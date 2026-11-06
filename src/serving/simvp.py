"""SimVP-v1 推理封装（服务化）。

复用项目既有训练产物：
  - 数据：data/processed/weddell_icemotion_daily.zarr
  - 模型：models/simvp-v1/ckpt_best.pt（SimVP 风格 SeaIceMotionNet，tin=10, tout=5, chid=128）
  - 结构：src/models.py::build_model / src/dataset.py 的预处理口径

对外提供与 `src.predict.engine.PredictionEngine` 同构的接口：
  - predict_point(model 无关) / predict_field / backtest / data_time / state

设计要点
--------
1. **绝不训练**：只 torch.load 既有 checkpoint 并 eval()。
2. 预处理严格复刻 `WeddellMotionDataset._frame/__getitem__`：
   u/v 用 1978-2017 逐像元 z-score（stats），NaN→0，pad 到 120×160，
   每帧额外拼 sin/cos(2π·doy) 两个通道，帧堆进 batch 维。
3. 输出 (1, 5, 2, 120, 160) 为标准化空间；取 day-1（index 0），裁回 116×158，
   再按逐像元 mean/std 反标准化 → cm/s。
4. 预计算场 `simvp_day1_fields.npz` 存在时优先使用（毫秒级响应）；
   否则对任意合法历史日期做实时推理（~0.3-1s）。
5. 线程安全：所有推理在 RLock 内串行（MPS 多线程不安全）。
"""

from __future__ import annotations

import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

GRID_H, GRID_W = 116, 158
PAD_H, PAD_W = 120, 160
TIN, TOUT = 10, 5
DEFAULT_ZARR = ROOT / "data/processed/weddell_icemotion_daily.zarr"
DEFAULT_CKPT = ROOT / "models/simvp-v1/ckpt_best.pt"
DEFAULT_PRECOMPUTED = ROOT / "data/processed/dashboard/simvp_day1_fields.npz"
DEFAULT_METRICS = ROOT / "data/processed/dashboard/metrics_simvp.json"
DEFAULT_CELL_STATS = ROOT / "data/processed/dashboard/cell_stats.npz"
TEST_YEARS = (2022, 2025)

# 24h 位移换算：speed(cm/s) → km/24h = speed * 0.036 * 24 = speed * 0.864
_KM_PER_CM_S_DAY = 3600.0 * 24.0 / 100.0 / 1000.0  # 0.864 km / (cm/s) / 24h
# 像元位移：u(cm/s) 一天漂移的像元数 = u*86400/(25067.525*100) ≈ u*0.034470
_CELLS_PER_CM_S_DAY = 86400.0 / (25067.525 * 100.0)


def _pad(a: np.ndarray) -> np.ndarray:
    """(H,W) 或 (T,H,W) → pad 到 (120,160)（与 dataset.py 相同：上2/下2、左1/右1）。"""
    ph, pw = (PAD_H - GRID_H) // 2, (PAD_W - GRID_W) // 2
    if a.ndim == 2:
        return np.pad(a, ((ph, PAD_H - GRID_H - ph), (pw, PAD_W - GRID_W - pw)))
    return np.pad(a, ((0, 0), (ph, PAD_H - GRID_H - ph), (pw, PAD_W - GRID_W - pw)))


def _unpad(a: np.ndarray) -> np.ndarray:
    ph, pw = (PAD_H - GRID_H) // 2, (PAD_W - GRID_W) // 2
    return a[..., ph:ph + GRID_H, pw:pw + GRID_W]


def _direction_deg(u: np.ndarray | float, v: np.ndarray | float):
    """矢量方向（度，0=东，逆时针为正，EASE x/y 分量）。"""
    return np.mod(np.degrees(np.arctan2(v, u)), 360.0)


def _bilinear(grid: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """在 (H,W) 网格上对小数 (row, col) 双线性插值（越界裁剪）。"""
    r = np.clip(np.asarray(rows, dtype=np.float64), 0, GRID_H - 1.000001)
    c = np.clip(np.asarray(cols, dtype=np.float64), 0, GRID_W - 1.000001)
    r0 = np.floor(r).astype(int)
    c0 = np.floor(c).astype(int)
    r1 = np.minimum(r0 + 1, GRID_H - 1)
    c1 = np.minimum(c0 + 1, GRID_W - 1)
    dr = r - r0
    dc = c - c0
    out = (grid[r0, c0] * (1 - dr) * (1 - dc) + grid[r0, c1] * (1 - dr) * dc
           + grid[r1, c0] * dr * (1 - dc) + grid[r1, c1] * dr * dc)
    return out


class SimVPPredictor:
    """既有 SimVP checkpoint 的服务化推理器（懒加载 / 线程安全 / 预计算场优先）。"""

    def __init__(self, zarr_path=DEFAULT_ZARR, ckpt=DEFAULT_CKPT,
                 device: str = "auto", precomputed=DEFAULT_PRECOMPUTED,
                 metrics_path=DEFAULT_METRICS, cell_stats_path=DEFAULT_CELL_STATS):
        self.zarr_path = Path(zarr_path)
        self.ckpt_path = Path(ckpt)
        self.precomputed_path = Path(precomputed) if precomputed else None
        self.metrics_path = Path(metrics_path) if metrics_path else None
        self.cell_stats_path = Path(cell_stats_path) if cell_stats_path else None
        self.device_name = device
        self._lock = threading.RLock()
        self._zarr = None
        self._torch = None
        self._model = None
        self._pre = None            # 预计算 dict: dates/target_idx/u/v
        self._field_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._field_cache_order: list[int] = []
        self._metrics: dict | None = None
        self._cell_stats: dict | None = None
        self._state = {
            "loaded": False,
            "device": None,
            "ckpt": str(self.ckpt_path),
            "error": None,
            "loaded_at": None,
            "precomputed": False,
        }

    # ------------------------------------------------------------------ 状态
    @property
    def state(self) -> dict:
        return dict(self._state)

    def load(self) -> dict:
        """显式加载（服务启动时调用；失败不抛异常，只记录 state.error）。"""
        try:
            self._ensure_zarr()
            self._ensure_model()
            self._state["loaded"] = True
            self._state["loaded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            if self._ensure_precomputed():
                self._state["precomputed"] = True
        except Exception as exc:  # noqa: BLE001
            self._state["error"] = f"{type(exc).__name__}: {exc}"
            self._state["loaded"] = False
        return dict(self._state)

    # ------------------------------------------------------------------ 懒加载
    def _ensure_zarr(self):
        if self._zarr is not None:
            return
        import zarr
        z = zarr.open_group(str(self.zarr_path), mode="r")
        self._zarr = z
        self._time = z["time"][:]
        self._dates = self._time.astype("datetime64[D]")
        self._doy = z["doy"][:]
        self._stats = {k: np.asarray(z["stats"][k][:], dtype=np.float32)
                       for k in ("u_mean", "u_std", "v_mean", "v_std")}
        self._lat = np.asarray(z["latitude"][:], dtype=np.float32)
        self._lon = np.asarray(z["longitude"][:], dtype=np.float32)
        self.n_frames = len(self._time)
        # 预构建全网格像元坐标（供整场预测）
        self._all_rows = np.repeat(np.arange(GRID_H), GRID_W)
        self._all_cols = np.tile(np.arange(GRID_W), GRID_H)

    def _ensure_torch(self):
        if self._torch is not None:
            return
        import torch
        self._torch = torch

    def _ensure_model(self):
        if self._model is not None:
            return
        self._ensure_torch()
        torch = self._torch
        from models import build_model, count_params  # ROOT/src 已在 sys.path
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        args = dict(ckpt.get("args") or {})
        tin = int(args.get("tin", TIN))
        tout = int(args.get("tout", TOUT))
        chid = int(args.get("chid", 128))
        n_s = int(args.get("n_s", 2))
        n_t = int(args.get("n_t", 4))
        add_doy = bool(args.get("add_doy", 1))
        cin = 4 if add_doy else 2
        self.tin, self.tout, self.chid = tin, tout, chid
        self.add_doy = add_doy
        model = build_model("simvp", tin=tin, tout=tout, cin=cin,
                            chid=chid, n_s=n_s, n_t=n_t)
        model.load_state_dict(ckpt["model"])
        model.eval()
        if self.device_name == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            device = self.device_name
        self.device = torch.device(device)
        model = model.to(self.device)
        # mps 的 CPU 回退：禁用（显式失败更易诊断）
        self._model = model
        self._state["device"] = device
        self._ckpt_epoch = ckpt.get("epoch")
        self._ckpt_best_val = ckpt.get("best_val")

    def _ensure_precomputed(self) -> bool:
        if self._pre is not None:
            return True
        if self.precomputed_path is None or not self.precomputed_path.exists():
            return False
        try:
            data = np.load(self.precomputed_path, allow_pickle=False)
            self._pre = {
                "dates": data["dates"],
                "target_idx": data["target_idx"].astype(np.int64),
                "u": data["u"],   # (N,116,158) float16
                "v": data["v"],
            }
            return True
        except Exception:  # noqa: BLE001
            self._pre = None
            return False

    def _ensure_metrics(self):
        if self._metrics is None:
            try:
                with open(self.metrics_path, "r", encoding="utf-8") as f:
                    self._metrics = json.load(f)
            except Exception:  # noqa: BLE001
                self._metrics = {}
        return self._metrics

    def _ensure_cell_stats(self):
        if self._cell_stats is None:
            try:
                d = np.load(self.cell_stats_path, allow_pickle=False)
                stats = {}
                for k in d.files:
                    arr = np.asarray(d[k])
                    # 兼容 (H,W) 与展平 (H*W,) 两种存法（ml-tabular artifact 为 1D）
                    if arr.ndim == 1 and arr.size == GRID_H * GRID_W:
                        arr = arr.reshape(GRID_H, GRID_W)
                    stats[k] = arr
                self._cell_stats = stats
            except Exception:  # noqa: BLE001
                self._cell_stats = {}
        return self._cell_stats

    # ------------------------------------------------------------------ 数据工具
    def idx_of_date(self, date: str) -> int | None:
        self._ensure_zarr()
        try:
            d = np.datetime64(str(date), "D")
        except Exception:  # noqa: BLE001
            return None
        hit = np.where(self._dates == d)[0]
        return int(hit[0]) if len(hit) else None

    def data_time(self) -> dict:
        self._ensure_zarr()
        built = None
        try:
            import zarr
            z = zarr.open_group(str(self.zarr_path), mode="r")
            built = z.attrs.get("created_utc")
        except Exception:  # noqa: BLE001
            built = None
        latest = str(self._dates[-1])
        try:
            age = int((np.datetime64("today", "D") - np.datetime64(latest, "D")).astype(int))
        except Exception:  # noqa: BLE001
            age = None
        return {
            "first_date": str(self._dates[0]),
            "latest_date": latest,
            "n_frames": int(self.n_frames),
            "built_at": built,
            "batch_age_days": age,
            "source": "NSIDC-0116 v4 Polar Pathfinder Daily 25km EASE-Grid Sea Ice Motion Vectors (Weddell subset)",
            "doi": "10.5067/INAWUWO7QH7B",
        }

    def _raw_window(self, last_obs_idx: int):
        """读取 [last_obs_idx-tin+1, last_obs_idx] 的原始 u,v 与 doy。"""
        self._ensure_zarr()
        lo = last_obs_idx - self.tin + 1
        if lo < 0:
            raise ValueError("insufficient_history")
        u = np.nan_to_num(np.asarray(self._zarr["u"][lo:last_obs_idx + 1], dtype=np.float32), nan=0.0)
        v = np.nan_to_num(np.asarray(self._zarr["v"][lo:last_obs_idx + 1], dtype=np.float32), nan=0.0)
        doy = np.asarray(self._doy[lo:last_obs_idx + 1], dtype=np.float32)
        return u, v, doy

    def _to_input_tensor(self, u_raw: np.ndarray, v_raw: np.ndarray, doy: np.ndarray):
        """(B? 首维为时间) → torch (1, T, C, 120, 160) 标准化输入。"""
        torch = self._torch
        mu, su = self._stats["u_mean"], self._stats["u_std"]
        mv, sv = self._stats["v_mean"], self._stats["v_std"]
        un = (u_raw - mu) / (su + 1e-6)
        vn = (v_raw - mv) / (sv + 1e-6)
        un, vn = _pad(un), _pad(vn)
        a = 2.0 * np.pi * doy[:, None, None]
        sin_c = np.broadcast_to(np.sin(a), un.shape).astype(np.float32)
        cos_c = np.broadcast_to(np.cos(a), un.shape).astype(np.float32)
        x = np.stack([un, vn, sin_c, cos_c], axis=1)  # (T,4,H,W)
        return torch.from_numpy(x).unsqueeze(0).to(self.device)

    def _forward_day1(self, x):
        """模型前向 → day-1 u,v（cm/s，116×158）。"""
        torch = self._torch
        with torch.no_grad():
            out = self._model(x)                    # (1, 5, 2, 120, 160) 标准化
            du = out[0, 0, 0].float().cpu().numpy()
            dv = out[0, 0, 1].float().cpu().numpy()
        du, dv = _unpad(du), _unpad(dv)
        u = du * self._stats["u_std"] + self._stats["u_mean"]
        v = dv * self._stats["v_std"] + self._stats["v_mean"]
        return u.astype(np.float32), v.astype(np.float32)

    def _predict_indices(self, last_obs_indices) -> list[tuple[np.ndarray, np.ndarray]]:
        """批量预测 day-1 场（list of (u,v) cm/s）。"""
        self._ensure_model()
        outs = []
        with self._lock:
            for idx in last_obs_indices:
                u_raw, v_raw, doy = self._raw_window(int(idx))
                x = self._to_input_tensor(u_raw, v_raw, doy)
                outs.append(self._forward_day1(x))
        return outs

    # ------------------------------------------------------------------ 核心
    def _field_for_last_obs(self, last_obs_idx: int) -> tuple[np.ndarray, np.ndarray, int, str]:
        """返回目标帧（last_obs+1）的 u,v 场 + target_idx + 来源（precomputed/live）。"""
        target_idx = last_obs_idx + 1
        self._ensure_zarr()
        if target_idx >= self.n_frames:
            raise ValueError("no_future_frame")
        if self._ensure_precomputed() and self._pre is not None:
            hit = np.where(self._pre["target_idx"] == target_idx)[0]
            if len(hit):
                k = int(hit[0])
                u = self._pre["u"][k].astype(np.float32)
                v = self._pre["v"][k].astype(np.float32)
                return u, v, target_idx, "precomputed"
        # LRU 缓存
        if target_idx in self._field_cache:
            u, v = self._field_cache[target_idx]
            return u, v, target_idx, "cache"
        u, v = self._predict_indices([last_obs_idx])[0]
        self._field_cache[target_idx] = (u, v)
        self._field_cache_order.append(target_idx)
        while len(self._field_cache_order) > 8:
            old = self._field_cache_order.pop(0)
            self._field_cache.pop(old, None)
        return u, v, target_idx, "live"

    def _mask(self, idx: int) -> np.ndarray:
        return np.asarray(self._zarr["valid_mask"][int(idx)], dtype=bool)

    def _cell_payload(self, row: int, col: int, idx: int) -> dict:
        u = float(np.asarray(self._zarr["u"][idx, row, col], dtype=np.float32))
        v = float(np.asarray(self._zarr["v"][idx, row, col], dtype=np.float32))
        return {
            "u": round(u, 3), "v": round(v, 3),
            "speed": round(float(math.hypot(u, v)), 3),
            "direction_deg": round(float(_direction_deg(u, v)), 2),
            "lat": round(float(self._lat[row, col]), 4),
            "lon": round(float(self._lon[row, col]), 4),
        }

    def _warnings(self, row, col, last_obs_idx, target_idx, pred_speed, valid: bool = True):
        warns = []
        full = np.asarray(self._zarr["valid_fraction"][last_obs_idx], dtype=np.float32)
        coverage = float(full)
        if coverage < 0.15:
            warns.append({"code": "low_coverage", "level": "warning",
                          "message": f"起报日有效海冰覆盖仅 {coverage * 100:.1f}%，预测可信度下降"})
        gap = int((self._dates[target_idx] - self._dates[last_obs_idx]).astype(int))
        if gap > 1:
            warns.append({"code": "frame_gap", "level": "info",
                          "message": f"下一有效帧距起报日 {gap} 天（数据缺失日），24h 曲线为近似"})
        cs = self._ensure_cell_stats()
        p99 = None
        if cs and "speed_p99" in cs:
            try:
                p99 = float(cs["speed_p99"][row, col])
            except Exception:  # noqa: BLE001
                p99 = None
        if valid and p99 is not None and np.isfinite(p99) and p99 > 0.5 and pred_speed > p99:
            warns.append({"code": "high_speed", "level": "warning",
                          "message": f"预测漂移速度 {pred_speed:.1f} cm/s 高于该像元历史 99% 分位（{p99:.1f} cm/s），可能为极端事件"})
        return warns

    @staticmethod
    def _series_24h(u: float, v: float, row: int, col: int, lat_grid, lon_grid):
        hours = list(range(25))
        speed = math.hypot(u, v)
        cum = [round(speed * 0.036 * h, 4) for h in hours]  # km
        speed_series = [round(speed, 4) for _ in hours]
        rows = [row + v * _CELLS_PER_CM_S_DAY * (h / 24.0) for h in hours]
        cols = [col + u * _CELLS_PER_CM_S_DAY * (h / 24.0) for h in hours]
        lats = [round(float(x), 5) for x in _bilinear(lat_grid, np.array(rows), np.array(cols))]
        lons = [round(float(x), 5) for x in _bilinear(lon_grid, np.array(rows), np.array(cols))]
        return {
            "hours": hours,
            "cumulative_km": cum,
            "speed_cm_s": speed_series,
            "row": [round(x, 5) for x in rows],
            "col": [round(x, 5) for x in cols],
            "lat": lats,
            "lon": lons,
            "uncertainty": None,
            "interpolation": "日尺度产品：逐时位置/位移由 24h 平均漂移矢量线性积分",
            "target_date": None,
        }

    # ------------------------------------------------------------------ 公共 API
    def predict_field(self, date: str) -> dict:
        t_start = time.perf_counter()
        last_obs = self.idx_of_date(date)
        if last_obs is None:
            return {"ok": False, "error": "date_not_in_dataset", "message": f"数据集中不存在日期 {date}"}
        try:
            u, v, target_idx, source = self._field_for_last_obs(last_obs)
        except ValueError as exc:
            code = str(exc)
            msg = {"no_future_frame": "该日期已是数据集最后一帧，之后没有可用帧，无法预测",
                   "insufficient_history": "该日期之前历史帧不足（SimVP 需要 10 帧输入）"}.get(
                code, str(exc))
            return {"ok": False, "error": code, "message": msg}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": "inference_failed", "message": f"{type(exc).__name__}: {exc}"}
        speed = np.hypot(u, v).astype(np.float32)
        mask = self._mask(last_obs)
        u = np.where(mask, u, np.nan)
        v = np.where(mask, v, np.nan)
        speed = np.where(mask, speed, np.nan)
        coverage = float(mask.mean())
        warnings = self._warnings_global(last_obs)
        return {
            "ok": True, "target_idx": target_idx, "target_date": str(self._dates[target_idx]),
            "u": u, "v": v, "speed": speed, "coverage": coverage,
            "source": source, "warnings": warnings,
            "elapsed_ms": round((time.perf_counter() - t_start) * 1000.0, 1),
        }

    def _warnings_global(self, last_obs_idx: int):
        warns = []
        cov = float(self._zarr["valid_fraction"][last_obs_idx])
        if cov < 0.15:
            warns.append({"code": "low_coverage", "level": "warning",
                          "message": f"起报日有效海冰覆盖仅 {cov * 100:.1f}%，预测可信度下降"})
        return warns

    def predict_point(self, date: str, row: int, col: int) -> dict:
        t_start = time.perf_counter()
        last_obs = self.idx_of_date(date)
        if last_obs is None:
            return {"ok": False, "error": "date_not_in_dataset",
                    "message": f"数据集中不存在日期 {date}"}
        try:
            u, v, target_idx, source = self._field_for_last_obs(last_obs)
        except ValueError as exc:
            code = str(exc)
            msg = {"no_future_frame": "该日期已是数据集最后一帧，之后没有可用帧，无法预测",
                   "insufficient_history": "该日期之前历史帧不足（SimVP 需要 10 帧输入）"}.get(
                code, str(exc))
            return {"ok": False, "error": code, "message": msg}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": "inference_failed",
                    "message": f"{type(exc).__name__}: {exc}"}
        mask_obs = self._mask(last_obs)
        valid = bool(mask_obs[row, col])
        inp = self._cell_payload(row, col, last_obs)
        pred = {"u": round(float(u[row, col]), 3), "v": round(float(v[row, col]), 3)}
        pred["speed"] = round(float(math.hypot(pred["u"], pred["v"])), 3)
        pred["direction_deg"] = round(float(_direction_deg(pred["u"], pred["v"])), 2)
        series = self._series_24h(pred["u"], pred["v"], row, col, self._lat, self._lon)
        series["target_date"] = str(self._dates[target_idx])
        warns = self._warnings(row, col, last_obs, target_idx, pred["speed"], valid=valid)
        if not valid:
            warns.insert(0, {"code": "no_observation", "level": "info",
                             "message": "起报日该像元无有效观测（u/v=0），预测主要依赖邻域/季节特征"})
        coverage = float(np.asarray(self._zarr["valid_fraction"][last_obs], dtype=np.float32))
        return {
            "ok": True,
            "date": str(self._dates[last_obs]),
            "next_date": str(self._dates[target_idx]),
            "next_idx": int(target_idx),
            "u": pred["u"], "v": pred["v"], "speed": pred["speed"],
            "direction_deg": pred["direction_deg"],
            "input": inp | {"row": row, "col": col, "valid": valid,
                            "coverage": round(coverage, 4),
                            "date": str(self._dates[last_obs])},
            "series_24h": series,
            "source": source,
            "warnings": warns,
            "elapsed_ms": round((time.perf_counter() - t_start) * 1000.0, 1),
        }

    def backtest(self, row: int, col: int, start: str, end: str,
                 max_points: int = 180) -> dict:
        """历史回测：仅覆盖预计算场存在的 test 期（2022–2025）。"""
        if not self._ensure_precomputed() or self._pre is None:
            return {"ok": False, "error": "precomputed_unavailable",
                    "message": "SimVP 回测需要预计算场 simvp_day1_fields.npz"}
        target_idx = self._pre["target_idx"]
        dates = np.asarray(self._pre["dates"]).astype("datetime64[D]")
        d0 = np.datetime64(str(start), "D")
        d1 = np.datetime64(str(end), "D")
        avail_start, avail_end = str(dates.min()), str(dates.max())
        sel = np.where((dates >= d0) & (dates <= d1))[0]
        warnings = []
        if len(sel) == 0:
            return {"ok": False, "error": "backtest_out_of_range",
                    "message": f"SimVP 回测可用范围 {avail_start} ~ {avail_end}",
                    "available_start": avail_start, "available_end": avail_end}
        if d0 < dates.min() or d1 > dates.max():
            warnings.append({"code": "range_clipped", "level": "info",
                             "message": f"回测范围已裁剪至可用范围 {avail_start} ~ {avail_end}"})
        n_total = len(sel)
        if len(sel) > max_points:
            step = len(sel) / float(max_points)
            sel = sel[np.linspace(0, len(sel) - 1, max_points).round().astype(int)]
        # 用原始 u/v 读取真值与持续性（目标帧在 zarr 中是连续区间，一次切片读出）
        series = []
        t_idx = target_idx[sel]
        lo, hi = int(t_idx.min()), int(t_idx.max())
        u_seg = np.asarray(self._zarr["u"][lo:hi + 1, row, col], dtype=np.float32)
        v_seg = np.asarray(self._zarr["v"][lo:hi + 1, row, col], dtype=np.float32)
        pers_u = np.asarray(self._zarr["u"][lo - 1:hi, row, col], dtype=np.float32)
        pers_v = np.asarray(self._zarr["v"][lo - 1:hi, row, col], dtype=np.float32)
        obs_u = u_seg[t_idx - lo]
        obs_v = v_seg[t_idx - lo]
        pers_u = pers_u[t_idx - lo]
        pers_v = pers_v[t_idx - lo]
        pred_u = np.asarray(self._pre["u"][sel, row, col], dtype=np.float32)
        pred_v = np.asarray(self._pre["v"][sel, row, col], dtype=np.float32)
        for k, si in enumerate(sel):
            series.append({
                "date": str(dates[si]),
                "obs_u": float(round(float(obs_u[k]), 3)),
                "obs_v": float(round(float(obs_v[k]), 3)),
                "obs_speed": float(round(float(math.hypot(obs_u[k], obs_v[k])), 3)),
                "pred_u": float(round(float(pred_u[k]), 3)),
                "pred_v": float(round(float(pred_v[k]), 3)),
                "pred_speed": float(round(float(math.hypot(pred_u[k], pred_v[k])), 3)),
            })
        summary = _summary_from_series(series, pers_u, pers_v)
        return {
            "ok": True,
            "available_start": avail_start,
            "available_end": avail_end,
            "n_days_total": int(n_total),
            "series": series,
            "summary": summary,
            "warnings": warnings,
        }


def _summary_from_series(series: list[dict], pers_u=None, pers_v=None) -> dict:
    if not series:
        return {"n": 0}
    ou = np.array([s["obs_u"] for s in series], dtype=np.float64)
    ov = np.array([s["obs_v"] for s in series], dtype=np.float64)
    pu = np.array([s["pred_u"] for s in series], dtype=np.float64)
    pv = np.array([s["pred_v"] for s in series], dtype=np.float64)
    du, dv = pu - ou, pv - ov
    d2 = du ** 2 + dv ** 2
    rmse_u = float(np.sqrt(np.mean(du ** 2)))
    rmse_v = float(np.sqrt(np.mean(dv ** 2)))
    vec = float(np.sqrt(np.mean(d2)))
    mae_vec = float(np.mean(np.sqrt(d2)))
    sst_u = float(np.sum((ou - ou.mean()) ** 2))
    sst_v = float(np.sum((ov - ov.mean()) ** 2))
    r2_u = 1.0 - float(np.sum(du ** 2)) / sst_u if sst_u > 0 else float("nan")
    r2_v = 1.0 - float(np.sum(dv ** 2)) / sst_v if sst_v > 0 else float("nan")
    skill = None
    if pers_u is not None and pers_v is not None:
        pers_u = np.asarray(pers_u, dtype=np.float64)
        pers_v = np.asarray(pers_v, dtype=np.float64)
        pers_vec = float(np.sqrt(np.mean((pers_u - ou) ** 2 + (pers_v - ov) ** 2)))
        skill = round(1.0 - vec / (pers_vec + 1e-12), 4)
    return {
        "n": len(series),
        "rmse_u": round(rmse_u, 4), "rmse_v": round(rmse_v, 4),
        "vec_rmse": round(vec, 4), "mae_vec": round(mae_vec, 4),
        "r2_u": round(r2_u, 4), "r2_v": round(r2_v, 4),
        "skill_vs_persistence": skill,
    }
