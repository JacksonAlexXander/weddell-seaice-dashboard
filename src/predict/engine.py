"""统一预测引擎 `PredictionEngine`：单点 / 场 / 回测 / 像元统计 / 数据时间。

接口契约见 docs/PREDICT_DASHBOARD_DESIGN.md §5.1（backend-api 按此调用）。
服务端只加载离线 artifact、**绝不训练、不写模型文件**：

- `ridge` → `models/ml/ridge.joblib`（Pipeline: StandardScaler + Ridge，多输出 u,v）
- `random_forest` → `models/ml/rf.joblib`（RandomForestRegressor 多输出）
- `persistence` → 当前帧（零训练基线）
- `climatology` → `data/processed/dashboard/climatology.npz`（1978–2017 逐月平均 u,v）

错误约定（便于后端映射 HTTP 状态码）：引擎抛 `PredictError` 子类，带 `.code` 与 `.http_status`：
`UnknownModelError`(404) / `InvalidCellError`(400) / `DateError`(422) /
`ModelUnavailableError`(409) / `InsufficientHistoryError`(422)。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np

from .baselines import ClimatologyPredictor, PersistencePredictor
from .common import (
    next_calendar_day,
    ARTIFACT_DIR,
    EXTREME_SPEED_ABS,
    GRID_H,
    GRID_W,
    KM_PER_CM_S_24H,
    LOW_COVERAGE_THRESHOLD,
    MODEL_DIR,
    PERSISTENCE_DELTA_WARN,
    ROOT,
    resolve_path,
)
from .data import DataStore
from .features import N_HISTORY, build_features
from .metrics import vec_metrics

# ---------------------------------------------------------------- 模型注册表
MODEL_IDS: tuple[str, ...] = ("ridge", "random_forest", "persistence", "climatology")
ML_MODEL_IDS: tuple[str, ...] = ("ridge", "random_forest")
BASELINE_IDS: tuple[str, ...] = ("persistence", "climatology")

MODEL_META: dict[str, dict] = {
    "ridge": {
        "id": "ridge", "name": "Ridge 岭回归", "name_zh": "Ridge 岭回归", "short": "Ridge",
        "category": "linear", "color": "#22d3ee", "is_baseline": False, "is_highlight": True,
        "description": "标准化特征上的多输出线性回归（34 维滞后/邻域/季节特征），推理毫秒级，可解释性强。",
    },
    "random_forest": {
        "id": "random_forest", "name": "Random Forest 随机森林",
        "name_zh": "Random Forest 随机森林", "short": "Random Forest",
        "category": "tree", "color": "#a78bfa", "is_baseline": False, "is_highlight": True,
        "description": "多输出随机森林（bootstrap + feature subsampling），用树间离散度估计不确定度。",
    },
    "persistence": {
        "id": "persistence", "name": "持续性基线", "name_zh": "持续性基线", "short": "Persistence",
        "category": "baseline", "color": "#64748b", "is_baseline": True, "is_highlight": False,
        "description": "下一帧 = 当前帧（海冰漂移短期预测的强基准）。",
    },
    "climatology": {
        "id": "climatology", "name": "气候态基线", "name_zh": "气候态基线", "short": "Climatology",
        "category": "baseline", "color": "#475569", "is_baseline": True, "is_highlight": False,
        "description": "1978–2017 逐月平均漂移场（每像元）。",
    },
}

DEFAULT_POINT = {"model": "random_forest", "date": "2025-06-26", "row": 65, "col": 60}

# 不确定度缺省标定（model_card.json 缺失时回退）
DEFAULT_RIDGE_RESIDUAL_VEC_RMSE = 8.0

SOURCE = ("NSIDC-0116 v4 Polar Pathfinder Daily 25km EASE-Grid Sea Ice Motion "
          "Vectors (Weddell subset)")
DOI = "10.5067/INAWUWO7QH7B"


# ---------------------------------------------------------------- 异常
class PredictError(ValueError):
    """引擎错误基类；`.code` / `.http_status` 供后端映射 ErrorResponse。"""
    code = "inference_failed"
    http_status = 500

    def __init__(self, message: str, *, hint: str | None = None):
        super().__init__(message)
        self.message = message
        self.hint = hint

    def to_dict(self) -> dict:
        d = {"status": "error",
             "error": {"code": self.code, "message": self.message}}
        if self.hint:
            d["error"]["hint"] = self.hint
        return d


class UnknownModelError(PredictError):
    code = "unknown_model"
    http_status = 404


class ModelUnavailableError(PredictError):
    code = "model_unavailable"
    http_status = 409


class InvalidCellError(PredictError):
    code = "invalid_cell"
    http_status = 400


class DateError(PredictError):
    code = "date_invalid"
    http_status = 422


class InsufficientHistoryError(PredictError):
    code = "insufficient_history"
    http_status = 422


# ---------------------------------------------------------------- 工具
def _direction_deg(u: float, v: float) -> float:
    """矢量方向（度，0=东，逆时针为正）。"""
    return float((np.degrees(np.arctan2(v, u)) + 360.0) % 360.0)


def _vec_speed(u: float, v: float) -> float:
    return float(np.hypot(u, v))


def _finite(x, default=None):
    v = float(x)
    return v if np.isfinite(v) else default


class PredictionEngine:
    """预测引擎（线程安全；模型/气候态/统计量均惰性加载并缓存）。"""

    def __init__(self, zarr_path: str | Path | None = None,
                 model_dir: str | Path | None = None,
                 artifact_dir: str | Path | None = None):
        self.store = DataStore(zarr_path)
        self.model_dir = resolve_path(model_dir, default=MODEL_DIR)
        self.artifact_dir = resolve_path(artifact_dir, default=ARTIFACT_DIR)
        self._lock = threading.RLock()
        self._models: dict[str, dict] = {}            # model_id -> {"pipeline","payload","path","mtime"}
        self._load_errors: dict[str, str] = {}
        self._clim: ClimatologyPredictor | None = None
        self._clim_error: str | None = None
        self._pers = PersistencePredictor()
        self._cell_stats: dict | None = None
        self._cell_stats_error: str | None = None
        self._model_card: dict | None = None
        self._metrics: dict | None = None

    # ------------------------------------------------------------ 元信息
    @property
    def model_ids(self) -> list[str]:
        return list(MODEL_IDS)

    def available(self, model_id: str) -> bool:
        """模型 artifact 是否存在且可加载（惰性：首次调用会尝试加载并缓存结果）。"""
        if model_id not in MODEL_IDS:
            return False
        if model_id == "persistence":
            return True
        if model_id == "climatology":
            try:
                self._ensure_clim()
                return True
            except Exception as e:  # noqa: BLE001
                self._clim_error = f"{type(e).__name__}: {e}"
                return False
        try:
            self._ensure_loaded(model_id)
            return True
        except Exception as e:  # noqa: BLE001
            self._load_errors[model_id] = f"{type(e).__name__}: {e}"
            return False

    def _artifact_path(self, model_id: str) -> Path:
        return self.model_dir / f"{'rf' if model_id == 'random_forest' else model_id}.joblib"

    def _ensure_loaded(self, model_id: str) -> dict:
        if model_id in self._models:
            return self._models[model_id]
        if model_id not in ML_MODEL_IDS:
            raise ModelUnavailableError(f"{model_id} 不是 joblib 模型")
        path = self._artifact_path(model_id)
        if not path.exists():
            raise ModelUnavailableError(f"模型 artifact 缺失: {path}",
                                        hint="请先运行 scripts/build_predict_artifacts.py 训练/生成")
        import joblib
        with self._lock:
            if model_id in self._models:
                return self._models[model_id]
            try:
                payload = joblib.load(path)
            except Exception as e:  # noqa: BLE001
                raise ModelUnavailableError(f"模型加载失败: {path} ({type(e).__name__}: {e})")
            if isinstance(payload, dict) and "pipeline" in payload:
                pipe = payload["pipeline"]
            else:                                     # 兼容直接 dump 的 sklearn 对象
                payload = {"pipeline": payload}
                pipe = payload["pipeline"]
            payload = dict(payload)
            payload["path"] = str(path)
            payload["mtime"] = path.stat().st_mtime
            self._models[model_id] = payload
            return payload

    def _ensure_clim(self) -> ClimatologyPredictor:
        if self._clim is not None:
            return self._clim
        with self._lock:
            if self._clim is None:
                self._clim = ClimatologyPredictor(self.artifact_dir / "climatology.npz")
        return self._clim

    def _load_model_card(self) -> dict:
        if self._model_card is not None:
            return self._model_card
        path = self.model_dir / "model_card.json"
        try:
            with open(path, "r", encoding="utf-8") as fh:
                self._model_card = json.load(fh)
        except Exception:  # noqa: BLE001
            self._model_card = {}
        return self._model_card

    def _load_metrics(self) -> dict:
        if self._metrics is not None:
            return self._metrics
        path = self.artifact_dir / "metrics_tabular.json"
        try:
            with open(path, "r", encoding="utf-8") as fh:
                self._metrics = json.load(fh)
        except Exception:  # noqa: BLE001
            self._metrics = {}
        return self._metrics

    def reload(self) -> None:
        """丢弃已加载模型（artifact 更新后调用；服务端一般不需要）。"""
        with self._lock:
            self._models.clear()
            self._load_errors.clear()
            self._clim = None
            self._model_card = None
            self._metrics = None

    def model_info(self, model_id: str) -> dict:
        """单个模型的元信息（后端 /api/models 可直接使用）。"""
        if model_id not in MODEL_IDS:
            raise UnknownModelError(f"未知模型: {model_id}",
                                    hint=f"可选模型: {', '.join(MODEL_IDS)}")
        meta = dict(MODEL_META[model_id])
        info = {
            **meta,
            "available": self.available(model_id),
            "trained_at": None,
            "training_samples": None,
            "metrics": None,
        }
        card = self._load_model_card()
        if model_id in ML_MODEL_IDS:
            info["trained_at"] = card.get("created_at")
            info["training_samples"] = (card.get("training", {})
                                        .get("random_forest" if model_id == "random_forest"
                                             else "ridge", {})
                                        .get("training_samples"))
            info["params"] = (card.get("training", {})
                              .get("random_forest" if model_id == "random_forest"
                                   else "ridge", {}).get("params")
                              or ({"alpha": (card.get("training", {}).get("ridge", {})
                                             .get("alpha"))} if model_id == "ridge" else None))
        metrics = self._load_metrics()
        for row in metrics.get("models", []):
            if row.get("id") == model_id:
                info["metrics"] = {"vec_rmse": row.get("vec_rmse"),
                                   "skill_vs_persistence": row.get("skill_vs_persistence"),
                                   "latency_ms": row.get("latency_ms")}
                break
        return info

    def models_info(self) -> list[dict]:
        return [self.model_info(m) for m in MODEL_IDS]

    # ------------------------------------------------------------ 数据时间 / 像元统计
    def data_time(self) -> dict:
        latest = self.store.date_of(self.store.n_frames - 1)
        first = self.store.date_of(0)
        built_at = self.store.created_utc
        if not built_at:
            try:
                mtime = self.store.zarr_path.stat().st_mtime
                built_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(mtime))
            except OSError:  # pragma: no cover
                built_at = None
        try:
            age = int((np.datetime64("today", "D") - np.datetime64(latest, "D"))
                      / np.timedelta64(1, "D"))
        except Exception:  # pragma: no cover
            age = None
        return {
            "first_date": first,
            "latest_date": latest,
            "n_frames": int(self.store.n_frames),
            "built_at": built_at,
            "batch_age_days": age,
            "source": SOURCE,
            "doi": DOI,
        }

    def _load_cell_stats(self) -> dict:
        if self._cell_stats is not None:
            return self._cell_stats
        path = self.artifact_dir / "cell_stats.npz"
        with self._lock:
            if self._cell_stats is not None:
                return self._cell_stats
            try:
                d = np.load(path)

                def _as2d(a):
                    a = np.asarray(a)
                    if a.ndim == 1 and a.size == GRID_H * GRID_W:   # 兼容旧 1D schema
                        return a.reshape(GRID_H, GRID_W)
                    return a

                self._cell_stats = {k: _as2d(d[k]) for k in
                                    ("speed_mean", "speed_std", "speed_p99", "valid_days_count")}
            except Exception:  # noqa: BLE001 — artifact 未生成时用 zarr stats 兜底
                stats = self.store.zg["stats"]
                u_mean = np.asarray(stats["u_mean"][:], np.float32)
                v_mean = np.asarray(stats["v_mean"][:], np.float32)
                u_std = np.asarray(stats["u_std"][:], np.float32)
                v_std = np.asarray(stats["v_std"][:], np.float32)
                smean = np.hypot(u_mean, v_mean).astype(np.float32)
                sstd = np.sqrt(u_std ** 2 + v_std ** 2).astype(np.float32)
                self._cell_stats = {
                    "speed_mean": smean,
                    "speed_std": sstd,
                    "speed_p99": (smean + 2.33 * sstd).astype(np.float32),
                    "valid_days_count": np.asarray(stats["valid_days_count"][:], np.int32),
                }
            return self._cell_stats

    def cell_stats(self, row: int, col: int) -> dict:
        row, col = self._validate_cell(row, col)
        cs = self._load_cell_stats()
        lat, lon = self.store.cell_latlon(row, col)
        return {
            "speed_mean": float(cs["speed_mean"][row, col]),
            "speed_std": float(cs["speed_std"][row, col]),
            "speed_p99": float(cs["speed_p99"][row, col]),
            "valid_days": int(cs["valid_days_count"][row, col]),
            "lat": lat,
            "lon": lon,
        }

    # ------------------------------------------------------------ 校验
    def _validate_model(self, model_id: str) -> str:
        if model_id not in MODEL_IDS:
            raise UnknownModelError(f"未知模型: {model_id!r}",
                                    hint=f"可选模型: {', '.join(MODEL_IDS)}")
        return model_id

    @staticmethod
    def _validate_cell(row: int, col: int) -> tuple[int, int]:
        try:
            r, c = int(row), int(col)
        except (TypeError, ValueError):
            raise InvalidCellError(f"row/col 必须为整数: row={row!r}, col={col!r}")
        if not (0 <= r < GRID_H and 0 <= c < GRID_W):
            raise InvalidCellError(f"像元越界: (row={r}, col={c})",
                                   hint=f"row 0–{GRID_H - 1}, col 0–{GRID_W - 1}")
        return r, c

    def _resolve_last_idx(self, date: str) -> int:
        idx = self.store.idx_of_date(date)
        if idx is None:
            first, last = self.store.date_of(0), self.store.date_of(self.store.n_frames - 1)
            raise DateError(f"数据集中无该日期: {date}", hint=f"可选日期 {first} ~ {last}")
        if idx < N_HISTORY - 1:
            raise InsufficientHistoryError(
                f"{date} 之前历史帧不足（需要 {N_HISTORY} 帧，仅有 {idx + 1} 帧）",
                hint=f"最早可用起报日 {self.store.date_of(N_HISTORY - 1)}")
        return idx

    # ------------------------------------------------------------ 特征 / 预测
    def _features(self, last_idx: int, rows, cols) -> np.ndarray:
        win = self.store.window(int(last_idx), N_HISTORY)
        return build_features(win, rows, cols)

    def _predict_rows(self, model_id: str, last_idx: int, rows, cols,
                      X: np.ndarray | None) -> np.ndarray:
        """返回 (n,2) float64 的 (u,v) 预测。"""
        rows = np.asarray(rows, np.int64)
        cols = np.asarray(cols, np.int64)
        if model_id in ML_MODEL_IDS:
            if X is None:
                X = self._features(last_idx, rows, cols)
            payload = self._ensure_loaded(model_id)
            pred = np.asarray(payload["pipeline"].predict(X), np.float64).reshape(-1, 2)
            return pred
        if model_id == "persistence":
            f = self.store.frame(int(last_idx))
            return np.stack([np.asarray(f["u"][rows, cols], np.float64),
                             np.asarray(f["v"][rows, cols], np.float64)], axis=1)
        if model_id == "climatology":
            clim = self._ensure_clim()
            target_day = next_calendar_day(self.store.dates[last_idx])
            k = int(target_day.astype("datetime64[M]").astype(int) % 12)
            return np.stack([np.asarray(clim.u[k][rows, cols], np.float64),
                             np.asarray(clim.v[k][rows, cols], np.float64)], axis=1)
        raise UnknownModelError(f"未知模型: {model_id!r}")

    # ------------------------------------------------------------ 不确定度
    def _rf_tree_distances(self, payload: dict, X: np.ndarray) -> np.ndarray:
        """RF 每棵树的 24h 位移模长 (n_trees, n) km。"""
        rf = payload["pipeline"]
        n_trees = len(rf.estimators_)
        out = np.empty((n_trees, X.shape[0]), np.float64)
        for k, est in enumerate(rf.estimators_):
            p = np.asarray(est.predict(X), np.float64).reshape(X.shape[0], -1)
            out[k] = KM_PER_CM_S_24H * np.hypot(p[:, 0], p[:, 1])
        return out

    def _uncertainty(self, model_id: str, pred: np.ndarray, X: np.ndarray | None) -> dict | None:
        u, v = float(pred[0, 0]), float(pred[0, 1])
        dist_km = KM_PER_CM_S_24H * _vec_speed(u, v)
        if model_id == "random_forest":
            if X is None:
                return None
            payload = self._ensure_loaded(model_id)
            disp = self._rf_tree_distances(payload, X)[:, 0]
            lo, hi = np.percentile(disp, [10, 90])
            return {
                "method": "RF 树间离散度 10–90 分位（24h 位移 km）",
                "lower_km": float(max(0.0, lo)),
                "upper_km": float(max(hi, lo)),
                "pred_km": float(dist_km),
                "n_trees": int(disp.size),
            }
        if model_id == "ridge":
            card = self._load_model_card()
            cal = (card.get("training", {})
                   .get("ridge_uncertainty_calibration", {}) or {})
            resid = float(cal.get("vec_residual_rmse", DEFAULT_RIDGE_RESIDUAL_VEC_RMSE))
            band = KM_PER_CM_S_24H * resid
            return {
                "method": f"ridge 验证集残差 RMSE 近似（±1σ = ±{band:.2f} km / 24h）",
                "lower_km": float(max(0.0, dist_km - band)),
                "upper_km": float(dist_km + band),
                "pred_km": float(dist_km),
            }
        return None   # 基线不给不确定度（前端按 null 处理）

    # ------------------------------------------------------------ 预警
    def _warnings(self, model_id: str, last_idx: int, row: int, col: int,
                  pred_u: float, pred_v: float, *, cur_valid: bool, coverage: float,
                  include_quality: bool = True) -> list[dict]:
        out: list[dict] = []
        if not cur_valid:
            msg = ("当前像元在最后观测日无效：特征使用历史回填/邻域值，预测可信度下降"
                   if model_id in ML_MODEL_IDS else
                   "当前像元在最后观测日无效，基线在该像元的预测退化为代表性数值（0/气候态）")
            out.append({"code": "input_invalid", "level": "warning", "message": msg})
        if coverage < LOW_COVERAGE_THRESHOLD:
            out.append({
                "code": "low_coverage", "level": "warning",
                "message": (f"当日有效覆盖仅 {coverage:.1%}（< {LOW_COVERAGE_THRESHOLD:.0%}），"
                            f"空间上下文稀疏，预测不确定性增加"),
            })
        pred_speed = _vec_speed(pred_u, pred_v)
        if include_quality:
            cs = self._load_cell_stats()
            p99 = float(cs["speed_p99"][row, col])
            if np.isfinite(p99) and p99 > 0 and pred_speed > p99:
                out.append({
                    "code": "high_speed", "level": "warning",
                    "message": (f"预测漂移速度 {pred_speed:.1f} cm/s 高于该像元历史 99% "
                                f"分位（{p99:.1f} cm/s），可能为极端事件"),
                })
            if cur_valid:
                f = self.store.frame(int(last_idx))
                cu = float(f["u"][row, col])
                cv = float(f["v"][row, col])
                delta = float(np.hypot(pred_u - cu, pred_v - cv))
                if delta > PERSISTENCE_DELTA_WARN:
                    out.append({
                        "code": "persistence_divergence", "level": "warning",
                        "message": (f"预测与持续性基线差异较大（Δ={delta:.1f} cm/s），"
                                    f"模型可能过度外推"),
                    })
        if pred_speed > EXTREME_SPEED_ABS:
            out.append({
                "code": "extreme_speed", "level": "danger",
                "message": f"预测漂移速度 {pred_speed:.1f} cm/s 超出物理合理范围（>{EXTREME_SPEED_ABS:.0f}），请谨慎使用",
            })
        return out

    # ------------------------------------------------------------ 单点预测
    def predict_point(self, model_id: str, date: str, row: int, col: int) -> dict:
        """预测 (date, row, col) 的下一自然日 (u,v)。

        返回兼容设计文档 §5.1 的字段：ok/next_date/next_idx/u/v/speed/direction_deg/
        input/uncertainty/warnings，并额外提供 displacement_km_24h / delta_speed_vs_persistence。
        """
        model_id = self._validate_model(model_id)
        row, col = self._validate_cell(row, col)
        last_idx = self._resolve_last_idx(date)
        target_date = str(next_calendar_day(self.store.dates[last_idx]))
        next_idx = self.store.idx_of_date(target_date)
        if model_id in ML_MODEL_IDS:
            self._ensure_loaded(model_id)

        win = self.store.window(int(last_idx), N_HISTORY)
        X = build_features(win, np.array([row]), np.array([col]))
        pred = self._predict_rows(model_id, last_idx, [row], [col], X)
        pred_u, pred_v = float(pred[0, 0]), float(pred[0, 1])
        speed = _vec_speed(pred_u, pred_v)
        direction = _direction_deg(pred_u, pred_v)

        f = self.store.frame(int(last_idx))
        cur_u, cur_v = float(f["u"][row, col]), float(f["v"][row, col])
        cur_speed = _vec_speed(cur_u, cur_v)
        cur_valid = bool(f["mask"][row, col] > 0)
        coverage = float(f["coverage"])
        lat, lon = self.store.cell_latlon(row, col)
        pers = self._predict_rows("persistence", last_idx, [row], [col], None)[0]
        pers_speed = _vec_speed(float(pers[0]), float(pers[1]))
        warnings = self._warnings(model_id, last_idx, row, col, pred_u, pred_v,
                                  cur_valid=cur_valid, coverage=coverage)

        return {
            "ok": True,
            "model_id": model_id,
            "date": self.store.date_of(last_idx),
            "next_date": target_date,
            "next_idx": next_idx,
            "last_obs_idx": int(last_idx),
            "u": float(pred_u),
            "v": float(pred_v),
            "speed": float(speed),
            "direction_deg": float(direction),
            "displacement_km_24h": float(KM_PER_CM_S_24H * speed),
            "delta_speed_vs_persistence": float(speed - pers_speed),
            "input": {
                "row": int(row), "col": int(col), "lat": lat, "lon": lon,
                "date": self.store.date_of(last_idx),
                "u": float(cur_u), "v": float(cur_v), "speed": float(cur_speed),
                "direction_deg": _direction_deg(cur_u, cur_v),
                "valid": cur_valid, "coverage": coverage,
            },
            "uncertainty": self._uncertainty(model_id, pred, X),
            "warnings": warnings,
        }

    # ------------------------------------------------------------ 场预测
    def predict_field(self, model_id: str, date: str) -> dict:
        """预测 (date) 的下一帧整场。返回 (H,W) 数组（无效像元 NaN）+ coverage + warnings。"""
        model_id = self._validate_model(model_id)
        last_idx = self._resolve_last_idx(date)
        target_date = str(next_calendar_day(self.store.dates[last_idx]))
        next_idx = self.store.idx_of_date(target_date)
        if model_id in ML_MODEL_IDS:
            self._ensure_loaded(model_id)
        rows, cols = self.store.valid_cells(int(last_idx))
        X = self._features(last_idx, rows, cols) if model_id in ML_MODEL_IDS else None
        pred = self._predict_rows(model_id, last_idx, rows, cols, X)
        u = np.full((GRID_H, GRID_W), np.nan, np.float32)
        v = np.full((GRID_H, GRID_W), np.nan, np.float32)
        if rows.size:
            u[rows, cols] = pred[:, 0]
            v[rows, cols] = pred[:, 1]
        speed = np.hypot(u, v).astype(np.float32)
        coverage = float(rows.size) / float(GRID_H * GRID_W)
        warnings: list[dict] = []
        if coverage < LOW_COVERAGE_THRESHOLD:
            warnings.append({"code": "low_coverage", "level": "warning",
                             "message": f"当日有效覆盖仅 {coverage:.1%}，预测场稀疏"})
        return {
            "ok": True,
            "model_id": model_id,
            "date": self.store.date_of(last_idx),
            "target_date": target_date,
            "target_idx": next_idx,
            "u": u,
            "v": v,
            "speed": speed,
            "coverage": coverage,
            "n_valid_cells": int(rows.size),
            "warnings": warnings,
        }

    # ------------------------------------------------------------ 回测
    def backtest(self, model_id: str, row: int, col: int,
                 start: str | None = None, end: str | None = None,
                 max_points: int = 180) -> dict:
        """逐日历史回测（真实 vs 预测）。series[i].date 为目标日（预测日）。"""
        model_id = self._validate_model(model_id)
        row, col = self._validate_cell(row, col)
        if model_id in ML_MODEL_IDS:
            self._ensure_loaded(model_id)
        try:
            max_points = int(max_points)
        except (TypeError, ValueError):
            raise DateError(f"max_points 非法: {max_points!r}")
        max_points = max(1, min(max_points, 365))

        latest = self.store.date_of(self.store.n_frames - 1)
        if end is None:
            end = latest
        try:
            target_hi = int(self.store.nearest_idx(end))
        except Exception:  # pragma: no cover
            raise DateError(f"end 日期非法: {end!r}")
        if start is None:
            target_lo = max(N_HISTORY, target_hi - 179)
        else:
            try:
                target_lo = int(self.store.nearest_idx(start))
            except Exception:  # pragma: no cover
                raise DateError(f"start 日期非法: {start!r}")
        avail_lo = N_HISTORY
        avail_hi = self.store.n_frames - 1
        target_lo = max(avail_lo, min(target_lo, avail_hi))
        target_hi = max(avail_lo, min(target_hi, avail_hi))
        if target_hi < target_lo:
            raise DateError("回测区间无有效目标日",
                            hint=f"可用目标日 {self.store.date_of(avail_lo)} ~ {self.store.date_of(avail_hi)}")
        targets = np.arange(target_lo, target_hi + 1, dtype=np.int64)
        one_day = self.store.dates[targets] - self.store.dates[targets - 1] == np.timedelta64(1, "D")
        targets = targets[one_day]
        n_days_total = len(targets)
        if not n_days_total:
            raise DateError("回测区间没有相邻观测日")
        if n_days_total > max_points:
            picks = np.linspace(0, n_days_total - 1, max_points).round().astype(int)
            targets = targets[picks]
        # 批量构造特征（每个目标日 j 从 j-1 帧出发）
        X = None
        if model_id in ML_MODEL_IDS:
            Xs = np.empty((targets.size, 34), np.float32)
            for i, j in enumerate(targets):
                win = self.store.window(int(j) - 1, N_HISTORY)
                Xs[i] = build_features(win, np.array([row]), np.array([col]))[0]
            X = Xs
        # 预测（基线逐目标日取不同帧）
        preds = np.empty((targets.size, 2), np.float64)
        pers = np.empty((targets.size, 2), np.float64)
        obs = np.empty((targets.size, 2), np.float64)
        obs_valid = np.zeros(targets.size, bool)
        for i, j in enumerate(targets):
            preds[i] = self._predict_rows(model_id, int(j) - 1, [row], [col],
                                          None if X is None else X[i:i + 1])[0]
            pers[i] = self._predict_rows("persistence", int(j) - 1, [row], [col], None)[0]
            fj = self.store.frame(int(j))
            obs[i] = [float(fj["u"][row, col]), float(fj["v"][row, col])]
            obs_valid[i] = bool(fj["mask"][row, col] > 0)

        valid = obs_valid & np.isfinite(preds).all(axis=1)
        summary: dict = {"n": int(valid.sum())}
        if valid.any():
            summary.update(vec_metrics(preds[valid], obs[valid], pers=pers[valid]))
        else:  # pragma: no cover — 回测区间内该像元全无效
            summary.update({k: None for k in ("rmse_u", "rmse_v", "vec_rmse", "mae_vec",
                                              "r2_u", "r2_v", "skill_vs_persistence")})
        series = []
        for i, j in enumerate(targets):
            err = np.nan
            if obs_valid[i] and np.isfinite(preds[i]).all():
                err = float(np.hypot(preds[i, 0] - obs[i, 0], preds[i, 1] - obs[i, 1]))
            series.append({
                "date": self.store.date_of(int(j)),
                "obs_u": _finite(obs[i, 0]) if obs_valid[i] else None,
                "obs_v": _finite(obs[i, 1]) if obs_valid[i] else None,
                "obs_speed": _finite(_vec_speed(obs[i, 0], obs[i, 1])) if obs_valid[i] else None,
                "pred_u": round(float(preds[i, 0]), 4),
                "pred_v": round(float(preds[i, 1]), 4),
                "pred_speed": round(_vec_speed(preds[i, 0], preds[i, 1]), 4),
                "err_vec": round(err, 4) if np.isfinite(err) else None,
            })
        requested_start = start if start is not None else self.store.date_of(int(targets[0]))
        requested_end = end if end is not None else latest
        warnings: list[dict] = []
        clipped = (start is not None and int(self.store.nearest_idx(start)) < avail_lo) or \
                  (int(self.store.nearest_idx(end)) > avail_hi)
        if clipped:
            warnings.append({
                "code": "range_clipped", "level": "info",
                "message": (f"请求区间已裁剪到可用范围 "
                            f"{self.store.date_of(avail_lo)} ~ {self.store.date_of(avail_hi)}"),
            })
        if not obs_valid.any():
            warnings.append({"code": "no_obs", "level": "warning",
                             "message": "所选像元在回测区间内没有有效观测"})
        for k in ("rmse_u", "rmse_v", "vec_rmse", "mae_vec", "r2_u", "r2_v", "skill_vs_persistence"):
            if summary.get(k) is not None:
                summary[k] = round(float(summary[k]), 4)
        return {
            "ok": True,
            "model_id": model_id,
            "cell": {"row": int(row), "col": int(col),
                     "lat": self.store.cell_latlon(row, col)[0],
                     "lon": self.store.cell_latlon(row, col)[1]},
            "range": {
                "requested_start": requested_start,
                "requested_end": requested_end,
                "available_start": self.store.date_of(avail_lo),
                "available_end": self.store.date_of(avail_hi),
                "n_points": int(targets.size),
                "n_days_total": int(n_days_total),
            },
            "series": series,
            "summary": summary,
            "warnings": warnings,
        }

    # ------------------------------------------------------------ 预热 / 基准
    def warmup(self, date: str = DEFAULT_POINT["date"],
               row: int = DEFAULT_POINT["row"], col: int = DEFAULT_POINT["col"]) -> dict:
        """预热：加载 all 模型 + 解码默认日期所在 chunk + 每个模型跑一次单点预测。"""
        status: dict = {"loaded": [], "errors": {}, "elapsed_ms": {}}
        t_all = time.perf_counter()
        for m in MODEL_IDS:
            try:
                if m in ML_MODEL_IDS:
                    self._ensure_loaded(m)
                elif m == "climatology":
                    self._ensure_clim()
                self._load_cell_stats()
                t0 = time.perf_counter()
                self.predict_point(m, date, row, col)
                status["loaded"].append(m)
                status["elapsed_ms"][m] = round((time.perf_counter() - t0) * 1000, 2)
            except Exception as e:  # noqa: BLE001
                status["errors"][m] = f"{type(e).__name__}: {e}"
        status["total_ms"] = round((time.perf_counter() - t_all) * 1000, 2)
        return status

    def benchmark_point(self, model_id: str, date: str = DEFAULT_POINT["date"],
                        row: int = DEFAULT_POINT["row"], col: int = DEFAULT_POINT["col"],
                        n: int = 30, warmup: int = 3) -> dict:
        """单点预测延迟基准（毫秒；含特征构建，不含 HTTP 开销）。"""
        model_id = self._validate_model(model_id)
        if model_id in ML_MODEL_IDS:
            self._ensure_loaded(model_id)
        for _ in range(max(1, warmup)):
            self.predict_point(model_id, date, row, col)
        times = []
        for _ in range(max(1, n)):
            t0 = time.perf_counter()
            self.predict_point(model_id, date, row, col)
            times.append((time.perf_counter() - t0) * 1000.0)
        arr = np.asarray(times)
        return {
            "model_id": model_id,
            "n": int(arr.size),
            "median_ms": round(float(np.median(arr)), 3),
            "mean_ms": round(float(arr.mean()), 3),
            "p95_ms": round(float(np.percentile(arr, 95)), 3),
            "min_ms": round(float(arr.min()), 3),
            "max_ms": round(float(arr.max()), 3),
        }

    def benchmark_field(self, model_id: str, date: str = DEFAULT_POINT["date"],
                        n: int = 3) -> dict:
        """整场预测延迟基准（毫秒）。"""
        model_id = self._validate_model(model_id)
        if model_id in ML_MODEL_IDS:
            self._ensure_loaded(model_id)
        self.predict_field(model_id, date)
        times = []
        for _ in range(max(1, n)):
            t0 = time.perf_counter()
            self.predict_field(model_id, date)
            times.append((time.perf_counter() - t0) * 1000.0)
        arr = np.asarray(times)
        return {"model_id": model_id, "n": int(arr.size),
                "median_ms": round(float(np.median(arr)), 2),
                "min_ms": round(float(arr.min()), 2),
                "max_ms": round(float(arr.max()), 2)}

    def __repr__(self) -> str:  # pragma: no cover
        return (f"PredictionEngine(store={self.store!r}, model_dir={self.model_dir}, "
                f"artifact_dir={self.artifact_dir})")
