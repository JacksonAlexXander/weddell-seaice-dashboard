"""ArtifactStore：模型引擎 / 指标 artifacts / zarr 数据访问的统一入口。

设计目标
--------
1. **启动绝不崩**：``src.predict`` / ``src.serving`` / artifacts 缺失时只记录错误，
   ``/api/health`` 仍正常，``/api/models`` 标 ``available:false``，预测端点返回 409/503。
2. **懒加载 + 线程安全**：引擎在首次用到时（或启动后台 warmup）加载，之后复用；
   每次 ``/api/predict`` 只做推理，绝不训练、绝不写 ``models/``、``data/``。
3. **artifact 热更新**：JSON/NPZ 按 mtime 缓存，Lead 最后合并出 ``metrics.json`` /
   ``overview.json`` 后无需重启服务。
4. **数据访问降级**：优先用 ``prediction_engine.store``（设计 §5.1），不可用时回退到
   内置 :class:`ZarrFrameReader`（直接只读 zarr，接口同构）。
"""

from __future__ import annotations

import copy
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import (
    DATASET_DOI,
    DATASET_SOURCE,
    MODEL_IDS,
    MODEL_CATALOG,
    TABULAR_MODEL_IDS,
    GRID_COLS,
    GRID_ROWS,
    Settings,
    get_settings,
    model_catalog_map,
)

logger = logging.getLogger("backend.app.store")

_DEEP_ID = "simvp"


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(ts)))
    except (OverflowError, OSError, ValueError):
        return None


def _file_iso(path: Path) -> str | None:
    try:
        return _iso(path.stat().st_mtime)
    except OSError:
        return None


def _as_float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


def _error_warning(code: str, message: str) -> dict:
    return {"code": code, "level": "info", "message": message}


# ---------------------------------------------------------------------------
# fallback 数据读取器（引擎不可用时使用；接口与 src.predict.DataStore 同构）
# ---------------------------------------------------------------------------
class ZarrFrameReader:
    """只读 zarr 的轻量帧读取器（线程安全、懒打开、无全量读入）。

    对外接口与设计 §5.1 的 ``DataStore`` 子集兼容：
    ``n_frames`` / ``dates`` / ``latitude`` / ``longitude`` / ``idx_of_date`` /
    ``next_idx`` / ``frame`` / ``valid_cells`` / ``data_time``。
    """

    def __init__(self, zarr_path: Path | str):
        self.path = Path(zarr_path)
        self._lock = threading.RLock()
        self._z = None
        self._dates: np.ndarray | None = None
        self._lat: np.ndarray | None = None
        self._lon: np.ndarray | None = None
        self._doy: np.ndarray | None = None
        self._valid_fraction: np.ndarray | None = None
        self._n_frames = 0
        self._built_at: str | None = None
        self._attrs: dict = {}

    # -- lazy open ---------------------------------------------------------
    def _ensure(self) -> None:
        if self._z is not None:
            return
        with self._lock:
            if self._z is not None:
                return
            import zarr  # 延迟 import：zarr 缺失也不影响服务启动

            z = zarr.open_group(str(self.path), mode="r")
            time_arr = np.asarray(z["time"][:])
            self._dates = time_arr.astype("datetime64[D]")
            self._n_frames = int(self._dates.shape[0])
            self._doy = np.asarray(z["doy"][:], dtype=np.float32)
            self._valid_fraction = np.asarray(z["valid_fraction"][:], dtype=np.float32)
            self._lat = np.asarray(z["latitude"][:], dtype=np.float32)
            self._lon = np.asarray(z["longitude"][:], dtype=np.float32)
            try:
                attrs = dict(z.attrs)
            except Exception:  # noqa: BLE001
                attrs = {}
            self._attrs = attrs
            built = attrs.get("created_utc")
            self._built_at = str(built) if built else None
            self._z = z

    # -- 元数据 ------------------------------------------------------------
    @property
    def n_frames(self) -> int:
        self._ensure()
        return self._n_frames

    @property
    def dates(self) -> np.ndarray:
        self._ensure()
        assert self._dates is not None
        return self._dates

    @property
    def latitude(self) -> np.ndarray:
        self._ensure()
        assert self._lat is not None
        return self._lat

    @property
    def longitude(self) -> np.ndarray:
        self._ensure()
        assert self._lon is not None
        return self._lon

    @property
    def attrs(self) -> dict:
        self._ensure()
        return dict(self._attrs)

    def date_str(self, idx: int) -> str:
        return str(self.dates[int(idx)])

    def idx_of_date(self, date: str) -> int | None:
        try:
            d = np.datetime64(str(date), "D")
        except (TypeError, ValueError):
            return None
        hit = np.where(self.dates == d)[0]
        return int(hit[0]) if len(hit) else None

    def next_idx(self, idx: int) -> int:
        nxt = int(idx) + 1
        if nxt >= self.n_frames:
            raise ValueError("no_future_frame")
        return nxt

    def valid_cells(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        frame = self.frame(idx)
        rows, cols = np.where(frame["mask"])
        return rows, cols

    # -- 帧读取 ------------------------------------------------------------
    def frame(self, idx: int) -> dict:
        """读取单帧 → {u,v,speed,mask,doy,date,coverage}（均为原始单位 cm/s）。"""
        self._ensure()
        i = int(idx)
        with self._lock:
            z = self._z
            assert z is not None
            u = np.asarray(z["u"][i], dtype=np.float32)
            v = np.asarray(z["v"][i], dtype=np.float32)
            mask = np.asarray(z["valid_mask"][i], dtype=bool)
            speed_raw = z["speed"]
            try:
                speed = np.asarray(speed_raw[i], dtype=np.float32)
            except Exception:  # noqa: BLE001
                speed = np.hypot(u, v).astype(np.float32)
            doy = float(self._doy[i]) if self._doy is not None else 0.0
            coverage = float(self._valid_fraction[i]) if self._valid_fraction is not None else (
                float(mask.mean()) if mask.size else 0.0
            )
        return {
            "u": u,
            "v": v,
            "speed": speed,
            "mask": mask,
            "doy": doy,
            "date": str(self._dates[i]),
            "coverage": float(coverage),
        }

    def data_time(self) -> dict:
        self._ensure()
        latest = str(self.dates[-1])
        try:
            age = int((np.datetime64("today", "D") - np.datetime64(latest, "D")).astype(int))
        except (TypeError, ValueError):
            age = None
        return {
            "first_date": str(self.dates[0]),
            "latest_date": latest,
            "n_frames": int(self.n_frames),
            "built_at": self._built_at,
            "batch_age_days": age,
            "source": DATASET_SOURCE,
            "doi": DATASET_DOI,
        }


class _EngineStoreAdapter:
    """包一层 ``PredictionEngine.store``，字段/方法缺失时逐项回退 reader。"""

    def __init__(self, inner: Any, reader: ZarrFrameReader):
        self._inner = inner
        self._reader = reader

    @property
    def n_frames(self) -> int:
        value = getattr(self._inner, "n_frames", None)
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
        return self._reader.n_frames

    @property
    def dates(self) -> np.ndarray:
        try:
            dates = getattr(self._inner, "dates")
            if dates is not None and np.asarray(dates).ndim == 1:
                return np.asarray(dates).astype("datetime64[D]")
        except Exception:  # noqa: BLE001
            pass
        return self._reader.dates

    @property
    def latitude(self) -> np.ndarray:
        try:
            lat = getattr(self._inner, "latitude")
            if lat is not None:
                return np.asarray(lat, dtype=np.float32)
        except Exception:  # noqa: BLE001
            pass
        return self._reader.latitude

    @property
    def longitude(self) -> np.ndarray:
        try:
            lon = getattr(self._inner, "longitude")
            if lon is not None:
                return np.asarray(lon, dtype=np.float32)
        except Exception:  # noqa: BLE001
            pass
        return self._reader.longitude

    def date_str(self, idx: int) -> str:
        try:
            return str(self.dates[int(idx)])
        except Exception:  # noqa: BLE001
            return self._reader.date_str(idx)

    def idx_of_date(self, date: str) -> int | None:
        try:
            value = self._inner.idx_of_date(date)
            if value is not None:
                return int(value)
            return None
        except Exception:  # noqa: BLE001
            return self._reader.idx_of_date(date)

    def next_idx(self, idx: int) -> int:
        try:
            return int(self._inner.next_idx(int(idx)))
        except ValueError:
            raise
        except Exception:  # noqa: BLE001
            return self._reader.next_idx(idx)

    def frame(self, idx: int) -> dict:
        try:
            raw = self._inner.frame(int(idx))
            return self._normalize(raw, int(idx))
        except Exception:  # noqa: BLE001
            return self._reader.frame(idx)

    def _normalize(self, raw: Any, idx: int) -> dict:
        if not isinstance(raw, dict):
            return self._reader.frame(idx)
        u = np.asarray(raw.get("u"), dtype=np.float32)
        v = np.asarray(raw.get("v"), dtype=np.float32)
        if u.shape != (GRID_ROWS, GRID_COLS) or v.shape != (GRID_ROWS, GRID_COLS):
            return self._reader.frame(idx)
        mask_raw = raw.get("mask", raw.get("valid_mask"))
        if mask_raw is None:
            mask = np.isfinite(u) & np.isfinite(v)
        else:
            mask = np.asarray(mask_raw, dtype=bool)
        speed_raw = raw.get("speed")
        speed = (
            np.asarray(speed_raw, dtype=np.float32)
            if speed_raw is not None and np.shape(speed_raw) == (GRID_ROWS, GRID_COLS)
            else np.hypot(u, v).astype(np.float32)
        )
        coverage = _as_float(raw.get("coverage"))
        if coverage is None:
            coverage = float(mask.mean())
        date = raw.get("date") or self.date_str(idx)
        doy = raw.get("doy")
        return {
            "u": u,
            "v": v,
            "speed": speed,
            "mask": mask,
            "doy": float(doy) if doy is not None else 0.0,
            "date": str(date),
            "coverage": float(coverage),
        }

    def data_time(self) -> dict:
        try:
            dt = self._inner.data_time()
            if isinstance(dt, dict) and dt.get("latest_date"):
                return dt
        except Exception:  # noqa: BLE001
            pass
        return self._reader.data_time()

    def valid_cells(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return np.where(self.frame(idx)["mask"])


# ---------------------------------------------------------------------------
# ArtifactStore
# ---------------------------------------------------------------------------
class ArtifactStore:
    """进程级单例（挂在 ``app.state.store``）：引擎 / artifacts / 数据的唯一访问点。"""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.started_at = time.time()

        # 引擎（懒加载，双检锁）
        self._engine_lock = threading.RLock()
        self._engine: Any = None
        self._engine_attempted = False
        self._engine_error: str | None = None
        self._engine_load_ms: float | None = None
        self._engine_generation = 0
        self._engine_snapshot: tuple | None = None

        # SimVP（懒加载）
        self._simvp_lock = threading.RLock()
        self._simvp: Any = None
        self._simvp_attempted = False
        self._simvp_error: str | None = None
        self._simvp_load_ms: float | None = None

        # 数据读取
        self._reader_lock = threading.RLock()
        self._reader: ZarrFrameReader | None = None

        # artifact 缓存（mtime 失效）
        self._json_lock = threading.RLock()
        self._json_cache: dict[str, tuple[float | None, Any]] = {}
        self._npz_lock = threading.RLock()
        self._npz_cache: dict[str, tuple[float | None, Any]] = {}
        self._data_time_lock = threading.RLock()
        self._data_time_cached: dict | None = None
        self._warmup_done = False

    # ==================================================================
    # 数据访问
    # ==================================================================
    def reader(self) -> ZarrFrameReader:
        """fallback 只读 zarr 读取器（同时用于经纬度等静态网格）。"""
        with self._reader_lock:
            if self._reader is None:
                self._reader = ZarrFrameReader(self.settings.zarr_path)
                self._reader._ensure()  # 早失败：zarr 缺失立即抛错，由上层转 503
            return self._reader

    def data_adapter(self):
        """观测数据访问器：优先 engine.store，回退 ZarrFrameReader。"""
        engine = self.engine_or_none()
        if engine is not None:
            inner = getattr(engine, "store", None)
            if inner is not None:
                try:
                    return _EngineStoreAdapter(inner, self.reader())
                except Exception:  # noqa: BLE001
                    logger.warning("engine.store 包装失败，回退 zarr reader", exc_info=True)
        return self.reader()

    def data_time(self) -> dict | None:
        """DataTime（优先引擎 / zarr，最后 artifacts 兜底）。"""
        with self._data_time_lock:
            if self._data_time_cached is not None:
                return copy.deepcopy(self._data_time_cached)

            dt: dict | None = None
            engine = self.engine_or_none()
            if engine is not None:
                try:
                    raw = engine.data_time()
                    if isinstance(raw, dict) and raw.get("latest_date"):
                        dt = self._normalize_data_time(raw)
                except Exception:  # noqa: BLE001
                    logger.warning("engine.data_time() 失败，回退 zarr", exc_info=True)
            if dt is None:
                try:
                    dt = self._normalize_data_time(self.reader().data_time())
                except Exception:  # noqa: BLE001
                    logger.warning("zarr data_time 读取失败", exc_info=True)
            if dt is None:
                # artifacts 兜底（timeseries / overview 里的 data_time）
                for name in ("overview.json", "timeseries.json"):
                    payload = self._read_json(name)
                    if isinstance(payload, dict) and isinstance(payload.get("data_time"), dict):
                        dt = self._normalize_data_time(payload["data_time"])
                        if dt.get("latest_date"):
                            break
                        dt = None
            if dt is None:
                return None
            self._data_time_cached = dt
            return copy.deepcopy(dt)

    def _normalize_data_time(self, raw: dict) -> dict:
        latest = raw.get("latest_date")
        first = raw.get("first_date")
        n_frames = raw.get("n_frames")
        batch_age = raw.get("batch_age_days")
        if batch_age is None and latest:
            try:
                batch_age = int((np.datetime64("today", "D") - np.datetime64(str(latest), "D")).astype(int))
            except (TypeError, ValueError):
                batch_age = None
        return {
            "first_date": str(first) if first else "",
            "latest_date": str(latest) if latest else "",
            "n_frames": int(n_frames) if n_frames is not None else 0,
            "built_at": raw.get("built_at"),
            "batch_age_days": int(batch_age) if batch_age is not None else None,
            "source": raw.get("source") or DATASET_SOURCE,
            "doi": raw.get("doi") or DATASET_DOI,
        }

    # ==================================================================
    # 引擎（tab 表格模型）
    # ==================================================================
    def engine_or_none(self):
        return self._engine

    def engine_error(self) -> str | None:
        return self._engine_error

    def ensure_engine(self):
        """懒加载 PredictionEngine；失败返回 None（错误记录在 engine_error）。"""
        if self._engine is not None:
            self.maybe_reload_engine()
            return self._engine
        with self._engine_lock:
            if self._engine is not None:
                return self._engine
            if self._engine_attempted and self._engine is None:
                # 已尝试失败：允许后续 warmup 再试一次
                pass
            t0 = time.perf_counter()
            try:
                from src.predict.engine import PredictionEngine  # 延迟 import

                engine = PredictionEngine(
                    zarr_path=str(self.settings.zarr_path),
                    model_dir=str(self.settings.models_ml_dir),
                    artifact_dir=str(self.settings.dashboard_dir),
                )
                self._engine = engine
                self._engine_error = None
                self._engine_load_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                self._engine_snapshot = self._artifact_snapshot()
                self._engine_generation += 1
                logger.info("PredictionEngine 已加载（%.1f ms）", self._engine_load_ms)
            except Exception as exc:  # noqa: BLE001
                self._engine = None
                self._engine_error = f"{type(exc).__name__}: {exc}"
                self._engine_load_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                logger.warning("PredictionEngine 加载失败：%s", self._engine_error)
            finally:
                self._engine_attempted = True
            return self._engine

    def simvp_or_none(self):
        return self._simvp

    def simvp_error(self) -> str | None:
        return self._simvp_error

    def ensure_simvp(self):
        """懒加载 SimVPPredictor；失败返回 None（错误记录在 simvp_error）。"""
        if self._simvp is not None:
            try:
                if self._simvp.state.get("loaded"):
                    return self._simvp
            except Exception:  # noqa: BLE001
                pass
        with self._simvp_lock:
            if self._simvp is not None and self._simvp.state.get("loaded"):
                return self._simvp
            t0 = time.perf_counter()
            try:
                from src.serving.simvp import SimVPPredictor  # 延迟 import

                predictor = SimVPPredictor(
                    zarr_path=str(self.settings.zarr_path),
                    ckpt=str(self.settings.simvp_ckpt),
                    precomputed=str(self.settings.dashboard_dir / "simvp_day1_fields.npz"),
                )
                state = predictor.load() if hasattr(predictor, "load") else {}
                self._simvp_load_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                if isinstance(state, dict) and state.get("loaded"):
                    self._simvp = predictor
                    self._simvp_error = None
                    logger.info(
                        "SimVPPredictor 已加载（device=%s, precomputed=%s, %.1f ms）",
                        state.get("device"),
                        state.get("precomputed"),
                        self._simvp_load_ms,
                    )
                else:
                    self._simvp = predictor
                    self._simvp_error = (
                        str(state.get("error")) if isinstance(state, dict) and state.get("error")
                        else "SimVPPredictor.state.loaded=False"
                    )
                    logger.warning("SimVPPredictor 加载失败：%s", self._simvp_error)
            except Exception as exc:  # noqa: BLE001
                self._simvp = None
                self._simvp_error = f"{type(exc).__name__}: {exc}"
                self._simvp_load_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                logger.warning("SimVPPredictor 加载失败：%s", self._simvp_error)
            finally:
                self._simvp_attempted = True
            return self._simvp if self._simvp_error is None else None

    # ==================================================================
    # 可用性
    # ==================================================================
    def _engine_module_present(self) -> bool:
        return (self.settings.workspace_root / "src/predict/engine.py").exists()

    def _simvp_module_present(self) -> bool:
        return (self.settings.workspace_root / "src/serving/simvp.py").exists()

    def _model_artifact_paths(self, model_id: str) -> tuple[Path, ...]:
        ml = self.settings.models_ml_dir
        dash = self.settings.dashboard_dir
        if model_id == "ridge":
            return (ml / "ridge.joblib",)
        if model_id == "random_forest":
            return (ml / "rf.joblib", ml / "random_forest.joblib")
        if model_id == "climatology":
            return (dash / "climatology.npz",)
        return ()

    def _provisional_model_available(self, model_id: str) -> bool:
        """引擎尚未加载时的乐观可用性（只看文件，毫秒级，避免首屏等待模型加载）。"""
        if not self._engine_module_present():
            return False
        paths = self._model_artifact_paths(model_id)
        if not paths:
            return model_id in TABULAR_MODEL_IDS  # persistence 无需 artifact
        # 任一候选文件存在即可（random_forest 兼容两种命名）
        if model_id == "random_forest" and len(paths) > 1:
            return any(p.exists() for p in paths)
        return paths[0].exists()

    def model_available(self, model_id: str) -> bool:
        if model_id not in MODEL_IDS:
            return False
        if model_id == _DEEP_ID:
            if self._simvp_attempted:
                simvp = self._simvp
                if simvp is None:
                    return False
                try:
                    return bool(simvp.state.get("loaded"))
                except Exception:  # noqa: BLE001
                    return False
            return self.settings.simvp_ckpt.exists() and self._simvp_module_present()
        # 表格模型
        if self._engine_attempted:
            engine = self._engine
            if engine is not None:
                self.maybe_reload_engine()
                engine = self._engine
            if engine is None:
                return False
            try:
                return bool(engine.available(model_id))
            except Exception:  # noqa: BLE001
                return False
        return self._provisional_model_available(model_id)

    def loaded_model_ids(self) -> list[str]:
        """已真实加载且可用的模型 id（按契约顺序）。"""
        out: list[str] = []
        engine = self._engine
        if engine is not None:
            engine_ids = getattr(engine, "model_ids", TABULAR_MODEL_IDS) or TABULAR_MODEL_IDS
            for model_id in engine_ids:
                try:
                    if engine.available(model_id):
                        out.append(model_id)
                except Exception:  # noqa: BLE001
                    continue
        simvp = self._simvp
        if simvp is not None:
            try:
                if simvp.state.get("loaded"):
                    out.append(_DEEP_ID)
            except Exception:  # noqa: BLE001
                pass
        order = {mid: i for i, mid in enumerate(MODEL_IDS)}
        return sorted(set(out), key=lambda m: order.get(m, 99))

    def availability_map(self) -> dict[str, bool]:
        return {model_id: self.model_available(model_id) for model_id in MODEL_IDS}

    def _artifact_snapshot(self) -> tuple:
        """模型/气候态 artifact 的 (name, mtime_ns) 快照，用于检测训练完成后的原子替换。"""
        s = self.settings
        paths = (
            s.models_ml_dir / "ridge.joblib",
            s.models_ml_dir / "rf.joblib",
            s.models_ml_dir / "model_card.json",
            s.models_ml_dir / "feature_spec.json",
            s.dashboard_dir / "climatology.npz",
            s.dashboard_dir / "cell_stats.npz",
        )
        snapshot = []
        for path in paths:
            try:
                snapshot.append((path.name, path.stat().st_mtime_ns))
            except OSError:
                snapshot.append((path.name, None))
        return tuple(snapshot)

    @property
    def engine_generation(self) -> int:
        """引擎代数：artifact 热更新后 +1（预测缓存据此自动失效）。"""
        return self._engine_generation

    def maybe_reload_engine(self) -> bool:
        """检测模型 artifact 是否被替换（如冒烟模型 → 正式训练模型），必要时 reload。

        只在文件 mtime 变化时触发（每次调用仅几次 stat），返回是否执行了 reload。
        """
        engine = self._engine
        if engine is None or not hasattr(engine, "reload"):
            return False
        snapshot = self._artifact_snapshot()
        if self._engine_snapshot is None:
            self._engine_snapshot = snapshot
            return False
        if snapshot == self._engine_snapshot:
            return False
        with self._engine_lock:
            snapshot = self._artifact_snapshot()
            if snapshot == self._engine_snapshot:
                return False
            try:
                engine.reload()
                try:
                    engine.warmup()
                except Exception:  # noqa: BLE001
                    logger.warning("engine.reload 后 warmup 失败", exc_info=True)
                self._engine_generation += 1
                self._engine_snapshot = self._artifact_snapshot()
                logger.info("检测到模型 artifact 更新，PredictionEngine 已 reload（generation=%s）",
                            self._engine_generation)
                return True
            except Exception:  # noqa: BLE001
                logger.warning("PredictionEngine reload 失败", exc_info=True)
                return False

    def engine_status(self) -> dict:
        return {
            "attempted": self._engine_attempted,
            "loaded": self._engine is not None,
            "error": self._engine_error,
            "load_ms": self._engine_load_ms,
            "module_present": self._engine_module_present(),
        }

    def simvp_status(self) -> dict:
        state = {}
        if self._simvp is not None:
            try:
                state = dict(self._simvp.state)
            except Exception:  # noqa: BLE001
                state = {}
        return {
            "attempted": self._simvp_attempted,
            "loaded": bool(state.get("loaded")),
            "error": self._simvp_error,
            "load_ms": self._simvp_load_ms,
            "module_present": self._simvp_module_present(),
            **{k: state.get(k) for k in ("device", "precomputed", "ckpt") if k in state},
        }

    # ==================================================================
    # JSON / NPZ artifacts
    # ==================================================================
    def _read_json_path(self, path: Path) -> Any:
        path = Path(path)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return None
        key = str(path)
        with self._json_lock:
            cached = self._json_cache.get(key)
            if cached is not None and cached[0] == mtime:
                return cached[1]
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("JSON artifact 读取失败 %s: %s", path, exc)
            data = None
        with self._json_lock:
            self._json_cache[key] = (mtime, data)
        return data

    def _read_json(self, name: str) -> Any:
        return self._read_json_path(self.settings.dashboard_dir / name)

    def _read_npz_path(self, path: Path) -> dict | None:
        path = Path(path)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return None
        key = str(path)
        with self._npz_lock:
            cached = self._npz_cache.get(key)
            if cached is not None and cached[0] == mtime:
                return cached[1]
        try:
            with np.load(path, allow_pickle=False) as data:
                arrays = {name: np.array(data[name]) for name in data.files}
        except Exception as exc:  # noqa: BLE001
            logger.warning("NPZ artifact 读取失败 %s: %s", path, exc)
            arrays = None
        with self._npz_lock:
            self._npz_cache[key] = (mtime, arrays)
        return arrays

    # -- metrics -----------------------------------------------------------
    @staticmethod
    def _metrics_entries(source: Any) -> list[dict]:
        """兼容 artifacts 的两种结构：``{"models": [...]}`` 或 ``{"model": {...}}``。"""
        if not isinstance(source, dict):
            return []
        models = source.get("models")
        if isinstance(models, list):
            return [m for m in models if isinstance(m, dict) and m.get("id")]
        model = source.get("model")
        if isinstance(model, dict) and model.get("id"):
            return [model]
        return []

    @staticmethod
    def _merge_protocol(protocol: dict | None, fallback: dict | None) -> tuple[dict | None, bool]:
        """用兜底 protocol 补齐缺失字段（不覆盖已有值），返回 (合并结果, 是否补齐)。"""
        if not fallback:
            return protocol, False
        if not isinstance(protocol, dict):
            return copy.deepcopy(fallback), True
        out = copy.deepcopy(protocol)
        filled = False
        for key, value in fallback.items():
            if key not in out or out[key] is None:
                out[key] = copy.deepcopy(value)
                filled = True
            elif isinstance(value, dict) and isinstance(out[key], dict):
                for sub_key, sub_value in value.items():
                    if sub_key not in out[key] or out[key][sub_key] is None:
                        out[key][sub_key] = copy.deepcopy(sub_value)
                        filled = True
        return out, filled

    def metrics(self) -> tuple[dict | None, list[dict]]:
        """返回 (metrics payload, warnings)。

        优先 ``metrics.json``（Lead 最终合并）；缺失/不完整时回退
        ``metrics_tabular.json`` 并可合并 ``metrics_simvp.json``，
        保证 ``models`` 为契约要求的数组（5 个模型按契约顺序）。
        """
        warnings: list[dict] = []
        primary = self._read_json("metrics.json")
        simvp_metrics = self._read_json("metrics_simvp.json")
        tabular_metrics = self._read_json("metrics_tabular.json")

        payload: dict | None = None
        used_primary = False
        if isinstance(primary, dict) and (
            self._metrics_entries(primary) or isinstance(primary.get("protocol"), dict)
        ):
            payload = copy.deepcopy(primary)
            used_primary = True
        elif isinstance(tabular_metrics, dict):
            payload = copy.deepcopy(tabular_metrics)
            warnings.append(
                _error_warning(
                    "metrics_fallback",
                    "metrics.json 尚未生成（Lead 最终合并中），当前回退使用 metrics_tabular.json",
                )
            )
        if payload is None and isinstance(simvp_metrics, dict):
            simvp_entries = self._metrics_entries(simvp_metrics)
            payload = {
                "generated_at": simvp_metrics.get("generated_at") or _iso(time.time()),
                "models": copy.deepcopy(simvp_entries),
            }
            warnings.append(
                _error_warning(
                    "metrics_fallback",
                    "metrics.json / metrics_tabular.json 尚未生成，当前仅返回 SimVP 指标",
                )
            )
        if payload is None:
            return None, warnings

        # 统一成 models 数组并补齐缺失模型
        entries: list[dict] = []
        seen: set[str] = set()
        for entry in self._metrics_entries(payload):
            entries.append(entry)
            seen.add(str(entry["id"]))
        for source, source_name in ((tabular_metrics, "metrics_tabular.json"),
                                    (simvp_metrics, "metrics_simvp.json")):
            for entry in self._metrics_entries(source):
                model_id = str(entry["id"])
                if model_id in seen:
                    continue
                entries.append(copy.deepcopy(entry))
                seen.add(model_id)
                if used_primary:
                    warnings.append(
                        _error_warning(
                            "metrics_partial_merge",
                            f"metrics.json 缺少 {model_id} 指标，已从 {source_name} 补齐",
                        )
                    )
        order = {mid: i for i, mid in enumerate(MODEL_IDS)}
        entries.sort(key=lambda e: order.get(str(e.get("id")), 99))
        payload["models"] = entries

        # protocol 补齐（split/metric_defs 等字段前端直接展示）
        protocol, filled = self._merge_protocol(
            payload.get("protocol") if isinstance(payload.get("protocol"), dict) else None,
            self._fallback_protocol(),
        )
        if protocol is not None:
            payload["protocol"] = protocol
        if filled:
            warnings.append(
                _error_warning(
                    "protocol_fallback",
                    "指标 artifact 的 protocol 缺少部分字段，已按设计文档补齐统一评估口径",
                )
            )
        if not payload.get("generated_at"):
            payload["generated_at"] = _iso(time.time())
        if warnings:
            payload["warnings"] = warnings
        return payload, warnings

    def metrics_models_map(self) -> dict[str, dict]:
        payload, _ = self.metrics()
        out: dict[str, dict] = {}
        if not isinstance(payload, dict):
            return out
        for entry in payload.get("models") or []:
            if isinstance(entry, dict) and entry.get("id"):
                out[str(entry["id"])] = entry
        return out

    def _fallback_protocol(self) -> dict | None:
        windows = self._read_json("eval_windows.json")
        n_windows = None
        if isinstance(windows, dict) and isinstance(windows.get("windows"), list):
            n_windows = len(windows["windows"])
        elif isinstance(windows, list):
            n_windows = len(windows)
        return {
            "task": "未来 24 小时（下一日帧）海冰漂移矢量预测",
            "target": "u(t+1), v(t+1)，cm/s，25 km EASE-Grid South 像元",
            "split": {"train": "1978–2017", "val": "2018–2021", "test": "2022–2025"},
            "sampling": f"test split 每 5 日一个起报窗口（{n_windows or 287} 个窗口），评估全部有效像元",
            "n_windows": n_windows or 287,
            "n_samples": 1556493,
            "metric_defs": {
                "vec_rmse": "√mean(Δu²+Δv²)，cm/s",
                "mae_vec": "mean(√(Δu²+Δv²))，cm/s",
                "skill_vs_persistence": "1 − RMSE_model/RMSE_persistence（矢量 RMSE 比口径，与项目 skill_pers 一致）",
                "r2_u": "u 分量决定系数",
            },
        }

    # -- overview ----------------------------------------------------------
    def overview(self) -> tuple[dict | None, list[dict]]:
        """返回 (overview payload, warnings)；优先 overview.json，否则由 timeseries.json 组装。"""
        warnings: list[dict] = []
        payload = self._read_json("overview.json")
        if isinstance(payload, dict) and (payload.get("dataset") or payload.get("annual")):
            out = copy.deepcopy(payload)
            if not isinstance(out.get("data_time"), dict):
                out["data_time"] = self.data_time()
            return out, warnings

        ts = self._read_json("timeseries.json")
        if not isinstance(ts, dict):
            return None, warnings
        out: dict = {}
        for key in ("dataset", "kpis", "annual", "monthly", "year_month", "data_time"):
            if key in ts:
                out[key] = copy.deepcopy(ts[key])
        data_time = self.data_time() or {}
        if not isinstance(out.get("dataset"), dict):
            out["dataset"] = self._dataset_fallback(ts, data_time)
        if not isinstance(out.get("kpis"), list):
            out["kpis"] = self._kpis_fallback(out["dataset"], data_time)
        out.setdefault("annual", ts.get("yearly") or ts.get("annual_trend") or [])
        out.setdefault("monthly", ts.get("monthly_climatology") or [])
        if "year_month" not in out:
            out["year_month"] = ts.get("year_month_matrix") or {}
        if not isinstance(out.get("data_time"), dict):
            out["data_time"] = data_time
        warnings.append(
            _error_warning(
                "overview_fallback",
                "overview.json 尚未生成（Lead 最终合并中），当前由 timeseries.json + 数据时间动态组装",
            )
        )
        out["warnings"] = warnings
        return out, warnings

    def _dataset_fallback(self, ts: dict, data_time: dict) -> dict:
        attrs: dict = {}
        try:
            attrs = self.reader().attrs
        except Exception:  # noqa: BLE001
            pass
        first = data_time.get("first_date") or ""
        latest = data_time.get("latest_date") or ""
        years = None
        try:
            years = int(np.datetime64(latest, "Y").astype(int) - np.datetime64(first, "Y").astype(int)) + 1
        except (TypeError, ValueError):
            pass
        meta = ts.get("dataset") if isinstance(ts.get("dataset"), dict) else {}
        return {
            "name": meta.get("name") or "NSIDC-0116 v4 威德尔海海冰运动数据集",
            "region": meta.get("region") or "60°W–20°E, 60°S–83°S",
            "grid": meta.get("grid") or {"rows": GRID_ROWS, "cols": GRID_COLS, "cell_km": self.settings.cell_km},
            "n_frames": data_time.get("n_frames"),
            "first_date": first or None,
            "latest_date": latest or None,
            "years": years,
            "missing_days": int(attrs.get("missing_days_n")) if attrs.get("missing_days_n") else None,
            "mean_valid_fraction": meta.get("mean_valid_fraction"),
            "mean_speed_cm_s": meta.get("mean_speed_cm_s"),
            "u_units": "cm/s",
            "source": "NSIDC-0116 v4",
        }

    def _kpis_fallback(self, dataset: dict, data_time: dict) -> list[dict]:
        return [
            {"id": "years", "label": "覆盖年限", "value": dataset.get("years"), "unit": "年"},
            {"id": "days", "label": "有效日数", "value": data_time.get("n_frames"), "unit": "天"},
            {"id": "mean_speed", "label": "平均漂移速度",
             "value": dataset.get("mean_speed_cm_s"), "unit": "cm/s"},
            {"id": "coverage", "label": "平均有效覆盖",
             "value": round(float(dataset["mean_valid_fraction"]) * 100, 1)
             if isinstance(dataset.get("mean_valid_fraction"), (int, float)) else None,
             "unit": "%"},
        ]

    # -- model card --------------------------------------------------------
    def model_card(self) -> dict:
        card = self._read_json_path(self.settings.models_ml_dir / "model_card.json")
        return card if isinstance(card, dict) else {}

    def model_meta(self) -> dict[str, dict]:
        """每个模型的人工元数据：trained_at / training_samples / latency_ms。"""
        card = self.model_card()
        containers: list[dict] = [card]
        for key in ("models", "model_cards", "trained", "cards", "per_model", "training"):
            value = card.get(key)
            if isinstance(value, dict):
                containers.append(value)
            elif isinstance(value, list):
                for entry in value:
                    if isinstance(entry, dict) and entry.get("id"):
                        containers.append({str(entry["id"]): entry})
        out: dict[str, dict] = {}
        for model_id in MODEL_IDS:
            entry: dict = {}
            for container in containers:
                cand = container.get(model_id)
                if isinstance(cand, dict):
                    entry = cand
                    break
            trained_at = None
            for key in ("trained_at", "created_at", "trained", "timestamp", "date"):
                if entry.get(key):
                    trained_at = str(entry[key])
                    break
            if trained_at is None:
                joblib = self._model_artifact_paths(model_id)
                for path in joblib:
                    if path.exists():
                        trained_at = _file_iso(path)
                        break
            training_samples = None
            for key in ("training_samples", "n_samples", "n_train", "samples", "train_samples"):
                if isinstance(entry.get(key), (int, float)):
                    training_samples = int(entry[key])
                    break
            latency = None
            metrics = entry.get("metrics")
            if isinstance(metrics, dict):
                latency = _as_float(metrics.get("latency_ms"))
            latency = latency if latency is not None else _as_float(entry.get("latency_ms"))
            out[model_id] = {
                "trained_at": trained_at,
                "training_samples": training_samples,
                "latency_ms": latency,
            }
        if "simvp" in out and not out["simvp"].get("trained_at"):
            out["simvp"]["trained_at"] = _file_iso(self.settings.simvp_ckpt)
        # 冒烟/初期 model_card.json 可能是顶层扁平结构（created_at / training_samples）
        card_created = card.get("created_at") or card.get("trained_at") or card.get("created")
        card_samples = card.get("training_samples") or card.get("n_samples")
        for model_id in ("ridge", "random_forest"):
            if out[model_id]["trained_at"] is None and card_created:
                out[model_id]["trained_at"] = str(card_created)
            if out[model_id]["training_samples"] is None and isinstance(card_samples, (int, float)):
                out[model_id]["training_samples"] = int(card_samples)
        return out

    # ==================================================================
    # 像元统计 / 指标辅助
    # ==================================================================
    def cell_stats(self, row: int, col: int) -> dict | None:
        """逐像元历史统计（speed_mean/std/p99/valid_days/lat/lon）。"""
        engine = self.engine_or_none()
        if engine is not None:
            try:
                stats = engine.cell_stats(int(row), int(col))
                if isinstance(stats, dict) and stats:
                    return {k: v for k, v in stats.items()}
            except Exception:  # noqa: BLE001
                logger.debug("engine.cell_stats 失败，回退 npz", exc_info=True)
        arrays = self._read_npz_path(self.settings.dashboard_dir / "cell_stats.npz")
        if not arrays:
            return None
        return _lookup_cell_stats(arrays, int(row), int(col))

    def simvp_backtest_range(self) -> tuple[str, str] | None:
        """SimVP 预计算场覆盖的目标日范围（供契约 backtest_out_of_range 提示用）。"""
        predictor = self.simvp_or_none()
        if predictor is not None:
            pre = getattr(predictor, "_pre", None)
            if isinstance(pre, dict):
                dates = pre.get("dates")
                try:
                    if dates is not None and len(dates):
                        arr = np.asarray(dates).astype("datetime64[D]")
                        return str(arr.min()), str(arr.max())
                except Exception:  # noqa: BLE001
                    pass
        path = self.settings.dashboard_dir / "simvp_day1_fields.npz"
        try:
            with np.load(path, allow_pickle=False) as data:
                if "dates" in data.files:
                    arr = np.asarray(data["dates"]).astype("datetime64[D]")
                    if arr.size:
                        return str(arr.min()), str(arr.max())
        except Exception:  # noqa: BLE001
            pass
        return None

    def model_metrics(self, model_id: str) -> dict | None:
        """契约 predict.metrics：{vec_rmse, skill_vs_persistence}（可 None）。"""
        entry = self.metrics_models_map().get(model_id)
        if not entry:
            return None
        return {
            "vec_rmse": _as_float(entry.get("vec_rmse")),
            "skill_vs_persistence": _as_float(entry.get("skill_vs_persistence")),
        }

    def persistence_vec_rmse(self) -> float | None:
        entry = self.metrics_models_map().get("persistence")
        if not entry:
            return None
        return _as_float(entry.get("vec_rmse"))

    # ==================================================================
    # 状态 / 预热
    # ==================================================================
    def artifact_status(self) -> dict:
        s = self.settings
        metrics_ok = (s.dashboard_dir / "metrics.json").exists() or (
            s.dashboard_dir / "metrics_tabular.json"
        ).exists() or (s.dashboard_dir / "metrics_simvp.json").exists()
        overview_ok = (s.dashboard_dir / "overview.json").exists() or (
            s.dashboard_dir / "timeseries.json"
        ).exists()
        return {
            "zarr": s.zarr_path.exists(),
            "models_ml": s.models_ml_dir.exists(),
            "simvp_ckpt": s.simvp_ckpt.exists(),
            "metrics": metrics_ok,
            "overview": overview_ok,
            "dashboard_dist": (s.dashboard_dist / "index.html").exists(),
        }

    def artifacts_ok(self) -> bool:
        status = self.artifact_status()
        return bool(status["zarr"] and status["metrics"] and status["overview"])

    def warmup(self) -> dict:
        """后台预热：数据读取 → artifacts → 两个引擎。任何异常都不外抛。"""
        t0 = time.perf_counter()
        report: dict[str, Any] = {}
        try:
            self.reader()
            report["data_time"] = bool(self.data_time())
        except Exception as exc:  # noqa: BLE001
            report["data_time"] = False
            logger.warning("warmup: zarr 数据读取失败：%s", exc)
        try:
            metrics, _ = self.metrics()
            report["metrics"] = metrics is not None
        except Exception:  # noqa: BLE001
            report["metrics"] = False
        try:
            overview, _ = self.overview()
            report["overview"] = overview is not None
        except Exception:  # noqa: BLE001
            report["overview"] = False
        try:
            engine = self.ensure_engine()
            report["engine_loaded"] = engine is not None
            if engine is not None and hasattr(engine, "warmup"):
                try:
                    engine.warmup()
                    report["engine_warmed"] = True
                except Exception as exc:  # noqa: BLE001
                    report["engine_warmed"] = False
                    logger.warning("warmup: engine.warmup() 失败：%s", exc)
                report["engine_status"] = self.engine_status()
        except Exception:  # noqa: BLE001
            report["engine_loaded"] = False
        try:
            report["simvp_loaded"] = self.ensure_simvp() is not None
        except Exception:  # noqa: BLE001
            report["simvp_loaded"] = False
        report["elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        self._warmup_done = True
        logger.info("warmup 完成：%s", report)
        return report

    def warmup_done(self) -> bool:
        return self._warmup_done


# ---------------------------------------------------------------------------
# NPZ 结构解析辅助
# ---------------------------------------------------------------------------
def _find_array(arrays: dict, *needles: str) -> np.ndarray | None:
    for name, arr in arrays.items():
        low = name.lower()
        if all(n in low for n in needles):
            return arr
    return None


def _lookup_cell_stats(arrays: dict, row: int, col: int) -> dict | None:
    """兼容多种 NPZ 命名/形状，返回单像元统计。"""
    h, w = GRID_ROWS, GRID_COLS

    def pick(*needles: str) -> float | None:
        arr = _find_array(arrays, *needles)
        if arr is None:
            return None
        a = np.asarray(arr)
        if a.shape == (h, w):
            return _as_float(a[row, col])
        if a.shape == (h * w,):
            return _as_float(a[row * w + col])
        return None

    speed_mean = pick("speed", "mean")
    if speed_mean is None:
        speed_mean = pick("mean")
    speed_std = pick("speed", "std")
    if speed_std is None:
        speed_std = pick("std")
    speed_p99 = pick("p99")
    valid_days = pick("valid", "days")
    if valid_days is None:
        valid_days = pick("count")
    lat = None
    lon = None
    try:
        lat_arr = arrays.get("latitude")
        lon_arr = arrays.get("longitude")
        reader_lat = _find_array(arrays, "lat")
        reader_lon = _find_array(arrays, "lon")
        if lat_arr is None and reader_lat is not None and reader_lat.shape == (h, w):
            lat = _as_float(reader_lat[row, col])
        if lon_arr is None and reader_lon is not None and reader_lon.shape == (h, w):
            lon = _as_float(reader_lon[row, col])
    except Exception:  # noqa: BLE001
        pass
    result = {
        "speed_mean": speed_mean,
        "speed_std": speed_std,
        "speed_p99": speed_p99,
        "valid_days": int(valid_days) if valid_days is not None else None,
        "lat": lat,
        "lon": lon,
    }
    if all(v is None for v in result.values()):
        return None
    return result


def build_store(settings: Settings | None = None) -> ArtifactStore:
    return ArtifactStore(settings)


__all__ = [
    "ArtifactStore",
    "ZarrFrameReader",
    "build_store",
    "MODEL_CATALOG",
    "model_catalog_map",
]
