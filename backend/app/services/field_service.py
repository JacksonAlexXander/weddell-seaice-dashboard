"""观测场 / 预测场服务（地图数据组装，字段严格按契约 /api/field*）。"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import numpy as np
from pydantic_core import PydanticCustomError

from ..config import (
    CELLS_PER_CM_S_DAY,
    GRID_COLS,
    GRID_ROWS,
    KM_PER_CM_S_HOUR,
    INTERPOLATION_NOTE,
    model_catalog_map,
)
from ..schemas import APIError, validate_date_string
from .artifact_store import ArtifactStore

logger = logging.getLogger("backend.app.field")


def _msg_text(value: object) -> str:
    try:
        return str(value)
    except Exception:  # noqa: BLE001
        return ""


# ---------------------------------------------------------------------------
# 几何 / 序列化工具
# ---------------------------------------------------------------------------
def direction_deg(u: float, v: float) -> float:
    """矢量方向（度，0=东，逆时针为正，EASE x/y 分量）。"""
    return float(np.mod(np.degrees(np.arctan2(v, u)), 360.0))


def bilinear(grid: np.ndarray, rows: np.ndarray | float, cols: np.ndarray | float) -> np.ndarray:
    """在 (H,W) 网格上对小数 (row, col) 双线性插值（越界裁剪到边缘）。"""
    h, w = grid.shape
    r = np.clip(np.asarray(rows, dtype=np.float64), 0, h - 1.000001)
    c = np.clip(np.asarray(cols, dtype=np.float64), 0, w - 1.000001)
    r0 = np.floor(r).astype(int)
    c0 = np.floor(c).astype(int)
    r1 = np.minimum(r0 + 1, h - 1)
    c1 = np.minimum(c0 + 1, w - 1)
    dr = r - r0
    dc = c - c0
    return (
        grid[r0, c0] * (1 - dr) * (1 - dc)
        + grid[r0, c1] * (1 - dr) * dc
        + grid[r1, c0] * dr * (1 - dc)
        + grid[r1, c1] * dr * dc
    )


def serialize_speed_grid(speed: np.ndarray, mask: np.ndarray) -> list[list[float | None]]:
    """(H,W) speed → 契约 speed 矩阵：无效像元 null，保留 2 位小数。"""
    out: list[list[float | None]] = []
    speed = np.asarray(speed, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    finite = np.isfinite(speed)
    for r in range(speed.shape[0]):
        row: list[float | None] = []
        for c in range(speed.shape[1]):
            if mask[r, c] and finite[r, c]:
                row.append(round(float(speed[r, c]), 2))
            else:
                row.append(None)
        out.append(row)
    return out


def sample_vectors(
    u: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray,
    stride: int,
) -> dict:
    """按 stride 抽样矢量（只保留有效像元）。"""
    h, w = u.shape
    rows = list(range(0, h, stride))
    cols = list(range(0, w, stride))
    rr, cc = np.meshgrid(np.asarray(rows), np.asarray(cols), indexing="ij")
    sub_u = np.asarray(u, dtype=np.float64)[np.ix_(rows, cols)]
    sub_v = np.asarray(v, dtype=np.float64)[np.ix_(rows, cols)]
    sub_mask = np.asarray(mask, dtype=bool)[np.ix_(rows, cols)]
    keep = sub_mask & np.isfinite(sub_u) & np.isfinite(sub_v)
    speed = np.hypot(sub_u, sub_v)
    out_rows = rr[keep]
    out_cols = cc[keep]
    out_u = sub_u[keep]
    out_v = sub_v[keep]
    out_speed = speed[keep]
    return {
        "stride": int(stride),
        "rows": [int(x) for x in out_rows],
        "cols": [int(x) for x in out_cols],
        "u": [round(float(x), 2) for x in out_u],
        "v": [round(float(x), 2) for x in out_v],
        "speed": [round(float(x), 2) for x in out_speed],
    }


def axes_row_lat_col_lon(lat: np.ndarray, lon: np.ndarray) -> tuple[list[float], list[float]]:
    """row_lat（按中列取纬度）、col_lon（按中行取经度）。"""
    h, w = lat.shape
    row_lat = [round(float(x), 5) for x in lat[:, w // 2]]
    col_lon = [round(float(x), 5) for x in lon[h // 2, :]]
    return row_lat, col_lon


def grid_info(settings) -> dict:
    return {"rows": settings.grid_rows, "cols": settings.grid_cols, "cell_km": settings.cell_km}


def stats_block(mask: np.ndarray, speed: np.ndarray) -> dict:
    """有效像元统计（valid_cells / coverage / mean_speed / max_speed）。"""
    mask = np.asarray(mask, dtype=bool)
    speed = np.asarray(speed, dtype=np.float64)
    valid = mask & np.isfinite(speed)
    values = speed[valid]
    total = int(mask.size) if mask.size else 1
    return {
        "valid_cells": int(valid.sum()),
        "coverage": round(float(valid.sum()) / total, 4),
        "mean_speed": round(float(values.mean()), 2) if values.size else None,
        "max_speed": round(float(values.max()), 2) if values.size else None,
    }


def series_24h(
    u: float,
    v: float,
    row: int,
    col: int,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    target_date: str,
    uncertainty: Any = None,
) -> dict:
    """构造 24h 逐时曲线：日尺度均值矢量线性积分（设计 §2）。"""
    hours = list(range(25))
    speed = math.hypot(u, v)
    cumulative = [round(speed * KM_PER_CM_S_HOUR * h, 3) for h in hours]
    speed_series = [round(speed, 3) for _ in hours]
    rows = [row + v * CELLS_PER_CM_S_DAY * (h / 24.0) for h in hours]
    cols = [col + u * CELLS_PER_CM_S_DAY * (h / 24.0) for h in hours]
    lats = [round(float(x), 5) for x in bilinear(lat_grid, np.asarray(rows), np.asarray(cols))]
    lons = [round(float(x), 5) for x in bilinear(lon_grid, np.asarray(rows), np.asarray(cols))]
    uncertainty_block = None
    if isinstance(uncertainty, dict):
        method = str(uncertainty.get("method") or "模型不确定度")
        lower_raw = uncertainty.get("lower_km")
        upper_raw = uncertainty.get("upper_km")
        if isinstance(lower_raw, (list, tuple)) and len(lower_raw) == 25:
            lower = [round(float(x), 3) for x in lower_raw]
        elif lower_raw is not None:
            lower = [round(max(0.0, float(lower_raw)) * h / 24.0, 3) for h in hours]
        else:
            lower = [0.0 for _ in hours]
        if isinstance(upper_raw, (list, tuple)) and len(upper_raw) == 25:
            upper = [round(float(x), 3) for x in upper_raw]
        elif upper_raw is not None:
            upper = [round(max(0.0, float(upper_raw)) * h / 24.0, 3) for h in hours]
        else:
            upper = [0.0 for _ in hours]
        # 保证 lower <= cumulative <= upper 的单调包络
        upper = [max(u_bound, c, l) for u_bound, c, l in zip(upper, cumulative, lower)]
        lower = [min(l_bound, c) for l_bound, c in zip(lower, cumulative)]
        uncertainty_block = {"method": method, "lower_km": lower, "upper_km": upper}
    return {
        "hours": hours,
        "cumulative_km": cumulative,
        "speed_cm_s": speed_series,
        "row": [round(x, 4) for x in rows],
        "col": [round(x, 4) for x in cols],
        "lat": lats,
        "lon": lons,
        "uncertainty": uncertainty_block,
        "interpolation": INTERPOLATION_NOTE,
        "target_date": target_date,
    }


# ---------------------------------------------------------------------------
# 引擎错误码 → APIError
# ---------------------------------------------------------------------------
_ENGINE_ERROR_MESSAGES: dict[str, tuple[int, str, str]] = {
    "date_not_in_dataset": (422, "date_not_in_dataset", "数据集中不存在该日期"),
    "date_invalid": (422, "date_invalid", "日期格式错误"),
    "no_future_frame": (422, "no_future_frame", "该日期之后没有可用帧（通常是数据集最后一天）"),
    "insufficient_history": (422, "insufficient_history", "该日期之前历史帧不足，无法构造模型输入"),
    "precomputed_unavailable": (
        409,
        "model_unavailable",
        "SimVP 历史回测需要预计算场 simvp_day1_fields.npz，目前尚未生成",
    ),
    "backtest_out_of_range": (422, "backtest_out_of_range", "回测日期超出该模型可用范围"),
    "inference_failed": (500, "inference_failed", "模型推理异常"),
    "model_unavailable": (409, "model_unavailable", "模型 artifact 缺失或加载失败"),
}


def map_engine_error(
    store: ArtifactStore,
    model_id: str,
    error_code: Any,
    message: Any = None,
    extra: dict | None = None,
) -> APIError:
    """把引擎/预测器返回的错误字典转成契约 APIError。"""
    code = str(error_code or "inference_failed")
    status, api_code, default_message = _ENGINE_ERROR_MESSAGES.get(
        code, (500, "inference_failed", "模型推理异常")
    )
    detail = f"（{message}）" if message else ""
    hint = None
    if api_code == "backtest_out_of_range" and isinstance(extra, dict):
        start = extra.get("available_start")
        end = extra.get("available_end")
        if start and end:
            hint = f"可用回测范围 {start} ~ {end}"
    elif api_code == "model_unavailable":
        hint = _unavailable_hint(store, model_id)
    if message:
        logger.info("模型 %s 返回错误 %s: %s", model_id, code, message)
    return APIError(status, api_code, f"{default_message}{detail}", hint)


def _unavailable_hint(store: ArtifactStore, model_id: str) -> str:
    if model_id == "simvp":
        err = store.simvp_error()
    else:
        err = store.engine_error()
    hint = "模型 artifact 缺失/加载失败（可能仍在训练或生成中，稍后重试）"
    if err:
        err = err.strip().replace("\n", " ")
        hint = f"{hint}：{err[:180]}"
    return hint


def unavailable_error(store: ArtifactStore, model_id: str) -> APIError:
    name = model_catalog_map().get(model_id, {}).get("name", model_id)
    return APIError(
        409,
        "model_unavailable",
        f"模型 {name}（{model_id}）当前不可用",
        _unavailable_hint(store, model_id),
    )

def exception_to_api_error(
    store: ArtifactStore, model_id: str, exc: BaseException
) -> APIError | None:
    """把引擎/预测器抛出的异常（PredictError 体系）映射成契约 APIError。

    ``src.predict`` 的异常带 ``.code`` / ``.http_status`` / ``.message`` / ``.hint``；
    未识别异常返回 None，由调用方按 500 处理。
    """
    raw_code = getattr(exc, "code", None)
    http_status = getattr(exc, "http_status", None)
    message = getattr(exc, "message", None) or _msg_text(exc)
    hint = getattr(exc, "hint", None)
    if raw_code:
        code = str(raw_code)
        if code == "date_invalid":
            # 引擎把「日期不存在」与「没有下一帧」都归为 DateError，这里按文案细分
            if "下一帧" in message:
                return APIError(422, "date_invalid", message, hint or "请选择更早的起报日期")
            if "无该日期" in message or "不存在" in message or "没有该日期" in message:
                return APIError(422, "date_not_in_dataset", message, hint)
            return APIError(422, "date_invalid", message, hint)
        if code in ("unknown_model", "model_unavailable", "invalid_cell",
                    "insufficient_history", "backtest_out_of_range"):
            status = int(http_status or 422)
            if code == "model_unavailable":
                hint = hint or _unavailable_hint(store, model_id)
            return APIError(status, code, message, hint)
        if code == "date_not_in_dataset":
            return APIError(422, "date_not_in_dataset", message, hint)
        if code == "no_future_frame":
            return APIError(422, "no_future_frame", message, hint)
        status = int(http_status or 500)
        return APIError(status, "inference_failed" if status >= 500 else code, message, hint)
    text = _msg_text(exc)
    if "insufficient_history" in text:
        return APIError(422, "insufficient_history", "该日期之前历史帧不足，无法构造模型输入")
    return None


# ---------------------------------------------------------------------------
# FieldService
# ---------------------------------------------------------------------------
class FieldService:
    """观测场 / 预测场组装（地图 UI 数据）。"""

    def __init__(self, store: ArtifactStore):
        self.store = store
        self.settings = store.settings

    # -- 公共辅助 ----------------------------------------------------------
    def _adapter(self):
        try:
            return self.store.data_adapter()
        except Exception as exc:  # noqa: BLE001
            logger.warning("数据访问失败：%s", exc)
            raise APIError(
                503,
                "artifact_unavailable",
                "海冰数据集 zarr 不可用",
                f"缺少或无法读取 {self.settings.zarr_path}",
            ) from exc

    def _data_time(self) -> dict:
        dt = self.store.data_time()
        if not dt:
            raise APIError(
                503,
                "artifact_unavailable",
                "数据时间信息不可用",
                f"无法读取 {self.settings.zarr_path}",
            )
        return dt

    def _grids(self, adapter) -> tuple[np.ndarray, np.ndarray]:
        try:
            lat = np.asarray(adapter.latitude, dtype=np.float32)
            lon = np.asarray(adapter.longitude, dtype=np.float32)
            if lat.shape == (GRID_ROWS, GRID_COLS) and lon.shape == (GRID_ROWS, GRID_COLS):
                return lat, lon
        except Exception:  # noqa: BLE001
            pass
        reader = self.store.reader()
        return reader.latitude, reader.longitude

    def _date_arg(self, date: str) -> str:
        try:
            return validate_date_string(date, "date")
        except PydanticCustomError:
            raise APIError(
                422, "date_invalid", f"日期格式错误：{date}，须为 YYYY-MM-DD"
            ) from None

    def resolve_date(self, adapter, date: str | None) -> tuple[str, int]:
        """解析日期 → (date_str, idx)；None 表示最新有效日。"""
        if date is None:
            dates = adapter.dates
            if len(dates) == 0:
                raise APIError(503, "artifact_unavailable", "数据集为空")
            idx = int(len(dates) - 1)
            return str(dates[idx]), idx
        date = self._date_arg(date)
        idx = adapter.idx_of_date(date)
        if idx is None:
            dt = self._data_time()
            raise APIError(
                422,
                "date_not_in_dataset",
                f"数据集中不存在日期 {date}",
                f"可选日期 {dt.get('first_date')} ~ {dt.get('latest_date')}",
            )
        return date, idx

    def _frame(self, adapter, idx: int) -> dict:
        try:
            return adapter.frame(idx)
        except ValueError as exc:
            raise APIError(422, "date_out_of_range", f"帧索引越界：{exc}") from exc
        except Exception as exc:  # noqa: BLE001
            logger.warning("帧读取失败 idx=%s: %s", idx, exc)
            raise APIError(503, "artifact_unavailable", "数据帧读取失败") from exc

    def _target_date(self, adapter, idx: int) -> str | None:
        try:
            return adapter.date_str(adapter.next_idx(idx))
        except ValueError:
            return None
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _warnings(raw: Any) -> list[dict]:
        out: list[dict] = []
        if not isinstance(raw, list):
            return out
        for item in raw:
            if isinstance(item, dict) and item.get("code"):
                out.append(
                    {
                        "code": str(item.get("code")),
                        "level": str(item.get("level") or "info"),
                        "message": str(item.get("message") or ""),
                    }
                )
        return out

    # -- 观测场 ------------------------------------------------------------
    def observed_field(self, date: str | None, stride: int = 5) -> dict:
        adapter = self._adapter()
        date_str, idx = self.resolve_date(adapter, date)
        frame = self._frame(adapter, idx)
        lat, lon = self._grids(adapter)
        u = np.asarray(frame["u"], dtype=np.float32)
        v = np.asarray(frame["v"], dtype=np.float32)
        mask = np.asarray(frame["mask"], dtype=bool) & np.isfinite(u) & np.isfinite(v)
        speed = np.hypot(u, v).astype(np.float32)
        row_lat, col_lon = axes_row_lat_col_lon(lat, lon)
        return {
            "kind": "observed",
            "date": date_str,
            "grid": grid_info(self.settings),
            "row_axis": list(range(GRID_ROWS)),
            "col_axis": list(range(GRID_COLS)),
            "row_lat": row_lat,
            "col_lon": col_lon,
            "speed": serialize_speed_grid(speed, mask),
            "vectors": sample_vectors(u, v, mask, stride),
            "stats": stats_block(mask, speed),
            "data_time": self._data_time(),
        }

    # -- 预测场 ------------------------------------------------------------
    def predicted_field(self, model_id: str, date: str, stride: int = 5) -> dict:
        t0 = time.perf_counter()
        catalog = model_catalog_map()
        meta = catalog.get(model_id)
        if meta is None:
            raise APIError(
                404,
                "unknown_model",
                f"未知模型 id：{model_id}",
                f"可选模型：{', '.join(catalog)}",
            )
        if not self.store.model_available(model_id):
            raise unavailable_error(self.store, model_id)
        adapter = self._adapter()
        date_str, idx = self.resolve_date(adapter, date)
        target_date = self._target_date(adapter, idx)
        if target_date is None:
            raise APIError(
                422,
                "date_invalid",
                "该日期之后没有可用帧，无法生成预测场（通常是数据集最后一天）",
                "请选择更早的日期",
            )

        try:
            if model_id == "simvp":
                predictor = self.store.ensure_simvp()
                if predictor is None:
                    raise unavailable_error(self.store, model_id)
                raw = predictor.predict_field(date_str)
            else:
                engine = self.store.ensure_engine()
                if engine is None:
                    raise unavailable_error(self.store, model_id)
                raw = engine.predict_field(model_id, date_str)
        except APIError:
            raise
        except Exception as exc:  # noqa: BLE001
            mapped = exception_to_api_error(self.store, model_id, exc)
            if mapped is not None:
                raise mapped from exc
            logger.exception("预测场推理失败 model=%s date=%s", model_id, date_str)
            raise APIError(500, "inference_failed", "模型推理异常") from exc

        if isinstance(raw, dict) and raw.get("ok") is False:
            raise map_engine_error(self.store, model_id, raw.get("error"), raw.get("message"), raw)
        if not isinstance(raw, dict):
            raise APIError(500, "inference_failed", "模型返回格式异常")

        u = np.asarray(raw.get("u"), dtype=np.float32)
        v = np.asarray(raw.get("v"), dtype=np.float32)
        if u.shape != (GRID_ROWS, GRID_COLS) or v.shape != (GRID_ROWS, GRID_COLS):
            raise APIError(500, "inference_failed", "模型返回的预测场形状异常")
        mask = np.isfinite(u) & np.isfinite(v)
        speed = np.hypot(u, v).astype(np.float32)
        target_date = str(raw.get("target_date") or target_date)

        comparison = None
        try:
            trap = self._frame(adapter, adapter.next_idx(idx))
            truth_mask = np.asarray(trap["mask"], dtype=bool)
            valid = mask & truth_mask & np.isfinite(trap["u"]) & np.isfinite(trap["v"])
            if bool(valid.any()):
                du = u[valid].astype(np.float64) - np.asarray(trap["u"], dtype=np.float32)[valid]
                dv = v[valid].astype(np.float64) - np.asarray(trap["v"], dtype=np.float32)[valid]
                vec_rmse = float(np.sqrt(np.mean(du * du + dv * dv)))
                comparison = {"n": int(valid.sum()), "vec_rmse": round(vec_rmse, 2)}
            else:
                comparison = {"n": 0, "vec_rmse": None}
        except APIError:
            comparison = None

        lat, lon = self._grids(adapter)
        row_lat, col_lon = axes_row_lat_col_lon(lat, lon)
        warnings = self._warnings(raw.get("warnings"))
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        return {
            "kind": "predicted",
            "model": {"id": model_id, "name": meta["name"]},
            "date": date_str,
            "target_date": target_date,
            "grid": grid_info(self.settings),
            "row_axis": list(range(GRID_ROWS)),
            "col_axis": list(range(GRID_COLS)),
            "row_lat": row_lat,
            "col_lon": col_lon,
            "speed": serialize_speed_grid(speed, mask),
            "vectors": sample_vectors(u, v, mask, stride),
            "stats": {**stats_block(mask, speed), "comparison": comparison},
            "elapsed_ms": elapsed_ms,
            "warnings": warnings,
        }


__all__ = [
    "FieldService",
    "exception_to_api_error",
    "bilinear",
    "direction_deg",
    "serialize_speed_grid",
    "sample_vectors",
    "axes_row_lat_col_lon",
    "grid_info",
    "stats_block",
    "series_24h",
    "map_engine_error",
    "unavailable_error",
]
