"""预测 / 回测服务：真实模型推理 + 契约响应组装（绝不训练、绝不写模型文件）。

- 表格模型（ridge / random_forest / persistence / climatology）→ ``PredictionEngine``
- SimVP → ``SimVPPredictor``
- 引擎/预测器返回的错误字典或异常统一映射为契约 ``APIError``
- 确定性推理结果做小内存 LRU 缓存（key: model + date + cell），命中时返回相同字段
"""

from __future__ import annotations

import copy
import logging
import math
import threading
import time
import uuid
from collections import OrderedDict
from typing import Any

import numpy as np

from ..config import (
    HORIZON_HOURS,
    KM_PER_CM_S_HOUR,
    model_catalog_map,
)
from ..schemas import APIError, BacktestRequest, PredictRequest
from .artifact_store import ArtifactStore
from .field_service import (
    FieldService,
    direction_deg,
    exception_to_api_error,
    map_engine_error,
    series_24h,
    unavailable_error,
)

logger = logging.getLogger("backend.app.predict")

_DEFAULT_INTERPOLATION = "日尺度产品：逐时位置/位移由 24h 平均漂移矢量线性积分"


# ---------------------------------------------------------------------------
# 小内存 LRU（线程安全，deepcopy 保护缓存对象不被下游修改）
# ---------------------------------------------------------------------------
class LRUCache:
    def __init__(self, maxsize: int = 128):
        self.maxsize = max(1, int(maxsize))
        self._data: "OrderedDict[Any, Any]" = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key):
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return copy.deepcopy(self._data[key])

    def put(self, key, value) -> None:
        with self._lock:
            self._data[key] = copy.deepcopy(value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


def _num(value: Any) -> float | None:
    """安全转 float（None / nan → None）。"""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _round(value: Any, digits: int = 3) -> float | None:
    out = _num(value)
    return None if out is None else round(out, digits)


def _msg_text(value: Any) -> str:
    try:
        return str(value)
    except Exception:  # noqa: BLE001
        return ""


class PredictService:
    """预测 / 回测业务逻辑（单例挂在 ``app.state.predict_service``）。"""

    def __init__(self, store: ArtifactStore):
        self.store = store
        self.settings = store.settings
        self.fields = FieldService(store)
        self._predict_cache = LRUCache(self.settings.predict_cache_size)
        self._backtest_cache = LRUCache(self.settings.backtest_cache_size)

    def clear_cache(self) -> None:
        self._predict_cache.clear()
        self._backtest_cache.clear()

    # ==================================================================
    # 公共校验
    # ==================================================================
    @staticmethod
    def _catalog_model(model_id: str) -> dict:
        catalog = model_catalog_map()
        meta = catalog.get(str(model_id))
        if meta is None:
            raise APIError(
                404,
                "unknown_model",
                f"未知模型 id：{model_id}",
                f"可选模型：{', '.join(catalog)}",
            )
        return meta

    def _require_available(self, model_id: str) -> None:
        if not self.store.model_available(model_id):
            raise unavailable_error(self.store, model_id)

    def _data(self):
        return self.fields._adapter()

    def _data_time_block(self, adapter, idx: int, date_str: str, next_frame_date: str) -> dict:
        dt = self.store.data_time() or {}
        latest = dt.get("latest_date") or adapter.date_str(adapter.n_frames - 1)
        n_frames = dt.get("n_frames") or int(adapter.n_frames)
        return {
            "last_obs_date": date_str,
            "last_obs_index": int(idx),
            "next_frame_date": next_frame_date,
            "horizon_hours": HORIZON_HOURS,
            "latest_date": str(latest),
            "n_frames": int(n_frames),
        }

    # ------------------------------------------------------------------
    # 引擎调用 & 错误归一
    # ------------------------------------------------------------------
    def _engine_predict_point(self, model_id: str, date_str: str, row: int, col: int) -> dict:
        try:
            if model_id == "simvp":
                predictor = self.store.ensure_simvp()
                if predictor is None:
                    raise unavailable_error(self.store, model_id)
                raw = predictor.predict_point(date_str, row, col)
            else:
                engine = self.store.ensure_engine()
                if engine is None:
                    raise unavailable_error(self.store, model_id)
                raw = engine.predict_point(model_id, date_str, row, col)
        except APIError:
            raise
        except Exception as exc:  # noqa: BLE001
            mapped = exception_to_api_error(self.store, model_id, exc)
            if mapped is not None:
                raise mapped from exc
            logger.exception("单点推理异常 model=%s date=%s cell=(%s,%s)", model_id, date_str, row, col)
            raise APIError(500, "inference_failed", "模型推理异常") from exc

        if isinstance(raw, dict) and raw.get("ok") is False:
            raise map_engine_error(self.store, model_id, raw.get("error"), raw.get("message"), raw)
        if not isinstance(raw, dict):
            raise APIError(500, "inference_failed", "模型返回格式异常")
        return raw

    def _engine_backtest(
        self, model_id: str, row: int, col: int, start: str, end: str, max_points: int
    ) -> dict:
        try:
            if model_id == "simvp":
                predictor = self.store.ensure_simvp()
                if predictor is None:
                    raise unavailable_error(self.store, model_id)
                raw = predictor.backtest(row, col, start, end, max_points)
            else:
                engine = self.store.ensure_engine()
                if engine is None:
                    raise unavailable_error(self.store, model_id)
                raw = engine.backtest(model_id, row, col, start, end, max_points)
        except APIError:
            raise
        except Exception as exc:  # noqa: BLE001
            mapped = exception_to_api_error(self.store, model_id, exc)
            if mapped is not None:
                raise mapped from exc
            logger.exception("回测推理异常 model=%s cell=(%s,%s) %s~%s", model_id, row, col, start, end)
            raise APIError(500, "inference_failed", "历史回测推理异常") from exc

        if isinstance(raw, dict) and raw.get("ok") is False:
            raise map_engine_error(self.store, model_id, raw.get("error"), raw.get("message"), raw)
        if not isinstance(raw, dict):
            raise APIError(500, "inference_failed", "模型返回格式异常")
        return raw

    # ==================================================================
    # POST /api/predict
    # ==================================================================
    def predict(self, req: PredictRequest) -> dict:
        t0 = time.perf_counter()
        request_id = uuid.uuid4().hex[:8]
        model_id = str(req.model)
        meta = self._catalog_model(model_id)
        self._require_available(model_id)

        adapter = self._data()
        date_str, idx = self.fields.resolve_date(adapter, req.date)
        try:
            next_idx = adapter.next_idx(idx)
        except ValueError:
            raise APIError(
                422,
                "date_invalid",
                "该日期之后没有可用帧，无法做 24h 预测（通常是数据集最后一天）",
                "请选择更早的起报日期（默认推荐 2025-06-26）",
            ) from None
        next_frame_date = adapter.date_str(next_idx)
        row, col = int(req.cell.row), int(req.cell.col)

        # SimVP 需要 10 帧历史输入（含当日）
        if model_id == "simvp" and idx < 9:
            raise APIError(
                422,
                "insufficient_history",
                "SimVP 需要至少 10 帧历史观测（含起报日），该日期之前历史帧不足",
                f"起报日 index={idx}，需 ≥9",
            )

        cache_key = (self.store.engine_generation, model_id, date_str, row, col)
        cached = self._predict_cache.get(cache_key)
        if cached is not None:
            cached["request_id"] = request_id
            cached["elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
            return cached

        raw = self._engine_predict_point(model_id, date_str, row, col)
        payload = self._build_predict_response(
            meta=meta,
            raw=raw,
            adapter=adapter,
            date_str=date_str,
            idx=idx,
            next_idx=next_idx,
            next_frame_date=next_frame_date,
            row=row,
            col=col,
            request_id=request_id,
            elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 1),
        )
        self._predict_cache.put(cache_key, payload)
        return payload

    def _build_predict_response(
        self,
        *,
        meta: dict,
        raw: dict,
        adapter,
        date_str: str,
        idx: int,
        next_idx: int,
        next_frame_date: str,
        row: int,
        col: int,
        request_id: str,
        elapsed_ms: float,
    ) -> dict:
        model_id = str(meta["id"])
        # --- 观测帧（输入上下文） ---
        try:
            frame = adapter.frame(idx)
        except Exception:  # noqa: BLE001
            frame = {"u": np.zeros((1, 1)), "v": np.zeros((1, 1)), "mask": np.zeros((1, 1), bool),
                     "coverage": 0.0}
        raw_input = raw.get("input") if isinstance(raw.get("input"), dict) else {}
        lat_grid, lon_grid = self.fields._grids(adapter)
        lat = _round(raw_input.get("lat"), 4)
        lon = _round(raw_input.get("lon"), 4)
        if lat is None:
            lat = round(float(lat_grid[row, col]), 4)
        if lon is None:
            lon = round(float(lon_grid[row, col]), 4)

        mask = frame.get("mask")
        valid = bool(mask[row, col]) if hasattr(mask, "shape") and mask.shape[0] > row else bool(
            raw_input.get("valid", True)
        )
        obs_u = _num(raw_input.get("u"))
        obs_v = _num(raw_input.get("v"))
        if obs_u is None or obs_v is None:
            try:
                obs_u = float(frame["u"][row, col])
                obs_v = float(frame["v"][row, col])
            except Exception:  # noqa: BLE001
                obs_u, obs_v = 0.0, 0.0
        input_speed = _num(raw_input.get("speed"))
        if input_speed is None:
            input_speed = math.hypot(obs_u, obs_v)
        input_direction = _num(raw_input.get("direction_deg"))
        if input_direction is None:
            input_direction = direction_deg(obs_u, obs_v)
        coverage = _num(raw_input.get("coverage"))
        if coverage is None:
            coverage = _num(frame.get("coverage"))
        if coverage is None:
            coverage = 0.0

        # --- 预测矢量 ---
        pred_u = _num(raw.get("u"))
        pred_v = _num(raw.get("v"))
        if pred_u is None or pred_v is None:
            raise APIError(500, "inference_failed", "模型返回的预测矢量缺失")
        pred_speed = _num(raw.get("speed"))
        if pred_speed is None:
            pred_speed = math.hypot(pred_u, pred_v)
        pred_direction = _num(raw.get("direction_deg"))
        if pred_direction is None:
            pred_direction = direction_deg(pred_u, pred_v)
        displacement_km = _round(raw.get("displacement_km_24h"), 3)
        if displacement_km is None:
            displacement_km = round(pred_speed * KM_PER_CM_S_HOUR * HORIZON_HOURS, 3)
        delta_speed = _round(raw.get("delta_speed_vs_persistence"), 3)
        if delta_speed is None:
            delta_speed = round(pred_speed - input_speed, 3) if valid else None

        # --- 24h 曲线：优先引擎的逐时序列（若有），否则线性积分 ---
        uncertainty = raw.get("uncertainty")
        raw_series = raw.get("series_24h") if isinstance(raw.get("series_24h"), dict) else None
        if uncertainty is None and raw_series is not None:
            uncertainty = raw_series.get("uncertainty")
        built_series = series_24h(
            pred_u, pred_v, row, col, lat_grid, lon_grid, next_frame_date, uncertainty
        )
        # 若引擎已经返回完整 25 点序列，保留其 target_date / interpolation 文案口径
        if raw_series is not None:
            if not raw_series.get("interpolation"):
                built_series["interpolation"] = _DEFAULT_INTERPOLATION
            if not built_series["uncertainty"] and isinstance(raw_series.get("uncertainty"), dict):
                built_series["uncertainty"] = raw_series["uncertainty"]

        # --- warnings：引擎 warnings + 高速度 / 无观测 / 时间缺口 ---
        warnings = self._normalize_warnings(raw.get("warnings"))
        have = {w["code"] for w in warnings}
        if not valid and not ({"no_observation", "input_invalid"} & have):
            warnings.insert(
                0,
                {
                    "code": "no_observation",
                    "level": "info",
                    "message": "起报日该像元无有效观测（u/v=0），预测主要依赖邻域/季节特征",
                },
            )
        try:
            gap = int((adapter.dates[next_idx] - adapter.dates[idx]).astype(int))
        except Exception:  # noqa: BLE001
            gap = 1
        if gap > 1 and not ({"frame_gap", "temporal_gap"} & have):
            warnings.append(
                {
                    "code": "frame_gap",
                    "level": "info",
                    "message": f"下一有效帧距起报日 {gap} 天（数据缺失日），24h 曲线为近似",
                }
            )
        if "high_speed" not in have:
            stats = self.store.cell_stats(row, col)
            if stats:
                p99 = _num(stats.get("speed_p99"))
                if p99 is not None and p99 > 0 and pred_speed > p99:
                    warnings.append(
                        {
                            "code": "high_speed",
                            "level": "warning",
                            "message": (
                                f"预测漂移速度 {pred_speed:.1f} cm/s 高于该像元历史 99% 分位"
                                f"（{p99:.1f} cm/s），可能为极端事件"
                            ),
                        }
                    )

        data_time_block = self._data_time_block(adapter, idx, date_str, next_frame_date)
        return {
            "status": "ok",
            "status_text": "预测成功",
            "request_id": request_id,
            "elapsed_ms": elapsed_ms,
            "model": {"id": model_id, "name": meta["name"], "category": meta["category"]},
            "data_time": data_time_block,
            "input": {
                "row": row,
                "col": col,
                "lat": lat,
                "lon": lon,
                "date": date_str,
                "u": round(obs_u, 3),
                "v": round(obs_v, 3),
                "speed": round(input_speed, 3),
                "direction_deg": round(input_direction, 2),
                "valid": valid,
                "coverage": round(coverage, 4),
            },
            "prediction": {
                "u": round(pred_u, 3),
                "v": round(pred_v, 3),
                "speed": round(pred_speed, 3),
                "direction_deg": round(pred_direction, 2),
                "unit": "cm/s",
                "displacement_km_24h": displacement_km,
                "bearing_deg": round(pred_direction, 2),
                "delta_speed_vs_persistence": delta_speed,
            },
            "series_24h": built_series,
            "metrics": self.store.model_metrics(model_id),
            "warnings": warnings,
        }

    @staticmethod
    def _normalize_warnings(raw: Any) -> list[dict]:
        out: list[dict] = []
        if not isinstance(raw, list):
            return out
        for item in raw:
            if not isinstance(item, dict) or not item.get("code"):
                continue
            level = str(item.get("level") or "info").lower()
            if level not in ("info", "warning", "danger"):
                level = "info"
            out.append(
                {
                    "code": str(item["code"]),
                    "level": level,
                    "message": str(item.get("message") or ""),
                }
            )
        return out

    # ==================================================================
    # POST /api/backtest
    # ==================================================================
    def backtest(self, req: BacktestRequest) -> dict:
        model_id = str(req.model)
        meta = self._catalog_model(model_id)
        self._require_available(model_id)
        adapter = self._data()
        row, col = int(req.cell.row), int(req.cell.col)
        dt = self.store.data_time() or {}
        latest = dt.get("latest_date") or adapter.date_str(adapter.n_frames - 1)
        first = dt.get("first_date") or adapter.date_str(0)

        end = req.end or str(latest)
        if req.start:
            start = req.start
        else:
            end_np = np.datetime64(end, "D")
            start = str(end_np - np.timedelta64(180, "D"))
        # 格式化校验（默认值也校验）
        try:
            start = self.fields._date_arg(start)
            end = self.fields._date_arg(end)
        except APIError:
            raise
        if start > end:
            raise APIError(
                422,
                "invalid_range",
                f"start（{start}）不能晚于 end（{end}）",
                "请检查回测日期范围",
            )
        max_points = int(req.max_points or 180)

        # SimVP 回测仅覆盖 test 期（2022–2025），契约要求请求 2022 之前日期返回 422 + 可用范围
        if model_id == "simvp":
            avail_start, avail_end = self.store.simvp_backtest_range() or ("2022-01-01", str(latest))
            if start < "2022-01-01" or end < avail_start:
                raise APIError(
                    422,
                    "backtest_out_of_range",
                    f"SimVP 回测仅覆盖 test 期（2022–2025），请求范围 {start} ~ {end} 超出可用范围",
                    f"可用回测范围 {avail_start} ~ {avail_end}",
                )

        cache_key = (self.store.engine_generation, model_id, row, col, start, end, max_points)
        cached = self._backtest_cache.get(cache_key)
        if cached is not None:
            return cached

        raw = self._engine_backtest(model_id, row, col, start, end, max_points)
        payload = self._build_backtest_response(
            meta=meta,
            raw=raw,
            adapter=adapter,
            start=start,
            end=end,
            row=row,
            col=col,
            first=first,
            latest=latest,
        )
        self._backtest_cache.put(cache_key, payload)
        return payload

    def _build_backtest_response(
        self,
        *,
        meta: dict,
        raw: dict,
        adapter,
        start: str,
        end: str,
        row: int,
        col: int,
        first: str,
        latest: str,
    ) -> dict:
        model_id = str(meta["id"])
        lat_grid, lon_grid = self.fields._grids(adapter)
        series_raw = raw.get("series") if isinstance(raw.get("series"), list) else []
        series: list[dict] = []
        for item in series_raw:
            if not isinstance(item, dict):
                continue
            obs_u = _round(item.get("obs_u"))
            obs_v = _round(item.get("obs_v"))
            pred_u = _round(item.get("pred_u"))
            pred_v = _round(item.get("pred_v"))
            obs_speed = _round(item.get("obs_speed"))
            if obs_speed is None and obs_u is not None and obs_v is not None:
                obs_speed = round(math.hypot(obs_u, obs_v), 3)
            pred_speed = _round(item.get("pred_speed"))
            if pred_speed is None and pred_u is not None and pred_v is not None:
                pred_speed = round(math.hypot(pred_u, pred_v), 3)
            err_vec = _round(item.get("err_vec"))
            if err_vec is None and None not in (obs_u, obs_v, pred_u, pred_v):
                err_vec = round(math.hypot(pred_u - obs_u, pred_v - obs_v), 3)
            series.append(
                {
                    "date": str(item.get("date") or ""),
                    "obs_u": obs_u,
                    "obs_v": obs_v,
                    "obs_speed": obs_speed,
                    "pred_u": pred_u,
                    "pred_v": pred_v,
                    "pred_speed": pred_speed,
                    "err_vec": err_vec,
                }
            )

        summary_raw = raw.get("summary") if isinstance(raw.get("summary"), dict) else {}
        summary = {
            "n": int(summary_raw.get("n") or len(series)),
            "rmse_u": _round(summary_raw.get("rmse_u"), 4),
            "rmse_v": _round(summary_raw.get("rmse_v"), 4),
            "vec_rmse": _round(summary_raw.get("vec_rmse"), 4),
            "mae_vec": _round(summary_raw.get("mae_vec"), 4),
            "r2_u": _round(summary_raw.get("r2_u"), 4),
            "r2_v": _round(summary_raw.get("r2_v"), 4),
            "skill_vs_persistence": _round(summary_raw.get("skill_vs_persistence"), 4),
        }
        # skill 兜底：1 - 本模型 vec_rmse / 持续性 vec_rmse（统一评估口径）
        if summary["skill_vs_persistence"] is None and summary["vec_rmse"] is not None:
            pers = self.store.persistence_vec_rmse()
            if pers and pers > 0:
                summary["skill_vs_persistence"] = round(1.0 - summary["vec_rmse"] / pers, 4)

        range_raw = raw.get("range") if isinstance(raw.get("range"), dict) else {}
        requested_start = str(range_raw.get("requested_start") or start)
        requested_end = str(range_raw.get("requested_end") or end)
        available_start = str(range_raw.get("available_start") or raw.get("available_start") or first)
        available_end = str(range_raw.get("available_end") or raw.get("available_end") or latest)
        n_points = int(range_raw.get("n_points") or len(series))
        n_days_total = range_raw.get("n_days_total")
        if n_days_total is None:
            n_days_total = raw.get("n_days_total")
        if n_days_total is None:
            try:
                dates = adapter.dates
                d0 = np.datetime64(start, "D")
                d1 = np.datetime64(end, "D")
                n_days_total = int(np.count_nonzero((dates >= d0) & (dates <= d1)))
            except Exception:  # noqa: BLE001
                n_days_total = len(series)

        return {
            "status": "ok",
            "model": {"id": model_id, "name": meta["name"]},
            "cell": {
                "row": row,
                "col": col,
                "lat": round(float(lat_grid[row, col]), 4),
                "lon": round(float(lon_grid[row, col]), 4),
            },
            "range": {
                "requested_start": requested_start,
                "requested_end": requested_end,
                "available_start": available_start,
                "available_end": available_end,
                "n_points": n_points,
                "n_days_total": int(n_days_total),
            },
            "series": series,
            "summary": summary,
            "warnings": self._normalize_warnings(raw.get("warnings")),
        }


__all__ = ["PredictService", "LRUCache"]
