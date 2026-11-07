"""Near-real-time forecast using the existing weights and a separate OSI SAF window."""
from __future__ import annotations

import logging
import math
import time
from datetime import UTC, datetime, timedelta
from functools import cached_property
from threading import Lock
from uuid import uuid4

import numpy as np

from src.serving.recent_observations import (
    PRODUCT, PRODUCT_URL, doy_phase, ease_latlon, ease_xy, load_snapshot, refresh_snapshot,
)

from ..config import CELL_KM, model_catalog_map
from ..schemas import APIError
from .field_service import (
    axes_row_lat_col_lon, direction_deg, grid_info, initial_bearing, sample_vectors,
    serialize_speed_grid, series_24h, stats_block, unavailable_error,
)

MODEL_IDS = ("simvp", "ridge", "random_forest")
MAX_DAYS = 3
logger = logging.getLogger(__name__)


def _path(fields, mask, row, col, latitude, longitude, end):
    """Advance at the daily modeled velocity sampled at each day's starting position."""
    x0, y0 = ease_xy(latitude[row, col], longitude[row, col])
    rows, cols, km, speed = [float(row)], [float(col)], [0.0], []
    for u, v in zip(*fields):
        r, c = round(rows[-1]), round(cols[-1])
        if not (0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and mask[r, c]):
            break
        uu, vv = float(u[r, c]), float(v[r, c])
        ss = math.hypot(uu, vv)
        speed.extend([ss] * 24)
        for _ in range(24):
            rows.append(rows[-1] + vv * 0.036 / CELL_KM)
            cols.append(cols[-1] + uu * 0.036 / CELL_KM)
            km.append(km[-1] + ss * 0.036)
    lat, lon = ease_latlon(x0 + (np.asarray(cols) - col) * CELL_KM * 1000,
                           y0 + (np.asarray(rows) - row) * CELL_KM * 1000)
    valid_at = end + timedelta(hours=len(speed))
    return {
        "hours": list(range(len(rows))),
        "cumulative_km": np.round(km, 3).tolist(),
        "speed_cm_s": np.round([speed[0], *speed], 3).tolist(),
        "row": np.round(rows, 5).tolist(), "col": np.round(cols, 5).tolist(),
        "lat": np.round(lat, 7).tolist(), "lon": np.round(lon, 7).tolist(),
        "uncertainty": None, "target_date": valid_at.date().isoformat(),
        "interpolation": "按各日预测矢量分段积分，次日起点按前一日轨迹更新。逐时位置为估计值。",
    }, valid_at


class RecentService:
    def __init__(self, store):
        self.store = store
        self.settings = store.settings
        self.path = self.settings.dashboard_dir / "recent_osisaf.npz"
        self._refresh_lock = Lock()
        self._forecast_lock = Lock()
        self._cached_forecasts = {}
        self._forecast_batch = None
        try:
            self._batch = load_snapshot(self.path) if self.path.is_file() else None
        except (ValueError, TypeError, KeyError, OSError):
            logger.warning("近期观测缓存读取失败", exc_info=True)
            self._batch = None

    @cached_property
    def grid(self):
        import zarr
        dataset = zarr.open_group(str(self.settings.zarr_path), mode="r")
        return dataset["latitude"][:], dataset["longitude"][:]

    def metadata(self):
        batch = self._batch
        if batch is None:
            return {"available": False, "source": PRODUCT, "source_url": PRODUCT_URL}
        mask = batch["mask"][-1]
        cells = np.argwhere(mask)
        preferred = (self.settings.default_row, self.settings.default_col)
        row, col = cells[np.square(cells - preferred).sum(axis=1).argmin()]
        meta = {key: value for key, value in batch["meta"].items() if key != "urls"}
        return {
            "available": True, **meta,
            "first_date": str(batch["dates"][0]), "latest_date": str(batch["dates"][-1]),
            "n_frames": len(batch["dates"]), "valid_cells": int(mask.sum()),
            "default_cell": {"row": int(row), "col": int(col)},
        }

    def refresh(self):
        with self._refresh_lock:
            try:
                self._batch = refresh_snapshot(self.settings.zarr_path, self.path, self._batch)
            except (ValueError, TypeError, RuntimeError, OSError, ImportError) as exc:
                logger.warning("OSI SAF 更新失败: %s", exc)
                raise APIError(
                    503, "recent_refresh_failed", "近期漂移观测更新失败",
                    "请检查网络与 OSI SAF 数据发布情况；已有批次可按原观测时间查看。",
                ) from exc
        return self.metadata()

    def _require_batch(self):
        if self._batch is None:
            raise APIError(503, "recent_unavailable", "暂无连续近期漂移观测", "请先点击「更新近期观测」。")
        return self._batch

    def _forecast(self, batch, model):
        if model not in MODEL_IDS:
            raise APIError(422, "recent_model_invalid", "近期预报支持 SimVP、Ridge 和 Random Forest")
        with self._forecast_lock:
            if self._forecast_batch is not batch:
                self._cached_forecasts.clear()
                self._forecast_batch = batch
            if model not in self._cached_forecasts:
                self._cached_forecasts[model] = self._run(batch, model)
            return self._cached_forecasts[model]

    def _run(self, batch, model):
        latitude, longitude = self.grid
        window = {name: batch[name] for name in ("u", "v", "mask", "doy")}
        if model == "simvp":
            predictor = self.store.ensure_simvp()
            if predictor is None:
                raise unavailable_error(self.store, model)
            return predictor.predict_window(window, MAX_DAYS)

        engine = self.store.ensure_engine()
        if engine is None or not self.store.model_available(model):
            raise unavailable_error(self.store, model)
        mask = window["mask"][-1]
        rows, cols = np.nonzero(mask)
        u_days = np.zeros((MAX_DAYS, *mask.shape), dtype=np.float32)
        v_days = np.zeros_like(u_days)
        window.update(latitude=latitude, longitude=longitude,
                      speed=np.hypot(window["u"], window["v"]))
        for lead in range(MAX_DAYS):
            pred = engine.predict_window(model, window, rows, cols)
            u, v = u_days[lead], v_days[lead]
            u[rows, cols], v[rows, cols] = pred.T
            if lead < MAX_DAYS - 1:
                date = batch["dates"][-1] + np.timedelta64(lead + 1, "D")
                updates = {"u": u, "v": v, "speed": np.hypot(u, v), "mask": mask,
                           "doy": doy_phase(date)}
                for key, next_frame in updates.items():
                    window[key] = np.concatenate((window[key][1:], np.expand_dims(next_frame, 0)))
        return u_days, v_days

    @staticmethod
    def _check_date(batch, date):
        latest = str(batch["dates"][-1])
        if date not in (None, "latest", latest):
            raise APIError(422, "recent_date_invalid", f"当前观测批次的起报日为 {latest}",
                           "用 date=latest 获取最新可用观测。")
        return latest

    def predict(self, request):
        started = time.perf_counter()
        batch = self._require_batch()
        latest = self._check_date(batch, request.date)
        row, col = request.cell.row, request.cell.col
        mask = batch["mask"][-1]
        if not mask[row, col]:
            raise APIError(422, "recent_cell_unobserved", "该像元缺少近期海冰漂移观测",
                           "请在近期观测图上选择有颜色的像元。")
        lat, lon = self.grid
        u_days, v_days = self._forecast(batch, request.model)
        days = request.lead_days
        end = datetime.fromisoformat(batch["meta"]["latest_at"])
        target = end + timedelta(days=days)
        target24 = (end + timedelta(days=1)).date().isoformat()
        path, path_end = _path((u_days[:days], v_days[:days]), mask, row, col, lat, lon, end)
        u, v = float(u_days[0, row, col]), float(v_days[0, row, col])
        ou, ov = float(batch["u"][-1, row, col]), float(batch["v"][-1, row, col])
        speed, obs_speed = math.hypot(u, v), math.hypot(ou, ov)
        series = series_24h(u, v, row, col, lat, lon, target24)
        bearing = initial_bearing(series["lat"][0], series["lon"][0],
                                  series["lat"][-1], series["lon"][-1])
        warnings = [{"code": "recent_product_shift", "level": "info",
                     "message": "OSI SAF 的 62.5 km、48h 平均漂移已转换到训练网格。历史 NSIDC 测试指标不适用于此数据源。"}]
        if days > 1:
            method = "SimVP 直接多时效推理" if request.model == "simvp" else "后续日使用上一日模型预测递推"
            warnings.append({"code": "multi_day_experimental", "level": "warning",
                             "message": f"{method}。多日预报为试验性结果，误差可能随时效增加。"})
        if path_end != target:
            warnings.insert(0, {"code": "path_outside_support", "level": "warning",
                                "message": "轨迹进入近期观测覆盖之外，已在有支持的最后一天结束。"})
        extends = path_end > datetime.now(UTC)
        if not extends:
            warnings.insert(0, {"code": "forecast_already_ended", "level": "warning",
                                "message": "此观测批次与时效的轨迹已经结束，请更新观测或选更长时效。"})
        meta = model_catalog_map()[request.model]
        daily = [
            {"lead_hours": (i + 1) * 24, "valid_at": (end + timedelta(days=i + 1)).isoformat(),
             "u": round(float(uu), 4), "v": round(float(vv), 4), "speed": round(math.hypot(uu, vv), 4)}
            for i, (uu, vv) in enumerate(zip(u_days[:days, row, col], v_days[:days, row, col]))
        ]
        return {
            "status": "ok", "status_text": "近期观测预报成功",
            "request_id": uuid4().hex[:8], "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "model": {key: meta[key] for key in ("id", "name", "category")},
            "data_time": {
                "last_obs_date": latest, "last_obs_index": len(batch["dates"]) - 1,
                "latest_date": latest, "next_frame_date": target24, "horizon_hours": days * 24,
                "n_frames": len(batch["dates"]), "target_observed": False, "source": PRODUCT,
                "observation_end_at": end.isoformat(), "forecast_valid_until": target.isoformat(),
                "extends_past_now": extends,
            },
            "input": {"row": row, "col": col, "lat": float(lat[row, col]), "lon": float(lon[row, col]),
                      "date": latest, "u": round(ou, 4), "v": round(ov, 4), "speed": round(obs_speed, 4),
                      "direction_deg": direction_deg(ou, ov), "valid": True, "coverage": float(mask.mean())},
            "prediction": {"u": round(u, 4), "v": round(v, 4), "speed": round(speed, 4),
                           "direction_deg": direction_deg(u, v), "bearing_deg": round(bearing, 2) % 360.0,
                           "displacement_km_24h": round(speed * .864, 3),
                           "delta_speed_vs_persistence": round(speed - obs_speed, 4)},
            "series_24h": series, "series_horizon": path, "daily_forecasts": daily,
            "metrics": None, "warnings": warnings,
        }

    def field(self, model, lead, stride, date):
        batch = self._require_batch()
        mask = batch["mask"][-1]
        lat, lon = self.grid
        date = self._check_date(batch, date)
        row_lat, col_lon = axes_row_lat_col_lon(lat, lon)
        if model is None:
            u, v = batch["u"][-1], batch["v"][-1]
        else:
            u_days, v_days = self._forecast(batch, model)
            u, v = u_days[lead - 1], v_days[lead - 1]
        speed = np.hypot(u, v)
        result = {
            "kind": "observed" if model is None else "predicted", "date": date,
            "grid": grid_info(self.settings), "row_axis": list(range(mask.shape[0])),
            "col_axis": list(range(mask.shape[1])), "row_lat": row_lat, "col_lon": col_lon,
            "speed": serialize_speed_grid(speed, mask), "vectors": sample_vectors(u, v, mask, stride),
            "stats": stats_block(mask, speed), "warnings": [], "source": PRODUCT,
        }
        if model is not None:
            target = datetime.fromisoformat(batch["meta"]["latest_at"]) + timedelta(days=lead)
            result.update(model={"id": model, "name": model_catalog_map()[model]["name"]},
                          target_date=target.date().isoformat(), valid_at=target.isoformat())
        return result
