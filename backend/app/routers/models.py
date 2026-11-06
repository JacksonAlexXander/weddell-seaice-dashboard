"""GET /api/models —— 模型列表 + 数据时间 + 默认预测参数 + 展示组合。"""

from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, Depends

from ..config import MODEL_CATALOG, MODEL_IDS
from ..deps import get_store
from ..schemas import APIError, Cell, Combo, DataTime, DefaultParams, ModelInfo, ModelMetricsBrief, ModelsResponse
from ..services.artifact_store import ArtifactStore

logger = logging.getLogger("backend.app.routes.models")

router = APIRouter(tags=["models"])


def _num(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _resolve_default_date(store: ArtifactStore, data_time: dict) -> str:
    """默认日期：优先设计推荐日（2025-06-26），否则倒数第二个有效帧（保证有下一帧可预测）。"""
    preferred = store.settings.preferred_default_date
    try:
        adapter = store.data_adapter()
        idx = adapter.idx_of_date(preferred)
        if idx is not None and idx < adapter.n_frames - 1:
            return preferred
        if adapter.n_frames >= 2:
            return adapter.date_str(adapter.n_frames - 2)
        return adapter.date_str(0)
    except Exception:  # noqa: BLE001
        logger.debug("默认日期解析失败，回退 latest_date", exc_info=True)
    return str(data_time.get("latest_date") or preferred)


def build_models_payload(store: ArtifactStore) -> ModelsResponse:
    data_time_raw = store.data_time()
    if not data_time_raw:
        raise APIError(
            503,
            "artifact_unavailable",
            "数据时间信息不可用",
            f"无法读取 {store.settings.zarr_path}",
        )
    data_time = DataTime(**data_time_raw)
    availability = store.availability_map()
    metrics_map = store.metrics_models_map()
    meta_map = store.model_meta()
    # 引擎已加载时优先使用其 model_info（含 model_card 解析出的训练信息/指标）
    engine = store.engine_or_none()
    if engine is not None:
        for model_id in MODEL_IDS:
            try:
                info = engine.model_info(model_id)
            except Exception:  # noqa: BLE001
                continue
            merged = dict(meta_map.get(model_id) or {})
            if info.get("trained_at") and not merged.get("trained_at"):
                merged["trained_at"] = str(info["trained_at"])
            if info.get("training_samples") and not merged.get("training_samples"):
                try:
                    merged["training_samples"] = int(info["training_samples"])
                except (TypeError, ValueError):
                    pass
            engine_metrics = info.get("metrics")
            if isinstance(engine_metrics, dict):
                entry = dict(metrics_map.get(model_id) or {})
                for key in ("vec_rmse", "skill_vs_persistence", "latency_ms"):
                    if entry.get(key) is None and engine_metrics.get(key) is not None:
                        entry[key] = engine_metrics[key]
                if entry:
                    metrics_map[model_id] = entry
            meta_map[model_id] = merged

    models: list[ModelInfo] = []
    for entry in MODEL_CATALOG:
        model_id = entry["id"]
        metrics_entry = metrics_map.get(model_id) or {}
        meta_entry = meta_map.get(model_id) or {}
        latency = _num(metrics_entry.get("latency_ms"))
        if latency is None:
            latency = _num(meta_entry.get("latency_ms"))
        models.append(
            ModelInfo(
                id=model_id,
                name=entry["name"],
                short=entry["short"],
                category=entry["category"],
                color=entry["color"],
                available=bool(availability.get(model_id, False)),
                description=entry["description"],
                trained_at=meta_entry.get("trained_at"),
                training_samples=meta_entry.get("training_samples"),
                metrics=ModelMetricsBrief(
                    vec_rmse=_num(metrics_entry.get("vec_rmse")),
                    skill_vs_persistence=_num(metrics_entry.get("skill_vs_persistence")),
                    latency_ms=latency,
                ),
                is_baseline=bool(entry["is_baseline"]),
            )
        )

    default_model = "random_forest" if availability.get("random_forest") else next(
        (mid for mid in MODEL_IDS if availability.get(mid)), "random_forest"
    )
    default_date = _resolve_default_date(store, data_time_raw)
    default_cell = Cell(row=store.settings.default_row, col=store.settings.default_col)

    combos = [
        Combo(model=m.id, date=default_date, cell=default_cell)
        for m in models
        if m.available
    ]

    return ModelsResponse(
        data_time=data_time,
        default=DefaultParams(model=default_model, date=default_date, cell=default_cell),
        models=models,
        combos=combos,
    )


@router.get(
    "/models",
    response_model=ModelsResponse,
    summary="模型列表与默认参数",
    description="合并 PredictionEngine / SimVPPredictor 可用性 + metrics.json 指标，生成模型列表、数据时间、默认预测参数。",
)
def list_models(store: ArtifactStore = Depends(get_store)) -> ModelsResponse:
    return build_models_payload(store)
