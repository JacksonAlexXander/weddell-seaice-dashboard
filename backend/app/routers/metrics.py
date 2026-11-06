"""GET /api/metrics —— 统一评估协议指标（Lead 合并的 metrics.json，缺失时回退）。"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_store
from ..schemas import APIError
from ..services.artifact_store import ArtifactStore

router = APIRouter(tags=["metrics"])


@router.get(
    "/metrics",
    summary="模型统一指标",
    description=(
        "直接读取 data/processed/dashboard/metrics.json（Lead 最终合并）；"
        "缺失时回退 metrics_tabular.json / metrics_simvp.json 并在 warnings 中提示。"
    ),
)
def get_metrics(store: ArtifactStore = Depends(get_store)) -> dict:
    payload, warnings = store.metrics()
    if payload is None:
        raise APIError(
            503,
            "artifact_unavailable",
            "指标 artifact 尚未生成",
            "等待离线评估产物 data/processed/dashboard/metrics.json（或 metrics_tabular.json）",
        )
    if warnings and not payload.get("warnings"):
        payload["warnings"] = warnings
    return payload
