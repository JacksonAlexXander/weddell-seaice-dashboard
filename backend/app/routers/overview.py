"""GET /api/overview —— 大屏总览（48 年趋势 / 月度气候态 / KPI / 数据时间）。"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_store
from ..schemas import APIError
from ..services.artifact_store import ArtifactStore

router = APIRouter(tags=["overview"])


@router.get(
    "/overview",
    summary="数据总览",
    description=(
        "优先读取 data/processed/dashboard/overview.json（Lead 最终合并）；"
        "缺失时由 timeseries.json + 数据时间动态组装并在 warnings 中提示。"
    ),
)
def get_overview(store: ArtifactStore = Depends(get_store)) -> dict:
    payload, warnings = store.overview()
    if payload is None:
        raise APIError(
            503,
            "artifact_unavailable",
            "总览 artifact 尚未生成",
            "等待 data/processed/dashboard/overview.json（或 timeseries.json）",
        )
    if warnings and not payload.get("warnings"):
        payload["warnings"] = warnings
    return payload
