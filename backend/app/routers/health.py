"""GET /api/health —— 健康检查（服务启动绝不崩，降级信息也如实返回）。"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from ..deps import get_started_at, get_store
from ..schemas import HealthResponse
from ..services.artifact_store import ArtifactStore

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="健康检查",
    description="服务存活状态、已加载模型列表与 artifacts 完整性。任何 artifact 缺失都不会导致本端点失败。",
)
def health(
    store: ArtifactStore = Depends(get_store),
    started_at: float = Depends(get_started_at),
) -> HealthResponse:
    now = time.time()
    uptime = round(now - started_at, 1) if started_at else 0.0
    return HealthResponse(
        status="ok",
        service=store.settings.service_name,
        version=store.settings.version,
        uptime_s=max(uptime, 0.0),
        models_loaded=store.loaded_model_ids(),
        artifacts_ok=store.artifacts_ok(),
    )
