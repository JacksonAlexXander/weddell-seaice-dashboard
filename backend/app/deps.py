"""FastAPI 依赖注入：从 ``app.state`` 取进程级单例（store / services）。"""

from __future__ import annotations

from fastapi import Request

from .config import Settings, get_settings
from .schemas import APIError
from .services.artifact_store import ArtifactStore
from .services.field_service import FieldService
from .services.predict_service import PredictService


def get_app_settings() -> Settings:
    return get_settings()


def get_store(request: Request) -> ArtifactStore:
    store = getattr(request.app.state, "store", None)
    if store is None:  # 正常不会发生（lifespan 初始化），保底不 500
        raise APIError(503, "artifact_unavailable", "服务尚未完成初始化，请稍后重试")
    return store


def get_field_service(request: Request) -> FieldService:
    service = getattr(request.app.state, "field_service", None)
    if service is None:
        service = FieldService(get_store(request))
    return service


def get_predict_service(request: Request) -> PredictService:
    service = getattr(request.app.state, "predict_service", None)
    if service is None:
        service = PredictService(get_store(request))
    return service


def get_started_at(request: Request) -> float:
    return float(getattr(request.app.state, "started_at", 0.0))


__all__ = [
    "get_app_settings",
    "get_store",
    "get_field_service",
    "get_predict_service",
    "get_started_at",
]
