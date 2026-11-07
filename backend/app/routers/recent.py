"""近期观测状态、手动更新和漂移场。"""
from fastapi import APIRouter, Query, Request

router = APIRouter(tags=["recent"])


@router.get("/recent")
def recent(request: Request):
    return request.app.state.recent_service.metadata()


@router.post("/recent/refresh")
def refresh(request: Request):
    return request.app.state.recent_service.refresh()


@router.get("/recent/field")
def field(request: Request, model: str | None = None, date: str | None = None, lead_days: int = Query(1, ge=1, le=3),
          stride: int = Query(5, ge=1, le=30)):
    return request.app.state.recent_service.field(model, lead_days, stride, date)
