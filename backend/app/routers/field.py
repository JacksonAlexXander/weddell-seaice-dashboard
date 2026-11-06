"""GET /api/field 与 /api/field/predicted —— 观测场 / 预测场（地图数据）。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ..deps import get_field_service
from ..schemas import ObservedFieldResponse, PredictedFieldResponse
from ..services.field_service import FieldService

router = APIRouter(tags=["field"])


@router.get(
    "/field",
    response_model=ObservedFieldResponse,
    summary="观测漂移场",
    description="读取数据集某一帧的 u/v/speed 场（无效像元 null），附抽样矢量、经纬度轴、覆盖统计与数据时间。",
)
def observed_field(
    date: str | None = Query(None, description="观测日 YYYY-MM-DD（默认最新有效日）"),
    stride: int = Query(5, ge=1, le=10, description="矢量抽样步长 1–10"),
    service: FieldService = Depends(get_field_service),
) -> ObservedFieldResponse:
    return service.observed_field(date, stride)


@router.get(
    "/field/predicted",
    response_model=PredictedFieldResponse,
    summary="预测漂移场",
    description=(
        "调用已加载模型预测下一帧整场；若目标帧真值存在，stats.comparison 给出该日矢量 RMSE。"
    ),
)
def predicted_field(
    model: str = Query(..., description="模型 id（/api/models 中 available=true）"),
    date: str = Query(..., description="起报日（最后观测日）YYYY-MM-DD"),
    stride: int = Query(5, ge=1, le=10, description="矢量抽样步长 1–10"),
    service: FieldService = Depends(get_field_service),
) -> PredictedFieldResponse:
    return service.predicted_field(model, date, stride)
