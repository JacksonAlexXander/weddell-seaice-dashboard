"""POST /api/predict —— 真实模型推理（单点下一帧漂移矢量 + 24h 逐时曲线）。"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_predict_service
from ..schemas import PredictRequest, PredictResponse
from ..services.predict_service import PredictService

router = APIRouter(tags=["predict"])


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="单点预测（真实推理）",
    description=(
        "读取起报日前真实历史数据 → 调用已训练 artifact（ridge/random_forest/persistence/"
        "climatology/simvp）推理下一帧 u,v → 返回预测卡 + 24h 逐时线性积分曲线 + 不确定度 + 提示。"
        "绝不重新训练、绝不写模型文件。"
    ),
)
def predict(
    request: PredictRequest,
    service: PredictService = Depends(get_predict_service),
) -> PredictResponse:
    return service.predict(request)
