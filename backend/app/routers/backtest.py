"""POST /api/backtest —— 历史回测（真实值 vs 模型逐日预测）。"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_predict_service
from ..schemas import BacktestRequest, BacktestResponse
from ..services.predict_service import PredictService

router = APIRouter(tags=["backtest"])


@router.post(
    "/backtest",
    response_model=BacktestResponse,
    summary="历史回测",
    description=(
        "对指定像元回测一段时间：series[i].date 是目标日（预测日），"
        "obs_* 为目标日真值，pred_* 为从上一帧出发的模型预测；附 RMSE / 技能分摘要。"
    ),
)
def backtest(
    request: BacktestRequest,
    service: PredictService = Depends(get_predict_service),
) -> BacktestResponse:
    return service.backtest(request)
