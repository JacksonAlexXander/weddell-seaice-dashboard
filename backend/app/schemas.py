"""Pydantic v2 请求/响应模型（逐字段对齐 docs/PREDICT_API_CONTRACT.md）。

约定
----
- 正常响应 HTTP 200；错误统一 ``ErrorResponse``（见 :class:`ErrorResponse` / :class:`APIError`）。
- 请求参数校验用 ``PydanticCustomError`` 携带契约错误码：
  ``invalid_cell``（400）、``date_invalid`` / ``invalid_range``（422）。
- 响应模型字段名严格按契约，前端硬编码依赖。
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from .config import get_settings

DATE_FMT = "%Y-%m-%d"

WarningLevel = Literal["info", "warning", "danger"]


# ---------------------------------------------------------------------------
# 通用
# ---------------------------------------------------------------------------
class WarningItem(BaseModel):
    """契约 Warning：异常/提示。"""

    code: str = Field(..., description="机器可读的提示码，如 high_speed")
    level: WarningLevel = Field("info", description="info | warning | danger")
    message: str = Field(..., description="面向用户的中文提示")


class DataTime(BaseModel):
    """DataTime —— 数据时间（header / 预测状态区显示）。"""

    first_date: str
    latest_date: str
    n_frames: int
    built_at: str | None = None
    batch_age_days: int | None = None
    source: str = ""
    doi: str = ""


class ErrorDetail(BaseModel):
    code: str = Field(..., description="契约错误码，如 unknown_model")
    message: str
    hint: str | None = None


class ErrorResponse(BaseModel):
    """所有错误响应的统一 body。"""

    status: Literal["error"] = "error"
    error: ErrorDetail


class APIError(Exception):
    """业务错误：会被 main.py 的异常处理器转成 ErrorResponse。"""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        hint: str | None = None,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(f"{code}: {message}")
        self.status_code = int(status_code)
        self.code = code
        self.message = message
        self.hint = hint
        self.headers = headers

    def to_response(self) -> ErrorResponse:
        return ErrorResponse(error=ErrorDetail(code=self.code, message=self.message, hint=self.hint))


# 错误码 -> HTTP 状态码（契约错误表 + 降级附加码）
ERROR_STATUS: dict[str, int] = {
    "invalid_cell": 400,
    "unknown_model": 404,
    "not_found": 404,
    "method_not_allowed": 405,
    "model_unavailable": 409,
    "date_invalid": 422,
    "date_not_in_dataset": 422,
    "date_out_of_range": 422,
    "insufficient_history": 422,
    "no_future_frame": 422,
    "invalid_range": 422,
    "backtest_out_of_range": 422,
    "validation_error": 422,
    "artifact_unavailable": 503,
    "inference_failed": 500,
    "internal_error": 500,
    "response_validation_error": 500,
}


def error_status(code: str, default: int = 500) -> int:
    return ERROR_STATUS.get(code, default)


# ---------------------------------------------------------------------------
# 请求体校验工具
# ---------------------------------------------------------------------------
def validate_date_string(value: object, field_name: str = "date") -> str:
    """严格校验 ``YYYY-MM-DD``；失败抛 ``date_invalid``（422）。"""
    if not isinstance(value, str):
        raise PydanticCustomError(
            "date_invalid", "{field} 必须为 YYYY-MM-DD 字符串", {"field": field_name}
        )
    if len(value) != 10:
        raise PydanticCustomError(
            "date_invalid",
            "{field} 日期格式错误：{value}，须为 YYYY-MM-DD",
            {"field": field_name, "value": value},
        )
    try:
        parsed = datetime.strptime(value, DATE_FMT)
    except ValueError as exc:  # noqa: F841
        raise PydanticCustomError(
            "date_invalid",
            "{field} 日期格式错误：{value}，须为 YYYY-MM-DD",
            {"field": field_name, "value": value},
        ) from None
    if parsed.strftime(DATE_FMT) != value:
        raise PydanticCustomError(
            "date_invalid",
            "{field} 日期无效：{value}",
            {"field": field_name, "value": value},
        )
    return value


def _coerce_cell_index(value: object, name: str, upper: int) -> int:
    """把 row/col 转成合法整数；非整数/越界抛 ``invalid_cell``（400）。"""
    if isinstance(value, bool):
        raise PydanticCustomError("invalid_cell", "{name} 必须为整数", {"name": name})
    if isinstance(value, int):
        idx = value
    elif isinstance(value, float):
        if not float(value).is_integer():
            raise PydanticCustomError(
                "invalid_cell", "{name} 必须为整数（收到 {value}）", {"name": name, "value": value}
            )
        idx = int(value)
    elif isinstance(value, str):
        try:
            idx = int(value.strip())
        except (TypeError, ValueError):
            raise PydanticCustomError(
                "invalid_cell", "{name} 必须为整数（收到 {value}）", {"name": name, "value": value}
            ) from None
    else:
        raise PydanticCustomError("invalid_cell", "{name} 必须为整数", {"name": name})
    if not (0 <= idx < upper):
        raise PydanticCustomError(
            "invalid_cell",
            "{name} 越界：须在 0–{upper_minus} 之间（收到 {value}）",
            {"name": name, "upper": upper, "upper_minus": upper - 1, "value": idx},
        )
    return idx


class Cell(BaseModel):
    """网格像元坐标。row 0–115（行增大=北），col 0–157（列增大=东）。"""

    row: int = Field(..., description="行号 0–115（行增大 = 纬度增大/北）")
    col: int = Field(..., description="列号 0–157（列增大 = 经度增大/东）")

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _check_cell(cls, data):
        if not isinstance(data, dict):
            raise PydanticCustomError("invalid_cell", "cell 必须是包含 row/col 的对象")
        settings = get_settings()
        out = dict(data)
        if "row" in data and data["row"] is not None:
            out["row"] = _coerce_cell_index(data["row"], "row", settings.grid_rows)
        if "col" in data and data["col"] is not None:
            out["col"] = _coerce_cell_index(data["col"], "col", settings.grid_cols)
        return out


class PredictRequest(BaseModel):
    """POST /api/predict 请求体。"""

    model: str = Field(..., description="模型 id（/api/models 中 available=true）")
    date: str = Field(..., description="最后观测日 YYYY-MM-DD，须存在于数据集")
    cell: Cell

    model_config = ConfigDict(extra="ignore")

    @field_validator("date")
    @classmethod
    def _valid_date(cls, v: str) -> str:
        return validate_date_string(v, "date")


class BacktestRequest(BaseModel):
    """POST /api/backtest 请求体。"""

    model: str = Field(..., description="模型 id")
    cell: Cell
    start: str | None = Field(None, description="起始目标日 YYYY-MM-DD；默认 end 前 180 天")
    end: str | None = Field(None, description="结束目标日 YYYY-MM-DD；默认数据集最新日")
    max_points: int | None = Field(180, description="返回点数上限 1–365")

    model_config = ConfigDict(extra="ignore")

    @field_validator("start", "end")
    @classmethod
    def _valid_optional_date(cls, v: str | None, info) -> str | None:
        if v is None:
            return None
        try:
            return validate_date_string(v, info.field_name)
        except PydanticCustomError:
            # 回测日期格式问题按 invalid_range（422）返回，符合契约错误表
            raise PydanticCustomError(
                "invalid_range",
                "{field} 日期格式错误：{value}，须为 YYYY-MM-DD",
                {"field": info.field_name, "value": v},
            ) from None

    @field_validator("max_points", mode="before")
    @classmethod
    def _valid_max_points(cls, v):
        if v is None:
            return None
        if isinstance(v, bool):
            raise PydanticCustomError("invalid_range", "max_points 必须为 1–365 的整数")
        if isinstance(v, float):
            if not v.is_integer():
                raise PydanticCustomError("invalid_range", "max_points 必须为 1–365 的整数")
            v = int(v)
        if not isinstance(v, int):
            raise PydanticCustomError("invalid_range", "max_points 必须为 1–365 的整数")
        if not (1 <= v <= 365):
            raise PydanticCustomError(
                "invalid_range", "max_points 越界：须在 1–365 之间（收到 {value}）", {"value": v}
            )
        return v

    @model_validator(mode="after")
    def _check_range_order(self):
        if self.start and self.end and self.start > self.end:
            raise PydanticCustomError(
                "invalid_range",
                "start（{start}）不能晚于 end（{end}）",
                {"start": self.start, "end": self.end},
            )
        return self


# ---------------------------------------------------------------------------
# /api/health
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = "ok"
    service: str
    version: str
    uptime_s: float
    models_loaded: list[str]
    artifacts_ok: bool


# ---------------------------------------------------------------------------
# /api/models
# ---------------------------------------------------------------------------
class ModelMetricsBrief(BaseModel):
    vec_rmse: float | None = None
    skill_vs_persistence: float | None = None
    latency_ms: float | None = None


class ModelInfo(BaseModel):
    id: str
    name: str
    short: str
    category: str
    color: str
    available: bool
    description: str
    trained_at: str | None = None
    training_samples: int | None = None
    metrics: ModelMetricsBrief = Field(default_factory=ModelMetricsBrief)
    is_baseline: bool = False


class Combo(BaseModel):
    model: str
    date: str
    cell: Cell


class DefaultParams(BaseModel):
    model: str
    date: str
    cell: Cell


class ModelsResponse(BaseModel):
    data_time: DataTime
    default: DefaultParams
    models: list[ModelInfo]
    combos: list[Combo]


# ---------------------------------------------------------------------------
# /api/field
# ---------------------------------------------------------------------------
class GridInfo(BaseModel):
    rows: int
    cols: int
    cell_km: float


class Vectors(BaseModel):
    stride: int
    rows: list[int]
    cols: list[int]
    u: list[float]
    v: list[float]
    speed: list[float]


class FieldStats(BaseModel):
    valid_cells: int
    coverage: float
    mean_speed: float | None = None
    max_speed: float | None = None


class ObservedFieldResponse(BaseModel):
    kind: Literal["observed"] = "observed"
    date: str
    grid: GridInfo
    row_axis: list[int]
    col_axis: list[int]
    row_lat: list[float]
    col_lon: list[float]
    speed: list[list[float | None]]
    vectors: Vectors
    stats: FieldStats
    data_time: DataTime


class Comparison(BaseModel):
    n: int
    vec_rmse: float | None = None


class PredictedFieldStats(FieldStats):
    comparison: Comparison | None = None


class ModelRef(BaseModel):
    id: str
    name: str


class ModelRefWithCategory(ModelRef):
    category: str


class PredictedFieldResponse(BaseModel):
    kind: Literal["predicted"] = "predicted"
    model: ModelRef
    date: str
    target_date: str
    grid: GridInfo
    row_axis: list[int]
    col_axis: list[int]
    row_lat: list[float]
    col_lon: list[float]
    speed: list[list[float | None]]
    vectors: Vectors
    stats: PredictedFieldStats
    elapsed_ms: float
    warnings: list[WarningItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# POST /api/predict
# ---------------------------------------------------------------------------
class PredictInput(BaseModel):
    row: int
    col: int
    lat: float
    lon: float
    date: str
    u: float
    v: float
    speed: float
    direction_deg: float
    valid: bool
    coverage: float


class PredictDataTime(BaseModel):
    last_obs_date: str
    last_obs_index: int
    next_frame_date: str
    target_observed: bool
    horizon_hours: int = 24
    latest_date: str
    n_frames: int


class Prediction(BaseModel):
    u: float
    v: float
    speed: float
    direction_deg: float
    unit: str = "cm/s"
    displacement_km_24h: float
    bearing_deg: float
    delta_speed_vs_persistence: float | None = None


class UncertaintySeries(BaseModel):
    method: str
    lower_km: list[float]
    upper_km: list[float]


class Series24h(BaseModel):
    hours: list[int]
    cumulative_km: list[float]
    speed_cm_s: list[float]
    row: list[float]
    col: list[float]
    lat: list[float]
    lon: list[float]
    uncertainty: UncertaintySeries | None = None
    interpolation: str
    target_date: str


class PredictMetrics(BaseModel):
    vec_rmse: float | None = None
    skill_vs_persistence: float | None = None


class PredictResponse(BaseModel):
    status: str = "ok"
    status_text: str = "预测成功"
    request_id: str
    elapsed_ms: float
    model: ModelRefWithCategory
    data_time: PredictDataTime
    input: PredictInput
    prediction: Prediction
    series_24h: Series24h
    metrics: PredictMetrics | None = None
    warnings: list[WarningItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# POST /api/backtest
# ---------------------------------------------------------------------------
class BacktestCell(BaseModel):
    row: int
    col: int
    lat: float
    lon: float


class BacktestRange(BaseModel):
    requested_start: str
    requested_end: str
    available_start: str
    available_end: str
    n_points: int
    n_days_total: int


class BacktestSeriesItem(BaseModel):
    date: str
    obs_u: float | None = None
    obs_v: float | None = None
    obs_speed: float | None = None
    pred_u: float | None = None
    pred_v: float | None = None
    pred_speed: float | None = None
    err_vec: float | None = None


class BacktestSummary(BaseModel):
    n: int = 0
    rmse_u: float | None = None
    rmse_v: float | None = None
    vec_rmse: float | None = None
    mae_vec: float | None = None
    r2_u: float | None = None
    r2_v: float | None = None
    skill_vs_persistence: float | None = None


class BacktestResponse(BaseModel):
    status: str = "ok"
    model: ModelRef
    cell: BacktestCell
    range: BacktestRange
    series: list[BacktestSeriesItem]
    summary: BacktestSummary
    warnings: list[WarningItem] = Field(default_factory=list)
