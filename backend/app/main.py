"""FastAPI 应用入口。

职责：
- 注册 7 个 API 路由（/api/*，字段严格按 docs/PREDICT_API_CONTRACT.md）
- 统一错误体 ErrorResponse、请求校验错误映射、request_id / elapsed_ms
- CORS 允许 Vite dev（5173）、/docs OpenAPI
- 静态托管 dashboard/dist（SPA fallback）；未构建时返回 JSON 构建提示
- lifespan：初始化 ArtifactStore + services；后台 warmup 引擎（不阻塞启动、失败不崩）

启动：``.venv/bin/python scripts/serve_dashboard.py --host 127.0.0.1 --port 8000``
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import Settings, get_settings
from .schemas import APIError, ErrorDetail, ErrorResponse, error_status
from .services.artifact_store import ArtifactStore
from .services.field_service import FieldService
from .services.predict_service import PredictService

logger = logging.getLogger("backend.app")


class UTF8JSONResponse(JSONResponse):
    """契约要求所有 JSON 响应 Content-Type: application/json; charset=utf-8。"""

    media_type = "application/json; charset=utf-8"


# ---------------------------------------------------------------------------
# 日志（不覆盖宿主已配置的 handler）
# ---------------------------------------------------------------------------
def _configure_logging() -> None:
    level_name = os.environ.get("PREDICT_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    if not logging.root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    logging.getLogger("backend").setLevel(level)


# ---------------------------------------------------------------------------
# 统一错误响应
# ---------------------------------------------------------------------------
def _error_json(
    status_code: int,
    code: str,
    message: str,
    hint: str | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body = ErrorResponse(error=ErrorDetail(code=code, message=message, hint=hint))
    return UTF8JSONResponse(status_code=status_code, content=body.model_dump(), headers=headers)


_KNOWN_VALIDATION_CODES = {"invalid_cell", "date_invalid", "invalid_range"}


def _validation_error_payload(exc: RequestValidationError) -> tuple[int, str, str, str | None]:
    """把 pydantic/FastAPI 校验错误映射成契约错误码（invalid_cell→400 等）。"""
    errors = list(exc.errors() or [])
    for err in errors:
        etype = str(err.get("type") or "")
        if etype in _KNOWN_VALIDATION_CODES:
            loc = ".".join(str(part) for part in (err.get("loc") or ()) if part != "body")
            message = str(err.get("msg") or "请求参数不合法")
            hint = f"字段：{loc}" if loc else None
            return error_status(etype, 422), etype, message, hint
    if errors:
        first = errors[0]
        loc = ".".join(str(part) for part in (first.get("loc") or ()) if part != "body")
        message = str(first.get("msg") or "请求参数校验失败")
        detail = f"{loc}：{message}" if loc else message
    else:
        detail = "请求参数校验失败"
    return 422, "validation_error", f"请求参数校验失败：{detail}", "检查请求体字段类型、必填项与取值范围"


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(APIError)
    async def _api_error_handler(request: Request, exc: APIError) -> JSONResponse:  # noqa: ANN001
        logger.info(
            "APIError %s %s -> %s %s", request.method, request.url.path, exc.status_code, exc.code
        )
        return _error_json(exc.status_code, exc.code, exc.message, exc.hint, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(  # noqa: ANN001
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        status_code, code, message, hint = _validation_error_payload(exc)
        logger.info("请求校验失败 %s %s -> %s %s", request.method, request.url.path, code, message)
        return _error_json(status_code, code, message, hint)

    @app.exception_handler(ResponseValidationError)
    async def _response_validation_handler(  # noqa: ANN001
        request: Request, exc: ResponseValidationError
    ) -> JSONResponse:
        logger.error("响应模型校验失败 %s %s: %s", request.method, request.url.path, exc)
        return _error_json(500, "response_validation_error", "服务响应格式校验失败", "请查看后端日志")

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception_handler(  # noqa: ANN001
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        code_map = {
            400: "bad_request",
            401: "unauthorized",
            403: "forbidden",
            404: "not_found",
            405: "method_not_allowed",
        }
        code = code_map.get(exc.status_code, "http_error")
        detail = exc.detail if isinstance(exc.detail, str) else "请求失败"
        hint = None
        if exc.status_code == 404 and request.url.path.startswith("/api"):
            hint = "可用端点见 GET /openapi.json 或 /docs"
        return _error_json(exc.status_code, code, detail, hint, getattr(exc, "headers", None))

    @app.exception_handler(Exception)
    async def _unhandled_handler(request: Request, exc: Exception) -> JSONResponse:  # noqa: ANN001
        logger.exception("未捕获异常 %s %s", request.method, request.url.path)
        return _error_json(500, "internal_error", "服务内部错误", "请查看后端日志")


# ---------------------------------------------------------------------------
# 静态托管 dashboard/dist（SPA fallback）
# ---------------------------------------------------------------------------
def _safe_static_file(dist: Path, rel_path: str) -> Path | None:
    """在 dist 内安全解析文件（防目录穿越）。"""
    if not rel_path or "\x00" in rel_path:
        return None
    try:
        base = dist.resolve()
        target = (base / rel_path).resolve()
        if not target.is_relative_to(base):
            return None
        if target.is_file():
            return target
    except (OSError, ValueError):
        return None
    return None


def _frontend_hint(settings: Settings) -> JSONResponse:
    dist = settings.dashboard_dist
    body = {
        "status": "error",
        "error": {
            "code": "frontend_not_built",
            "message": "前端构建产物不存在，先构建大屏或访问 /docs 查看 API",
            "hint": "cd dashboard && pnpm install && pnpm run build",
        },
        "dashboard_dist": str(dist),
        "build_command": "cd dashboard && pnpm install && pnpm run build",
    }
    return UTF8JSONResponse(status_code=200, content=body)


def _register_static_routes(app: FastAPI, settings: Settings) -> None:
    dist = settings.dashboard_dist
    index = dist / "index.html"

    @app.get("/", include_in_schema=False)
    def root_page():
        if index.is_file():
            return FileResponse(index)
        return _frontend_hint(settings)

    @app.get("/assets/{asset_path:path}", include_in_schema=False)
    def assets(asset_path: str):
        target = _safe_static_file(dist, f"assets/{asset_path}")
        if target is not None:
            return FileResponse(target)
        raise HTTPException(status_code=404, detail="静态资源不存在")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str):
        if full_path in ("api/predict", "api/backtest"):
            # 这些端点只接受 POST：GET 落到 catch-all 时按 405 返回（而非 404）
            raise HTTPException(status_code=405, detail="该 API 端点仅接受 POST")
        if full_path == "api" or full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API 路径不存在")
        # dist 内真实文件（favicon / manifest / vite.svg 等）
        target = _safe_static_file(dist, full_path)
        if target is not None:
            return FileResponse(target)
        # SPA 路由 fallback → index.html
        if index.is_file():
            return FileResponse(index)
        return _frontend_hint(settings)


# ---------------------------------------------------------------------------
# 应用工厂
# ---------------------------------------------------------------------------
def create_app(settings: Settings | None = None) -> FastAPI:
    _configure_logging()
    active_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # noqa: ANN202
        store = ArtifactStore(active_settings)
        app.state.settings = active_settings
        app.state.store = store
        app.state.field_service = FieldService(store)
        app.state.predict_service = PredictService(store)
        app.state.started_at = time.time()
        app.state.warmup_thread = None
        logger.info(
            "服务启动：zarr=%s dashboard=%s dist=%s warmup=%s",
            active_settings.zarr_path,
            active_settings.dashboard_dir,
            active_settings.dashboard_dist,
            active_settings.warmup,
        )
        if active_settings.warmup:
            thread = threading.Thread(target=store.warmup, name="predict-warmup", daemon=True)
            app.state.warmup_thread = thread
            thread.start()
        try:
            yield
        finally:
            logger.info("服务关闭")

    app = FastAPI(
        title="威德尔海海冰漂移 ML 预测服务",
        description=(
            "真实机器学习推理后端：Ridge / Random Forest / SimVP / 持续性 / 气候态。\n\n"
            "**绝不重新训练**：所有模型离线训练并落盘，服务只加载推理。\n\n"
            "字段级契约见 `docs/PREDICT_API_CONTRACT.md`。"
        ),
        version=active_settings.version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        default_response_class=UTF8JSONResponse,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(active_settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    @app.middleware("http")
    async def _access_log(request: Request, call_next):  # noqa: ANN001, ANN202
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        path = request.url.path
        if path.startswith("/api") or response.status_code >= 400:
            logger.info(
                "%s %s -> %s (%.1f ms)", request.method, path, response.status_code, elapsed_ms
            )
        return response

    _register_exception_handlers(app)

    # API 路由（务必先于静态 catch-all 注册）
    from .routers import backtest, field, health, metrics, models, overview, predict

    app.include_router(health.router, prefix="/api")
    app.include_router(models.router, prefix="/api")
    app.include_router(metrics.router, prefix="/api")
    app.include_router(overview.router, prefix="/api")
    app.include_router(field.router, prefix="/api")
    app.include_router(predict.router, prefix="/api")
    app.include_router(backtest.router, prefix="/api")

    _register_static_routes(app, active_settings)
    return app


app = create_app()


__all__ = ["app", "create_app"]
