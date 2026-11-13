"""Serve the browser dashboard with a public-facing route allowlist."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from starlette.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.main import app  # noqa: E402

PUBLIC_ROUTES = {
    ("GET", "/"),
    ("GET", "/api/health"),
    ("GET", "/api/models"),
    ("GET", "/api/metrics"),
    ("GET", "/api/overview"),
    ("GET", "/api/field"),
    ("GET", "/api/field/predicted"),
    ("GET", "/api/recent"),
    ("GET", "/api/recent/field"),
    ("POST", "/api/predict"),
    ("POST", "/api/backtest"),
}


async def browser_app(scope, receive, send):
    if scope["type"] != "http":
        await app(scope, receive, send)
        return
    method, path = scope["method"], scope["path"]
    allowed = (method, path) in PUBLIC_ROUTES or (
        method in {"GET", "HEAD"} and path.startswith("/assets/")
    )
    if not allowed:
        if (method, path) == ("POST", "/api/recent/refresh"):
            response = JSONResponse(
                {"error": {"code": "owner_refresh_only", "message": "浏览器演示的观测数据由维护者更新。"}},
                status_code=403,
            )
        else:
            response = JSONResponse({"detail": "Not found"}, status_code=404)
        await response(scope, receive, send)
        return
    await app(scope, receive, send)


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the Weddell Sea browser dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8786")))
    args = parser.parse_args()
    uvicorn.run(browser_app, host=args.host, port=args.port, access_log=False, limit_concurrency=24)
