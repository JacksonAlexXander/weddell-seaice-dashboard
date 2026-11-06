#!/usr/bin/env python
"""海冰漂移预测大屏后端启动入口。

用法::

    .venv/bin/python scripts/serve_dashboard.py                 # http://127.0.0.1:8000
    .venv/bin/python scripts/serve_dashboard.py --port 8010     # 换端口
    .venv/bin/python scripts/serve_dashboard.py --reload        # 开发热重载

启动后：
    - 大屏（若 dashboard/dist 已构建）：http://127.0.0.1:8000/
    - OpenAPI 文档：                    http://127.0.0.1:8000/docs
    - 健康检查：                        http://127.0.0.1:8000/api/health

本脚本只加载已训练 artifact 做推理，绝不触发训练，也绝不写 models/ 与 data/。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# 确保可 import backend.app.* 与 src.*（src 为命名空间包）
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="启动海冰漂移 ML 预测服务（FastAPI + uvicorn）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default=os.environ.get("PREDICT_HOST", "127.0.0.1"), help="监听地址")
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("PREDICT_PORT", "8000")), help="监听端口"
    )
    parser.add_argument(
        "--log-level", default=os.environ.get("PREDICT_LOG_LEVEL", "info"), help="uvicorn 日志级别"
    )
    parser.add_argument("--reload", action="store_true", help="开发模式热重载（仅监听代码变更）")
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="启动时不在后台预热模型引擎（首个预测请求会变慢）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.no_warmup:
        os.environ["PREDICT_WARMUP"] = "0"

    # uvicorn 以字符串方式导入 backend.app.main:app（与 ROOT 在同一进程，sys.path 已就绪）
    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=args.reload,
        reload_dirs=[str(ROOT / "backend"), str(ROOT / "src")] if args.reload else None,
        app_dir=str(ROOT),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
