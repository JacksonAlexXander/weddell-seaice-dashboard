"""pytest 全局配置（backend/tests）。

- 把工作区根目录加入 sys.path，保证 ``import backend.app.*`` / ``import src.*`` 可用；
- 测试默认关闭启动后台 warmup（PREDICT_WARMUP=0），避免 session 级 fixture 与
  引擎加载竞态；需要真实引擎的用例通过请求触发懒加载。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("PREDICT_WARMUP", "0")
os.environ.setdefault("PREDICT_LOG_LEVEL", "WARNING")
