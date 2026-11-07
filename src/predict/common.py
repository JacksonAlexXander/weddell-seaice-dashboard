"""src.predict 公共常量 / 路径解析 / 原子写工具。

所有路径默认相对**工作区根目录**解析（`Path(__file__).resolve().parents[2]`），
因此从任意 cwd（后端 uvicorn、pytest、notebook）导入本包都能找到 zarr 与 artifact。
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- 路径与常量
ROOT = Path(__file__).resolve().parents[2]          # 工作区根目录
DEFAULT_ZARR = "data/processed/weddell_icemotion_daily.zarr"
MODEL_DIR = "models/ml"
ARTIFACT_DIR = "data/processed/dashboard"
CACHE_DIR = "models/ml/_cache"
EVAL_WINDOWS_PATH = "data/processed/dashboard/eval_windows.json"

GRID_H, GRID_W = 116, 158
CELL_KM = 25.067525                # EASE-Grid South 实际像元间距（m 25067.525）
KM_PER_CM_S_24H = 0.036 * 24.0     # cm/s -> 24h 位移 km（0.864 km 每 cm/s）
KM_PER_CM_S_HOUR = 0.036           # cm/s -> km/h

SEED = 42
TRAIN_YEARS = (1978, 2017)
VAL_YEARS = (2018, 2021)
TEST_YEARS = (2022, 2025)
SPLITS = {"train": TRAIN_YEARS, "val": VAL_YEARS, "test": TEST_YEARS}

# 缺测日间隔 / 预警阈值（engine.predict_point 的 warnings 使用）
LOW_COVERAGE_THRESHOLD = 0.10       # 当日有效覆盖 < 10% -> low_coverage
PERSISTENCE_DELTA_WARN = 10.0       # 预测与持续性差异 > 10 cm/s -> persistence_divergence
EXTREME_SPEED_ABS = 60.0            # 预测速度绝对上限（cm/s）-> danger


def next_calendar_day(date) -> np.datetime64:
    return np.datetime64(date, "D") + np.timedelta64(1, "D")


def resolve_path(path: str | os.PathLike | None, *, default: str | None = None) -> Path:
    """把相对路径解析到工作区根；绝对路径原样返回。"""
    if path is None:
        if default is None:
            raise ValueError("path 与 default 不能同时为 None")
        path = default
    p = Path(path)
    if p.is_absolute():
        return p
    # 优先 cwd 相对（用户显式给出，如 models/ml），否则回退到工作区根
    cand = Path.cwd() / p
    if cand.exists():
        return cand
    return ROOT / p


def ensure_parent(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def json_safe(obj):
    """递归把 numpy / Path 转成可 json.dump 的原生类型。"""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def atomic_write_json(path: str | os.PathLike, obj) -> Path:
    """原子写 JSON（tmp + os.replace），避免服务端读到半截文件。"""
    path = ensure_parent(path)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(json_safe(obj), fh, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def atomic_joblib_dump(obj, path: str | os.PathLike, *, compress: int = 3) -> Path:
    """原子写 joblib（训练脚本落盘时使用，服务端加载不会读到半截文件）。"""
    import joblib

    path = ensure_parent(path)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    os.close(fd)
    try:
        joblib.dump(obj, tmp, compress=compress)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def atomic_savez(path: str | os.PathLike, **arrays) -> Path:
    """原子写 .npz。"""
    path = ensure_parent(path)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp.npz")
    os.close(fd)
    try:
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def load_json(path: str | os.PathLike):
    with open(resolve_path(path), "r", encoding="utf-8") as fh:
        return json.load(fh)


def year_of_dates(dates: np.ndarray) -> np.ndarray:
    """datetime64[D] -> 年份 int array。"""
    return dates.astype("datetime64[Y]").astype(np.int64) + 1970


def date_str(d: np.datetime64 | str) -> str:
    return str(np.datetime64(d, "D"))


def f4(x) -> float:
    """转原生 float，非有限值 -> None（JSON 安全）。"""
    v = float(x)
    return v if np.isfinite(v) else None


def r4(x, ndigits: int = 4):
    v = float(x)
    return round(v, ndigits) if np.isfinite(v) else None
