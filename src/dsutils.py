"""本工作区的通用数据 I/O 与可复现性小工具。

通过 .venv 的 .pth 文件把 <project>/src 加进了 sys.path，因此任何脚本/notebook
都可以直接 ``import dsutils``（无需安装、无需设置 PYTHONPATH）。

示例::

    import dsutils as du
    du.set_seed(42)
    lf = du.scan_any("data/processed/events.parquet")
    df = lf.group_by("user_id").agg(pl.len()).collect()
"""

from __future__ import annotations

import csv
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
METRICS_CSV = REPORT_DIR / "metrics.csv"

_READERS = {
    ".parquet": "parquet",
    ".pq": "parquet",
    ".arrow": "arrow",
    ".ipc": "arrow",
    ".feather": "arrow",
    ".csv": "csv",
    ".tsv": "csv",
    ".json": "json",
    ".ndjson": "json",
    ".jsonl": "json",
}


def set_seed(seed: int = 42) -> None:
    """固定 Python / NumPy 随机性（sklearn 的 random_state 仍需显式传入）。"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass


def scan_any(path: str | Path, *, glob: bool = False):
    """按扩展名返回 polars LazyFrame（parquet / arrow / csv / json）。"""
    import polars as pl

    path = Path(path)
    readers = {
        "parquet": pl.scan_parquet,
        "arrow": pl.scan_ipc,
        "csv": pl.scan_csv,
        "json": pl.scan_ndjson,
    }
    suffix = path.suffix.lower()
    if suffix not in _READERS:
        raise ValueError(f"不支持的文件类型: {path.suffix}（支持 {sorted(_READERS)}）")
    return readers[_READERS[suffix]](path)


def read_pandas(path: str | Path, *, columns: Iterable[str] | None = None, chunksize: int | None = None):
    """pandas 读取入口；超过内存的 CSV 请传 chunksize 分块处理。"""
    import pandas as pd

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=list(columns) if columns else None)
    if suffix in {".csv", ".tsv"}:
        kwargs: dict[str, Any] = {}
        if columns:
            kwargs["usecols"] = list(columns)
        if chunksize:
            kwargs["chunksize"] = chunksize
        return pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",", **kwargs)
    if suffix in {".json", ".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=suffix in {".jsonl", ".ndjson"})
    raise ValueError(f"pandas 读取不支持: {path.suffix}")


def write_parquet(df: Any, path: str | Path, *, compression: str = "zstd") -> Path:
    """统一写 parquet（polars / pandas 皆可），返回绝对路径。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import polars as pl

        if isinstance(df, pl.DataFrame):
            df.write_parquet(path, compression=compression)
            return path.resolve()
    except ImportError:  # pragma: no cover
        pass
    df.to_parquet(path, compression=compression, index=False)
    return path.resolve()


def parquet_summary(path: str | Path) -> dict[str, Any]:
    """不读全量数据，直接给出 parquet 的行数/列数/大小。"""
    import pyarrow.parquet as pq

    path = Path(path)
    meta = pq.ParquetFile(path).metadata
    return {
        "path": str(path.resolve()),
        "rows": meta.num_rows,
        "row_groups": meta.num_row_groups,
        "columns": meta.num_columns,
        "size_mb": round(path.stat().st_size / 1024**2, 2),
    }


def append_metrics(
    values: Mapping[str, Any],
    *,
    experiment: str = "default",
    params: Mapping[str, Any] | None = None,
    path: str | Path = METRICS_CSV,
) -> Path:
    """把一次实验的指标追加到 reports/metrics.csv（自动建表头）。

    每行形如: timestamp, experiment, metric, value, params
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    params_json = ""
    if params:
        import json

        params_json = json.dumps(dict(params), ensure_ascii=False, sort_keys=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if not exists:
            writer.writerow(["timestamp", "experiment", "metric", "value", "params"])
        for metric, value in values.items():
            writer.writerow([stamp, experiment, metric, value, params_json])
    return path.resolve()
