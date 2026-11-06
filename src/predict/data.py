"""zarr 数据访问层 `DataStore`：日期索引 / 帧 / 滑窗 / 有效像元 / 分块 LRU 缓存。

数据（见 zarr `.zattrs`）：
- `u`, `v`, `speed` (T,116,158) float32 cm/s，**无效像元存 0**（有效性一律看 `valid_mask`）
- `valid_mask` (T,116,158) uint8；`doy`（年内相位 [0,1)）、`month`（1..12）、`time`（日）
- `latitude`, `longitude` (116,158)；`stats/*` 为 1978–2017 每像元 z-score 统计

性能设计：zarr 的 chunk 沿时间轴为 256 帧，逐帧读会把同一 chunk 反复解压。因此本类按
**时间 chunk 整块缓存**（LRU，默认 2 块 ≈ 40MB），窗口/帧读取全部命中缓存；单点推理冷启动
约 100ms（解压一块），之后 1–3ms。
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import zarr

from .common import (
    DEFAULT_ZARR,
    GRID_H,
    GRID_W,
    date_str,
    resolve_path,
    year_of_dates,
)

_FRAME_VARS = ("u", "v", "speed", "mask", "doy")


class DataStore:
    """zarr 只读访问 + 日期索引 + 窗口读取 + 有效像元。线程安全（内部 RLock）。"""

    def __init__(self, zarr_path: str | Path | None = DEFAULT_ZARR,
                 *, max_cached_chunks: int = 2):
        self.zarr_path = resolve_path(zarr_path, default=DEFAULT_ZARR)
        if not self.zarr_path.exists():
            raise FileNotFoundError(f"zarr 不存在: {self.zarr_path}")
        self.zg = zarr.open_group(str(self.zarr_path), mode="r")
        self._u = self.zg["u"]
        self._v = self.zg["v"]
        self._speed = self.zg["speed"]
        self._mask = self.zg["valid_mask"]
        self._doy = self.zg["doy"]
        self._month = self.zg["month"]
        self._valid_fraction = self.zg["valid_fraction"]

        self.n_frames: int = int(self._u.shape[0])
        self.shape: tuple[int, int] = (int(self._u.shape[1]), int(self._u.shape[2]))
        self.dates: np.ndarray = self.zg["time"][:].astype("datetime64[D]")   # (T,)
        self.years: np.ndarray = year_of_dates(self.dates)                    # (T,)
        self.doy_all: np.ndarray = np.asarray(self._doy[:], dtype=np.float32)  # (T,)
        self.month_all: np.ndarray = np.asarray(self._month[:], dtype=np.int16)
        self.valid_fraction_all: np.ndarray = np.asarray(self._valid_fraction[:], dtype=np.float32)
        self.latitude: np.ndarray = np.asarray(self.zg["latitude"][:], dtype=np.float32)
        self.longitude: np.ndarray = np.asarray(self.zg["longitude"][:], dtype=np.float32)
        self.attrs: dict = dict(self.zg.attrs)

        # 时间 chunk 大小（zarr v2 为 (256,116,158)）；按 chunk 整块缓存
        chunks = getattr(self._u, "chunks", None)
        self.chunk_frames: int = int(chunks[0]) if chunks else 256
        self._max_cached_chunks = int(max_cached_chunks)
        self._cache: "OrderedDict[int, dict]" = OrderedDict()
        self._lock = threading.RLock()

    # ------------------------------------------------------------ 基础属性
    @property
    def created_utc(self) -> str | None:
        """zarr 构建时间（.zattrs created_utc），供 data_time() 使用。"""
        return self.attrs.get("created_utc")

    @property
    def grid(self) -> tuple[int, int]:
        return self.shape

    def date_of(self, i: int) -> str:
        return date_str(self.dates[int(i)])

    # ------------------------------------------------------------ 日期索引
    @staticmethod
    def _to_day(date) -> np.datetime64:
        if isinstance(date, np.datetime64):
            return date.astype("datetime64[D]")
        if isinstance(date, str):
            return np.datetime64(date, "D")
        # datetime / pd.Timestamp
        return np.datetime64(str(date)[:10], "D")

    def idx_of_date(self, date) -> int | None:
        """精确匹配日期 -> 帧索引；不存在返回 None。"""
        d = self._to_day(date)
        i = int(np.searchsorted(self.dates, d))
        if 0 <= i < self.n_frames and self.dates[i] == d:
            return i
        return None

    def nearest_idx(self, date) -> int:
        """最近帧索引（超出范围时夹到端点）。"""
        d = self._to_day(date)
        i = int(np.searchsorted(self.dates, d))
        if i >= self.n_frames:
            return self.n_frames - 1
        if i <= 0:
            return 0
        before = self.dates[i - 1]
        return i - 1 if (d - before) <= (self.dates[i] - d) else i

    def next_idx(self, i: int) -> int:
        """下一帧索引；越界抛 ValueError（dataset 最后一帧没有"下一帧"）。"""
        i = int(i)
        if i < 0 or i >= self.n_frames - 1:
            raise ValueError(f"next_idx 越界: {i}（n_frames={self.n_frames}，最后一帧无下一帧）")
        return i + 1

    def gap_days(self, i: int) -> int:
        """帧 i 与下一帧的自然日间隔（正常 1，数据集缺失日导致 2–8）。"""
        j = self.next_idx(i)
        return int((self.dates[j] - self.dates[i]) / np.timedelta64(1, "D"))

    def indices_of_years(self, year_lo: int, year_hi: int) -> np.ndarray:
        """年份闭区间内的帧索引（连续区间）。"""
        return np.nonzero((self.years >= int(year_lo)) & (self.years <= int(year_hi)))[0]

    def month_of(self, i: int) -> int:
        return int(self.month_all[int(i)])

    def valid_fraction_of(self, i: int) -> float:
        return float(self.valid_fraction_all[int(i)])

    # ------------------------------------------------------------ 块缓存
    def _chunk_index(self, i: int) -> int:
        return int(i) // self.chunk_frames

    def _load_chunk(self, ci: int) -> dict:
        start = ci * self.chunk_frames
        stop = min(start + self.chunk_frames, self.n_frames)
        blk = {
            "u": np.asarray(self._u[start:stop], dtype=np.float32),
            "v": np.asarray(self._v[start:stop], dtype=np.float32),
            "speed": np.asarray(self._speed[start:stop], dtype=np.float32),
            "mask": np.asarray(self._mask[start:stop], dtype=np.uint8),
            "doy": np.asarray(self._doy[start:stop], dtype=np.float32),
            "start": start,
            "stop": stop,
        }
        return blk

    def _get_chunk(self, ci: int) -> dict:
        with self._lock:
            blk = self._cache.get(ci)
            if blk is not None:
                self._cache.move_to_end(ci)
                return blk
        blk = self._load_chunk(ci)          # 在锁外解压，避免阻塞其他读
        with self._lock:
            cur = self._cache.get(ci)
            if cur is not None:             # 并发重复加载：以先到者为准
                self._cache.move_to_end(ci)
                return cur
            self._cache[ci] = blk
            while len(self._cache) > self._max_cached_chunks:
                self._cache.popitem(last=False)
        return blk

    def read_range(self, start: int, stop: int) -> dict:
        """读取 [start, stop) 连续帧，返回 {u,v,speed,mask,doy} 数组 (n,H,W)/(n,)。"""
        start, stop = int(start), int(stop)
        if start < 0:
            raise ValueError(f"read_range start<0: {start}")
        stop = min(stop, self.n_frames)
        if stop <= start:
            raise ValueError(f"read_range 空区间: [{start}, {stop})")
        n = stop - start
        out = {
            "u": np.empty((n, GRID_H, GRID_W), np.float32),
            "v": np.empty((n, GRID_H, GRID_W), np.float32),
            "speed": np.empty((n, GRID_H, GRID_W), np.float32),
            "mask": np.empty((n, GRID_H, GRID_W), np.uint8),
            "doy": np.empty((n,), np.float32),
        }
        pos = 0
        while pos < n:
            gi = start + pos
            ci = self._chunk_index(gi)
            blk = self._get_chunk(ci)
            lo = gi - blk["start"]
            m = min(blk["stop"] - gi, n - pos)
            for k in _FRAME_VARS:
                out[k][pos:pos + m] = blk[k][lo:lo + m]
            pos += m
        return out

    # ------------------------------------------------------------ 帧 / 窗口 / 像元
    def frame(self, i: int) -> dict:
        """单帧：{u,v,speed,mask,doy,date,coverage,index}（无效像元 u/v/speed 为 0）。"""
        i = int(i)
        r = self.read_range(i, i + 1)
        return {
            "u": r["u"][0],
            "v": r["v"][0],
            "speed": r["speed"][0],
            "mask": r["mask"][0],
            "doy": float(r["doy"][0]),
            "date": self.date_of(i),
            "coverage": self.valid_fraction_of(i),
            "index": i,
        }

    def window(self, end_idx: int, n_days: int) -> dict:
        """以 end_idx 为最后一帧的滑窗（旧→新），返回
        {u,v,speed,mask,doy} 数组 (n_days,H,W)/(n_days,) 以及 latitude/longitude/date/coverage。
        """
        end_idx = int(end_idx)
        n_days = int(n_days)
        start = end_idx - n_days + 1
        if start < 0:
            raise ValueError(f"window 需要 {n_days} 帧历史，但 end_idx={end_idx}")
        r = self.read_range(start, end_idx + 1)
        r["date"] = self.date_of(end_idx)
        r["end_idx"] = end_idx
        r["coverage"] = self.valid_fraction_of(end_idx)
        r["coverage_hist"] = np.asarray(self.valid_fraction_all[start:end_idx + 1], np.float32)
        r["month"] = int(self.month_all[end_idx])
        r["latitude"] = self.latitude
        r["longitude"] = self.longitude
        return r

    def valid_cells(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """帧 i 的有效像元 (rows, cols)，intp 数组；行/列为 0..H-1 / 0..W-1。"""
        i = int(i)
        blk = self._get_chunk(self._chunk_index(i))
        m = blk["mask"][i - blk["start"]]
        rows, cols = np.nonzero(m)
        return rows, cols

    def valid_mask(self, i: int) -> np.ndarray:
        i = int(i)
        blk = self._get_chunk(self._chunk_index(i))
        return blk["mask"][i - blk["start"]]

    def speed_field(self, i: int) -> np.ndarray:
        i = int(i)
        blk = self._get_chunk(self._chunk_index(i))
        return blk["speed"][i - blk["start"]]

    def cell_latlon(self, row: int, col: int) -> tuple[float, float]:
        return float(self.latitude[row, col]), float(self.longitude[row, col])

    # ------------------------------------------------------------ 便捷统计
    def n_valid_cells(self, i: int) -> int:
        return int(np.count_nonzero(self.valid_mask(i)))

    def train_frame_indices(self) -> np.ndarray:
        """1978–2017 全部帧索引。"""
        return self.indices_of_years(1978, 2017)

    def __repr__(self) -> str:  # pragma: no cover
        return (f"DataStore({self.zarr_path.name}, n_frames={self.n_frames}, "
                f"grid={self.shape}, chunk_frames={self.chunk_frames})")
