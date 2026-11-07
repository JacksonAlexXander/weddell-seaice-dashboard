"""Fetch and regrid recent EUMETSAT OSI SAF Antarctic drift observations."""
from __future__ import annotations

import json
import os
import re
import ssl
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
from scipy.spatial import cKDTree

PRODUCT = "EUMETSAT OSI SAF OSI-405-d"
PRODUCT_URL = "https://osi-saf.eumetsat.int/products/osi-405-d"
DOI = "10.15770/EUM_SAF_OSI_NRT_2007"
SERVER = "https://thredds.met.no/thredds"
PREFIX = "osisaf/met.no/ice/drift_lr/merged"
FILENAME = re.compile(r"ice_drift_sh_polstere-625_multi-oi_(\d{12})-(\d{12})\.nc")
HISTORY_DAYS = 10
RADIUS_M = 6371228.0


def ease_xy(lat, lon):
    """EPSG:3409 projected coordinates from longitude and latitude in degrees."""
    lat, lon = np.radians(lat), np.radians(lon)
    rho = 2 * RADIUS_M * np.sin((np.pi / 2 + lat) / 2)
    return rho * np.sin(lon), rho * np.cos(lon)


def ease_latlon(x, y):
    lat = np.degrees(2 * np.arcsin(np.hypot(x, y) / (2 * RADIUS_M)) - np.pi / 2)
    return lat, np.degrees(np.arctan2(x, y))


def doy_phase(dates):
    dates = np.asarray(dates, dtype="datetime64[D]")
    year = dates.astype("datetime64[Y]")
    start = year.astype("datetime64[D]")
    end = (year + np.timedelta64(1, "Y")).astype("datetime64[D]")
    return ((dates - start) / (end - start)).astype(np.float32)


def _download(url, limit):
    import certifi

    context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(url, timeout=20, context=context) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("OSI SAF 文件超出读取上限")
    return data


def recent_files(now):
    """Resolve an uninterrupted 10-day run from two official monthly catalogs."""
    last_month = now.replace(day=1) - timedelta(days=1)
    months = (last_month.strftime("%Y/%m"), now.strftime("%Y/%m"))
    files = {}
    for month in months:
        try:
            catalog = _download(f"{SERVER}/catalog/{PREFIX}/{month}/catalog.xml", 1_000_000)
            root = ET.fromstring(catalog)
        except HTTPError as exc:
            if exc.code == 404:
                continue
            raise
        except ET.ParseError as exc:
            raise ValueError("OSI SAF 目录格式异常") from exc
        for node in root.iter():
            path = node.get("urlPath", "")
            match = FILENAME.fullmatch(path.rsplit("/", 1)[-1])
            if not match or path != f"{PREFIX}/{month}/{match[0]}":
                continue
            end = datetime.strptime(match[2], "%Y%m%d%H%M").replace(tzinfo=UTC)
            if end <= now:
                files[end] = path
    for end in sorted(files, reverse=True):
        dates = [end - timedelta(days=i) for i in range(HISTORY_DAYS - 1, -1, -1)]
        if all(date in files for date in dates):
            return [(date, files[date]) for date in dates]
    raise ValueError("OSI SAF 暂无连续 10 天的南极漂移观测")


def _read_frame(data, expected_end, x_grid, y_grid, domain):
    # NetCDF/HDF5 decoding is serialized. Network downloads run concurrently.
    from netCDF4 import Dataset

    with Dataset("osisaf.nc", memory=data) as nc:
        start = datetime.fromisoformat(nc.time_coverage_start)
        end = datetime.fromisoformat(nc.time_coverage_end)
        if nc.product_id != "OSI-405-d" or end != expected_end or end - start != timedelta(days=2):
            raise ValueError("OSI SAF 产品或时间元数据与目录不匹配")

        def field(name):
            return np.ma.asarray(nc[name][:], dtype=np.float64).filled(np.nan).squeeze()

        x0, y0 = ease_xy(field("lat"), field("lon"))
        x1, y1 = ease_xy(field("lat1"), field("lon1"))
        duration = (end - start).total_seconds() + field("dt1") - field("dt0")
        valid = np.isin(field("status_flag"), [20, 21, 22, 30])
        valid &= np.isfinite(x1) & np.isfinite(y1) & (duration > 36 * 3600) & (duration < 60 * 3600)
        u = np.divide((x1 - x0) * 100, duration, out=np.zeros_like(x0), where=valid)
        v = np.divide((y1 - y0) * 100, duration, out=np.zeros_like(y0), where=valid)

    # Use nearest source cell, including rejected cells in the search so that
    # regridding does not replace the nearest rejected observation with distant ice.
    source = np.isfinite(x0) & np.isfinite(y0)
    tree = cKDTree(np.column_stack((x0[source], y0[source])))
    distances, indices = tree.query(np.column_stack((x_grid.ravel(), y_grid.ravel())))
    mask = (valid[source][indices].reshape(domain.shape) & domain
            & (distances.reshape(domain.shape) <= 50_000))
    if not mask.any():
        raise ValueError(f"{end.date()} 在威德尔海没有有效漂移像元")
    return tuple(np.where(mask, data[source][indices].reshape(domain.shape), 0).astype(np.float32)
                 for data in (u, v)) + (mask,)


def refresh_snapshot(zarr_path, output, existing=None, now=None):
    """Download, validate and atomically replace the cached grid window."""
    import zarr

    now = now or datetime.now(UTC)
    entries = recent_files(now)
    urls = [f"{SERVER}/fileServer/{path}" for _, path in entries]
    if existing is not None and existing["meta"]["urls"] == urls:
        return existing
    grid = zarr.open_group(str(zarr_path), mode="r")
    x, y = np.meshgrid(grid["x"][:], grid["y"][:])
    domain = grid["stats"]["valid_days_count"][:] > 0

    def fetch(url):
        return _download(url, 4_000_000)

    with ThreadPoolExecutor(max_workers=4) as pool:
        data = list(pool.map(fetch, urls))
    frames = [_read_frame(blob, date, x, y, domain) for (date, _), blob in zip(entries, data)]
    dates = np.array([date.date().isoformat() for date, _ in entries], dtype="datetime64[D]")
    meta = {
        "source": PRODUCT, "source_url": PRODUCT_URL, "doi": DOI,
        "latest_at": entries[-1][0].isoformat(), "downloaded_at": now.isoformat(),
        "resolution_km": 62.5, "observation_span_hours": 48, "urls": urls,
    }
    u, v, mask = map(np.stack, zip(*frames))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".npz", delete=False) as handle:
        tmp = Path(handle.name)
        try:
            np.savez_compressed(handle, dates=dates, u=u, v=v, mask=mask, meta=json.dumps(meta))
            handle.flush()
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
    try:
        os.replace(tmp, output)
    finally:
        tmp.unlink(missing_ok=True)
    return load_snapshot(output)


def load_snapshot(path):
    with np.load(path, allow_pickle=False) as cache:
        dates, u, v, mask = (cache[key] for key in ("dates", "u", "v", "mask"))
        meta = json.loads(cache["meta"].item())
    if (dates.shape != (HISTORY_DAYS,) or u.shape != (HISTORY_DAYS, 116, 158)
            or u.shape != v.shape or u.shape != mask.shape
            or not mask.reshape(HISTORY_DAYS, -1).any(axis=1).all()
            or not np.isfinite(u).all() or not np.isfinite(v).all()
            or not np.all(np.diff(dates) == np.timedelta64(1, "D"))
            or meta["source"] != PRODUCT
            or str(dates[-1]) != datetime.fromisoformat(meta["latest_at"]).date().isoformat()):
        raise ValueError("近期观测缓存无效")
    return {"dates": dates, "u": u, "v": v, "mask": mask.astype(bool),
            "doy": doy_phase(dates), "meta": meta}
