"""离线 artifact 构建 CLI（数据总览 / 气候态 / 像元统计 / 统一协议评估 / 指标）。

用法（工作区根目录，后台运行）::

    # 全量：三份数据 artifact + 统一协议评估 + metrics_tabular.json + eval_day1.parquet
    PYTHONPATH=src nohup .venv/bin/python -m predict.build_artifacts \
        > logs/ml_tabular_artifacts.log 2>&1 &

    # 也可以经由一次性入口脚本：
    .venv/bin/python scripts/build_predict_artifacts.py

产物（只写约定路径）：
- ``data/processed/dashboard/timeseries.json``  ：dataset KPI + annual(48) + monthly(12) + year_month(48×12) + data_time
- ``data/processed/dashboard/climatology.npz``  ：train 期逐月平均 u,v (12,116,158) + 逐月 speed std
- ``data/processed/dashboard/cell_stats.npz``   ：train 期逐像元 speed mean/std/p99/valid_days_count
- ``data/processed/dashboard/metrics_tabular.json``：4 个模型的统一协议指标（字段按 API 契约 /api/metrics）
- ``data/processed/dashboard/eval_day1.parquet``   ：评估明细（287 窗口 × 全部目标帧有效像元）

扫描纪律：timeseries / 气候态 / 像元统计用**一次分块扫描**（每块一个 zarr 时间 chunk ≈ 256 帧），
逐帧只保留标量、逐像元只累加 sum/sumsq/直方图，绝不一次性载入整张 (T,H,W)。
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

from .common import (
    ARTIFACT_DIR,
    CACHE_DIR,
    EVAL_WINDOWS_PATH,
    GRID_H,
    GRID_W,
    KM_PER_CM_S_24H,
    MODEL_DIR,
    ROOT,
    SEED,
    TEST_YEARS,
    TRAIN_YEARS,
    VAL_YEARS,
    atomic_savez,
    atomic_write_json,
    ensure_parent,
    load_json,
    resolve_path,
    utc_now_iso,
)
from .data import DataStore
from .engine import MODEL_IDS, MODEL_META, PredictionEngine
from .features import N_HISTORY, build_features
from .metrics import metric_defs, vec_metrics

CACHE = resolve_path(CACHE_DIR, default=CACHE_DIR)
SCAN_CACHE = CACHE / "scan_agg.npz"

# 像元 p99 直方图：0–200 cm/s，1000 个 0.2 cm/s 宽的 bin（误差 ±0.2 cm/s）
P99_HMAX = 200.0
P99_NBINS = 1000
P99_WIDTH = P99_HMAX / P99_NBINS


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024.0
    return f"{n:.1f}GB"


def _maxrss() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


# ---------------------------------------------------------------- 一次分块扫描
def scan_dataset(store: DataStore, *, force: bool = False) -> dict:
    """单次分块扫描：帧级统计 + train 期逐月场 + train 期逐像元统计（含 p99 直方图）。

    返回聚合 dict（并缓存到 ``models/ml/_cache/scan_agg.npz``，可续跑/复用）。
    """
    if SCAN_CACHE.exists() and not force:
        log(f"复用扫描缓存: {SCAN_CACHE}")
        with np.load(SCAN_CACHE) as d:
            return {k: d[k] for k in d.files}
    t0 = time.time()
    n = store.n_frames
    step = store.chunk_frames
    years, months = store.years, store.month_all

    frame_count = np.zeros(n, np.int32)          # 有效像元数
    frame_speed_sum = np.zeros(n, np.float64)    # 有效像元 speed 之和（cm/s）
    frame_coverage = np.zeros(n, np.float32)
    n_cells = GRID_H * GRID_W

    train_mask = (years >= TRAIN_YEARS[0]) & (years <= TRAIN_YEARS[1])
    mon = np.zeros((12, GRID_H, GRID_W), np.float64)      # 有效样本数（u/v/speed 共用）
    u_sum = np.zeros((12, GRID_H, GRID_W), np.float64)
    v_sum = np.zeros((12, GRID_H, GRID_W), np.float64)
    s_sum = np.zeros((12, GRID_H, GRID_W), np.float64)
    s2_sum = np.zeros((12, GRID_H, GRID_W), np.float64)

    cell_cnt = np.zeros(n_cells, np.int64)
    cell_s_sum = np.zeros(n_cells, np.float64)
    cell_s2_sum = np.zeros(n_cells, np.float64)
    hist = np.zeros((n_cells, P99_NBINS), np.int32)

    flat_cells = np.arange(n_cells, dtype=np.int64)
    for b0 in range(0, n, step):
        b1 = min(b0 + step, n)
        d = store.read_range(b0, b1)
        u, v, sp = d["u"], d["v"], d["speed"]
        mk = d["mask"] > 0
        cnt = mk.sum(axis=(1, 2)).astype(np.int32)
        frame_count[b0:b1] = cnt
        frame_coverage[b0:b1] = cnt / float(n_cells)
        frame_speed_sum[b0:b1] = (sp.astype(np.float64) * mk).sum(axis=(1, 2))

        # --- train 期：逐月场 + 逐像元统计
        sel = np.nonzero(train_mask[b0:b1])[0]
        if sel.size:
            mm = months[b0:b1][sel]
            for m in range(1, 13):
                fm = sel[mm == m]
                if fm.size == 0:
                    continue
                mkk = mk[fm]
                uu = u[fm] * mkk
                vv = v[fm] * mkk
                ss = sp[fm] * mkk
                k = m - 1
                mon[k] += mkk.sum(axis=0)
                u_sum[k] += uu.sum(axis=0)
                v_sum[k] += vv.sum(axis=0)
                s_sum[k] += ss.sum(axis=0)
                s2_sum[k] += (ss * ss).sum(axis=0)
            # 逐像元：用整块 mask 一次性摊平（train 帧）
            sub_mk = mk[sel].reshape(-1)
            sub_sp = sp[sel].reshape(-1)
            valid = sub_mk
            cells_v = np.tile(flat_cells, sel.size)[valid]
            sp_v = sub_sp[valid].astype(np.float64)
            cell_cnt += np.bincount(cells_v, minlength=n_cells)
            cell_s_sum += np.bincount(cells_v, weights=sp_v, minlength=n_cells)
            cell_s2_sum += np.bincount(cells_v, weights=sp_v * sp_v, minlength=n_cells)
            bins = np.clip((sp_v / P99_HMAX * P99_NBINS).astype(np.int32), 0, P99_NBINS - 1)
            np.add.at(hist, (cells_v, bins), 1)
        if (b0 // step) % 8 == 0 or b1 == n:
            log(f"  scan {b1:>6}/{n} 帧  elapsed={time.time() - t0:5.1f}s  "
                f"maxrss={_fmt_bytes(_maxrss())}")

    agg = {
        "frame_year": years.astype(np.int16),
        "frame_month": months.astype(np.int8),
        "frame_count": frame_count,
        "frame_speed_sum": frame_speed_sum,
        "frame_coverage": frame_coverage,
        "monthly_count": mon,
        "monthly_u_sum": u_sum,
        "monthly_v_sum": v_sum,
        "monthly_speed_sum": s_sum,
        "monthly_speed_sq_sum": s2_sum,
        "cell_count": cell_cnt,
        "cell_speed_sum": cell_s_sum,
        "cell_speed_sq_sum": cell_s2_sum,
        "cell_speed_hist": hist,
        "p99_hmax": np.array(P99_HMAX),
        "p99_nbins": np.array(P99_NBINS),
    }
    atomic_savez(SCAN_CACHE, **agg)
    log(f"扫描完成 {time.time() - t0:.1f}s -> {SCAN_CACHE} "
        f"({_fmt_bytes(SCAN_CACHE.stat().st_size)})")
    return agg


# ---------------------------------------------------------------- 写 artifacts
def write_timeseries(store: DataStore, agg: dict) -> Path:
    """48 年年度 / 12 月 / 年月矩阵 + KPI + 数据时间（API /api/overview 结构）。"""
    years = agg["frame_year"].astype(np.int32)
    months = agg["frame_month"].astype(np.int32)
    cnt = agg["frame_count"].astype(np.float64)
    ssum = agg["frame_speed_sum"].astype(np.float64)
    cov = agg["frame_coverage"].astype(np.float64)

    def grp_mean_speed(sel):
        c = float(cnt[sel].sum())
        return float(ssum[sel].sum() / c) if c > 0 else None

    def grp_cov(sel):
        return float(cov[sel].mean()) if sel.any() else None

    annual = []
    for y in range(int(years.min()), int(years.max()) + 1):
        sel = years == y
        annual.append({
            "year": int(y),
            "mean_speed": round(grp_mean_speed(sel), 4) if grp_mean_speed(sel) is not None else None,
            "coverage": round(grp_cov(sel), 6) if grp_cov(sel) is not None else None,
            "n_days": int(sel.sum()),
        })
    monthly = []
    for m in range(1, 13):
        sel = months == m
        monthly.append({
            "month": int(m),
            "mean_speed": round(grp_mean_speed(sel), 4) if grp_mean_speed(sel) is not None else None,
            "coverage": round(grp_cov(sel), 6) if grp_cov(sel) is not None else None,
        })
    annual_years = [a["year"] for a in annual]
    ym_values: list[list] = []
    for y in annual_years:
        row = []
        for m in range(1, 13):
            sel = (years == y) & (months == m)
            v = grp_mean_speed(sel)
            row.append(round(v, 4) if v is not None else None)
        ym_values.append(row)

    total_cnt = float(cnt.sum())
    mean_valid_fraction = float(total_cnt / (store.n_frames * GRID_H * GRID_W))
    mean_speed = float(ssum.sum() / total_cnt) if total_cnt > 0 else None

    dt = {
        "first_date": store.date_of(0),
        "latest_date": store.date_of(store.n_frames - 1),
        "n_frames": int(store.n_frames),
        "built_at": store.created_utc,
        "batch_age_days": int((np.datetime64("today", "D")
                               - store.dates[-1]) / np.timedelta64(1, "D")),
        "source": ("NSIDC-0116 v4 Polar Pathfinder Daily 25km EASE-Grid Sea Ice Motion "
                   "Vectors (Weddell subset)"),
        "doi": "10.5067/INAWUWO7QH7B",
    }
    obj = {
        "dataset": {
            "name": "NSIDC-0116 v4 威德尔海海冰运动数据集",
            "region": "60°W–20°E, 60°S–83°S",
            "grid": {"rows": GRID_H, "cols": GRID_W, "cell_km": 25.07},
            "n_frames": int(store.n_frames),
            "first_date": store.date_of(0),
            "latest_date": store.date_of(store.n_frames - 1),
            "years": int(annual_years[-1] - annual_years[0] + 1),
            "missing_days": int(store.attrs.get("missing_days_n", 0) or 0),
            "mean_valid_fraction": round(mean_valid_fraction, 4),
            "mean_speed_cm_s": round(mean_speed, 4) if mean_speed is not None else None,
            "u_units": "cm/s",
            "source": "NSIDC-0116 v4",
        },
        "kpis": [
            {"id": "years", "label": "覆盖年限",
             "value": int(annual_years[-1] - annual_years[0] + 1), "unit": "年"},
            {"id": "days", "label": "有效日数", "value": int(store.n_frames), "unit": "天"},
            {"id": "mean_speed", "label": "平均漂移速度",
             "value": round(mean_speed, 2) if mean_speed is not None else None, "unit": "cm/s"},
            {"id": "coverage", "label": "平均有效覆盖",
             "value": round(mean_valid_fraction * 100.0, 1), "unit": "%"},
        ],
        "annual": annual,
        "monthly": monthly,
        "year_month": {"years": annual_years, "months": list(range(1, 13)), "values": ym_values},
        "data_time": dt,
        "generated_at": utc_now_iso(),
    }
    path = resolve_path(f"{ARTIFACT_DIR}/timeseries.json",
                        default=f"{ARTIFACT_DIR}/timeseries.json")
    atomic_write_json(path, obj)
    log(f"已写 {path}（annual={len(annual)} monthly={len(monthly)} "
        f"year_month={len(ym_values)}×12, mean_speed={mean_speed:.3f}, "
        f"coverage={mean_valid_fraction:.4f}）")
    return path


def write_climatology(store: DataStore, agg: dict) -> Path:
    """train 期逐月平均 u,v（12,116,158）+ 逐月 speed mean/std。"""
    mon = agg["monthly_count"]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(mon > 0, agg["monthly_u_sum"] / mon, 0.0)
        v = np.where(mon > 0, agg["monthly_v_sum"] / mon, 0.0)
        sm = np.where(mon > 0, agg["monthly_speed_sum"] / mon, 0.0)
        var = np.where(mon > 0, agg["monthly_speed_sq_sum"] / mon - sm * sm, 0.0)
    speed_std = np.sqrt(np.maximum(var, 0.0))
    path = resolve_path(f"{ARTIFACT_DIR}/climatology.npz",
                        default=f"{ARTIFACT_DIR}/climatology.npz")
    atomic_savez(path,
                 u=u.astype(np.float32), v=v.astype(np.float32),
                 speed_mean=sm.astype(np.float32), speed_std=speed_std.astype(np.float32),
                 valid_count=mon.astype(np.float32),
                 months=np.arange(1, 13, dtype=np.int16),
                 train_years=np.array(TRAIN_YEARS, np.int16),
                 created_at=np.array(utc_now_iso()))
    log(f"已写 {path}（12×{GRID_H}×{GRID_W}; speed_mean mean={sm.mean():.2f}, "
        f"speed_std mean={speed_std.mean():.2f}）")
    return path


def write_cell_stats(store: DataStore, agg: dict) -> Path:
    """train 期逐像元 speed mean/std/p99/valid_days_count。"""
    cnt = agg["cell_count"].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(cnt > 0, agg["cell_speed_sum"] / cnt, np.nan)
        var = np.where(cnt > 0, agg["cell_speed_sq_sum"] / cnt - mean * mean, np.nan)
    std = np.sqrt(np.maximum(var, 0.0))
    # p99：直方图累计（0.2 cm/s 分辨率），再插值到 bin 右边界
    hist = agg["cell_speed_hist"].astype(np.int64)
    cum = np.cumsum(hist, axis=1)
    p99_idx = np.argmax(cum >= 0.99 * np.maximum(cnt, 1)[:, None], axis=1)
    p99 = (p99_idx + 1).astype(np.float64) * P99_WIDTH
    p99[cnt <= 0] = np.nan
    path = resolve_path(f"{ARTIFACT_DIR}/cell_stats.npz",
                        default=f"{ARTIFACT_DIR}/cell_stats.npz")
    # schema 固定 2D (116,158)：逐像元物理量，下游不要再各自 reshape
    atomic_savez(path,
                 speed_mean=np.nan_to_num(mean, nan=0.0).astype(np.float32).reshape(GRID_H, GRID_W),
                 speed_std=np.nan_to_num(std, nan=0.0).astype(np.float32).reshape(GRID_H, GRID_W),
                 speed_p99=np.nan_to_num(p99, nan=0.0).astype(np.float32).reshape(GRID_H, GRID_W),
                 valid_days_count=cnt.astype(np.int32).reshape(GRID_H, GRID_W),
                 grid_shape=np.array([GRID_H, GRID_W], np.int16),
                 train_years=np.array(TRAIN_YEARS, np.int16),
                 created_at=np.array(utc_now_iso()))
    log(f"已写 {path}（mean={np.nanmean(mean):.2f} std={np.nanmean(std):.2f} "
        f"p99 mean={np.nanmean(p99):.2f}）")
    return path


# ---------------------------------------------------------------- 统一协议评估
def _load_eval_windows(store: DataStore) -> dict:
    path = resolve_path(EVAL_WINDOWS_PATH, default=EVAL_WINDOWS_PATH)
    obj = load_json(path)
    windows = obj["windows"]
    if len(windows) != obj.get("n_windows", len(windows)):
        raise ValueError("eval_windows.json 窗口数与 n_windows 不一致")
    return obj


def _eval_predictions(store: DataStore, windows: list[dict]) -> dict:
    """对统一协议的每个窗口、全部目标帧有效像元计算 4 个模型的预测。

    返回 {obs, pers, ridge, rf, clim, rows, cols, window_id, input_valid, obs_speed}，
    以及窗口级元信息。
    """
    import joblib

    model_dir = resolve_path(MODEL_DIR, default=MODEL_DIR)
    artifact_dir = resolve_path(ARTIFACT_DIR, default=ARTIFACT_DIR)

    def load(model_id: str):
        path = model_dir / f"{'rf' if model_id == 'random_forest' else model_id}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"缺少模型 artifact: {path}（先运行 train_tabular --stage final）")
        payload = joblib.load(path)
        return payload["pipeline"] if isinstance(payload, dict) and "pipeline" in payload else payload

    ridge = load("ridge")
    rf = load("random_forest")
    clim_path = artifact_dir / "climatology.npz"
    if not clim_path.exists():
        raise FileNotFoundError(f"缺少气候态 artifact: {clim_path}（先运行本脚本数据部分）")
    cd = np.load(clim_path)
    clim_u = np.asarray(cd["u"], np.float32)   # (12,H,W)
    clim_v = np.asarray(cd["v"], np.float32)

    n_total = 0
    obs_l, pers_l, ridge_l, rf_l, clim_l = [], [], [], [], []
    row_l, col_l, wid_l, in_valid_l = [], [], [], []
    t_start = time.time()
    for wi, w in enumerate(windows):
        last_idx = int(w["last_obs_idx"])
        truth_idx = int(w["truth_idx"])
        truth_mask = store.valid_mask(truth_idx) > 0
        rows, cols = np.nonzero(truth_mask)
        win = store.window(last_idx, N_HISTORY)
        X = build_features(win, rows, cols)
        f_truth = store.frame(truth_idx)
        obs = np.stack([np.asarray(f_truth["u"][rows, cols], np.float32),
                        np.asarray(f_truth["v"][rows, cols], np.float32)], axis=1)
        f_last = store.frame(last_idx)
        pers = np.stack([np.asarray(f_last["u"][rows, cols], np.float32),
                         np.asarray(f_last["v"][rows, cols], np.float32)], axis=1)
        month_k = int(store.month_of(truth_idx)) - 1
        clim = np.stack([clim_u[month_k][rows, cols], clim_v[month_k][rows, cols]], axis=1)
        ridge_p = np.asarray(ridge.predict(X), np.float32).reshape(-1, 2)
        rf_p = np.asarray(rf.predict(X), np.float32).reshape(-1, 2)
        obs_l.append(obs)
        pers_l.append(pers)
        ridge_l.append(ridge_p)
        rf_l.append(rf_p)
        clim_l.append(clim)
        row_l.append(rows.astype(np.int16))
        col_l.append(cols.astype(np.int16))
        wid_l.append(np.full(rows.size, wi, np.int16))
        in_valid_l.append((f_last["mask"][rows, cols] > 0).astype(np.int8))
        n_total += rows.size
        if wi % 20 == 0 or wi == len(windows) - 1:
            log(f"  eval {wi + 1:>3}/{len(windows)} 窗口  n={n_total:,}  "
                f"elapsed={time.time() - t_start:5.1f}s  maxrss={_fmt_bytes(_maxrss())}")

    out = {
        "obs": np.concatenate(obs_l),
        "persistence": np.concatenate(pers_l),
        "ridge": np.concatenate(ridge_l),
        "random_forest": np.concatenate(rf_l),
        "climatology": np.concatenate(clim_l),
        "rows": np.concatenate(row_l),
        "cols": np.concatenate(col_l),
        "window_idx": np.concatenate(wid_l),
        "input_valid": np.concatenate(in_valid_l),
        "n_samples": n_total,
        "elapsed_s": round(time.time() - t_start, 1),
    }
    log(f"评估预测完成：{n_total:,} 样本 / {len(windows)} 窗口，用时 {out['elapsed_s']}s，"
        f"maxrss={_fmt_bytes(_maxrss())}")
    return out


def write_eval_parquet(store: DataStore, windows: list[dict], pred: dict) -> Path:
    import pandas as pd

    win_idx = pred["window_idx"]
    last_dates = np.array([windows[int(i)]["last_obs_date"] for i in np.unique(win_idx)])
    truth_dates = np.array([windows[int(i)]["truth_date"] for i in np.unique(win_idx)])
    uniq = np.unique(win_idx)
    last_map = dict(zip(uniq.tolist(), last_dates.tolist()))
    truth_map = dict(zip(uniq.tolist(), truth_dates.tolist()))
    df = pd.DataFrame({
        "window": win_idx.astype(np.int16),
        "last_obs_date": pd.Categorical([last_map[int(i)] for i in win_idx]),
        "truth_date": pd.Categorical([truth_map[int(i)] for i in win_idx]),
        "row": pred["rows"].astype(np.int16),
        "col": pred["cols"].astype(np.int16),
        "input_valid": pred["input_valid"].astype(np.int8),
        "obs_u": pred["obs"][:, 0].astype(np.float32),
        "obs_v": pred["obs"][:, 1].astype(np.float32),
        "persistence_u": pred["persistence"][:, 0].astype(np.float32),
        "persistence_v": pred["persistence"][:, 1].astype(np.float32),
        "ridge_u": pred["ridge"][:, 0].astype(np.float32),
        "ridge_v": pred["ridge"][:, 1].astype(np.float32),
        "random_forest_u": pred["random_forest"][:, 0].astype(np.float32),
        "random_forest_v": pred["random_forest"][:, 1].astype(np.float32),
        "climatology_u": pred["climatology"][:, 0].astype(np.float32),
        "climatology_v": pred["climatology"][:, 1].astype(np.float32),
    })
    path = resolve_path(f"{ARTIFACT_DIR}/eval_day1.parquet",
                        default=f"{ARTIFACT_DIR}/eval_day1.parquet")
    ensure_parent(path)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)
    log(f"已写 {path}（{len(df):,} 行 × {df.shape[1]} 列，"
        f"{_fmt_bytes(path.stat().st_size)}）")
    return path


def _benchmark_latency() -> dict:
    """用 PredictionEngine 实测单点 / 整场推理延迟（毫秒）。"""
    engine = PredictionEngine()
    point_lat, field_lat = {}, {}
    for m in MODEL_IDS:
        point_lat[m] = engine.benchmark_point(m, n=30, warmup=3)["median_ms"]
    for m in MODEL_IDS:
        field_lat[m] = engine.benchmark_field(m, n=3)["median_ms"]
    return {"point_ms": point_lat, "field_ms": field_lat,
            "detail": {m: engine.benchmark_point(m, n=10, warmup=1) for m in MODEL_IDS}}


def write_metrics(store: DataStore, windows_obj: dict, pred: dict) -> dict:
    """写 metrics_tabular.json（4 行，字段严格按 API 契约 /api/metrics）并更新 model_card。"""
    artifact_dir = resolve_path(ARTIFACT_DIR, default=ARTIFACT_DIR)
    model_dir = resolve_path(MODEL_DIR, default=MODEL_DIR)
    card_path = model_dir / "model_card.json"
    card = {}
    if card_path.exists():
        with open(card_path, "r", encoding="utf-8") as fh:
            card = json.load(fh)

    obs = pred["obs"].astype(np.float64)
    pers = pred["persistence"].astype(np.float64)
    lat = _benchmark_latency()
    log(f"延迟实测 point_ms={lat['point_ms']}, field_ms={lat['field_ms']}")

    train_cfg = card.get("training", {})
    ridge_cfg = train_cfg.get("ridge", {}) or {}
    rf_cfg = train_cfg.get("random_forest", {}) or {}
    training_samples = {
        "ridge": ridge_cfg.get("training_samples"),
        "random_forest": rf_cfg.get("training_samples"),
    }

    rows = []
    test_metrics = {}
    for mid in MODEL_IDS:
        p = pred[mid].astype(np.float64)
        m = vec_metrics(p, obs, pers=pers)
        test_metrics[mid] = m
        meta = MODEL_META[mid]
        if mid == "ridge":
            params = {"alpha": ridge_cfg.get("alpha"),
                      "pipeline": "StandardScaler + Ridge", "n_features": 34}
        elif mid == "random_forest":
            params = dict(rf_cfg.get("params") or {})   # 以 model_card 实际训练参数为准
        else:
            params = None
        rows.append({
            "id": mid,
            "name": meta["name"],
            "name_zh": meta["name_zh"],
            "category": meta["category"],
            "vec_rmse": round(m["vec_rmse"], 4),
            "rmse_u": round(m["rmse_u"], 4),
            "rmse_v": round(m["rmse_v"], 4),
            "mae_vec": round(m["mae_vec"], 4),
            "r2_u": round(m["r2_u"], 4),
            "r2_v": round(m["r2_v"], 4),
            "skill_vs_persistence": round(m.get("skill_vs_persistence", 0.0), 4),
            "latency_ms": lat["point_ms"][mid],
            "params": params,
            "training_samples": training_samples.get(mid),
            "is_highlight": meta["is_highlight"],
            "is_baseline": meta["is_baseline"],
        })
        log(f"  {mid:<14} vec_rmse={m['vec_rmse']:.4f} rmse_u={m['rmse_u']:.4f} "
            f"rmse_v={m['rmse_v']:.4f} r2_u={m['r2_u']:.4f} "
            f"skill={m.get('skill_vs_persistence', 0.0):+.4f}")

    protocol = {
        "task": "未来 24 小时（下一日帧）海冰漂移矢量预测",
        "target": "u(t+1), v(t+1)，cm/s，25 km EASE-Grid South 像元",
        "split": {"train": "1978–2017", "val": "2018–2021", "test": "2022–2025"},
        "sampling": ("test split 每 5 日一个起报窗口（287 个窗口），"
                     "评估每个窗口目标帧的全部有效像元"),
        "n_windows": int(windows_obj.get("n_windows", len(windows_obj["windows"]))),
        "n_samples": int(pred["n_samples"]),
        "metric_defs": metric_defs(),
        "eval_windows_file": "data/processed/dashboard/eval_windows.json",
        "window_definition": "target = eval_windows.json 的 truth_idx；输入历史 = last_obs_idx 及其前 7 帧",
        "note": "test 2022–2025 仅用于本表评估；标准化/调参未接触 test 年份",
    }
    obj = {
        "protocol": protocol,
        "models": rows,
        "generated_at": utc_now_iso(),
        "training": {
            "seed": SEED,
            "n_train_samples": ridge_cfg.get("training_samples"),
            "val_rows": train_cfg.get("val_rows"),
            "ridge_val": (train_cfg.get("val_metrics", {}) or {}).get("ridge"),
            "rf_val": (train_cfg.get("val_metrics", {}) or {}).get("random_forest"),
            "latency_point_ms": lat["point_ms"],
            "latency_field_ms": lat["field_ms"],
            "eval_elapsed_s": pred["elapsed_s"],
        },
    }
    path = artifact_dir / "metrics_tabular.json"
    atomic_write_json(path, obj)
    log(f"已写 {path}")

    # ---- 更新 model_card.json（补 test 指标与延迟；不改变训练信息）
    card.setdefault("evaluation", {})
    card["evaluation"].update({
        "generated_at": utc_now_iso(),
        "protocol": {k: v for k, v in protocol.items() if k != "metric_defs"},
        "test_metrics": {mid: {k: round(float(v), 6) if isinstance(v, (int, float)) else v
                               for k, v in m.items()} for mid, m in test_metrics.items()},
        "latency_point_ms": lat["point_ms"],
        "latency_field_ms": lat["field_ms"],
        "model_card_updated_at": utc_now_iso(),
    })
    card["test_metrics"] = {mid: {k: round(float(v), 6) for k, v in m.items()}
                            for mid, m in test_metrics.items()}
    card["metric_conventions"] = {
        "pooled": ("本项目统一口径：vec_rmse = sqrt(Σ(Δu²+Δv²)/N)，N=1,556,493，"
                   "按全部目标帧有效像元一次性池化；metrics_tabular.json 全部采用该口径"),
        "batch_equal_weight_legacy": ("旧项目 reports/final_report.md 的 evaluate() 采用 batch=8 "
                                      "等权平均口径（每个 batch 权重相同），与 pooled 不可直接比较；"
                                      "本包未使用该口径"),
        "cross_check": {
            "persistence_pooled_vec_rmse": round(test_metrics["persistence"]["vec_rmse"], 4),
            "persistence_legacy_batch_equal_weight": 15.2875,
            "note": ("legacy 15.2875 为旧 5 日多步 batch 等权口径（lead 参考值）；"
                     "本表 persistence 为 day-1 pooled 口径，逐像元样本 ∈ 目标帧有效像元"),
        },
        "definition_note": "persistence 预测 = 起报帧（last_obs）u,v；climatology = 目标帧月份 train 期月均场",
    }
    atomic_write_json(card_path, card)
    log(f"已更新 {card_path}")
    return obj


# ---------------------------------------------------------------- main
def build_data_artifacts(store: DataStore, *, force_scan: bool = False) -> dict:
    agg = scan_dataset(store, force=force_scan)
    paths = {
        "timeseries": write_timeseries(store, agg),
        "climatology": write_climatology(store, agg),
        "cell_stats": write_cell_stats(store, agg),
    }
    return {"agg": agg, "paths": paths}


def evaluate(store: DataStore) -> dict:
    windows_obj = _load_eval_windows(store)
    windows = windows_obj["windows"]
    log(f"统一评估协议：{len(windows)} 窗口，{windows_obj.get('n_valid_samples')} 有效样本（声明）")
    pred = _eval_predictions(store, windows)
    if pred["n_samples"] != int(windows_obj.get("n_valid_samples", pred["n_samples"])):
        raise RuntimeError(f"评估样本数不一致: {pred['n_samples']} != {windows_obj.get('n_valid_samples')}")
    write_eval_parquet(store, windows, pred)
    obj = write_metrics(store, windows_obj, pred)
    return {"metrics": obj, "pred": pred}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="构建 dashboard 数据 artifact 与统一协议评估")
    ap.add_argument("--zarr", default=None)
    ap.add_argument("--skip-data", action="store_true", help="跳过 timeseries/climatology/cell_stats")
    ap.add_argument("--skip-eval", action="store_true", help="跳过统一协议评估")
    ap.add_argument("--force-scan", action="store_true", help="强制重扫 zarr")
    args = ap.parse_args(argv)

    log("===== predict.build_artifacts 开始 =====")
    t_all = time.time()
    store = DataStore(args.zarr)
    log(f"DataStore: {store}")
    if not args.skip_data:
        build_data_artifacts(store, force_scan=args.force_scan)
    if not args.skip_eval:
        evaluate(store)
    log(f"===== 完成，总耗时 {time.time() - t_all:.1f}s，maxrss={_fmt_bytes(_maxrss())} =====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
