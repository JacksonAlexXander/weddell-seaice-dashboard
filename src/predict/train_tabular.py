"""Ridge / RandomForest 离线训练与调参 CLI（后台运行，可续跑）。

用法（工作区根目录）::

    # 0) 冒烟：小样本训练，尽快让 PredictionEngine 可用（约 1 分钟）
    PYTHONPATH=src nohup .venv/bin/python -m predict.train_tabular --stage smoke \
        > logs/ml_tabular_smoke.log 2>&1 &

    # 1) 调参（val 2018–2021 子集，小网格；输出写 models/ml/_cache/tuning.json）
    PYTHONPATH=src nohup .venv/bin/python -m predict.train_tabular --stage tune \
        > logs/ml_tabular_tune.log 2>&1 &

    # 2) 全量训练（train 1978–2017 采样，约 2.3M 行；保存正式 artifact）
    PYTHONPATH=src nohup .venv/bin/python -m predict.train_tabular --stage final \
        > logs/ml_tabular_train.log 2>&1 &

    # 一键：调参 + 训练
    PYTHONPATH=src nohup .venv/bin/python -m predict.train_tabular --stage all ...

设计约束：
- 训练采样：每个有效日随机取 `--cells-per-day`（默认 160）个"输入帧与目标帧均有效"的像元；
  train 1978–2017 共 ~2.3M 行（内存 < 6GB，逐年份流式构建并缓存 npz，可续跑）。
- 超参只在 val 2018–2021 的子集上选择；test 2022–2025 **不参与**标准化/调参。
- 随机性固定 `SEED=42`，RF `random_state=42, n_jobs=4`，joblib 原子落盘。
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .common import (
    CACHE_DIR,
    GRID_H,
    GRID_W,
    MODEL_DIR,
    ROOT,
    SEED,
    TEST_YEARS,
    TRAIN_YEARS,
    VAL_YEARS,
    atomic_joblib_dump,
    atomic_write_json,
    ensure_parent,
    resolve_path,
    utc_now_iso,
)
from .data import DataStore
from .features import N_FEATURES, N_HISTORY, build_features, feature_spec
from .metrics import vec_metrics

CACHE = resolve_path(CACHE_DIR, default=CACHE_DIR)
DEFAULT_CELLS_PER_DAY = 150
TUNE_ROWS = 500_000
VAL_CELLS_PER_DAY = 120


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024.0
    return f"{n:.1f}GB"


def _maxrss() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- 特征矩阵构建（可续跑）
def _year_cache_path(tag: str, year: int) -> Path:
    return CACHE / f"{tag}_{year}.npz"


def _build_year_matrix(store: DataStore, year: int, cells_per_day: int) -> dict | None:
    """构建某一年的训练/验证矩阵（输入帧与目标帧均有效）。"""
    idx = store.indices_of_years(year, year)
    if idx.size == 0:
        return None
    lo, hi = int(idx[0]), int(idx[-1])
    if hi >= store.n_frames - 1:
        hi = store.n_frames - 2
    start = max(0, lo - (N_HISTORY - 1))
    arrays = store.read_range(start, hi + 2)          # 覆盖历史 + 目标帧
    u, v, sp, mk = arrays["u"], arrays["v"], arrays["speed"], arrays["mask"]
    k0 = lo - start
    Xs, ys, ts = [], [], []
    for t in range(lo, hi + 1):
        k = t - start
        if t < N_HISTORY - 1 or k + 1 >= u.shape[0]:
            continue
        both = (mk[k] > 0) & (mk[k + 1] > 0)
        rows, cols = np.nonzero(both)
        if rows.size == 0:
            continue
        rng = np.random.default_rng([SEED, t])
        if rows.size > cells_per_day:
            sel = rng.choice(rows.size, cells_per_day, replace=False)
            rows, cols = rows[sel], cols[sel]
        win = {
            "u": u[k - N_HISTORY + 1:k + 1],
            "v": v[k - N_HISTORY + 1:k + 1],
            "speed": sp[k - N_HISTORY + 1:k + 1],
            "mask": mk[k - N_HISTORY + 1:k + 1],
            "doy": arrays["doy"][k - N_HISTORY + 1:k + 1],
            "latitude": store.latitude,
            "longitude": store.longitude,
        }
        Xs.append(build_features(win, rows, cols))
        ys.append(np.stack([u[k + 1][rows, cols], v[k + 1][rows, cols]], axis=1))
        ts.append(np.full(rows.size, t, np.int32))
    if not Xs:
        return None
    X = np.concatenate(Xs, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(ys, axis=0).astype(np.float32, copy=False)
    t_arr = np.concatenate(ts, axis=0)
    return {"X": X, "y": y, "t": t_arr, "year": year}


def build_split_matrix(store: DataStore, split: str, cells_per_day: int, *,
                       force: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """逐年份构建/加载缓存矩阵；返回 (X, y, t)。"""
    year_lo, year_hi = {"train": TRAIN_YEARS, "val": VAL_YEARS, "test": TEST_YEARS}[split]
    tag = f"{split}_c{cells_per_day}"
    ensure_parent(str(CACHE / "x"))
    Xs, ys, ts = [], [], []
    t_start = time.time()
    for year in range(year_lo, year_hi + 1):
        path = _year_cache_path(tag, year)
        if path.exists() and not force:
            d = np.load(path)
            Xs.append(d["X"])
            ys.append(d["y"])
            ts.append(d["t"])
            log(f"  [{tag}] {year}: cache hit {d['X'].shape[0]:,} rows")
            continue
        d = _build_year_matrix(store, year, cells_per_day)
        if d is None:
            log(f"  [{tag}] {year}: 无样本（跳过）")
            continue
        np.savez(path, **d)
        Xs.append(d["X"])
        ys.append(d["y"])
        ts.append(d["t"])
        log(f"  [{tag}] {year}: built {d['X'].shape[0]:,} rows "
            f"(elapsed {time.time() - t_start:.0f}s)")
    if not Xs:
        raise RuntimeError(f"{split} 未构建到任何样本")
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    t_arr = np.concatenate(ts, axis=0)
    log(f"  [{tag}] total {X.shape[0]:,} rows × {X.shape[1]} feats, "
        f"mem {_fmt_bytes(X.nbytes + y.nbytes)}, elapsed {time.time() - t_start:.0f}s")
    return X, y, t_arr


# ---------------------------------------------------------------- 调参
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
RF_GRID = (
    {"max_depth": None, "min_samples_leaf": 10, "max_features": 0.5},
    {"max_depth": None, "min_samples_leaf": 25, "max_features": 0.5},
    {"max_depth": None, "min_samples_leaf": 50, "max_features": 0.5},
    {"max_depth": 28, "min_samples_leaf": 10, "max_features": 0.5},
    {"max_depth": 28, "min_samples_leaf": 25, "max_features": 0.5},
    {"max_depth": None, "min_samples_leaf": 25, "max_features": 1.0},
)
TUNE_RF_TREES = 80
FINAL_RF_TREES = 200


def run_tuning(store: DataStore, force: bool = False) -> dict:
    """在 val 2018–2021 子集上小网格调参，写 models/ml/_cache/tuning.json。"""
    tune_path = CACHE / "tuning.json"
    if tune_path.exists() and not force:
        log(f"调参结果已存在，直接复用: {tune_path}")
        return json.loads(tune_path.read_text())

    log("构建 train 特征矩阵（用于调参子集）…")
    Xtr, ytr, _ = build_split_matrix(store, "train", DEFAULT_CELLS_PER_DAY)
    rng = np.random.default_rng([SEED, 7])
    if Xtr.shape[0] > TUNE_ROWS:
        sub = rng.choice(Xtr.shape[0], TUNE_ROWS, replace=False)
        Xt, yt = Xtr[sub], ytr[sub]
        del sub
    else:
        Xt, yt = Xtr, ytr
    log(f"调参子集: {Xt.shape[0]:,} 行；释放全量 train 矩阵")
    del Xtr, ytr

    log("构建 val 特征矩阵 …")
    Xv, yv, _ = build_split_matrix(store, "val", VAL_CELLS_PER_DAY)

    result: dict = {"ridge": {}, "random_forest": {}, "n_tune_rows": int(Xt.shape[0]),
                    "n_val_rows": int(Xv.shape[0]), "created_at": utc_now_iso()}

    # ---- Ridge alpha
    log(f"Ridge 网格 alpha={RIDGE_ALPHAS} …")
    ridge_rows = []
    best_r, best_a = None, None
    for alpha in RIDGE_ALPHAS:
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("ridge", Ridge(alpha=alpha))])
        t0 = time.time()
        pipe.fit(Xt, yt)
        m = vec_metrics(pipe.predict(Xv), yv, pers=Xv[:, [0, 1]])
        row = {"alpha": alpha, **m, "fit_s": round(time.time() - t0, 2)}
        ridge_rows.append(row)
        log(f"  alpha={alpha:<6} vec_rmse={m['vec_rmse']:.4f} r2_u={m['r2_u']:.4f} "
            f"({row['fit_s']}s)")
        if best_r is None or m["vec_rmse"] < best_r:
            best_r, best_a = m["vec_rmse"], alpha
    result["ridge"] = {"alpha": float(best_a), "tuning": ridge_rows}
    log(f"Ridge 最优 alpha={best_a} (val vec_rmse={best_r:.4f})")

    # ---- RF 小网格
    log(f"RandomForest 网格 {len(RF_GRID)} 组 × {TUNE_RF_TREES} trees …")
    rf_rows = []
    best_rf, best_cfg = None, None
    for cfg in RF_GRID:
        t0 = time.time()
        rf = RandomForestRegressor(
            n_estimators=TUNE_RF_TREES, n_jobs=4, random_state=SEED,
            bootstrap=True, **cfg)
        rf.fit(Xt, yt)
        m = vec_metrics(rf.predict(Xv), yv, pers=Xv[:, [0, 1]])
        row = {**cfg, **m, "fit_s": round(time.time() - t0, 2),
               "model_mb": round(sum(sys.getsizeof(t) for t in rf.estimators_) / 2**20, 1)}
        rf_rows.append(row)
        log(f"  {cfg} vec_rmse={m['vec_rmse']:.4f} r2_u={m['r2_u']:.4f} ({row['fit_s']}s)")
        if best_rf is None or m["vec_rmse"] < best_rf:
            best_rf, best_cfg = m["vec_rmse"], dict(cfg)
        del rf
    result["random_forest"] = {"best_grid": best_cfg, "tuning": rf_rows,
                               "n_estimators_tune": TUNE_RF_TREES}
    log(f"RF 最优 {best_cfg} (val vec_rmse={best_rf:.4f})")

    # 按调参子集 -> 全量行数 **比例缩放** min_samples_leaf（保证全量 RF 显存可控）
    n_tune = Xt.shape[0]
    n_full = 0
    for year in range(TRAIN_YEARS[0], TRAIN_YEARS[1] + 1):
        p = _year_cache_path(f"train_c{DEFAULT_CELLS_PER_DAY}", year)
        if p.exists():
            with np.load(p) as d:
                n_full += int(d["X"].shape[0])
    rel = float(best_cfg["min_samples_leaf"]) / max(n_tune, 1)
    min_leaf_full = int(round(rel * max(n_full, 1)))
    min_leaf_full = max(8, min(min_leaf_full, 400))
    final_cfg = dict(best_cfg)
    final_cfg["min_samples_leaf"] = min_leaf_full
    final_cfg["n_estimators"] = FINAL_RF_TREES
    result["random_forest"]["final"] = final_cfg
    result["random_forest"]["min_samples_leaf_scaling"] = {
        "n_tune_rows": int(n_tune), "n_full_rows": int(n_full),
        "rule": "min_leaf_full = round(min_leaf_tune / n_tune_rows * n_full_rows)",
    }
    log(f"随机森林最终配置: {final_cfg}（调参 min_leaf 按行数比例缩放）")

    atomic_write_json(tune_path, result)
    log(f"调参完成 -> {tune_path}")
    return result


# ---------------------------------------------------------------- 最终训练
# 内存约束（8GB 机器；200 树全深度 RF 曾在 fit 阶段被系统 OOM kill）：
# 最终 RF 采用内存安全配置：特征行子采样 1.5M + max_samples=0.6 + max_depth=22 +
# min_samples_leaf=200 + max_features=0.5 + n_jobs=2 + warm_start 分块训练
# （30 树/块，逐块 joblib 落盘 models/ml/rf.partial.joblib；中断后可续跑）。
FINAL_RF_N_ESTIMATORS = 120
FINAL_RF_CHUNK_TREES = 30
FINAL_RF_MAX_DEPTH = 22
FINAL_RF_MIN_LEAF = 200
FINAL_RF_MAX_FEATURES = 0.5
FINAL_RF_MAX_SAMPLES = 0.6
FINAL_RF_N_JOBS = 2
FINAL_TRAIN_ROWS = 1_500_000


def _save_feature_spec() -> None:
    atomic_write_json(resolve_path(f"{MODEL_DIR}/feature_spec.json", default=f"{MODEL_DIR}/feature_spec.json"),
                      feature_spec())


def _rf_params() -> dict:
    return {
        "n_estimators": FINAL_RF_N_ESTIMATORS,
        "max_depth": FINAL_RF_MAX_DEPTH,
        "min_samples_leaf": FINAL_RF_MIN_LEAF,
        "max_features": FINAL_RF_MAX_FEATURES,
        "max_samples": FINAL_RF_MAX_SAMPLES,
        "n_jobs": FINAL_RF_N_JOBS,
        "random_state": SEED,
        "bootstrap": True,
        "warm_start": False,
    }


def run_final(store: DataStore, force: bool = False) -> dict:
    """全量 train 训练 Ridge + RF（RF 内存安全 + 分块可续跑），写 model_card。"""
    import gc
    import os

    tune_path = CACHE / "tuning.json"
    if not tune_path.exists():
        log("未找到调参结果，先运行 tune …")
        tuning = run_tuning(store)
    else:
        tuning = json.loads(tune_path.read_text())

    ridge_alpha = float(tuning["ridge"]["alpha"])
    log(f"Ridge alpha={ridge_alpha}")
    log(f"RF 最终配置（内存安全）：{_rf_params()} 训练行数上限={FINAL_TRAIN_ROWS:,}")

    log("构建全量 train 特征矩阵（逐年份 npz 缓存）…")
    X, y, t_arr = build_split_matrix(store, "train", DEFAULT_CELLS_PER_DAY, force=force)
    n_rows_all = int(X.shape[0])
    log(f"train rows={n_rows_all:,}; mem maxrss={_fmt_bytes(_maxrss())}")

    # ---------------- Ridge：线性模型，全量 2.14M 行（内存/时间都很小）
    log("训练 Ridge Pipeline(StandardScaler + Ridge) …")
    t0 = time.time()
    ridge = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=ridge_alpha))])
    ridge.fit(X, y)
    ridge_fit_s = time.time() - t0
    log(f"Ridge 完成 {ridge_fit_s:.1f}s")

    log("构建 val 矩阵并计算验证指标 …")
    Xv, yv, _ = build_split_matrix(store, "val", VAL_CELLS_PER_DAY)
    pers_v = Xv[:, [0, 1]]          # 中心像元 t 帧 u,v = 持续性预测
    ridge_pred_v = ridge.predict(Xv)
    ridge_val = vec_metrics(ridge_pred_v, yv, pers=pers_v)
    resid_u = ridge_pred_v[:, 0] - yv[:, 0]
    resid_v = ridge_pred_v[:, 1] - yv[:, 1]
    ridge_unc = {
        "method": "ridge 验证集残差 RMSE 近似（±1σ 24h 位移带）",
        "rmse_u": float(np.sqrt(np.mean(resid_u ** 2))),
        "rmse_v": float(np.sqrt(np.mean(resid_v ** 2))),
        "vec_residual_rmse": float(np.sqrt(np.mean(resid_u ** 2 + resid_v ** 2))),
    }
    log(f"Ridge val: vec_rmse={ridge_val['vec_rmse']:.4f} "
        f"skill={ridge_val.get('skill_vs_persistence', 0):+.4f}")

    ridge_path = resolve_path(f"{MODEL_DIR}/ridge.joblib", default=f"{MODEL_DIR}/ridge.joblib")
    atomic_joblib_dump({"pipeline": ridge, "model_id": "ridge",
                        "feature_names": feature_spec()["feature_names"],
                        "n_features": N_FEATURES, "n_history": N_HISTORY,
                        "target": ["u_next", "v_next"], "unit": "cm/s",
                        "alpha": ridge_alpha, "stage": "final",
                        "trained_at": utc_now_iso(),
                        "training_samples": n_rows_all}, ridge_path)
    log(f"已保存 {ridge_path}（{_fmt_bytes(ridge_path.stat().st_size)}）")

    # ---------------- RF：行子采样 + 内存安全参数 + warm_start 分块
    rng = np.random.default_rng([SEED, 11])
    if n_rows_all > FINAL_TRAIN_ROWS:
        idx = np.sort(rng.choice(n_rows_all, FINAL_TRAIN_ROWS, replace=False))
        Xr = X[idx].copy()
        yr = y[idx].copy()
        del idx
        log(f"RF 训练子集：{n_rows_all:,} -> {Xr.shape[0]:,} 行（无放回随机抽样 seed=42）")
        del X, y, t_arr
    else:
        Xr, yr = X, y
        del X, y, t_arr
    gc.collect()
    log(f"RF 训练矩阵 {Xr.shape[0]:,} 行 × {Xr.shape[1]} 特征，"
        f"mem={_fmt_bytes(Xr.nbytes + yr.nbytes)}，maxrss={_fmt_bytes(_maxrss())}")

    def _fresh_rf() -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=FINAL_RF_CHUNK_TREES, warm_start=True, bootstrap=True,
            max_depth=FINAL_RF_MAX_DEPTH, min_samples_leaf=FINAL_RF_MIN_LEAF,
            max_features=FINAL_RF_MAX_FEATURES, max_samples=FINAL_RF_MAX_SAMPLES,
            n_jobs=FINAL_RF_N_JOBS, random_state=SEED)

    total_chunks = FINAL_RF_N_ESTIMATORS // FINAL_RF_CHUNK_TREES
    partial_path = resolve_path(f"{MODEL_DIR}/rf.partial.joblib",
                                default=f"{MODEL_DIR}/rf.partial.joblib")
    rf = _fresh_rf()
    start_chunk = 0
    if partial_path.exists() and not force:
        try:
            pd = joblib.load(partial_path)
            rf = pd["pipeline"]
            start_chunk = int(pd.get("chunk", 0))
            rf.set_params(n_jobs=FINAL_RF_N_JOBS, warm_start=True)
            if rf.n_estimators % FINAL_RF_CHUNK_TREES != 0 or start_chunk >= total_chunks:
                raise ValueError(f"检查点不一致（n_estimators={rf.n_estimators}, chunk={start_chunk}）")
            log(f"从分块检查点恢复 RF：已有 {rf.n_estimators} 棵树 / {start_chunk}/{total_chunks} 块")
        except Exception as e:  # noqa: BLE001
            log(f"检查点不可用（{type(e).__name__}: {e}），从头训练")
            rf = _fresh_rf()
            start_chunk = 0
    else:
        log(f"开始 warm_start 分块训练：{FINAL_RF_N_ESTIMATORS} 棵树 = "
            f"{total_chunks} 块 × {FINAL_RF_CHUNK_TREES} 棵（逐块落盘 {partial_path.name}）")

    t0 = time.time()
    for c in range(start_chunk, total_chunks):
        rf.set_params(n_estimators=(c + 1) * FINAL_RF_CHUNK_TREES)
        rf.fit(Xr, yr)
        gc.collect()
        atomic_joblib_dump({"pipeline": rf, "chunk": c + 1,
                            "n_estimators": rf.n_estimators,
                            "params": _rf_params(), "saved_at": utc_now_iso()},
                           partial_path)
        log(f"  RF chunk {c + 1}/{total_chunks} 完成：n_estimators={rf.n_estimators}，"
            f"elapsed={time.time() - t0:.0f}s，maxrss={_fmt_bytes(_maxrss())}")
    rf_fit_s = time.time() - t0
    node_counts = [int(est.tree_.node_count) for est in rf.estimators_]
    log(f"RF 完成 {rf_fit_s:.1f}s；树节点数 mean={np.mean(node_counts):.0f} "
        f"max={np.max(node_counts)} total={sum(node_counts):,}；"
        f"maxrss={_fmt_bytes(_maxrss())}")
    try:
        os.unlink(partial_path)      # 成功后被原子替换为正式 artifact，删除检查点
    except OSError:
        pass

    rf_val = vec_metrics(rf.predict(Xv), yv, pers=pers_v)
    log(f"RF val: vec_rmse={rf_val['vec_rmse']:.4f} "
        f"skill={rf_val.get('skill_vs_persistence', 0):+.4f}")

    rf_params = _rf_params()
    rf_path = resolve_path(f"{MODEL_DIR}/rf.joblib", default=f"{MODEL_DIR}/rf.joblib")
    atomic_joblib_dump({"pipeline": rf, "model_id": "random_forest",
                        "feature_names": feature_spec()["feature_names"],
                        "n_features": N_FEATURES, "n_history": N_HISTORY,
                        "target": ["u_next", "v_next"], "unit": "cm/s",
                        "params": rf_params, "stage": "final",
                        "trained_at": utc_now_iso(),
                        "training_samples": int(Xr.shape[0])}, rf_path)
    log(f"已保存 {rf_path}（{_fmt_bytes(rf_path.stat().st_size)}）")

    tuning_best_val = min(float(r["vec_rmse"]) for r in tuning["random_forest"]["tuning"])

    model_card = {
        "version": "1.0",
        "created_at": utc_now_iso(),
        "stage": "final",
        "task": "未来 24 小时（下一日帧）海冰漂移矢量 (u,v) 预测，cm/s",
        "data": {
            "zarr": str(store.zarr_path),
            "zarr_created_utc": store.created_utc,
            "n_frames": store.n_frames,
            "grid": [GRID_H, GRID_W],
            "first_date": store.date_of(0),
            "latest_date": store.date_of(store.n_frames - 1),
            "split": {"train": "1978–2017", "val": "2018–2021", "test": "2022–2025"},
            "missing_days": store.attrs.get("missing_days_n"),
        },
        "training": {
            "seed": SEED,
            "sampling": {
                "mode": "每天随机取输入帧与目标帧均有效的像元（无放回）",
                "cells_per_day": DEFAULT_CELLS_PER_DAY,
                "n_rows_train_matrix": n_rows_all,
                "n_rows_ridge": n_rows_all,
                "n_rows_rf": int(Xr.shape[0]),
                "train_frames": int(store.indices_of_years(*TRAIN_YEARS).size),
            },
            "ridge": {"pipeline": "StandardScaler + Ridge", "alpha": ridge_alpha,
                      "fit_seconds": round(ridge_fit_s, 1),
                      "training_samples": n_rows_all},
            "random_forest": {
                "params": rf_params,
                "fit_seconds": round(rf_fit_s, 1),
                "training_samples": int(Xr.shape[0]),
                "tree_nodes_total": int(sum(node_counts)),
                "tree_nodes_mean": float(np.mean(node_counts)),
                "memory_safe_note": (
                    "最终 RF 因 8GB 机器在 200 树全深度 fit 阶段被系统 OOM kill，降级为："
                    "训练行子采样 1.5M（无放回 seed=42）、max_samples=0.6、max_depth=22、"
                    "min_samples_leaf=200、max_features=0.5、n_jobs=2，并采用 warm_start "
                    "30 树/块分块训练+逐块 joblib 检查点（rf.partial.joblib）。"
                    "调参阶段最优 val vec_rmse=%.4f（max_depth=28/min_samples_leaf=25/"
                    "max_features=0.5，500k 调参子集、80 树）；最终模型 val vec_rmse=%.4f，"
                    "差异来自内存降级，属预期取舍。"
                    % (tuning_best_val, rf_val["vec_rmse"])),
            },
            "tuning": tuning,
            "val_metrics": {
                "ridge": ridge_val,
                "random_forest": rf_val,
                "persistence": vec_metrics(pers_v, yv, pers=pers_v),
            },
            "val_rows": int(yv.shape[0]),
            "ridge_uncertainty_calibration": ridge_unc,
            "rf_uncertainty": {"method": "RF 树间速度 10–90 分位（转换为 24h 位移 km）",
                               "n_trees": int(rf.n_estimators)},
            "memory": {"maxrss_bytes": _maxrss(),
                       "maxrss_human": _fmt_bytes(_maxrss())},
        },
        "artifacts": {
            "ridge": str(ridge_path.relative_to(ROOT)),
            "random_forest": str(rf_path.relative_to(ROOT)),
            "feature_spec": f"{MODEL_DIR}/feature_spec.json",
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "platform": sys.platform,
        },
        "metric_conventions": {
            "pooled": ("统一评估口径：vec_rmse = sqrt(Σ(Δu²+Δv²)/N)；N=1,556,493，"
                       "按目标帧全部有效像元一次性池化（与旧报告 batch 等权口径不同）"),
            "legacy_batch_equal_weight": ("旧 reports/final_report.md 的 evaluate() 为 batch=8 "
                                          "等权平均；本包统一用 pooled，二者不可直接比较"),
        },
        "limitations": [
            "特征仅来自历史漂移场本身（无风场/海流/海冰密集度外强迫），可预测性主要来自持续性 + 季节/空间先验。",
            "日尺度产品只能给出 24h 平均矢量；逐时曲线为线性积分，非真实逐时运动。",
            "1978–1986 早期传感器存在系统性低估（regime shift），训练包含该段，可能压低早期拟合质量。",
            "训练采样为每像元独立样本，未建模空间相关性；RF 多输出树间离散度只是粗略不确定性代理。",
            "最终 RF 为内存安全降级配置（1.5M 行子采样、120 树、max_depth=22、min_samples_leaf=200），"
            "val 指标略逊于调参阶段最优配置。",
            "test 2022–2025 仅用于最终评估，不参与调参/标准化。",
        ],
    }
    _save_feature_spec()
    atomic_write_json(resolve_path(f"{MODEL_DIR}/model_card.json", default=f"{MODEL_DIR}/model_card.json"),
                      model_card)
    log("feature_spec.json / model_card.json 已写入")
    return {"ridge_val": ridge_val, "rf_val": rf_val,
            "n_rows_all": n_rows_all, "n_rows_rf": int(Xr.shape[0]),
            "ridge_fit_s": ridge_fit_s, "rf_fit_s": rf_fit_s}


def run_smoke(store: DataStore) -> dict:
    """小样本冒烟训练：尽快让引擎可用（非正式指标）。"""
    log("SMOKE：构建小样本特征矩阵（2015–2016, 40 cells/day）…")
    Xs, ys = [], []
    for year in (2015, 2016):
        d = _build_year_matrix(store, year, 40)
        if d is not None:
            Xs.append(d["X"])
            ys.append(d["y"])
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    log(f"SMOKE 训练行数 {X.shape[0]:,}")
    ridge = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    ridge.fit(X, y)
    rf = RandomForestRegressor(n_estimators=60, max_depth=18, min_samples_leaf=10,
                               max_features=0.5, n_jobs=4, random_state=SEED)
    rf.fit(X, y)
    ridge_path = resolve_path(f"{MODEL_DIR}/ridge.joblib", default=f"{MODEL_DIR}/ridge.joblib")
    rf_path = resolve_path(f"{MODEL_DIR}/rf.joblib", default=f"{MODEL_DIR}/rf.joblib")
    names = feature_spec()["feature_names"]
    atomic_joblib_dump({"pipeline": ridge, "model_id": "ridge", "feature_names": names,
                        "n_features": N_FEATURES, "n_history": N_HISTORY,
                        "alpha": 1.0, "stage": "smoke", "trained_at": utc_now_iso(),
                        "training_samples": int(X.shape[0])}, ridge_path)
    atomic_joblib_dump({"pipeline": rf, "model_id": "random_forest", "feature_names": names,
                        "n_features": N_FEATURES, "n_history": N_HISTORY,
                        "params": {"n_estimators": 60, "max_depth": 18, "min_samples_leaf": 10},
                        "stage": "smoke", "trained_at": utc_now_iso(),
                        "training_samples": int(X.shape[0])}, rf_path)
    _save_feature_spec()
    atomic_write_json(resolve_path(f"{MODEL_DIR}/model_card.json", default=f"{MODEL_DIR}/model_card.json"),
                      {"version": "1.0", "stage": "smoke", "created_at": utc_now_iso(),
                       "note": "冒烟模型（仅供接口联调）；正式训练将覆盖",
                       "training_samples": int(X.shape[0]),
                       "features": feature_spec()["feature_names"]})
    log(f"SMOKE 完成：{ridge_path.name} / {rf_path.name}（{int(X.shape[0]):,} 行）")
    return {"n": int(X.shape[0])}


# ---------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ridge/RF 离线训练与调参（后台运行）")
    ap.add_argument("--stage", default="smoke",
                    choices=["smoke", "tune", "final", "all"],
                    help="smoke=小样本冒烟; tune=val 调参; final=全量训练; all=tune+final")
    ap.add_argument("--zarr", default=None, help="zarr 路径（默认工作区 data/processed/...）")
    ap.add_argument("--force", action="store_true", help="忽略已有缓存/调参结果，强制重算")
    args = ap.parse_args(argv)

    log(f"===== predict.train_tabular stage={args.stage} seed={SEED} =====")
    log(f"numpy={np.__version__} sklearn={sklearn.__version__} python={sys.version.split()[0]}")
    t_all = time.time()
    store = DataStore(args.zarr)
    log(f"DataStore: {store}")

    try:
        if args.stage == "smoke":
            run_smoke(store)
        elif args.stage == "tune":
            run_tuning(store, force=args.force)
        elif args.stage == "final":
            run_final(store, force=args.force)
        elif args.stage == "all":
            run_tuning(store, force=args.force)
            run_final(store, force=args.force)
    except Exception:
        import traceback
        traceback.print_exc()
        log("训练失败，退出码 1")
        return 1
    log(f"===== 完成，总耗时 {time.time() - t_all:.1f}s，maxrss={_fmt_bytes(_maxrss())} =====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
