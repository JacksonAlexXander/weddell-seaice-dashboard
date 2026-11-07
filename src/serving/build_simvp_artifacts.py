"""SimVP 服务化 artifacts 构建：统一口径评估指标 + test 期逐日 day-1 预测场预计算。

用法（后台跑）：
  .venv/bin/python src/serving/build_simvp_artifacts.py --eval --precompute > logs/simvp_serving_build.log 2>&1

产物：
  data/processed/dashboard/metrics_simvp.json      # 统一评估协议上的 SimVP 指标（Lead 合并进 metrics.json）
  data/processed/dashboard/simvp_day1_fields.npz   # 预计算 day-1 场（test 期，供回测/预测场秒级响应）

评估协议：从项目 2022–2025 测试窗中保留连续 10 日输入及次日观测。
使用 --update-combined 生成同一组样本上的 Ridge、RF、SimVP 和基线指标。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.serving.simvp import DEFAULT_CKPT, SimVPPredictor  # noqa: E402
from src.predict.common import atomic_write_json
from src.predict.metrics import vec_metrics

EVAL_WINDOWS = ROOT / "data/processed/dashboard/eval_windows.json"
OUT_METRICS = ROOT / "data/processed/dashboard/metrics_simvp.json"
OUT_FIELDS = ROOT / "data/processed/dashboard/simvp_day1_fields.npz"


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_eval_windows() -> list[dict]:
    with open(EVAL_WINDOWS, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta["windows"]


class MetricAccumulator:
    """在全部评估样本上累加误差统计量（与设计文档指标定义一致）。"""

    def __init__(self):
        self.n = 0
        self.s_du2 = 0.0
        self.s_dv2 = 0.0
        self.s_d2 = 0.0
        self.s_abs = 0.0
        self.s_ou = 0.0
        self.s_ou2 = 0.0
        self.s_ov = 0.0
        self.s_ov2 = 0.0
        self.s_pdu2 = 0.0
        self.s_pdv2 = 0.0
        self.s_pd2 = 0.0

    def add(self, ou, ov, pu, pv, pers_u, pers_v):
        du, dv = pu - ou, pv - ov
        d2 = du * du + dv * dv
        self.n += ou.size
        self.s_du2 += float(np.sum(du * du))
        self.s_dv2 += float(np.sum(dv * dv))
        self.s_d2 += float(np.sum(d2))
        self.s_abs += float(np.sum(np.sqrt(d2)))
        self.s_ou += float(np.sum(ou))
        self.s_ou2 += float(np.sum(ou * ou))
        self.s_ov += float(np.sum(ov))
        self.s_ov2 += float(np.sum(ov * ov))
        self.s_pdu2 += float(np.sum((pers_u - ou) ** 2))
        self.s_pdv2 += float(np.sum((pers_v - ov) ** 2))
        self.s_pd2 += float(np.sum((pers_u - ou) ** 2 + (pers_v - ov) ** 2))

    def result(self) -> dict:
        n = max(self.n, 1)
        rmse_u = math.sqrt(self.s_du2 / n)
        rmse_v = math.sqrt(self.s_dv2 / n)
        vec = math.sqrt(self.s_d2 / n)
        mae = self.s_abs / n
        mu, mv = self.s_ou / n, self.s_ov / n
        sst_u = self.s_ou2 - n * mu * mu
        sst_v = self.s_ov2 - n * mv * mv
        r2_u = 1.0 - (self.s_du2 / sst_u) if sst_u > 0 else float("nan")
        r2_v = 1.0 - (self.s_dv2 / sst_v) if sst_v > 0 else float("nan")
        pers_vec = math.sqrt(self.s_pd2 / n)
        pers_rmse_u = math.sqrt(self.s_pdu2 / n)
        pers_rmse_v = math.sqrt(self.s_pdv2 / n)
        return {
            "n_samples": int(self.n),
            "rmse_u": round(rmse_u, 4),
            "rmse_v": round(rmse_v, 4),
            "vec_rmse": round(vec, 4),
            "mae_vec": round(mae, 4),
            "r2_u": round(float(r2_u), 4),
            "r2_v": round(float(r2_v), 4),
            "skill_vs_persistence": round(1.0 - vec / (pers_vec + 1e-12), 4),
            "persistence": {
                "vec_rmse": round(pers_vec, 4),
                "rmse_u": round(pers_rmse_u, 4),
                "rmse_v": round(pers_rmse_v, 4),
            },
        }


def run_eval(predictor: SimVPPredictor, windows: list[dict], limit: int = 0) -> dict:
    acc = MetricAccumulator()
    times = []
    wlist = windows[:limit] if limit else windows
    for k, w in enumerate(wlist):
        last_obs = int(w["last_obs_idx"])
        truth = int(w["truth_idx"])
        t0 = time.perf_counter()
        u_p, v_p, target_idx, source = predictor._field_for_last_obs(last_obs)
        times.append((time.perf_counter() - t0) * 1000.0)
        if target_idx != truth:
            raise RuntimeError(f"target_idx mismatch: {target_idx} != {truth} @window {k}")
        mask = predictor._mask(truth)
        ou = np.asarray(predictor._zarr["u"][truth], dtype=np.float32)[mask]
        ov = np.asarray(predictor._zarr["v"][truth], dtype=np.float32)[mask]
        # 持续性基线：起报帧（last_obs）的 u/v
        pers_u = np.asarray(predictor._zarr["u"][last_obs], dtype=np.float32)[mask]
        pers_v = np.asarray(predictor._zarr["v"][last_obs], dtype=np.float32)[mask]
        acc.add(ou.astype(np.float64), ov.astype(np.float64),
                u_p[mask].astype(np.float64), v_p[mask].astype(np.float64),
                pers_u.astype(np.float64), pers_v.astype(np.float64))
        if (k + 1) % 50 == 0 or k == len(wlist) - 1:
            log(f"[eval] {k + 1}/{len(wlist)} windows, n={acc.n}, "
                f"vec_rmse≈{acc.result()['vec_rmse']:.3f}")
    res = acc.result()
    res["n_windows"] = len(wlist)
    res["latency_ms"] = {
        "median": round(float(np.median(times[1:]) if len(times) > 1 else times[0]), 2),
        "p90": round(float(np.percentile(times[1:], 90)) if len(times) > 1 else round(times[0], 2), 2),
    }
    return res


def run_precompute(predictor: SimVPPredictor, out_fields: Path, limit: int = 0) -> dict:
    predictor._ensure_zarr()
    dates = predictor._dates
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    test_mask = (years >= 2022) & (years <= 2025)
    target_idx = np.where(test_mask)[0]
    target_idx = target_idx[target_idx >= 1]  # 需要前一帧作为起报帧
    adjacent = dates[target_idx] - dates[target_idx - 1] == np.timedelta64(1, "D")
    target_idx = target_idx[adjacent]
    if limit:
        target_idx = target_idx[:limit]
    n = len(target_idx)
    log(f"[precompute] {n} 个 test 期目标帧，开始推理 ...")
    u_out = np.zeros((n, 116, 158), dtype=np.float16)
    v_out = np.zeros((n, 116, 158), dtype=np.float16)
    t_all = time.time()
    for k, ti in enumerate(target_idx):
        u, v, target, _src = predictor._field_for_last_obs(int(ti) - 1)
        if target != int(ti):
            raise RuntimeError(f"precompute target mismatch {target} != {ti}")
        u_out[k] = u.astype(np.float16)
        v_out[k] = v.astype(np.float16)
        if (k + 1) % 100 == 0 or k == n - 1:
            el = time.time() - t_all
            eta = el / (k + 1) * (n - k - 1)
            log(f"[precompute] {k + 1}/{n} ({el:.0f}s, ETA {eta:.0f}s)")
    out_dates = dates[target_idx].astype("datetime64[s]")
    with predictor.ckpt_path.open("rb") as checkpoint:
        digest = hashlib.file_digest(checkpoint, "sha256").hexdigest()
    tmp = out_fields.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, dates=out_dates, target_idx=target_idx.astype(np.int32),
                        u=u_out, v=v_out, checkpoint_sha256=digest)
    tmp.replace(out_fields)
    log(f"[precompute] saved {out_fields} ({out_fields.stat().st_size / 1e6:.1f} MB)")
    return {"n": int(n), "range": [str(dates[target_idx].min()), str(dates[target_idx].max())]}


def same_test_windows(predictor: SimVPPredictor) -> list[dict]:
    """从原测试协议中保留连续 10 日输入和严格 24h 目标的窗口。"""
    predictor._ensure_zarr()
    dates = predictor._dates
    return [
        window for window in load_eval_windows()
        if dates[window["truth_idx"]] - dates[window["input_start"]]
        == np.timedelta64(predictor.tin, "D")
    ]


def update_combined(out_dir: Path, windows: list[dict], model_entry: dict, result: dict) -> None:
    import pyarrow.parquet as pq

    ids = ("ridge", "random_forest", "persistence", "climatology")
    columns = ["window", "obs_u", "obs_v"] + [f"{mid}_{axis}" for mid in ids for axis in ("u", "v")]
    data = pq.read_table(EVAL_WINDOWS.parent / "eval_day1.parquet", columns=columns)
    accepted = {window["truth_idx"] for window in windows}
    keep = np.isin(data["window"].to_numpy(), [
        i for i, w in enumerate(load_eval_windows()) if w["truth_idx"] in accepted
    ])
    data = data.filter(keep)

    def pairs(name):
        return np.column_stack([data[f"{name}_{axis}"].to_numpy() for axis in ("u", "v")])

    truth, persistence = pairs("obs"), pairs("persistence")
    if len(truth) != result["eval_day1"]["n_samples"]:
        raise RuntimeError("时空模型和表格模型评估样本数不一致")
    tabular = json.loads((EVAL_WINDOWS.parent / "metrics_tabular.json").read_text())
    tabular["generated_at"] = result["generated_at"]
    tabular["protocol"].update(
        sampling=f"2022–2025 留出期：每 5 帧抽样，仅保留连续 10 日输入及次日真值（{len(windows)} 窗口），使用全部有效像元",
        n_windows=len(windows),
        n_samples=len(truth),
    )
    tabular["protocol"].pop("reference", None)
    for item in tabular["models"]:
        item.update({key: round(value, 4) for key, value in
                     vec_metrics(pairs(item["id"]), truth, pers=persistence).items()})
    atomic_write_json(out_dir / "metrics_tabular.json", tabular)
    combined = {**tabular, "models": [*tabular["models"], model_entry]}
    atomic_write_json(out_dir / "metrics.json", combined)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--precompute", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 个（调试）")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--compare-ckpt", type=Path)
    ap.add_argument("--out-dir", type=Path, default=OUT_METRICS.parent)
    ap.add_argument("--update-combined", action="store_true")
    args = ap.parse_args()
    if not (args.eval or args.precompute):
        args.eval = args.precompute = True
    if args.update_combined and (not args.eval or args.limit):
        ap.error("--update-combined 需要完整评估")

    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    predictor = SimVPPredictor(device=args.device, ckpt=args.ckpt, precomputed=None)
    st = predictor.load()
    if not st["loaded"]:
        raise SystemExit(f"SimVP checkpoint 加载失败: {st['error']}")
    log(f"[load] device={st['device']} ckpt_epoch={predictor._ckpt_epoch} best_val={predictor._ckpt_best_val}")
    checkpoint = args.ckpt.resolve().relative_to(ROOT)
    tuned = predictor._init_weights is not None
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ckpt": str(checkpoint),
        "checkpoint_epoch": predictor._ckpt_epoch,
        "checkpoint_best_val": predictor._ckpt_best_val,
        "device": st["device"],
    }
    if args.eval:
        windows = same_test_windows(predictor)
        log(f"[eval] {len(windows)} consecutive windows")
        out["eval_day1"] = run_eval(predictor, windows, args.limit)
        log(f"[eval] result: {json.dumps(out['eval_day1'], ensure_ascii=False)}")
        if args.compare_ckpt:
            baseline = SimVPPredictor(device=args.device, ckpt=args.compare_ckpt, precomputed=None)
            baseline._ensure_zarr()
            out["original_checkpoint_eval"] = run_eval(baseline, windows, args.limit)
            del baseline
            log(f"[eval/original] {json.dumps(out['original_checkpoint_eval'], ensure_ascii=False)}")
    if args.precompute:
        out["precompute"] = run_precompute(predictor, args.out_dir / OUT_FIELDS.name, args.limit)

    name = "SimVP 24h 微调" if tuned else "SimVP-v1 深度学习"
    model_entry = {
        "id": "simvp", "name": name, "name_zh": name,
        "short": "SimVP 24h" if tuned else "SimVP",
        "category": "deep", "color": "#f59e0b", "is_baseline": False,
        "description": "10 帧日漂移场输入，取第 1 天预测。保留空间门控、时间混合与季节通道。",
        "trained_at": datetime.fromtimestamp(args.ckpt.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
        "params": {"tin": predictor.tin, "tout": predictor.tout, "chid": predictor.chid,
                   "ckpt": str(checkpoint), "device": st["device"], "epoch": predictor._ckpt_epoch},
        "artifact_source": str(checkpoint),
    }
    if "eval_day1" in out:
        ev = out["eval_day1"]
        model_entry.update({key: ev[key] for key in
                            ("vec_rmse", "rmse_u", "rmse_v", "mae_vec", "r2_u", "r2_v", "skill_vs_persistence")})
        model_entry["latency_ms"] = ev["latency_ms"]["median"]
    atomic_write_json(args.out_dir / OUT_METRICS.name, {**out, "model": model_entry})
    if args.update_combined:
        update_combined(args.out_dir, windows, model_entry, out)
    log(f"[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
