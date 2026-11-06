"""SimVP 服务化 artifacts 构建：统一口径评估指标 + test 期逐日 day-1 预测场预计算。

用法（后台跑）：
  .venv/bin/python src/serving/build_simvp_artifacts.py --eval --precompute > logs/simvp_serving_build.log 2>&1

产物：
  data/processed/dashboard/metrics_simvp.json      # 统一评估协议上的 SimVP 指标（Lead 合并进 metrics.json）
  data/processed/dashboard/simvp_day1_fields.npz   # 预计算 day-1 场（test 期，供回测/预测场秒级响应）

评估协议：data/processed/dashboard/eval_windows.json（项目 test split + stride=5，287 个窗口，
与 reports/final_report.md 的 SimVP 测试口径一致）。预期 day-1 vec_rmse ≈ 9.79 cm/s（±0.1）。
"""

from __future__ import annotations

import argparse
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

from src.serving.simvp import SimVPPredictor  # noqa: E402

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


def run_precompute(predictor: SimVPPredictor, limit: int = 0) -> dict:
    predictor._ensure_zarr()
    dates = predictor._dates
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    test_mask = (years >= 2022) & (years <= 2025)
    target_idx = np.where(test_mask)[0]
    target_idx = target_idx[target_idx >= 1]  # 需要前一帧作为起报帧
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
    tmp = OUT_FIELDS.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, dates=out_dates, target_idx=target_idx.astype(np.int32),
                        u=u_out, v=v_out)
    tmp.replace(OUT_FIELDS)
    log(f"[precompute] saved {OUT_FIELDS} ({OUT_FIELDS.stat().st_size / 1e6:.1f} MB)")
    return {"n": int(n), "range": [str(dates[target_idx].min()), str(dates[target_idx].max())]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--precompute", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 个（调试）")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    if not (args.eval or args.precompute):
        args.eval = args.precompute = True

    t0 = time.time()
    predictor = SimVPPredictor(device=args.device)
    st = predictor.load()
    if not st["loaded"]:
        raise SystemExit(f"SimVP checkpoint 加载失败: {st['error']}")
    log(f"[load] device={st['device']} ckpt_epoch={predictor._ckpt_epoch} best_val={predictor._ckpt_best_val}")
    log(f"[load] state={json.dumps(st, ensure_ascii=False)}")

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ckpt": str(predictor.ckpt_path.relative_to(ROOT)),
        "ckspt_epoch": predictor._ckpt_epoch,
        "ckpt_best_val": predictor._ckpt_best_val,
        "device": st["device"],
    }
    if args.eval:
        windows = load_eval_windows()
        log(f"[eval] {len(windows)} windows from eval_windows.json")
        ev = run_eval(predictor, windows, args.limit)
        out["eval_day1"] = ev
        log(f"[eval] result: {json.dumps(ev, ensure_ascii=False)}")
    if args.precompute:
        pc = run_precompute(predictor, args.limit)
        out["precompute"] = pc

    # 模型卡片（字段与设计 §3 一致；Lead 合并进最终 metrics.json）
    model_entry = {
        "id": "simvp",
        "name": "SimVP-v1 深度学习",
        "name_zh": "SimVP-v1 深度学习",
        "short": "SimVP",
        "category": "deep",
        "description": "项目既有 SimVP-gSTA 风格时空预测模型（~9.3M 参数，10 帧输入→5 天场，取 day-1）；"
                       "25 km 原生网格，加载 ckpt_best@ep22 推理，不重新训练。",
        "trained_at": "2026-10-26T00:20:00Z",
        "training_samples": None,
        "params": {"tin": 10, "tout": 5, "chid": 128, "params_m": 9.3,
                   "ckpt": str(predictor.ckpt_path.relative_to(ROOT)),
                   "device": st["device"], "epoch": predictor._ckpt_epoch},
        "artifact_source": "models/simvp-v1/ckpt_best.pt（项目既有训练产物）",
        "is_baseline": False,
        "reference_report": {
            "note": "项目 reports/final_report.md §5.2 的 day-1 vec_rmse=9.7938 采用 evaluate() 的"
                    "「逐 batch MSE 等权」聚合（batch=8，等价于每个起报窗口等权，夏季低覆盖窗口被放大权重）；"
                    "本统一协议按全部有效像元池化（pooled）聚合，供 Ridge/RF/SimVP 公平比较。",
            "report_day1_vec_rmse_batchweighted": 9.7938,
        },
    }
    if "eval_day1" in out:
        ev = out["eval_day1"]
        model_entry.update({
            "vec_rmse": ev["vec_rmse"], "rmse_u": ev["rmse_u"], "rmse_v": ev["rmse_v"],
            "mae_vec": ev["mae_vec"], "r2_u": ev["r2_u"], "r2_v": ev["r2_v"],
            "skill_vs_persistence": ev["skill_vs_persistence"],
            "latency_ms": ev["latency_ms"]["median"],
        })
    with open(OUT_METRICS, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": out["generated_at"],
            "model": model_entry,
            "eval": out.get("eval_day1"),
            "precompute": out.get("precompute"),
            "protocol": {
                "task": "未来 24 小时（下一日帧）海冰漂移矢量预测",
                "sampling": "test split 每 5 日一个起报窗口（287 窗口），评估全部有效像元",
                "aggregation": "按全部有效像元池化（pooled：vec_rmse=sqrt(mean(Δu²+Δv²)))",
                "n_samples": out.get("eval_day1", {}).get("n_samples"),
                "source_windows": "data/processed/dashboard/eval_windows.json",
            },
        }, f, ensure_ascii=False, indent=1)
    log(f"[save] {OUT_METRICS}")
    log(f"[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
