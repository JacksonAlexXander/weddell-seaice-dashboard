"""验证「点击预测不重新训练」：对已训练 artifact 做 SHA256/mtime 快照，
调用各模型预测接口若干次后再次快照，必须完全不变，且 models/ 无新增文件。

用法：.venv/bin/python scripts/qa/check_no_retrain.py --base http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
WATCH = [
    ROOT / "models/ml/ridge.joblib",
    ROOT / "models/ml/rf.joblib",
    ROOT / "models/simvp-v1/ckpt_best.pt",
    ROOT / "models/simvp-24h/ckpt_best.pt",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot() -> dict:
    snap = {}
    for p in WATCH:
        if p.exists():
            st = p.stat()
            snap[str(p.relative_to(ROOT))] = {
                "sha256": sha256(p), "mtime_ns": st.st_mtime_ns, "size": st.st_size,
            }
    files = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "models").rglob("*")
                   if p.is_file() and not p.name.startswith("."))
    return {"files": snap, "all_files": files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--date", default="2025-06-26")
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    c = httpx.Client(base_url=args.base.rstrip("/"), timeout=180.0)

    before = snapshot()
    print(f"快照模型文件 {len(before['files'])} 个，models/ 下 {len(before['all_files'])} 个文件")
    for k, v in before["files"].items():
        print(f"  {k}: {v['sha256'][:16]}… {v['size'] / 1e6:.1f}MB")

    calls = 0
    latencies = []
    for _ in range(args.rounds):
        for mid in ("ridge", "random_forest", "simvp"):
            body = {"model": mid, "date": args.date, "cell": {"row": 65, "col": 60}}
            t0 = time.perf_counter()
            r = c.post("/api/predict", json=body)
            ms = (time.perf_counter() - t0) * 1000.0
            if r.status_code != 200:
                raise RuntimeError(f"predict {mid} -> {r.status_code}: {r.text[:200]}")
            latencies.append((mid, ms))
            calls += 1
        # 变化像元，触发不同推理路径
        for row, col in ((40, 40), (80, 120)):
            c.post("/api/predict", json={"model": "random_forest", "date": args.date,
                                         "cell": {"row": row, "col": col}}).raise_for_status()
            calls += 1
    print(f"共调用 /api/predict {calls} 次；延迟: " + ", ".join(
        f"{m}:{ms:.0f}ms" for m, ms in latencies[::3]))

    after = snapshot()
    if before != after:
        print("❌ 检测到模型 artifact 发生变化（疑似重训/重写）！")
        for k in before["files"]:
            if before["files"].get(k) != after["files"].get(k):
                print("  变化:", k)
        if before["all_files"] != after["all_files"]:
            added = set(after["all_files"]) - set(before["all_files"])
            removed = set(before["all_files"]) - set(after["all_files"])
            print("  新增:", added, "删除:", removed)
        sys.exit(1)
    print("✅ 所有模型 artifact 的 SHA256/mtime/size 完全未变，models/ 无新增文件 → 预测未触发训练")


if __name__ == "__main__":
    main()
