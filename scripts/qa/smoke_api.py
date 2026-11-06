"""API 冒烟测试：逐端点验证 + 关键字段断言 + 延迟统计 + 错误码检查。

用法：
  .venv/bin/python scripts/qa/smoke_api.py                 # 默认 http://127.0.0.1:8000
  .venv/bin/python scripts/qa/smoke_api.py --base http://127.0.0.1:8010
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import httpx

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = ""):
    if cond:
        PASS.append(name)
        print(f"  ✓ {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAIL.append(name)
        print(f"  ✗ {name}" + (f"  ({detail})" if detail else ""))


def timed(fn):
    t0 = time.perf_counter()
    r = fn()
    return r, (time.perf_counter() - t0) * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--predict-date", default="2025-06-26")
    ap.add_argument("--cell", default="65,60")
    args = ap.parse_args()
    base = args.base.rstrip("/")
    row, col = (int(x) for x in args.cell.split(","))
    c = httpx.Client(base_url=base, timeout=120.0)

    print(f"== 冒烟测试 {base} ==")
    # ---------------- health
    print("[health]")
    r = c.get("/api/health")
    check("GET /api/health 200", r.status_code == 200, str(r.status_code))
    h = r.json()
    check("health.status=ok", h.get("status") == "ok", str(h.get("status")))
    print(f"     models_loaded={h.get('models_loaded')}")

    # ---------------- models
    print("[models]")
    r = c.get("/api/models")
    check("GET /api/models 200", r.status_code == 200)
    m = r.json()
    ids = [x["id"] for x in m.get("models", [])]
    check("models >= 5", len(ids) >= 5, str(ids))
    for need in ("ridge", "random_forest", "simvp", "persistence", "climatology"):
        check(f"model {need} 存在", need in ids)
    check("models.default 存在", bool(m.get("default")), json.dumps(m.get("default"), ensure_ascii=False))
    check("data_time.latest_date", bool((m.get("data_time") or {}).get("latest_date")),
          str((m.get("data_time") or {}).get("latest_date")))
    avail = {x["id"]: x.get("available") for x in m.get("models", [])}
    print(f"     available={avail}")

    # ---------------- metrics
    print("[metrics]")
    r = c.get("/api/metrics")
    check("GET /api/metrics 200", r.status_code == 200)
    mm = r.json()
    check("metrics.protocol.n_samples=1556493",
          int((mm.get("protocol") or {}).get("n_samples") or 0) == 1556493,
          str((mm.get("protocol") or {}).get("n_samples")))
    rows = {x["id"]: x for x in mm.get("models", [])}
    for need in ("ridge", "random_forest"):
        check(f"metrics 含 {need}", need in rows)
        if need in rows:
            check(f"{need}.vec_rmse 数值", isinstance(rows[need].get("vec_rmse"), (int, float)),
                  str(rows[need].get("vec_rmse")))
            check(f"{need}.skill_vs_persistence 数值",
                  isinstance(rows[need].get("skill_vs_persistence"), (int, float)),
                  str(rows[need].get("skill_vs_persistence")))
    if "ridge" in rows and "random_forest" in rows:
        print(f"     ridge vec_rmse={rows['ridge'].get('vec_rmse')} "
              f"| rf vec_rmse={rows['random_forest'].get('vec_rmse')}")

    # ---------------- overview
    print("[overview]")
    r = c.get("/api/overview")
    check("GET /api/overview 200", r.status_code == 200)
    ov = r.json()
    check("overview.annual=48", len(ov.get("annual", [])) == 48, str(len(ov.get("annual", []))))
    check("overview.monthly=12", len(ov.get("monthly", [])) == 12, str(len(ov.get("monthly", []))))
    check("overview.kpis=4", len(ov.get("kpis", [])) == 4, str(len(ov.get("kpis", []))))
    check("overview.data_time 最新日期", bool((ov.get("data_time") or {}).get("latest_date")),
          str((ov.get("data_time") or {}).get("latest_date")))

    # ---------------- field
    print("[field]")
    r = c.get("/api/field", params={"date": args.predict_date})
    check("GET /api/field 200", r.status_code == 200)
    f = r.json()
    check("field.kind=observed", f.get("kind") == "observed")
    check("field.speed 116x158",
          len(f.get("speed", [])) == 116 and len(f["speed"][0]) == 158,
          f"{len(f.get('speed', []))}x{len((f.get('speed') or [[]])[0])}")
    check("field.vectors 非空", len((f.get("vectors") or {}).get("rows", [])) > 100,
          str(len((f.get("vectors") or {}).get("rows", []))))
    check("field.stats.valid_cells>0", int(f["stats"]["valid_cells"]) > 0,
          str(f["stats"]["valid_cells"]))

    # ---------------- predict（真实推理）
    print("[predict]")
    predict_times = {}
    for mid in ("ridge", "random_forest", "simvp"):
        if mid not in ids:
            continue
        body = {"model": mid, "date": args.predict_date, "cell": {"row": row, "col": col}}
        try:
            r, ms = timed(lambda: c.post("/api/predict", json=body))
            predict_times[mid] = round(ms, 1)
            ok = r.status_code == 200
            check(f"POST /api/predict[{mid}] 200", ok, f"{r.status_code} {r.text[:160]}")
            if ok:
                p = r.json()
                check(f"predict[{mid}].status=ok", p.get("status") == "ok")
                check(f"predict[{mid}].prediction 数值",
                      all(isinstance(p.get("prediction", {}).get(k), (int, float))
                          for k in ("u", "v", "speed")))
                s = p.get("series_24h") or {}
                check(f"predict[{mid}].series_24h 25 点", len(s.get("hours", [])) == 25,
                      str(len(s.get("hours", []))))
                check(f"predict[{mid}].data_time.next_frame_date",
                      bool((p.get("data_time") or {}).get("next_frame_date")),
                      str((p.get("data_time") or {}).get("next_frame_date")))
                check(f"predict[{mid}] 含 warnings 字段", isinstance(p.get("warnings"), list))
                check(f"predict[{mid}] 延迟 < 5s", ms < 5000, f"{ms:.0f}ms")
        except Exception as e:  # noqa: BLE001
            check(f"POST /api/predict[{mid}]", False, f"异常 {e}")
    print(f"     predict 延迟(含HTTP): {predict_times}")

    # ---------------- field/predicted
    print("[field/predicted]")
    r, ms = timed(lambda: c.get("/api/field/predicted",
                                params={"model": "random_forest", "date": args.predict_date}))
    check("GET /api/field/predicted 200", r.status_code == 200, f"{r.status_code} {r.text[:160]}")
    if r.status_code == 200:
        pf = r.json()
        check("predicted.kind=predicted", pf.get("kind") == "predicted")
        check("predicted.speed 116x158",
              len(pf.get("speed", [])) == 116 and len(pf["speed"][0]) == 158)
        print(f"     field/predicted 延迟 {ms:.0f}ms")

    # ---------------- backtest
    print("[backtest]")
    for mid in ("ridge", "random_forest", "simvp"):
        if mid not in ids:
            continue
        body = {"model": mid, "cell": {"row": row, "col": col},
                "start": "2025-01-01", "end": "2025-12-30", "max_points": 90}
        r, ms = timed(lambda: c.post("/api/backtest", json=body))
        ok = r.status_code == 200
        check(f"POST /api/backtest[{mid}] 200", ok, f"{r.status_code} {r.text[:160]}")
        if ok:
            bt = r.json()
            check(f"backtest[{mid}].series 非空", len(bt.get("series", [])) > 0,
                  str(len(bt.get("series", []))))
            check(f"backtest[{mid}].summary.n", (bt.get("summary") or {}).get("n", 0) > 0)
        print(f"     backtest[{mid}] 延迟 {ms:.0f}ms")

    # ---------------- 错误路径
    print("[错误路径]")
    r = c.post("/api/predict", json={"model": "nope", "date": args.predict_date,
                                     "cell": {"row": row, "col": col}})
    check("未知模型 404", r.status_code == 404, str(r.status_code))
    check("错误体含 error.code", "error" in r.json() and "code" in r.json()["error"])
    r = c.post("/api/predict", json={"model": "ridge", "date": args.predict_date,
                                     "cell": {"row": 999, "col": 999}})
    check("越界像元 400", r.status_code == 400, str(r.status_code))
    r = c.post("/api/predict", json={"model": "ridge", "date": "1900-01-01",
                                     "cell": {"row": row, "col": col}})
    check("非法日期 422", r.status_code == 422, str(r.status_code))

    # ---------------- 静态首页/文档
    print("[静态/文档]")
    r = c.get("/")
    check("GET / 200", r.status_code == 200)
    if r.status_code == 200:
        ctype = r.headers.get("content-type", "")
        if "text/html" in ctype:
            check("首页为 HTML（前端已构建）", "<div id=\"app\"" in r.text or "<div id='app'" in r.text)
        else:
            print(f"     [warn] 首页非 HTML（前端 dist 未构建？）: {ctype}")
    r = c.get("/docs")
    check("GET /docs 200", r.status_code == 200)
    r = c.get("/openapi.json")
    check("GET /openapi.json 200", r.status_code == 200)

    print("\n== 结果 ==")
    print(f"PASS {len(PASS)} / FAIL {len(FAIL)}")
    if FAIL:
        print("失败项：")
        for x in FAIL:
            print("  -", x)
        sys.exit(1)
    print("全部通过 ✅  predict 延迟: " + json.dumps(predict_times, ensure_ascii=False))


if __name__ == "__main__":
    main()
