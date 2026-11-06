"""统一评估指标（与 docs/PREDICT_API_CONTRACT.md 的 metric_defs 一一对应）。

所有模型（Ridge / RF / persistence / climatology / SimVP）在统一评估协议
`data/processed/dashboard/eval_windows.json` 上计算，指标定义：

- `rmse_u` / `rmse_v`：分量 RMSE（cm/s）
- `vec_rmse`：sqrt(mean(Δu²+Δv²))（cm/s）
- `mae_vec`：mean(sqrt(Δu²+Δv²))（cm/s）
- `r2_u` / `r2_v`：分量决定系数（以评估样本均值为参考）
- `skill_vs_persistence`：1 − vec_rmse / vec_rmse_persistence
"""

from __future__ import annotations

import numpy as np


def vec_metrics(pred: np.ndarray, obs: np.ndarray, pers: np.ndarray | None = None) -> dict:
    """pred / obs / pers: (n,2) 数组（列 = u,v，cm/s）。返回指标 dict。"""
    pred = np.asarray(pred, np.float64)
    obs = np.asarray(obs, np.float64)
    du = pred[:, 0] - obs[:, 0]
    dv = pred[:, 1] - obs[:, 1]
    d2 = du * du + dv * dv
    n = int(obs.shape[0])
    out = {
        "n": n,
        "rmse_u": float(np.sqrt(np.mean(du * du))),
        "rmse_v": float(np.sqrt(np.mean(dv * dv))),
        "vec_rmse": float(np.sqrt(np.mean(d2))),
        "mae_vec": float(np.mean(np.sqrt(d2))),
    }
    mu = float(np.mean(obs[:, 0]))
    mv = float(np.mean(obs[:, 1]))
    ss_u = float(np.sum((obs[:, 0] - mu) ** 2))
    ss_v = float(np.sum((obs[:, 1] - mv) ** 2))
    out["r2_u"] = float(1.0 - np.sum(du * du) / max(ss_u, 1e-12))
    out["r2_v"] = float(1.0 - np.sum(dv * dv) / max(ss_v, 1e-12))
    if pers is not None:
        pers = np.asarray(pers, np.float64)
        d2p = (pers[:, 0] - obs[:, 0]) ** 2 + (pers[:, 1] - obs[:, 1]) ** 2
        base = float(np.sqrt(np.mean(d2p)))
        out["persistence_vec_rmse"] = base
        out["skill_vs_persistence"] = float(1.0 - out["vec_rmse"] / max(base, 1e-12))
    return out


def metric_defs() -> dict:
    """写入 metrics_tabular.json 的 protocol.metric_defs（后端/前端直接引用）。"""
    return {
        "vec_rmse": "√mean(Δu²+Δv²)，cm/s",
        "mae_vec": "mean(√(Δu²+Δv²))，cm/s",
        "skill_vs_persistence": "1 − vec_rmse_model/vec_rmse_persistence（矢量 RMSE 比口径，与项目 skill_pers 一致）",
        "r2_u": "u 分量决定系数（以评估集均值为参考）",
        "latency_ms": "单次单点预测端到端推理耗时中位数（离线实测，不含 HTTP 开销）",
    }
