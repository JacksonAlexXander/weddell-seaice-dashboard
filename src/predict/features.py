"""表格特征工程（Ridge / RandomForest 共用）：约 34 维，全部向量化（无逐像元 Python 循环）。

任务：给定 `last_obs_idx = t`（最后观测帧）与像元 (row,col)，预测 t+1 帧的 (u,v) cm/s。
特征（见 docs/PREDICT_DASHBOARD_DESIGN.md §3.1）：

- 中心像元滞后 u,v,speed @ t,t-1,t-2,t-3                     12
- 滚动均值 u,v,speed 近 3 日、近 7 日                          6
- 邻域（3×3 除中心）均值 u,v,speed @ t 与 t-1；邻域 u,v 标准差 @ t  8
- 趋势 u,v,speed 的 (t − t-3) 差分                             3
- 季节 sin(2π·doy), cos(2π·doy)                               2
- 位置/质量 latitude, longitude, 当日 valid_fraction            3
合计 34 维（设计文档"约 31 维"的实现值）。

缺失处理：zarr 中无效像元 u/v/speed 为 0，本模块内部一律以 mask 判定无效；对每像元在
N_HISTORY 帧窗口内做**前向填充**（用最近的有效历史值），窗口内全无效则填 0。邻域统计只使用
有效邻居；邻域全无效时回退到中心像元填充值。边缘像元用"夹取复制"（clip）近似。
"""

from __future__ import annotations

import numpy as np

GRID_H, GRID_W = 116, 158
N_HISTORY = 8  # 需要的历史天数（含当帧）：满足 t-3 滞后 + 近 7 日均值 + 邻域 t-1

FEATURE_NAMES: list[str] = [
    # 中心像元滞后 (12)
    "u_t", "v_t", "speed_t",
    "u_t1", "v_t1", "speed_t1",
    "u_t2", "v_t2", "speed_t2",
    "u_t3", "v_t3", "speed_t3",
    # 滚动均值 (6)
    "u_mean3", "v_mean3", "speed_mean3",
    "u_mean7", "v_mean7", "speed_mean7",
    # 邻域 (8)
    "nb_u_t", "nb_v_t", "nb_speed_t",
    "nb_u_t1", "nb_v_t1", "nb_speed_t1",
    "nb_std_u_t", "nb_std_v_t",
    # 趋势 (3)
    "du_t3", "dv_t3", "dspeed_t3",
    # 季节 (2)
    "sin_doy", "cos_doy",
    # 位置/质量 (3)
    "latitude", "longitude", "valid_fraction",
]
N_FEATURES = len(FEATURE_NAMES)
assert N_FEATURES == 34, N_FEATURES


# ---------------------------------------------------------------- 内部工具（向量化）
def _ffill_hist(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """沿时间轴（第 0 维）前向填充：用最近的有效历史值替代无效值；全无效填 0。

    x: (n_hist, n_cells) float32；valid: (n_hist, n_cells) bool。
    """
    nh, n = x.shape
    last = np.where(valid, np.arange(nh, dtype=np.int32)[:, None], -1)
    np.maximum.accumulate(last, axis=0, out=last)          # 每个位置最近有效帧号
    has = last >= 0
    idx = np.clip(last, 0, max(nh - 1, 0))
    filled = np.take_along_axis(x, idx, axis=0)
    return np.where(has, filled, 0.0).astype(np.float32, copy=False)


def _neighbor_stats(values: dict, mask: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                    fallback: dict, height: int, width: int) -> dict:
    """3×3 邻域（排除中心）有效像元均值/标准差。

    values: {"u","v","speed"}: (H,W) float32；mask: (H,W) bool；fallback: 中心像元填充值 (n,)。
    返回 {"u","v","speed","std_u","std_v"} 均为 (n,) float32。
    """
    n = rows.size
    f32 = np.float32
    areas = {k: np.zeros(n, f32) for k in ("u", "v", "speed", "u2", "v2")}
    cnt = np.zeros(n, f32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            rr = np.clip(rows + dy, 0, height - 1)
            cc = np.clip(cols + dx, 0, width - 1)
            ok = mask[rr, cc]
            w = ok.astype(f32)
            uu = np.where(ok, values["u"][rr, cc], 0.0).astype(f32, copy=False)
            vv = np.where(ok, values["v"][rr, cc], 0.0).astype(f32, copy=False)
            ss = np.where(ok, values["speed"][rr, cc], 0.0).astype(f32, copy=False)
            areas["u"] += uu
            areas["v"] += vv
            areas["speed"] += ss
            areas["u2"] += uu * uu
            areas["v2"] += vv * vv
            cnt += w
    # 有效邻居数至少 1：用倒数权重求均值；否则回退中心填充值
    inv = np.zeros(n, f32)
    pos = cnt > 0
    inv[pos] = 1.0 / cnt[pos]
    out = {}
    u_m = areas["u"] * inv
    v_m = areas["v"] * inv
    s_m = areas["speed"] * inv
    out["u"] = np.where(pos, u_m, fallback["u"]).astype(f32, copy=False)
    out["v"] = np.where(pos, v_m, fallback["v"]).astype(f32, copy=False)
    out["speed"] = np.where(pos, s_m, fallback["speed"]).astype(f32, copy=False)
    var_u = np.maximum(areas["u2"] * inv - u_m * u_m, 0.0)
    var_v = np.maximum(areas["v2"] * inv - v_m * v_m, 0.0)
    out["std_u"] = np.where(pos, np.sqrt(var_u), 0.0).astype(f32, copy=False)
    out["std_v"] = np.where(pos, np.sqrt(var_v), 0.0).astype(f32, copy=False)
    return out


def _gather_hist(win: dict) -> dict:
    """把 window 字典裁剪/校验成最后一帧为 t 的 N_HISTORY 帧历史。"""
    if "u" not in win or "mask" not in win:
        raise KeyError("build_features 需要 win['u'], win['v'], win['speed'], win['mask']")
    u = np.asarray(win["u"])
    n = u.shape[0]
    if n < N_HISTORY:
        raise ValueError(f"build_features 需要 >= {N_HISTORY} 帧历史（当前 {n}）")
    sl = slice(n - N_HISTORY, n)
    out = {
        "u": np.asarray(win["u"][sl], np.float32),
        "v": np.asarray(win["v"][sl], np.float32),
        "speed": np.asarray(win["speed"][sl], np.float32),
        "mask": np.asarray(win["mask"][sl]).astype(bool, copy=False),
        "doy": np.asarray(win["doy"])[-N_HISTORY:].astype(np.float32),
    }
    return out


def build_features(win: dict, rows, cols) -> np.ndarray:
    """构造特征矩阵。

    Parameters
    ----------
    win : dict
        滑窗字典（旧→新），至少包含 ``u/v/speed`` (n,H,W)、``mask`` (n,H,W)、``doy`` (n,)；
        可选 ``latitude`` / ``longitude`` (H,W)。``n >= N_HISTORY``，只使用最后 N_HISTORY 帧。
    rows, cols : array-like[int]
        像元行/列索引（等长）。

    Returns
    -------
    np.ndarray, shape (len(rows), 34), float32
    """
    rows = np.asarray(rows, dtype=np.int64).ravel()
    cols = np.asarray(cols, dtype=np.int64).ravel()
    n_cells = rows.size
    if n_cells == 0:
        return np.zeros((0, N_FEATURES), np.float32)

    h = _gather_hist(win)
    u, v, sp, mk, doy = h["u"], h["v"], h["speed"], h["mask"], h["doy"]

    # 中心像元时间序列 (N_HISTORY, n_cells)
    U = u[:, rows, cols]
    V = v[:, rows, cols]
    S = sp[:, rows, cols]
    M = mk[:, rows, cols]
    Uf = _ffill_hist(U, M)
    Vf = _ffill_hist(V, M)
    Sf = _ffill_hist(S, M)

    # 邻域：t（最后一帧）与 t-1
    fb = {"u": Uf[-1], "v": Vf[-1], "speed": Sf[-1]}
    nb_t = _neighbor_stats({"u": u[-1], "v": v[-1], "speed": sp[-1]}, mk[-1],
                           rows, cols, fb, u.shape[1], u.shape[2])
    fb1 = {"u": Uf[-2], "v": Vf[-2], "speed": Sf[-2]}
    nb_t1 = _neighbor_stats({"u": u[-2], "v": v[-2], "speed": sp[-2]}, mk[-2],
                            rows, cols, fb1, u.shape[1], u.shape[2])

    feats: list[np.ndarray] = [
        # 中心像元滞后 t, t-1, t-2, t-3
        Uf[-1], Vf[-1], Sf[-1],
        Uf[-2], Vf[-2], Sf[-2],
        Uf[-3], Vf[-3], Sf[-3],
        Uf[-4], Vf[-4], Sf[-4],
        # 滚动均值 3 / 7
        Uf[-3:].mean(axis=0), Vf[-3:].mean(axis=0), Sf[-3:].mean(axis=0),
        Uf[-7:].mean(axis=0), Vf[-7:].mean(axis=0), Sf[-7:].mean(axis=0),
        # 邻域
        nb_t["u"], nb_t["v"], nb_t["speed"],
        nb_t1["u"], nb_t1["v"], nb_t1["speed"],
        nb_t["std_u"], nb_t["std_v"],
        # 趋势 (t - t-3)
        Uf[-1] - Uf[-4], Vf[-1] - Vf[-4], Sf[-1] - Sf[-4],
        # 季节
        np.sin(2.0 * np.pi * doy[-1]) * np.ones(n_cells, np.float32),
        np.cos(2.0 * np.pi * doy[-1]) * np.ones(n_cells, np.float32),
    ]

    # 位置
    lat = win.get("latitude")
    lon = win.get("longitude")
    if lat is not None and lon is not None:
        lat = np.asarray(lat, np.float32)
        lon = np.asarray(lon, np.float32)
        feats.append(np.asarray(lat[rows, cols], np.float32))
        feats.append(np.asarray(lon[rows, cols], np.float32))
    else:  # 兼容：无位置时填 0（训练/推理都走同一条路径）
        feats.append(np.zeros(n_cells, np.float32))
        feats.append(np.zeros(n_cells, np.float32))

    # 当日全局有效覆盖
    cov = float(np.asarray(win["mask"])[-1].mean()) if "mask" in win else 0.0
    feats.append(np.full(n_cells, cov, np.float32))

    X = np.column_stack(feats).astype(np.float32, copy=False)
    if X.shape[1] != N_FEATURES:  # pragma: no cover - 防御性
        raise RuntimeError(f"特征维度错误: {X.shape[1]} != {N_FEATURES}")
    return X


def build_features_multi(win: dict, day_offsets, rows, cols) -> np.ndarray:
    """批量构造多个"起报日"的特征（同一次 read_range 覆盖多天时使用，可选工具）。

    day_offsets[i] 表示以 win 内第 day_offsets[i] 帧为 t 的特征；rows/cols 为每行像元。
    """
    day_offsets = np.asarray(day_offsets, dtype=np.int64)
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    if day_offsets.size != rows.size:
        raise ValueError("day_offsets 与 rows/cols 长度需一致")
    out = np.empty((rows.size, N_FEATURES), np.float32)
    for t in np.unique(day_offsets):
        sel = day_offsets == t
        end = int(t) + 1
        sub = {k: win[k][end - N_HISTORY:end] if k in ("u", "v", "speed", "mask") else
               (win[k][end - N_HISTORY:end] if k == "doy" else win[k])
               for k in ("u", "v", "speed", "mask", "doy", "latitude", "longitude")}
        out[sel] = build_features(sub, rows[sel], cols[sel])
    return out


def feature_spec() -> dict:
    """特征说明（写入 models/ml/feature_spec.json）。"""
    return {
        "version": "1.0",
        "task": "next-frame (24h) sea-ice drift vector prediction",
        "target": ["u(t+1)", "v(t+1)"],
        "target_unit": "cm/s",
        "n_features": N_FEATURES,
        "n_history": N_HISTORY,
        "feature_names": FEATURE_NAMES,
        "groups": {
            "center_lags": ["u_t", "v_t", "speed_t", "u_t1", "v_t1", "speed_t1",
                            "u_t2", "v_t2", "speed_t2", "u_t3", "v_t3", "speed_t3"],
            "rolling_mean": ["u_mean3", "v_mean3", "speed_mean3",
                             "u_mean7", "v_mean7", "speed_mean7"],
            "neighborhood_3x3": ["nb_u_t", "nb_v_t", "nb_speed_t",
                                 "nb_u_t1", "nb_v_t1", "nb_speed_t1",
                                 "nb_std_u_t", "nb_std_v_t"],
            "trend": ["du_t3", "dv_t3", "dspeed_t3"],
            "season": ["sin_doy", "cos_doy"],
            "position_quality": ["latitude", "longitude", "valid_fraction"],
        },
        "missing_value_policy": {
            "center": "在 N_HISTORY 帧窗口内按有效 mask 前向填充；全无效填 0",
            "neighborhood": "只统计有效邻居；全无效回退中心填充值；边界 clip 复制",
            "valid_fraction": "当日全网格有效像元占比（mask.mean）",
        },
        "normalization": "特征直接在 cm/s 原始空间构建；Ridge 在 Pipeline 内 StandardScaler（仅 train 拟合），RF 不缩放",
        "doy_convention": "zarr 的 doy 为 (dayofyear-1)/days_in_year ∈ [0,1)，特征用 sin/cos(2π·doy)",
    }
