"""自回归场推演与轨迹评估的共用工具（供 evaluate/predict 脚本调用）。

约定：
- 所有返回场为原生网格 (H=116, W=158) 的 cm/s 张量 (T_out, H, W)
- 模型输入为 dataset 的标准化窗口；输出经 unnorm 还原后裁掉 pad
"""

from __future__ import annotations

import numpy as np
import torch


def unpad_native(t, pad_h=120, pad_w=160, h=116, w=158):
    """把 (...,pad_h,pad_w) 裁回原生网格 (...,h,w)。"""
    ph, pw = (pad_h - h) // 2, (pad_w - w) // 2
    return t[..., ph:ph + h, pw:pw + w]


@torch.no_grad()
def rollout_fields(model, dataset, start_idx: int, horizon: int, device,
                   window_days: int | None = None,
                   residual: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """自回归推演未来 horizon 天的 u/v 场（cm/s，原生网格）。

    start_idx: 数据集帧索引，表示"最后观测帧"
    每轮预测 tout 帧，然后把预测帧滚动进输入窗口继续预测。
    """
    tin, tout = dataset.tin, dataset.tout
    window_days = window_days or tout
    t0 = start_idx - (tin - 1)
    win = dataset._frame(t0, t0 + tin)
    cur_u = torch.from_numpy(win[0])
    cur_v = torch.from_numpy(win[1])
    d0 = float(win[3][-1])
    pred_u, pred_v = [], []
    n_rounds = int(np.ceil(horizon / window_days))
    for r in range(n_rounds):
        n_step = min(window_days, horizon - r * window_days)
        H, W = cur_u.shape[1], cur_u.shape[2]
        k0 = r * window_days - (tin - 1)
        doy = (d0 + (k0 + np.arange(tin)) / 365.25) % 1.0
        a = 2 * np.pi * doy[:, None, None]
        ones = np.ones((1, H, W), np.float32)
        chans = [cur_u, cur_v,
                 torch.from_numpy(np.sin(a).astype(np.float32) * ones),
                 torch.from_numpy(np.cos(a).astype(np.float32) * ones)]
        x = torch.stack(chans, 1).unsqueeze(0).to(device)
        out = model(x)[0].cpu()                    # 标准化输出
        if residual:                               # 残差模型：增量 + 最后一帧
            out = out + x[0, -1:, :2].cpu().expand_as(out)
        cur_u = torch.cat([cur_u[window_days:], out[:, 0]], 0)
        cur_v = torch.cat([cur_v[window_days:], out[:, 1]], 0)
        pu = dataset.unnorm(out[:n_step, 0], 0)
        pv = dataset.unnorm(out[:n_step, 1], 1)
        pred_u.append(unpad_native(pu))
        pred_v.append(unpad_native(pv))
    return torch.cat(pred_u, 0)[:horizon], torch.cat(pred_v, 0)[:horizon]


def true_fields(dataset, start_idx: int, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
    """真值未来场（cm/s，原生网格）。注意：zarr 内已是 cm/s 原始值（无效像元=0）。"""
    tu = torch.from_numpy(np.nan_to_num(
        dataset._u[start_idx + 1:start_idx + 1 + horizon], nan=0.0)).float()
    tv = torch.from_numpy(np.nan_to_num(
        dataset._v[start_idx + 1:start_idx + 1 + horizon], nan=0.0)).float()
    return tu, tv


def persistence_fields(dataset, start_idx: int, horizon: int, window: int = 1):
    """持续性基线：未来 horizon 天恒定 = 最后 window 天平均场（cm/s，原生网格）。"""
    u = np.nan_to_num(dataset._u[start_idx - window + 1:start_idx + 1], nan=0.0).mean(0)
    v = np.nan_to_num(dataset._v[start_idx - window + 1:start_idx + 1], nan=0.0).mean(0)
    tu = torch.from_numpy(np.asarray(u)).float()
    tv = torch.from_numpy(np.asarray(v)).float()
    return tu.unsqueeze(0).expand(horizon, -1, -1), tv.unsqueeze(0).expand(horizon, -1, -1)


def climatology_fields(dataset, start_idx: int, horizon: int,
                       clim_npz: str = "data/interim/climatology_monthly_mean.npz"):
    """气候学基线：未来各天 = 该月 1978–2017 月平均 u/v（cm/s，原生网格）。"""
    arr = np.load(clim_npz)["monthly_mean"]          # (12,2,H,W) cm/s
    months = dataset._month[start_idx + 1:start_idx + 1 + horizon]  # 1..12
    u = torch.from_numpy(arr[months - 1, 0].astype(np.float32))
    v = torch.from_numpy(arr[months - 1, 1].astype(np.float32))
    return u, v


def seed_points(dataset, start_idx: int, n: int = 200, seed: int = 0):
    """在起始观测帧的有效像元中随机抽 n 个轨迹种子点（原生网格像素坐标 (col,row)）。"""
    valid = np.argwhere(dataset._mask[start_idx] > 0)
    rng = np.random.default_rng(seed)
    k = min(n, len(valid))
    picks = valid[rng.choice(len(valid), k, replace=False)]
    return torch.from_numpy(picks[:, [1, 0]].astype(np.float32))
