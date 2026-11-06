#!/usr/bin/env python3
"""海冰轨迹推演与误差评估工具（torch 实现，供训练/评估代码直接 import）。

坐标约定（已对源数据实测确认）
------------------------------
- 网格：威德尔海 25 km EASE-Grid South 子集，**H=116 行 × W=158 列**。
- 像素坐标 ``(x, y)``：**x = 列号（0..W-1），y = 行号（0..H-1）**，均指像元中心，
  与 NumPy/张量索引 ``array[..., y, x]`` 一致。
- 方向实测（对 NSIDC-0116 1978 年文件校验，全网格严格单调）：
    * **列号增大 → 经度增大（西 → 东）**；
    * **行号增大 → 纬度增大（南 → 北）**。
  即图像按常规"北上右东"方向放置，但注意威德尔区域在 EASE 投影下格网北与
  地理北有偏转，u/v 只是大致对应向东/向北。
- 场变量 u/v：**cm/s**，为 EASE-Grid 投影 x/y 分量（u 沿列方向、v 沿行方向），
  不是严格的东西/南北分量。场为 (T, H, W)，T 是"连续 valid 日"序列——
  中间可能跳过无记录日，因此 dt_days 计的是帧步数，真实日历间隔可能更大。
- 无效像元（valid_mask=0）在 zarr 中 u/v=0：粒子漂入开放水域后速度≈0 而停滞；
  起始点应选在有效像元内，必要时把场与 valid_mask 相乘以抑制假流场。
- 像元大小：任务约定按 25.0 km 换算（x_res_km=25.0）；实际 EASE 格距为
  25067.525 m，相对误差约 0.27%。

采样约定
--------
双线性采样用 ``torch.nn.functional.grid_sample``（``align_corners=False``）：
标准化坐标 [-1, 1] 中，-1 对应首像元边缘、+1 对应末像元边缘，像元中心
``(col, row)`` 映射为 ``((col+0.5)/W*2-1, (row+0.5)/H*2-1)``；grid 的最后一维
顺序是 **(x, y)**（x 对应列/W，y 对应行/H）。``padding_mode="border"``：
越界点取边界像元值。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "advect_torch",
    "grid_to_geo",
    "endpoint_error_km",
    "path_error_km",
    "mean_speed_km_per_day",
    "speed_error_km_per_day",
]


def _pix_to_norm(pts: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """像素坐标 (...,2)（x=列, y=行）→ grid_sample 标准化坐标 (...,2)（顺序 x,y）。"""
    xn = (pts[..., 0] + 0.5) * (2.0 / w) - 1.0
    yn = (pts[..., 1] + 0.5) * (2.0 / h) - 1.0
    return torch.stack([xn, yn], dim=-1)


def advect_torch(
    field_u: torch.Tensor,
    field_v: torch.Tensor,
    start_pts: torch.Tensor,
    n_steps: int,
    dt_days: float = 1.0,
    x_res_km: float = 25.0,
) -> torch.Tensor:
    """对给定速度场做半拉格朗日轨迹推演（逐帧前向欧拉 + 双线性采样）。

    参数
    ----
    field_u, field_v : (T, H, W) torch 张量
        连续日序列的 u/v 场，单位 cm/s（EASE-Grid x/y 分量），T >= n_steps。
        无效像元应为 0（与 /valid_mask 区分），否则推演会使用 NaN。
    start_pts : (N, 2)
        起始点像素坐标（x=列, y=行），可含小数（亚像元精度）。
    n_steps : int
        推演步数；返回路径长度 n_steps+1（含起点）。
    dt_days : float
        每步跨过的帧数（天），默认 1.0。
    x_res_km : float
        像元边长（km），默认 25.0。位移换算：
        ``u_cells = u_cm_s * 86400 * dt_days / (x_res_km * 1000 * 100)``。

    返回
    ----
    path : (N, n_steps+1, 2) torch 张量（float32/与 start_pts 同 dtype）
        像素坐标轨迹，path[:, 0] == start_pts；path[:, t] 为第 t 步后位置。
        （t 步使用 field[t] 的速度，要求 T >= n_steps。）
    """
    field_u = torch.as_tensor(field_u)
    field_v = torch.as_tensor(field_v)
    start_pts = torch.as_tensor(start_pts)
    if field_u.dim() != 3 or field_u.shape != field_v.shape:
        raise ValueError("field_u/field_v 必须为 (T,H,W) 且形状一致")
    if start_pts.dim() != 2 or start_pts.shape[1] != 2:
        raise ValueError("start_pts 必须为 (N,2)")
    if n_steps < 0:
        raise ValueError("n_steps 不能为负")
    t_dim, h, w = field_u.shape
    if n_steps > t_dim:
        raise ValueError(f"n_steps={n_steps} 超过场帧数 T={t_dim}")

    out_dtype = start_pts.dtype if start_pts.is_floating_point() else torch.float32
    fu = field_u.float()
    fv = field_v.float()
    n_pts = start_pts.shape[0]
    scale = 86400.0 * dt_days / (x_res_km * 1000.0 * 100.0)  # cm/s -> 像元/步

    path = torch.empty((n_steps + 1, n_pts, 2), dtype=torch.float32, device=start_pts.device)
    pts = start_pts.float().clone()
    path[0] = pts
    for step in range(n_steps):
        grid = _pix_to_norm(pts, h, w).view(n_pts, 1, 1, 2)
        u = F.grid_sample(
            fu[step].view(1, 1, h, w).expand(n_pts, 1, h, w),
            grid, mode="bilinear", align_corners=False, padding_mode="border",
        ).reshape(n_pts)
        v = F.grid_sample(
            fv[step].view(1, 1, h, w).expand(n_pts, 1, h, w),
            grid, mode="bilinear", align_corners=False, padding_mode="border",
        ).reshape(n_pts)
        pts = pts + torch.stack([u, v], dim=-1) * scale
        path[step + 1] = pts
    return path.permute(1, 0, 2).to(dtype=out_dtype)  # (N, n_steps+1, 2)


def grid_to_geo(
    path_pts: torch.Tensor,
    lon2d: torch.Tensor,
    lat2d: torch.Tensor,
) -> torch.Tensor:
    """把像素坐标轨迹 (N, T, 2) 转成经纬度轨迹 (N, T, 2)（双线性采样）。

    参数
    ----
    path_pts : (N, T, 2) 像素坐标（x=列, y=行，可小数）。
    lon2d, lat2d : (H, W) 经/纬度场（度，与速度场同一网格，如 zarr 的
        /longitude、/latitude；numpy 数组亦可，内部转 torch）。

    返回
    ----
    geo : (N, T, 2) torch 张量，[..., 0]=经度、[..., 1]=纬度（度）。
    """
    path = torch.as_tensor(path_pts)
    lon2d = torch.as_tensor(lon2d)
    lat2d = torch.as_tensor(lat2d)
    if path.dim() != 3 or path.shape[2] != 2:
        raise ValueError("path_pts 必须为 (N,T,2)")
    if lon2d.dim() != 2 or lon2d.shape != lat2d.shape:
        raise ValueError("lon2d/lat2d 必须为 (H,W) 且形状一致")
    h, w = lon2d.shape
    n_pts, n_t = path.shape[0], path.shape[1]
    grid = _pix_to_norm(path.float(), h, w).unsqueeze(2)  # (N,T,1,2)，H_out=T, W_out=1
    base_lon = lon2d.float().view(1, 1, h, w).expand(n_pts, 1, h, w)
    base_lat = lat2d.float().view(1, 1, h, w).expand(n_pts, 1, h, w)
    lon = F.grid_sample(base_lon, grid, mode="bilinear", align_corners=False,
                        padding_mode="border").reshape(n_pts, n_t)
    lat = F.grid_sample(base_lat, grid, mode="bilinear", align_corners=False,
                        padding_mode="border").reshape(n_pts, n_t)
    out = torch.stack([lon, lat], dim=-1)
    return out.to(dtype=path.dtype if path.is_floating_point() else torch.float32)


# ---------------------------------------------------------------- 误差指标
def _as_pair(pred: torch.Tensor, true: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    p = torch.as_tensor(pred)
    t = torch.as_tensor(true)
    if p.shape != t.shape or p.dim() != 3 or p.shape[2] != 2:
        raise ValueError("pred/true 必须为形状相同的 (N,T,2) 轨迹")
    return p.float(), t.float()


def endpoint_error_km(pred_path: torch.Tensor, true_path: torch.Tensor,
                      km_per_cell: float = 25.0) -> torch.Tensor:
    """终点误差：末点欧氏距离（像元）×km_per_cell，返回每轨迹 (N,) km。"""
    p, t = _as_pair(pred_path, true_path)
    return torch.norm(p[:, -1] - t[:, -1], dim=-1) * km_per_cell


def path_error_km(pred_path: torch.Tensor, true_path: torch.Tensor,
                  km_per_cell: float = 25.0) -> torch.Tensor:
    """整条路径平均误差：逐点欧氏距离在时间维平均（含起点），返回每轨迹 (N,) km。"""
    p, t = _as_pair(pred_path, true_path)
    return torch.norm(p - t, dim=-1).mean(dim=1) * km_per_cell


def mean_speed_km_per_day(path: torch.Tensor, km_per_cell: float = 25.0,
                          dt_days: float = 1.0) -> torch.Tensor:
    """轨迹平均日位移速度（km/天）：各步位移取模平均后 /dt_days，返回每轨迹 (N,)。"""
    p = torch.as_tensor(path)
    if p.dim() != 3 or p.shape[2] != 2:
        raise ValueError("path 必须为 (N,T,2)")
    step = torch.norm(p[:, 1:] - p[:, :-1], dim=-1).float()  # (N, T-1) 像元
    return step.mean(dim=1) * km_per_cell / dt_days


def speed_error_km_per_day(pred_path: torch.Tensor, true_path: torch.Tensor,
                           km_per_cell: float = 25.0, dt_days: float = 1.0) -> torch.Tensor:
    """速度误差：预测与真实轨迹的平均日位移速度之差的绝对值（km/天），返回每轨迹 (N,)。"""
    return (mean_speed_km_per_day(pred_path, km_per_cell, dt_days)
            - mean_speed_km_per_day(true_path, km_per_cell, dt_days)).abs()


# ------------------------------------------------------------------- 自检
if __name__ == "__main__":  # pragma: no cover
    import numpy as np

    # 1) 常数场：u=1 cm/s, v=0 → 每步位移 86400/2.5e6 = 0.03456 像元
    T, H, W = 16, 8, 10
    fu = torch.ones(T, H, W) * 1.0
    fv = torch.zeros(T, H, W)
    starts = torch.tensor([[0.0, 0.0], [4.5, 3.0], [9.0, 7.0]])
    path = advect_torch(fu, fv, starts, n_steps=10)
    expected_dx = 86400.0 / (25.0 * 1000.0 * 100.0) * 10
    assert torch.allclose(path[:, -1, 0] - starts[:, 0], torch.full_like(starts[:, 0], expected_dx), atol=1e-6), path
    assert torch.allclose(path[:, -1, 1], starts[:, 1], atol=1e-6)
    assert torch.equal(path[:, 0], starts.float())

    # 2) 整数像元点采样 == 该像元值；中点 (0.5,0.5) 双线性 == 四邻均值
    field = torch.arange(H * W, dtype=torch.float32).reshape(1, H, W)
    pts = torch.tensor([[2.0, 3.0], [0.5, 0.5], [9.99, 0.0]])
    grid = _pix_to_norm(pts, H, W).view(3, 1, 1, 2)
    got = F.grid_sample(field.expand(3, 1, H, W), grid, align_corners=False,
                        padding_mode="border").reshape(3)
    assert torch.allclose(got[0], field[0, 3, 2]), (got[0], field[0, 3, 2])
    assert torch.allclose(got[1], (field[0, 0, 0] + field[0, 0, 1] + field[0, 1, 0] + field[0, 1, 1]) / 4), got[1]

    # 3) grid_to_geo：整数像元 → 恰好返回 lon/lat 该像元值
    lon = torch.arange(W, dtype=torch.float32).repeat(H, 1)
    lat = (torch.arange(H, dtype=torch.float32) * 10).unsqueeze(1).repeat(1, W)
    gpath = torch.tensor([[[2.0, 3.0], [5.0, 1.0]], [[0.0, 0.0], [9.0, 7.0]]])
    geo = grid_to_geo(gpath, lon, lat)
    assert torch.allclose(geo[0, 0], torch.tensor([lon[3, 2], lat[3, 2]])), geo[0, 0]
    assert torch.allclose(geo[1, 1], torch.tensor([lon[7, 9], lat[7, 9]])), geo[1, 1]

    # 4) 误差指标
    p = torch.tensor([[[0., 0.], [1., 0.], [2., 0.]], [[0., 0.], [0., 2.], [0., 4.]]])
    t = torch.tensor([[[0., 0.], [1., 0.], [3., 0.]], [[0., 0.], [0., 2.], [0., 2.]]])
    assert torch.allclose(endpoint_error_km(p, t), torch.tensor([25.0, 50.0]))
    assert torch.allclose(path_error_km(p, t), torch.tensor([25.0 / 3, 50.0 / 3]))
    assert torch.allclose(speed_error_km_per_day(p, t), torch.tensor([12.5, 25.0]))
    print("trajectory.py 自检全部通过")
