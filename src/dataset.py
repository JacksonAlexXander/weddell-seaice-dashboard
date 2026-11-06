"""zarr 训练数据加载：48 年日尺度海冰运动 (u, v) 滑窗样本。

数据来源: data/processed/weddell_icemotion_daily.zarr（datamaster 构建）
- /u, /v (T,116,158) float32 cm/s（无效像元为 NaN）
- /valid_mask (T,116,158) uint8
- /time (T,) datetime64[s], /doy (T,) float32, /month (T,) int16
- /stats/u_mean, u_std, v_mean, v_std (116,158)（1978–2017 训练段统计）
- /latitude, /longitude (116,158)

样本: 输入 T_in 天 -> 输出 T_out 天；按年切分 1978–2017 / 2018–2021 / 2022–2025。
预处理: 每像元 z-score（NaN->0）-> pad 到 (120,160) -> 可选平均池化降采样。
可选输入通道: sin/cos day-of-year（帮助模型捕捉季节循环）。
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

GRID_H, GRID_W = 116, 158
PAD_H, PAD_W = 120, 160
TRAIN_YEARS = (1978, 2017)
VAL_YEARS = (2018, 2021)
TEST_YEARS = (2022, 2025)
SPLITS = {"train": TRAIN_YEARS, "val": VAL_YEARS, "test": TEST_YEARS}


class WeddellMotionDataset(Dataset):
    """滑窗采样 zarr；num_workers=0 时单进程持有 zarr 句柄（线程读）。"""

    def __init__(self, zarr_path: str, split: str = "train", *, tin: int = 6,
                 tout: int = 6, stride: int = 3, downsample: int = 1,
                 add_doy: bool = True, synth: bool = False,
                 seed: int = 42, pad: bool = True):
        assert split in SPLITS, split
        self.split = split
        self.tin, self.tout, self.stride = tin, tout, stride
        self.ds = downsample
        self.add_doy = add_doy
        self.pad = pad
        self.synth = synth
        self.rng = np.random.default_rng(seed)
        # pad 目标：原生 116x158 -> 补齐到 8*ds 的倍数，保证 encoder 三次
        # 2x 下采样后各层尺寸仍为整数（ds=1 -> 120x160；ds=2 -> 128x160 -> 池化 64x80）
        self.pad_h = ((GRID_H + 8 * downsample - 1) // (8 * downsample)) * 8 * downsample
        self.pad_w = ((GRID_W + 8 * downsample - 1) // (8 * downsample)) * 8 * downsample

        if synth:
            self.n_frames = 48 * 366
            self._synth = True
            self._u = None
            self.stats = {"u_mean": np.zeros((GRID_H, GRID_W), np.float32),
                          "u_std": np.ones((GRID_H, GRID_W), np.float32),
                          "v_mean": np.zeros((GRID_H, GRID_W), np.float32),
                          "v_std": np.ones((GRID_H, GRID_W), np.float32)}
        else:
            import zarr
            self.store = zarr.open_group(zarr_path, mode="r")
            self._u = self.store["u"]
            self._v = self.store["v"]
            self._mask = self.store["valid_mask"]
            self._time = self.store["time"][:]
            self._doy = self.store["doy"][:]
            self._month = self.store["month"][:]
            self.n_frames = self._u.shape[0]
            self.stats = {k: self.store["stats"][k][:] for k in
                          ("u_mean", "u_std", "v_mean", "v_std")}
        # 划分样本起点索引（按年过滤）
        years = self._years()
        lo, hi = SPLITS[split]
        idx = np.where((years >= lo) & (years <= hi))[0]
        self.idx = idx[idx <= self.n_frames - (tin + tout)]
        self._synth = synth

    def _years(self):
        if self.synth:
            t = np.datetime64("1978-10-25", "D") + np.arange(self.n_frames)
            t = t.astype("datetime64[Y]").astype(int) + 1970
        else:
            t = self._time.astype("datetime64[Y]").astype(int) + 1970
        return np.asarray(t)

    def __len__(self):
        return max(0, (len(self.idx) - 1) // self.stride + 1)

    def _frame(self, t0: int, t1: int):
        """读 u/v/mask/doy 帧并做标准化 + pad + 降采样。"""
        if self.synth:
            u = self.rng.standard_normal((t1 - t0, GRID_H, GRID_W)).astype(np.float32)
            v = self.rng.standard_normal((t1 - t0, GRID_H, GRID_W)).astype(np.float32)
            mask = (self.rng.random((t1 - t0, GRID_H, GRID_W)) > 0.25).astype(np.float32)
            doy = np.linspace(0, 1, t1 - t0, dtype=np.float32)
        else:
            u = np.nan_to_num(self._u[t0:t1], nan=0.0)
            v = np.nan_to_num(self._v[t0:t1], nan=0.0)
            mask = self._mask[t0:t1].astype(np.float32)
            doy = self._doy[t0:t1]
            u = (u - self.stats["u_mean"]) / (self.stats["u_std"] + 1e-6)
            v = (v - self.stats["v_mean"]) / (self.stats["v_std"] + 1e-6)
        if self.pad:
            ph, pw = (self.pad_h - GRID_H) // 2, (self.pad_w - GRID_W) // 2
            u = np.pad(u, ((0, 0), (ph, self.pad_h - GRID_H - ph), (pw, self.pad_w - GRID_W - pw)))
            v = np.pad(v, ((0, 0), (ph, self.pad_h - GRID_H - ph), (pw, self.pad_w - GRID_W - pw)))
            mask = np.pad(mask, ((0, 0), (ph, self.pad_h - GRID_H - ph), (pw, self.pad_w - GRID_W - pw)))
        if self.ds > 1:
            u = u.reshape(t1 - t0, u.shape[1] // self.ds, self.ds, u.shape[2] // self.ds, self.ds).mean((2, 4))
            v = v.reshape(t1 - t0, v.shape[1] // self.ds, self.ds, v.shape[2] // self.ds, self.ds).mean((2, 4))
            mask = (mask.reshape(t1 - t0, mask.shape[1] // self.ds, self.ds, mask.shape[2] // self.ds, self.ds).mean((2, 4)) > 0.5).astype(np.float32)
        return u, v, mask, doy

    def __getitem__(self, i: int):
        t0 = int(self.idx[i * self.stride])
        u, v, mask, doy = self._frame(t0, t0 + self.tin + self.tout)
        xu, xv, xm, xd = u[:self.tin], v[:self.tin], mask[:self.tin], doy[:self.tin]
        yu, yv, ym = u[self.tin:], v[self.tin:], mask[self.tin:]
        chans = [torch.from_numpy(xu), torch.from_numpy(xv)]
        if self.add_doy:
            a = 2 * np.pi * xd[:, None, None]
            H, W = xu.shape[1], xu.shape[2]
            ones = np.ones((1, H, W), np.float32)
            chans += [torch.from_numpy(np.sin(a).astype(np.float32) * ones),
                      torch.from_numpy(np.cos(a).astype(np.float32) * ones)]
        x = torch.stack(chans, dim=1)                       # (T_in, C_in, H, W)
        y = torch.stack([torch.from_numpy(yu), torch.from_numpy(yv)], dim=1)
        ymask = torch.from_numpy(ym).unsqueeze(1)           # (T_out, 1, H, W)
        return {"x": x, "y": y, "mask": ymask, "t0": t0}

    def _aligned_stats(self, u_or_v: int):
        """返回与 pad/降采样后网格对齐的 (mean, std) 张量 (H, W)。"""
        key = "u" if u_or_v == 0 else "v"
        mean = self.stats[f"{key}_mean"].astype(np.float32)
        std = self.stats[f"{key}_std"].astype(np.float32)
        if self.pad:
            ph, pw = (self.pad_h - GRID_H) // 2, (self.pad_w - GRID_W) // 2
            mean = np.pad(mean, ((ph, self.pad_h - GRID_H - ph), (pw, self.pad_w - GRID_W - pw)))
            std = np.pad(std, ((ph, self.pad_h - GRID_H - ph), (pw, self.pad_w - GRID_W - pw)))
        if self.ds > 1:
            def pool(a):
                return a.reshape(a.shape[0] // self.ds, self.ds,
                                 a.shape[1] // self.ds, self.ds).mean((1, 3))
            mean, std = pool(mean), pool(std)
        return torch.from_numpy(mean), torch.from_numpy(std)

    def unnorm(self, y_norm: torch.Tensor, u_or_v: int) -> torch.Tensor:
        """把标准化场还原为 cm/s（按每像元 mean/std）。"""
        mean, std = self._aligned_stats(u_or_v)
        s = std.to(y_norm.device).unsqueeze(0)
        m = mean.to(y_norm.device).unsqueeze(0)
        return y_norm * s + m
