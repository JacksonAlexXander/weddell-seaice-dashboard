"""威德尔海海冰运动矢量场时空预测模型。

主模型 SeaIceMotionNet：SimVP 风格的 encoder-translator-decoder 预测架构
（全卷积 + 门控时空注意力），translator 内的时间注意力模块与 TAU 的
temporal unit 同思路，可用散度正则（训练脚本 --div_reg）做物理约束。

参考文献（CCF-A）：
- SimVP: Simpler yet Better Video Prediction, CVPR 2022
- TAU: Temporal Attention Unit, CVPR 2023
- ConvLSTM: Convolutional LSTM Network, NeurIPS 2015
- OpenSTL 开源实现（SimVP/gSTA 模块设计参考）

输入: (B, T_in, C_in, H, W)，C_in = 2 (u, v) 或 4 (+sin/cos day-of-year)
输出: (B, T_out, 2, H, W) 即未来 T_out 天的 u/v 场（cm/s，已按每像元标准化）
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- 基础模块


class ConvNormAct(nn.Module):
    """Conv2d + GroupNorm + LeakyReLU（保持分辨率或 stride 下采样）。"""

    def __init__(self, cin: int, cout: int, k: int = 3, stride: int = 1,
                 groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, padding=k // 2,
                              groups=groups, bias=False)
        ng = cout // 16
        self.norm = nn.GroupNorm(max(1, ng), cout)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class TemporalAttention(nn.Module):
    """沿时间维的 1x1-conv 注意力（gSTA 的时间支 / TAU temporal unit 同思路）。

    x: (B, T, C, H, W) -> 把 T 放到最后做共享 Linear(T->T)，残差连接。
    参数量 O(T^2)，可忽略；对海冰漂移的逐日演化显式建模时间交互。
    """

    def __init__(self, t: int):
        super().__init__()
        self.w = nn.Parameter(torch.empty(t, t))
        nn.init.xavier_uniform_(self.w)

    def forward(self, x):
        B, T, C, H, W = x.shape
        y = x.permute(0, 2, 3, 4, 1).reshape(-1, T)  # (B*C*H*W, T)
        y = y @ self.w
        y = y.view(B, C, H, W, T).permute(0, 4, 1, 2, 3)
        return x + y


class SpatialGate(nn.Module):
    """空间门控注意力（gSTA 空间支）：深度可分 3x3 分支产生门控，逐通道相乘。"""

    def __init__(self, c: int):
        super().__init__()
        c1 = c // 2
        self.dw = nn.Conv2d(c1, c1, 3, padding=1, groups=c1, bias=False)
        self.norm = nn.GroupNorm(max(1, c1 // 16), c1)
        self.fuse = nn.Conv2d(c, c, 1, bias=False)

    def forward(self, x):
        # 支持 (B,C,H,W) 或 (B,T,C,H,W)：折叠时间维到 batch 后按通道门控
        squeezed = x.dim() == 4
        if not squeezed:
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)
        a, b = torch.chunk(x, 2, dim=1)
        g = torch.sigmoid(self.norm(self.dw(a)))
        y = self.fuse(torch.cat([g * b, a], dim=1))
        if not squeezed:
            y = y.reshape(B, T, *y.shape[1:])
            x = x.reshape(B, T, *x.shape[1:])
        return x + y


class MetaBlock(nn.Module):
    """translator 基本块：时间注意力 + 空间门控注意力（残差）。"""

    def __init__(self, c: int, t: int):
        super().__init__()
        self.ta = TemporalAttention(t)
        self.sg = SpatialGate(c)

    def forward(self, x):
        return self.sg(self.ta(x))


# ---------------------------------------------------------------- SimVP 类主模型


class Encoder(nn.Module):
    """3 次 2x 下采样（1/8 分辨率），输出每层 skip 供 decoder 拼接。"""

    def __init__(self, cin: int, chid: int, n_s: int):
        super().__init__()
        self.d1 = ConvNormAct(cin, chid, stride=2)
        self.d2 = ConvNormAct(chid, 2 * chid, stride=2)
        self.d3 = ConvNormAct(2 * chid, 2 * chid, stride=2)
        self.tail = nn.Sequential(
            *[ConvNormAct(2 * chid, 2 * chid) for _ in range(n_s)])

    def forward(self, x):
        s1 = self.d1(x)
        s2 = self.d2(s1)
        s3 = self.d3(s2)
        return [s1, s2, s3], self.tail(s3)


class Decoder(nn.Module):
    """3 次 2x 上采样并拼接对应层 skip，输出 T_out*2 通道预测场。"""

    def __init__(self, cout: int, chid: int, n_s: int, cin: int | None = None):
        super().__init__()
        cin = cin if cin is not None else 2 * chid
        self.head = nn.Sequential(
            ConvNormAct(cin, 2 * chid),
            *[ConvNormAct(2 * chid, 2 * chid) for _ in range(n_s)])
        self.up3 = ConvNormAct(2 * chid, 2 * chid)
        self.c3 = ConvNormAct(4 * chid, 2 * chid)
        self.up2 = ConvNormAct(2 * chid, chid)
        self.c2 = ConvNormAct(2 * chid, chid)
        self.up1 = ConvNormAct(chid, chid // 2)
        self.c1 = ConvNormAct(chid // 2, chid // 2)
        self.out = nn.Conv2d(chid // 2, cout, 1)

    def forward(self, z, skips):
        # skips = [s1(1/2), s2(1/4), s3(1/8)]；z 在 1/8 尺度
        z = self.head(z)
        z = self.c3(torch.cat([self.up3(F.interpolate(
            z, scale_factor=2, mode="bilinear", align_corners=False)), skips[1]], 1))
        z = self.c2(torch.cat([self.up2(F.interpolate(
            z, scale_factor=2, mode="bilinear", align_corners=False)), skips[0]], 1))
        z = self.c1(self.up1(F.interpolate(
            z, scale_factor=2, mode="bilinear", align_corners=False)))
        return self.out(z)


class SeaIceMotionNet(nn.Module):
    """SimVP 风格海冰运动预测模型（encoder-translator-decoder）。

    forward 输入 (B, T_in, C_in, H, W)（帧已拼进通道），
    输出 (B, T_out, 2, H, W)。H, W 需能被 8 整除（训练数据已 pad）。
    """

    def __init__(self, cin: int, tin: int, tout: int, chid: int = 64,
                 n_s: int = 2, n_t: int = 4):
        super().__init__()
        self.tin, self.tout, self.chid = tin, tout, chid
        self.enc = Encoder(cin * tin, chid, n_s)
        # 学习式帧扩展：把编码器底部特征 (B,2chid,h,w) 扩展成 T_out 个帧特征
        self.expand = nn.Conv2d(2 * chid, tout * 2 * chid, 1)
        self.mid = nn.Sequential(
            *[MetaBlock(2 * chid, tout) for _ in range(n_t)])
        self.dec = Decoder(tout * 2, chid, n_s, cin=tout * 2 * chid)

    def forward(self, x):
        # x: (B, T_in, C_in, H, W) —— 内部把输入帧拼到通道维
        B, T, C, H, W = x.shape
        skips, z = self.enc(x.reshape(B, T * C, H, W))   # z: (B, 2chid, h, w)
        # 帧扩展 + 时间注意力: (B, T_out, 2chid, h, w)
        z5 = self.expand(z).reshape(B, self.tout, 2 * self.chid,
                                    z.shape[2], z.shape[3])
        z5 = self.mid(z5)
        z5 = z5.reshape(B, self.tout * 2 * self.chid, z.shape[2], z.shape[3])
        out = self.dec(z5, skips)          # (B, T_out*2, H, W)
        return out.reshape(B, self.tout, 2, H, W)


# ---------------------------------------------------------------- ConvLSTM 基线


class ConvLSTMCell(nn.Module):
    """标准 ConvLSTM 单元（NeurIPS 2015）。"""

    def __init__(self, cin: int, chid: int, k: int = 5):
        super().__init__()
        self.chid = chid
        self.gx = nn.Conv2d(cin, 4 * chid, k, padding=k // 2, bias=False)
        self.gh = nn.Conv2d(chid, 4 * chid, k, padding=k // 2, bias=False)

    def forward(self, x, state):
        h, c = state
        gates = self.gx(x) + self.gh(h)
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, (h, c)


class ConvLSTM(nn.Module):
    """两层 ConvLSTM，输入 (B, T_in, 2, H, W)，逐帧更新后滚推 T_out 步。"""

    def __init__(self, chid: int = 64, k: int = 5):
        super().__init__()
        self.cell1 = ConvLSTMCell(2, chid, k)
        self.cell2 = ConvLSTMCell(chid, chid, k)
        self.head = nn.Conv2d(chid, 2, 1)
        self.chid = chid

    def forward(self, x, tout: int):
        x = x[:, :, :2]          # ConvLSTM 只用 u/v 通道（doy 通道不参与递推）
        B, T, _, H, W = x.shape
        h1 = x.new_zeros(B, self.chid, H, W)
        c1 = x.new_zeros(B, self.chid, H, W)
        h2 = x.new_zeros(B, self.chid, H, W)
        c2 = x.new_zeros(B, self.chid, H, W)
        for t in range(T):
            h1, (h1, c1) = self.cell1(x[:, t], (h1, c1))
            h2, (h2, c2) = self.cell2(h1, (h2, c2))
        outs = []
        inp = x[:, -1]
        for _ in range(tout):
            h1, (h1, c1) = self.cell1(inp, (h1, c1))
            h2, (h2, c2) = self.cell2(h1, (h2, c2))
            pred = self.head(h2)
            outs.append(pred)
            inp = pred
        return torch.stack(outs, dim=1)


# ---------------------------------------------------------------- 简单基线


class Persistence(nn.Module):
    """持续性预报：把最后一帧输入复制 T_out 次。"""

    def forward(self, x, tout: int):
        return x[:, -1:].expand(-1, tout, -1, -1, -1).contiguous()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(name: str, *, tin: int, tout: int, cin: int = 4,
                chid: int = 64, **kw) -> nn.Module:
    """按名称构建模型。"""
    if name == "simvp":
        return SeaIceMotionNet(cin=cin, tin=tin, tout=tout, chid=chid,
                               n_s=kw.get("n_s", 2), n_t=kw.get("n_t", 4))
    if name == "convlstm":
        return ConvLSTM(chid=kw.get("chid", 64))
    if name == "persistence":
        return Persistence()
    raise ValueError(f"unknown model {name}")
