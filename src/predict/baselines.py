"""零训练基线：持续性（persistence）与气候态（climatology）。

- `PersistencePredictor`：下一帧漂移 = 当前帧（最后观测帧）值；当前像元无效时按 zarr 零填充约定取 0。
- `ClimatologyPredictor`：1978–2017 逐月平均 u/v 场（`data/processed/dashboard/climatology.npz`），
  预测目标帧所在月份的气候态。artifact 缺失时回退到项目既有
  `data/interim/climatology_monthly_mean.npz`（形状 (12,2,H,W)），保证引擎在数据 artifact
  尚未生成时也可用。

两个基线都不需要训练，也不产生模型文件；`PredictEngine` 通过它们提供 `persistence` /
`climatology` 两个 model_id。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import ARTIFACT_DIR, GRID_H, GRID_W, ROOT, resolve_path


class PersistencePredictor:
    """下一帧 = 当前帧。"""

    model_id = "persistence"

    def predict_point(self, store, last_idx: int, row: int, col: int) -> tuple[float, float]:
        """返回 (u, v)；像元在当前帧无效时按 zarr 约定返回 (0, 0)。"""
        f = store.frame(int(last_idx))
        return float(f["u"][row, col]), float(f["v"][row, col])

    def predict_field(self, store, last_idx: int) -> dict:
        """整场 = 当前帧 u/v，无效像元置 NaN。"""
        f = store.frame(int(last_idx))
        mask = f["mask"] > 0
        u = np.where(mask, f["u"], np.nan).astype(np.float32)
        v = np.where(mask, f["v"], np.nan).astype(np.float32)
        return {"u": u, "v": v}


class ClimatologyPredictor:
    """目标帧月份的 1978–2017 平均 u/v 场。"""

    model_id = "climatology"

    def __init__(self, climatology_path: str | Path | None = None):
        self.path = resolve_path(climatology_path, default=f"{ARTIFACT_DIR}/climatology.npz")
        if self.path.exists():
            d = np.load(self.path)
            self.u = np.asarray(d["u"], np.float32)        # (12,H,W)
            self.v = np.asarray(d["v"], np.float32)
            self.source = str(self.path)
        else:
            fallback = ROOT / "data/interim/climatology_monthly_mean.npz"
            if not fallback.exists():
                raise FileNotFoundError(
                    f"气候态 artifact 缺失: {self.path}（也找不到回退 {fallback}）")
            d = np.load(fallback)
            mm = np.asarray(d["monthly_mean"], np.float32)   # (12,2,H,W)
            self.u = mm[:, 0]
            self.v = mm[:, 1]
            self.source = str(fallback)
        if self.u.shape != (12, GRID_H, GRID_W):  # pragma: no cover - 防御性
            raise ValueError(f"气候态形状异常: {self.u.shape}")

    def _month_index(self, store, target_idx: int) -> int:
        m = int(store.month_of(int(target_idx)))          # 1..12
        return min(max(m - 1, 0), 11)

    def predict_point(self, store, target_idx: int, row: int, col: int) -> tuple[float, float]:
        k = self._month_index(store, target_idx)
        return float(self.u[k, row, col]), float(self.v[k, row, col])

    def predict_field(self, store, target_idx: int) -> dict:
        k = self._month_index(store, target_idx)
        return {"u": self.u[k].copy(), "v": self.v[k].copy()}
