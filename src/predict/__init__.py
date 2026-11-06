"""src.predict：表格 ML 预测包（Ridge / RandomForest / 持续性 / 气候态）。

对外主要入口：
- ``predict.engine.PredictionEngine``：统一点预测/场预测/回测/像元统计/数据时间
- ``predict.data.DataStore``：zarr 读取
- ``predict.features``：34 维特征
- ``predict.baselines``：零训练基线

本包同时提供离线训练 CLI（``predict.train_tabular``）与 artifact 构建 CLI
（``predict.build_artifacts``），服务端只加载产物、绝不训练。
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = [
    "DataStore",
    "PredictionEngine",
    "PersistencePredictor",
    "ClimatologyPredictor",
    "FEATURE_NAMES",
    "N_FEATURES",
    "N_HISTORY",
    "build_features",
]


def __getattr__(name):  # PEP 562 惰性导入，避免 import 包就加载 sklearn/zarr
    if name == "DataStore":
        from .data import DataStore
        return DataStore
    if name == "PredictionEngine":
        from .engine import PredictionEngine
        return PredictionEngine
    if name in ("PersistencePredictor", "ClimatologyPredictor"):
        from . import baselines
        return getattr(baselines, name)
    if name in ("FEATURE_NAMES", "N_FEATURES", "N_HISTORY", "build_features"):
        from . import features
        return getattr(features, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
