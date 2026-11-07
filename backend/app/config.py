"""服务配置：路径、网格常量、模型目录、CORS、缓存参数。

所有路径默认相对工作区根目录解析；可用环境变量覆盖（便于测试与部署）：

    PREDICT_ZARR_PATH        zarr 数据集路径
    PREDICT_DASHBOARD_DIR    data/processed/dashboard artifacts 目录
    PREDICT_MODELS_ML_DIR    models/ml 模型目录
    PREDICT_SIMVP_CKPT       models/simvp-24h/ckpt_best.pt
    PREDICT_DASHBOARD_DIST   dashboard/dist 前端构建产物目录
    PREDICT_WARMUP           1/0 是否在启动后后台预热引擎（默认 1）
    PREDICT_LOG_LEVEL        uvicorn/应用日志级别
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

# ---------------------------------------------------------------------------
# 工作区根目录：backend/app/config.py -> [0]=app [1]=backend [2]=workspace root
# ---------------------------------------------------------------------------
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

# 保证 `import src.predict...` / `import src.serving...` 可用（src 为命名空间包）。
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


# ---------------------------------------------------------------------------
# 契约常量（docs/PREDICT_API_CONTRACT.md / §2 设计文档）
# ---------------------------------------------------------------------------
GRID_ROWS = 116
GRID_COLS = 158
CELL_KM = 25.07                       # 契约展示值；实际 25.0675 km
KM_PER_CM_S_HOUR = 0.036              # speed(cm/s) * 0.036 * h = km(h)
CELLS_PER_CM_S_DAY = 0.034470         # u(cm/s) 一天像元位移
HORIZON_HOURS = 24
INTERPOLATION_NOTE = "日尺度产品：逐时位置/位移由 24h 平均漂移矢量线性积分"

ISOLATION_WARNING = (
    "日尺度产品只能给出 24h 平均漂移矢量；逐时曲线为线性积分近似，非 1h 分辨率观测"
)

DATASET_SOURCE = (
    "NSIDC-0116 v4 Polar Pathfinder Daily 25km EASE-Grid Sea Ice Motion Vectors "
    "(Weddell subset)"
)
DATASET_DOI = "10.5067/INAWUWO7QH7B"

# /api/models 的模型元数据（字段与契约逐字段对应，不可改名）
MODEL_CATALOG: tuple[dict, ...] = (
    {
        "id": "ridge",
        "name": "Ridge 岭回归",
        "short": "Ridge",
        "category": "linear",
        "color": "#22d3ee",
        "description": "标准化特征上的多输出线性回归，推理毫秒级，可解释性强。",
        "is_baseline": False,
    },
    {
        "id": "random_forest",
        "name": "Random Forest 随机森林",
        "short": "Random Forest",
        "category": "tree",
        "color": "#a78bfa",
        "description": "多输出随机森林（特征子采样），树间离散度可给出预测不确定度。",
        "is_baseline": False,
    },
    {
        "id": "simvp",
        "name": "SimVP 24h 微调",
        "short": "SimVP 24h",
        "category": "deep",
        "color": "#f59e0b",
        "description": "沿用原 SimVP 时空网络，针对首日漂移微调。输入 10 帧日漂移场，输出次日 u/v。",
        "is_baseline": False,
    },
    {
        "id": "persistence",
        "name": "持续性基线",
        "short": "Persistence",
        "category": "baseline",
        "color": "#64748b",
        "description": "下一帧 = 当前帧（海冰漂移短期预测的强基准）。",
        "is_baseline": True,
    },
    {
        "id": "climatology",
        "name": "气候态基线",
        "short": "Climatology",
        "category": "baseline",
        "color": "#475569",
        "description": "1978–2017 逐月平均漂移场（每像元）。",
        "is_baseline": True,
    },
)

MODEL_IDS: tuple[str, ...] = tuple(m["id"] for m in MODEL_CATALOG)
TABULAR_MODEL_IDS: tuple[str, ...] = ("ridge", "random_forest", "persistence", "climatology")
DEEP_MODEL_IDS: tuple[str, ...] = ("simvp",)


def model_catalog_map() -> dict[str, dict]:
    return {m["id"]: dict(m) for m in MODEL_CATALOG}


# 每个表格模型依赖的 artifact（相对工作区根目录；persistence 无 artifact）
_MODEL_ARTIFACT_NAMES: dict[str, tuple[str, ...]] = {
    "ridge": ("models/ml/ridge.joblib",),
    "random_forest": ("models/ml/rf.joblib",),
    "persistence": (),
    "climatology": ("data/processed/dashboard/climatology.npz",),
}


@dataclass(frozen=True)
class Settings:
    """不可变运行时配置（由 :func:`load_settings` 构建）。"""

    workspace_root: Path = WORKSPACE_ROOT
    zarr_path: Path = WORKSPACE_ROOT / "data/processed/weddell_icemotion_daily.zarr"
    dashboard_dir: Path = WORKSPACE_ROOT / "data/processed/dashboard"
    models_ml_dir: Path = WORKSPACE_ROOT / "models/ml"
    simvp_ckpt: Path = WORKSPACE_ROOT / "models/simvp-24h/ckpt_best.pt"
    dashboard_dist: Path = WORKSPACE_ROOT / "dashboard/dist"

    # 服务元信息
    service_name: str = "weddell-seaice-ml"
    version: str = "1.2.0"

    # 网格 / 物理常量
    grid_rows: int = GRID_ROWS
    grid_cols: int = GRID_COLS
    cell_km: float = CELL_KM

    # 默认预测参数（契约 /api/models 的 default）
    default_row: int = 65
    default_col: int = 60
    preferred_default_date: str = "2025-06-26"

    # 缓存 / 预热
    predict_cache_size: int = 128
    backtest_cache_size: int = 24
    warmup: bool = True

    # CORS：Vite dev server（契约要求 5173，额外兼容 vite preview 4173）
    cors_origins: tuple[str, ...] = (
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:4173",
        "http://localhost:4173",
    )

    def artifact_paths(self, model_id: str) -> tuple[Path, ...]:
        """某模型实际 artifact 路径（仅返回配置指向的路径，不保证存在）。"""
        names = _MODEL_ARTIFACT_NAMES.get(model_id, ())
        return tuple(self.workspace_root / n for n in names)


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    if not raw:
        return default
    return Path(raw).expanduser().resolve()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def load_settings() -> Settings:
    """从环境变量加载配置（每次调用重新读取，便于测试）。"""
    default_zarr = WORKSPACE_ROOT / "data/processed/weddell_icemotion_daily.zarr"
    default_dashboard = WORKSPACE_ROOT / "data/processed/dashboard"
    default_models = WORKSPACE_ROOT / "models/ml"
    default_ckpt = WORKSPACE_ROOT / "models/simvp-24h/ckpt_best.pt"
    default_dist = WORKSPACE_ROOT / "dashboard/dist"
    return Settings(
        zarr_path=_env_path("PREDICT_ZARR_PATH", default_zarr),
        dashboard_dir=_env_path("PREDICT_DASHBOARD_DIR", default_dashboard),
        models_ml_dir=_env_path("PREDICT_MODELS_ML_DIR", default_models),
        simvp_ckpt=_env_path("PREDICT_SIMVP_CKPT", default_ckpt),
        dashboard_dist=_env_path("PREDICT_DASHBOARD_DIST", default_dist),
        warmup=_env_bool("PREDICT_WARMUP", True),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例配置。"""
    return load_settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
