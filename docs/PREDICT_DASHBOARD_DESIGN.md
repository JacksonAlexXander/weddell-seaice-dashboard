# 海冰漂移 ML 预测服务 + 可视化大屏 —— 总体设计

本文记录原版设计。现行模型、评估窗口和数据批次见[训练与回测记录](训练与回测.md)，接口以当前 API 契约为准。

> 版本 v1（2026-10-26）｜Owner: Lead｜读者：ml-tabular / backend-api / frontend-ui 三个 teammate
> 本文件是**唯一设计事实源**；API 字段级契约见 [`PREDICT_API_CONTRACT.md`](PREDICT_API_CONTRACT.md)。

## 0. 背景与目标

工作区已有「威德尔海海冰运动轨迹预测」项目（NSIDC-0116 v4，1978–2025，日尺度，25 km EASE-Grid South，
116×158 网格；SimVP / ConvLSTM 模型已训练完成，报告在 `reports/final_report.md`）。

本期在其基础上交付**真实机器学习预测接口 + Vue3/ECharts 可视化大屏**，要求：

1. `POST /api/predict` 真实推理接口（不是 mock、不是前端本地计算）。
2. 支持选择不同模型（Ridge、Random Forest、SimVP、持续性、气候态）。
3. 后端接收请求数据 → 调用**已训练 artifact** 推理 → 返回 JSON。
4. **点击预测时绝不重新训练**：所有模型离线训练一次并落盘（joblib / pt），服务启动只加载。
5. 大屏在原有可视化内容（48 年数据总览、海冰漂移场、气候态）之上，新增：
   模型选择、一键预测、未来 24h 预测曲线、历史回测真实对比、Ridge vs RF 指标对比、
   预测状态 / 异常提示 / 数据时间显示。

> 说明：工作区中此前没有 Vue3 大屏代码（已核实），因此本大屏的「原有可视化」部分 =
> 用真实数据重建原项目的核心可视化（48 年趋势、月度气候态、漂移矢量场、数据质检 KPI），
> 新功能叠加在其上。

## 1. 数据事实（已实测）

| 项 | 值 |
|---|---|
| zarr | `data/processed/weddell_icemotion_daily.zarr`（872 MB，zarr v2，blosc） |
| 帧数 | 17,188 个有效日 |
| 时间范围 | 1978-10-25 ～ 2025-12-30（48 年） |
| 网格 | 116 行 × 158 列，EASE-Grid South 25.07 km；行增大=北，列增大=东 |
| 变量 | `u`,`v`,`speed` (T,116,158) float32 cm/s；`valid_mask` uint8；`doy`,`month`,`time`；`latitude`,`longitude`；`stats/u_mean,u_std,v_mean,v_std`（1978–2017 每像元 z-score 统计） |
| 缺失日 | 36 天（已跳过）；相邻帧间隔 1 天为主，少量 2–8 天缺口 |
| 有效覆盖 | 全期平均 30.9%（季节 5.7%–43.7%）；平均每天有效像元约 5,600 |
| 划分 | train 1978–2017（14,283 帧）/ val 2018–2021（1,457）/ test 2022–2025（1,448） |
| 推荐默认预测点 | date=`2025-06-26`，cell=(row 65, col 60)，lat≈-68.04, lon≈-33.2（冬季高覆盖日） |

## 2. 预测任务定义（统一点预测口径）

**任务**：给定「最后观测日」`date`（zarr 中的某一帧）与网格像元 `(row, col)`，
预测**下一有效帧（≈未来 24 小时）**该像元的海冰漂移矢量 `(u, v)`，单位 cm/s。

- 目标帧 = `date` 在 zarr 时间轴上的下一帧（正常情况恰为 +1 天；遇到缺失日则间隔 >1 天，返回 warning）。
- 「未来 24 小时曲线」：日尺度产品只能给出 24h 平均漂移矢量，因此逐时（0–24 h，25 个点）位置/位移
  由该矢量**线性积分**得到（物理约束，UI 与接口中明确标注 `interpolation` 说明），同时给出模型不确定度带。
  - 位移换算：`km(h) = speed_cm_s * 0.036 * h`（cm/s × 3600 s = 36·speed m/h = 0.036·speed km/h）；
  - 像元位移：`d_cells(24h) = u * 86400 / (25067.525 * 100) = u * 0.034470`（u 分量沿列、v 沿行）；
  - 经纬度由 `latitude/longitude` 网格双线性插值。
- 统一评估协议（所有模型共用，保证指标可比）：
  `data/processed/dashboard/eval_windows.json`：沿用项目 test split + stride=5 的 287 个起报窗口
  （与 `reports/final_report.md` 中 SimVP 测试口径一致），评估每个窗口「目标帧」的全部有效像元，
  共 **1,556,493** 个样本。样本行 = (窗口, 像元)，记录 obs / 各模型 pred 的 u、v。

指标定义（写入 `metrics.json` 的 `protocol.metric_defs`，并被后端/前端直接引用）：

| 指标 | 定义 |
|---|---|
| `rmse_u` / `rmse_v` | 分量 RMSE（cm/s），在全部评估样本上 |
| `vec_rmse` | `sqrt(mean(Δu²+Δv²))`（cm/s） |
| `mae_vec` | `mean(sqrt(Δu²+Δv²))`（cm/s） |
| `r2_u` / `r2_v` | 分量决定系数 |
| `skill_vs_persistence` | `1 - vec_rmse / vec_rmse_persistence`（与项目 `skill_pers` 同口径） |
| `latency_ms` | 单次「单点预测」端到端推理耗时中位数（离线实测，不含 HTTP 开销） |

## 3. 模型与训练口径

| id | 名称 | 类型 | 说明 | artifact |
|---|---|---|---|---|
| `ridge` | Ridge 岭回归 | 线性 | 每像元 z-score 后多变量线性回归，多输出 (u,v) | `models/ml/ridge.joblib` |
| `random_forest` | Random Forest | 树集成 | 多输出随机森林，feature subsampling；可选树间离散度做不确定度 | `models/ml/rf.joblib` |
| `simvp` | SimVP-v1（深度学习） | 时空场预测 | 复用已有 `models/simvp-v1/ckpt_best.pt`（9.3M 参数），输入 10 帧 → 输出 5 天场，取 day-1 | 既有 ckpt |
| `persistence` | 持续性基线 | 基线 | 下一帧 = 当前帧 | 无需训练 |
| `climatology` | 气候态基线 | 基线 | 1978–2017 逐月平均场（每像元） | `data/processed/dashboard/climatology.npz` |

### 3.1 Ridge / Random Forest 特征工程（ml-tabular 负责）

对 (date=t, 像元) 预测 t+1 帧。特征约 31 维：

- 中心像元滞后：u,v,speed @ t, t-1, t-2, t-3（12）
- 滚动均值：u,v,speed 近 3 日、近 7 日均值（6）
- 邻域（3×3 除中心）均值：u,v,speed @ t 与 t-1（6）；邻域 u,v 标准差 @ t（2）
- 趋势：u,v,speed 的 (t − t-3) 差分（3）
- 季节：`sin(2π·doy)`, `cos(2π·doy)`（2）
- 位置/质量：`latitude`, `longitude`, 当日 `valid_fraction`（3）

> 注意：`src/dataset.py` 的 stats 是**每像元 z-score**（训练段统计），特征/目标统一用该标准化空间或
> 原始 cm/s 均可，但**目标必须还原为 cm/s 评估**。推荐直接在 cm/s 空间训练（u/v 量级 0–50，树模型无需缩放；
> Ridge 用 `StandardScaler` Pipeline 处理）。

训练/验证/测试：与项目一致 train 1978–2017 / val 2018–2021 / test 2022–2025。
训练采样：每天随机取约 150–220 个有效像元（共 ~2–3M 行），`random_state=42`；
超参在 val 子集上小网格选择后固定；**最终指标只在 test 统一评估协议上报告**。
训练脚本必须：可续跑、日志到 `logs/`、artifact 到 `models/ml/`、指标 JSON 到 `data/processed/dashboard/`。

### 3.2 SimVP 统一口径评估（Lead 负责）

复用既有 `src/models.py` + `src/dataset.py`：对 287 个评估窗口，加载 10 帧输入（含 doy 通道、
z-score、pad 120×160），模型一次前向取 day-1 的 u/v，反标准化后与目标帧真值在有效像元比较。
预期可复现报告中的 day-1 `vec_rmse ≈ 9.79`（±0.1）——作为实现正确性自检。
另预计算 test 期（2022–2025）全部目标日的 day-1 场，供大屏「预测场」与「历史回测」秒级响应。

## 4. 代码 / 产物布局（谁写什么）

```
src/predict/                      # ml-tabular 独占（新包）
  __init__.py
  data.py         DataStore       # zarr 读取、日期索引、窗口、有效像元、字段缓存
  features.py     FEATURE_NAMES, build_features()
  engine.py       PredictionEngine# 统一点预测/场预测/回测/像元统计/数据时间（ridge,rf,persistence,climatology）
  baselines.py    PersistencePredictor, ClimatologyPredictor
  train_tabular.py                # 离线训练 + 评估 CLI（后台跑）
  build_artifacts.py              # 生成 timeseries.json / metrics_tabular.json / cell_stats.npz / eval parquet
scripts/build_predict_artifacts.py   # ml-tabular 的一次性入口（可调用 build_artifacts）
models/ml/                        # ml-tabular：ridge.joblib, rf.joblib, feature_spec.json, model_card.json
data/processed/dashboard/         # 共享产物目录（按文件名切分，禁止改别人的文件）
  eval_windows.json               # Lead ✅ 已生成（统一评估协议）
  timeseries.json                 # ml-tabular：48 年年度/月度/年月矩阵 + KPI + 数据时间
  climatology.npz                 # ml-tabular：月度平均 u,v（12,116,158）+ 月度 std
  cell_stats.npz                  # ml-tabular：逐像元 speed mean/std/p99（训练段）
  metrics_tabular.json            # ml-tabular：4 个模型的统一指标
  eval_day1.parquet               # ml-tabular：评估明细（窗口、像元、obs 与各模型 pred），供复核
  metrics_simvp.json              # Lead：SimVP 指标
  simvp_day1_fields.npz           # Lead：test 期逐日 day-1 预测场（供回测/预测场）
  metrics.json                    # Lead 合并：最终指标（后端只读这个）
  overview.json                   # Lead 合并：大屏总览（由 timeseries.json + 数据时间派生）
src/serving/                      # Lead 独占（新包）
  simvp.py         SimVPPredictor # 既有 SimVP ckpt 的推理封装（线程安全、MPS/CPU 自适应、场/点预测）
  build_simvp_artifacts.py        # 预计算 + 统一口径指标
backend/                          # backend-api 独占（新）
  __init__.py  app/{__init__.py, main.py, config.py, schemas.py, deps.py}
  app/routers/{health.py, models.py, metrics.py, overview.py, field.py, predict.py, backtest.py}
  app/services/{artifact_store.py, field_service.py, predict_service.py}
  tests/test_api.py
scripts/serve_dashboard.py        # backend-api：uvicorn 启动入口（--host 127.0.0.1 --port 8000）
dashboard/                        # frontend-ui 独占（新，Vue3 + Vite + ECharts）
  package.json vite.config.js index.html src/... dist/（构建产物）
docs/, reports/                   # Lead 汇总
```

**写入纪律**：每个 teammate 只写自己 scope；共享目录 `data/processed/dashboard/` 只写约定文件名；
`src/`、`models/`、`reports/` 已有文件**只读**（除各自约定新增文件）。

## 5. 模块接口契约（backend-api 调用这些模块，ml-tabular/Lead 实现）

### 5.1 `src/predict`（ml-tabular）

```python
# src/predict/data.py
class DataStore:
    def __init__(self, zarr_path: str = "data/processed/weddell_icemotion_daily.zarr")
    n_frames: int
    dates: np.ndarray           # datetime64[D], shape (T,)
    latitude: np.ndarray        # (116,158) float32
    longitude: np.ndarray
    def idx_of_date(self, date: str) -> int | None          # 精确匹配，无则 None
    def nearest_idx(self, date: str) -> int                 # 最近帧
    def next_idx(self, i: int) -> int                       # i+1（越界抛 ValueError）
    def frame(self, i: int) -> dict                         # {u,v,speed,mask,doy,date,coverage}
    def window(self, end_idx: int, n_days: int) -> dict      # {u,v,speed,mask,doy} 数组 (n,H,W)，旧→新
    def valid_cells(self, i: int) -> tuple[np.ndarray, np.ndarray]  # (rows, cols)

# src/predict/features.py
FEATURE_NAMES: list[str]
N_HISTORY: int                  # 需要的历史天数（含当日，>=8）
def build_features(win: dict, rows: np.ndarray, cols: np.ndarray) -> np.ndarray
    # win["u"/"v"/"speed"]: (n,H,W) 旧→新，win["doy"]: (n,), n>=N_HISTORY
    # 返回 (len(rows), len(FEATURE_NAMES)) float32

# src/predict/engine.py
class PredictionEngine:
    def __init__(self, zarr_path="data/processed/weddell_icemotion_daily.zarr",
                 model_dir="models/ml", artifact_dir="data/processed/dashboard")
    store: DataStore
    model_ids: list[str]        # ["ridge","random_forest","persistence","climatology"]
    def available(self, model_id: str) -> bool
    def data_time(self) -> dict          # {"first_date","latest_date","n_frames","built_at","batch_age_days"}
    def predict_point(self, model_id, date: str, row: int, col: int) -> dict
        # {"ok": True, "next_date": "YYYY-MM-DD", "next_idx": int,
        #  "u": float, "v": float, "speed": float, "direction_deg": float,
        #  "input": {"u","v","speed","direction_deg","lat","lon","valid","coverage","date"},
        #  "uncertainty": {"method": str, "lower_km": float, "upper_km": float} | None,
        #  "warnings": [{"code","level","message"}]}
    def predict_field(self, model_id, date: str) -> dict
        # {"ok": True, "u": (H,W) float32 (无效处 nan), "v": (H,W), "speed": (H,W),
        #  "coverage": float, "warnings": [...]}
    def backtest(self, model_id, row: int, col: int, start: str, end: str,
                 max_points: int = 180) -> dict
        # {"ok": True,
        #  "series": [{"date","obs_u","obs_v","obs_speed","pred_u","pred_v","pred_speed"}...],
        #  "summary": {"n","rmse_u","rmse_v","vec_rmse","mae_vec","r2_u","r2_v","skill_vs_persistence"},
        #  "available_start": str, "available_end": str, "warnings": [...]}
    def cell_stats(self, row: int, col: int) -> dict
        # {"speed_mean","speed_std","speed_p99","valid_days","lat","lon"}
```

### 5.2 `src/serving/simvp.py`（Lead）

```python
class SimVPPredictor:
    def __init__(self, zarr_path="data/processed/weddell_icemotion_daily.zarr",
                 ckpt="models/simvp-v1/ckpt_best.pt", device="auto",
                 precomputed="data/processed/dashboard/simvp_day1_fields.npz")
    state: dict                 # {"loaded": bool, "device": str, "ckpt": str, "error": str|None}
    def predict_point(self, date: str, row: int, col: int) -> dict
        # 与 5.1 predict_point 同 schema（uncertainty 用 test day-1 vec_rmse 构造）
    def predict_field(self, date: str) -> dict
        # 与 5.1 predict_field 同 schema；优先用预计算场（test 期），否则实时推理
    def backtest(self, row: int, col: int, start: str, end: str, max_points: int = 180) -> dict
        # 与 5.1 backtest 同 schema（仅 test 期 2022–2025 有覆盖，越界返回 warning）
```

> backend-api 在 import 失败或 `state["loaded"]=False` 时，`/api/models` 中把 simvp 标为
> `available:false` 并降级返回 503 风格错误，不得让整个服务崩溃。

## 6. 大屏信息架构（frontend-ui）

1920×1080 深色科技风，CSS Grid 三栏；所有数据来自后端 API，**不内置假数据**（开发期可用 mock fixtures，
联调后必须切真实接口）：

- **Header**：标题「南极威德尔海海冰漂移智能预测大屏」＋数据集徽标（NSIDC-0116 v4 / 48 年）；
  右侧：数据时间（最新日期 + 数据集构建时间）、实时时钟、后端连接状态点 / 预测状态。
- **中栏（核心）**：
  1) 控制条：模型选择（Ridge / Random Forest / SimVP / 持续性 / 气候态，来自 `/api/models`）、
     日期选择、像元选择（row/col，支持点击地图选点）、**一键运行预测**按钮、预测状态（idle/loading/success/error + 耗时）。
  2) 海冰漂移场（ECharts 热力图 speed + 矢量箭头 + 选中像元 + 24h 预测路径；观测/预测切换）。
  3) **未来 24 小时预测曲线**（ECharts：逐时累计位移 km + 不确定度带；副轴 speed；明确标注积分口径）。
- **左栏**：数据 KPI 卡（48 年 / 17,188 日 / 平均速度 / 有效覆盖）；48 年年度平均漂移速度趋势；
  月度气候态；数据覆盖/缺失信息。
- **右栏**：预测结果卡（u/v/speed/方向/24h 位移/模型/耗时）；**异常提示**（warnings，分级配色）+
  数据时间；**历史回测真实对比**（真实 vs 预测逐日曲线 + RMSE/技能分摘要）；
  **Ridge vs Random Forest 指标对比**（分组柱状 + 表格 + 技能分，来自 `/api/metrics`）。
- **Footer**：数据来源 DOI、预测口径声明（日尺度 → 24h 平均矢量，逐时位置为线性积分）、生成时间。

## 7. 验收标准（Definition of Done）

1. `bash scripts/start_dashboard.sh`（或文档中的等价命令）一键启动，浏览器打开 `http://127.0.0.1:8000/`
   即可看到大屏；后端 `/docs` 有 OpenAPI。
2. 六项新增功能全部可用：模型选择、一键预测、24h 曲线、历史回测、Ridge/RF 指标对比、状态/异常/数据时间。
3. `/api/predict` 真实调用已训练模型：Ridge/RF 为 joblib 推理，SimVP 为 torch 推理；返回真实数值，
   且**反复点击预测时模型文件 mtime 不变、无任何训练日志/训练耗时**（QA 验证）。
4. `pytest backend/tests` 全绿；`curl` 冒烟覆盖全部端点。
5. 指标表含 Ridge vs RF 的 vec_rmse/mae/r²/skill，数值与离线评估 JSON 完全一致（同一 artifact）。
6. 截图轮（headless Chrome 1920×1080）视觉检查通过：无溢出/遮挡/空白图表、中文正常、状态与时间可见。
7. 文档：README 增补 + `reports/final_report_ml_dashboard.md`（数据/方法/指标/复现/局限）。
