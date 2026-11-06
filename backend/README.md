# 海冰漂移 ML 预测服务后端（FastAPI）

威德尔海海冰运动（NSIDC-0116 v4，1978–2025，日尺度，25 km EASE-Grid South，116×158）
的**真实机器学习推理后端**与大屏静态托管。

> 字段级契约：[`docs/PREDICT_API_CONTRACT.md`](../docs/PREDICT_API_CONTRACT.md)
> 总体设计：[`docs/PREDICT_DASHBOARD_DESIGN.md`](../docs/PREDICT_DASHBOARD_DESIGN.md)

**核心承诺**：`/api/predict`/`/api/backtest` 只对**已加载的离线训练 artifact** 做推理，
绝不训练、绝不写 `models/` 与 `data/`（`pytest` 有专门的 mtime / 文件集不变量测试）。

---

## 1. 启动

```bash
# 工作区根目录执行（.venv 固定解释器）
.venv/bin/python scripts/serve_dashboard.py --host 127.0.0.1 --port 8000

# 开发热重载 / 换端口 / 关闭启动预热
.venv/bin/python scripts/serve_dashboard.py --port 8010 --reload
.venv/bin/python scripts/serve_dashboard.py --no-warmup
```

启动后：

| 地址 | 说明 |
|---|---|
| `http://127.0.0.1:8000/` | 大屏（若 `dashboard/dist` 已构建；未构建返回 JSON 构建提示） |
| `http://127.0.0.1:8000/docs` | Swagger UI（OpenAPI） |
| `http://127.0.0.1:8000/openapi.json` | OpenAPI schema |
| `http://127.0.0.1:8000/api/health` | 健康检查 |

启动时会在**后台线程**预热引擎（FastAPI lifespan 不阻塞、失败不崩）；首个预测请求前
通常已完成加载（Ridge/RF joblib + SimVP torch ckpt，约 2–4 s，视磁盘/设备而定）。

环境变量（均可选）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `PREDICT_ZARR_PATH` | `data/processed/weddell_icemotion_daily.zarr` | 数据集 |
| `PREDICT_DASHBOARD_DIR` | `data/processed/dashboard` | 指标/气候态/预计算场 artifacts |
| `PREDICT_MODELS_ML_DIR` | `models/ml` | Ridge/RF joblib |
| `PREDICT_SIMVP_CKPT` | `models/simvp-v1/ckpt_best.pt` | SimVP checkpoint |
| `PREDICT_DASHBOARD_DIST` | `dashboard/dist` | 前端构建产物 |
| `PREDICT_WARMUP` | `1` | `0` 关闭启动后台预热 |
| `PREDICT_LOG_LEVEL` | `INFO` | 日志级别 |

---

## 2. API 一览（字段严格按契约）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 存活、已加载模型、artifacts 完整性 |
| GET | `/api/models` | DataTime + 默认预测参数 + 5 模型（含可用性/指标/latency）+ combos |
| GET | `/api/metrics` | 统一评估协议指标（`metrics.json` → 回退 `metrics_tabular.json` + warning） |
| GET | `/api/overview` | 大屏总览（`overview.json` → 回退 `timeseries.json` 动态组装） |
| GET | `/api/field?date=&stride=` | 观测漂移场：speed 116×158（无效 null，2 位小数）+ 抽样矢量 + 经纬度轴 + 统计 |
| GET | `/api/field/predicted?model=&date=&stride=` | 预测漂移场；目标帧真值存在时 `stats.comparison` 给该日 vec_rmse |
| POST | `/api/predict` | **真实推理**：下一帧单点矢量 + 24h 逐时积分曲线 + 不确定度 + warnings |
| POST | `/api/backtest` | 历史回测：目标日真值 vs 预测逐日序列 + RMSE/技能分摘要 |

模型 id：`ridge` / `random_forest` / `persistence` / `climatology`（`src.predict.PredictionEngine`）、
`simvp`（`src.serving.SimVPPredictor`）。

### 统一错误体

所有错误响应：`HTTP 4xx/5xx` + `{"status":"error","error":{"code","message","hint?"}}`：

| HTTP | code | 场景 |
|---|---|---|
| 400 | `invalid_cell` | row/col 越界或非整数 |
| 404 | `unknown_model` | 模型 id 不存在 |
| 409 | `model_unavailable` | artifact 缺失/加载失败（如训练中） |
| 422 | `date_invalid` | 日期格式错 / 数据集最后一帧无下一帧 |
| 422 | `date_not_in_dataset` | 数据集无该日（缺失日） |
| 422 | `insufficient_history` | 历史帧不足（SimVP 需 10 帧） |
| 422 | `validation_error` | 请求体字段缺失/类型错误 |
| 422 | `invalid_range` / `backtest_out_of_range` | 回测范围非法 / SimVP 请求 2022 之前 |
| 500 | `inference_failed` / `internal_error` | 推理异常（traceback 只写日志，不回前端） |
| 503 | `artifact_unavailable` | zarr/指标 artifact 尚未就绪（优雅降级） |

其它：CORS 允许 `http://127.0.0.1:5173`、`http://localhost:5173`（Vite dev，另兼容 4173）；
所有 JSON 响应 `Content-Type: application/json; charset=utf-8`。

### 预测响应关键口径（设计 §2）

- 目标帧 = 起报日在 zarr 时间轴的**下一帧**（正常 +1 天，遇缺失日间隔 >1 天给 `frame_gap` warning）；
- 位移：`km(h) = speed_cm_s × 0.036 × h`；像元位移 `u × 0.034470`（沿列/东）、`v × 0.034470`（沿行/北）；
- `series_24h` 25 点，日尺度产品逐时速度为**恒定 24h 平均矢量**（`interpolation` 文案明示线性积分）；
- 不确定度：Ridge 用验证集残差 σ、RF 用树间离散度 10–90 分位；基线模型 `uncertainty: null`；
- 确定性结果有进程内小内存 LRU 缓存（key: model + date + cell，模型 artifact 更新后自动失效）。

---

## 3. 目录结构

```
backend/
  app/
    main.py              FastAPI 应用：路由注册 / 统一错误 / 静态托管 / lifespan 预热
    config.py            路径、网格常量、模型目录、CORS、缓存参数
    schemas.py           pydantic v2 请求/响应模型 + APIError（逐字段对齐契约）
    deps.py              app.state 单例依赖
    routers/             health models metrics overview field predict backtest
    services/
      artifact_store.py  引擎懒加载/降级、artifacts(mtime 缓存)、zarr fallback 读取器
      field_service.py   观测场/预测场组装、24h 序列、双线性插值、错误码映射
      predict_service.py 单点预测/回测、契约响应、LRU 缓存
  tests/
    conftest.py          sys.path / PREDICT_WARMUP=0
    test_api.py          pytest + TestClient（31 用例，含不重训 mtime 断言）
scripts/serve_dashboard.py   uvicorn 启动入口（自动把工作区根目录加入 sys.path）
```

### 降级策略（依赖未就绪时服务绝不崩）

- `src.predict` / `src.serving` 未就绪或 artifacts 缺失：
  `/api/health` 仍 200；`/api/models` 对应模型 `available:false`（文件存在时先乐观 true，
  引擎加载失败后翻 false）；预测类端点 409 `model_unavailable`、数据不可用时 503。
- `metrics.json` 未合并：回退 `metrics_tabular.json`（并合并 `metrics_simvp.json`），
  `models` 自动补齐并按契约顺序排序，响应带 `warnings`。
- `overview.json` 未生成：由 `timeseries.json` + 数据时间动态组装，响应带 `warnings`。
- 模型 artifact 被原子替换（如冒烟模型 → 正式训练模型）：按 mtime 检测并自动
  `engine.reload()`，预测缓存随代数失效，服务无需重启。
- `dashboard/dist` 未构建：`GET /` 与 SPA fallback 返回 JSON 构建提示（
  `cd dashboard && pnpm install && pnpm run build`）。

---

## 4. 测试

```bash
.venv/bin/python -m pytest backend/tests -q
```

覆盖：health/models/metrics/overview/field/predict/backtest 成功路径、未知模型 404、
非法 cell 400、非法日期 422、错误体格式、predict 响应字段结构与数值范围（24h 序列单调、
位移换算、方向、不确定度带包络）、CORS、静态托管、假引擎路由逻辑 + LRU 缓存命中，
以及**不重训验证**：predict/backtest 前后 `models/ml/*.joblib`、`models/**/*.pt`
mtime 与文件集不变。引擎/artifacts 缺失时相关用例自动 `skip`（`pytest.mark.skipif`）。

---

## 5. 依赖说明

- 运行时：FastAPI 0.162 / pydantic v2 / uvicorn / numpy（zarr、sklearn、torch 由
  `src/predict`、`src/serving` 在加载模型时按需 import）。
- `src/predict`（ml-tabular 维护）：`PredictionEngine.predict_point/predict_field/backtest/
  cell_stats/data_time/available`。
- `src/serving/simvp.py`（Lead 维护）：`SimVPPredictor.predict_point/predict_field/backtest/load`。
- 两者接口与设计 §5 一致；后端只调用、不修改、不重新训练。
