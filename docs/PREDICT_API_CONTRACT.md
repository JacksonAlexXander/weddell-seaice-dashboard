# 海冰漂移 ML 预测服务 —— API 契约 v1

> 服务基址：`http://127.0.0.1:8000`（FastAPI）｜前端开发期 Vite 代理 `/api` → 该地址
> 所有响应 `Content-Type: application/json; charset=utf-8`；字段名不可改动（前端按此硬编码）。
> 正常响应 HTTP 200；错误响应 HTTP 4xx/5xx，body 统一为 `ErrorResponse`。
> 设计文档见 [`PREDICT_DASHBOARD_DESIGN.md`](PREDICT_DASHBOARD_DESIGN.md)。

## 通用对象

```jsonc
// DataTime —— 数据时间（大屏 header / 预测状态区显示）
{
  "first_date": "1978-10-25",
  "latest_date": "2025-12-30",
  "n_frames": 17188,
  "built_at": "2026-10-25T14:40:44Z",     // 数据集 zarr 构建时间（.zattrs created_utc）
  "batch_age_days": 300,                   // 最新数据距今天数（信息性，不一定是异常）
  "source": "NSIDC-0116 v4 Polar Pathfinder Daily 25km EASE-Grid Sea Ice Motion Vectors (Weddell subset)",
  "doi": "10.5067/INAWUWO7QH7B"
}

// Warning —— 异常/提示
{ "code": "high_speed", "level": "warning", "message": "预测漂移速度 25.3 cm/s 高于该像元历史 99% 分位（21.8 cm/s），可能为极端事件" }
// level: "info" | "warning" | "danger"

// ErrorResponse
{ "status": "error", "error": { "code": "date_out_of_range", "message": "日期超出数据集范围", "hint": "可选日期 1978-11-01 ~ 2025-12-30" } }
```

## `GET /api/health`

```jsonc
{
  "status": "ok",
  "service": "weddell-seaice-ml",
  "version": "1.0.0",
  "uptime_s": 123.4,
  "models_loaded": ["ridge", "random_forest", "persistence", "climatology", "simvp"],
  "artifacts_ok": true
}
```

## `GET /api/models`

```jsonc
{
  "data_time": { /* DataTime */ },
  "default": { "model": "random_forest", "date": "2025-06-26", "cell": { "row": 65, "col": 60 } },
  "models": [
    {
      "id": "ridge",
      "name": "Ridge 岭回归",
      "short": "Ridge",
      "category": "linear",              // "linear" | "tree" | "deep" | "baseline"
      "color": "#22d3ee",
      "available": true,
      "description": "标准化特征上的多输出线性回归，推理毫秒级，可解释性强。",
      "trained_at": "2026-10-26T03:12:00Z",
      "training_samples": 2143000,
      "metrics": { "vec_rmse": 10.62, "skill_vs_persistence": 0.03, "latency_ms": 0.8 },
      "is_baseline": false
    },
    {
      "id": "random_forest", "name": "Random Forest 随机森林", "short": "Random Forest",
      "category": "tree", "color": "#a78bfa", "available": true, "description": "多输出随机森林……",
      "trained_at": "2026-10-26T03:15:00Z", "training_samples": 2143000,
      "metrics": { "vec_rmse": 10.15, "skill_vs_persistence": 0.07, "latency_ms": 34.2 },
      "is_baseline": false
    },
    {
      "id": "simvp", "name": "SimVP-v1 深度学习", "short": "SimVP",
      "category": "deep", "color": "#f59e0b", "available": true, "description": "项目既有 9.3M 参数时空预测模型……",
      "trained_at": "2026-10-26T00:20:00Z", "training_samples": null,
      "metrics": { "vec_rmse": 9.794, "skill_vs_persistence": 0.10, "latency_ms": 320.0 },
      "is_baseline": false
    },
    {
      "id": "persistence", "name": "持续性基线", "short": "Persistence",
      "category": "baseline", "color": "#64748b", "available": true,
      "description": "下一帧 = 当前帧（海冰漂移短期预测的强基准）。",
      "trained_at": null, "training_samples": null,
      "metrics": { "vec_rmse": 10.91, "skill_vs_persistence": 0.0, "latency_ms": 0.1 },
      "is_baseline": true
    },
    {
      "id": "climatology", "name": "气候态基线", "short": "Climatology",
      "category": "baseline", "color": "#475569", "available": true,
      "description": "1978–2017 逐月平均漂移场（每像元）。",
      "trained_at": null, "training_samples": null,
      "metrics": { "vec_rmse": 12.36, "skill_vs_persistence": -0.13, "latency_ms": 0.1 },
      "is_baseline": true
    }
  ],
  "combos": [                                  // 大屏可直接展示的模型×日期×像元组合
    { "model": "random_forest", "date": "2025-06-26", "cell": { "row": 65, "col": 60 } }
  ]
}
```

## `GET /api/metrics`

```jsonc
{
  "protocol": {
    "task": "未来 24 小时（下一日帧）海冰漂移矢量预测",
    "target": "u(t+1), v(t+1)，cm/s，25 km EASE-Grid South 像元",
    "split": { "train": "1978–2017", "val": "2018–2021", "test": "2022–2025" },
    "sampling": "test split 每 5 日一个起报窗口（287 个窗口），评估全部有效像元",
    "n_windows": 287,
    "n_samples": 1556493,
    "metric_defs": {
      "vec_rmse": "√mean(Δu²+Δv²)，cm/s",
      "mae_vec": "mean(√(Δu²+Δv²))，cm/s",
      "skill_vs_persistence": "1 − RMSE_model/RMSE_persistence（矢量 RMSE 比口径，与项目 skill_pers 一致）",
      "r2_u": "u 分量决定系数"
    }
  },
  "models": [
    {
      "id": "ridge", "name": "Ridge 岭回归", "name_zh": "Ridge 岭回归", "category": "linear",
      "vec_rmse": 10.62, "rmse_u": 7.55, "rmse_v": 7.47,
      "mae_vec": 8.71, "r2_u": 0.42, "r2_v": 0.40,
      "skill_vs_persistence": 0.03, "latency_ms": 0.8,
      "params": { "alpha": 1.0 }, "training_samples": 2143000,
      "is_highlight": true, "is_baseline": false
    },
    {
      "id": "random_forest", "name": "Random Forest 随机森林", "name_zh": "Random Forest 随机森林",
      "category": "tree", "vec_rmse": 10.15, "rmse_u": 7.22, "rmse_v": 7.14,
      "mae_vec": 8.30, "r2_u": 0.46, "r2_v": 0.44,
      "skill_vs_persistence": 0.07, "latency_ms": 34.2,
      "params": { "n_estimators": 250, "max_depth": null, "min_samples_leaf": 4 }, "training_samples": 2143000,
      "is_highlight": true, "is_baseline": false
    },
    { "id": "simvp", "name": "SimVP-v1 深度学习", "category": "deep", "vec_rmse": 9.79, "...": "同结构", "is_baseline": false },
    { "id": "persistence", "name": "持续性基线", "category": "baseline", "vec_rmse": 10.91, "skill_vs_persistence": 0.0, "is_baseline": true },
    { "id": "climatology", "name": "气候态基线", "category": "baseline", "vec_rmse": 12.36, "is_baseline": true }
  ],
  "generated_at": "2026-10-26T06:00:00Z"
}
```

> 前端必须把 `id` 为 `ridge` 与 `random_forest` 的两行高亮对比（柱状图 + 表格），其他模型作为参照行。

## `GET /api/overview`

```jsonc
{
  "dataset": {
    "name": "NSIDC-0116 v4 威德尔海海冰运动数据集",
    "region": "60°W–20°E, 60°S–83°S",
    "grid": { "rows": 116, "cols": 158, "cell_km": 25.07 },
    "n_frames": 17188, "first_date": "1978-10-25", "latest_date": "2025-12-30",
    "years": 48, "missing_days": 36,
    "mean_valid_fraction": 0.309, "mean_speed_cm_s": 8.42,
    "u_units": "cm/s", "source": "NSIDC-0116 v4"
  },
  "kpis": [
    { "id": "years", "label": "覆盖年限", "value": 48, "unit": "年" },
    { "id": "days", "label": "有效日数", "value": 17188, "unit": "天" },
    { "id": "mean_speed", "label": "平均漂移速度", "value": 8.42, "unit": "cm/s" },
    { "id": "coverage", "label": "平均有效覆盖", "value": 30.9, "unit": "%" }
  ],
  "annual": [ { "year": 1978, "mean_speed": 5.12, "coverage": 0.281, "n_days": 68 } ],          // 48 项
  "monthly": [ { "month": 1, "mean_speed": 4.51, "coverage": 0.19 } ],                            // 12 项
  "year_month": { "years": [1978], "months": [1,2,3,4,5,6,7,8,9,10,11,12], "values": [[5.1]] },      // 48×12
  "data_time": { /* DataTime */ }
}
```

## `GET /api/field`

观测场（大屏地图）。

Query：`date`（YYYY-MM-DD，可选，默认最新有效日）｜`stride`（矢量抽样步长，默认 5，范围 1–10）

```jsonc
{
  "kind": "observed",
  "date": "2025-06-26",
  "grid": { "rows": 116, "cols": 158, "cell_km": 25.07 },
  "row_axis": [0, 1],          // 116 项
  "col_axis": [0, 1],          // 158 项
  "row_lat": [ /* 116 项，按中列取纬度 */ ],
  "col_lon": [ /* 158 项，按中行取经度 */ ],
  "speed": [ [null, 3.2], [] ],      // 116×158，无效像元 null，保留 2 位小数
  "vectors": {                        // 抽样后的矢量箭头
    "stride": 5,
    "rows": [0], "cols": [0], "u": [1.2], "v": [0.3], "speed": [1.24]
  },
  "stats": { "valid_cells": 7568, "coverage": 0.413, "mean_speed": 12.3, "max_speed": 42.1 },
  "data_time": { /* DataTime */ }
}
```

## `GET /api/field/predicted`

预测场（地图切换「预测」时使用）。

Query：`model`（必填，任意已可用模型 id）｜`date`（必填）

```jsonc
{
  "kind": "predicted",
  "model": { "id": "random_forest", "name": "Random Forest 随机森林" },
  "date": "2025-06-26",
  "target_date": "2025-06-27",
  "grid": { "rows": 116, "cols": 158, "cell_km": 25.07 },
  "row_axis": [], "col_axis": [], "row_lat": [], "col_lon": [],
  "speed": [], "vectors": { "stride": 5, "rows": [], "cols": [], "u": [], "v": [], "speed": [] },
  "stats": {
    "valid_cells": 7568, "coverage": 0.413, "mean_speed": 11.9, "max_speed": 40.8,
    "comparison": { "n": 7568, "vec_rmse": 10.2 }      // 若目标帧真值存在，给出该日预测场误差；否则 null
  },
  "elapsed_ms": 812.0,
  "warnings": []
}
```

## `POST /api/predict`

**真实推理接口**。请求指定模型 + 最后观测日 + 像元；后端读取该日期前的真实历史数据、
构造特征/输入场、调用**已训练模型**推理（不训练、不写模型文件），返回 JSON。

### 请求

```jsonc
{
  "model": "random_forest",       // 必填；/api/models 中 available=true 的 id
  "date": "2025-06-26",           // 必填；最后观测日（YYYY-MM-DD，须存在于数据集）
  "cell": { "row": 65, "col": 60 } // 必填；row 0–115，col 0–157
}
```

### 响应 200（成功）

```jsonc
{
  "status": "ok",
  "status_text": "预测成功",
  "request_id": "b3f8c2d1",
  "elapsed_ms": 42.7,
  "model": { "id": "random_forest", "name": "Random Forest 随机森林", "category": "tree" },
  "data_time": {
    "last_obs_date": "2025-06-26",
    "last_obs_index": 17000,
    "next_frame_date": "2025-06-27",
    "horizon_hours": 24,
    "latest_date": "2025-12-30",
    "n_frames": 17188
  },
  "input": {
    "row": 65, "col": 60, "lat": -68.04, "lon": -33.2,
    "date": "2025-06-26",
    "u": 17.21, "v": 5.04, "speed": 17.93, "direction_deg": 16.3,   // 矢量方向（度，0=东，逆时针）
    "valid": true, "coverage": 0.413
  },
  "prediction": {
    "u": 16.72, "v": 4.81, "speed": 17.40, "direction_deg": 16.1,
    "unit": "cm/s",
    "displacement_km_24h": 15.03,
    "bearing_deg": 16.1,
    "delta_speed_vs_persistence": -0.53
  },
  "series_24h": {
    "hours": [0, 1, 2, "...", 24],          // 25 项整数
    "cumulative_km": [0.0, 0.63, "...", 15.03],   // 25 项
    "speed_cm_s": [17.4, "..."],                  // 25 项（日尺度均值→恒定）
    "row": [65.0, "..."], "col": [60.0, "..."],   // 25 项，逐时像元位置
    "lat": [-68.04, "..."], "lon": [-33.2, "..."],// 25 项
    "uncertainty": {
      "method": "RF 树间离散度 10–90 分位",
      "lower_km": [0.0, "..."], "upper_km": [0.0, "..."]   // 25 项；无不确定度模型返回 null
    },
    "interpolation": "日尺度产品：逐时位置/位移由 24h 平均漂移矢量线性积分",
    "target_date": "2025-06-27"
  },
  "metrics": { "vec_rmse": 10.15, "skill_vs_persistence": 0.07 },   // 该模型统一口径指标（可 null）
  "warnings": [
    { "code": "high_speed", "level": "warning", "message": "预测漂移速度 17.4 cm/s 高于该像元历史 90% 分位" }
  ]
}
```

### 错误

| HTTP | code 示例 | 场景 |
|---|---|---|
| 400 | `invalid_cell` | row/col 越界或非整数 |
| 404 | `unknown_model` | 模型 id 不存在 |
| 409 | `model_unavailable` | 模型 artifact 缺失/加载失败（如 simvp 未就绪） |
| 422 | `date_invalid` / `date_not_in_dataset` | 日期格式错 / 数据集无该日 |
| 422 | `insufficient_history` | 该日之前历史帧不足（SimVP 需 10 帧） |
| 500 | `inference_failed` | 推理异常（含 traceback 摘要写入日志，不返回前端） |

错误 body 见「通用对象 ErrorResponse」。

## `POST /api/backtest`

历史回测：真实值 vs 模型预测（逐日）。返回的 `series[i].date` 是**目标日**（预测日），
`obs_*` 是目标日真值，`pred_*` 是模型从上一帧出发的预测。

### 请求

```jsonc
{
  "model": "ridge",
  "cell": { "row": 65, "col": 60 },
  "start": "2025-01-01",      // 可选，默认 end 前 180 天
  "end": "2025-12-30",        // 可选，默认最新日
  "max_points": 180           // 可选，返回点数上限（1–365）
}
```

### 响应 200

```jsonc
{
  "status": "ok",
  "model": { "id": "ridge", "name": "Ridge 岭回归" },
  "cell": { "row": 65, "col": 60, "lat": -68.04, "lon": -33.2 },
  "range": {
    "requested_start": "2025-01-01", "requested_end": "2025-12-30",
    "available_start": "1978-11-08", "available_end": "2025-12-30",
    "n_points": 180, "n_days_total": 357
  },
  "series": [
    { "date": "2025-01-02", "obs_u": 12.1, "obs_v": 3.2, "obs_speed": 12.5,
      "pred_u": 11.8, "pred_v": 3.4, "pred_speed": 12.3, "err_vec": 0.45 }
  ],
  "summary": {
    "n": 180, "rmse_u": 6.1, "rmse_v": 5.8, "vec_rmse": 8.4,
    "mae_vec": 6.7, "r2_u": 0.31, "r2_v": 0.29, "skill_vs_persistence": 0.02
  },
  "warnings": []
}
```

错误：`unknown_model`(404)、`model_unavailable`(409)、`invalid_cell`(400)、`invalid_range`(422)、
`backtest_out_of_range`(422，仅 SimVP 请求 2022 之前日期时返回，并附可用范围)。

## 静态资源 / 前端

- `GET /` → `dashboard/dist/index.html`（构建产物；若未构建返回 JSON 提示与构建命令）。
- `GET /assets/*` → 前端静态资源；SPA 路由 fallback 到 `index.html`。
- CORS：允许 `http://127.0.0.1:5173`、`http://localhost:5173`（Vite dev）。
- OpenAPI：`GET /docs`、`GET /openapi.json`。

## 前端交互约定

1. 首屏：`GET /api/models`（填充模型选择 + 数据时间 + 默认预测参数）、`GET /api/overview`（KPI/趋势）、
   `GET /api/metrics`（对比图）、`GET /api/field?date=默认日期`（地图）。
2. 「运行预测」：`POST /api/predict` → 更新 24h 曲线、结果卡、状态（loading→success/error）、异常提示；
   随后自动拉取 `POST /api/backtest` 更新回测面板。
3. 地图选点：点击热力图 → 更新 `cell.row/col`；地图日期切换 → `GET /api/field`；「观测/预测」切换 →
   `GET /api/field/predicted?model=...&date=...`。
4. 所有请求失败要在 UI 显示错误状态与后端 `error.message`，不得静默失败。
