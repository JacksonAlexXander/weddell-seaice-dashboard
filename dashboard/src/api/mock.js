/**
 * 开发期本地 Mock（仅通过 ?mock=1 启用，生产构建中不会被打包）。
 * 数据来源：
 *  - fixtures/field_2025-06-26.json：从 data/processed/weddell_icemotion_daily.zarr 抽取的真实
 *    观测帧（2025-06-26，116×158，有效像元 7,568），用于开发期地图/预测的视觉预览；
 *  - fixtures/overview_stats.json：从真实 zarr 聚合的 48 年年度/月度统计；
 *  - 其余（预测/回测/指标）为按 API 契约生成的确定性模拟值，仅用于视觉联调，不是真实推理结果。
 * 联调与生产环境默认走真实 FastAPI 接口（见 src/api/client.js）。
 */
import frameData from './fixtures/field_2025-06-26.json'
import overviewStats from './fixtures/overview_stats.json'
import { ApiError } from './client.js'
import { shiftDate } from '../utils/format.js'

const H = frameData.rows
const W = frameData.cols
const CELL_KM = frameData.cell_km
const FRAME_DATE = frameData.date

const DATA_TIME = {
  first_date: '1978-10-25',
  latest_date: '2025-12-30',
  n_frames: 17188,
  built_at: '2026-10-25T14:40:44Z',
  batch_age_days: 300,
  source:
    'NSIDC-0116 v4 Polar Pathfinder Daily 25km EASE-Grid Sea Ice Motion Vectors (Weddell subset)',
  doi: '10.5067/INAWUWO7QH7B',
}

const MISSING_DAYS = new Set([
  '1980-03-15', '1980-03-16', '1980-03-17', '1980-03-18', '1982-01-31', '1984-03-20',
  '1984-03-21', '1984-08-22', '1984-08-23', '1984-08-24', '1986-03-28', '1987-01-26',
  '1987-01-27', '1987-12-23', '1987-12-26', '1987-12-27', '1987-12-28', '1988-02-28',
  '1988-02-29', '1992-02-13', '1994-11-19', '1994-11-20', '1994-11-21', '1996-11-30',
  '1996-12-01', '2010-06-27', '2010-06-28', '2024-05-21', '2024-05-22', '2024-09-11',
  '2024-09-12', '2024-09-13', '2024-09-14', '2024-09-15', '2024-09-16', '2024-09-17',
])

// ---------- 真实观测帧解析 ----------
const speedGrid = frameData.speed_rows.map((row) =>
  row.split(',').map((v) => (v === '' ? null : Number(v)))
)
const maskGrid = frameData.mask.map((row) => row.split('').map((v) => v === '1'))
const uGrid = frameData.u
const vGrid = frameData.v
const rowLat = frameData.row_lat
const colLon = frameData.col_lon
const rowAxis = Array.from({ length: H }, (_, i) => i)
const colAxis = Array.from({ length: W }, (_, j) => j)

let validCells = 0
let speedSum = 0
let maxSpeed = 0
for (let r = 0; r < H; r += 1) {
  for (let c = 0; c < W; c += 1) {
    const s = speedGrid[r][c]
    if (s !== null) {
      validCells += 1
      speedSum += s
      if (s > maxSpeed) maxSpeed = s
    }
  }
}
const frameStats = {
  valid_cells: validCells,
  coverage: Number((validCells / (H * W)).toFixed(4)),
  mean_speed: Number((speedSum / Math.max(validCells, 1)).toFixed(2)),
  max_speed: Number(maxSpeed.toFixed(2)),
}

// ---------- 模型 / 指标 ----------
const MODEL_DEFS = [
  {
    id: 'ridge',
    name: 'Ridge 岭回归',
    short: 'Ridge',
    category: 'linear',
    color: '#22d3ee',
    available: true,
    description: '标准化特征上的多输出线性回归，推理毫秒级，可解释性强。',
    trained_at: '2026-10-26T03:12:00Z',
    training_samples: 2143000,
    latency_ms: 0.8,
  },
  {
    id: 'random_forest',
    name: 'Random Forest 随机森林',
    short: 'Random Forest',
    category: 'tree',
    color: '#a78bfa',
    available: true,
    description: '多输出随机森林（250 棵树），对非线性漂移响应更敏感，树间离散度给出不确定度。',
    trained_at: '2026-10-26T03:15:00Z',
    training_samples: 2143000,
    latency_ms: 34.2,
  },
  {
    id: 'simvp',
    name: 'SimVP-v1 深度学习',
    short: 'SimVP',
    category: 'deep',
    color: '#f59e0b',
    available: true,
    description: '项目既有 9.3M 参数时空预测模型，输入 10 帧输出 5 天场，取 day-1 预测。',
    trained_at: '2026-10-26T00:20:00Z',
    training_samples: null,
    latency_ms: 320.0,
  },
  {
    id: 'persistence',
    name: '持续性基线',
    short: 'Persistence',
    category: 'baseline',
    color: '#64748b',
    available: true,
    description: '下一帧 = 当前帧（海冰漂移短期预测的强基准）。',
    trained_at: null,
    training_samples: null,
    latency_ms: 0.1,
  },
  {
    id: 'climatology',
    name: '气候态基线',
    short: 'Climatology',
    category: 'baseline',
    color: '#475569',
    available: true,
    description: '1978–2017 逐月平均漂移场（每像元）。',
    trained_at: null,
    training_samples: null,
    latency_ms: 0.1,
  },
]

const METRIC_VALUES = {
  ridge: {
    vec_rmse: 10.62, rmse_u: 7.55, rmse_v: 7.47, mae_vec: 8.71,
    r2_u: 0.42, r2_v: 0.4, skill_vs_persistence: 0.03,
    params: { alpha: 1.0 }, training_samples: 2143000,
  },
  random_forest: {
    vec_rmse: 10.15, rmse_u: 7.22, rmse_v: 7.14, mae_vec: 8.3,
    r2_u: 0.46, r2_v: 0.44, skill_vs_persistence: 0.07,
    params: { n_estimators: 250, max_depth: null, min_samples_leaf: 4 },
    training_samples: 2143000,
  },
  simvp: {
    vec_rmse: 9.79, rmse_u: 7.05, rmse_v: 6.79, mae_vec: 7.95,
    r2_u: 0.49, r2_v: 0.47, skill_vs_persistence: 0.1,
    params: { tin: 10, tout: 5, day1: true }, training_samples: null,
  },
  persistence: {
    vec_rmse: 10.91, rmse_u: 7.72, rmse_v: 7.71, mae_vec: 8.94,
    r2_u: 0.39, r2_v: 0.37, skill_vs_persistence: 0.0,
    params: { type: 'baseline' }, training_samples: null,
  },
  climatology: {
    vec_rmse: 12.36, rmse_u: 8.74, rmse_v: 8.74, mae_vec: 9.98,
    r2_u: 0.22, r2_v: 0.2, skill_vs_persistence: -0.13,
    params: { years: '1978-2017' }, training_samples: null,
  },
}

function modelById(id) {
  return MODEL_DEFS.find((m) => m.id === id) || null
}

function modelListPayload() {
  return {
    data_time: DATA_TIME,
    default: { model: 'random_forest', date: FRAME_DATE, cell: { row: 65, col: 60 } },
    models: MODEL_DEFS.map((m) => ({
      id: m.id,
      name: m.name,
      short: m.short,
      category: m.category,
      color: m.color,
      available: m.available,
      description: m.description,
      trained_at: m.trained_at,
      training_samples: m.training_samples,
      metrics: {
        vec_rmse: METRIC_VALUES[m.id].vec_rmse,
        skill_vs_persistence: METRIC_VALUES[m.id].skill_vs_persistence,
        latency_ms: m.latency_ms,
      },
      is_baseline: m.category === 'baseline',
    })),
    combos: [
      { model: 'random_forest', date: FRAME_DATE, cell: { row: 65, col: 60 } },
      { model: 'ridge', date: FRAME_DATE, cell: { row: 65, col: 60 } },
      { model: 'simvp', date: FRAME_DATE, cell: { row: 78, col: 90 } },
    ],
  }
}

function metricsPayload() {
  return {
    protocol: {
      task: '未来 24 小时（下一日帧）海冰漂移矢量预测',
      target: 'u(t+1), v(t+1)，cm/s，25 km EASE-Grid South 像元',
      split: { train: '1978–2017', val: '2018–2021', test: '2022–2025' },
      sampling: 'test split 每 5 日一个起报窗口（287 个窗口），评估全部有效像元',
      n_windows: 287,
      n_samples: 1556493,
      metric_defs: {
        vec_rmse: '√mean(Δu²+Δv²)，cm/s',
        mae_vec: 'mean(√(Δu²+Δv²))，cm/s',
        skill_vs_persistence: '1 − RMSE_model/RMSE_persistence（矢量 RMSE 比口径）',
        r2_u: 'u 分量决定系数',
      },
    },
    models: MODEL_DEFS.map((m) => ({
      id: m.id,
      name: m.name,
      name_zh: m.name,
      short: m.short,
      category: m.category,
      color: m.color,
      ...METRIC_VALUES[m.id],
      latency_ms: m.latency_ms,
      is_highlight: m.id === 'ridge' || m.id === 'random_forest',
      is_baseline: m.category === 'baseline',
    })),
    generated_at: '2026-10-26T06:00:00Z',
  }
}

function overviewPayload() {
  return {
    dataset: {
      name: 'NSIDC-0116 v4 威德尔海海冰运动数据集',
      region: '60°W–20°E, 60°S–83°S',
      grid: { rows: H, cols: W, cell_km: CELL_KM },
      n_frames: DATA_TIME.n_frames,
      first_date: DATA_TIME.first_date,
      latest_date: DATA_TIME.latest_date,
      years: 48,
      missing_days: 36,
      mean_valid_fraction: overviewStats.mean_valid_fraction,
      mean_speed_cm_s: overviewStats.mean_speed_cm_s,
      u_units: 'cm/s',
      source: 'NSIDC-0116 v4',
    },
    kpis: [
      { id: 'years', label: '覆盖年限', value: 48, unit: '年' },
      { id: 'days', label: '有效日数', value: DATA_TIME.n_frames, unit: '天' },
      { id: 'mean_speed', label: '平均漂移速度', value: overviewStats.mean_speed_cm_s, unit: 'cm/s' },
      {
        id: 'coverage',
        label: '平均有效覆盖',
        value: Number((overviewStats.mean_valid_fraction * 100).toFixed(1)),
        unit: '%',
      },
    ],
    annual: overviewStats.annual,
    monthly: overviewStats.monthly,
    year_month: overviewStats.year_month,
    data_time: DATA_TIME,
  }
}

// ---------- 场数据 ----------
function modelFactor(modelId) {
  return { ridge: 0.985, random_forest: 0.965, simvp: 0.945, persistence: 1.0, climatology: 0.82 }[
    modelId
  ] || 1
}

function buildFieldPayload({ kind, date, modelId, stride = 5 }) {
  const strideN = Math.max(1, Math.min(10, Number(stride) || 5))
  const factor = kind === 'predicted' ? modelFactor(modelId) : 1
  const speed = new Array(H)
  for (let r = 0; r < H; r += 1) {
    const row = new Array(W)
    for (let c = 0; c < W; c += 1) {
      const s = speedGrid[r][c]
      if (s === null) {
        row[c] = null
      } else if (kind === 'predicted') {
        const mod = 1 + 0.06 * Math.sin((r + c) * 0.11 + modelId.length) + 0.03 * Math.cos(r * 0.23)
        row[c] = Number((s * factor * mod).toFixed(2))
      } else {
        row[c] = Number(s.toFixed(2))
      }
    }
    speed[r] = row
  }

  const vRows = []
  const vCols = []
  const vU = []
  const vV = []
  const vSpeed = []
  for (let r = 0; r < H; r += strideN) {
    for (let c = 0; c < W; c += strideN) {
      const u = uGrid[r][c]
      const v = vGrid[r][c]
      if (u === null || v === null) continue
      const mod = kind === 'predicted' ? factor : 1
      vRows.push(r)
      vCols.push(c)
      vU.push(Number((u * mod).toFixed(2)))
      vV.push(Number((v * mod).toFixed(2)))
      vSpeed.push(Number(Math.hypot(u * mod, v * mod).toFixed(2)))
    }
  }

  let sum = 0
  let mx = 0
  let n = 0
  for (let r = 0; r < H; r += 1) {
    for (let c = 0; c < W; c += 1) {
      const s = speed[r][c]
      if (s !== null) {
        sum += s
        n += 1
        if (s > mx) mx = s
      }
    }
  }

  const targetDate = shiftDate(date, 1)
  const payload = {
    kind,
    date,
    grid: { rows: H, cols: W, cell_km: CELL_KM },
    row_axis: rowAxis,
    col_axis: colAxis,
    row_lat: rowLat,
    col_lon: colLon,
    speed,
    vectors: {
      stride: strideN,
      rows: vRows,
      cols: vCols,
      u: vU,
      v: vV,
      speed: vSpeed,
    },
    stats: {
      valid_cells: n,
      coverage: Number((n / (H * W)).toFixed(4)),
      mean_speed: Number((sum / Math.max(n, 1)).toFixed(2)),
      max_speed: Number(mx.toFixed(2)),
    },
    data_time: DATA_TIME,
  }
  if (kind === 'predicted') {
    payload.model = { id: modelId, name: modelById(modelId)?.name || modelId }
    payload.target_date = targetDate
    payload.stats.comparison = {
      n,
      vec_rmse: Number((METRIC_VALUES[modelId]?.vec_rmse ?? 10.5).toFixed(2)),
    }
    payload.elapsed_ms = Number((modelById(modelId)?.latency_ms || 40).toFixed(1))
    payload.warnings = []
  }
  return payload
}

// ---------- 单点预测 ----------
function hashStr(str) {
  let h = 2166136261
  for (let i = 0; i < str.length; i += 1) {
    h ^= str.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return h >>> 0
}

function det(seedStr, lo = -1, hi = 1) {
  const v = (hashStr(seedStr) % 10000) / 10000
  return lo + v * (hi - lo)
}

function interp(arr, x) {
  const i = Math.max(0, Math.min(arr.length - 2, Math.floor(x)))
  const t = Math.max(0, Math.min(1, x - i))
  return arr[i] * (1 - t) + arr[i + 1] * t
}

const UNCERTAINTY_FACTOR = {
  ridge: 0.26,
  random_forest: 0.36,
  simvp: 0.32,
  persistence: null,
  climatology: null,
}

function predictPayload(body) {
  const { model, date, cell } = body || {}
  const m = modelById(model)
  if (!m) throw new ApiError(`未知模型 id: ${model}`, { code: 'unknown_model', status: 404 })
  if (!m.available) {
    throw new ApiError(`模型 ${m.name} 当前不可用（artifact 缺失或加载失败）`, {
      code: 'model_unavailable',
      status: 409,
    })
  }
  const row = Number(cell?.row)
  const col = Number(cell?.col)
  if (!Number.isInteger(row) || !Number.isInteger(col) || row < 0 || row >= H || col < 0 || col >= W) {
    throw new ApiError('像元 row/col 越界或非整数', {
      code: 'invalid_cell',
      status: 400,
      hint: `row 0–${H - 1}，col 0–${W - 1}`,
    })
  }
  assertDate(date)

  const u0 = uGrid[row][col] || 0
  const v0 = vGrid[row][col] || 0
  const obsSpeed = Math.hypot(u0, v0)
  const lat = rowLat[row]
  const lon = colLon[col]
  const seed = `${model}|${date}|${row}|${col}`
  const jitterU = det(`${seed}|u`, -1.4, 1.4)
  const jitterV = det(`${seed}|v`, -1.4, 1.4)

  let predU = u0
  let predV = v0
  if (model === 'climatology') {
    predU = u0 * 0.55 + 0.6 + jitterU
    predV = v0 * 0.55 + 0.3 + jitterV
  } else if (model !== 'persistence') {
    const shrink = model === 'random_forest' ? 0.93 : model === 'simvp' ? 0.91 : 0.96
    predU = u0 * shrink + jitterU
    predV = v0 * shrink + jitterV
  }
  predU = Number(predU.toFixed(2))
  predV = Number(predV.toFixed(2))
  const predSpeed = Number(Math.hypot(predU, predV).toFixed(2))
  const dispKm = Number((predSpeed * 0.036 * 24).toFixed(2))
  const dirDeg = ((Math.atan2(predV, predU) * 180) / Math.PI + 360) % 360
  const bearingDeg = ((Math.atan2(predU, predV) * 180) / Math.PI + 360) % 360
  const obsDir = ((Math.atan2(v0, u0) * 180) / Math.PI + 360) % 360

  const hours = Array.from({ length: 25 }, (_, i) => i)
  const cum = hours.map((h) => Number(((predSpeed * 0.036 * h) / 1).toFixed(2)))
  const speedSeries = hours.map(() => Number(predSpeed.toFixed(2)))
  const dRowPerDay = predV * 0.034466
  const dColPerDay = predU * 0.034466
  const rowSeries = hours.map((h) => Number((row + (dRowPerDay * h) / 24).toFixed(3)))
  const colSeries = hours.map((h) => Number((col + (dColPerDay * h) / 24).toFixed(3)))
  const latSeries = rowSeries.map((x) => Number(interp(rowLat, x).toFixed(4)))
  const lonSeries = colSeries.map((x) => Number(interp(colLon, x).toFixed(4)))

  const uncFactor = UNCERTAINTY_FACTOR[model]
  let uncertainty = null
  if (uncFactor !== null && uncFactor !== undefined) {
    const sigma24 = METRIC_VALUES[model].vec_rmse * uncFactor * 0.864
    uncertainty = {
      method:
        model === 'random_forest'
          ? 'RF 树间离散度 10–90 分位'
          : model === 'simvp'
            ? 'test day-1 vec RMSE 经验不确定度'
            : 'val 集残差 10–90 分位',
      lower_km: hours.map((h) => Number(Math.max(0, cum[h] - (sigma24 * h) / 24).toFixed(2))),
      upper_km: hours.map((h) => Number((cum[h] + (sigma24 * h) / 24).toFixed(2))),
    }
  }

  const warnings = []
  if (predSpeed > 15) {
    warnings.push({
      code: 'high_speed',
      level: 'warning',
      message: `预测漂移速度 ${predSpeed.toFixed(1)} cm/s 高于该像元历史 90% 分位（约 15.0 cm/s），可能为极端事件`,
    })
  }
  if (DATA_TIME.batch_age_days > 30) {
    warnings.push({
      code: 'data_batch_age',
      level: 'info',
      message: `数据集最新日期距当前 ${DATA_TIME.batch_age_days} 天，属归档数据批次（不影响历史预测复现）`,
    })
  }
  if (date !== DATA_TIME.latest_date) {
    warnings.push({
      code: 'historical_date',
      level: 'info',
      message: `当前使用历史观测日 ${date} 起报（最新可用观测为 ${DATA_TIME.latest_date}）`,
    })
  }

  const nextDate = shiftDate(date, 1)
  const nextOk = nextDate <= DATA_TIME.latest_date
  return {
    status: 'ok',
    status_text: nextOk ? '预测成功' : '预测成功（目标帧为外推日）',
    request_id: hashStr(seed).toString(16).slice(0, 8),
    elapsed_ms: Number((m.latency_ms + 1.5 + Math.abs(det(seed, 0, 4))).toFixed(1)),
    model: { id: m.id, name: m.name, category: m.category },
    data_time: {
      last_obs_date: date,
      last_obs_index: 17000,
      next_frame_date: nextDate,
      horizon_hours: 24,
      latest_date: DATA_TIME.latest_date,
      n_frames: DATA_TIME.n_frames,
    },
    input: {
      row,
      col,
      lat: Number(lat.toFixed(3)),
      lon: Number(lon.toFixed(3)),
      date,
      u: Number(u0.toFixed(2)),
      v: Number(v0.toFixed(2)),
      speed: Number(obsSpeed.toFixed(2)),
      direction_deg: Number(obsDir.toFixed(1)),
      valid: speedGrid[row][col] !== null,
      coverage: frameStats.coverage,
    },
    prediction: {
      u: predU,
      v: predV,
      speed: predSpeed,
      direction_deg: Number(dirDeg.toFixed(1)),
      unit: 'cm/s',
      displacement_km_24h: dispKm,
      bearing_deg: Number(bearingDeg.toFixed(1)),
      delta_speed_vs_persistence: Number((predSpeed - obsSpeed).toFixed(2)),
    },
    series_24h: {
      hours,
      cumulative_km: cum,
      speed_cm_s: speedSeries,
      row: rowSeries,
      col: colSeries,
      lat: latSeries,
      lon: lonSeries,
      uncertainty,
      interpolation: '日尺度产品：逐时位置/位移由 24h 平均漂移矢量线性积分',
      target_date: nextDate,
    },
    metrics: {
      vec_rmse: METRIC_VALUES[model].vec_rmse,
      skill_vs_persistence: METRIC_VALUES[model].skill_vs_persistence,
    },
    warnings,
  }
}

// ---------- 历史回测 ----------
function assertDate(date) {
  if (!date || !/^\d{4}-\d{2}-\d{2}$/.test(String(date))) {
    throw new ApiError('日期格式错误，应为 YYYY-MM-DD', { code: 'date_invalid', status: 422 })
  }
  if (date < DATA_TIME.first_date || date > DATA_TIME.latest_date) {
    throw new ApiError('日期超出数据集范围', {
      code: 'date_out_of_range',
      status: 422,
      hint: `可选日期 ${DATA_TIME.first_date} ~ ${DATA_TIME.latest_date}`,
    })
  }
  if (MISSING_DAYS.has(date)) {
    throw new ApiError(`数据集在 ${date} 无有效观测帧`, {
      code: 'date_not_in_dataset',
      status: 422,
      hint: '该日属于已知缺失日，请选择相邻有效日期',
    })
  }
}

function dayOfYear(dateStr) {
  const d = new Date(`${dateStr}T00:00:00Z`)
  const start = Date.UTC(d.getUTCFullYear(), 0, 1)
  return Math.floor((d.getTime() - start) / 86400000) + 1
}

function backtestPayload(body) {
  const { model, cell, start, end, max_points } = body || {}
  const m = modelById(model)
  if (!m) throw new ApiError(`未知模型 id: ${model}`, { code: 'unknown_model', status: 404 })
  if (!m.available) {
    throw new ApiError(`模型 ${m.name} 当前不可用`, { code: 'model_unavailable', status: 409 })
  }
  const row = Number(cell?.row)
  const col = Number(cell?.col)
  if (!Number.isInteger(row) || !Number.isInteger(col) || row < 0 || row >= H || col < 0 || col >= W) {
    throw new ApiError('像元 row/col 越界或非整数', { code: 'invalid_cell', status: 400 })
  }

  const availableStart = '1978-11-08'
  const availableEnd = DATA_TIME.latest_date
  let reqStart = start || shiftDate(availableEnd, -180)
  let reqEnd = end || availableEnd
  if (!/^\d{4}-\d{2}-\d{2}$/.test(reqStart) || !/^\d{4}-\d{2}-\d{2}$/.test(reqEnd) || reqStart > reqEnd) {
    throw new ApiError('回测区间非法：start 必须早于 end', { code: 'invalid_range', status: 422 })
  }
  const warnings = []
  if (model === 'simvp' && reqStart < '2022-01-10') {
    throw new ApiError('SimVP 回测仅覆盖 test 期（2022–2025）', {
      code: 'backtest_out_of_range',
      status: 422,
      hint: '可用范围 2022-01-10 ~ 2025-12-30',
    })
  }
  if (reqStart < availableStart) {
    reqStart = availableStart
    warnings.push({
      code: 'range_clamped',
      level: 'info',
      message: `请求起始日早于可用范围，已裁剪到 ${availableStart}`,
    })
  }
  if (reqEnd > availableEnd) {
    reqEnd = availableEnd
    warnings.push({
      code: 'range_clamped',
      level: 'info',
      message: `请求结束日晚于数据集最新日期，已裁剪到 ${availableEnd}`,
    })
  }

  const totalDays = Math.round(
    (new Date(`${reqEnd}T00:00:00Z`) - new Date(`${reqStart}T00:00:00Z`)) / 86400000
  )
  const limit = Math.max(1, Math.min(365, Number(max_points) || 180))
  const nPoints = Math.min(limit, totalDays + 1)
  const offsets = []
  if (nPoints >= totalDays + 1) {
    for (let i = 0; i <= totalDays; i += 1) offsets.push(i)
  } else {
    for (let k = 0; k < nPoints; k += 1) offsets.push(Math.round((k * totalDays) / (nPoints - 1)))
    offsets[offsets.length - 1] = totalDays
  }

  const series = []
  const rngSeed = hashStr(`${model}|${row}|${col}|${reqStart}|${reqEnd}`)
  let state = rngSeed
  const rnd = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0
    return state / 4294967296
  }
  const gauss = () => {
    const u1 = Math.max(rnd(), 1e-9)
    const u2 = rnd()
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
  }

  // AR(1) 逐日矢量扰动：使「持续性基线」的矢量 RMSE 量级 ≈ 10–11 cm/s（与真实数据一致）
  const AR = 0.72
  const INNO = 7.3
  const bearingBase = (108 * Math.PI) / 180
  let nu = 0
  let nv = 0
  let prevU = null
  let prevV = null
  const obsPairs = []
  const predPairs = []
  const persPairs = []

  for (const i of offsets) {
    const d = new Date(`${reqStart}T00:00:00Z`)
    d.setUTCDate(d.getUTCDate() + i)
    const dateStr = d.toISOString().slice(0, 10)
    const doy = dayOfYear(dateStr)
    const seasonalSpeed = 8.1 + 3.7 * Math.sin((2 * Math.PI * (doy - 205)) / 365)
    nu = AR * nu + INNO * gauss()
    nv = AR * nv + INNO * gauss()
    const obsU = seasonalSpeed * Math.sin(bearingBase) + nu
    const obsV = seasonalSpeed * Math.cos(bearingBase) + nv
    const obsSpeedR = Math.hypot(obsU, obsV)

    let predU
    let predV
    if (model === 'persistence') {
      predU = prevU ?? obsU
      predV = prevV ?? obsV
    } else if (model === 'climatology') {
      // 月气候态：无逐日信息，误差为全部逐日扰动
      predU = seasonalSpeed * Math.sin(bearingBase)
      predV = seasonalSpeed * Math.cos(bearingBase)
    } else {
      const mv = METRIC_VALUES[model]
      predU = obsU + gauss() * mv.rmse_u
      predV = obsV + gauss() * mv.rmse_v
    }
    const predSpeed = Math.hypot(predU, predV)
    const errVec = Math.hypot(obsU - predU, obsV - predV)

    series.push({
      date: dateStr,
      obs_u: Number(obsU.toFixed(2)),
      obs_v: Number(obsV.toFixed(2)),
      obs_speed: Number(obsSpeedR.toFixed(2)),
      pred_u: Number(predU.toFixed(2)),
      pred_v: Number(predV.toFixed(2)),
      pred_speed: Number(predSpeed.toFixed(2)),
      err_vec: Number(errVec.toFixed(2)),
    })
    obsPairs.push([obsU, obsV])
    predPairs.push([predU, predV])
    if (prevU !== null) persPairs.push([prevU, prevV])
    prevU = obsU
    prevV = obsV
  }

  const mean = (arr) => arr.reduce((a, b) => a + b, 0) / Math.max(arr.length, 1)
  const errVecs = obsPairs.map(([ou, ov], idx) =>
    Math.hypot(ou - predPairs[idx][0], ov - predPairs[idx][1])
  )
  const maeVecVal = mean(errVecs)
  const rmseU = Math.sqrt(mean(obsPairs.map(([ou], i) => (ou - predPairs[i][0]) ** 2)))
  const rmseV = Math.sqrt(mean(obsPairs.map(([, ov], i) => (ov - predPairs[i][1]) ** 2)))
  const vecRmse = Math.sqrt(mean(obsPairs.map(([ou, ov], i) => (ou - predPairs[i][0]) ** 2 + (ov - predPairs[i][1]) ** 2)))
  const persRmse = persPairs.length
    ? Math.sqrt(
        mean(
          persPairs.map(([pu, pv], i) => {
            const [ou, ov] = obsPairs[i + 1]
            return (ou - pu) ** 2 + (ov - pv) ** 2
          })
        )
      )
    : vecRmse
  const obsU = obsPairs.map(([u]) => u)
  const predUArr = predPairs.map(([u]) => u)
  const obsV = obsPairs.map(([, v]) => v)
  const predVArr = predPairs.map(([, v]) => v)
  const r2 = (o, p) => {
    const mo = mean(o)
    const ssTot = o.reduce((a, b) => a + (b - mo) ** 2, 0)
    const ssRes = o.reduce((a, b, i) => a + (b - p[i]) ** 2, 0)
    return ssTot > 0 ? 1 - ssRes / ssTot : 0
  }

  return {
    status: 'ok',
    model: { id: m.id, name: m.name },
    cell: { row, col, lat: Number(rowLat[row].toFixed(3)), lon: Number(colLon[col].toFixed(3)) },
    range: {
      requested_start: start || reqStart,
      requested_end: end || reqEnd,
      available_start: availableStart,
      available_end: availableEnd,
      n_points: series.length,
      n_days_total: totalDays,
    },
    series,
    summary: {
      n: series.length,
      rmse_u: Number(rmseU.toFixed(2)),
      rmse_v: Number(rmseV.toFixed(2)),
      vec_rmse: Number(vecRmse.toFixed(2)),
      mae_vec: Number(maeVecVal.toFixed(2)),
      r2_u: Number(r2(obsU, predUArr).toFixed(3)),
      r2_v: Number(r2(obsV, predVArr).toFixed(3)),
      skill_vs_persistence: Number((1 - vecRmse / Math.max(persRmse, 1e-6)).toFixed(3)),
    },
    warnings,
  }
}

// ---------- 其他 ----------
function healthPayload() {
  return {
    status: 'ok',
    service: 'weddell-seaice-ml',
    version: '1.0.0',
    uptime_s: Number((performance.now() / 1000).toFixed(1)),
    models_loaded: MODEL_DEFS.map((m) => m.id),
    artifacts_ok: true,
  }
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

export async function handleMock(method, path, body) {
  await sleep(60 + Math.random() * 110)
  const url = new URL(path, 'http://mock.local')
  const p = url.pathname
  const q = url.searchParams

  if (method === 'GET' && p === '/api/health') return healthPayload()
  if (method === 'GET' && p === '/api/models') return modelListPayload()
  if (method === 'GET' && p === '/api/metrics') return metricsPayload()
  if (method === 'GET' && p === '/api/overview') return overviewPayload()
  if (method === 'GET' && p === '/api/field') {
    const date = q.get('date') || FRAME_DATE
    assertDate(date)
    return buildFieldPayload({ kind: 'observed', date, stride: q.get('stride') })
  }
  if (method === 'GET' && p === '/api/field/predicted') {
    const date = q.get('date') || FRAME_DATE
    const model = q.get('model')
    if (!model) throw new ApiError('缺少 model 参数', { code: 'missing_param', status: 400 })
    if (!modelById(model)) throw new ApiError(`未知模型 id: ${model}`, { code: 'unknown_model', status: 404 })
    assertDate(date)
    return buildFieldPayload({ kind: 'predicted', modelId: model, date, stride: q.get('stride') })
  }
  if (method === 'POST' && p === '/api/predict') return predictPayload(body)
  if (method === 'POST' && p === '/api/backtest') return backtestPayload(body)

  throw new ApiError(`Mock 未实现的端点：${method} ${p}`, { code: 'mock_not_implemented', status: 404 })
}
