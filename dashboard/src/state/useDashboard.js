import { computed, reactive } from 'vue'
import { api, describeError, usingMock } from '../api/client.js'
import { clamp, fmtClock, shiftDate } from '../utils/format.js'

const FALLBACK = {
  model: 'random_forest',
  date: '2025-06-26',
  cell: { row: 65, col: 60 },
}

const RECENT_MODELS = new Set(['simvp', 'ridge', 'random_forest'])
const ROWS = 116
const COLS = 158

export function useDashboard() {
  const state = reactive({
    clock: new Date(),
    booting: true,
    booted: false,
    selectionTouched: false,
    mode: 'archive',
    leadDays: 3,
    recentBatch: null,
    recentLoading: false,
    recentError: null,

    health: { ok: false, checked: false, error: null, data: null, checkedAt: null },

    models: null,
    modelsLoading: false,
    modelsError: null,

    overview: null,
    overviewLoading: false,
    overviewError: null,

    metrics: null,
    metricsLoading: false,
    metricsError: null,

    selection: {
      model: FALLBACK.model,
      date: FALLBACK.date,
      row: FALLBACK.cell.row,
      col: FALLBACK.cell.col,
    },

    mapMode: 'observed',
    observedField: null,
    fieldLoading: false,
    fieldError: null,
    fieldAt: null,

    predictedField: null,
    predictedLoading: false,
    predictedError: null,
    predictedKey: '',

    prediction: null,
    predictStatus: 'idle', // idle | running | success | error
    predictError: null,
    predictElapsedMs: null,
    predictClientMs: null,
    predictAt: null,

    backtest: null,
    backtestLoading: false,
    backtestError: null,
    backtestAt: null,
    backtestRange: 180,
  })

  let clockTimer = 0
  let healthTimer = 0
  let fieldToken = 0
  let predFieldToken = 0
  let predictRunId = 0
  let backtestToken = 0
  let disposed = false
  let archiveSelection = null

  // ---------------- 派生数据 ----------------
  const dataTime = computed(
    () => state.models?.data_time || state.overview?.data_time || null
  )

  const modelsList = computed(() => {
    const models = state.models?.models || []
    return state.mode === 'recent' ? models.filter(m => RECENT_MODELS.has(m.id)) : models
  })
  const activeDataTime = computed(() => state.mode === 'archive' ? dataTime.value : {
    ...state.recentBatch, built_at: state.recentBatch?.downloaded_at,
  })
  const recentExpired = computed(() => Boolean(state.recentBatch?.available &&
    Date.parse(state.recentBatch.latest_at) + 3 * 86400000 <= state.clock.getTime()))

  const availableModels = computed(() => modelsList.value.filter((m) => m.available !== false))

  const selectedModel = computed(
    () =>
      modelsList.value.find((m) => m.id === state.selection.model) ||
      availableModels.value[0] ||
      { id: state.selection.model, name: state.selection.model, color: '#22d3ee', category: 'unknown' }
  )

  const currentField = computed(() =>
    state.mapMode === 'predicted' ? state.predictedField : state.observedField
  )
  const currentFieldLoading = computed(() =>
    state.mapMode === 'predicted' ? state.predictedLoading : state.fieldLoading
  )
  const currentFieldError = computed(() =>
    state.mapMode === 'predicted' ? state.predictedError : state.fieldError
  )

  const selectedLatLon = computed(() => {
    const field = state.observedField || state.predictedField
    if (!field) return null
    const lat = field.row_lat?.[state.selection.row]
    const lon = field.col_lon?.[state.selection.col]
    if (lat === undefined || lon === undefined) return null
    return { lat, lon }
  })

  /** 地图上的 24h 预测路径（仅当预测日期与地图观测日一致时显示） */
  const mapPredictionPath = computed(() => {
    const p = state.prediction
    if (!p || !p.series_24h) return null
    if (p.data_time?.last_obs_date && p.data_time.last_obs_date !== state.selection.date) return null
    return p.series_horizon || p.series_24h
  })

  const warnings = computed(() => {
    const list = [
      ...(state.prediction?.warnings || []),
      ...(state.backtest?.warnings || []),
      ...(state.predictedField?.warnings || []),
      ...(state.observedField?.warnings || []),
    ]
    const seen = new Set()
    return list.filter((w) => {
      if (!w) return false
      const key = `${w.code}|${w.message}`
      if (seen.has(key)) return false
      seen.add(key)
      return true
    })
  })

  const banner = computed(() => {
    if (usingMock) {
      return {
        level: 'info',
        text: '本地 Mock 预览模式（?mock=1，仅开发期可用）：地图基于真实观测帧，预测/回测/指标为契约模拟值；联调与生产构建默认走真实 API。',
      }
    }
    if (state.health.checked && !state.health.ok) {
      return {
        level: 'danger',
        text: `后端服务未连接：${state.health.error || '未知原因'}。请启动 FastAPI（.venv/bin/python scripts/serve_dashboard.py --port 8000）后点击「重新连接」。`,
      }
    }
    if (state.modelsError && !state.models) {
      return { level: 'danger', text: `模型列表加载失败：${state.modelsError}` }
    }
    return null
  })

  const predictStatusText = computed(() => {
    switch (state.predictStatus) {
      case 'running':
        return '推理中…'
      case 'success':
        return `成功 · ${state.predictElapsedMs ?? '—'} ms · ${state.predictAt ? fmtClock(state.predictAt).slice(11) : ''}`
      case 'error':
        return `失败 · ${state.predictError || ''}`
      default:
        return '待运行'
    }
  })

  // ---------------- 基础请求 ----------------
  async function loadHealth() {
    try {
      const res = await api.health()
      state.health = { ok: true, checked: true, error: null, data: res, checkedAt: new Date() }
    } catch (err) {
      state.health = {
        ok: false,
        checked: true,
        error: describeError(err),
        data: null,
        checkedAt: new Date(),
      }
    }
  }

  async function loadModels() {
    state.modelsLoading = true
    state.modelsError = null
    try {
      state.models = await api.models()
      applyDefaults()
    } catch (err) {
      state.modelsError = describeError(err)
    } finally {
      state.modelsLoading = false
    }
  }

  async function loadOverview() {
    state.overviewLoading = true
    state.overviewError = null
    try {
      state.overview = await api.overview()
    } catch (err) {
      state.overviewError = describeError(err)
    } finally {
      state.overviewLoading = false
    }
  }

  async function loadMetrics() {
    state.metricsLoading = true
    state.metricsError = null
    try {
      state.metrics = await api.metrics()
    } catch (err) {
      state.metricsError = describeError(err)
    } finally {
      state.metricsLoading = false
    }
  }

  async function loadField() {
    const token = ++fieldToken
    state.fieldLoading = true
    state.observedField = null
    state.fieldError = null
    try {
      if (state.mode === 'recent' && !state.recentBatch?.available) return
      const res = state.mode === 'recent' ? await api.recentField({ date: state.selection.date }) :
        await api.field({ date: state.selection.date, stride: 5 })
      if (token !== fieldToken || disposed) return
      state.observedField = res
      state.fieldAt = new Date()
    } catch (err) {
      if (token !== fieldToken || disposed) return
      state.fieldError = describeError(err)
    } finally {
      if (token === fieldToken) state.fieldLoading = false
    }
  }

  async function loadPredictedField(force = false) {
    const model = state.selection.model
    const date = state.selection.date
    const key = `${state.mode}|${model}|${date}|${state.leadDays}`
    if (!force && state.predictedField && state.predictedKey === key) return
    const token = ++predFieldToken
    state.predictedLoading = true
    state.predictedField = null
    state.predictedError = null
    try {
      if (state.mode === 'recent' && !state.recentBatch?.available) return
      const res = state.mode === 'recent'
        ? await api.recentField({ model, date, lead_days: state.leadDays })
        : await api.predictedField({ model, date, stride: 5 })
      if (token !== predFieldToken || disposed) return
      state.predictedField = res
      state.predictedKey = key
    } catch (err) {
      if (token !== predFieldToken || disposed) return
      state.predictedField = null
      state.predictedKey = ''
      state.predictedError = describeError(err)
    } finally {
      if (token === predFieldToken) state.predictedLoading = false
    }
  }

  async function loadRecent() {
    if (usingMock) return
    try {
      state.recentBatch = await api.recent()
    } catch (err) {
      state.recentError = describeError(err)
    }
  }

  function applyRecentBatch() {
    const batch = state.recentBatch
    if (!batch?.available) return
    state.selection.date = batch.latest_date
    state.selection.row = batch.default_cell.row
    state.selection.col = batch.default_cell.col
  }

  async function refreshRecent() {
    if (state.recentLoading) return
    state.recentLoading = true
    state.recentError = null
    try {
      state.recentBatch = await api.refreshRecent()
      if (state.mode === 'recent') {
        applyRecentBatch()
        clearPrediction()
        state.predictedKey = ''
        await loadField()
        if (state.mapMode === 'predicted') await loadPredictedField()
      }
    } catch (err) {
      state.recentError = describeError(err)
    } finally {
      state.recentLoading = false
    }
  }

  function setMode(mode, reload = true) {
    if (mode === state.mode) return
    if (mode === 'recent') archiveSelection = { ...state.selection }
    state.mode = mode
    if (mode === 'recent') {
      if (!RECENT_MODELS.has(state.selection.model)) {
        state.selection.model = availableModels.value[0]?.id || 'simvp'
      }
      applyRecentBatch()
    } else if (archiveSelection) {
      Object.assign(state.selection, archiveSelection)
    }
    clearPrediction()
    ++predFieldToken
    ++backtestToken
    state.backtest = null
    state.backtestAt = null
    state.backtestError = null
    state.backtestLoading = false
    state.mapMode = 'observed'
    state.predictedField = null
    state.predictedKey = ''
    state.predictedError = null
    state.predictedLoading = false
    if (reload) loadField()
  }

  function setLeadDays(days) {
    state.leadDays = Number(days)
    clearPrediction()
    if (state.mapMode === 'predicted') loadPredictedField()
  }

  // ---------------- 交互 ----------------
  function applyDefaults() {
    const res = state.models
    if (!res) return
    const def = res.default || {}
    const avail = availableModels.value
    const ids = new Set(avail.map((m) => m.id))

    if (!state.selectionTouched && state.mode === 'archive') {
      if (def.model && ids.has(def.model)) state.selection.model = def.model
      else if (avail.length && !ids.has(state.selection.model)) state.selection.model = avail[0].id
      if (def.date) state.selection.date = String(def.date).slice(0, 10)
      if (def.cell) {
        state.selection.row = clamp(def.cell.row, 0, ROWS - 1)
        state.selection.col = clamp(def.cell.col, 0, COLS - 1)
      }
    } else if (avail.length && !ids.has(state.selection.model)) {
      state.selection.model = avail[0].id
    }

    const dt = dataTime.value
    if (dt && state.mode === 'archive') {
      if (state.selection.date < dt.first_date) state.selection.date = dt.first_date
      if (state.selection.date > dt.latest_date) state.selection.date = dt.latest_date
    }
  }

  function clearPrediction() {
    ++predictRunId
    state.prediction = null
    state.predictStatus = 'idle'
    state.predictError = null
    state.predictElapsedMs = null
    state.predictClientMs = null
    state.predictAt = null
  }

  function setModel(modelId) {
    if (!modelId || modelId === state.selection.model) return
    const model = modelsList.value.find((m) => m.id === modelId)
    if (model && model.available === false) return
    state.selection.model = modelId
    state.selectionTouched = true
    clearPrediction()
    if (state.mapMode === 'predicted') loadPredictedField()
    if (state.backtest || state.backtestLoading) loadBacktest()
  }

  function setDate(date) {
    if (!date || date === state.selection.date) return
    const dt = dataTime.value
    let next = String(date).slice(0, 10)
    if (dt) {
      if (next < dt.first_date) next = dt.first_date
      if (next > dt.latest_date) next = dt.latest_date
    }
    state.selection.date = next
    state.selectionTouched = true
    clearPrediction()
    loadField()
    if (state.mapMode === 'predicted') loadPredictedField()
  }

  function setCell(row, col) {
    const nextRow = clamp(Math.round(Number(row)), 0, ROWS - 1)
    const nextCol = clamp(Math.round(Number(col)), 0, COLS - 1)
    if (nextRow === state.selection.row && nextCol === state.selection.col) return
    state.selection.row = nextRow
    state.selection.col = nextCol
    state.selectionTouched = true
    clearPrediction()
    ++backtestToken
    state.backtest = null
    state.backtestAt = null
    state.backtestError = null
    state.backtestLoading = false
  }

  function setMapMode(mode) {
    if (mode !== 'observed' && mode !== 'predicted') return
    if (state.mapMode === mode) return
    state.mapMode = mode
    if (mode === 'predicted') loadPredictedField()
  }

  function setRange(days) {
    const n = Number(days)
    if (!Number.isFinite(n) || n <= 0) return
    state.backtestRange = n
    loadBacktest()
  }

  // ---------------- 预测 ----------------
  async function runPredict() {
    if (state.predictStatus === 'running') return
    if (!state.models) {
      state.predictStatus = 'error'
      state.predictError = state.modelsError || '模型列表尚未加载'
      return
    }
    const runId = ++predictRunId
    const started = performance.now()
    state.predictStatus = 'running'
    state.predictError = null
    state.predictAt = null
    try {
      const res = await api.predict({
        model: state.selection.model,
        date: state.selection.date,
        cell: { row: state.selection.row, col: state.selection.col },
        ...(state.mode === 'recent' ? { source: 'recent', lead_days: state.leadDays } : {}),
      })
      if (runId !== predictRunId || disposed) return
      const clientMs = Math.round(performance.now() - started)
      state.prediction = res
      state.predictStatus = 'success'
      state.predictElapsedMs = res.elapsed_ms ?? clientMs
      state.predictClientMs = clientMs
      state.predictAt = new Date()
      if (state.mode === 'archive') await loadBacktest()
    } catch (err) {
      if (runId !== predictRunId || disposed) return
      state.predictStatus = 'error'
      state.predictError = describeError(err)
      state.prediction = null
    }
  }

  async function loadBacktest() {
    if (!state.models) return
    const token = ++backtestToken
    const range = state.backtestRange
    const end = dataTime.value?.latest_date || state.selection.date
    const start = shiftDate(end, 1 - range)
    state.backtestLoading = true
    state.backtest = null
    state.backtestAt = null
    state.backtestError = null
    try {
      const res = await api.backtest({
        model: state.selection.model,
        cell: { row: state.selection.row, col: state.selection.col },
        start,
        end,
        max_points: Math.min(365, Math.max(1, range)),
      })
      if (token !== backtestToken || disposed) return
      state.backtest = res
      state.backtestAt = new Date()
    } catch (err) {
      if (token !== backtestToken || disposed) return
      state.backtestError = describeError(err)
    } finally {
      if (token === backtestToken) state.backtestLoading = false
    }
  }

  // ---------------- 生命周期 ----------------
  function startTimers() {
    if (!clockTimer) {
      clockTimer = window.setInterval(() => {
        state.clock = new Date()
      }, 1000)
    }
    if (!healthTimer) {
      healthTimer = window.setInterval(() => loadHealth(), 25000)
    }
  }

  async function bootstrap() {
    if (state.booted) return
    startTimers()
    state.booting = true
    await Promise.allSettled([loadHealth(), loadModels(), loadOverview(), loadMetrics(), loadRecent()])
    if (state.recentBatch?.available && new URLSearchParams(window.location.search).get('mode') !== 'archive') {
      setMode('recent', false)
    }
    await loadField()
    state.booting = false
    state.booted = true
    if (state.models && !(state.mode === 'recent' && recentExpired.value)) await runPredict()
  }

  async function retryConnection() {
    await loadHealth()
    if (!state.health.ok) return
    await Promise.allSettled([loadModels(), loadOverview(), loadMetrics()])
    await loadField()
    if (state.models && (!state.prediction || state.predictStatus === 'error')) await runPredict()
  }

  function dispose() {
    disposed = true
    if (clockTimer) window.clearInterval(clockTimer)
    if (healthTimer) window.clearInterval(healthTimer)
    clockTimer = 0
    healthTimer = 0
  }

  return {
    state,
    usingMock,
    dataTime,
    activeDataTime,
    recentExpired,
    modelsList,
    availableModels,
    selectedModel,
    currentField,
    currentFieldLoading,
    currentFieldError,
    selectedLatLon,
    mapPredictionPath,
    warnings,
    banner,
    predictStatusText,
    actions: {
      bootstrap,
      retryConnection,
      loadHealth,
      loadModels,
      setMode,
      setLeadDays,
      refreshRecent,
      loadOverview,
      loadMetrics,
      loadField,
      loadPredictedField,
      loadBacktest,
      runPredict,
      setModel,
      setDate,
      setCell,
      setMapMode,
      setRange,
      dispose,
    },
  }
}
