// 统一 API 客户端：同源 /api（生产由 FastAPI 托管，开发由 Vite proxy 转发到 127.0.0.1:8000）
// 开发期可通过 URL 参数 ?mock=1 启用 dashboard/src/api/mock.js 本地 mock（默认关闭，
// 生产构建中 mock 分支会被 Vite 静态剔除，绝不内置假数据）。

const DEV = import.meta.env.DEV
const API_BASE = (import.meta.env.VITE_API_BASE || '').replace(/\/$/, '')
const search = typeof window !== 'undefined' ? new URLSearchParams(window.location.search) : new URLSearchParams()
const MOCK_FLAG = DEV && (search.get('mock') === '1' || (import.meta.env.VITE_USE_MOCK === '1' && search.get('mock') !== '0'))

export const usingMock = Boolean(MOCK_FLAG)

export class ApiError extends Error {
  constructor(message, { code = 'request_failed', status = 0, hint = '', path = '' } = {}) {
    super(message)
    this.name = 'ApiError'
    this.code = code
    this.status = status
    this.hint = hint
    this.path = path
  }
}

/** 把错误转成 UI 可展示的一行文本 */
export function describeError(err) {
  if (!err) return '未知错误'
  if (err instanceof ApiError) {
    const parts = []
    if (err.message) parts.push(err.message)
    if (err.code && err.code !== 'request_failed') parts.push(`[${err.code}]`)
    if (err.hint) parts.push(`· ${err.hint}`)
    return parts.join(' ')
  }
  return err.message || String(err)
}

function parseErrorPayload(payload, status, path) {
  // 契约错误体：{ status:'error', error:{ code, message, hint } }
  if (payload && payload.error && typeof payload.error === 'object') {
    const e = payload.error
    return new ApiError(e.message || `HTTP ${status}`, {
      code: e.code || 'http_error',
      hint: e.hint || '',
      status,
      path,
    })
  }
  // FastAPI 默认校验错误：{ detail: [...] } 或 { detail: '...' }
  if (payload && payload.detail !== undefined) {
    let message = ''
    if (typeof payload.detail === 'string') {
      message = payload.detail
    } else if (Array.isArray(payload.detail)) {
      message = payload.detail
        .slice(0, 3)
        .map((d) => {
          const loc = Array.isArray(d.loc) ? d.loc.filter((x) => x !== 'body').join('.') : ''
          return `${loc ? loc + ': ' : ''}${d.msg || ''}`
        })
        .join('；')
    } else {
      message = JSON.stringify(payload.detail)
    }
    return new ApiError(message || `HTTP ${status}`, { code: 'validation_error', status, path })
  }
  return new ApiError(`HTTP ${status}`, { code: 'http_error', status, path })
}

async function rawRequest(method, path, body, { timeout = 45000 } = {}) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeout)
  const init = {
    method,
    headers: { Accept: 'application/json' },
    signal: controller.signal,
  }
  if (body !== undefined) {
    init.headers['Content-Type'] = 'application/json'
    init.body = JSON.stringify(body)
  }
  try {
    const res = await fetch(`${API_BASE}${path}`, init)
    const text = await res.text()
    let payload = null
    if (text) {
      try {
        payload = JSON.parse(text)
      } catch {
        payload = null
      }
    }
    if (!res.ok) throw parseErrorPayload(payload, res.status, path)
    if (payload === null) throw new ApiError('后端返回了非 JSON 响应', { code: 'bad_response', status: res.status, path })
    return payload
  } catch (err) {
    if (err instanceof ApiError) throw err
    if (err && err.name === 'AbortError') {
      throw new ApiError(`请求超时（>${Math.round(timeout / 1000)}s）`, { code: 'timeout', path })
    }
    throw new ApiError(`无法连接后端服务：${err?.message || err}`, { code: 'network_error', path })
  } finally {
    clearTimeout(timer)
  }
}

// ---- 开发期 mock（动态 import；生产构建中该分支被静态剔除）----
let mockLoader = null
if (import.meta.env.DEV) {
  mockLoader = () => import('./mock.js')
}

async function request(method, path, body, opts = {}) {
  if (usingMock && mockLoader) {
    const mod = await mockLoader()
    return mod.handleMock(method, path, body)
  }
  return rawRequest(method, path, body, opts)
}

function qs(params) {
  const sp = new URLSearchParams()
  Object.entries(params || {}).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== '') sp.set(k, String(v))
  })
  const s = sp.toString()
  return s ? `?${s}` : ''
}

export const api = {
  health: () => request('GET', '/api/health', undefined, { timeout: 8000 }),
  models: () => request('GET', '/api/models'),
  metrics: () => request('GET', '/api/metrics'),
  overview: () => request('GET', '/api/overview'),
  field: ({ date, stride = 5 } = {}) => request('GET', `/api/field${qs({ date, stride })}`),
  predictedField: ({ model, date, stride = 5 }) =>
    request('GET', `/api/field/predicted${qs({ model, date, stride })}`, undefined, { timeout: 90000 }),
  predict: (payload) => request('POST', '/api/predict', payload, { timeout: 120000 }),
  backtest: (payload) => request('POST', '/api/backtest', payload, { timeout: 120000 }),
}
