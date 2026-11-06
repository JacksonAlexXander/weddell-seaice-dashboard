// 格式化工具：数字、日期、方向、时钟等
const DASH = '—'

export function fmt(v, digits = 2, fallback = DASH) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return fallback
  const n = Number(v)
  return n.toFixed(digits)
}

export function fmtInt(v, fallback = DASH) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return fallback
  return Math.round(Number(v)).toLocaleString('zh-CN')
}

/** 0.3094 -> 30.9 */
export function fracToPct(v, digits = 1) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return DASH
  return (Number(v) * 100).toFixed(digits)
}

/** 0.07 -> +7.0% */
export function signedPct(v, digits = 1) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return DASH
  const n = Number(v) * 100
  const sign = n > 0 ? '+' : ''
  return `${sign}${n.toFixed(digits)}%`
}

/** 0.07 -> +0.070 */
export function signedNum(v, digits = 2, fallback = DASH) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return fallback
  const n = Number(v)
  return `${n > 0 ? '+' : ''}${n.toFixed(digits)}`
}

/** '2025-06-26' -> '2025-06-26'；ISO -> 'YYYY-MM-DD' */
export function toDateStr(v) {
  if (!v) return DASH
  if (typeof v === 'string') return v.slice(0, 10)
  const d = new Date(v)
  if (Number.isNaN(d.getTime())) return DASH
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`
}

/** ISO UTC -> '2026-10-25 14:40 UTC' */
export function fmtUtc(iso) {
  if (!iso) return DASH
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return String(iso)
  return `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())} ${pad(
    d.getUTCHours()
  )}:${pad(d.getUTCMinutes())} UTC`
}

/** 北京时间 'YYYY-MM-DD HH:mm:ss' */
export function fmtClock(date) {
  const d = date instanceof Date ? date : new Date()
  const parts = new Intl.DateTimeFormat('zh-CN', {
    timeZone: 'Asia/Shanghai',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  }).formatToParts(d)
  const get = (t) => parts.find((p) => p.type === t)?.value ?? ''
  return `${get('year')}-${get('month')}-${get('day')} ${get('hour')}:${get('minute')}:${get('second')}`
}

export function fmtTimeOnly(date) {
  const d = date instanceof Date ? date : new Date()
  const parts = new Intl.DateTimeFormat('zh-CN', {
    timeZone: 'Asia/Shanghai',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  }).formatToParts(d)
  const get = (t) => parts.find((p) => p.type === t)?.value ?? ''
  return `${get('hour')}:${get('minute')}:${get('second')}`
}

export function fmtLocalDateTime(date) {
  const d = date instanceof Date ? date : new Date()
  const pad2 = (n) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())} ${pad2(
    d.getHours()
  )}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`
}

function pad(n) {
  return String(n).padStart(2, '0')
}

const SECTORS = ['北', '东北', '东', '东南', '南', '西南', '西', '西北']
const SECTORS_EN = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

/** 方位角（0=北，顺时针 0–360）-> 中文方位 + 英文缩写 */
export function bearingLabel(bearing) {
  if (bearing === null || bearing === undefined || Number.isNaN(Number(bearing))) {
    return { zh: DASH, en: DASH }
  }
  const b = ((Number(bearing) % 360) + 360) % 360
  const idx = Math.round(b / 45) % 8
  return { zh: SECTORS[idx], en: SECTORS_EN[idx] }
}

/** direction_deg 契约（0=东，逆时针）-> 屏幕/罗盘角度 */
export function directionToBearing(deg) {
  if (deg === null || deg === undefined || Number.isNaN(Number(deg))) return DASH
  const math = Number(deg)
  const bearing = (90 - math + 360) % 360
  return bearing
}

export function shiftDate(dateStr, days) {
  const d = new Date(`${dateStr}T00:00:00Z`)
  if (Number.isNaN(d.getTime())) return dateStr
  d.setUTCDate(d.getUTCDate() + days)
  return `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())}`
}

export function clamp(v, lo, hi) {
  const n = Number(v)
  if (Number.isNaN(n)) return lo
  return Math.min(hi, Math.max(lo, n))
}

/** '2025-06-26' -> '06-26' */
export function shortDate(dateStr) {
  if (!dateStr || typeof dateStr !== 'string') return DASH
  const s = dateStr.slice(0, 10)
  return s.length >= 10 ? s.slice(5) : s
}

/** 截断长文本 */
export function truncate(text, n = 60) {
  if (!text) return ''
  const s = String(text)
  return s.length > n ? `${s.slice(0, n - 1)}…` : s
}
