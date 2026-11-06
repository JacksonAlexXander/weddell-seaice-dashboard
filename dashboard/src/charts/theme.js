// ECharts 通用主题工具：深色科技风坐标轴 / tooltip / 网格
export const C = {
  cyan: '#22d3ee',
  sky: '#38bdf8',
  blue: '#3b82f6',
  purple: '#a78bfa',
  amber: '#f59e0b',
  green: '#34d399',
  red: '#f87171',
  slate: '#64748b',
  text: '#dbe7f5',
  muted: '#7b8ba6',
  gridLine: 'rgba(148, 163, 184, 0.10)',
  axisLine: 'rgba(148, 163, 184, 0.22)',
}

export const FONT =
  "'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', system-ui, sans-serif"

export function hexToRgba(hex, alpha = 1) {
  const h = hex.replace('#', '')
  const full = h.length === 3 ? h.split('').map((c) => c + c).join('') : h
  const n = parseInt(full, 16)
  const r = (n >> 16) & 255
  const g = (n >> 8) & 255
  const b = n & 255
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}

/** 渐变填充（自上而下） */
export function areaGradient(color, topAlpha = 0.32, bottomAlpha = 0.02) {
  return {
    type: 'linear',
    x: 0,
    y: 0,
    x2: 0,
    y2: 1,
    colorStops: [
      { offset: 0, color: hexToRgba(color, topAlpha) },
      { offset: 1, color: hexToRgba(color, bottomAlpha) },
    ],
  }
}

/** 柱状渐变 */
export function barGradient(color, topAlpha = 0.95, bottomAlpha = 0.28) {
  return {
    type: 'linear',
    x: 0,
    y: 0,
    x2: 0,
    y2: 1,
    colorStops: [
      { offset: 0, color: hexToRgba(color, topAlpha) },
      { offset: 1, color: hexToRgba(color, bottomAlpha) },
    ],
  }
}

export function baseTooltip(overrides = {}) {
  return {
    trigger: 'axis',
    confine: true,
    appendToBody: false,
    backgroundColor: 'rgba(6, 14, 27, 0.96)',
    borderColor: 'rgba(56, 189, 248, 0.38)',
    borderWidth: 1,
    padding: [7, 10],
    textStyle: { color: C.text, fontSize: 11.5, fontFamily: FONT },
    axisPointer: {
      lineStyle: { color: 'rgba(56, 189, 248, 0.45)', width: 1, type: 'dashed' },
      crossStyle: { color: 'rgba(56, 189, 248, 0.45)' },
      label: {
        backgroundColor: 'rgba(14, 116, 144, 0.9)',
        color: '#e0f2fe',
        fontSize: 10,
      },
    },
    extraCssText: 'border-radius:4px;box-shadow:0 6px 22px rgba(0,0,0,.5);',
    ...overrides,
  }
}

export function catAxis(data, overrides = {}) {
  return {
    type: 'category',
    data,
    boundaryGap: true,
    axisLine: { lineStyle: { color: C.axisLine } },
    axisTick: { show: false },
    axisLabel: { color: C.muted, fontSize: 9.5, fontFamily: FONT, hideOverlap: true },
    splitLine: { show: false },
    ...overrides,
  }
}

export function valAxis(overrides = {}) {
  return {
    type: 'value',
    axisLine: { show: false },
    axisTick: { show: false },
    axisLabel: { color: C.muted, fontSize: 9.5, fontFamily: FONT },
    splitLine: { show: true, lineStyle: { color: C.gridLine, type: 'dashed' } },
    nameTextStyle: { color: C.muted, fontSize: 9, fontFamily: FONT },
    ...overrides,
  }
}

export function baseGrid(overrides = {}) {
  return { left: 6, right: 8, top: 14, bottom: 4, containLabel: true, ...overrides }
}

export function legendStyle(overrides = {}) {
  return {
    itemWidth: 9,
    itemHeight: 6,
    itemGap: 10,
    icon: 'roundRect',
    textStyle: { color: C.muted, fontSize: 9.5, fontFamily: FONT },
    ...overrides,
  }
}

/** html 转义（tooltip formatter 中使用） */
export function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => {
    const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }
    return map[c]
  })
}

/** tooltip 行样式 */
export function tipRow(label, value, color) {
  return `<div style="display:flex;justify-content:space-between;gap:14px;line-height:1.75;">
    <span style="color:#7b8ba6;">${esc(label)}</span>
    <span style="font-family:'SF Mono',Menlo,monospace;color:${color || C.text};">${esc(value)}</span>
  </div>`
}
