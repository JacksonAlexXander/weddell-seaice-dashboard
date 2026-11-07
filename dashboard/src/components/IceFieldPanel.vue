<script setup>
import { computed, ref } from 'vue'
import PanelShell from './PanelShell.vue'
import { useEChart } from '../composables/useEChart.js'
import { C, baseTooltip, catAxis, esc, tipRow } from '../charts/theme.js'
import { fmtInt } from '../utils/format.js'

const props = defineProps({
  mode: { type: String, default: 'observed' },
  observedField: { type: Object, default: null },
  predictedField: { type: Object, default: null },
  selectedCell: { type: Object, required: true },
  selectedLatLon: { type: Object, default: null },
  path: { type: Object, default: null },
  selectedModel: { type: Object, default: null },
  predictionInput: { type: Object, default: null },
  loading: { type: Boolean, default: false },
  error: { type: String, default: '' },
})

const emit = defineEmits(['mode-change', 'select-cell', 'retry'])

const el = ref(null)

const field = computed(() => (props.mode === 'predicted' ? props.predictedField : props.observedField))

const HEAT_STOPS = [
  [0.0, '#0b2536'],
  [0.16, '#155e75'],
  [0.34, '#0e7490'],
  [0.5, '#0891b2'],
  [0.66, '#22d3ee'],
  [0.82, '#a3e635'],
  [1.0, '#fbbf24'],
]

function hexRgb(hex) {
  const n = parseInt(hex.slice(1), 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}

/** 速度 → 热力色（0–vMax 线性映射到冷色→暖色） */
function heatColor(sp, max) {
  const t = Math.min(1, Math.max(0, sp / Math.max(max, 1)))
  for (let i = 1; i < HEAT_STOPS.length; i += 1) {
    const [t1, c1] = HEAT_STOPS[i]
    if (t <= t1) {
      const [t0, c0] = HEAT_STOPS[i - 1]
      const f = (t - t0) / Math.max(t1 - t0, 1e-6)
      const a = hexRgb(c0)
      const b = hexRgb(c1)
      return `rgb(${Math.round(a[0] + (b[0] - a[0]) * f)},${Math.round(
        a[1] + (b[1] - a[1]) * f
      )},${Math.round(a[2] + (b[2] - a[2]) * f)})`
    }
  }
  return HEAT_STOPS[HEAT_STOPS.length - 1][1]
}

function speedColor(sp, max) {
  const t = Math.min(1, Math.max(0, sp / Math.max(max, 1)))
  if (t < 0.22) return 'rgba(147,197,253,0.82)'
  if (t < 0.45) return '#67e8f9'
  if (t < 0.7) return '#a3e635'
  if (t < 0.88) return '#fbbf24'
  return '#fb7185'
}

function lonLabel(lon, digits = 0) {
  if (lon === null || lon === undefined) return ''
  return lon < 0 ? `${Math.abs(lon).toFixed(digits)}°W` : `${lon.toFixed(digits)}°E`
}

function latLabel(lat, digits = 1) {
  if (lat === null || lat === undefined) return ''
  return lat < 0 ? `${Math.abs(lat).toFixed(digits)}°S` : `${lat.toFixed(digits)}°N`
}

const option = computed(() => {
  const f = field.value
  if (!f || !Array.isArray(f.speed) || !f.speed.length) return {}

  const rows = f.grid?.rows || f.speed.length
  const cols = f.grid?.cols || f.speed[0].length
  const rowLat = f.row_lat || []
  const colLon = f.col_lon || []
  const heat = []
  let dataMax = 0
  for (let r = 0; r < rows; r += 1) {
    const rowVals = f.speed[r]
    if (!rowVals) continue
    for (let c = 0; c < cols; c += 1) {
      const s = rowVals[c]
      if (s === null || s === undefined || Number.isNaN(Number(s))) continue
      heat.push([c, r, Number(s)])
      if (s > dataMax) dataMax = Number(s)
    }
  }
  if (!heat.length) return {}

  const statsMax = Number(f.stats?.max_speed) || dataMax || 25
  const vMax = Math.ceil(statsMax / 5) * 5
  const heatData = heat.map(([c, r, sp]) => ({
    value: [c, r, sp],
    itemStyle: { color: heatColor(sp, vMax) },
  }))

  const vec = f.vectors || {}
  const vRows = vec.rows || []
  const arrowData = vRows.map((r, i) => {
    const u = Number(vec.u?.[i] ?? 0)
    const v = Number(vec.v?.[i] ?? 0)
    const sp = Number(vec.speed?.[i] ?? Math.hypot(u, v))
    const bearing = ((Math.atan2(u, v) * 180) / Math.PI + 360) % 360
    const size = 5 + 7.5 * Math.min(1, sp / Math.max(vMax, 1))
    return {
      value: [Number(vec.cols?.[i] ?? 0), r],
      uv: { u, v, speed: sp },
      symbolRotate: bearing,
      symbolSize: Number(size.toFixed(1)),
      itemStyle: { color: speedColor(sp, vMax), opacity: 0.95 },
    }
  })

  const sel = [Number(props.selectedCell.col), Number(props.selectedCell.row)]

  const series = [
    {
      name: '海冰漂移速度',
      type: 'heatmap',
      data: heatData,
      z: 1,
      animation: false,
      emphasis: { disabled: true },
      progressive: 4000,
    },
  ]

  // 24h 预测路径（日尺度积分位移通常 < 1 个像元，按格网索引取整绘制）
  if (props.path && Array.isArray(props.path.col) && props.path.col.length) {
    const pts = props.path.col.map((c, i) => [
      Math.round(Number(c)),
      Math.round(Number(props.path.row?.[i] ?? 0)),
    ])
    const last = pts[pts.length - 1]
    const disp = props.path.cumulative_km?.[props.path.cumulative_km.length - 1]
    const modelColor = props.selectedModel?.color || C.amber
    series.push({
      name: '预测路径',
      type: 'line',
      data: pts,
      z: 6,
      animation: false,
      symbol: 'circle',
      symbolSize: 2,
      lineStyle: { color: modelColor, width: 1.8, type: 'dashed', opacity: 0.95 },
      itemStyle: { color: modelColor },
      silent: true,
    })
    series.push({
      name: '路径终点',
      type: 'scatter',
      data: [
        {
          value: last,
          symbolSize: 7,
          itemStyle: { color: modelColor, borderColor: '#04070f', borderWidth: 1 },
          label: {
            show: true,
            position: 'right',
            formatter: `+${props.path.hours.at(-1)}h ${disp === undefined ? '' : '+' + Number(disp).toFixed(1) + ' km'}`,
            color: modelColor,
            fontSize: 9,
            fontWeight: 600,
            backgroundColor: 'rgba(4,7,15,0.72)',
            padding: [2, 4],
            borderRadius: 2,
          },
        },
      ],
      z: 7,
      animation: false,
      silent: true,
    })
  }

  series.push({
    name: '选中像元',
    type: 'scatter',
    data: [
      {
        value: sel,
        symbolSize: 15,
        itemStyle: { color: 'transparent', borderColor: '#f59e0b', borderWidth: 1.6 },
        label: {
          show: true,
          position: 'top',
          distance: 5,
          formatter: `(${sel[1]}, ${sel[0]})`,
          color: '#fcd34d',
          fontSize: 9,
          fontWeight: 600,
          backgroundColor: 'rgba(4,7,15,0.78)',
          padding: [1, 4],
          borderRadius: 2,
        },
      },
    ],
    z: 8,
    animation: false,
    silent: true,
  })
  series.push({
    name: '选中像元点',
    type: 'scatter',
    data: [{ value: sel, symbolSize: 5, itemStyle: { color: '#f59e0b' } }],
    z: 9,
    animation: false,
    silent: true,
  })

  return {
    animation: false,
    grid: { left: 2, right: 6, top: 5, bottom: 22, containLabel: true },
    tooltip: baseTooltip({
      trigger: 'item',
      formatter: (p) => {
        if (p.seriesType === 'heatmap') {
          const [c, r, s] = p.value
          return `<div style="font-weight:600;color:#a5f3fc;margin-bottom:2px;">漂移速度 ${esc(
            Number(s).toFixed(2)
          )} cm/s</div>${tipRow('行 / 列', `${r} / ${c}`)}${tipRow(
            '纬度',
            latLabel(rowLat[r])
          )}${tipRow('经度', lonLabel(colLon[c]))}${tipRow('日期', f.date || '—')}`
        }
        if (p.data?.uv) {
          const { u, v, speed } = p.data.uv
          const bearing = ((Math.atan2(u, v) * 180) / Math.PI + 360) % 360
          return `<div style="font-weight:600;color:#a5f3fc;margin-bottom:2px;">抽样矢量</div>${tipRow(
            'u / v',
            `${Number(u).toFixed(2)} / ${Number(v).toFixed(2)} cm/s`
          )}${tipRow('速度', `${Number(speed).toFixed(2)} cm/s`)}${tipRow(
            '网格方位角',
            `${bearing.toFixed(0)}°`
          )}`
        }
        return ''
      },
    }),
    xAxis: catAxis(
      Array.from({ length: cols }, (_, i) => i),
      {
        splitArea: { show: false },
        axisLabel: {
          color: C.muted,
          fontSize: 8.5,
          interval: Math.max(1, Math.floor(cols / 7)),
          formatter: (val, idx) => lonLabel(colLon[idx]),
        },
      }
    ),
    yAxis: catAxis(
      Array.from({ length: rows }, (_, i) => i),
      {
        splitArea: { show: false },
        axisLabel: {
          color: C.muted,
          fontSize: 8.5,
          interval: Math.max(1, Math.floor(rows / 7)),
          formatter: (val, idx) => latLabel(rowLat[idx]),
        },
      }
    ),
    series: [...series, arrowSeries(arrowData)],
  }
})

function arrowSeries(arrowData) {
  return {
    name: '抽样矢量',
    type: 'scatter',
    data: arrowData,
    symbol: 'arrow',
    z: 5,
    animation: false,
    emphasis: { disabled: true },
    tooltip: { show: true },
  }
}

function onZrClick(e) {
  const chart = chartRef.value
  if (!chart) return
  const pt = [e.offsetX, e.offsetY]
  if (!chart.containPixel({ gridIndex: 0 }, pt)) return
  const conv = chart.convertFromPixel({ xAxisIndex: 0, yAxisIndex: 0 }, pt)
  if (!Array.isArray(conv) || conv.length < 2) return
  const col = Math.round(Number(conv[0]))
  const row = Math.round(Number(conv[1]))
  const f = field.value
  if (!f) return
  const nRows = f.grid?.rows || 116
  const nCols = f.grid?.cols || 158
  if (row < 0 || row >= nRows || col < 0 || col >= nCols) return
  // 只允许选中海冰有效像元（热力图上无数据的陆地/缺测格点）
  const speed = f.speed?.[row]?.[col]
  if (speed === null || speed === undefined || Number.isNaN(Number(speed))) return
  emit('select-cell', { row, col })
}

const { chart: chartRef } = useEChart(el, option, { events: { 'zr:click': onZrClick } })

const subtitle = computed(() => {
  const f = field.value
  if (!f) return props.mode === 'predicted' ? '预测场未加载' : '观测场未加载'
  if (props.mode === 'predicted') {
    const m = f.model?.name || props.selectedModel?.name || ''
    const cmp = f.stats?.comparison
    return `${f.date} → ${f.target_date || '—'} · ${m}${
      cmp?.vec_rmse ? ` · 对比 vec RMSE ${cmp.vec_rmse} cm/s` : ''
    } · 平均 ${f.stats?.mean_speed ?? '—'} cm/s`
  }
  return `${f.date} · 有效像元 ${fmtInt(f.stats?.valid_cells)} · 覆盖 ${(
    (f.stats?.coverage || 0) * 100
  ).toFixed(1)}% · 平均 ${f.stats?.mean_speed ?? '—'} cm/s`
})

const title = computed(() =>
  props.mode === 'predicted' ? '海冰漂移场 · 预测' : '海冰漂移场 · 观测'
)

const legendMax = computed(() => {
  const st = field.value?.stats?.max_speed
  return Math.ceil((Number(st) || 25) / 5) * 5
})

const selectedText = computed(() => {
  const { row, col } = props.selectedCell
  const inp = props.predictionInput
  const ll =
    inp && Number(inp.row) === Number(row) && Number(inp.col) === Number(col)
      ? { lat: inp.lat, lon: inp.lon }
      : props.selectedLatLon
  return ll
    ? `选中 (${row}, ${col}) · ${latLabel(ll.lat)} ${lonLabel(ll.lon, 1)}`
    : `选中 (${row}, ${col})`
})

const hasField = computed(() => Boolean(field.value && field.value.speed))
</script>

<template>
  <PanelShell
    :title="title"
    :subtitle="subtitle"
    :loading="loading"
    :error="error"
    :empty="!loading && !error && !hasField"
    empty-text="暂无场数据：请选择有效观测日，或切换模型后重试（/api/field、/api/field/predicted）"
    flush
    @retry="$emit('retry')"
  >
    <template #extra>
      <span class="mode-tag tag" :class="mode === 'predicted' ? 'amber' : ''">
        {{ mode === 'predicted' ? '模型预测场' : '真实观测场' }}
      </span>
      <div class="seg">
        <button class="mini-btn" :class="{ active: mode === 'observed' }" @click="$emit('mode-change', 'observed')">
          观测
        </button>
        <button class="mini-btn" :class="{ active: mode === 'predicted' }" @click="$emit('mode-change', 'predicted')">
          预测
        </button>
      </div>
    </template>

    <div ref="el" class="chart-fill map-chart" :class="{ hidden: !hasField }"></div>

    <div v-if="hasField" class="map-overlay map-overlay-left">
      <span class="map-chip">{{ selectedText }}</span>
    </div>
    <div v-if="hasField" class="map-overlay map-overlay-right">
      <span class="map-hint">抽样矢量 stride = {{ field?.vectors?.stride || 5 }} · 点击地图选择像元</span>
    </div>
    <div v-if="hasField" class="map-legend">
      <span class="map-legend-label">0</span>
      <span class="map-legend-bar"></span>
      <span class="map-legend-label">{{ legendMax }} cm/s</span>
    </div>
  </PanelShell>
</template>

<style scoped>
.map-chart {
  cursor: crosshair;
}

.hidden {
  visibility: hidden;
}

.seg {
  display: flex;
  gap: 3px;
}

.map-overlay {
  position: absolute;
  top: 6px;
  z-index: 3;
  pointer-events: none;
}

.map-overlay-left {
  left: 10px;
}

.map-overlay-right {
  right: 10px;
}

.map-chip {
  font-size: 9.5px;
  color: #fcd34d;
  padding: 2px 6px;
  border-radius: 3px;
  background: rgba(4, 7, 15, 0.72);
  border: 1px solid rgba(245, 158, 11, 0.28);
  font-family: var(--font-mono);
}

.map-hint {
  font-size: 9px;
  color: var(--text-3);
  padding: 2px 6px;
  border-radius: 3px;
  background: rgba(4, 7, 15, 0.6);
  border: 1px solid rgba(56, 189, 248, 0.12);
}

.map-legend {
  position: absolute;
  bottom: 4px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 3;
  display: flex;
  align-items: center;
  gap: 5px;
  pointer-events: none;
}

.map-legend-label {
  font-size: 8.5px;
  color: var(--text-3);
  font-family: var(--font-mono);
}

.map-legend-bar {
  width: 110px;
  height: 5px;
  border-radius: 3px;
  background: linear-gradient(
    90deg,
    #0b2536,
    #155e75,
    #0e7490,
    #0891b2,
    #22d3ee,
    #a3e635,
    #fbbf24
  );
  box-shadow: 0 0 8px rgba(34, 211, 238, 0.35);
}

.mode-tag {
  display: none;
}
</style>
