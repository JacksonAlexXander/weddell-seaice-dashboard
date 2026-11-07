<script setup>
import { computed, ref } from 'vue'
import PanelShell from './PanelShell.vue'
import { useEChart } from '../composables/useEChart.js'
import { C, baseTooltip, catAxis, hexToRgba, tipRow, valAxis } from '../charts/theme.js'

const props = defineProps({
  prediction: { type: Object, default: null },
  running: { type: Boolean, default: false },
})

const el = ref(null)

const series24 = computed(() => props.prediction?.series_horizon || props.prediction?.series_24h || null)

const option = computed(() => {
  const s = series24.value
  if (!s || !Array.isArray(s.hours)) return {}
  const hours = s.hours.map((h) => String(h))
  const cum = s.cumulative_km || []
  const speed = s.speed_cm_s || []
  const unc = s.uncertainty
  const lower = unc?.lower_km
  const upper = unc?.upper_km
  const hasBand = Array.isArray(lower) && Array.isArray(upper) && lower.length === cum.length

  const series = []

  if (hasBand) {
    series.push({
      name: '不确定度下界',
      type: 'line',
      data: lower,
      stack: 'uncertainty',
      symbol: 'none',
      silent: true,
      lineStyle: { opacity: 0 },
      areaStyle: { opacity: 0 },
      tooltip: { show: false },
      z: 1,
    })
    series.push({
      name: '不确定度带',
      type: 'line',
      data: upper.map((v, i) => Number((v - lower[i]).toFixed(3))),
      stack: 'uncertainty',
      symbol: 'none',
      silent: true,
      lineStyle: { opacity: 0 },
      areaStyle: { color: hexToRgba(C.cyan, 0.18) },
      tooltip: { show: false },
      z: 1,
    })
  }

  series.push({
    name: '逐时累计位移',
    type: 'line',
    data: cum,
    smooth: props.prediction?.series_horizon ? false : 0.25,
    showSymbol: true,
    symbol: 'circle',
    symbolSize: 3.2,
    lineStyle: {
      width: 2.4,
      color: C.cyan,
      shadowColor: hexToRgba(C.cyan, 0.7),
      shadowBlur: 10,
    },
    itemStyle: { color: C.cyan },
    areaStyle: {
      color: {
        type: 'linear',
        x: 0,
        y: 0,
        x2: 0,
        y2: 1,
        colorStops: [
          { offset: 0, color: hexToRgba(C.cyan, 0.24) },
          { offset: 1, color: hexToRgba(C.cyan, 0.02) },
        ],
      },
    },
    z: 3,
    markPoint: {
      symbol: 'circle',
      symbolSize: 7,
      itemStyle: { color: C.cyan, borderColor: '#04070f', borderWidth: 1.5 },
      label: {
        show: true,
        position: 'left',
        distance: 4,
        formatter: `${Number(cum[cum.length - 1] ?? 0).toFixed(1)} km`,
        color: '#a5f3fc',
        fontSize: 9.5,
        fontWeight: 600,
      },
      data: [{ coord: [hours.length - 1, cum[cum.length - 1] ?? 0] }],
    },
  })

  const speedLast = Number(speed[speed.length - 1] ?? 0)
  series.push({
    name: '漂移速度',
    type: 'line',
    step: props.prediction?.series_horizon ? 'start' : false,
    yAxisIndex: 1,
    data: speed,
    smooth: false,
    showSymbol: false,
    lineStyle: { width: 1.3, color: C.amber, type: 'dashed', opacity: 0.85 },
    itemStyle: { color: C.amber },
    z: 2,
    tooltip: { show: true },
    markPoint: {
      symbol: 'circle',
      symbolSize: 5,
      itemStyle: { color: C.amber, borderColor: '#04070f', borderWidth: 1 },
      label: {
        show: true,
        position: 'top',
        distance: 3,
        formatter: `${speedLast.toFixed(1)} cm/s 速度`,
        color: '#fcd34d',
        fontSize: 9,
        fontWeight: 600,
      },
      data: [{ coord: [hours.length - 1, speedLast] }],
    },
  })

  return {
    animationDuration: 500,
    grid: { left: 2, right: 4, top: 16, bottom: 2, containLabel: true },
    tooltip: baseTooltip({
      formatter: (params) => {
        const idx = params[0]?.dataIndex ?? 0
        const h = s.hours[idx]
        const rows = [
          `<div style="font-weight:600;color:#a5f3fc;margin-bottom:2px;">+${h} h${
            s.target_date ? ` · 目标日 ${s.target_date}` : ''
          }</div>`,
          tipRow('累计位移', `${Number(cum[idx] ?? 0).toFixed(2)} km`, '#67e8f9'),
          tipRow('漂移速度', `${Number(speed[idx] ?? 0).toFixed(2)} cm/s`, '#fcd34d'),
        ]
        if (hasBand) rows.push(tipRow('不确定度带', `${lower[idx].toFixed(1)} ~ ${upper[idx].toFixed(1)} km`))
        if (s.lat && s.lon) rows.push(tipRow('像元位置', `${Number(s.row[idx]).toFixed(2)}, ${Number(s.col[idx]).toFixed(2)}`))
        if (s.lat && s.lon) rows.push(tipRow('经纬度', `${Number(s.lat[idx]).toFixed(3)}°, ${Number(s.lon[idx]).toFixed(3)}°`))
        return rows.join('')
      },
    }),
    xAxis: catAxis(hours, {
      boundaryGap: false,
      name: '小时 (h)',
      nameGap: 5,
      nameLocation: 'end',
      nameTextStyle: { color: C.muted, fontSize: 9 },
      axisLabel: { color: C.muted, fontSize: 9, interval: hours.length > 49 ? 11 : 4 },
    }),
    yAxis: [
      valAxis({
        name: 'km',
        nameGap: 6,
        scale: true,
        axisLabel: { color: C.muted, fontSize: 9 },
      }),
      valAxis({
        scale: true,
        splitLine: { show: false },
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: { show: false },
        name: '',
      }),
    ],
    series,
  }
})

useEChart(el, option)

const subtitle = computed(() => {
  const p = props.prediction
  if (!p) return '日尺度 → 逐时线性积分'
  const m = p.model?.name || ''
  const dt = p.data_time || {}
  return `${m} · ${dt.last_obs_date || '—'} → ${series24.value?.target_date || dt.next_frame_date || '—'}`
})

function downloadForecast() {
  const forecast = props.prediction
  const url = URL.createObjectURL(new Blob([JSON.stringify(forecast, null, 2)], { type: 'application/json' }))
  const link = document.createElement('a')
  link.href = url
  link.download = `weddell-${forecast.model.id}-${forecast.data_time.last_obs_date}.json`
  link.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

const hasSeries = computed(() => Boolean(series24.value && series24.value.hours?.length))
</script>

<template>
  <PanelShell
    :title="prediction?.series_horizon ? `近期漂移预报 · +${series24.hours.at(-1)} 小时` : '24 小时预测曲线'"
    :subtitle="subtitle"
    :loading="running"
    :empty="!running && !hasSeries"
    empty-text="尚未运行预测：点击「一键运行预测」生成逐时累计位移与不确定度带（POST /api/predict）"
  >
    <template #extra>
      <span v-if="prediction?.metrics" class="tag">该模型 vec RMSE {{ prediction.metrics.vec_rmse }}</span>
      <button v-if="prediction?.series_horizon" class="mini-btn" @click="downloadForecast">下载预报 JSON</button>
      <span class="tag amber">逐时积分</span>
    </template>

    <div class="forecast-body" :class="{ 'is-empty': !hasSeries }">
      <div ref="el" class="forecast-chart"></div>
      <div v-if="prediction?.daily_forecasts" class="forecast-note">
        <span v-for="day in prediction.daily_forecasts" :key="day.lead_hours">
          +{{ day.lead_hours }}h · u {{ day.u.toFixed(2) }} / v {{ day.v.toFixed(2) }} cm/s
        </span>
      </div>
      <div v-if="hasSeries" class="forecast-note">
        <span class="dot cyan"></span>
        {{ series24.interpolation }}
        <template v-if="series24?.uncertainty">不确定度：{{ series24.uncertainty.method }}</template>
      </div>
    </div>
  </PanelShell>
</template>

<style scoped>
.forecast-body {
  display: flex;
  flex-direction: column;
  min-height: 0;
  flex: 1 1 auto;
}

.forecast-body.is-empty .forecast-chart {
  visibility: hidden;
}

.forecast-chart {
  flex: 1 1 auto;
  min-height: 0;
}

.forecast-note {
  flex: 0 0 auto;
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 9.5px;
  color: var(--text-3);
  padding-top: 3px;
  border-top: 1px dashed rgba(56, 189, 248, 0.1);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.dot.cyan {
  background: var(--cyan);
  box-shadow: 0 0 6px rgba(34, 211, 238, 0.8);
}
</style>
