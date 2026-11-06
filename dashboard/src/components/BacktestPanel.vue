<script setup>
import { computed, ref } from 'vue'
import PanelShell from './PanelShell.vue'
import { useEChart } from '../composables/useEChart.js'
import { C, baseTooltip, catAxis, hexToRgba, legendStyle, tipRow, valAxis } from '../charts/theme.js'
import { fmt, shortDate, signedPct } from '../utils/format.js'

const props = defineProps({
  backtest: { type: Object, default: null },
  loading: { type: Boolean, default: false },
  error: { type: String, default: '' },
  range: { type: Number, default: 180 },
  model: { type: Object, default: null },
  cell: { type: Object, required: true },
})
const emit = defineEmits(['range-change', 'refresh', 'retry'])

const el = ref(null)

const rangeOptions = [90, 180, 365]

const series = computed(() => props.backtest?.series || [])

const option = computed(() => {
  const rows = series.value
  if (!rows.length) return {}
  const dates = rows.map((r) => shortDate(r.date))
  return {
    animationDuration: 400,
    grid: { left: 2, right: 4, top: 20, bottom: 2, containLabel: true },
    legend: {
      ...legendStyle({ top: 0, left: 0, itemGap: 10 }),
      data: ['真实观测速度', '模型预测速度'],
    },
    tooltip: baseTooltip({
      formatter: (params) => {
        const idx = params[0]?.dataIndex ?? 0
        const r = rows[idx]
        return `<div style="font-weight:600;color:#a5f3fc;margin-bottom:3px;">${r.date}</div>${tipRow(
          '真实速度',
          `${fmt(r.obs_speed)} cm/s`,
          '#e2e8f0'
        )}${tipRow('预测速度', `${fmt(r.pred_speed)} cm/s`, '#fcd34d')}${tipRow(
          '误差 |Δ|',
          `${fmt(r.err_vec)} cm/s`
        )}${tipRow('真实 u / v', `${fmt(r.obs_u)} / ${fmt(r.obs_v)}`)}${tipRow(
          '预测 u / v',
          `${fmt(r.pred_u)} / ${fmt(r.pred_v)}`
        )}`
      },
    }),
    xAxis: catAxis(dates, {
      boundaryGap: false,
      axisLabel: {
        color: C.muted,
        fontSize: 8.5,
        interval: Math.max(0, Math.ceil(dates.length / 6) - 1),
      },
    }),
    yAxis: valAxis({
      name: 'cm/s',
      nameGap: 6,
      scale: true,
      axisLabel: { color: C.muted, fontSize: 8.5 },
    }),
    series: [
      {
        name: '真实观测速度',
        type: 'line',
        data: rows.map((r) => r.obs_speed),
        smooth: 0.2,
        showSymbol: false,
        lineStyle: { width: 1.8, color: '#e2e8f0', shadowColor: 'rgba(226,232,240,.35)', shadowBlur: 6 },
        itemStyle: { color: '#e2e8f0' },
        areaStyle: { color: hexToRgba('#e2e8f0', 0.07) },
        z: 3,
      },
      {
        name: '模型预测速度',
        type: 'line',
        data: rows.map((r) => r.pred_speed),
        smooth: 0.2,
        showSymbol: false,
        lineStyle: { width: 1.6, color: C.amber, type: 'dashed' },
        itemStyle: { color: C.amber },
        z: 2,
      },
    ],
  }
})

useEChart(el, option)

const hasData = computed(() => series.value.length > 0)

const subtitle = computed(() => {
  const m = props.model?.short || props.model?.name || '—'
  return `${m} · 像元 (${props.cell.row}, ${props.cell.col}) · 近 ${props.range} 天`
})

const summary = computed(() => props.backtest?.summary || null)

const rangeText = computed(() => {
  const r = props.backtest?.range
  if (!r) return ''
  return `${r.requested_start || '—'} ~ ${r.requested_end || '—'}`
})
</script>

<template>
  <PanelShell
    title="历史回测 · 真实 vs 预测"
    :subtitle="subtitle"
    :loading="loading"
    :error="error"
    :empty="!loading && !error && !hasData"
    empty-text="运行预测后自动生成回测对比（POST /api/backtest）；也可点击右上角刷新"
    @retry="$emit('retry')"
  >
    <template #extra>
      <div class="seg">
        <button
          v-for="opt in rangeOptions"
          :key="opt"
          class="mini-btn"
          :class="{ active: range === opt }"
          @click="emit('range-change', opt)"
        >
          {{ opt }}d
        </button>
      </div>
      <button class="mini-btn" title="刷新回测" :disabled="loading" @click="emit('refresh')">↻</button>
    </template>

    <div class="bt-chart">
      <div ref="el" class="bt-chart-inner" :class="{ hidden: !hasData }"></div>
    </div>

    <div v-if="summary" class="bt-summary">
      <span>样本 <b class="mono">{{ summary.n }}</b> 天</span>
      <span class="chip" :class="summary.vec_rmse <= 11 ? 'success' : 'idle'">
        vec RMSE {{ fmt(summary.vec_rmse) }} cm/s
      </span>
      <span class="chip" :class="summary.skill_vs_persistence > 0 ? 'success' : 'error'">
        技能分 {{ signedPct(summary.skill_vs_persistence) }}
      </span>
      <span>R²_u {{ fmt(summary.r2_u, 3) }}</span>
      <span class="dim" style="margin-left: auto">{{ rangeText }}</span>
    </div>
  </PanelShell>
</template>

<style scoped>
.seg {
  display: flex;
  gap: 3px;
}

.bt-chart {
  position: relative;
  flex: 1 1 auto;
  min-height: 0;
}

.bt-chart-inner {
  position: absolute;
  inset: 0;
}

.hidden {
  visibility: hidden;
}
</style>
