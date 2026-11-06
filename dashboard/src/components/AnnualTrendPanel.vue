<script setup>
import { computed, ref } from 'vue'
import PanelShell from './PanelShell.vue'
import { useEChart } from '../composables/useEChart.js'
import { C, areaGradient, baseTooltip, catAxis, hexToRgba, legendStyle, tipRow, valAxis } from '../charts/theme.js'

const props = defineProps({
  annual: { type: Array, default: null },
  loading: { type: Boolean, default: false },
  error: { type: String, default: '' },
})
defineEmits(['retry'])

const el = ref(null)

const option = computed(() => {
  const annual = props.annual || []
  if (!annual.length) return {}
  const years = annual.map((a) => String(a.year))
  const speed = annual.map((a) => a.mean_speed)
  const coverage = annual.map((a) => a.coverage)
  return {
    animationDuration: 700,
    grid: { left: 2, right: 4, top: 22, bottom: 2, containLabel: true },
    legend: {
      ...legendStyle({ top: 0, right: 0, itemGap: 8 }),
      data: ['平均漂移速度', '有效覆盖率'],
    },
    tooltip: baseTooltip({
      formatter: (params) => {
        const idx = params[0]?.dataIndex ?? 0
        const a = annual[idx]
        return `<div style="font-weight:600;color:#a5f3fc;margin-bottom:3px;">${a.year} 年</div>${tipRow(
          '平均漂移速度',
          `${a.mean_speed?.toFixed?.(2) ?? a.mean_speed} cm/s`
        )}${tipRow('有效覆盖率', `${(a.coverage * 100).toFixed(1)}%`)}${tipRow(
          '有效观测日',
          `${a.n_days} 天`
        )}`
      },
    }),
    xAxis: catAxis(years, {
      axisLabel: { color: C.muted, fontSize: 9, fontFamily: undefined, interval: 7 },
    }),
    yAxis: [
      valAxis({
        name: 'cm/s',
        scale: true,
        axisLabel: { color: C.muted, fontSize: 9 },
      }),
      valAxis({
        name: '覆盖',
        scale: true,
        splitLine: { show: false },
        axisLabel: {
          color: C.muted,
          fontSize: 9,
          formatter: (v) => `${(v * 100).toFixed(0)}%`,
        },
      }),
    ],
    series: [
      {
        name: '平均漂移速度',
        type: 'line',
        data: speed,
        smooth: 0.35,
        symbol: 'none',
        lineStyle: { width: 2, color: C.cyan, shadowColor: hexToRgba(C.cyan, 0.6), shadowBlur: 8 },
        areaStyle: { color: areaGradient(C.cyan, 0.32, 0.02) },
        emphasis: { focus: 'series' },
      },
      {
        name: '有效覆盖率',
        type: 'line',
        yAxisIndex: 1,
        data: coverage,
        smooth: 0.35,
        symbol: 'none',
        lineStyle: { width: 1.2, color: C.purple, type: 'dashed' },
        emphasis: { focus: 'series' },
      },
    ],
  }
})

useEChart(el, option)
</script>

<template>
  <PanelShell
    title="48 年年度平均漂移速度趋势"
    subtitle="area：年均速度 · 虚线（右轴）：年均覆盖率"
    :loading="loading"
    :error="error"
    :empty="!loading && !error && !(annual && annual.length)"
    empty-text="暂无年度趋势数据（/api/overview）"
    flush
    @retry="$emit('retry')"
  >
    <div ref="el" class="chart-fill" :class="{ hidden: !(annual && annual.length) }"></div>
  </PanelShell>
</template>

<style scoped>
.hidden {
  visibility: hidden;
}
</style>
