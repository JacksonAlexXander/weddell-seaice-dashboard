<script setup>
import { computed, ref } from 'vue'
import PanelShell from './PanelShell.vue'
import { useEChart } from '../composables/useEChart.js'
import { C, barGradient, baseTooltip, catAxis, legendStyle, tipRow, valAxis } from '../charts/theme.js'

const props = defineProps({
  monthly: { type: Array, default: null },
  loading: { type: Boolean, default: false },
  error: { type: String, default: '' },
})
defineEmits(['retry'])

const el = ref(null)

const option = computed(() => {
  const monthly = props.monthly || []
  if (!monthly.length) return {}
  const months = monthly.map((m) => `${m.month}月`)
  const speed = monthly.map((m) => m.mean_speed)
  const coverage = monthly.map((m) => m.coverage)
  return {
    animationDuration: 700,
    grid: { left: 2, right: 4, top: 22, bottom: 2, containLabel: true },
    legend: {
      ...legendStyle({ top: 0, right: 0, itemGap: 8 }),
      data: ['月均漂移速度', '月均覆盖率'],
    },
    tooltip: baseTooltip({
      formatter: (params) => {
        const idx = params[0]?.dataIndex ?? 0
        const m = monthly[idx]
        return `<div style="font-weight:600;color:#a5f3fc;margin-bottom:3px;">${m.month} 月气候态</div>${tipRow(
          '平均漂移速度',
          `${m.mean_speed?.toFixed?.(2) ?? m.mean_speed} cm/s`
        )}${tipRow('平均有效覆盖', `${(m.coverage * 100).toFixed(1)}%`)}`
      },
    }),
    xAxis: catAxis(months, { axisLabel: { color: C.muted, fontSize: 9 } }),
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
        axisLabel: { color: C.muted, fontSize: 9, formatter: (v) => `${(v * 100).toFixed(0)}%` },
      }),
    ],
    series: [
      {
        name: '月均漂移速度',
        type: 'bar',
        data: speed,
        barWidth: '48%',
        itemStyle: {
          color: barGradient(C.sky, 0.95, 0.22),
          borderRadius: [2, 2, 0, 0],
          shadowColor: 'rgba(56,189,248,0.35)',
          shadowBlur: 6,
        },
        emphasis: { itemStyle: { color: barGradient('#7dd3fc', 1, 0.35) } },
      },
      {
        name: '月均覆盖率',
        type: 'line',
        yAxisIndex: 1,
        data: coverage,
        smooth: 0.3,
        symbol: 'circle',
        symbolSize: 4,
        lineStyle: { width: 1.6, color: C.purple },
        itemStyle: { color: C.purple, borderColor: '#0b1220', borderWidth: 1 },
      },
    ],
  }
})

useEChart(el, option)
</script>

<template>
  <PanelShell
    title="月度气候态"
    subtitle="bar：平均速度 · line：有效覆盖率"
    :loading="loading"
    :error="error"
    :empty="!loading && !error && !(monthly && monthly.length)"
    empty-text="暂无月度气候态数据（/api/overview）"
    flush
    @retry="$emit('retry')"
  >
    <div ref="el" class="chart-fill" :class="{ hidden: !(monthly && monthly.length) }"></div>
  </PanelShell>
</template>

<style scoped>
.hidden {
  visibility: hidden;
}
</style>
