<script setup>
import { computed } from 'vue'
import PanelShell from './PanelShell.vue'

const props = defineProps({
  overview: { type: Object, default: null },
  loading: { type: Boolean, default: false },
  error: { type: String, default: '' },
})
defineEmits(['retry'])

const ICONS = {
  years:
    'M4 6.5h12M6.5 4v2M13.5 4v2M4 8.5v5A1.5 1.5 0 0 0 5.5 15h9a1.5 1.5 0 0 0 1.5-1.5v-5M6.5 11.5h2M10 11.5h1.5',
  days: 'M10 4.5a5.5 5.5 0 1 0 0 11 5.5 5.5 0 0 0 0-11zM10 7.5V10l2 1.2',
  mean_speed: 'M3 13.5c2.2-5 3.8-7 7-7s4.8 2 7 7M7 10.5l2 1.5M13 10.5l-2 1.5M3 6l2-1.5M17 6l-2-1.5',
  coverage: 'M4 4h12v12H4zM4 8.5h12M8.5 4v12M11.5 11.5h2.5v2.5h-2.5z',
}

const kpis = computed(() => {
  if (!props.overview?.kpis) return []
  return props.overview.kpis.map((k) => ({
    ...k,
    icon: ICONS[k.id] || ICONS.days,
    display: formatKpi(k),
  }))
})

function formatKpi(kpi) {
  if (kpi.value === null || kpi.value === undefined) return '—'
  const n = Number(kpi.value)
  if (!Number.isFinite(n)) return '—'
  if (kpi.id === 'days') return Math.round(n).toLocaleString('zh-CN')
  if (kpi.id === 'years') return String(Math.round(n))
  return n.toLocaleString('zh-CN', { minimumFractionDigits: 1, maximumFractionDigits: 2 })
}
</script>

<template>
  <PanelShell
    title="数据总览 KPI"
    subtitle="NSIDC-0116 v4 · 1978–2025"
    :loading="loading"
    :error="error"
    :empty="!loading && !error && !kpis.length"
    empty-text="暂无 KPI 数据（/api/overview）"
    @retry="$emit('retry')"
  >
    <div class="kpi-grid">
      <div v-for="kpi in kpis" :key="kpi.id" class="kpi-card">
        <div class="kpi-icon">
          <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.15" stroke-linecap="round" stroke-linejoin="round">
            <path :d="kpi.icon" />
          </svg>
        </div>
        <div class="kpi-body">
          <div class="kpi-value-row">
            <span class="kpi-value">{{ kpi.display }}</span>
            <span class="kpi-unit">{{ kpi.unit }}</span>
          </div>
          <span class="kpi-label">{{ kpi.label }}</span>
        </div>
      </div>
    </div>
  </PanelShell>
</template>
