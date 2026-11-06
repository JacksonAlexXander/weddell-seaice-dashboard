<script setup>
import { computed } from 'vue'
import PanelShell from './PanelShell.vue'
import { fmtInt } from '../utils/format.js'

const props = defineProps({
  overview: { type: Object, default: null },
  dataTime: { type: Object, default: null },
  loading: { type: Boolean, default: false },
  error: { type: String, default: '' },
})
defineEmits(['retry'])

const dataset = computed(() => props.overview?.dataset || null)

const coveragePct = computed(() => {
  const f = dataset.value?.mean_valid_fraction
  if (f === null || f === undefined) return null
  return Number((f * 100).toFixed(1))
})

const seasonal = computed(() => {
  const monthly = props.overview?.monthly
  if (!monthly || monthly.length < 12) return null
  const mean = (ids) => {
    const rows = monthly.filter((m) => ids.includes(m.month))
    if (!rows.length) return null
    return rows.reduce((a, b) => a + b.coverage, 0) / rows.length
  }
  const summer = mean([12, 1, 2])
  const winter = mean([6, 7, 8])
  if (summer === null || winter === null) return null
  return { summer: Number((summer * 100).toFixed(1)), winter: Number((winter * 100).toFixed(1)) }
})

const rows = computed(() => {
  const d = dataset.value
  if (!d) return []
  return [
    { label: '时段', value: `${d.first_date} → ${d.latest_date}`, wide: true },
    { label: '网格', value: `${d.grid?.rows} × ${d.grid?.cols} · ${d.grid?.cell_km} km` },
    { label: '有效日数', value: `${fmtInt(d.n_frames)} 天 / 48 年` },
    { label: '缺失日', value: `${d.missing_days} 天（整帧跳过）` },
    { label: '最新批次', value: props.dataTime?.latest_date || d.latest_date },
  ]
})
</script>

<template>
  <PanelShell
    title="数据覆盖与缺失日"
    subtitle="数据质检 · EASE-Grid South"
    :loading="loading"
    :error="error"
    :empty="!loading && !error && !dataset"
    empty-text="暂无数据覆盖信息（/api/overview）"
    @retry="$emit('retry')"
  >
    <div class="coverage-head">
      <span class="cov-label">全期平均有效覆盖</span>
      <span class="cov-value mono">{{ coveragePct === null ? '—' : coveragePct + '%' }}</span>
    </div>
    <div class="coverage-bar">
      <i :style="{ width: Math.min(100, coveragePct || 0) + '%' }"></i>
    </div>

    <div class="dl-grid">
      <div v-for="row in rows" :key="row.label" class="dl-item" :class="{ wide: row.wide }">
        <span class="dl-label">{{ row.label }}</span>
        <span class="dl-value" :title="row.value">{{ row.value }}</span>
      </div>
    </div>

    <div v-if="seasonal" class="dl-note">
      区域 {{ dataset?.region || '60°W–20°E, 60°S–83°S' }}；季节差异显著：冬季（JJA）覆盖约
      <b>{{ seasonal.winter }}%</b>，夏季（DJF）约 <b>{{ seasonal.summer }}%</b>。
    </div>
  </PanelShell>
</template>

<style scoped>
.dl-item.wide {
  grid-column: 1 / -1;
}

.coverage-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  margin-bottom: -2px;
}

.cov-label {
  font-size: 10px;
  color: var(--text-3);
}

.cov-value {
  font-size: 15px;
  font-weight: 700;
  color: #a5f3fc;
  text-shadow: 0 0 12px rgba(34, 211, 238, 0.4);
}
</style>
