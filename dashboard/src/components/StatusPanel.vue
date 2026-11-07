<script setup>
import { computed } from 'vue'
import PanelShell from './PanelShell.vue'
import { fmtInt, fmtTimeOnly, fmtUtc } from '../utils/format.js'

const props = defineProps({
  recent: { type: Boolean, default: false },
  warnings: { type: Array, default: () => [] },
  dataTime: { type: Object, default: null },
  prediction: { type: Object, default: null },
  backtestAt: { type: Date, default: null },
  predictAt: { type: Date, default: null },
  selection: { type: Object, required: true },
})

const LEVEL_LABEL = { info: '提示', warning: '警告', danger: '危险' }

const showWarnings = computed(() => (props.warnings || []).slice(0, 3))

const dtRows = computed(() => {
  const p = props.prediction?.data_time
  const d = props.dataTime
  if (props.recent) return [
    { label: '近期数据源', value: d?.source || 'OSI SAF' },
    { label: '观测结束', value: fmtUtc(p?.observation_end_at || d?.latest_at), cy: true },
    { label: '预报有效至', value: fmtUtc(p?.forecast_valid_until), cy: true },
    { label: '轨迹覆盖未来', value: p ? (p.extends_past_now ? '是' : '时效已结束') : '—' },
    { label: '观测口径', value: '62.5 km · 48h 平均' },
    { label: '连续观测帧', value: d?.n_frames || '—' },
  ]
  const rows = [
    { label: '起报日', value: p?.last_obs_date || props.selection.date, cy: true },
    { label: '24h 目标日', value: p?.next_frame_date || '—', cy: true },
    { label: '目标日观测', value: !p ? '—' : p.target_observed === false ? '本批次无观测' : '本批次已收录' },
    { label: '最新数据', value: d?.latest_date || '—' },
    { label: '数据批次', value: d?.built_at ? fmtUtc(d.built_at).slice(0, 10) : '—' },
    { label: '总帧数', value: d?.n_frames ? fmtInt(d.n_frames) : '—' },
    {
      label: '批次滞后',
      value: d?.batch_age_days === undefined ? '—' : `${d.batch_age_days} 天前`,
    },
  ]
  return rows
})
</script>

<template>
  <PanelShell title="异常提示与数据时间" subtitle="warnings · data_time">
    <template #extra>
      <span class="tag" :class="showWarnings.length ? 'amber' : 'green'">
        {{ showWarnings.length ? showWarnings.length + ' 条提示' : '无异常' }}
      </span>
    </template>

    <div class="warn-list">
      <div
        v-for="(w, i) in showWarnings"
        :key="`${w.code}-${i}`"
        class="warn-item"
        :class="w.level || 'info'"
      >
        <span class="warn-code">{{ LEVEL_LABEL[w.level] || w.level || 'INFO' }}</span>
        <span class="warn-msg">{{ w.message }}</span>
      </div>
      <div v-if="!showWarnings.length" class="warn-empty">
        <span class="dot ok"></span>
        无异常提示 · 数据与推理链路正常
      </div>
    </div>

    <div class="dt-section">
      <span class="dt-caption">
        <a v-if="recent" href="https://doi.org/10.15770/EUM_SAF_OSI_NRT_2007" target="_blank"
          rel="noopener noreferrer">EUMETSAT OSI SAF · UTC</a>
        <template v-else>归档批次 {{ fmtUtc(dataTime?.built_at) }}</template>
      </span>
      <div class="dt-grid">
        <div v-for="row in dtRows" :key="row.label" class="dt-item">
          <span class="dt-label">{{ row.label }}</span>
          <span class="dt-value" :class="{ cy: row.cy }" :title="row.value">{{ row.value }}</span>
        </div>
      </div>
      <div class="dt-updated">
        <span>预测更新 {{ predictAt ? fmtTimeOnly(predictAt) : '—' }}</span>
        <span>回测更新 {{ backtestAt ? fmtTimeOnly(backtestAt) : '—' }}</span>
      </div>
    </div>
  </PanelShell>
</template>

<style scoped>
.dt-updated {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  font-size: 9px;
  color: var(--text-3);
  padding-top: 2px;
}
</style>
