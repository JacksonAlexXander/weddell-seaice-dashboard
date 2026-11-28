<script setup>
import { computed } from 'vue'
import { pagesMode } from '../api/client.js'
import PanelShell from './PanelShell.vue'
import { fmtInt, fmtTimeOnly, fmtUtc } from '../utils/format.js'

const props = defineProps({
  recent: { type: Boolean, default: false },
  batchExpired: { type: Boolean, default: false },
  validUntil: { type: Number, default: null },
  horizonExpired: { type: Boolean, default: false },
  leadDays: { type: Number, default: 1 },
  warnings: { type: Array, default: () => [] },
  dataTime: { type: Object, default: null },
  prediction: { type: Object, default: null },
  backtestAt: { type: Date, default: null },
  predictAt: { type: Date, default: null },
  selection: { type: Object, required: true },
})

const LEVEL_LABEL = { info: '提示', warning: '警告', danger: '危险' }

const CONTEXT_CODES = new Set(['recent_product_shift', 'multi_day_experimental', 'forecast_already_ended'])
const showWarnings = computed(() => {
  if (!props.recent) return props.warnings
  const warnings = props.warnings.filter(item => !CONTEXT_CODES.has(item.code))
  if (!props.horizonExpired) return warnings
  const resolution = props.batchExpired
    ? pagesMode ? '正在等待更新的发布批次。' : '请更新近期观测。'
    : warnings.some(item => item.code === 'path_outside_support')
      ? '可选择覆盖范围内的其他像元。'
      : '可选择更长的预报时效。'
  return [{ code: 'forecast_expired', level: 'warning',
    message: `所选轨迹已于 ${fmtUtc(props.validUntil)} 结束。${resolution}` }, ...warnings]
})
const horizonText = computed(() => {
  if (props.validUntil === null) return '—'
  return props.horizonExpired ? '已结束 · 存档' : '有效'
})

const dtRows = computed(() => {
  const p = props.prediction?.data_time
  const d = props.dataTime
  if (props.recent) return [
    { label: '近期数据源', value: d?.source || 'OSI SAF' },
    { label: '观测结束', value: fmtUtc(p?.observation_end_at || d?.latest_at), cy: true },
    { label: '所选时效有效至', value: fmtUtc(props.validUntil), cy: true },
    { label: '预报状态', value: horizonText.value },
    { label: '观测口径', value: d?.available ? `${d.resolution_km ?? 62.5} km · ${d.observation_span_hours ?? 48}h 平均` : '—' },
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
  <PanelShell title="异常提示与数据时间">
    <template #extra>
      <span class="tag" :class="showWarnings.length ? 'amber' : 'green'">
        {{ showWarnings.length ? showWarnings.length + ' 条异常' : recent && validUntil === null ? '待载入' : '无异常' }}
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
      <div v-if="!showWarnings.length && (!recent || validUntil !== null)" class="warn-empty">
        <span class="dot ok"></span>
        当前无异常
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
      <p v-if="recent && dataTime?.available" class="data-method">
        OSI SAF 观测已映射到 25 km 训练网格。近期预报精度需用同期 OSI SAF 观测检验。
        <template v-if="leadDays > 1">多日预报为试验性结果，误差可能随时效增加。</template>
      </p>
      <div class="dt-updated">
        <span v-if="recent && pagesMode">网站发布 {{ fmtUtc(dataTime?.published_at) }}</span>
        <span v-else>预测更新 {{ predictAt ? fmtTimeOnly(predictAt) : '—' }}</span>
        <span v-if="!pagesMode">回测更新 {{ backtestAt ? fmtTimeOnly(backtestAt) : '—' }}</span>
      </div>
    </div>
  </PanelShell>
</template>

<style scoped>
.data-method {
  margin: 5px 0 3px;
  font-size: 10px;
  line-height: 1.4;
  color: var(--text-3);
}

.dt-updated {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  font-size: 9px;
  color: var(--text-3);
  padding-top: 2px;
}
</style>
