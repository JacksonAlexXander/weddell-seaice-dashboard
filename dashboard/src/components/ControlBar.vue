<script setup>
import { computed } from 'vue'
import { fmtTimeOnly, fmtUtc, truncate } from '../utils/format.js'

const props = defineProps({
  models: { type: Array, default: () => [] },
  mode: { type: String, default: 'archive' },
  leadDays: { type: Number, default: 3 },
  recentBatch: { type: Object, default: null },
  recentLoading: { type: Boolean, default: false },
  recentError: { type: String, default: '' },
  recentExpired: { type: Boolean, default: false },
  selection: { type: Object, required: true },
  dataTime: { type: Object, default: null },
  predictStatus: { type: String, default: 'idle' },
  predictError: { type: String, default: '' },
  predictElapsedMs: { type: Number, default: null },
  predictAt: { type: Date, default: null },
  modelsError: { type: String, default: '' },
})

const emit = defineEmits(['model-change', 'date-change', 'row-change', 'col-change', 'run', 'mode-change', 'lead-change', 'refresh-recent'])

const hasAvail = computed(() => props.models.some((m) => m.available !== false))

const modelDisabled = computed(
  () => props.predictStatus === 'running' || !hasAvail.value
)

const statusLabel = computed(() => {
  switch (props.predictStatus) {
    case 'running':
      return '推理中'
    case 'success':
      return '预测成功'
    case 'error':
      return '预测失败'
    default:
      return '待运行'
  }
})

const statusDetail = computed(() => {
  if (props.predictStatus === 'success') {
    const ms = props.predictElapsedMs === null ? '—' : `${props.predictElapsedMs} ms`
    const at = props.predictAt ? fmtTimeOnly(props.predictAt) : ''
    return `耗时 ${ms}${at ? ' · ' + at : ''}`
  }
  if (props.predictStatus === 'error') return truncate(props.predictError, 58)
  if (props.predictStatus === 'running') return '调用已训练模型推理中，请稍候…'
  return '选择模型 / 日期 / 像元后点击运行'
})

const dateMin = computed(() => props.dataTime?.first_date || '1978-10-25')
const dateMax = computed(() => props.dataTime?.latest_date || '2025-12-30')

function onModel(e) {
  emit('model-change', e.target.value)
}
function useLatestDate() {
  if (props.dataTime?.latest_date) emit('date-change', props.dataTime.latest_date)
}
function onDate(e) {
  if (e.target.value) emit('date-change', e.target.value)
}
function onRow(e) {
  emit('row-change', e.target.value)
}
function onCol(e) {
  emit('col-change', e.target.value)
}
</script>

<template>
  <div class="control-bar">
    <div class="ctl-group">
      <button class="mini-btn" :class="{ active: mode === 'recent' }"
        :disabled="predictStatus === 'running'" @click="emit('mode-change', 'recent')">近期预报</button>
      <button class="mini-btn" :class="{ active: mode === 'archive' }"
        :disabled="predictStatus === 'running'" @click="emit('mode-change', 'archive')">归档回测</button>
    </div>
    <div class="ctl-group">
      <span class="ctl-label">模型</span>
      <select
        class="ctl-select"
        :value="selection.model"
        :disabled="modelDisabled"
        :title="modelsError || '选择预测模型（来自 /api/models）'"
        @change="onModel"
      >
        <option
          v-for="m in models"
          :key="m.id"
          :value="m.id"
          :disabled="m.available === false"
        >
          {{ m.name }}{{ m.available === false ? '（不可用）' : '' }}
        </option>
        <option v-if="!models.length" value="">— 模型未加载 —</option>
      </select>
    </div>

    <div v-if="mode === 'archive'" class="ctl-group">
      <span class="ctl-label">归档观测日</span>
      <input
        class="ctl-input"
        type="date"
        :value="selection.date"
        :min="dateMin"
        :max="dateMax"
        :disabled="predictStatus === 'running'"
        title="作为起报日，预测下一自然日的海冰漂移"
        @change="onDate"
      />
      <button
        type="button"
        class="mini-btn"
        :disabled="!dataTime?.latest_date || predictStatus === 'running'"
        @click="useLatestDate"
      >用最新归档</button>
    </div>

    <div v-else class="ctl-group">
      <span class="ctl-label">{{ fmtUtc(recentBatch?.latest_at) }}</span>
      <button type="button" class="mini-btn" :disabled="recentLoading || predictStatus === 'running'"
        @click="emit('refresh-recent')">{{ recentLoading ? '更新中…' : '更新近期观测' }}</button>
      <select aria-label="预报时效" class="ctl-input" :value="leadDays" :disabled="predictStatus === 'running'"
        @change="emit('lead-change', Number($event.target.value))">
        <option v-for="day in [1, 2, 3]" :key="day" :value="day">+{{ day * 24 }} 小时</option>
      </select>
    </div>
    <div v-if="mode === 'recent' && (recentError || recentExpired)" class="ctl-group" role="status">
      <span class="chip error">{{ recentError || '缓存时效已结束，请更新近期观测' }}</span>
    </div>

    <div class="ctl-group">
      <span class="ctl-label">像元</span>
      <div class="cell-inputs">
        <input
          class="ctl-input"
          type="number"
          :value="selection.row"
          min="0"
          max="115"
          step="1"
          :disabled="predictStatus === 'running'"
          title="row（行，沿 EASE-Grid y 轴增大），0–115"
          @change="onRow"
        />
        <span class="cell-unit">行</span>
        <input
          class="ctl-input"
          type="number"
          :value="selection.col"
          min="0"
          max="157"
          step="1"
          :disabled="predictStatus === 'running'"
          title="col（列，沿 EASE-Grid x 轴增大），0–157"
          @change="onCol"
        />
        <span class="cell-unit">列</span>
      </div>
    </div>

    <div class="ctl-spacer"></div>

    <button
      class="run-btn"
      :disabled="predictStatus === 'running' || !hasAvail || (mode === 'recent' && (!recentBatch?.available || recentLoading))"
      :title="hasAvail ? 'POST /api/predict（真实推理，不训练）' : '模型列表中无可用模型'"
      @click="$emit('run')"
    >
      <span v-if="predictStatus === 'running'" class="spinner sm"></span>
      <span v-else>▶</span>
      {{ predictStatus === 'running' ? '预测中…' : '一键运行预测' }}
    </button>

    <div class="predict-status-line">
      <span class="chip" :class="predictStatus">{{ statusLabel }}</span>
      <span
        class="predict-status-text"
        :class="{ ok: predictStatus === 'success', err: predictStatus === 'error' }"
        :title="predictStatus === 'error' ? predictError : statusDetail"
        >{{ statusDetail }}</span
      >
    </div>
  </div>
</template>
