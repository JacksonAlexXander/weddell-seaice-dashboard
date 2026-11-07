<script setup>
import { computed } from 'vue'
import PanelShell from './PanelShell.vue'
import { bearingLabel, fmt } from '../utils/format.js'

const props = defineProps({
  prediction: { type: Object, default: null },
  status: { type: String, default: 'idle' },
  error: { type: String, default: '' },
  elapsedMs: { type: Number, default: null },
  predictAt: { type: Date, default: null },
  selectedModel: { type: Object, default: null },
})

defineEmits(['run'])

const pred = computed(() => props.prediction?.prediction || null)
const SHORT = {
  ridge: 'Ridge',
  random_forest: 'Random Forest',
  simvp: 'SimVP',
  persistence: '持续性基线',
  climatology: '气候态基线',
}
const modelName = computed(
  () => props.prediction?.model?.name || props.selectedModel?.name || '—'
)
const modelShort = computed(() => {
  const id = props.prediction?.model?.id || props.selectedModel?.id
  return SHORT[id] || modelName.value
})
const targetDate = computed(
  () => props.prediction?.data_time?.next_frame_date || props.prediction?.series_24h?.target_date || '—'
)
const lastObs = computed(() => props.prediction?.data_time?.last_obs_date || '—')

const bearing = computed(() => {
  const b = pred.value?.bearing_deg
  return b === null || b === undefined ? null : Number(b)
})

const compass = computed(() => bearingLabel(bearing.value))

const subTitle = computed(() => {
  if (!props.prediction) return '尚未运行'
  return `${modelShort.value} · ${lastObs.value} → ${targetDate.value}`
})

function fmtNum(v, d = 2) {
  return fmt(v, d)
}
</script>

<template>
  <PanelShell
    title="预测结果"
    :subtitle="subTitle"
    :loading="status === 'running'"
    :error="status === 'error' ? error : ''"
    @retry="$emit('run')"
  >
    <template #extra>
      <span v-if="prediction?.metrics" class="tag purple">
        skill {{ (Number(prediction.metrics.skill_vs_persistence) * 100).toFixed(1) }}%
      </span>
      <span v-if="status === 'success'" class="chip success">已返回</span>
    </template>

    <template v-if="pred">
      <div class="pred-hero">
        <div class="pred-speed">
          <span class="pred-speed-value">{{ fmtNum(pred.speed, 2) }}</span>
          <span class="pred-speed-unit">cm/s</span>
          <span class="pred-speed-label">预测漂移速度</span>
        </div>
        <div class="compass" :title="`地理方位角 ${fmtNum(bearing, 1)}°（${compass.zh}）`">
          <svg viewBox="0 0 40 40" fill="none" aria-hidden="true">
            <circle cx="20" cy="20" r="15" stroke="rgba(56,189,248,.18)" stroke-width=".8" />
            <path d="M20 5.5v3M20 31.5v3M5.5 20h3M31.5 20h3" stroke="rgba(125,211,252,.5)" stroke-width=".9" />
            <text x="20" y="9.5" text-anchor="middle" fill="rgba(125,211,252,.75)" font-size="4.6">N</text>
            <g :transform="`rotate(${bearing ?? 0} 20 20)`">
              <line x1="20" y1="29" x2="20" y2="12" stroke="#fbbf24" stroke-width="1.7" stroke-linecap="round" />
              <path d="M20 9.5l3.4 5.4h-6.8z" fill="#fbbf24" />
            </g>
          </svg>
          <span class="compass-label">{{ compass.zh }} {{ fmtNum(bearing, 1) }}°</span>
        </div>
      </div>

      <div class="pred-meta-grid">
        <div class="pred-meta">
          <span class="pred-meta-label">u 分量</span>
          <span class="pred-meta-value cy">{{ fmtNum(pred.u) }} <small>cm/s</small></span>
        </div>
        <div class="pred-meta">
          <span class="pred-meta-label">v 分量</span>
          <span class="pred-meta-value cy">{{ fmtNum(pred.v) }} <small>cm/s</small></span>
        </div>
        <div class="pred-meta">
          <span class="pred-meta-label">24h 位移</span>
          <span class="pred-meta-value amber">{{ fmtNum(pred.displacement_km_24h) }} <small>km</small></span>
        </div>
        <div class="pred-meta">
          <span class="pred-meta-label">相对持续性</span>
          <span class="pred-meta-value" style="color: #a5f3fc">
            {{ Number(pred.delta_speed_vs_persistence) >= 0 ? '+' : '' }}{{ fmtNum(pred.delta_speed_vs_persistence) }}
            <small>cm/s</small>
          </span>
        </div>
        <div class="pred-meta">
          <span class="pred-meta-label">推理耗时（服务端）</span>
          <span class="pred-meta-value">{{ elapsedMs ?? prediction?.elapsed_ms ?? '—' }} <small>ms</small></span>
        </div>
        <div class="pred-meta">
          <span class="pred-meta-label">目标帧</span>
          <span class="pred-meta-value">{{ targetDate }}</span>
        </div>
      </div>

      <div class="pred-foot">
        <span>
          输入像元 <b>({{ prediction.input?.row }}, {{ prediction.input?.col }})</b>
          · <b>{{ fmtNum(prediction.input?.lat, 2) }}°, {{ fmtNum(prediction.input?.lon, 2) }}°</b>
          · 观测 <b>{{ fmtNum(prediction.input?.speed) }} cm/s</b>
        </span>
        <span class="mono dim" style="margin-left: auto">req {{ prediction.request_id || '—' }}</span>
      </div>
    </template>

    <div v-else class="pred-placeholder">
      <span class="big-icon">◈</span>
      <span>尚未运行预测：选择模型 / 日期 / 像元后点击「一键运行预测」</span>
      <button v-if="status === 'error'" class="mini-btn" @click="$emit('run')">重新运行</button>
    </div>
  </PanelShell>
</template>

<style scoped>
.pred-meta-value small {
  font-size: 9px;
  color: var(--text-3);
  font-family: var(--font-sans);
}
</style>
