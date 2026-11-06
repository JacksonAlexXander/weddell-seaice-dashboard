<script setup>
import { computed } from 'vue'
import { fmtClock, fmtUtc } from '../utils/format.js'

const props = defineProps({
  dataTime: { type: Object, default: null },
  health: { type: Object, required: true },
  clock: { type: Date, required: true },
  predictStatus: { type: String, default: 'idle' },
  predictStatusText: { type: String, default: '待运行' },
})

const clockText = computed(() => fmtClock(props.clock))
const builtAt = computed(() => fmtUtc(props.dataTime?.built_at))

const connText = computed(() => {
  if (!props.health.checked) return '检测中…'
  return props.health.ok ? 'API 正常' : '未连接'
})

const statusText = computed(() => {
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
</script>

<template>
  <header class="app-header">
    <div class="brand">
      <svg class="brand-mark" viewBox="0 0 48 48" fill="none" aria-hidden="true">
        <path
          d="M24 3.5 42 13.6v20.8L24 44.5 6 34.4V13.6z"
          stroke="#22d3ee"
          stroke-width="1.3"
          stroke-opacity=".85"
        />
        <path
          d="M24 11 36 17.8v12.4L24 37 12 30.2V17.8z"
          fill="rgba(34,211,238,.12)"
          stroke="#22d3ee"
          stroke-width=".9"
          stroke-opacity=".6"
        />
        <path d="M24 16.5 30.5 20v8L24 31.5 17.5 28v-8z" fill="rgba(34,211,238,.28)" stroke="#67e8f9" stroke-width=".8" />
        <path d="M24 3.5v7M24 40v4.5M6 13.6l6.2 3.6M35.8 30.8 42 34.4M42 13.6l-6.2 3.6M12.2 30.8 6 34.4" stroke="#22d3ee" stroke-opacity=".45" stroke-width=".9" />
      </svg>
      <div class="brand-text">
        <h1>南极威德尔海海冰漂移智能预测大屏</h1>
        <div class="brand-sub">WEDDELL SEA ICE DRIFT · ML PREDICTION CENTER</div>
      </div>
      <div class="badges">
        <span class="badge">NSIDC-0116 v4</span>
        <span class="badge purple">48 年 · 17,188 帧</span>
        <span class="badge">25 km EASE-Grid</span>
      </div>
    </div>

    <div class="header-right">
      <div class="hstat">
        <span class="hstat-label">最新观测</span>
        <span class="hstat-value accent mono">{{ dataTime?.latest_date || '—' }}</span>
      </div>
      <div class="hstat">
        <span class="hstat-label">数据集构建</span>
        <span class="hstat-value small mono">{{ builtAt }}</span>
      </div>
      <div class="hstat">
        <span class="hstat-label">实时时钟 · 北京时间</span>
        <span class="hstat-value clock-value mono">{{ clockText }}</span>
      </div>
      <div class="hstat">
        <span class="hstat-label">后端连接</span>
        <span class="conn">
          <i class="dot" :class="health.ok ? 'ok' : health.checked ? 'bad' : 'idle'"></i>
          <span :style="{ color: health.ok ? '#6ee7b7' : health.checked ? '#fca5a5' : '#94a3b8' }">
            {{ connText }}
          </span>
        </span>
      </div>
      <div class="hstat">
        <span class="hstat-label">预测状态</span>
        <span class="chip" :class="predictStatus" :title="predictStatusText">
          <span v-if="predictStatus === 'running'" class="spinner sm"></span>
          {{ statusText }}
        </span>
      </div>
    </div>
  </header>
</template>
