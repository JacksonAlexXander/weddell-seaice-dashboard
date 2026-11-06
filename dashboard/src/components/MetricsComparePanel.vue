<script setup>
import { computed, ref } from 'vue'
import PanelShell from './PanelShell.vue'
import { useEChart } from '../composables/useEChart.js'
import { C, baseTooltip, catAxis, tipRow, valAxis } from '../charts/theme.js'
import { fmt, fmtInt, signedPct } from '../utils/format.js'

const props = defineProps({
  metrics: { type: Object, default: null },
  loading: { type: Boolean, default: false },
  error: { type: String, default: '' },
})
defineEmits(['retry'])

const el = ref(null)

const SHORT_BY_ID = {
  ridge: 'Ridge',
  random_forest: 'Random Forest',
  simvp: 'SimVP',
  persistence: 'Persistence',
  climatology: 'Climatology',
}
const COLOR_BY_ID = {
  ridge: '#22d3ee',
  random_forest: '#a78bfa',
  simvp: '#f59e0b',
  persistence: '#64748b',
  climatology: '#475569',
}

const models = computed(() => props.metrics?.models || [])

const ordered = computed(() => {
  const list = [...models.value]
  const rank = { ridge: 0, random_forest: 1, simvp: 2, persistence: 3, climatology: 4 }
  list.sort((a, b) => (rank[a.id] ?? 9) - (rank[b.id] ?? 9))
  return list
})

const highlights = computed(() => ordered.value.filter((m) => ['ridge', 'random_forest'].includes(m.id)))

const delta = computed(() => {
  const ridge = ordered.value.find((m) => m.id === 'ridge')
  const rf = ordered.value.find((m) => m.id === 'random_forest')
  if (!ridge || !rf) return null
  return {
    rmse: Number((rf.vec_rmse - ridge.vec_rmse).toFixed(2)),
    mae: Number((rf.mae_vec - ridge.mae_vec).toFixed(2)),
    skill: Number((rf.skill_vs_persistence - ridge.skill_vs_persistence).toFixed(3)),
  }
})

function isHl(m) {
  return m.is_highlight === true || m.id === 'ridge' || m.id === 'random_forest'
}

function displayName(m) {
  // 契约中 RF 的展示名是 "Random Forest"；后端个别版本 metrics.short 会截成 "Random"，
  // 这里按 id 归一化，避免表格/坐标轴出现截断名。
  if (m.id === 'random_forest') return 'Random Forest'
  return m.short || SHORT_BY_ID[m.id] || m.name || m.id
}

function modelColor(m) {
  return m.color || COLOR_BY_ID[m.id] || '#64748b'
}

const option = computed(() => {
  const list = ordered.value
  if (!list.length) return {}
  const names = list.map((m) => displayName(m))
  const rmse = list.map((m) => ({
    value: m.vec_rmse,
    itemStyle: {
      color: isHl(m) ? modelColor(m) : 'rgba(100, 116, 139, 0.42)',
      borderRadius: [2, 2, 0, 0],
      shadowBlur: isHl(m) ? 10 : 0,
      shadowColor: isHl(m) ? modelColor(m) : 'transparent',
    },
  }))
  const mae = list.map((m) => ({
    value: m.mae_vec,
    itemStyle: {
      color: isHl(m) ? 'rgba(148, 163, 184, 0.75)' : 'rgba(100, 116, 139, 0.2)',
      borderRadius: [2, 2, 0, 0],
    },
  }))
  return {
    animationDuration: 500,
    grid: { left: 2, right: 4, top: 16, bottom: 2, containLabel: true },
    tooltip: baseTooltip({
      formatter: (params) => {
        const idx = params[0]?.dataIndex ?? 0
        const m = list[idx]
        return `<div style="font-weight:600;color:${m.color || '#a5f3fc'};margin-bottom:3px;">${m.name}</div>${tipRow(
          '矢量 RMSE',
          `${fmt(m.vec_rmse)} cm/s`
        )}${tipRow('矢量 MAE', `${fmt(m.mae_vec)} cm/s`)}${tipRow('RMSE u / v', `${fmt(m.rmse_u)} / ${fmt(m.rmse_v)}`)}${tipRow(
          'R² u',
          fmt(m.r2_u, 3)
        )}${tipRow('技能分', signedPct(m.skill_vs_persistence))}${tipRow('推理延迟', `${fmt(m.latency_ms, 1)} ms`)}`
      },
    }),
    xAxis: catAxis(names, {
      axisLabel: { color: C.muted, fontSize: 8.5, interval: 0 },
    }),
    yAxis: [
      valAxis({
        name: 'cm/s',
        nameGap: 6,
        axisLabel: { color: C.muted, fontSize: 8.5 },
      }),
      valAxis({
        name: '技能',
        nameGap: 6,
        splitLine: { show: false },
        axisLabel: {
          color: 'rgba(245, 158, 11, 0.75)',
          fontSize: 8.5,
          formatter: (v) => `${Math.round(v * 100)}%`,
        },
        min: (v) => Math.min(0, Math.floor(v.min * 10) / 10 - 0.05),
      }),
    ],
    series: [
      {
        name: '矢量 RMSE',
        type: 'bar',
        data: rmse,
        barWidth: 9,
        barGap: '16%',
        emphasis: { itemStyle: { shadowBlur: 8, shadowColor: 'rgba(34,211,238,.5)' } },
      },
      {
        name: '矢量 MAE',
        type: 'bar',
        data: mae,
        barWidth: 9,
      },
      {
        name: '技能分',
        type: 'line',
        yAxisIndex: 1,
        data: list.map((m) => m.skill_vs_persistence),
        symbol: 'circle',
        symbolSize: 5,
        lineStyle: { width: 1.5, color: C.amber },
        itemStyle: { color: C.amber, borderColor: '#0b1220', borderWidth: 1 },
      },
    ],
  }
})

useEChart(el, option)

const subtitle = computed(() => {
  const p = props.metrics?.protocol
  if (!p) return '统一评估协议 · 柱：RMSE / MAE（cm/s）· 线：技能分'
  return `287 窗口 · ${(p.n_samples / 10000).toFixed(1)} 万样本 · 柱 RMSE/MAE · 线 技能分`
})

const hasData = computed(() => ordered.value.length > 0)
</script>

<template>
  <PanelShell
    title="Ridge vs Random Forest 指标对比"
    :subtitle="subtitle"
    :loading="loading"
    :error="error"
    :empty="!loading && !error && !hasData"
    empty-text="暂无指标数据（/api/metrics）"
    @retry="$emit('retry')"
  >
    <div class="chart-box" :style="{ height: '92px' }">
      <div ref="el" style="position: absolute; inset: 0"></div>
    </div>

    <div v-if="delta" class="metrics-delta">
      <span>RF 相对 Ridge：</span>
      <span>
        vec RMSE <b :class="delta.rmse < 0 ? 'up' : 'down'">{{ delta.rmse > 0 ? '+' : '' }}{{ delta.rmse }}</b>
        cm/s
      </span>
      <span>
        MAE <b :class="delta.mae < 0 ? 'up' : 'down'">{{ delta.mae > 0 ? '+' : '' }}{{ delta.mae }}</b>
      </span>
      <span>
        技能分 <b :class="delta.skill > 0 ? 'up' : 'down'">{{ signedPct(delta.skill) }}</b>
      </span>
    </div>

    <div v-if="hasData" class="metric-table-wrap">
      <table class="metric-table">
        <thead>
          <tr>
            <th>模型</th>
            <th>vec RMSE</th>
            <th>RMSE_u</th>
            <th>RMSE_v</th>
            <th>R²_u</th>
            <th>技能分</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="m in ordered"
            :key="m.id"
            :class="{ hl: isHl(m), baseline: m.is_baseline && !isHl(m) }"
          >
            <td>
              <span class="model-cell">
                <i class="model-dot" :style="{ background: modelColor(m) }"></i>{{ displayName(m) }}
              </span>
            </td>
            <td>{{ fmt(m.vec_rmse) }}</td>
            <td>{{ fmt(m.rmse_u) }}</td>
            <td>{{ fmt(m.rmse_v) }}</td>
            <td>{{ fmt(m.r2_u, 3) }}</td>
            <td :style="{ color: m.skill_vs_persistence > 0 ? '#6ee7b7' : m.skill_vs_persistence < 0 ? '#fca5a5' : '#94a3b8' }">
              {{ m.is_baseline && m.skill_vs_persistence === 0 ? '基准' : signedPct(m.skill_vs_persistence) }}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </PanelShell>
</template>
