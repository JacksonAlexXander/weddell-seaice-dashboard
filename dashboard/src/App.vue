<script setup>
import { onBeforeUnmount, onMounted } from 'vue'
import AppHeader from './components/AppHeader.vue'
import AppFooter from './components/AppFooter.vue'
import KpiPanel from './components/KpiPanel.vue'
import AnnualTrendPanel from './components/AnnualTrendPanel.vue'
import MonthlyClimatePanel from './components/MonthlyClimatePanel.vue'
import CoveragePanel from './components/CoveragePanel.vue'
import ControlBar from './components/ControlBar.vue'
import IceFieldPanel from './components/IceFieldPanel.vue'
import Forecast24hPanel from './components/Forecast24hPanel.vue'
import PredictionCard from './components/PredictionCard.vue'
import MetricsComparePanel from './components/MetricsComparePanel.vue'
import BacktestPanel from './components/BacktestPanel.vue'
import StatusPanel from './components/StatusPanel.vue'
import { useDashboard } from './state/useDashboard.js'

const {
  state,
  usingMock,
  dataTime,
  activeDataTime,
  recentExpired,
  modelsList,
  selectedModel,
  currentFieldLoading,
  currentFieldError,
  selectedLatLon,
  mapPredictionPath,
  warnings,
  banner,
  predictStatusText,
  actions,
} = useDashboard()

const generatedAt = new Date()

onMounted(() => {
  actions.bootstrap()
})
onBeforeUnmount(() => {
  actions.dispose()
})

function onRowChange(v) {
  actions.setCell(v, state.selection.col)
}
function onColChange(v) {
  actions.setCell(state.selection.row, v)
}
function onSelectCell(cell) {
  actions.setCell(cell.row, cell.col)
}
function onFieldRetry() {
  if (state.mapMode === 'predicted') actions.loadPredictedField(true)
  else actions.loadField()
}

const retryOverview = () => actions.loadOverview()
const retryMetrics = () => actions.loadMetrics()
const retryBacktest = () => actions.loadBacktest()
</script>

<template>
  <div class="app-shell">
    <AppHeader
      :data-time="activeDataTime"
      :recent="state.mode === 'recent'"
      :health="state.health"
      :clock="state.clock"
      :predict-status="state.predictStatus"
      :predict-status-text="predictStatusText"
    />

    <div v-if="banner" class="global-alert" :class="banner.level">
      <span class="ga-badge">{{ banner.level === 'info' ? 'DEV' : '连接异常' }}</span>
      <span class="ga-text" :title="banner.text">{{ banner.text }}</span>
      <button class="mini-btn" @click="actions.retryConnection">重新连接</button>
    </div>

    <main class="screen-grid">
      <!-- 左栏：数据总览 / 48 年趋势 / 月度气候态 / 覆盖与缺失 -->
      <div class="col-left">
        <KpiPanel
          :overview="state.overview"
          :loading="state.overviewLoading"
          :error="state.overviewError || ''"
          @retry="retryOverview"
        />
        <AnnualTrendPanel
          :annual="state.overview?.annual"
          :loading="state.overviewLoading"
          :error="state.overviewError || ''"
          @retry="retryOverview"
        />
        <MonthlyClimatePanel
          :monthly="state.overview?.monthly"
          :loading="state.overviewLoading"
          :error="state.overviewError || ''"
          @retry="retryOverview"
        />
        <CoveragePanel
          :overview="state.overview"
          :data-time="dataTime"
          :loading="state.overviewLoading"
          :error="state.overviewError || ''"
          @retry="retryOverview"
        />
      </div>

      <!-- 中栏：控制条 / 漂移场 / 24h 预测曲线 -->
      <div class="col-center">
        <ControlBar
          :models="modelsList"
          :mode="state.mode"
          :lead-days="state.leadDays"
          :recent-batch="state.recentBatch"
          :recent-loading="state.recentLoading"
          :recent-error="state.recentError || ''"
          :recent-expired="recentExpired"
          @mode-change="actions.setMode"
          @lead-change="actions.setLeadDays"
          @refresh-recent="actions.refreshRecent"
          :selection="state.selection"
          :data-time="dataTime"
          :predict-status="state.predictStatus"
          :predict-error="state.predictError || ''"
          :predict-elapsed-ms="state.predictElapsedMs"
          :predict-at="state.predictAt"
          :models-error="state.modelsError || ''"
          @model-change="actions.setModel"
          @date-change="actions.setDate"
          @row-change="onRowChange"
          @col-change="onColChange"
          @run="actions.runPredict"
        />
        <IceFieldPanel
          :mode="state.mapMode"
          :observed-field="state.observedField"
          :predicted-field="state.predictedField"
          :selected-cell="state.selection"
          :selected-lat-lon="selectedLatLon"
          :path="mapPredictionPath"
          :selected-model="selectedModel"
          :prediction-input="state.prediction?.input || null"
          :loading="currentFieldLoading"
          :error="currentFieldError || ''"
          @mode-change="actions.setMapMode"
          @select-cell="onSelectCell"
          @retry="onFieldRetry"
        />
        <Forecast24hPanel :prediction="state.prediction" :running="state.predictStatus === 'running'" />
      </div>

      <!-- 右栏：预测结果 / 指标对比 / 历史回测 / 异常与数据时间 -->
      <div class="col-right">
        <PredictionCard
          :prediction="state.prediction"
          :status="state.predictStatus"
          :error="state.predictError || ''"
          :elapsed-ms="state.predictElapsedMs"
          :predict-at="state.predictAt"
          :selected-model="selectedModel"
          @run="actions.runPredict"
        />
        <MetricsComparePanel
          :metrics="state.metrics"
          :loading="state.metricsLoading"
          :error="state.metricsError || ''"
          @retry="retryMetrics"
        />
        <BacktestPanel v-if="state.mode === 'archive'"
          :backtest="state.backtest"
          :loading="state.backtestLoading"
          :error="state.backtestError || ''"
          :range="state.backtestRange"
          :model="selectedModel"
          :cell="{ row: state.selection.row, col: state.selection.col }"
          @range-change="actions.setRange"
          @refresh="actions.loadBacktest"
          @retry="retryBacktest"
        />
        <StatusPanel
          :warnings="warnings"
          :data-time="activeDataTime"
          :recent="state.mode === 'recent'"
          :prediction="state.prediction"
          :backtest-at="state.backtestAt"
          :predict-at="state.predictAt"
          :selection="state.selection"
        />
      </div>
    </main>

    <AppFooter :data-time="activeDataTime" :recent="state.mode === 'recent'" :generated-at="generatedAt" />
  </div>
</template>
