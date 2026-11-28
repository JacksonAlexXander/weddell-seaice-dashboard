<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import PanelShell from './PanelShell.vue'
import { assets, cellAt, concentration, coordinates, drawWeddell, mapSize } from '../maps/weddell.js'
import { fmtInt, fmtUtc } from '../utils/format.js'

const props = defineProps({
  mode: { type: String, default: 'observed' },
  observedField: { type: Object, default: null },
  predictedField: { type: Object, default: null },
  selectedCell: { type: Object, required: true },
  path: { type: Object, default: null },
  selectedModel: { type: Object, default: null },
  loading: { type: Boolean, default: false },
  error: { type: String, default: '' },
})

const emit = defineEmits(['mode-change', 'select-cell', 'retry'])
const canvas = ref(null)
const layer = ref('concentration')
const hovered = ref(null)
const imageryError = ref('')
const field = computed(() => props.mode === 'predicted' ? props.predictedField : props.observedField)
const maximum = computed(() => Math.max(5, Math.ceil((field.value?.stats?.max_speed || 25) / 5) * 5))
const showConcentration = computed(() => layer.value === 'concentration')
const concentrationGradient = `linear-gradient(90deg, ${concentration.colorStops.map(
  ([value, rgb]) => `rgb(${rgb.join(',')}) ${(value - 15) / 85 * 100}%`,
).join(',')})`
const layers = [
  { id: 'concentration', label: '48 年平均密集度' },
  { id: 'speed', label: '漂移速度' },
  { id: 'satellite', label: '卫星底图' },
]

const subtitle = computed(() => {
  const f = field.value
  if (!f) return props.mode === 'predicted' ? '预测场' : '观测场'
  if (props.mode === 'predicted') {
    const target = f.valid_at ? fmtUtc(f.valid_at) : f.target_date
    return `${target || f.date} · ${f.model?.name || props.selectedModel?.name || ''}`
  }
  return `${f.date} · ${fmtInt(f.stats?.valid_cells)} 个有效像元`
})

function coordinateText({ row, col }) {
  const { lat, lon } = coordinates(row, col)
  return `${Math.abs(lat).toFixed(2)}°S ${Math.abs(lon).toFixed(2)}°${lon < 0 ? 'W' : 'E'}`
}
const selectedText = computed(() => {
  const { row, col } = props.selectedCell
  return `像元 (${row}, ${col}) · ${coordinateText(props.selectedCell)}`
})

let images
let iceMask
let frame = null
let observer
let disposed = false

function render() {
  frame = null
  const element = canvas.value
  if (!element || !images) return
  const w = element.clientWidth
  const h = element.clientHeight
  if (!(w && h)) return
  const density = Math.min(window.devicePixelRatio || 1, 2)
  element.width = Math.round(w * density)
  element.height = Math.round(h * density)
  const scale = Math.min((w - 48) / mapSize.width, (h - 38) / mapSize.height)
  if (scale <= 0) return
  const x = (w - scale * mapSize.width) / 2
  const y = (h - scale * mapSize.height) / 2 + 4
  const ctx = element.getContext('2d')
  ctx.setTransform(scale * density, 0, 0, scale * density, x * density, y * density)
  drawWeddell(ctx, { images, field: field.value, maximum: maximum.value, layer: layer.value,
    selected: props.selectedCell, path: props.path, unit: 1 / scale })
  viewport = { scale, x, y }
}

let viewport
function schedule() {
  if (frame === null && !disposed) frame = requestAnimationFrame(render)
}

function loadImage(url) {
  const image = new Image()
  image.src = url
  return image.decode().then(() => image)
}

async function loadImagery() {
  imageryError.value = ''
  try {
    const entries = await Promise.all(Object.entries(assets).map(async ([key, url]) => [key, await loadImage(url)]))
    if (disposed) return
    images = Object.fromEntries(entries)
    const maskCanvas = document.createElement('canvas')
    maskCanvas.width = mapSize.width
    maskCanvas.height = mapSize.height
    const ctx = maskCanvas.getContext('2d')
    ctx.drawImage(images.mask, 0, 0)
    iceMask = ctx.getImageData(0, 0, mapSize.width, mapSize.height).data
    schedule()
  } catch {
    if (!disposed) imageryError.value = '卫星地图资源加载失败'
  }
}

function hit(e) {
  if (!viewport) return null
  const { scale, x, y } = viewport
  const px = (e.offsetX - x) / scale
  const py = (e.offsetY - y) / scale
  if (px < 0 || py < 0 || px >= mapSize.width || py >= mapSize.height) return null
  const pixel = Math.floor(py) * mapSize.width + Math.floor(px)
  if (iceMask[pixel * 4 + 3]) return null
  const cell = cellAt(px, py)
  const speed = cell && field.value?.speed?.[cell.row]?.[cell.col]
  return Number.isFinite(speed) ? { ...cell, speed } : null
}

function selectCell(e) {
  const cell = hit(e)
  if (cell) emit('select-cell', { row: cell.row, col: cell.col })
}

function showCell(e) {
  const cell = hit(e)
  hovered.value = cell && {
    text: `(${cell.row}, ${cell.col}) · ${coordinateText(cell)} · ${cell.speed.toFixed(2)} cm/s`,
  }
}

onMounted(() => {
  observer = new ResizeObserver(schedule)
  observer.observe(canvas.value)
  loadImagery()
})

watch([field, layer, () => props.selectedCell.row, () => props.selectedCell.col, () => props.path], () => {
  hovered.value = null
  schedule()
})
onBeforeUnmount(() => {
  disposed = true
  observer?.disconnect()
  if (frame !== null) cancelAnimationFrame(frame)
})
</script>

<template>
  <PanelShell
    :title="mode === 'predicted' ? '海冰漂移场 · 预测' : '海冰漂移场 · 观测'"
    :subtitle="subtitle"
    :loading="loading"
    :error="imageryError"
    flush
    @retry="loadImagery"
  >
    <template #extra>
      <div class="map-controls" aria-label="漂移场类型">
        <button v-for="choice in [{ id: 'observed', label: '观测' }, { id: 'predicted', label: '预测' }]"
          :key="choice.id" type="button" class="mini-btn" :class="{ active: mode === choice.id }"
          :aria-pressed="mode === choice.id" @click="emit('mode-change', choice.id)">{{ choice.label }}</button>
      </div>
    </template>
    <div class="satellite-map">
      <div class="map-toolbar">
        <div class="map-controls" aria-label="地图色彩叠加层">
          <button v-for="item in layers" :key="item.id" type="button" class="mini-btn"
            :class="{ active: layer === item.id }" :aria-pressed="layer === item.id"
            @click="layer = item.id">{{ item.label }}</button>
        </div>
        <span class="map-point">{{ hovered?.text || selectedText }}</span>
      </div>
      <div class="map-viewport">
        <canvas ref="canvas" class="weddell-canvas"
          aria-label="威德尔海卫星地图与漂移矢量。可以点击有效海冰像元，或使用上方行列输入框选择。"
          @click="selectCell" @pointermove="showCell" @pointerleave="hovered = null" />
        <div v-if="error" class="map-warning" role="status">
          {{ error }} <button class="mini-btn" @click="emit('retry')">重试漂移场</button>
        </div>
      </div>
      <div class="map-key">
        <div v-if="showConcentration" class="map-scale">
          <span>密集度 15%</span>
          <span class="map-ramp" :style="{ background: concentrationGradient }"></span>
          <span>100%</span>
        </div>
        <div v-else-if="layer === 'speed'" class="map-scale">
          <span>速度 0</span>
          <span class="map-ramp speed-ramp"></span>
          <span>{{ maximum }} cm/s</span>
        </div>
        <span class="map-ground">陆地 / 冰盖</span>
        <span class="map-shelf">冰架</span>
        <span class="map-arrow">↗ {{ mode === 'predicted' ? '预测' : '观测' }}漂移</span>
      </div>
      <div class="map-sources">
        <span v-if="showConcentration">1978–2025 · {{ concentration.months }} 月 ·
          <a href="https://doi.org/10.7265/a98x-0f50">NOAA/NSIDC G02135 v4</a>
        </span>
        <a href="https://www.pgc.umn.edu/guides/misc/polar-satellite-imagery-resources/">Landsat LIMA / MODIS · USGS / BAS / NASA / PGC</a>
        <a href="https://doi.org/10.5285/96ffb7ba-69a9-4ec3-aede-07e746f1555a">SCAR ADD 7.13 · CC BY 4.0</a>
      </div>
    </div>
  </PanelShell>
</template>

<style scoped>
.satellite-map {
  height: 100%;
  min-height: 0;
  display: grid;
  grid-template-rows: auto minmax(0, 1fr) auto auto;
  padding: 5px 9px 6px;
  background: #081120;
}

.map-toolbar, .map-key, .map-sources, .map-controls, .map-scale {
  display: flex;
  align-items: center;
  gap: 5px;
  flex-wrap: wrap;
}
.map-toolbar { justify-content: space-between; }
.map-viewport { min-height: 0; position: relative; }
.weddell-canvas { display: block; width: 100%; height: 100%; cursor: crosshair; }
.map-point { color: #ffe19a; font: 10px var(--font-mono); }
.map-key, .map-sources { font-size: 10px; justify-content: center; color: #b7cadd; gap: 5px 11px; }
.map-sources { margin-top: 5px; font-size: 9px; }
.map-sources a { color: #9bc8e9; text-decoration: underline; text-underline-offset: 2px; }
.map-ramp { width: clamp(54px, 8vw, 105px); height: 6px; border-radius: 2px; }
.speed-ramp { background: linear-gradient(90deg, #12537a, #1692b4, #60e9ee, #ffd66c); }
.map-ground::before, .map-shelf::before { content: ''; display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 4px; }
.map-ground::before { background: #acbecb; }
.map-shelf::before { background: #bfdef2; }
.map-arrow { color: #f3faff; }
.map-warning {
  position: absolute;
  bottom: 4px;
  left: 8px;
  background: #181926;
  color: #fbb6bd;
  padding: 4px 6px;
  font-size: 10px;
}
</style>
