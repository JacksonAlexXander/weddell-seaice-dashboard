<script setup>
import { computed } from 'vue'
import { fmtClock } from '../utils/format.js'

const props = defineProps({
  dataTime: { type: Object, default: null },
  generatedAt: { type: Date, default: () => new Date() },
})

const genText = computed(() => fmtClock(props.generatedAt))
</script>

<template>
  <footer class="app-footer">
    <span class="foot-item">
      数据来源：NSIDC-0116 v4 Polar Pathfinder Daily 25km EASE-Grid Sea Ice Motion Vectors（Tschudi et al. 2019）· DOI
      <a href="https://doi.org/10.5067/INAWUWO7QH7B" target="_blank" rel="noreferrer">10.5067/INAWUWO7QH7B</a>
    </span>
    <span class="foot-sep"></span>
    <span class="foot-item grow">
      预测口径：日尺度产品仅给出未来 24h 平均漂移矢量，逐时位置 / 累计位移由该矢量线性积分得到（非逐时观测，不代表逐时真实路径）
    </span>
    <span class="foot-sep"></span>
    <span class="foot-item fixed">数据批次：{{ dataTime?.latest_date || '—' }} · 构建 {{ (dataTime?.built_at || '').slice(0, 10) || '—' }}</span>
    <span class="foot-sep"></span>
    <span class="foot-item fixed">生成时间 {{ genText }} · v1.1.0</span>
  </footer>
</template>
