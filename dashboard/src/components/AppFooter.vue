<script setup>
import { computed } from 'vue'
import { fmtClock } from '../utils/format.js'

const props = defineProps({
  recent: { type: Boolean, default: false },
  dataTime: { type: Object, default: null },
  generatedAt: { type: Date, default: () => new Date() },
})

const genText = computed(() => fmtClock(props.generatedAt))
</script>

<template>
  <footer class="app-footer">
    <span v-if="recent" class="foot-item">
      近期观测 EUMETSAT OSI SAF · OSI-405-d · CC BY 4.0
      <a href="https://doi.org/10.15770/EUM_SAF_OSI_NRT_2007" target="_blank" rel="noreferrer">数据 DOI</a>
    </span>
    <span v-else class="foot-item">
      数据来源：NSIDC-0116 v4 Polar Pathfinder Daily 25km EASE-Grid Sea Ice Motion Vectors（Tschudi et al. 2019）· DOI
      <a href="https://doi.org/10.5067/INAWUWO7QH7B" target="_blank" rel="noreferrer">10.5067/INAWUWO7QH7B</a>
    </span>
    <span class="foot-sep"></span>
    <span class="foot-item grow">
      {{ recent ? '从最近观测结束时刻起报，按各日预测漂移矢量逐时积分。' :
        '归档预报为次日漂移矢量，逐时位置由该矢量线性积分。' }}
    </span>
    <span class="foot-sep"></span>
    <span class="foot-item fixed">数据批次：{{ dataTime?.latest_date || '—' }} · 构建 {{ (dataTime?.built_at || '').slice(0, 10) || '—' }}</span>
    <span class="foot-sep"></span>
    <span class="foot-item fixed">生成时间 {{ genText }} · v1.2.0</span>
  </footer>
</template>
