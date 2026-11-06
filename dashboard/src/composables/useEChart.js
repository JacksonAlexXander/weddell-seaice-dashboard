import { onBeforeUnmount, onMounted, shallowRef, watch } from 'vue'
import * as echarts from 'echarts'

/**
 * ECharts 生命周期封装：
 * - 容器尺寸变化（ResizeObserver / window resize）自动 resize（rAF 节流）
 * - option 为 computed ref 时自动 setOption（notMerge）
 * - 支持绑定事件（'click' 等）与 zrender 事件（'zr:click'）
 */
export function useEChart(elRef, optionRef, { events } = {}) {
  const chart = shallowRef(null)
  let ro = null
  let disposed = false
  let rafId = 0

  function render() {
    if (!chart.value || disposed) return
    const option = optionRef.value
    if (!option || Object.keys(option).length === 0) {
      chart.value.clear()
      return
    }
    chart.value.setOption(option, { notMerge: true })
  }

  function resize() {
    if (!chart.value || disposed) return
    chart.value.resize()
  }

  function scheduleResize() {
    if (rafId) return
    rafId = requestAnimationFrame(() => {
      rafId = 0
      resize()
    })
  }

  onMounted(() => {
    if (!elRef.value) return
    chart.value = echarts.init(elRef.value, null, { renderer: 'canvas' })
    render()
    ro = new ResizeObserver(scheduleResize)
    ro.observe(elRef.value)
    window.addEventListener('resize', scheduleResize)

    if (events) {
      Object.entries(events).forEach(([key, handler]) => {
        if (key.startsWith('zr:')) chart.value.getZr().on(key.slice(3), handler)
        else chart.value.on(key, handler)
      })
    }
  })

  watch(optionRef, render, { flush: 'post' })

  onBeforeUnmount(() => {
    disposed = true
    if (rafId) cancelAnimationFrame(rafId)
    if (ro) ro.disconnect()
    window.removeEventListener('resize', scheduleResize)
    if (chart.value) {
      chart.value.dispose()
      chart.value = null
    }
  })

  return { chart, render, resize }
}
