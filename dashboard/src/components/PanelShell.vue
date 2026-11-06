<script setup>
defineProps({
  title: { type: String, default: '' },
  subtitle: { type: String, default: '' },
  loading: { type: Boolean, default: false },
  error: { type: String, default: '' },
  empty: { type: Boolean, default: false },
  emptyText: { type: String, default: '暂无数据' },
  flush: { type: Boolean, default: false },
})

defineEmits(['retry'])
</script>

<template>
  <section class="panel">
    <header class="panel-hd">
      <div class="panel-hd-left">
        <span class="panel-accent"></span>
        <h3 class="panel-title">{{ title }}</h3>
        <span v-if="subtitle" class="panel-subtitle">{{ subtitle }}</span>
      </div>
      <div class="panel-extra">
        <slot name="extra"></slot>
      </div>
    </header>

    <div class="panel-bd" :class="{ flush }">
      <slot></slot>
    </div>

    <div v-if="loading" class="panel-state">
      <span class="spinner"></span>
      <span>加载中…</span>
    </div>
    <div v-else-if="error" class="panel-state error">
      <span class="state-msg">✕ 加载失败：{{ error }}</span>
      <button class="mini-btn" @click="$emit('retry')">重试</button>
    </div>
    <div v-else-if="empty" class="panel-state empty">
      <span class="state-msg">{{ emptyText }}</span>
    </div>
  </section>
</template>
