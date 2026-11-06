import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// 开发/预览期把 /api 代理到 FastAPI 后端；生产构建由后端直接静态托管 dist，
// 前端默认使用同源 /api（可用 VITE_API_BASE 覆盖）。
const API_TARGET = process.env.VITE_API_PROXY_TARGET || 'http://127.0.0.1:8000'

const proxy = {
  '/api': {
    target: API_TARGET,
    changeOrigin: true,
  },
}

export default defineConfig({
  plugins: [vue()],
  base: '/',
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    proxy,
  },
  preview: {
    host: '127.0.0.1',
    port: 4173,
    strictPort: true,
    proxy,
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: false,
    chunkSizeWarningLimit: 1800,
  },
})
