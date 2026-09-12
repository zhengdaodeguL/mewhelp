import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'node:url'

/**
 * 构建产物直接落到后端静态目录 —— 这样 `python -m mewhelp.cli serve`
 * 不需要任何 Node/npm/网络就能把界面跑起来。前端只在**开发/构建时**需要 Node。
 */
const staticDir = fileURLToPath(new URL('../src/mewhelp/api/static', import.meta.url))

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: staticDir,
    emptyOutDir: true,
    // 产物文件名带 hash，避免浏览器缓存住旧版本
    assetsDir: 'assets',
  },
  server: {
    port: 5173,
    // 开发时前端跑 5173，接口转发到本地后端，免去跨域配置
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
})
