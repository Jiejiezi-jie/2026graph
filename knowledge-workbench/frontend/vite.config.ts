import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const apiTarget = loadEnv(mode, '.', 'API_PROXY_TARGET').API_PROXY_TARGET || 'http://127.0.0.1:8000'
  return {
  plugins: [react()],
  build: { rollupOptions: { output: { manualChunks: { graph: ['cytoscape'], markdown: ['react-markdown'] } } } },
  server: {
    port: 5173,
    strictPort: true,
    proxy: { '/api': { target: apiTarget, timeout: 900000, proxyTimeout: 900000 } },
  },
  preview: { proxy: { '/api': { target: apiTarget, timeout: 900000, proxyTimeout: 900000 } } },
  }
})
