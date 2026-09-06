import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: { rollupOptions: { output: { manualChunks: { graph: ['cytoscape'], markdown: ['react-markdown'] } } } },
  server: {
    port: 5173,
    strictPort: true,
    proxy: { '/api': { target: 'http://127.0.0.1:8000', timeout: 900000, proxyTimeout: 900000 } },
  },
  preview: { proxy: { '/api': { target: 'http://127.0.0.1:8000', timeout: 900000, proxyTimeout: 900000 } } },
})
