import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The dev server proxies /api/* to the FastAPI service, so the browser never needs CORS.
// VITE_STATIC_DEMO=1 builds the read-only static demo (deploy/hf-static) with relative asset paths.
export default defineConfig({
  base: process.env.VITE_STATIC_DEMO === '1' ? './' : '/',
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.API_URL ?? 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
