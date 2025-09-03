import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const isElectron = !!process.env.BUILD_TARGET_ELECTRON
  return {
    plugins: [react()],
    // IMPORTANT: relative base when loading from file:// in packaged desktop apps
    base: isElectron ? './' : '/',
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    server: {
      port: 3003,
      proxy: {
        '/api':    { target: 'http://localhost:3002', changeOrigin: true },
        '/auth':   { target: 'http://localhost:3002', changeOrigin: true },
        '/metrics':{ target: 'http://localhost:3002', changeOrigin: true },
        '/health': { target: 'http://localhost:3002', changeOrigin: true },
        '/ws':     { target: 'ws://localhost:3002', ws: true, changeOrigin: true },
        '/context':{ target: 'http://localhost:3002', changeOrigin: true },
        '/decisions':{ target: 'http://localhost:3002', changeOrigin: true },
        '/portfolio':{ target: 'http://localhost:3002', changeOrigin: true },
        '/safety':{ target: 'http://localhost:3002', changeOrigin: true },
        '/strategies':{ target: 'http://localhost:3002', changeOrigin: true },
        '/trades':{ target: 'http://localhost:3002', changeOrigin: true },
        '/alerts':{ target: 'http://localhost:3002', changeOrigin: true },
        '/data':{ target: 'http://localhost:3002', changeOrigin: true },
        '/watchlists':{ target: 'http://localhost:3002', changeOrigin: true },
        '/orders':{ target: 'http://localhost:3002', changeOrigin: true },
        '/jobs':{ target: 'http://localhost:3002', changeOrigin: true },
      },
    },
  }
})