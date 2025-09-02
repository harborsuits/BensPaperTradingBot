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
      port: 3005,
      proxy: {
        '/api':    { target: 'http://localhost:4000', changeOrigin: true },
        '/metrics':{ target: 'http://localhost:4000', changeOrigin: true },
        '/health': { target: 'http://localhost:4000', changeOrigin: true },
        '/ws':     { target: 'ws://localhost:4000', ws: true, changeOrigin: true },
      },
    },
  }
})
