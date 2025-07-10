import { defineConfig } from 'vite'

export default defineConfig({
  // Base URL for assets
  base: './',
  
  // Development server configuration
  server: {
    port: 3000,
    host: '127.0.0.1',
    open: false, // Don't auto-open browser
    cors: true,
    // Proxy API requests to backend during development
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        secure: false
      }
    }
  },
  
  // Build configuration
  build: {
    outDir: 'dist',
    sourcemap: true
  },
  
  // Path resolution
  resolve: {
    alias: {
      '@': '/src'
    }
  }
}) 