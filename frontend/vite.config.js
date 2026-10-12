import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import { env as processEnv } from 'node:process'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: Object.fromEntries(
      ['auth', 'predictions', 'incidents', 'sos', 'alerts', 'predict', 'db', 'health'].map((route) => [
        `/${route}`,
        processEnv.VITE_API_PROXY_TARGET || 'http://127.0.0.1:8000',
      ]),
    ),
  },
})
