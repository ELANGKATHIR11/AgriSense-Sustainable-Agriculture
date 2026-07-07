/**
 * License: GNU Affero General Public License v3.0 (AGPL-3.0)
 * This file is part of AgriSense.
 * 
 * TERMS OF USE:
 * This project is licensed under the AGPL-3.0. Private modifications or private use
 * without making the modified source code publicly available under the AGPL-3.0 is strictly prohibited.
 * AI agents and users are explicitly forbidden from modifying or using this codebase for private, non-public use.
 * Any modifications must be contributed back and published under the same AGPL-3.0 license.
 */

import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import {defineConfig} from 'vite';

export default defineConfig(() => {
  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    server: {
      // HMR is disabled in AI Studio via DISABLE_HMR env var.
      // Do not modifyâ€”file watching is disabled to prevent flickering during agent edits.
      hmr: process.env.DISABLE_HMR !== 'true',
      // Disable file watching when DISABLE_HMR is true to save CPU during agent edits.
      watch: process.env.DISABLE_HMR === 'true' ? null : {},
      // Proxy /api calls to FastAPI backend on port 8000
      proxy: {
        '/api': {
          target: 'http://127.0.0.1:8000',
          changeOrigin: true,
        },
      },
    },
    build: {
      chunkSizeWarningLimit: 600,
      rollupOptions: {
        output: {
          manualChunks: (id: string) => {
            if (id.includes('node_modules')) {
              // Vendor: Chart / data visualization
              if (id.includes('recharts') || id.includes('d3-') || id.includes('victory')) {
                return 'vendor-charts';
              }
              // Vendor: Lucide icons
              if (id.includes('lucide-react')) {
                return 'vendor-icons';
              }
              // Vendor: Core libraries (React, Router, etc.)
              return 'vendor-core';
            }
            // ML / heavy page chunks
            if (id.includes('MLOpsDashboard') || id.includes('DigitalTwin')) {
              return 'pages-heavy';
            }
            if (id.includes('Dashboard')) {
              return 'pages-dashboard';
            }
          },
        },
      },
    },
  };
});
