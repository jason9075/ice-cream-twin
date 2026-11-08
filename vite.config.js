import { defineConfig } from 'vite';

export default defineConfig({
  // 相對路徑，GitHub Pages 的子路徑 (/<repo>/) 也能正常載入
  base: './',
  build: {
    target: 'es2022',
  },
  // the simulation runs in the Rust server (just dev starts it on 8765); the page talks to it at /ws
  server: {
    proxy: { '/ws': { target: 'ws://localhost:8765', ws: true } },
  },
});
