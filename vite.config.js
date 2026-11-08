import { defineConfig } from 'vite';

export default defineConfig({
  // 相對路徑，GitHub Pages 的子路徑 (/<repo>/) 也能正常載入
  base: './',
  build: {
    target: 'es2022',
  },
});
