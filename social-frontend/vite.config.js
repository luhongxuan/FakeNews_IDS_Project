import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// 社群平台前端固定跑在 5174，關係圖前端維持 5173，兩者互不衝突、可各自獨立啟動
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    strictPort: true,
  },
});
