import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  publicDir: "mobile-public",
  server: {
    host: "127.0.0.1",
    port: 1421,
    strictPort: true,
    proxy: { "/api": "http://127.0.0.1:8787" },
  },
  build: {
    outDir: "dist-mobile",
    rollupOptions: { input: "mobile.html" },
  },
});
