import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Proxy backend calls so the UI can run anywhere (incl. SSH-forwarded ports)
// and still talk to the FastAPI server. UI calls relative URLs like
// `/personas` / `/chat/...`; vite forwards them to localhost:8000 server-side.
export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      "/personas": "http://127.0.0.1:8000",
      "/chat":     "http://127.0.0.1:8000",
      "/health":   "http://127.0.0.1:8000",
    },
  },
});
