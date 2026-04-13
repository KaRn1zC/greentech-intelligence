import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"
import tailwindcss from "@tailwindcss/vite"
import path from "path"

// Le bundle est decoupe en plusieurs vendor chunks pour ameliorer la mise en
// cache navigateur et supprimer le warning "chunk > 500 kB". Les chunks sont
// regroupes par frequence de changement (vendors quasi-fixes vs code applicatif).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks(id: string) {
          if (!id.includes("node_modules")) return undefined
          if (id.includes("recharts") || id.includes("d3-")) return "chart-vendor"
          if (id.includes("react-router")) return "react-vendor"
          if (id.match(/[\\/](react|react-dom|scheduler)[\\/]/)) return "react-vendor"
          if (
            id.includes("lucide-react") ||
            id.includes("sonner") ||
            id.includes("class-variance-authority") ||
            id.includes("tailwind-merge") ||
            id.includes("clsx")
          ) {
            return "ui-vendor"
          }
          return undefined
        },
      },
    },
  },
})
