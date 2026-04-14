import { Outlet } from "react-router-dom"
import { Header } from "./Header"
import { Footer } from "./Footer"

export function Layout() {
  return (
    <div className="relative flex min-h-screen flex-col">
      {/* Fond ambient avec quadrillage blueprint tres subtil sur toute la hauteur */}
      <div
        aria-hidden="true"
        className="pointer-events-none fixed inset-0 -z-10 opacity-[0.04]"
        style={{
          backgroundImage:
            "linear-gradient(oklch(0.72 0.18 155 / 0.6) 1px, transparent 1px), " +
            "linear-gradient(to right, oklch(0.72 0.18 155 / 0.6) 1px, transparent 1px)",
          backgroundSize: "64px 64px",
        }}
      />
      <Header />
      <main className="container mx-auto max-w-6xl flex-1 px-4 py-8">
        <Outlet />
      </main>
      <Footer />
    </div>
  )
}
