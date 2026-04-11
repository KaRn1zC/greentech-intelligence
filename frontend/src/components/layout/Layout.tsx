import { Outlet } from "react-router-dom"
import { Header } from "./Header"
import { Footer } from "./Footer"

export function Layout() {
  return (
    <div className="flex min-h-screen flex-col">
      <Header />
      <main className="container mx-auto max-w-5xl flex-1 px-4 py-6">
        <Outlet />
      </main>
      <Footer />
    </div>
  )
}
