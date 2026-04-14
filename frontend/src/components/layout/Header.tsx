import { useEffect, useState } from "react"
import { Link, NavLink, useNavigate } from "react-router-dom"
import { Command, LogOut, User } from "lucide-react"
import { Button } from "@/components/ui/button"
import { StatusIndicator } from "@/components/ui/status-indicator"
import { LeafCircuitLogo } from "@/components/layout/LeafCircuitLogo"
import { ModeToggle } from "@/components/layout/ModeToggle"
import { CommandPalette } from "@/components/layout/CommandPalette"
import { useAuth } from "@/hooks/useAuth"
import { cn } from "@/lib/utils"

type HealthStatus = "ok" | "degraded" | "down" | "unknown"

/**
 * Interroge le endpoint `/health` de l'API toutes les 30 secondes pour
 * alimenter l'indicateur de statut visible dans le header.
 */
function useApiHealth(): HealthStatus {
  const [status, setStatus] = useState<HealthStatus>("unknown")

  useEffect(() => {
    let cancelled = false
    const base = import.meta.env.VITE_API_URL ?? ""

    const check = async () => {
      try {
        const res = await fetch(`${base}/health`, { cache: "no-store" })
        if (cancelled) return
        setStatus(res.ok ? "ok" : "degraded")
      } catch {
        if (!cancelled) setStatus("down")
      }
    }

    check()
    const interval = setInterval(check, 30_000)
    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [])

  return status
}

export function Header() {
  const { isAuthenticated, user, logout } = useAuth()
  const navigate = useNavigate()
  const apiStatus = useApiHealth()

  const handleLogout = () => {
    logout()
    navigate("/login")
  }

  return (
    <>
      <header
        className={cn(
          "sticky top-0 z-50 w-full border-b border-border/60",
          "bg-background/80 backdrop-blur-md supports-[backdrop-filter]:bg-background/60",
        )}
      >
        <div className="container mx-auto flex h-16 max-w-6xl items-center justify-between gap-4 px-4">
          <Link to="/" className="group flex items-center gap-2.5 font-display font-semibold">
            <LeafCircuitLogo size={28} />
            <span className="text-base tracking-tight">
              GreenTech <span className="text-gradient-eco">Intelligence</span>
            </span>
          </Link>

          <nav className="hidden items-center gap-1 md:flex">
            <NavLink
              to="/"
              className={({ isActive }) =>
                cn(
                  "rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
                  isActive
                    ? "text-foreground"
                    : "text-muted-foreground hover:text-foreground",
                )
              }
              end
            >
              Dashboard
            </NavLink>
          </nav>

          <div className="flex items-center gap-2">
            <StatusIndicator
              status={apiStatus}
              label={apiStatus === "ok" ? "API" : undefined}
              className="hidden sm:inline-flex"
            />

            <kbd
              className={cn(
                "hidden items-center gap-1 rounded-md border border-border/60 bg-muted/40 px-2 py-1",
                "font-mono text-[10px] uppercase tracking-wider text-muted-foreground",
                "md:inline-flex",
              )}
              aria-hidden="true"
            >
              <Command className="h-3 w-3" />
              K
            </kbd>

            <ModeToggle />

            {isAuthenticated ? (
              <div className="flex items-center gap-2">
                <span
                  className="hidden items-center gap-1.5 font-mono text-xs text-muted-foreground lg:inline-flex"
                  title={user?.email}
                >
                  <User className="h-3.5 w-3.5" />
                  {user?.email}
                </span>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleLogout}
                  aria-label="Se deconnecter"
                >
                  <LogOut className="h-4 w-4" />
                  <span className="hidden sm:inline">Deconnexion</span>
                </Button>
              </div>
            ) : (
              <Button variant="outline" size="sm" onClick={() => navigate("/login")}>
                Connexion
              </Button>
            )}
          </div>
        </div>
      </header>
      <CommandPalette />
    </>
  )
}
