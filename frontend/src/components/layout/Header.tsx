import { Link, useNavigate } from "react-router-dom"
import { Leaf, LogOut, User } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useAuth } from "@/hooks/useAuth"

export function Header() {
  const { isAuthenticated, user, logout } = useAuth()
  const navigate = useNavigate()

  const handleLogout = () => {
    logout()
    navigate("/login")
  }

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto flex h-14 max-w-5xl items-center justify-between px-4">
        <Link to="/" className="flex items-center gap-2 font-semibold">
          <Leaf className="h-5 w-5 text-green-600" />
          <span>GreenTech Intelligence</span>
        </Link>

        <nav className="flex items-center gap-4">
          <Link to="/" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
            Dashboard
          </Link>

          {isAuthenticated ? (
            <div className="flex items-center gap-3">
              <span className="hidden sm:inline text-sm text-muted-foreground">
                <User className="mr-1 inline h-3.5 w-3.5" />
                {user?.email}
              </span>
              <Button variant="ghost" size="sm" onClick={handleLogout}>
                <LogOut className="h-4 w-4" />
                <span className="hidden sm:inline">Deconnexion</span>
              </Button>
            </div>
          ) : (
            <Button variant="outline" size="sm" onClick={() => navigate("/login")}>
              Connexion
            </Button>
          )}
        </nav>
      </div>
    </header>
  )
}
