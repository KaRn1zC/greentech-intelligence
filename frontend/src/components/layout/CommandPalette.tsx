import { useEffect, useState } from "react"
import { useNavigate } from "react-router-dom"
import { useTheme } from "next-themes"
import {
  BarChart3,
  FileText,
  Leaf,
  LogOut,
  Moon,
  Search,
  Sun,
} from "lucide-react"
import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
} from "@/components/ui/command"
import { useAuth } from "@/hooks/useAuth"

/**
 * Palette de commandes accessible via Ctrl/Cmd+K.
 *
 * Fournit un acces rapide aux principales pages et actions du projet :
 * navigation, bascule de theme, deconnexion, recherche rapide.
 */
export function CommandPalette() {
  const [open, setOpen] = useState(false)
  const navigate = useNavigate()
  const { setTheme } = useTheme()
  const { isAuthenticated, logout } = useAuth()

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        setOpen((prev) => !prev)
      }
    }
    document.addEventListener("keydown", handleKey)
    return () => document.removeEventListener("keydown", handleKey)
  }, [])

  const run = (fn: () => void) => {
    setOpen(false)
    fn()
  }

  return (
    <CommandDialog open={open} onOpenChange={setOpen}>
      <CommandInput placeholder="Tapez une commande ou une recherche..." />
      <CommandList>
        <CommandEmpty>Aucun resultat.</CommandEmpty>

        <CommandGroup heading="Navigation">
          <CommandItem onSelect={() => run(() => navigate("/"))}>
            <BarChart3 className="h-4 w-4" />
            <span>Dashboard</span>
          </CommandItem>
          <CommandItem onSelect={() => run(() => navigate("/"))}>
            <Search className="h-4 w-4" />
            <span>Analyser un article</span>
          </CommandItem>
          <CommandItem onSelect={() => run(() => navigate("/"))}>
            <FileText className="h-4 w-4" />
            <span>Articles recents</span>
          </CommandItem>
        </CommandGroup>

        <CommandSeparator />

        <CommandGroup heading="Apparence">
          <CommandItem onSelect={() => run(() => setTheme("dark"))}>
            <Moon className="h-4 w-4" />
            <span>Theme sombre (defaut)</span>
          </CommandItem>
          <CommandItem onSelect={() => run(() => setTheme("light"))}>
            <Sun className="h-4 w-4" />
            <span>Theme clair</span>
          </CommandItem>
        </CommandGroup>

        {isAuthenticated && (
          <>
            <CommandSeparator />
            <CommandGroup heading="Compte">
              <CommandItem onSelect={() => run(() => { logout(); navigate("/login") })}>
                <LogOut className="h-4 w-4" />
                <span>Se deconnecter</span>
              </CommandItem>
            </CommandGroup>
          </>
        )}

        <CommandSeparator />

        <CommandGroup heading="A propos">
          <CommandItem disabled>
            <Leaf className="h-4 w-4" />
            <span>GreenTech Intelligence - Plateforme d'analyse Green IT</span>
          </CommandItem>
        </CommandGroup>
      </CommandList>
    </CommandDialog>
  )
}
