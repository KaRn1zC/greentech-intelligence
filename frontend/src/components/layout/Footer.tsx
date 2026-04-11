import { Leaf } from "lucide-react"

export function Footer() {
  return (
    <footer className="border-t py-6 mt-auto">
      <div className="container mx-auto flex max-w-5xl flex-col items-center gap-2 px-4 text-sm text-muted-foreground sm:flex-row sm:justify-between">
        <div className="flex items-center gap-1">
          <Leaf className="h-3.5 w-3.5 text-green-600" />
          <span>GreenTech Intelligence</span>
        </div>
        <p>Plateforme d'analyse Green IT — Projet Chef d'Oeuvre</p>
      </div>
    </footer>
  )
}
