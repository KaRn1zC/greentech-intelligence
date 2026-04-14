import { LeafCircuitLogo } from "@/components/layout/LeafCircuitLogo"

export function Footer() {
  return (
    <footer className="border-t border-border/60 py-6 mt-auto">
      <div className="container mx-auto flex max-w-6xl flex-col items-center gap-3 px-4 text-sm sm:flex-row sm:justify-between">
        <div className="flex items-center gap-2 text-muted-foreground">
          <LeafCircuitLogo size={18} animated={false} />
          <span className="font-display font-medium">GreenTech Intelligence</span>
          <span className="font-mono text-xs opacity-70">v1.0.0</span>
        </div>
        <p className="font-mono text-xs text-muted-foreground">
          <span className="text-[oklch(0.72_0.18_155)]">&gt;</span>{" "}
          plateforme d'analyse Green IT - projet chef d'oeuvre - par KaRn1zC
        </p>
      </div>
    </footer>
  )
}
