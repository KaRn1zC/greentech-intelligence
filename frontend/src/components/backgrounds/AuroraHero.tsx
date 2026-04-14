import type { ReactNode } from "react"
import { cn } from "@/lib/utils"

interface AuroraHeroProps {
  children: ReactNode
  className?: string
}

/**
 * Fond "aurora" personnalise aux couleurs Tech/Ecologie du projet.
 *
 * Ce composant est inspire de l'AuroraBackground d'aceternity mais remplace
 * les violets/bleus par notre duo vert emeraude + cyan electrique, en
 * utilisant directement nos tokens OKLCH.
 *
 * L'animation est purement CSS (pas de JS), respectueuse de
 * `prefers-reduced-motion` via le media query standard.
 */
export function AuroraHero({ children, className }: AuroraHeroProps) {
  return (
    <div
      className={cn(
        "relative isolate overflow-hidden",
        "bg-[oklch(0.145_0.015_160)]",
        className,
      )}
    >
      {/* Couche aurora - nuages verts/cyans */}
      <div
        aria-hidden="true"
        className={cn(
          "pointer-events-none absolute -inset-20 opacity-60 blur-3xl",
          "motion-safe:animate-pulse",
        )}
        style={{
          background:
            "radial-gradient(ellipse 70% 50% at 20% 30%, oklch(0.72 0.18 155 / 0.35), transparent 65%), " +
            "radial-gradient(ellipse 60% 50% at 80% 20%, oklch(0.78 0.15 210 / 0.3), transparent 65%), " +
            "radial-gradient(ellipse 80% 60% at 60% 90%, oklch(0.6 0.16 165 / 0.25), transparent 70%)",
        }}
      />

      {/* Couche quadrillage blueprint subtil */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 opacity-[0.08]"
        style={{
          backgroundImage:
            "linear-gradient(oklch(0.72 0.18 155 / 0.5) 1px, transparent 1px), " +
            "linear-gradient(to right, oklch(0.72 0.18 155 / 0.5) 1px, transparent 1px)",
          backgroundSize: "60px 60px",
          maskImage: "radial-gradient(ellipse at center, black 40%, transparent 80%)",
        }}
      />

      {/* Contenu */}
      <div className="relative z-10">{children}</div>
    </div>
  )
}
