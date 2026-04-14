import type { ReactNode } from "react"
import { cn } from "@/lib/utils"

type GlowVariant = "green" | "cyan" | "warning" | "muted"

interface GlowBadgeProps {
  variant?: GlowVariant
  pulse?: boolean
  icon?: ReactNode
  children: ReactNode
  className?: string
}

const variantStyles: Record<GlowVariant, string> = {
  green:
    "border-[oklch(0.72_0.18_155_/_0.35)] bg-[oklch(0.72_0.18_155_/_0.12)] text-[oklch(0.88_0.12_150)] glow-green-sm",
  cyan:
    "border-[oklch(0.78_0.15_210_/_0.35)] bg-[oklch(0.78_0.15_210_/_0.12)] text-[oklch(0.88_0.1_210)] glow-cyan-sm",
  warning:
    "border-[oklch(0.78_0.16_75_/_0.4)] bg-[oklch(0.78_0.16_75_/_0.12)] text-[oklch(0.88_0.12_80)]",
  muted:
    "border-border bg-muted/60 text-muted-foreground",
}

/**
 * Badge avec un halo lumineux (glow) colore, adapte a l'identite visuelle
 * du projet : vert pour Green IT confirme, cyan pour accent tech, ambre
 * pour les articles en attente ou incertains.
 *
 * Le parametre `pulse` ajoute une pulsation lente pour les badges "en cours".
 */
export function GlowBadge({
  variant = "green",
  pulse = false,
  icon,
  children,
  className,
}: GlowBadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex shrink-0 items-center gap-1.5 rounded-full border px-2.5 py-0.5",
        "font-mono text-xs font-medium uppercase tracking-wider",
        "transition-colors",
        variantStyles[variant],
        pulse && variant === "green" && "animate-pulse-glow",
        className,
      )}
    >
      {icon && <span aria-hidden="true" className="shrink-0">{icon}</span>}
      {children}
    </span>
  )
}
