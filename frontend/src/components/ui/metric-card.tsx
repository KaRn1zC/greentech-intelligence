import type { ReactNode } from "react"
import { NumberTicker } from "@/components/ui/number-ticker"
import { cn } from "@/lib/utils"

interface MetricCardProps {
  label: string
  value: number
  suffix?: string
  decimals?: number
  icon?: ReactNode
  trend?: {
    value: number
    direction: "up" | "down" | "flat"
  }
  accent?: "green" | "cyan" | "warning" | "muted"
  className?: string
}

const accentRing: Record<NonNullable<MetricCardProps["accent"]>, string> = {
  green: "before:bg-[oklch(0.72_0.18_155)]",
  cyan: "before:bg-[oklch(0.78_0.15_210)]",
  warning: "before:bg-[oklch(0.78_0.16_75)]",
  muted: "before:bg-muted-foreground",
}

const trendColor = {
  up: "text-[oklch(0.82_0.18_140)]",
  down: "text-[oklch(0.78_0.2_25)]",
  flat: "text-muted-foreground",
} as const

/**
 * Carte statistique pour le Dashboard, avec un chiffre anime via NumberTicker.
 *
 * Une barre verticale d'accent sur le bord gauche renforce l'identite tech
 * et permet de grouper visuellement plusieurs metriques par categorie.
 */
export function MetricCard({
  label,
  value,
  suffix,
  decimals = 0,
  icon,
  trend,
  accent = "green",
  className,
}: MetricCardProps) {
  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-xl border bg-card p-5",
        "before:absolute before:inset-y-4 before:left-0 before:w-[3px] before:rounded-r-full",
        "transition-all hover:border-border hover:bg-card/80",
        accentRing[accent],
        className,
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <p className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
            {label}
          </p>
          <p className="mt-2 font-display text-3xl font-semibold tracking-tight text-foreground tabular-nums">
            <NumberTicker value={value} decimalPlaces={decimals} />
            {suffix && <span className="ml-1 text-xl text-muted-foreground">{suffix}</span>}
          </p>
          {trend && (
            <p className={cn("mt-1 font-mono text-xs tabular-nums", trendColor[trend.direction])}>
              {trend.direction === "up" && "+"}
              {trend.direction === "down" && "-"}
              {Math.abs(trend.value).toFixed(1)}% <span className="text-muted-foreground">vs 7j</span>
            </p>
          )}
        </div>
        {icon && (
          <div className="rounded-lg bg-muted/40 p-2 text-muted-foreground" aria-hidden="true">
            {icon}
          </div>
        )}
      </div>
    </div>
  )
}
