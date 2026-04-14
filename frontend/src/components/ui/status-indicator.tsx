import { cn } from "@/lib/utils"

type StatusLevel = "ok" | "degraded" | "down" | "unknown"

interface StatusIndicatorProps {
  status: StatusLevel
  label?: string
  className?: string
}

const levelStyles: Record<StatusLevel, { dot: string; text: string; aria: string }> = {
  ok: {
    dot: "bg-[oklch(0.72_0.18_155)] shadow-[0_0_8px_oklch(0.72_0.18_155_/_0.7)]",
    text: "text-[oklch(0.82_0.14_150)]",
    aria: "Systeme operationnel",
  },
  degraded: {
    dot: "bg-[oklch(0.78_0.16_75)] shadow-[0_0_8px_oklch(0.78_0.16_75_/_0.6)]",
    text: "text-[oklch(0.82_0.12_80)]",
    aria: "Systeme degrade",
  },
  down: {
    dot: "bg-[oklch(0.62_0.22_25)] shadow-[0_0_8px_oklch(0.62_0.22_25_/_0.6)]",
    text: "text-[oklch(0.78_0.2_25)]",
    aria: "Systeme indisponible",
  },
  unknown: {
    dot: "bg-muted-foreground",
    text: "text-muted-foreground",
    aria: "Statut inconnu",
  },
}

/**
 * LED pulsante de statut, typiquement branchee sur le endpoint `/health`.
 *
 * La pulsation ne se declenche que pour le statut "ok" afin d'eviter de
 * rendre les alertes plus discretes qu'elles ne devraient l'etre.
 */
export function StatusIndicator({ status, label, className }: StatusIndicatorProps) {
  const { dot, text, aria } = levelStyles[status]
  return (
    <span
      className={cn("inline-flex items-center gap-2 text-xs font-mono", text, className)}
      role="status"
      aria-label={label ?? aria}
    >
      <span className="relative flex h-2.5 w-2.5">
        {status === "ok" && (
          <span
            className={cn(
              "absolute inline-flex h-full w-full rounded-full opacity-50 animate-ping",
              dot,
            )}
            aria-hidden="true"
          />
        )}
        <span className={cn("relative inline-flex h-2.5 w-2.5 rounded-full", dot)} aria-hidden="true" />
      </span>
      {label && <span>{label}</span>}
    </span>
  )
}
