import { motion } from "motion/react"
import { cn } from "@/lib/utils"

interface LeafCircuitLogoProps {
  size?: number
  animated?: boolean
  className?: string
}

/**
 * Logo identite visuelle du projet : une feuille dont les nervures forment
 * des traces de circuit imprime. L'illustration PNG detouree est servie
 * depuis `public/` afin de partager la meme source que le favicon du site.
 *
 * Lorsque `animated` est actif, le logo apparait en fondu avec une legere
 * mise a l'echelle au montage, et reagit au hover du parent `.group` via
 * une transition CSS.
 */
export function LeafCircuitLogo({
  size = 28,
  animated = true,
  className,
}: LeafCircuitLogoProps) {
  const commonProps = {
    src: "/favicon-512.png",
    alt: "",
    width: size,
    height: size,
    draggable: false,
    "aria-hidden": true as const,
    className: cn(
      "shrink-0 select-none object-contain",
      "transition-transform duration-300 group-hover:scale-105",
      className,
    ),
    style: { width: size, height: size },
  }

  if (!animated) {
    return <img {...commonProps} />
  }

  return (
    <motion.img
      {...commonProps}
      initial={{ opacity: 0, scale: 0.85 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
    />
  )
}
