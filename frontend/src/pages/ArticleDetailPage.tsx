import { useEffect, useState } from "react"
import { useParams, Link } from "react-router-dom"
import { motion } from "motion/react"
import {
  ArrowLeft,
  Calendar,
  Clock,
  Cpu,
  ExternalLink,
  Globe,
  Leaf,
  User,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { GlowBadge } from "@/components/ui/glow-badge"
import { getArticle } from "@/lib/api"
import type { ArticleDetail } from "@/types/api"
import { cn } from "@/lib/utils"

export function ArticleDetailPage() {
  const { id } = useParams<{ id: string }>()
  const [article, setArticle] = useState<ArticleDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")

  useEffect(() => {
    if (!id) return
    getArticle(Number(id))
      .then(setArticle)
      .catch(() => setError("Article introuvable."))
      .finally(() => setLoading(false))
  }, [id])

  if (loading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-12 w-3/4" />
        <Skeleton className="h-64 w-full" />
      </div>
    )
  }

  if (error || !article) {
    return (
      <div className="space-y-4">
        <Link to="/">
          <Button variant="ghost" size="sm">
            <ArrowLeft className="mr-1 h-4 w-4" /> Retour
          </Button>
        </Link>
        <p className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          {error || "Article introuvable."}
        </p>
      </div>
    )
  }

  const formatDate = (d: string | null) =>
    d ? new Date(d).toLocaleDateString("fr-FR", {
      day: "numeric",
      month: "long",
      year: "numeric",
    }) : null

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
      className="space-y-6"
    >
      <Link to="/">
        <Button variant="ghost" size="sm" className="group -ml-2 font-mono text-xs uppercase tracking-wider text-muted-foreground">
          <ArrowLeft className="mr-1 h-3.5 w-3.5 transition-transform group-hover:-translate-x-0.5" />
          retour au dashboard
        </Button>
      </Link>

      {/* En-tete */}
      <header className="space-y-4">
        <div className="flex flex-wrap items-center gap-2">
          {article.est_green_it !== null && (
            article.est_green_it ? (
              <GlowBadge variant="green" pulse icon={<Leaf className="h-3 w-3" />}>Green IT confirme</GlowBadge>
            ) : (
              <GlowBadge variant="warning">Non Green IT</GlowBadge>
            )
          )}
          {article.nom_source && (
            <GlowBadge variant="muted">{article.nom_source}</GlowBadge>
          )}
          {article.langue && (
            <GlowBadge variant="cyan" icon={<Globe className="h-3 w-3" />}>{article.langue.toUpperCase()}</GlowBadge>
          )}
        </div>

        <h1 className="font-display text-3xl font-semibold leading-tight tracking-tight sm:text-4xl">
          {article.titre}
        </h1>

        <div className="flex flex-wrap gap-4 font-mono text-xs text-muted-foreground">
          {article.auteur && (
            <span className="flex items-center gap-1.5">
              <User className="h-3.5 w-3.5" /> {article.auteur}
            </span>
          )}
          {article.date_publication && (
            <span className="flex items-center gap-1.5">
              <Calendar className="h-3.5 w-3.5" /> {formatDate(article.date_publication)}
            </span>
          )}
          {article.url && (
            <a
              href={article.url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 transition-colors hover:text-[oklch(0.82_0.14_150)]"
            >
              <ExternalLink className="h-3.5 w-3.5" /> Source originale
            </a>
          )}
        </div>
      </header>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="space-y-6 lg:col-span-2">
          {article.resume && (
            <ContentCard title="Resume">
              <p className="text-sm leading-relaxed">{article.resume}</p>
            </ContentCard>
          )}

          {article.resume_ecologique && (
            <section
              className={cn(
                "relative overflow-hidden rounded-2xl border p-6",
                "border-[oklch(0.72_0.18_155_/_0.35)] bg-[oklch(0.72_0.18_155_/_0.05)]",
              )}
            >
              <div
                className="pointer-events-none absolute inset-y-6 left-0 w-[3px] rounded-r-full bg-[oklch(0.72_0.18_155)]"
                aria-hidden="true"
              />
              <header className="mb-3 flex items-center gap-2">
                <Leaf className="h-4 w-4 text-[oklch(0.82_0.14_150)]" aria-hidden="true" />
                <h2 className="font-mono text-xs uppercase tracking-wider text-[oklch(0.82_0.14_150)]">
                  aspects ecologiques identifies
                </h2>
              </header>
              <p className="text-sm leading-relaxed">
                {article.resume_ecologique}
              </p>
            </section>
          )}

          {article.contenu && (
            <ContentCard title="Contenu de l'article">
              <p className="whitespace-pre-line text-sm leading-relaxed text-muted-foreground">
                {article.contenu.length > 3000
                  ? `${article.contenu.slice(0, 3000)}...`
                  : article.contenu}
              </p>
            </ContentCard>
          )}
        </div>

        <aside className="space-y-4 lg:col-span-1">
          <section className="rounded-2xl border border-border/60 bg-card p-6">
            <header className="mb-4">
              <h2 className="font-display text-base font-semibold">Classification</h2>
              <p className="font-mono text-xs text-muted-foreground">resultats du modele</p>
            </header>

            <dl className="space-y-4 text-sm">
              {article.score_confiance !== null && (
                <div>
                  <dt className="mb-1.5 font-mono text-xs uppercase tracking-wider text-muted-foreground">
                    confiance
                  </dt>
                  <dd className="flex items-center gap-2">
                    <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
                      <motion.div
                        className="h-full rounded-full bg-gradient-to-r from-[oklch(0.72_0.18_155)] to-[oklch(0.78_0.15_210)]"
                        initial={{ width: 0 }}
                        animate={{ width: `${article.score_confiance * 100}%` }}
                        transition={{ duration: 0.8, ease: "easeOut" }}
                      />
                    </div>
                    <span className="font-mono text-sm font-medium tabular-nums">
                      {(article.score_confiance * 100).toFixed(1)}%
                    </span>
                  </dd>
                </div>
              )}

              {article.modele_classification && (
                <div>
                  <dt className="mb-1 font-mono text-xs uppercase tracking-wider text-muted-foreground">
                    modele
                  </dt>
                  <dd className="flex items-center gap-1.5 font-mono text-xs">
                    <Cpu className="h-3.5 w-3.5 text-[oklch(0.78_0.15_210)]" />
                    {article.modele_classification}
                  </dd>
                </div>
              )}

              {article.date_analyse && (
                <div>
                  <dt className="mb-1 font-mono text-xs uppercase tracking-wider text-muted-foreground">
                    analyse
                  </dt>
                  <dd className="flex items-center gap-1.5 font-mono text-xs">
                    <Clock className="h-3.5 w-3.5" />
                    {formatDate(article.date_analyse)}
                  </dd>
                </div>
              )}
            </dl>
          </section>
        </aside>
      </div>
    </motion.div>
  )
}

function ContentCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="rounded-2xl border border-border/60 bg-card p-6">
      <header className="mb-3">
        <h2 className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
          {title}
        </h2>
      </header>
      {children}
    </section>
  )
}
