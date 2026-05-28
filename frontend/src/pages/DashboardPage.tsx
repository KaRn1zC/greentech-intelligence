import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Link } from "react-router-dom"
import { motion } from "motion/react"
import { toast } from "sonner"
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from "recharts"
import {
  Activity,
  BarChart3,
  ExternalLink,
  FileUp,
  Gauge,
  Leaf,
  Loader2,
  Newspaper,
  Search,
  SendHorizonal,
  Sparkles,
  X,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Skeleton } from "@/components/ui/skeleton"
import emptyStateIllustration from "@/assets/illustrations/empty-state-no-articles.png"
import classificationLoadingIllustration from "@/assets/illustrations/classification-loading.png"
import { GlowBadge } from "@/components/ui/glow-badge"
import { MetricCard } from "@/components/ui/metric-card"
import { BorderBeam } from "@/components/ui/border-beam"
import { TypingAnimation } from "@/components/ui/typing-animation"
import { AnimatedGridPattern } from "@/components/ui/animated-grid-pattern"
import {
  getArticles,
  getStats,
  submitAnalysis,
  submitAnalysisFile,
  getAnalysisStatus,
} from "@/lib/api"
import type {
  ArticleListItem,
  AnalysisResult,
  GlobalStats,
} from "@/types/api"
import { cn } from "@/lib/utils"

const ACCEPTED_FILE_EXTENSIONS = ".txt,.md,.pdf,.docx,.html,.htm"
const MAX_FILE_SIZE_MB = 10

const PIE_COLORS = [
  "oklch(0.72 0.18 155)",
  "oklch(0.62 0.22 25)",
  "oklch(0.5 0.02 170)",
]

export function DashboardPage() {
  // Compteur incremente par AnalyzeSection a la fin de chaque analyse reussie.
  // Les sections qui lisent des donnees cote serveur (StatsGrid,
  // ClassificationBreakdown, RecentArticlesSection) l'utilisent comme cle de
  // dependance d'effet pour re-fetcher automatiquement et refleter
  // l'apparition du nouvel article sans rechargement manuel de la page.
  const [refreshToken, setRefreshToken] = useState(0)
  const notifyAnalysisComplete = useCallback(() => {
    setRefreshToken((token) => token + 1)
  }, [])

  return (
    <div className="space-y-8">
      <DashboardHeader />
      <StatsGrid refreshToken={refreshToken} />
      <AnalyzeSection onAnalysisComplete={notifyAnalysisComplete} />
      <div className="grid gap-6 lg:grid-cols-5">
        <ClassificationBreakdown className="lg:col-span-2" refreshToken={refreshToken} />
        <RecentArticlesSection className="lg:col-span-3" refreshToken={refreshToken} />
      </div>
    </div>
  )
}

// ============================================================================
// En-tete Dashboard avec typing animation terminal
// ============================================================================

function DashboardHeader() {
  return (
    <div className="relative overflow-hidden rounded-2xl border border-border/60 bg-card/40 p-6">
      <AnimatedGridPattern
        numSquares={30}
        maxOpacity={0.08}
        duration={3}
        className={cn(
          "inset-0 h-full w-full",
          "[mask-image:radial-gradient(500px_circle_at_center,white,transparent)]",
        )}
      />
      <div className="relative z-10 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.2em] text-muted-foreground">
            <span className="text-[oklch(0.72_0.18_155)]">&gt;</span> dashboard
          </p>
          <h1 className="mt-1 font-display text-3xl font-semibold tracking-tight">
            Intelligence <span className="text-gradient-eco">Green IT</span>
          </h1>
          <div className="mt-2 font-mono text-sm text-muted-foreground">
            <TypingAnimation duration={40} className="font-mono text-sm text-muted-foreground">
              Scanning sustainability signals across the tech landscape...
            </TypingAnimation>
          </div>
        </div>
        <GlowBadge variant="cyan" icon={<Activity className="h-3 w-3" />}>
          live
        </GlowBadge>
      </div>
    </div>
  )
}

// ============================================================================
// Grille de metriques principales (4 cards)
// ============================================================================

function StatsGrid({ refreshToken }: { refreshToken: number }) {
  const [stats, setStats] = useState<GlobalStats | null>(null)
  const [loading, setLoading] = useState(true)

  // Re-fetch a chaque incrementation de `refreshToken` (analyse terminee).
  // On n'affiche plus le Skeleton sur les refresh silencieux pour eviter le
  // flash visuel : le premier chargement est le seul cas ou `loading` reste
  // vrai jusqu'au succes, ensuite on met a jour `stats` en place.
  useEffect(() => {
    getStats()
      .then(setStats)
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [refreshToken])

  if (loading || !stats) {
    return (
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <Skeleton key={i} className="h-28 w-full rounded-xl" />
        ))}
      </div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4"
    >
      <MetricCard
        label="Articles analyses"
        value={stats.total_articles}
        icon={<Newspaper className="h-4 w-4" />}
        accent="cyan"
      />
      <MetricCard
        label="Green IT confirmes"
        value={stats.articles_green_it}
        icon={<Leaf className="h-4 w-4" />}
        accent="green"
      />
      <MetricCard
        label="Non Green IT"
        value={stats.articles_non_green_it}
        icon={<X className="h-4 w-4" />}
        accent="warning"
      />
      <MetricCard
        label="Ratio Green IT"
        value={stats.pourcentage_green_it ?? 0}
        suffix="%"
        decimals={1}
        icon={<Gauge className="h-4 w-4" />}
        accent="green"
      />
    </motion.div>
  )
}

// ============================================================================
// Section Analyse d'article (URL / texte / fichier)
// ============================================================================

function AnalyzeSection({ onAnalysisComplete }: { onAnalysisComplete: () => void }) {
  const [input, setInput] = useState("")
  const [file, setFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState("")
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)

  // Auto-grow de la zone de texte : la hauteur suit le contenu sans depasser
  // ~12 lignes (cap a 360 px), au-dela un scroll interne prend le relais.
  // Indispensable pour rendre les copier-coller d'articles longs lisibles
  // sans forcer un faux composant single-line.
  useEffect(() => {
    const node = textareaRef.current
    if (!node) return
    node.style.height = "auto"
    node.style.height = `${Math.min(node.scrollHeight, 360)}px`
  }, [input])

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [])

  useEffect(() => () => stopPolling(), [stopPolling])

  const startPolling = (jobId: string) => {
    pollRef.current = setInterval(async () => {
      try {
        const status = await getAnalysisStatus(jobId)
        if (status.statut === "termine") {
          stopPolling()
          setResult(status)
          setLoading(false)
          toast.success(
            status.est_green_it ? "Article classe Green IT" : "Article classe Non Green IT",
          )
          // Notifie le parent pour que les sections Stats / Repartition /
          // Derniers articles rafraichissent leur contenu et refletent
          // immediatement la presence du nouvel article classifie.
          onAnalysisComplete()
        } else if (status.statut === "erreur") {
          stopPolling()
          setResult(status)
          setLoading(false)
          toast.error(status.erreur || "Echec de l'analyse")
        }
      } catch {
        stopPolling()
        setError("Erreur lors du suivi de l'analyse.")
        setLoading(false)
        toast.error("Erreur lors du suivi de l'analyse")
      }
    }, 2000)
  }

  // Validation alignee sur les regles du backend (50 car. min. pour le texte
  // brut, sinon URL ou fichier). On laisse le bouton accessible aux lecteurs
  // d'ecran via aria-disabled, mais on bloque la soumission tant que la
  // saisie est invalide pour eviter le 422 et son toast inutilement bruyant.
  const trimmedInput = input.trim()
  const isUrl = /^https?:\/\//i.test(trimmedInput)
  const inputValid = !!file || isUrl || trimmedInput.length >= 50

  const handleSubmit = async () => {
    if (!inputValid) return
    setLoading(true)
    setError("")
    setResult(null)
    stopPolling()

    try {
      let job
      if (file) {
        job = await submitAnalysisFile(file)
      } else {
        const isUrl = input.startsWith("http://") || input.startsWith("https://")
        job = await submitAnalysis(isUrl ? { url: input } : { texte: input })
      }
      startPolling(job.job_id)
    } catch (err) {
      const message = err instanceof Error ? err.message : "Impossible de soumettre l'analyse."
      setError(message)
      setLoading(false)
      toast.error(message)
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0] ?? null
    if (!selected) {
      setFile(null)
      return
    }
    if (selected.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
      toast.error(`Fichier trop volumineux (max ${MAX_FILE_SIZE_MB} Mo)`)
      e.target.value = ""
      return
    }
    setFile(selected)
    setInput("")
  }

  const clearFile = () => {
    setFile(null)
    if (fileInputRef.current) fileInputRef.current.value = ""
  }

  return (
    <section
      className={cn(
        "relative overflow-hidden rounded-2xl border border-border/60 bg-card p-6",
        "transition-colors",
      )}
    >
      {loading && <BorderBeam size={180} duration={8} colorFrom="oklch(0.72 0.18 155)" colorTo="oklch(0.78 0.15 210)" />}

      <header className="mb-4 flex items-start justify-between gap-3">
        <div>
          <h2 className="flex items-center gap-2 font-display text-lg font-semibold">
            <Search className="h-4 w-4 text-[oklch(0.78_0.15_210)]" />
            Analyser un article
          </h2>
          <p className="mt-0.5 text-sm text-muted-foreground">
            Collez une URL, du texte (50 car. min.) ou deposez un fichier.
          </p>
        </div>
        {loading && (
          <GlowBadge variant="cyan" pulse icon={<Sparkles className="h-3 w-3" />}>
            classification en cours
          </GlowBadge>
        )}
      </header>

      <div className="flex flex-col gap-2">
        <label htmlFor="analyze-input" className="sr-only">
          URL ou texte de l'article a analyser
        </label>
        <Textarea
          ref={textareaRef}
          id="analyze-input"
          placeholder="https://... ou collez ici le texte complet de l'article (50 car. min.)"
          value={input}
          onChange={(e) => {
            setInput(e.target.value)
            if (e.target.value && file) clearFile()
          }}
          onKeyDown={(e) => {
            // Cmd/Ctrl + Entree envoie le formulaire ; Entree seul laisse passer
            // pour permettre les sauts de ligne dans un texte colle.
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
              e.preventDefault()
              handleSubmit()
            }
          }}
          aria-describedby="analyze-help"
          disabled={!!file}
          rows={3}
          className="min-h-[88px] max-h-[360px] resize-none overflow-y-auto"
        />
        <div className="flex items-center justify-between gap-2">
          <span className="hidden text-xs text-muted-foreground sm:inline">
            {input.length > 0 && (
              <>
                {input.length} caractere{input.length > 1 ? "s" : ""}
                {!isUrl && input.length > 0 && input.length < 50 && (
                  <span className="ml-2 text-amber-600">
                    (minimum 50 pour un texte)
                  </span>
                )}
                <span className="ml-2 hidden md:inline">
                  &middot; Ctrl/Cmd + Entree pour analyser
                </span>
              </>
            )}
          </span>
          <div className="ml-auto flex gap-2">
            <Button
              type="button"
              variant="outline"
              onClick={() => fileInputRef.current?.click()}
              disabled={loading}
              aria-label="Choisir un fichier a analyser"
            >
              <FileUp className="h-4 w-4" />
            </Button>
            <Button
              onClick={handleSubmit}
              disabled={loading || !inputValid}
              aria-label="Lancer l'analyse"
              className="font-display font-medium"
            >
              {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <SendHorizonal className="h-4 w-4" />}
            </Button>
          </div>
        </div>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept={ACCEPTED_FILE_EXTENSIONS}
        onChange={handleFileChange}
        className="hidden"
        aria-label="Fichier a analyser"
      />
      <p id="analyze-help" className="sr-only">
        Saisissez une URL commencant par http, du texte d'au moins 50 caracteres,
        ou deposez un fichier (.txt, .md, .pdf, .docx, .html)
      </p>

      {file && (
        <div className="mt-3 flex items-center justify-between rounded-md border border-border/60 bg-muted/40 px-3 py-2 text-sm">
          <span className="truncate font-mono">
            <FileUp className="mr-2 inline h-4 w-4 text-[oklch(0.78_0.15_210)]" />
            {file.name} <span className="text-muted-foreground">({(file.size / 1024).toFixed(1)} Ko)</span>
          </span>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={clearFile}
            aria-label="Retirer le fichier"
            disabled={loading}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      )}

      {error && (
        <p
          className="mt-3 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive"
          role="alert"
        >
          {error}
        </p>
      )}

      {loading && !result && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="mt-6 flex flex-col items-center gap-3 py-4"
        >
          <img
            src={classificationLoadingIllustration}
            alt=""
            className="h-40 w-auto select-none opacity-90"
            aria-hidden="true"
            draggable={false}
          />
          <p className="font-mono text-xs text-muted-foreground">
            <span className="text-[oklch(0.78_0.15_210)]">&gt;</span>{" "}
            classification en cours...
          </p>
        </motion.div>
      )}

      {result && result.statut === "termine" && <AnalysisResultCard result={result} />}

      {result && result.statut === "erreur" && (
        <p className="mt-4 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          Erreur : {result.erreur || "Echec de l'analyse"}
        </p>
      )}
    </section>
  )
}

function AnalysisResultCard({ result }: { result: AnalysisResult }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="mt-4 space-y-3 rounded-xl border border-border/60 bg-background/40 p-4"
    >
      <div className="flex items-center justify-between gap-3">
        <span className="truncate font-display font-medium">
          {result.titre || "Article analyse"}
        </span>
        {result.est_green_it !== null && (
          result.est_green_it ? (
            <GlowBadge variant="green" icon={<Leaf className="h-3 w-3" />}>Green IT</GlowBadge>
          ) : (
            <GlowBadge variant="warning">Non Green IT</GlowBadge>
          )
        )}
      </div>

      {result.score_confiance !== null && (
        <p className="font-mono text-xs text-muted-foreground">
          confiance : <span className="text-foreground tabular-nums">{(result.score_confiance * 100).toFixed(1)}%</span>
          {result.temps_inference_ms && (
            <> &bull; <span className="tabular-nums">{result.temps_inference_ms}ms</span></>
          )}
        </p>
      )}

      {result.resume && (
        <div className="space-y-1">
          <p className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
            resume
          </p>
          <p className="text-sm leading-relaxed">{result.resume}</p>
        </div>
      )}

      {result.resume_ecologique && (
        <div
          className="space-y-1 rounded-lg border border-[oklch(0.72_0.18_155_/_0.3)] bg-[oklch(0.72_0.18_155_/_0.05)] p-3"
        >
          <p className="flex items-center gap-1.5 font-mono text-xs uppercase tracking-wider text-[oklch(0.82_0.14_150)]">
            <Leaf className="h-3 w-3" aria-hidden="true" />
            aspects ecologiques
          </p>
          <p className="text-sm leading-relaxed text-foreground">
            {result.resume_ecologique}
          </p>
        </div>
      )}

      {result.id_article && (
        <Link
          to={`/articles/${result.id_article}`}
          className="inline-flex items-center gap-1 text-sm text-[oklch(0.82_0.14_150)] underline-offset-4 transition-colors hover:text-[oklch(0.88_0.12_150)] hover:underline"
        >
          Voir le detail <ExternalLink className="h-3 w-3" />
        </Link>
      )}
    </motion.div>
  )
}

// ============================================================================
// Repartition de classification (camembert)
// ============================================================================

function ClassificationBreakdown({
  className,
  refreshToken,
}: {
  className?: string
  refreshToken: number
}) {
  const [stats, setStats] = useState<GlobalStats | null>(null)

  useEffect(() => {
    getStats().then(setStats).catch(() => {})
  }, [refreshToken])

  const pieData = useMemo(() => {
    if (!stats) return []
    return [
      { name: "Green IT", value: stats.articles_green_it },
      { name: "Non Green IT", value: stats.articles_non_green_it },
      { name: "En attente", value: stats.en_attente_analyse },
    ]
  }, [stats])

  return (
    <section className={cn("overflow-hidden rounded-2xl border border-border/60 bg-card p-6", className)}>
      <header className="mb-4 flex items-center justify-between">
        <h2 className="flex items-center gap-2 font-display text-lg font-semibold">
          <BarChart3 className="h-4 w-4 text-[oklch(0.72_0.18_155)]" />
          Repartition
        </h2>
        {stats && (
          <p className="font-mono text-xs text-muted-foreground">
            {stats.total_articles} articles
          </p>
        )}
      </header>

      {!stats ? (
        <Skeleton className="h-40 w-full" />
      ) : (
        <div className="flex items-center gap-6">
          <ResponsiveContainer width={160} height={160}>
            <PieChart>
              <Pie
                data={pieData}
                dataKey="value"
                cx="50%"
                cy="50%"
                innerRadius={42}
                outerRadius={72}
                paddingAngle={3}
                stroke="oklch(0.145 0.015 160)"
                strokeWidth={2}
              >
                {pieData.map((_, i) => (
                  <Cell key={i} fill={PIE_COLORS[i]} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  backgroundColor: "oklch(0.205 0.02 170)",
                  border: "1px solid oklch(0.28 0.015 170)",
                  borderRadius: "0.5rem",
                  fontSize: "0.75rem",
                }}
                labelStyle={{ color: "oklch(0.95 0.01 150)" }}
              />
            </PieChart>
          </ResponsiveContainer>
          <div className="space-y-2.5 font-mono text-xs">
            <LegendRow color="oklch(0.72 0.18 155)" label="Green IT" value={stats.articles_green_it} />
            <LegendRow color="oklch(0.62 0.22 25)" label="Non Green IT" value={stats.articles_non_green_it} />
            <LegendRow color="oklch(0.5 0.02 170)" label="En attente" value={stats.en_attente_analyse} />
            {stats.pourcentage_green_it !== null && (
              <p className="pt-2 font-display text-base font-semibold text-[oklch(0.82_0.14_150)] tabular-nums">
                {stats.pourcentage_green_it.toFixed(1)}% <span className="text-xs text-muted-foreground">Green IT</span>
              </p>
            )}
          </div>
        </div>
      )}
    </section>
  )
}

function LegendRow({ color, label, value }: { color: string; label: string; value: number }) {
  return (
    <div className="flex items-center gap-2">
      <span
        className="h-2.5 w-2.5 rounded-full"
        style={{ backgroundColor: color, boxShadow: `0 0 8px ${color}` }}
        aria-hidden="true"
      />
      <span className="text-muted-foreground">{label}</span>
      <span className="ml-auto tabular-nums text-foreground">{value}</span>
    </div>
  )
}

// ============================================================================
// Articles recents
// ============================================================================

function RecentArticlesSection({
  className,
  refreshToken,
}: {
  className?: string
  refreshToken: number
}) {
  const [articles, setArticles] = useState<ArticleListItem[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getArticles({ limit: 8 })
      .then((data) => setArticles(data.articles))
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [refreshToken])

  return (
    <section className={cn("overflow-hidden rounded-2xl border border-border/60 bg-card p-6", className)}>
      <header className="mb-4 flex items-center justify-between">
        <h2 className="flex items-center gap-2 font-display text-lg font-semibold">
          <Newspaper className="h-4 w-4 text-[oklch(0.78_0.15_210)]" />
          Derniers articles
        </h2>
        {articles.length > 0 && (
          <p className="font-mono text-xs text-muted-foreground">
            {articles.length} resultats
          </p>
        )}
      </header>

      {loading ? (
        <div className="space-y-3">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-14 w-full" />
          ))}
        </div>
      ) : articles.length === 0 ? (
        <div className="flex flex-col items-center justify-center gap-3 py-8">
          <img
            src={emptyStateIllustration}
            alt=""
            className="h-40 w-auto select-none opacity-90"
            aria-hidden="true"
            draggable={false}
          />
          <p className="font-mono text-xs text-muted-foreground">
            <span className="text-[oklch(0.72_0.18_155)]">&gt;</span>{" "}
            aucun article collecté pour le moment
          </p>
        </div>
      ) : (
        <ul className="space-y-1">
          {articles.map((a, i) => (
            <motion.li
              key={a.id_article}
              initial={{ opacity: 0, x: -6 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.25, delay: i * 0.03 }}
            >
              <Link
                to={`/articles/${a.id_article}`}
                className={cn(
                  "group flex items-start justify-between gap-3 rounded-lg p-3",
                  "transition-colors hover:bg-muted/40",
                )}
              >
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium transition-colors group-hover:text-[oklch(0.82_0.14_150)]">
                    {a.titre}
                  </p>
                  <p className="mt-0.5 truncate font-mono text-xs text-muted-foreground">
                    {a.nom_source || "source inconnue"}
                    {a.date_publication &&
                      ` - ${new Date(a.date_publication).toLocaleDateString("fr-FR")}`}
                  </p>
                </div>
                <ArticleBadge value={a.est_green_it} />
              </Link>
            </motion.li>
          ))}
        </ul>
      )}
    </section>
  )
}

function ArticleBadge({ value }: { value: boolean | null }) {
  if (value === null) {
    return <GlowBadge variant="muted">en attente</GlowBadge>
  }
  return value ? (
    <GlowBadge variant="green" icon={<Leaf className="h-3 w-3" />}>Green IT</GlowBadge>
  ) : (
    <GlowBadge variant="warning">Non Green IT</GlowBadge>
  )
}
