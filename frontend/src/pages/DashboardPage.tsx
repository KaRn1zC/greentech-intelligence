import { useCallback, useEffect, useRef, useState } from "react"
import { Link } from "react-router-dom"
import { toast } from "sonner"
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from "recharts"
import {
  BarChart3,
  ExternalLink,
  FileUp,
  Leaf,
  Loader2,
  Search,
  SendHorizonal,
  X,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Skeleton } from "@/components/ui/skeleton"
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

const ACCEPTED_FILE_EXTENSIONS = ".txt,.md,.pdf,.docx,.html,.htm"
const MAX_FILE_SIZE_MB = 10

const PIE_COLORS = ["#16a34a", "#dc2626", "#a1a1aa"]

export function DashboardPage() {
  return (
    <div className="space-y-8">
      <h1 className="text-2xl font-bold">Dashboard</h1>
      <AnalyzeSection />
      <div className="grid gap-6 md:grid-cols-2">
        <StatsSection />
        <RecentArticlesSection />
      </div>
    </div>
  )
}

// === Section Analyse ===

function AnalyzeSection() {
  const [input, setInput] = useState("")
  const [file, setFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState("")
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)

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

  const handleSubmit = async () => {
    if (!input.trim() && !file) return
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
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Search className="h-5 w-5" />
          Analyser un article
        </CardTitle>
        <CardDescription>
          Collez une URL, du texte, ou deposez un fichier pour obtenir la classification Green IT
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex gap-2">
          <label htmlFor="analyze-input" className="sr-only">
            URL ou texte de l'article a analyser
          </label>
          <Input
            id="analyze-input"
            placeholder="https://... ou collez le texte de l'article (50 car. min)"
            value={input}
            onChange={(e) => {
              setInput(e.target.value)
              if (e.target.value && file) clearFile()
            }}
            onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
            aria-describedby="analyze-help"
            disabled={!!file}
          />
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
            disabled={loading || (!input.trim() && !file)}
            aria-label="Lancer l'analyse"
          >
            {loading ? (
              <Loader2 className="animate-spin" />
            ) : (
              <SendHorizonal className="h-4 w-4" />
            )}
          </Button>
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
          <div className="flex items-center justify-between rounded-md border bg-muted/50 px-3 py-2 text-sm">
            <span className="truncate">
              <FileUp className="mr-2 inline h-4 w-4" />
              {file.name} ({(file.size / 1024).toFixed(1)} Ko)
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

        {loading && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            Analyse en cours...
          </div>
        )}

        {error && <p className="text-sm text-destructive">{error}</p>}

        {result && result.statut === "termine" && (
          <div className="rounded-lg border p-4 space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-medium">{result.titre || "Article analyse"}</span>
              <GreenBadge value={result.est_green_it} />
            </div>
            {result.score_confiance !== null && (
              <p className="text-sm text-muted-foreground">
                Confiance : {(result.score_confiance * 100).toFixed(1)}%
                {result.temps_inference_ms && ` — ${result.temps_inference_ms}ms`}
              </p>
            )}
            {result.resume && (
              <div className="space-y-1">
                <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Resume de l'article
                </p>
                <p className="text-sm">{result.resume}</p>
              </div>
            )}
            {result.resume_ecologique && (
              <div className="space-y-1 rounded-md border border-green-200 bg-green-50 p-3 dark:border-green-900 dark:bg-green-950/30">
                <p className="flex items-center gap-1 text-xs font-semibold uppercase tracking-wide text-green-700 dark:text-green-400">
                  <Leaf className="h-3 w-3" aria-hidden="true" />
                  Aspects ecologiques identifies
                </p>
                <p className="text-sm text-green-900 dark:text-green-100">
                  {result.resume_ecologique}
                </p>
              </div>
            )}
            {result.id_article && (
              <Link
                to={`/articles/${result.id_article}`}
                className="inline-flex items-center gap-1 text-sm text-primary hover:underline"
              >
                Voir le detail <ExternalLink className="h-3 w-3" />
              </Link>
            )}
          </div>
        )}

        {result && result.statut === "erreur" && (
          <p className="text-sm text-destructive">
            Erreur : {result.erreur || "Echec de l'analyse"}
          </p>
        )}
      </CardContent>
    </Card>
  )
}

// === Section Statistiques ===

function StatsSection() {
  const [stats, setStats] = useState<GlobalStats | null>(null)

  useEffect(() => {
    getStats().then(setStats).catch(() => {})
  }, [])

  if (!stats) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" /> Statistiques
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <Skeleton className="h-40 w-full" />
        </CardContent>
      </Card>
    )
  }

  const pieData = [
    { name: "Green IT", value: stats.articles_green_it },
    { name: "Non Green IT", value: stats.articles_non_green_it },
    { name: "En attente", value: stats.en_attente_analyse },
  ]

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <BarChart3 className="h-5 w-5" /> Statistiques
        </CardTitle>
        <CardDescription>{stats.total_articles} articles au total</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex items-center gap-6">
          <ResponsiveContainer width={140} height={140}>
            <PieChart>
              <Pie
                data={pieData}
                dataKey="value"
                cx="50%"
                cy="50%"
                innerRadius={35}
                outerRadius={60}
                paddingAngle={2}
              >
                {pieData.map((_, i) => (
                  <Cell key={i} fill={PIE_COLORS[i]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
          <div className="space-y-2 text-sm">
            <div className="flex items-center gap-2">
              <span className="h-3 w-3 rounded-full bg-green-600" />
              Green IT : {stats.articles_green_it}
            </div>
            <div className="flex items-center gap-2">
              <span className="h-3 w-3 rounded-full bg-red-600" />
              Non Green IT : {stats.articles_non_green_it}
            </div>
            <div className="flex items-center gap-2">
              <span className="h-3 w-3 rounded-full bg-zinc-400" />
              En attente : {stats.en_attente_analyse}
            </div>
            {stats.pourcentage_green_it !== null && (
              <p className="font-medium pt-1">
                {stats.pourcentage_green_it.toFixed(1)}% Green IT
              </p>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

// === Section Articles Recents ===

function RecentArticlesSection() {
  const [articles, setArticles] = useState<ArticleListItem[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getArticles({ limit: 8 })
      .then((data) => setArticles(data.articles))
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Leaf className="h-5 w-5 text-green-600" /> Articles recents
        </CardTitle>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="space-y-3">
            {Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-12 w-full" />
            ))}
          </div>
        ) : articles.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            Aucun article pour l'instant.
          </p>
        ) : (
          <ul className="space-y-3">
            {articles.map((a) => (
              <li key={a.id_article}>
                <Link
                  to={`/articles/${a.id_article}`}
                  className="group flex items-start justify-between gap-2 rounded-md p-2 -mx-2 hover:bg-muted/50 transition-colors"
                >
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium truncate group-hover:text-primary transition-colors">
                      {a.titre}
                    </p>
                    <p className="text-xs text-muted-foreground truncate">
                      {a.nom_source || "Source inconnue"}
                      {a.date_publication &&
                        ` — ${new Date(a.date_publication).toLocaleDateString("fr-FR")}`}
                    </p>
                  </div>
                  <GreenBadge value={a.est_green_it} />
                </Link>
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  )
}

// === Composant Badge Green IT ===

function GreenBadge({ value }: { value: boolean | null }) {
  if (value === null) {
    return <Badge variant="secondary" className="shrink-0">En attente</Badge>
  }
  return value ? (
    <Badge className="shrink-0 bg-green-600 hover:bg-green-700">Green IT</Badge>
  ) : (
    <Badge variant="destructive" className="shrink-0">Non Green IT</Badge>
  )
}
