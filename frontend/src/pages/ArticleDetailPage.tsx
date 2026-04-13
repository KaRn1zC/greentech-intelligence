import { useEffect, useState } from "react"
import { useParams, Link } from "react-router-dom"
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
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { getArticle } from "@/lib/api"
import type { ArticleDetail } from "@/types/api"

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
        <Skeleton className="h-64 w-full" />
      </div>
    )
  }

  if (error || !article) {
    return (
      <div className="space-y-4">
        <Link to="/">
          <Button variant="ghost" size="sm">
            <ArrowLeft className="h-4 w-4 mr-1" /> Retour
          </Button>
        </Link>
        <p className="text-destructive">{error || "Article introuvable."}</p>
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
    <div className="space-y-6">
      <Link to="/">
        <Button variant="ghost" size="sm">
          <ArrowLeft className="h-4 w-4 mr-1" /> Retour au dashboard
        </Button>
      </Link>

      {/* En-tete */}
      <div className="space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          {article.est_green_it !== null && (
            article.est_green_it ? (
              <Badge className="bg-green-600 hover:bg-green-700">
                <Leaf className="mr-1 h-3 w-3" /> Green IT
              </Badge>
            ) : (
              <Badge variant="destructive">Non Green IT</Badge>
            )
          )}
          {article.nom_source && (
            <Badge variant="outline">{article.nom_source}</Badge>
          )}
          {article.langue && (
            <Badge variant="secondary">
              <Globe className="mr-1 h-3 w-3" />{article.langue.toUpperCase()}
            </Badge>
          )}
        </div>
        <h1 className="text-2xl font-bold leading-tight">{article.titre}</h1>
        <div className="flex flex-wrap gap-4 text-sm text-muted-foreground">
          {article.auteur && (
            <span className="flex items-center gap-1">
              <User className="h-3.5 w-3.5" /> {article.auteur}
            </span>
          )}
          {article.date_publication && (
            <span className="flex items-center gap-1">
              <Calendar className="h-3.5 w-3.5" /> {formatDate(article.date_publication)}
            </span>
          )}
          {article.url && (
            <a
              href={article.url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 hover:text-primary transition-colors"
            >
              <ExternalLink className="h-3.5 w-3.5" /> Source originale
            </a>
          )}
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-3">
        {/* Colonne principale */}
        <div className="space-y-6 md:col-span-2">
          {/* Resume IA */}
          {article.resume && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Resume IA</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm leading-relaxed">{article.resume}</p>
              </CardContent>
            </Card>
          )}

          {/* Resume aspects ecologiques (uniquement si article Green IT) */}
          {article.resume_ecologique && (
            <Card className="border-green-300 bg-green-50 dark:border-green-900 dark:bg-green-950/30">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base text-green-800 dark:text-green-300">
                  <Leaf className="h-4 w-4" aria-hidden="true" />
                  Aspects ecologiques identifies
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm leading-relaxed text-green-900 dark:text-green-100">
                  {article.resume_ecologique}
                </p>
              </CardContent>
            </Card>
          )}

          {/* Contenu */}
          {article.contenu && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Contenu de l'article</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm leading-relaxed whitespace-pre-line">
                  {article.contenu.length > 3000
                    ? `${article.contenu.slice(0, 3000)}...`
                    : article.contenu}
                </p>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Barre laterale — metriques IA */}
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Classification IA</CardTitle>
              <CardDescription>Resultats du modele</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {article.score_confiance !== null && (
                <div>
                  <p className="text-xs text-muted-foreground mb-1">Confiance</p>
                  <div className="flex items-center gap-2">
                    <div className="h-2 flex-1 rounded-full bg-muted overflow-hidden">
                      <div
                        className="h-full rounded-full bg-green-600 transition-all"
                        style={{ width: `${article.score_confiance * 100}%` }}
                      />
                    </div>
                    <span className="text-sm font-medium">
                      {(article.score_confiance * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              )}

              {article.modele_classification && (
                <div>
                  <p className="text-xs text-muted-foreground mb-1">Modele</p>
                  <p className="flex items-center gap-1 text-sm">
                    <Cpu className="h-3.5 w-3.5" />
                    {article.modele_classification}
                  </p>
                </div>
              )}

              {article.date_analyse && (
                <div>
                  <p className="text-xs text-muted-foreground mb-1">Analyse</p>
                  <p className="flex items-center gap-1 text-sm">
                    <Clock className="h-3.5 w-3.5" />
                    {formatDate(article.date_analyse)}
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
