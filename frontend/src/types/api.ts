// Types API — miroir des schemas Pydantic du backend FastAPI

// === Auth ===

export interface TokenResponse {
  access_token: string
  token_type: string
}

export interface UserResponse {
  id_utilisateur: string
  email: string
  est_actif: boolean
  est_verifie: boolean
  date_creation: string
}

// === Articles ===

export interface ArticleListItem {
  id_article: number
  uuid: string
  titre: string
  url: string
  resume: string | null
  auteur: string | null
  date_publication: string | null
  est_green_it: boolean | null
  score_confiance: number | null
  nom_source: string | null
  date_creation: string
}

export interface ArticleDetail extends ArticleListItem {
  contenu: string | null
  langue: string
  modele_classification: string | null
  date_analyse: string | null
  date_modification: string
}

export interface ArticleListResponse {
  articles: ArticleListItem[]
  total: number
  page: number
  limit: number
  pages: number
}

// === Stats ===

export interface GlobalStats {
  total_articles: number
  articles_green_it: number
  articles_non_green_it: number
  en_attente_analyse: number
  score_confiance_moyen: number | null
  pourcentage_green_it: number | null
}

export interface DailyStatsItem {
  date_stat: string
  total_articles: number
  articles_green_it: number
  articles_non_green_it: number
  score_confiance_moyen: number | null
}

export interface DailyStatsResponse {
  stats: DailyStatsItem[]
  periode_debut: string
  periode_fin: string
}

export interface SourceStatsItem {
  id_source: number
  nom: string
  type: string
  total_articles: number
  articles_green_it: number
  articles_non_green_it: number
  derniere_collecte: string | null
}

export interface SourceStatsResponse {
  sources: SourceStatsItem[]
}

// === Analyse ===

export type AnalysisStatus = "en_attente" | "en_cours" | "termine" | "erreur"

export interface AnalysisJobCreated {
  job_id: string
  statut: AnalysisStatus
  message: string
}

export interface AnalysisResult {
  job_id: string
  statut: AnalysisStatus
  id_article: number | null
  titre: string | null
  est_green_it: boolean | null
  score_confiance: number | null
  resume: string | null
  modele_classification: string | null
  temps_inference_ms: number | null
  date_analyse: string | null
  erreur: string | null
}
