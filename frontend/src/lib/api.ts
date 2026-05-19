import { getToken, removeToken } from "./auth"
import type {
  AnalysisJobCreated,
  AnalysisResult,
  ArticleDetail,
  ArticleListResponse,
  GlobalStats,
  TokenResponse,
  UserResponse,
} from "@/types/api"

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000"

class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.name = "ApiError"
    this.status = status
  }
}

async function apiFetch<T>(path: string, options: RequestInit = {}): Promise<T> {
  const token = getToken()
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...((options.headers as Record<string, string>) || {}),
  }

  if (token) {
    headers["Authorization"] = `Bearer ${token}`
  }

  const response = await fetch(`${API_URL}${path}`, {
    ...options,
    headers,
  })

  if (response.status === 401) {
    removeToken()
    window.location.href = "/login"
    throw new ApiError(401, "Session expiree")
  }

  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: "Erreur serveur" }))
    throw new ApiError(response.status, formatErrorDetail(body.detail, response.status))
  }

  return response.json() as Promise<T>
}

/**
 * FastAPI renvoie `detail` sous deux formes : une string pour les HTTPException
 * "metier" (401, 404, 409...) et un tableau d'objets pour les erreurs Pydantic
 * (422). Sans aplatissement, ce tableau finissait affiche en `[object Object]`
 * dans les toasts. Cette fonction normalise les deux cas en message lisible.
 */
function formatErrorDetail(detail: unknown, status: number): string {
  if (typeof detail === "string" && detail.length > 0) return detail

  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => {
        if (typeof item === "string") return item
        if (item && typeof item === "object" && "msg" in item) {
          return String((item as { msg: unknown }).msg)
        }
        return null
      })
      .filter((m): m is string => Boolean(m))
    if (messages.length > 0) return messages.join(" • ")
  }

  if (detail && typeof detail === "object" && "msg" in detail) {
    return String((detail as { msg: unknown }).msg)
  }

  return `Erreur ${status}`
}

// === Auth ===

export async function login(email: string, password: string): Promise<TokenResponse> {
  return apiFetch<TokenResponse>("/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  })
}

export async function register(email: string, password: string): Promise<UserResponse> {
  return apiFetch<UserResponse>("/auth/register", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  })
}

export async function getMe(): Promise<UserResponse> {
  return apiFetch<UserResponse>("/auth/me")
}

// === Articles ===

export async function getArticles(params?: {
  page?: number
  limit?: number
  is_green_it?: boolean
}): Promise<ArticleListResponse> {
  const searchParams = new URLSearchParams()
  if (params?.page) searchParams.set("page", String(params.page))
  if (params?.limit) searchParams.set("limit", String(params.limit))
  if (params?.is_green_it !== undefined)
    searchParams.set("is_green_it", String(params.is_green_it))

  const query = searchParams.toString()
  return apiFetch<ArticleListResponse>(`/articles${query ? `?${query}` : ""}`)
}

export async function getArticle(id: number): Promise<ArticleDetail> {
  return apiFetch<ArticleDetail>(`/articles/${id}`)
}

export async function searchArticles(
  q: string,
  page = 1,
): Promise<ArticleListResponse> {
  return apiFetch<ArticleListResponse>(
    `/articles/search?q=${encodeURIComponent(q)}&page=${page}`,
  )
}

// === Stats ===

export async function getStats(): Promise<GlobalStats> {
  return apiFetch<GlobalStats>("/stats")
}

// === Analyse ===

export async function submitAnalysis(input: {
  url?: string
  texte?: string
}): Promise<AnalysisJobCreated> {
  return apiFetch<AnalysisJobCreated>("/analyze", {
    method: "POST",
    body: JSON.stringify(input),
  })
}

export async function submitAnalysisFile(file: File): Promise<AnalysisJobCreated> {
  const token = getToken()
  const formData = new FormData()
  formData.append("fichier", file)

  const headers: Record<string, string> = {}
  if (token) {
    headers["Authorization"] = `Bearer ${token}`
  }

  const response = await fetch(`${API_URL}/analyze/file`, {
    method: "POST",
    headers,
    body: formData,
  })

  if (response.status === 401) {
    removeToken()
    window.location.href = "/login"
    throw new ApiError(401, "Session expiree")
  }

  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: "Erreur serveur" }))
    throw new ApiError(response.status, body.detail || `Erreur ${response.status}`)
  }

  return response.json() as Promise<AnalysisJobCreated>
}

export async function getAnalysisStatus(jobId: string): Promise<AnalysisResult> {
  return apiFetch<AnalysisResult>(`/analyze/${jobId}`)
}

export { ApiError }
