import { useState, type FormEvent } from "react"
import { Navigate, useNavigate } from "react-router-dom"
import { toast } from "sonner"
import { Leaf, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useAuth } from "@/hooks/useAuth"
import { register as apiRegister, ApiError } from "@/lib/api"

export function LoginPage() {
  const { login, isAuthenticated } = useAuth()
  const navigate = useNavigate()

  const [isRegister, setIsRegister] = useState(false)
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [error, setError] = useState("")
  const [loading, setLoading] = useState(false)

  // Redirection declarative pour eviter "setState during render" de react-router
  // (navigate(...) pendant le render declenche un update du routeur parent).
  if (isAuthenticated) {
    return <Navigate to="/" replace />
  }

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    setError("")
    setLoading(true)

    try {
      if (isRegister) {
        await apiRegister(email, password)
        toast.success("Compte cree avec succes")
      }
      await login(email, password)
      toast.success("Connexion reussie")
      navigate("/")
    } catch (err) {
      const message = err instanceof ApiError
        ? err.message
        : "Une erreur inattendue est survenue."
      setError(message)
      toast.error(message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex min-h-[70vh] items-center justify-center">
      <Card className="w-full max-w-sm">
        <CardHeader className="text-center">
          <div className="mx-auto mb-2 flex h-10 w-10 items-center justify-center rounded-full bg-green-100">
            <Leaf className="h-5 w-5 text-green-600" />
          </div>
          <CardTitle className="text-xl">
            {isRegister ? "Creer un compte" : "Connexion"}
          </CardTitle>
          <CardDescription>
            {isRegister
              ? "Inscrivez-vous pour analyser des articles"
              : "Connectez-vous a votre compte"}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="email">Email</Label>
              <Input
                id="email"
                type="email"
                placeholder="vous@exemple.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                autoComplete="email"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="password">Mot de passe</Label>
              <Input
                id="password"
                type="password"
                placeholder="8 caracteres minimum"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                minLength={8}
                autoComplete={isRegister ? "new-password" : "current-password"}
              />
            </div>

            {error && (
              <p className="text-sm text-destructive" role="alert">{error}</p>
            )}

            <Button type="submit" className="w-full" disabled={loading}>
              {loading && <Loader2 className="animate-spin" />}
              {isRegister ? "S'inscrire" : "Se connecter"}
            </Button>

            <p className="text-center text-sm text-muted-foreground">
              {isRegister ? "Deja inscrit ?" : "Pas encore de compte ?"}{" "}
              <button
                type="button"
                onClick={() => { setIsRegister(!isRegister); setError("") }}
                className="text-primary underline-offset-4 hover:underline"
              >
                {isRegister ? "Se connecter" : "S'inscrire"}
              </button>
            </p>
          </form>
        </CardContent>
      </Card>
    </div>
  )
}
