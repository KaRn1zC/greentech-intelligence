import { useState, type FormEvent } from "react"
import { Navigate, useNavigate } from "react-router-dom"
import { motion } from "motion/react"
import { toast } from "sonner"
import { Loader2, Lock, Mail } from "lucide-react"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { ShimmerButton } from "@/components/ui/shimmer-button"
import { Meteors } from "@/components/ui/meteors"
import { LeafCircuitLogo } from "@/components/layout/LeafCircuitLogo"
import carbonFootprintIllustration from "@/assets/illustrations/icon-carbon-footprint.png"
import { useAuth } from "@/hooks/useAuth"
import { register as apiRegister, ApiError } from "@/lib/api"
import { cn } from "@/lib/utils"

export function LoginPage() {
  const { login, isAuthenticated } = useAuth()
  const navigate = useNavigate()

  const [isRegister, setIsRegister] = useState(false)
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [error, setError] = useState("")
  const [loading, setLoading] = useState(false)

  // Redirection declarative pour eviter "setState during render" de react-router.
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
    <div className="relative -mx-4 -my-8 flex min-h-[calc(100vh-8rem)] items-center justify-center overflow-hidden px-4 py-8">
      {/* Fond aurora vert/cyan */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 -z-10"
        style={{
          background:
            "radial-gradient(ellipse 60% 50% at 20% 30%, oklch(0.72 0.18 155 / 0.25), transparent 60%), " +
            "radial-gradient(ellipse 50% 45% at 80% 25%, oklch(0.78 0.15 210 / 0.22), transparent 60%), " +
            "radial-gradient(ellipse 70% 55% at 50% 100%, oklch(0.6 0.16 165 / 0.18), transparent 65%)",
        }}
      />

      {/* Couche meteors discrete */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden="true">
        <Meteors number={12} />
      </div>

      {/* Empreinte carbone en watermark decoratif bas-gauche */}
      <img
        src={carbonFootprintIllustration}
        alt=""
        aria-hidden="true"
        draggable={false}
        className={cn(
          "pointer-events-none absolute bottom-6 left-6 z-0",
          "hidden h-40 w-40 select-none opacity-[0.08] md:block",
          "mix-blend-screen",
        )}
      />

      <motion.div
        initial={{ opacity: 0, y: 16, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        className="relative z-10 w-full max-w-md"
      >
        <div
          className={cn(
            "relative overflow-hidden rounded-2xl border border-border/60",
            "bg-surface-glass",
            "p-8 shadow-[0_0_40px_oklch(0.72_0.18_155_/_0.15)]",
          )}
        >
          {/* Bordure lumineuse au top */}
          <div
            className="pointer-events-none absolute inset-x-8 top-0 h-px bg-gradient-to-r from-transparent via-[oklch(0.72_0.18_155_/_0.8)] to-transparent"
            aria-hidden="true"
          />

          <header className="mb-6 text-center">
            <div className="mx-auto mb-4 inline-flex h-14 w-14 items-center justify-center rounded-2xl border border-border/50 bg-card/60">
              <LeafCircuitLogo size={36} />
            </div>
            <h1 className="font-display text-2xl font-semibold tracking-tight">
              {isRegister ? "Creer un compte" : "Connexion"}
            </h1>
            <p className="mt-1.5 font-mono text-xs text-muted-foreground">
              <span className="text-[oklch(0.72_0.18_155)]">&gt;</span>{" "}
              {isRegister ? "initialisation du profil..." : "authentification requise"}
            </p>
          </header>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="email" className="font-mono text-xs uppercase tracking-wider">
                Email
              </Label>
              <div className="relative">
                <Mail
                  className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground"
                  aria-hidden="true"
                />
                <Input
                  id="email"
                  type="email"
                  placeholder="vous@exemple.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  autoComplete="email"
                  className="pl-9"
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="password" className="font-mono text-xs uppercase tracking-wider">
                Mot de passe
              </Label>
              <div className="relative">
                <Lock
                  className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground"
                  aria-hidden="true"
                />
                <Input
                  id="password"
                  type="password"
                  placeholder="8 caracteres minimum"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  minLength={8}
                  autoComplete={isRegister ? "new-password" : "current-password"}
                  className="pl-9"
                />
              </div>
            </div>

            {error && (
              <p
                className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive"
                role="alert"
              >
                {error}
              </p>
            )}

            <ShimmerButton
              type="submit"
              disabled={loading}
              shimmerColor="oklch(0.88 0.12 150)"
              shimmerDuration="2.5s"
              background="oklch(0.205 0.02 170)"
              borderRadius="0.75rem"
              className="w-full font-display text-sm font-medium"
            >
              {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              {isRegister ? "Creer le compte" : "Se connecter"}
            </ShimmerButton>

            <p className="pt-2 text-center text-sm text-muted-foreground">
              {isRegister ? "Deja inscrit ?" : "Pas encore de compte ?"}{" "}
              <button
                type="button"
                onClick={() => { setIsRegister(!isRegister); setError("") }}
                className="font-medium text-[oklch(0.82_0.14_150)] underline-offset-4 transition-colors hover:text-[oklch(0.88_0.12_150)] hover:underline"
              >
                {isRegister ? "Se connecter" : "S'inscrire"}
              </button>
            </p>
          </form>
        </div>
      </motion.div>
    </div>
  )
}
