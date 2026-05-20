"""Validation end-to-end de la chaine GreenTech Intelligence (etape B4.6).

Verifie que le modele de production fonctionne reellement a travers toute la
chaine applicative deployee, et non plus seulement en benchmark hors-ligne :

    Client HTTP -> API FastAPI -> Celery/Redis -> resume LLM + classifieur
    Qwen3 de production -> PostgreSQL -> reponse JSON

Le script s'execute contre une stack `docker-compose` en marche (profil complet)
ou contre un deploiement distant via `--base-url`. Il enchaine des controles de
plus en plus profonds et termine sur un rapport synthetique + un code de sortie
exploitable en CI (0 = succes, 1 = au moins un controle critique echoue).

Controles effectues :
    1. Sante de l'API (`/health`) et connexion base de donnees
    2. Statistiques globales (`/stats`) et liste paginee (`/articles`)
    3. Authentification complete (register + login JWT)
    4. N analyses reelles via `POST /analyze` + suivi du job Celery jusqu'a
       son terme, avec controle de la coherence des reponses et de la polarite
       de classification (mini-evaluation green / non-green)
    5. Metriques Prometheus d'inference (`/metrics`)
    6. Datasources et tableaux de bord Grafana (controle non bloquant)

Example:
    ```bash
    # Validation locale par defaut (10 analyses contre la stack docker)
    uv run python scripts/validate_end_to_end.py

    # Validation rapide (3 analyses) contre une autre instance
    uv run python scripts/validate_end_to_end.py --count 3 --base-url http://localhost:8000
    ```
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from loguru import logger

# ---------------------------------------------------------------------------
# Jeux d'essai : un echantillon bilingue equilibre green / non-green. Les
# textes "green" doivent etre classes Green IT, les "non_green" non. Le modele
# n'etant pas parfait, la polarite sert de mini-evaluation (taux de reussite)
# et non de critere bloquant article par article.
# ---------------------------------------------------------------------------
ECHANTILLONS: list[dict[str, Any]] = [
    {
        "label": "green",
        "attendu": True,
        "payload": {
            "texte": (
                "A carbon-aware job scheduler shifts machine learning training "
                "workloads to periods when the electricity grid is powered by "
                "renewable energy, cutting the carbon footprint of data center "
                "GPU clusters by 40 percent without degrading model accuracy."
            )
        },
    },
    {
        "label": "green",
        "attendu": True,
        "payload": {
            "texte": (
                "L'eco-conception logicielle vise a reduire l'empreinte "
                "environnementale du numerique : optimisation du code, sobriete "
                "des requetes reseau, allongement de la duree de vie des "
                "terminaux et mesure de l'impact carbone des applications web."
            )
        },
    },
    {
        "label": "green",
        "attendu": True,
        "payload": {
            "texte": (
                "Researchers describe a liquid cooling architecture for "
                "hyperscale data centers that reuses waste heat to warm nearby "
                "buildings, improving the power usage effectiveness (PUE) and "
                "lowering the overall energy consumption of cloud infrastructure."
            )
        },
    },
    {
        "label": "green",
        "attendu": True,
        "payload": {
            "texte": (
                "Une etude compare l'empreinte carbone de plusieurs modeles de "
                "langage et propose des techniques de quantification et "
                "d'elagage permettant de diviser par trois la consommation "
                "energetique lors de l'inference, au service d'une IA frugale."
            )
        },
    },
    {
        "label": "green",
        "attendu": True,
        "payload": {
            "texte": (
                "A green software engineering guide explains how to measure the "
                "energy proportionality of microservices, schedule batch jobs in "
                "low-carbon-intensity regions, and adopt sustainable web design "
                "principles to shrink page weight and server load."
            )
        },
    },
    {
        "label": "non_green",
        "attendu": False,
        "payload": {
            "texte": (
                "The latest flagship smartphone introduces a 200-megapixel "
                "camera sensor with improved optical zoom and a faster "
                "fingerprint reader, alongside a brighter display and a redesigned "
                "titanium frame for a more premium feel."
            )
        },
    },
    {
        "label": "non_green",
        "attendu": False,
        "payload": {
            "texte": (
                "L'entreprise a publie ses resultats financiers trimestriels, "
                "annoncant un chiffre d'affaires en hausse de douze pour cent "
                "porte par les ventes de sa division jeux video et le succes de "
                "son dernier titre exclusif sur consoles de salon."
            )
        },
    },
    {
        "label": "non_green",
        "attendu": False,
        "payload": {
            "texte": (
                "A new streaming series breaks viewership records over its "
                "opening weekend, dominating social media conversations as fans "
                "debate the surprise plot twist and the casting choices for the "
                "upcoming second season."
            )
        },
    },
    {
        "label": "non_green",
        "attendu": False,
        "payload": {
            "texte": (
                "Le club a officialise le transfert de son nouvel attaquant pour "
                "un montant record, un contrat de cinq ans signe a quelques jours "
                "de l'ouverture du championnat, suscitant l'enthousiasme des "
                "supporters reunis devant le stade."
            )
        },
    },
    {
        "label": "non_green",
        "attendu": False,
        "payload": {
            "texte": (
                "A popular video game studio unveils its next open-world action "
                "title, showcasing a new combat system, photorealistic character "
                "models and a sprawling map, with a release date set for the "
                "holiday season on all major platforms."
            )
        },
    },
]


@dataclass
class Controle:
    """Resultat d'un controle individuel de la validation."""

    nom: str
    succes: bool
    detail: str = ""
    critique: bool = True


@dataclass
class Rapport:
    """Accumulateur des controles et synthese finale."""

    controles: list[Controle] = field(default_factory=list)

    def ajouter(self, nom: str, succes: bool, detail: str = "", *, critique: bool = True) -> None:
        """Enregistre un controle et le journalise immediatement."""
        self.controles.append(Controle(nom, succes, detail, critique))
        icone = "OK " if succes else ("KO " if critique else "WARN")
        log = logger.success if succes else (logger.error if critique else logger.warning)
        log(f"[{icone}] {nom}{' : ' + detail if detail else ''}")

    @property
    def succes_global(self) -> bool:
        """Vrai si aucun controle critique n'a echoue."""
        return all(c.succes for c in self.controles if c.critique)

    def afficher_synthese(self) -> None:
        """Affiche le tableau recapitulatif final."""
        total = len(self.controles)
        ok = sum(1 for c in self.controles if c.succes)
        logger.info("=" * 70)
        logger.info(f"SYNTHESE VALIDATION END-TO-END : {ok}/{total} controles au vert")
        for c in self.controles:
            etat = "OK  " if c.succes else ("ECHEC" if c.critique else "WARN ")
            logger.info(f"  [{etat}] {c.nom}{' — ' + c.detail if c.detail else ''}")
        logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Controles unitaires
# ---------------------------------------------------------------------------
async def verifier_sante(client: httpx.AsyncClient, rapport: Rapport) -> None:
    """Controle l'endpoint `/health` et la connexion base de donnees."""
    try:
        reponse = await client.get("/health", timeout=15)
        donnees = reponse.json()
        ok = reponse.status_code == 200 and donnees.get("status") == "ok"
        bdd = bool(donnees.get("database"))
        rapport.ajouter(
            "Sante API (/health)",
            ok and bdd,
            f"status={donnees.get('status')}, database={bdd}, version={donnees.get('version')}",
        )
    except Exception as exc:  # noqa: BLE001 - on veut un controle robuste
        rapport.ajouter("Sante API (/health)", False, f"exception : {exc}")


async def verifier_stats(client: httpx.AsyncClient, rapport: Rapport) -> None:
    """Controle l'endpoint `/stats` (donnees classifiees presentes en base)."""
    try:
        reponse = await client.get("/stats", timeout=15)
        donnees = reponse.json()
        total = int(donnees.get("total_articles", 0))
        pct = donnees.get("pourcentage_green_it")
        rapport.ajouter(
            "Statistiques globales (/stats)",
            reponse.status_code == 200 and total > 0,
            f"{total} articles, {donnees.get('articles_green_it')} Green IT ({pct} %)",
        )
    except Exception as exc:  # noqa: BLE001
        rapport.ajouter("Statistiques globales (/stats)", False, f"exception : {exc}")


async def verifier_articles(client: httpx.AsyncClient, rapport: Rapport) -> None:
    """Controle la liste paginee `/articles` et la forme des enregistrements."""
    try:
        reponse = await client.get("/articles", params={"limit": 5}, timeout=15)
        donnees = reponse.json()
        articles = donnees.get("articles", [])
        ok = reponse.status_code == 200 and len(articles) > 0
        champ_ok = bool(articles) and "titre" in articles[0]
        rapport.ajouter(
            "Liste paginee (/articles)",
            ok and champ_ok,
            f"{len(articles)} articles retournes sur {donnees.get('total')} au total",
        )
    except Exception as exc:  # noqa: BLE001
        rapport.ajouter("Liste paginee (/articles)", False, f"exception : {exc}")


async def authentifier(client: httpx.AsyncClient, rapport: Rapport) -> str | None:
    """Cree un compte de test ephemere et retourne un jeton JWT valide.

    Returns:
        Le jeton d'acces si l'authentification reussit, sinon ``None``.
    """
    email = f"e2e_validation_{int(time.time())}@example.com"
    mot_de_passe = "GreenTechE2E2026!"
    try:
        inscription = await client.post(
            "/auth/register",
            json={"email": email, "password": mot_de_passe},
            timeout=20,
        )
        # 201 (cree) ou 400 (deja existant lors d'un re-run rapide) sont acceptables
        if inscription.status_code not in (200, 201, 400):
            rapport.ajouter(
                "Authentification (register + login)",
                False,
                f"register HTTP {inscription.status_code} : {inscription.text[:120]}",
            )
            return None

        connexion = await client.post(
            "/auth/login",
            json={"email": email, "password": mot_de_passe},
            timeout=20,
        )
        jeton = connexion.json().get("access_token") if connexion.status_code == 200 else None
        rapport.ajouter(
            "Authentification (register + login)",
            bool(jeton),
            f"compte {email} cree, jeton JWT {'obtenu' if jeton else 'absent'}",
        )
        return jeton
    except Exception as exc:  # noqa: BLE001
        rapport.ajouter("Authentification (register + login)", False, f"exception : {exc}")
        return None


_STATUTS_TERMINAUX = {"termine", "echec", "erreur", "success", "failure", "error", "done"}


async def soumettre_analyse(
    client: httpx.AsyncClient, jeton: str, payload: dict[str, Any]
) -> str | None:
    """Soumet une analyse et retourne l'identifiant de job Celery."""
    reponse = await client.post(
        "/analyze",
        json=payload,
        headers={"Authorization": f"Bearer {jeton}"},
        timeout=30,
    )
    if reponse.status_code not in (200, 201, 202):
        logger.warning(f"Soumission refusee (HTTP {reponse.status_code}) : {reponse.text[:120]}")
        return None
    return reponse.json().get("job_id")


async def attendre_resultat(
    client: httpx.AsyncClient,
    jeton: str,
    job_id: str,
    *,
    deadline: float,
    intervalle: float,
) -> dict[str, Any] | None:
    """Interroge un job jusqu'a son terme ou jusqu'au depassement de la deadline."""
    entetes = {"Authorization": f"Bearer {jeton}"}
    while time.monotonic() < deadline:
        reponse = await client.get(f"/analyze/{job_id}", headers=entetes, timeout=15)
        if reponse.status_code == 200:
            donnees = reponse.json()
            statut = str(donnees.get("statut", "")).lower()
            if statut in _STATUTS_TERMINAUX or donnees.get("est_green_it") is not None:
                return donnees
        await asyncio.sleep(intervalle)
    return None


async def verifier_analyses(
    client: httpx.AsyncClient,
    jeton: str,
    rapport: Rapport,
    *,
    nombre: int,
    timeout_global: float,
    intervalle: float,
) -> None:
    """Lance N analyses reelles et controle leurs resultats de bout en bout."""
    echantillons = [ECHANTILLONS[i % len(ECHANTILLONS)] for i in range(nombre)]

    # Soumission concurrente : Celery traitera selon la concurrence du worker.
    job_ids = await asyncio.gather(
        *(soumettre_analyse(client, jeton, e["payload"]) for e in echantillons)
    )
    soumis = [(e, j) for e, j in zip(echantillons, job_ids, strict=True) if j]
    rapport.ajouter(
        "Soumission des analyses (POST /analyze)",
        len(soumis) == nombre,
        f"{len(soumis)}/{nombre} jobs acceptes",
    )
    if not soumis:
        return

    deadline = time.monotonic() + timeout_global
    resultats = await asyncio.gather(
        *(
            attendre_resultat(client, jeton, j, deadline=deadline, intervalle=intervalle)
            for _, j in soumis
        )
    )

    termines = 0
    coherents = 0
    bonnes_polarites = 0
    latences: list[float] = []
    for (echantillon, _), resultat in zip(soumis, resultats, strict=True):
        if resultat is None:
            continue
        termines += 1
        statut = str(resultat.get("statut", "")).lower()
        est_green = resultat.get("est_green_it")
        score = resultat.get("score_confiance")
        resume = resultat.get("resume") or ""
        modele = resultat.get("modele_classification") or ""
        latence = resultat.get("temps_inference_ms")

        coherent = (
            statut == "termine"
            and isinstance(est_green, bool)
            and isinstance(score, (int, float))
            and 0.0 <= float(score) <= 1.0
            and len(resume) > 20
            and "production" in modele
        )
        coherents += int(coherent)
        if isinstance(latence, (int, float)):
            latences.append(float(latence))
        if est_green == echantillon["attendu"]:
            bonnes_polarites += 1

    rapport.ajouter(
        "Analyses terminees dans le delai",
        termines == len(soumis),
        f"{termines}/{len(soumis)} jobs termines",
    )
    rapport.ajouter(
        "Coherence des reponses (schema + modele production)",
        coherents == termines and termines > 0,
        f"{coherents}/{termines} reponses coherentes",
    )
    if latences:
        moyenne = sum(latences) / len(latences)
        rapport.ajouter(
            "Latence d'inference relevee",
            True,
            f"moyenne {moyenne:.0f} ms (min {min(latences):.0f}, max {max(latences):.0f})",
            critique=False,
        )
    # La polarite est une mini-evaluation indicative (le modele n'est pas parfait).
    rapport.ajouter(
        "Polarite de classification (green / non-green)",
        bonnes_polarites >= max(1, int(0.7 * termines)),
        f"{bonnes_polarites}/{termines} classifications conformes a l'attendu",
        critique=False,
    )


async def verifier_metriques(client: httpx.AsyncClient, rapport: Rapport) -> None:
    """Controle que `/metrics` expose des compteurs d'inference Prometheus."""
    try:
        reponse = await client.get("/metrics", timeout=15)
        corps = reponse.text
        attendus = [m for m in ("inference", "analyze", "greentech", "http_request") if m in corps]
        rapport.ajouter(
            "Metriques Prometheus (/metrics)",
            reponse.status_code == 200 and len(attendus) > 0,
            f"familles detectees : {', '.join(attendus) or 'aucune'}",
        )
    except Exception as exc:  # noqa: BLE001
        rapport.ajouter("Metriques Prometheus (/metrics)", False, f"exception : {exc}")


async def verifier_grafana(url: str, utilisateur: str, mot_de_passe: str, rapport: Rapport) -> None:
    """Controle (non bloquant) la sante Grafana, ses datasources et dashboards."""
    try:
        async with httpx.AsyncClient(base_url=url, auth=(utilisateur, mot_de_passe)) as client:
            sante = await client.get("/api/health", timeout=10)
            if sante.status_code != 200:
                rapport.ajouter(
                    "Grafana (datasources + dashboards)",
                    False,
                    f"/api/health HTTP {sante.status_code}",
                    critique=False,
                )
                return
            datasources = await client.get("/api/datasources", timeout=10)
            recherche = await client.get("/api/search", params={"type": "dash-db"}, timeout=10)
            n_ds = len(datasources.json()) if datasources.status_code == 200 else 0
            n_dash = len(recherche.json()) if recherche.status_code == 200 else 0
            detail = f"{n_ds} datasources, {n_dash} dashboards"
            if datasources.status_code == 401 or recherche.status_code == 401:
                detail = "authentification Grafana refusee (verifier identifiants admin)"
                rapport.ajouter("Grafana (datasources + dashboards)", False, detail, critique=False)
                return
            rapport.ajouter(
                "Grafana (datasources + dashboards)",
                n_ds > 0 and n_dash > 0,
                detail,
                critique=False,
            )
    except Exception as exc:  # noqa: BLE001
        rapport.ajouter(
            "Grafana (datasources + dashboards)", False, f"exception : {exc}", critique=False
        )


async def main(args: argparse.Namespace) -> int:
    """Orchestre la validation end-to-end et retourne le code de sortie."""
    logger.info(f"Validation end-to-end contre {args.base_url} ({args.count} analyses)")
    rapport = Rapport()

    async with httpx.AsyncClient(base_url=args.base_url) as client:
        await verifier_sante(client, rapport)
        await verifier_stats(client, rapport)
        await verifier_articles(client, rapport)

        jeton = await authentifier(client, rapport)
        if jeton:
            await verifier_analyses(
                client,
                jeton,
                rapport,
                nombre=args.count,
                timeout_global=args.timeout,
                intervalle=args.poll_interval,
            )
        else:
            rapport.ajouter(
                "Analyses reelles (POST /analyze)",
                False,
                "ignorees faute de jeton d'authentification",
            )

        await verifier_metriques(client, rapport)

    await verifier_grafana(args.grafana_url, args.grafana_user, args.grafana_password, rapport)

    rapport.afficher_synthese()
    return 0 if rapport.succes_global else 1


def parser_arguments() -> argparse.Namespace:
    """Definit et lit les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Validation end-to-end GreenTech Intelligence (B4.6)"
    )
    parser.add_argument("--base-url", default="http://localhost:8000", help="URL de base de l'API")
    parser.add_argument("--count", type=int, default=10, help="Nombre d'analyses /analyze a lancer")
    parser.add_argument(
        "--timeout", type=float, default=900.0, help="Budget global (s) pour les jobs"
    )
    parser.add_argument(
        "--poll-interval", type=float, default=8.0, help="Intervalle de polling (s)"
    )
    parser.add_argument("--grafana-url", default="http://localhost:3000", help="URL Grafana")
    parser.add_argument("--grafana-user", default="admin", help="Utilisateur Grafana")
    parser.add_argument("--grafana-password", default="admin123", help="Mot de passe Grafana")
    return parser.parse_args()


if __name__ == "__main__":
    code = asyncio.run(main(parser_arguments()))
    sys.exit(code)
