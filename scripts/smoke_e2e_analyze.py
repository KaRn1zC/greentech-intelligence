"""Smoke test end-to-end : authentification + 10 analyses /analyze via Celery.

Valide la chaine complete API + Redis + Celery + worker + Postgres + Prometheus :

1. Cree un compte de test (ou se logue si deja cree).
2. Recupere un JWT.
3. Soumet 10 analyses variees (Green IT clair, Non-Green IT clair, borderline,
   FR et EN) via POST /analyze.
4. Polle GET /analyze/{job_id} jusqu'a SUCCESS ou FAILURE pour chaque tache.
5. Affiche un tableau recap : statut, latence end-to-end, score, modele.

Pas d'assert "strict" sur la prediction Green IT/Non-Green : la calibration
peut produire des faux positifs sur du texte court. L'objectif est de valider
que la chaine technique fonctionne (pas zero erreur, pas de timeout, pas de
deadlock Celery).

Usage
-----

::

    # Stack full Docker (API sur :8000)
    docker compose --profile full up -d

    # Mode hybride : infra Docker + API + worker locaux
    docker compose up -d
    uv run uvicorn src.greentech.api.main:app --port 8000 &
    uv run celery -A greentech.api.celery_app worker --pool=solo &

    # Puis lancer ce smoke test
    uv run python scripts/smoke_e2e_analyze.py
"""

from __future__ import annotations

import sys
import time
import uuid

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

import httpx

API_BASE = "http://localhost:8000"
# TLD valide (RFC 6761 reserve les TLD `.local` `.test` `.invalid` `.example`,
# certains validators email les refusent). On utilise `.io` (TLD reel).
TEST_EMAIL = "smoke-e2e@greentech.io"
TEST_PASSWORD = "smoke-test-12345"
POLL_INTERVAL_S = 3
POLL_TIMEOUT_S = 300

SAMPLES: list[tuple[str, str]] = [
    ("Green IT EN clair", "A novel approach to reduce datacenter energy consumption by 30% via dynamic VM scheduling. The method maintains performance while cutting power dissipation, applicable to cloud server farms."),
    ("Green IT FR clair", "Eco-conception web : guide complet pour reduire l empreinte carbone d un site internet. Techniques abordees : lazy loading, optimisation des images en WebP, mutualisation des CDN, hebergement vert et compression Brotli."),
    ("Non-Green clair (smartphone)", "Apple unveils the new iPhone 17 Pro Max with an improved triple-camera system, faster A19 Bionic chip and longer battery life. The phone supports Wi-Fi 7 and satellite messaging."),
    ("Non-Green clair (finance)", "The European Central Bank announced a 25 basis points interest rate cut today, citing slowing inflation across the eurozone. Markets reacted positively with the EURO STOXX 50 gaining 1.8%."),
    ("Green IT EN (cloud carbon)", "Carbon Aware Computing in Kubernetes: this article describes how to shift compute workloads geographically and temporally to align with grid renewable energy availability, reducing operational CO2 emissions."),
    ("Green IT FR (datacenter)", "Comment OVHcloud refroidit ses datacenters sans climatisation grace au water cooling direct et a la chaleur fatale recuperee pour chauffer des batiments voisins. Etude de l empreinte energetique reduite."),
    ("Borderline (climat sans IT)", "Climate change accelerates the melting of Arctic sea ice at a rate three times faster than predicted by IPCC models. The albedo feedback loop worsens warming in polar regions."),
    ("Borderline (energie sans IT)", "France a inaugure son plus grand parc eolien offshore au large de Saint-Nazaire, capable d alimenter en electricite environ 800 000 foyers grace a ses 80 turbines."),
    ("Green IT EN (ML carbon)", "MLPerf adds a new energy efficiency metric to its benchmark suite, allowing researchers to compare deep learning models not only on accuracy but also on power consumption per inference."),
    ("Non-Green clair (jeu video)", "The publisher Square Enix announced the release date for Final Fantasy XVII, set for early 2027 on PlayStation 6 and PC. The trailer showcases an expanded combat system and new summon abilities."),
]


def main() -> int:
    print("=" * 78)
    print("  Smoke E2E /analyze : 10 articles via Celery + Redis + worker")
    print("=" * 78)
    print()

    with httpx.Client(base_url=API_BASE, timeout=30.0) as client:
        # 1. Authentication
        print("[1/4] Authentification...")
        try:
            r = client.post(
                "/auth/register",
                json={"email": TEST_EMAIL, "password": TEST_PASSWORD},
            )
            if r.status_code in (200, 201):
                print(f"  Compte {TEST_EMAIL} cree")
            elif r.status_code == 400 and "already exists" in r.text.lower():
                print(f"  Compte {TEST_EMAIL} existe deja, login direct")
            else:
                print(f"  Register inattendu : {r.status_code} {r.text[:200]}")
        except Exception as exc:
            print(f"  Register error : {exc}")

        r = client.post(
            "/auth/login",
            json={"email": TEST_EMAIL, "password": TEST_PASSWORD},
        )
        if r.status_code != 200:
            print(f"  ECHEC login : {r.status_code} {r.text[:200]}")
            return 1
        token = r.json().get("access_token")
        if not token:
            print(f"  Pas d'access_token dans la reponse : {r.json()}")
            return 1
        print(f"  Token JWT recupere ({len(token)} chars)")
        print()

        auth = {"Authorization": f"Bearer {token}"}

        # 2. Submission of 10 analyses
        print(f"[2/4] Soumission de {len(SAMPLES)} analyses...")
        jobs: list[tuple[str, uuid.UUID, float]] = []
        t0_total = time.time()
        for label, texte in SAMPLES:
            t0 = time.time()
            r = client.post("/analyze", json={"texte": texte}, headers=auth)
            elapsed_ms = (time.time() - t0) * 1000
            if r.status_code != 202:
                print(f"  ECHEC POST {label} : {r.status_code} {r.text[:200]}")
                continue
            data = r.json()
            job_id = uuid.UUID(data["job_id"])
            jobs.append((label, job_id, time.time()))
            print(f"  {label:35s} -> job_id={str(job_id)[:8]}... ({elapsed_ms:.0f}ms)")
        total_submission_ms = (time.time() - t0_total) * 1000
        print(f"  Total submission : {total_submission_ms:.0f}ms (non-bloquant attendu)")
        print()

        if not jobs:
            print("Aucun job soumis avec succes, abandon")
            return 1

        # 3. Polling until completion
        print(f"[3/4] Polling jusqu'a completion (timeout {POLL_TIMEOUT_S}s, interval {POLL_INTERVAL_S}s)...")
        results: dict[uuid.UUID, dict] = {}
        deadline = time.time() + POLL_TIMEOUT_S
        while time.time() < deadline and len(results) < len(jobs):
            for label, job_id, _t_submit in jobs:
                if job_id in results:
                    continue
                r = client.get(f"/analyze/{job_id}", headers=auth)
                if r.status_code != 200:
                    print(f"  GET {label} : {r.status_code}")
                    continue
                data = r.json()
                statut = data.get("statut")
                if statut in ("termine", "erreur"):
                    results[job_id] = data
                    print(f"  {label:35s} -> {statut}")
            if len(results) < len(jobs):
                time.sleep(POLL_INTERVAL_S)

        if len(results) < len(jobs):
            print(f"  TIMEOUT : {len(results)}/{len(jobs)} taches terminees")
        else:
            print(f"  Toutes les taches terminees : {len(results)}/{len(jobs)}")
        print()

        # 4. Recap table
        print("[4/4] Recap des resultats")
        print("-" * 110)
        print(f"  {'Label':35s} {'Statut':10s} {'Green':6s} {'Proba':7s} {'Latence':10s} {'Modele':25s}")
        print("-" * 110)
        ok = 0
        ko = 0
        for label, job_id, t_submit in jobs:
            r = results.get(job_id)
            if not r:
                print(f"  {label:35s} {'TIMEOUT':10s}")
                ko += 1
                continue
            statut = r.get("statut", "?")
            green = r.get("est_green_it")
            proba = r.get("score_confiance")
            temps_inf = r.get("temps_inference_ms") or 0
            modele = r.get("modele_classification") or "n/a"
            modele_short = modele.split("\\")[-1].split("/")[-1] if modele else "n/a"
            green_str = "yes" if green is True else "no" if green is False else "?"
            proba_str = f"{proba:.3f}" if isinstance(proba, (int, float)) else "?"
            print(f"  {label:35s} {statut:10s} {green_str:6s} {proba_str:7s} {temps_inf:>6d}ms   {modele_short[:25]}")
            if statut == "termine":
                ok += 1
            else:
                ko += 1
        print("-" * 110)
        total_e2e_s = time.time() - t0_total
        print(f"  Total E2E : {total_e2e_s:.1f}s | OK={ok} | KO/TIMEOUT={ko}")
        print()

        if ko > 0:
            print(f"AVERTISSEMENT : {ko}/{len(jobs)} taches non terminees")
            return 2
        print("SUCCES : 10/10 taches traitees via la chaine Celery")
        return 0


if __name__ == "__main__":
    sys.exit(main())
