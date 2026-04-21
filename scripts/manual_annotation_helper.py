"""Outil d'annotation manuelle des articles borderline (B2.10 audit).

Ce script propose a l'annotateur humain (KaRn1zC) les articles classifies
par le LLM judge avec une confiance **borderline** (score 0.3-0.7), pour
valider ou corriger la decision du modele. Les corrections sont
persistees en base avec une tracabilite complete (``annotation_source``,
``annotated_at``, ``annotated_by``).

Usage
-----

Annotation complete (toutes sources, par ordre ascendant de score)::

    uv run python scripts/manual_annotation_helper.py

Filtrer par source (recommande : commencer par GreenIT.fr qui contient
969 des 1325 borderline — potentiels faux negatifs) ::

    uv run python scripts/manual_annotation_helper.py --source "GreenIT.fr"

Elargir la fenetre de score (ex: couvrir aussi 0.7-0.8) ::

    uv run python scripts/manual_annotation_helper.py --score-max 0.8

Restreindre le nombre d'articles par session (annotation courte) ::

    uv run python scripts/manual_annotation_helper.py --limit 50

Saisie interactive
------------------

Pour chaque article, l'outil affiche :

- Titre + URL + source
- Score LLM judge et decision actuelle
- Resume de classification (~200 mots)
- Contenu tronque (extrait des 1500 premiers caracteres)
- Raison du LLM (si disponible, peuplee lors d'un futur re-run classify)

Saisie attendue :

- ``g`` ou ``G`` : classer Green IT (ou corriger Non Green IT -> Green IT)
- ``n`` ou ``N`` : classer Non Green IT (ou confirmer la decision LLM)
- ``s`` ou ``S`` : passer cet article (revenir plus tard)
- ``o`` ou ``O`` : ouvrir l'URL dans le navigateur pour plus de contexte
- ``q`` ou ``Q`` : sauvegarder et quitter

Reprise de session
------------------

Les articles deja annotes (``annotation_source='manual'``) sont
automatiquement exclus lors du prochain lancement : on ne repasse jamais
deux fois sur le meme article sauf si on reinitialise ``annotation_source``
manuellement en base.

"""

from __future__ import annotations

import argparse
import asyncio
import webbrowser
from datetime import UTC, datetime

from loguru import logger
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from sqlalchemy import and_, or_, select, update

from greentech.data.storage.database import async_session_factory
from greentech.data.storage.models import Article, Source

# Valeurs par defaut pour la fenetre borderline
DEFAULT_SCORE_MIN = 0.3
DEFAULT_SCORE_MAX = 0.7
# Annotateur par defaut (configurable via --by)
DEFAULT_ANNOTATOR = "KaRn1zC"
# Longueur du contenu affiche pour le contexte (au-dela on tronque)
CONTENT_PREVIEW_CHARS = 1500


console = Console()


async def fetch_borderline_articles(
    *,
    score_min: float,
    score_max: float,
    source_filter: str | None,
    limit: int | None,
) -> list[dict]:
    """Recupere les articles borderline non encore annotes manuellement.

    Args:
        score_min: Score minimum (inclus) de la fenetre borderline.
        score_max: Score maximum (inclus).
        source_filter: Nom exact d'une source pour filtrer, ou None pour
            toutes les sources.
        limit: Plafond optionnel sur le nombre d'articles retournes.

    Returns:
        Liste de dicts ``{id_article, titre, url, contenu, resume,
        langue, score, est_green_it, raison, source_nom}``.
    """
    async with async_session_factory() as session:
        stmt = (
            select(
                Article.id_article,
                Article.titre,
                Article.url,
                Article.contenu,
                Article.resume,
                Article.langue,
                Article.score_confiance,
                Article.est_green_it,
                Article.raison_llm_judge,
                Source.nom,
            )
            .join(Source, Source.id_source == Article.id_source)
            .where(
                and_(
                    Article.modele_classification == "keyword_filter+qwen_llm_judge",
                    Article.score_confiance.between(score_min, score_max),
                    or_(
                        Article.annotation_source.is_(None),
                        Article.annotation_source != "manual",
                    ),
                )
            )
            .order_by(
                # Tri par source (pour grouper) puis par score croissant
                # (les plus incertains en premier).
                Source.nom,
                Article.score_confiance,
            )
        )
        if source_filter:
            stmt = stmt.where(Source.nom == source_filter)
        if limit is not None:
            stmt = stmt.limit(limit)

        result = await session.execute(stmt)
        return [
            {
                "id_article": row[0],
                "titre": row[1],
                "url": row[2],
                "contenu": row[3] or "",
                "resume": row[4] or "",
                "langue": row[5] or "",
                "score": float(row[6]) if row[6] is not None else 0.0,
                "est_green_it": bool(row[7]) if row[7] is not None else None,
                "raison": row[8] or "",
                "source_nom": row[9] or "(inconnue)",
            }
            for row in result.all()
        ]


async def persist_annotation(
    *,
    id_article: int,
    est_green_it: bool,
    annotator: str,
) -> None:
    """Ecrit une annotation manuelle en base avec tracabilite complete.

    Ne touche pas aux champs ``score_confiance`` / ``raison_llm_judge`` /
    ``modele_classification`` : on conserve l'historique de la decision
    LLM originale pour pouvoir mesurer le taux de correction humaine.

    Args:
        id_article: Identifiant de l'article.
        est_green_it: Nouvelle valeur binaire (True = Green IT).
        annotator: Identifiant de l'annotateur (defaut ``KaRn1zC``).
    """
    async with async_session_factory() as session:
        await session.execute(
            update(Article)
            .where(Article.id_article == id_article)
            .values(
                est_green_it=est_green_it,
                annotation_source="manual",
                annotated_at=datetime.now(UTC),
                annotated_by=annotator,
            )
        )
        await session.commit()


def _render_article_panel(article: dict, position: int, total: int) -> None:
    """Affiche un article avec mise en forme Rich pour faciliter la decision."""
    console.rule(f"[bold cyan]Article {position}/{total}")

    # Metadonnees
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_row("[bold]ID[/]", str(article["id_article"]))
    table.add_row("[bold]Source[/]", article["source_nom"])
    table.add_row("[bold]Langue[/]", article["langue"])
    table.add_row("[bold]URL[/]", f"[link]{article['url']}[/link]")
    table.add_row(
        "[bold]Decision LLM[/]",
        f"{'Green IT' if article['est_green_it'] else 'Non Green IT'} "
        f"(score {article['score']:.2f})",
    )
    if article["raison"]:
        table.add_row("[bold]Raison LLM[/]", article["raison"])
    console.print(table)

    # Titre en panneau distinct
    console.print(Panel(article["titre"], title="Titre", border_style="blue"))

    # Resume de classification (si present)
    if article["resume"]:
        console.print(
            Panel(
                Markdown(article["resume"][:CONTENT_PREVIEW_CHARS]),
                title="Resume de classification",
                border_style="green",
            )
        )

    # Extrait du contenu
    if article["contenu"]:
        preview = article["contenu"][:CONTENT_PREVIEW_CHARS]
        if len(article["contenu"]) > CONTENT_PREVIEW_CHARS:
            preview += "\n\n[...]"
        console.print(Panel(preview, title="Contenu (extrait)", border_style="yellow"))


async def run_session(
    *,
    score_min: float,
    score_max: float,
    source_filter: str | None,
    limit: int | None,
    annotator: str,
) -> None:
    """Boucle principale d'annotation interactive.

    Args:
        score_min: Borne basse de la fenetre score borderline.
        score_max: Borne haute.
        source_filter: Nom d'une source pour filtrer, ou None.
        limit: Nombre max d'articles a traiter dans la session.
        annotator: Identifiant humain (pour tracabilite ``annotated_by``).
    """
    articles = await fetch_borderline_articles(
        score_min=score_min,
        score_max=score_max,
        source_filter=source_filter,
        limit=limit,
    )

    if not articles:
        console.print(
            "[yellow]Aucun article borderline a annoter avec ces filtres.[/]\n"
            "Verifier la source ou elargir la fenetre de score."
        )
        return

    console.print(
        Panel(
            f"[bold]Session d'annotation manuelle[/]\n\n"
            f"Articles a traiter : [bold cyan]{len(articles)}[/]\n"
            f"Fenetre score    : [{score_min:.2f} ; {score_max:.2f}]\n"
            f"Source           : {source_filter or 'toutes'}\n"
            f"Annotateur       : {annotator}\n\n"
            "Saisies : [bold green]g[/]=Green IT  [bold red]n[/]=Non Green IT  "
            "[yellow]s[/]=skip  [blue]o[/]=ouvrir URL  [white]q[/]=quitter",
            border_style="bold",
        )
    )

    stats = {"green": 0, "non_green": 0, "skip": 0, "url_opened": 0}

    for idx, article in enumerate(articles, start=1):
        _render_article_panel(article, position=idx, total=len(articles))

        while True:
            choice = Prompt.ask(
                "\n[bold]Decision[/] (g/n/s/o/q)",
                choices=["g", "n", "s", "o", "q"],
                default="s",
                show_choices=False,
            ).lower()

            if choice == "o":
                # Ouvre l'URL et redemande la decision
                try:
                    webbrowser.open(article["url"])
                    stats["url_opened"] += 1
                    console.print(f"[blue]URL ouverte dans le navigateur : {article['url']}")
                except Exception as exc:
                    console.print(f"[red]Impossible d'ouvrir l'URL : {exc}")
                continue  # reboucle pour demander la decision apres consultation
            break

        if choice == "q":
            console.print("\n[bold]Session interrompue par l'utilisateur.[/]")
            break

        if choice == "s":
            stats["skip"] += 1
            console.print("[yellow]Article saute.[/]")
            continue

        est_green = choice == "g"
        await persist_annotation(
            id_article=article["id_article"],
            est_green_it=est_green,
            annotator=annotator,
        )
        if est_green:
            stats["green"] += 1
        else:
            stats["non_green"] += 1
        tag = "[green]Green IT[/]" if est_green else "[red]Non Green IT[/]"
        corrige = est_green != article["est_green_it"]
        if corrige:
            console.print(f"{tag} (correction du LLM - enregistree)")
        else:
            console.print(f"{tag} (confirmation du LLM - enregistree)")

    # Bilan de session
    console.rule("[bold]Bilan de session[/]")
    console.print(
        f"  Green IT      : [bold green]{stats['green']}[/]\n"
        f"  Non Green IT  : [bold red]{stats['non_green']}[/]\n"
        f"  Skip          : [yellow]{stats['skip']}[/]\n"
        f"  URL ouvertes  : [blue]{stats['url_opened']}[/]"
    )


def main() -> None:
    """Point d'entree CLI avec arguments."""
    from greentech.utils.logger import setup_logging

    setup_logging(level="INFO", enable_loki=False)

    parser = argparse.ArgumentParser(
        description="Annotation manuelle des articles borderline LLM judge (B2.10)."
    )
    parser.add_argument(
        "--score-min",
        type=float,
        default=DEFAULT_SCORE_MIN,
        help=f"Score minimum de la fenetre borderline (defaut {DEFAULT_SCORE_MIN})",
    )
    parser.add_argument(
        "--score-max",
        type=float,
        default=DEFAULT_SCORE_MAX,
        help=f"Score maximum (defaut {DEFAULT_SCORE_MAX})",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Filtrer par nom exact d'une source (ex: 'GreenIT.fr')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limiter le nombre d'articles a traiter dans cette session",
    )
    parser.add_argument(
        "--by",
        type=str,
        default=DEFAULT_ANNOTATOR,
        help=f"Identifiant de l'annotateur (defaut {DEFAULT_ANNOTATOR})",
    )
    args = parser.parse_args()

    if args.score_min < 0 or args.score_max > 1 or args.score_min >= args.score_max:
        logger.error(
            f"Fenetre score invalide : [{args.score_min}, {args.score_max}]. "
            "Attendu : 0 <= score_min < score_max <= 1."
        )
        raise SystemExit(2)

    asyncio.run(
        run_session(
            score_min=args.score_min,
            score_max=args.score_max,
            source_filter=args.source,
            limit=args.limit,
            annotator=args.by,
        )
    )


if __name__ == "__main__":
    main()
