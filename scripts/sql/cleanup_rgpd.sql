-- Procedure de purge RGPD automatique
--
-- Conformement au registre des traitements (docs/REGISTRE_RGPD.md) :
--   "Donnees brutes (MinIO raw-data) : 90 jours puis suppression automatique"
--
-- Cette procedure supprime les articles soumis via l'endpoint /analyze
-- (URL prefixee par `analyse-directe://`) plus anciens que 90 jours.
-- Les articles issus des sources publiques (API, scraping, arXiv) ne sont
-- pas concernes : ils proviennent de donnees publiques et sont conserves
-- pour l'entrainement du modele.
--
-- Execution recommandee : cron quotidien (ex. 02:00 UTC).
--   psql -U greentech -d greentech_db -f scripts/sql/cleanup_rgpd.sql
--
-- Date : 2026-04-13
-- Auteur : KaRn1zC

BEGIN;

-- Comptage avant purge (visible dans les logs)
SELECT
    COUNT(*) AS articles_a_purger,
    MIN(date_creation) AS plus_ancien,
    MAX(date_creation) AS plus_recent
FROM articles
WHERE url LIKE 'analyse-directe://%'
  AND date_creation < NOW() - INTERVAL '90 days';

-- Suppression effective
DELETE FROM articles
WHERE url LIKE 'analyse-directe://%'
  AND date_creation < NOW() - INTERVAL '90 days';

-- Suppression des logs d'analyse orphelins (articles supprimes)
-- La cle etrangere id_article dans analysis_logs doit etre ON DELETE CASCADE
-- pour que ce nettoyage se fasse automatiquement via la suppression ci-dessus.

COMMIT;
