-- Migration 004 : annotation manuelle + justification LLM judge
--
-- Cette migration ajoute quatre colonnes a la table ``articles`` pour
-- supporter la tracabilite de la classification et l'annotation manuelle
-- des articles borderline identifies lors de l'audit B2.10 (avril 2026).
--
-- 1. ``raison_llm_judge`` : justification textuelle produite par le LLM
--    Qwen lors du classement (etage 2 du pipeline hybride). Le LLM la
--    genere deja (``ClassifierVerdict.raison``) mais elle n'etait pas
--    persistee auparavant — seuls le verdict booleen et le score de
--    confiance l'etaient. Peuplee desormais par ``apply_verdicts`` dans
--    ``scripts/classify_candidates.py`` lors du prochain run LLM naturel.
--
-- 2. ``annotation_source`` : origine de la decision finale. Valeurs :
--       - ``"llm_judge"`` : decision posee par Qwen en etage 2
--       - ``"keyword_filter"`` : decision triviale du pre-filtre (Non Green IT)
--       - ``"manual"`` : decision posee a la main par un annotateur
--    Permet de distinguer les articles audites manuellement de ceux
--    laisses au LLM, notamment pour ne pas ecraser une annotation
--    manuelle lors d'un futur re-run du LLM judge.
--
-- 3. ``annotated_at`` : timestamp de l'annotation manuelle.
--
-- 4. ``annotated_by`` : identifiant de l'annotateur humain (``KaRn1zC``
--    par defaut pour le projet). Permet la tracabilite RGPD et la
--    comprehension a posteriori des decisions manuelles.
--
-- Toutes les colonnes sont idempotentes (``ADD COLUMN IF NOT EXISTS``).
-- Les valeurs sont NULL par defaut : les articles deja classes avant
-- cette migration ne sont pas modifies.
--
-- Date : 2026-04-22
-- Auteur : KaRn1zC

ALTER TABLE articles
    ADD COLUMN IF NOT EXISTS raison_llm_judge TEXT;

COMMENT ON COLUMN articles.raison_llm_judge IS
    'Justification textuelle du LLM Qwen (etage 2), peuplee au prochain run de classify_candidates.py';

ALTER TABLE articles
    ADD COLUMN IF NOT EXISTS annotation_source VARCHAR(20);

COMMENT ON COLUMN articles.annotation_source IS
    'Origine de la classification : llm_judge | keyword_filter | manual';

ALTER TABLE articles
    ADD COLUMN IF NOT EXISTS annotated_at TIMESTAMP WITH TIME ZONE;

COMMENT ON COLUMN articles.annotated_at IS
    'Horodatage de l''annotation manuelle (NULL si classe automatiquement)';

ALTER TABLE articles
    ADD COLUMN IF NOT EXISTS annotated_by VARCHAR(100);

COMMENT ON COLUMN articles.annotated_by IS
    'Identifiant de l''annotateur humain (NULL si classe automatiquement)';

-- Index partiel pour accelerer les requetes d'annotation manuelle sur
-- les articles borderline (score 0.3-0.7). Partiel = ne couvre que les
-- articles susceptibles d'etre audites, donc faible empreinte disque.
CREATE INDEX IF NOT EXISTS idx_articles_borderline_llm
    ON articles (score_confiance, id_source)
    WHERE modele_classification = 'keyword_filter+qwen_llm_judge'
      AND score_confiance BETWEEN 0.3 AND 0.7;

-- Backfill conservateur : les articles deja classes par le LLM judge
-- recoivent annotation_source='llm_judge'. Les articles Non Green IT
-- decides par le pre-filtre seul recoivent annotation_source='keyword_filter'.
-- Les articles encore NULL (non classes) restent NULL.
UPDATE articles
SET annotation_source = 'llm_judge'
WHERE modele_classification = 'keyword_filter+qwen_llm_judge'
  AND annotation_source IS NULL
  AND est_green_it IS NOT NULL;

UPDATE articles
SET annotation_source = 'keyword_filter'
WHERE modele_classification = 'keyword_filter'
  AND annotation_source IS NULL
  AND est_green_it IS NOT NULL;
