-- Migration 002 : enrichissement sources B2 (arXiv API + Crossref)
--
-- Ajoute les nouvelles sources REST/JSON et les mots-cles associes pour
-- la phase B2 d'enrichissement du dataset Green IT. Les spiders Scrapy
-- seront ajoutes dans une migration ulterieure (B2.3).
--
-- Idempotent : peut etre rejoue sans creer de doublons (INSERT ... ON
-- CONFLICT DO NOTHING sur les contraintes UNIQUE).
--
-- Date : 2026-04-19
-- Auteur : KaRn1zC

-- ============================================================================
-- Mise a jour de la CHECK constraint sur search_config.type_source
-- ============================================================================
-- La contrainte existante limitait type_source a {api, scraping, file,
-- guardian, devto, newsdata}. On y ajoute les nouveaux types B2 :
-- arxiv_api, crossref. Les spiders Scrapy futurs continueront a utiliser
-- 'scraping' (pas besoin d'un type dedie par site).

ALTER TABLE search_config DROP CONSTRAINT IF EXISTS search_config_type_source_check;
ALTER TABLE search_config ADD CONSTRAINT search_config_type_source_check
    CHECK (type_source IN (
        'api',
        'scraping',
        'file',
        'guardian',
        'devto',
        'newsdata',
        'arxiv_api',
        'crossref'
    ));

-- ============================================================================
-- Nouvelles sources
-- ============================================================================

-- arXiv API : collecteur REST/JSON vivant (complementaire de la source
-- 'arXiv Dataset' de type=file qui utilise le dump Kaggle historique).
INSERT INTO sources (nom, type, url_base, description, est_active)
VALUES (
    'arXiv API',
    'api',
    'https://export.arxiv.org/api/query',
    'API Atom XML arXiv pour recuperer les abstracts de publications '
    'scientifiques (preprints) en lien avec le Green IT. '
    'Complementaire du dataset Kaggle historique (source "arXiv Dataset"). '
    'Categories ciblees : cs.*, eess.*, stat.ML.',
    true
)
ON CONFLICT (nom) DO NOTHING;

-- Crossref : API JSON pour publications peer-reviewed d'editeurs
-- (Springer, Elsevier, IEEE, ACM, etc.). Utilise le Polite Pool via
-- mailto dans le User-Agent.
INSERT INTO sources (nom, type, url_base, description, est_active)
VALUES (
    'Crossref',
    'api',
    'https://api.crossref.org/works',
    'API JSON Crossref pour publications editoriales peer-reviewed. '
    'Filtre sur has-abstract:true + journal-article + from-pub-date:2020. '
    'Polite Pool active via CROSSREF_MAILTO.',
    true
)
ON CONFLICT (nom) DO NOTHING;

-- ============================================================================
-- Mots-cles arXiv API (type_source='arxiv_api')
-- ============================================================================
-- Choix des mots-cles : on privilegie les queries ciblees ayant <= 200
-- resultats sur arXiv pour garder un signal Green IT fort. Les queries
-- larges comme "efficient inference" (1499 resultats) ou "model
-- compression" (1585) sont volontairement ecartees : elles dilueraient
-- la pertinence et depasseraient notre plafond MAX_RESULTS_PER_KEYWORD.
--
-- Volume attendu (total) : 300-1500 articles bruts, 100-300 apres
-- dedup et classification LLM judge.

INSERT INTO search_config (mot_cle, type_source, priorite, actif)
VALUES
    ('green computing', 'arxiv_api', 1, true),
    ('sustainable AI', 'arxiv_api', 1, true),
    ('green AI', 'arxiv_api', 1, true),
    ('carbon-aware computing', 'arxiv_api', 2, true),
    ('energy-efficient ML', 'arxiv_api', 2, true),
    ('green software engineering', 'arxiv_api', 2, true),
    ('low-power neural network', 'arxiv_api', 3, true),
    ('data center sustainability', 'arxiv_api', 2, true),
    ('sustainable computing', 'arxiv_api', 1, true)
ON CONFLICT DO NOTHING;

-- ============================================================================
-- Mots-cles Crossref (type_source='crossref')
-- ============================================================================
-- Recherche par query.title pour precision. Chaque mot-cle capture les
-- top 200 articles pertinents (tries par relevance score Crossref).
--
-- Volume attendu (total) : 500-1500 candidats bruts, dedoublonnes par
-- DOI. Apres classification LLM judge : 100-400 confirmes Green IT.

INSERT INTO search_config (mot_cle, type_source, priorite, actif)
VALUES
    ('green computing', 'crossref', 1, true),
    ('sustainable AI', 'crossref', 1, true),
    ('carbon-aware computing', 'crossref', 2, true),
    ('green software', 'crossref', 2, true),
    ('energy-efficient inference', 'crossref', 2, true),
    ('green AI', 'crossref', 1, true),
    ('sustainable computing', 'crossref', 1, true),
    ('data center efficiency', 'crossref', 2, true)
ON CONFLICT DO NOTHING;

-- ============================================================================
-- Verification post-migration (pour les logs du deploiement)
-- ============================================================================

DO $$
DECLARE
    arxiv_count INT;
    crossref_count INT;
    new_sources INT;
BEGIN
    SELECT COUNT(*) INTO arxiv_count
    FROM search_config
    WHERE type_source = 'arxiv_api' AND actif;

    SELECT COUNT(*) INTO crossref_count
    FROM search_config
    WHERE type_source = 'crossref' AND actif;

    SELECT COUNT(*) INTO new_sources
    FROM sources
    WHERE nom IN ('arXiv API', 'Crossref');

    RAISE NOTICE 'Migration 002 : arXiv API keywords=%, Crossref keywords=%, nouvelles sources=%',
        arxiv_count, crossref_count, new_sources;
END $$;
