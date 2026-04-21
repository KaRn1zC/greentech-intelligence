-- =============================================================================
-- GreenTech Intelligence - Script d'initialisation de la base de données
-- PostgreSQL 15+
-- =============================================================================

-- Extension pour UUID
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- TABLES DE CONFIGURATION
-- =============================================================================

-- Table de configuration des recherches (Source SQL dynamique).
-- type_source accepte les categories generiques (api, scraping, file) ainsi
-- que les sous-types dedies a chaque API REST/JSON (guardian, devto,
-- newsdata legacy, arxiv_api, crossref). Cela permet aux collecteurs de
-- filtrer precisement leurs mots-cles sans se marcher dessus.
CREATE TABLE IF NOT EXISTS search_config (
    id_config SERIAL PRIMARY KEY,
    mot_cle VARCHAR(100) NOT NULL,
    url_source TEXT,
    type_source VARCHAR(20) CHECK (type_source IN (
        'api', 'scraping', 'file',
        'guardian', 'devto', 'newsdata',
        'arxiv_api', 'crossref'
    )),
    priorite INTEGER DEFAULT 1,
    actif BOOLEAN DEFAULT true,
    date_creation TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    date_modification TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Donnees initiales de configuration. Depuis avril 2026, les collecteurs
-- REST/JSON utilisent The Guardian Open Platform (tier Developer,
-- 5000 req/jour) et Dev.to / Forem API (pas de cle requise). NewsData.io
-- reste configurable via type_source='newsdata' pour usage futur, mais
-- n'est pas alimentee par defaut a cause de la troncature "ONLY AVAILABLE
-- IN PAID PLANS" en free tier.
INSERT INTO search_config (mot_cle, type_source, priorite) VALUES
    -- === The Guardian (API REST/JSON principale, Green IT / Sustainability) ===
    ('green IT', 'guardian', 1),
    ('sustainable AI', 'guardian', 1),
    ('data center sustainability', 'guardian', 2),
    ('carbon footprint computing', 'guardian', 2),
    ('energy efficient AI', 'guardian', 2),
    ('green cloud computing', 'guardian', 2),
    ('renewable energy technology', 'guardian', 3),
    ('eco-friendly software', 'guardian', 3),
    ('sustainable digital infrastructure', 'guardian', 3),
    ('e-waste recycling', 'guardian', 3),
    ('smart grid technology', 'guardian', 3),
    ('low power machine learning', 'guardian', 3),
    ('climate tech', 'guardian', 2),
    ('circular economy technology', 'guardian', 3),
    ('carbon neutral data center', 'guardian', 2),
    -- === Dev.to (tags Green IT / sustainability) ===
    ('greenit', 'devto', 1),
    ('sustainability', 'devto', 1),
    ('climatechange', 'devto', 2),
    ('webperf', 'devto', 2),
    ('environment', 'devto', 2),
    ('cleanenergy', 'devto', 3),
    ('sustainabletech', 'devto', 3),
    ('greensoftware', 'devto', 3),
    -- === TechCrunch (scraping hybride RSS + HTML) ===
    ('Carbon Footprint Software', 'scraping', 2),
    ('Energy Efficient Computing', 'scraping', 2),
    ('Data Center Sustainability', 'scraping', 3),
    -- === arXiv API (complementaire au dataset Kaggle historique) ===
    -- Queries ciblees (< 200 resultats attendus) pour preserver la pertinence
    -- Green IT. Volume attendu total : 300-1500 articles bruts.
    ('green computing', 'arxiv_api', 1),
    ('sustainable AI', 'arxiv_api', 1),
    ('green AI', 'arxiv_api', 1),
    ('carbon-aware computing', 'arxiv_api', 2),
    ('energy-efficient ML', 'arxiv_api', 2),
    ('green software engineering', 'arxiv_api', 2),
    ('low-power neural network', 'arxiv_api', 3),
    ('data center sustainability', 'arxiv_api', 2),
    ('sustainable computing', 'arxiv_api', 1),
    -- === Crossref (peer-reviewed, Polite Pool via CROSSREF_MAILTO) ===
    -- Recherche par query.title, top 200 par mot-cle tries par relevance.
    ('green computing', 'crossref', 1),
    ('sustainable AI', 'crossref', 1),
    ('carbon-aware computing', 'crossref', 2),
    ('green software', 'crossref', 2),
    ('energy-efficient inference', 'crossref', 2),
    ('green AI', 'crossref', 1),
    ('sustainable computing', 'crossref', 1),
    ('data center efficiency', 'crossref', 2)
ON CONFLICT DO NOTHING;

-- =============================================================================
-- TABLES DE DONNÉES
-- =============================================================================

-- Sources de données
CREATE TABLE IF NOT EXISTS sources (
    id_source SERIAL PRIMARY KEY,
    nom VARCHAR(100) NOT NULL UNIQUE,
    type VARCHAR(20) NOT NULL CHECK (type IN ('api', 'scraping', 'file')),
    url_base TEXT,
    description TEXT,
    est_active BOOLEAN DEFAULT true,
    derniere_collecte TIMESTAMP WITH TIME ZONE,
    date_creation TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Articles collectés
CREATE TABLE IF NOT EXISTS articles (
    id_article SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE,
    id_source INTEGER REFERENCES sources(id_source) ON DELETE SET NULL,

    -- Contenu
    titre VARCHAR(500) NOT NULL,
    url TEXT UNIQUE NOT NULL,
    contenu TEXT,
    resume TEXT,
    -- Résumé orienté "aspects écologiques" (LLM instructif via HF SaaS),
    -- rempli uniquement quand est_green_it = TRUE.
    resume_ecologique TEXT,

    -- Métadonnées
    auteur VARCHAR(200),
    date_publication TIMESTAMP WITH TIME ZONE,
    langue VARCHAR(10) DEFAULT 'en',

    -- Résultats IA
    est_green_it BOOLEAN,
    score_confiance FLOAT CHECK (score_confiance >= 0 AND score_confiance <= 1),
    modele_classification VARCHAR(100),

    -- Audit
    chemin_donnees_brutes TEXT,
    date_analyse TIMESTAMP WITH TIME ZONE,
    date_creation TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    date_modification TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index pour les recherches fréquentes
CREATE INDEX IF NOT EXISTS idx_articles_est_green_it ON articles(est_green_it);
CREATE INDEX IF NOT EXISTS idx_articles_date_publication ON articles(date_publication DESC);
CREATE INDEX IF NOT EXISTS idx_articles_id_source ON articles(id_source);
CREATE INDEX IF NOT EXISTS idx_articles_date_analyse ON articles(date_analyse) WHERE date_analyse IS NOT NULL;

-- =============================================================================
-- TABLES UTILISATEURS (FastAPI Users)
-- =============================================================================

CREATE TABLE IF NOT EXISTS users (
    id_utilisateur UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(320) UNIQUE NOT NULL,
    mot_de_passe_hash VARCHAR(1024) NOT NULL,
    est_actif BOOLEAN DEFAULT true,
    est_superuser BOOLEAN DEFAULT false,
    est_verifie BOOLEAN DEFAULT false,
    date_creation TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    date_modification TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================================================
-- TABLES ANALYTICS
-- =============================================================================

-- Statistiques quotidiennes
CREATE TABLE IF NOT EXISTS daily_stats (
    id_stats SERIAL PRIMARY KEY,
    date_stat DATE UNIQUE NOT NULL,
    total_articles INTEGER DEFAULT 0,
    articles_green_it INTEGER DEFAULT 0,
    articles_non_green_it INTEGER DEFAULT 0,
    score_confiance_moyen FLOAT,
    articles_par_source JSONB,
    date_creation TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Logs d'analyse IA
CREATE TABLE IF NOT EXISTS analysis_logs (
    id_log SERIAL PRIMARY KEY,
    id_article INTEGER REFERENCES articles(id_article) ON DELETE CASCADE,
    nom_modele VARCHAR(100) NOT NULL,
    version_modele VARCHAR(50),
    temps_inference_ms INTEGER,
    emissions_carbone_kg FLOAT,
    prediction BOOLEAN,
    confiance FLOAT,
    date_creation TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================================================
-- FONCTIONS & TRIGGERS
-- =============================================================================

-- Fonction pour mettre à jour date_modification automatiquement
CREATE OR REPLACE FUNCTION update_date_modification_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.date_modification = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers pour date_modification
DROP TRIGGER IF EXISTS update_articles_date_modification ON articles;
CREATE TRIGGER update_articles_date_modification
    BEFORE UPDATE ON articles
    FOR EACH ROW
    EXECUTE FUNCTION update_date_modification_column();

DROP TRIGGER IF EXISTS update_users_date_modification ON users;
CREATE TRIGGER update_users_date_modification
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_date_modification_column();

DROP TRIGGER IF EXISTS update_search_config_date_modification ON search_config;
CREATE TRIGGER update_search_config_date_modification
    BEFORE UPDATE ON search_config
    FOR EACH ROW
    EXECUTE FUNCTION update_date_modification_column();

-- =============================================================================
-- VUES
-- =============================================================================

-- Vue des statistiques globales
CREATE OR REPLACE VIEW v_global_stats AS
SELECT
    COUNT(*) as total_articles,
    COUNT(*) FILTER (WHERE est_green_it = true) as articles_green_it,
    COUNT(*) FILTER (WHERE est_green_it = false) as articles_non_green_it,
    COUNT(*) FILTER (WHERE est_green_it IS NULL) as en_attente_analyse,
    ROUND(AVG(score_confiance)::numeric, 3) as score_confiance_moyen,
    ROUND((COUNT(*) FILTER (WHERE est_green_it = true)::float / NULLIF(COUNT(*) FILTER (WHERE est_green_it IS NOT NULL), 0) * 100)::numeric, 2) as pourcentage_green_it
FROM articles;

-- Vue des articles récents
CREATE OR REPLACE VIEW v_articles_recents AS
SELECT
    a.id_article,
    a.uuid,
    a.titre,
    a.url,
    a.resume,
    a.auteur,
    a.date_publication,
    a.est_green_it,
    a.score_confiance,
    s.nom as nom_source,
    a.date_analyse
FROM articles a
LEFT JOIN sources s ON a.id_source = s.id_source
ORDER BY a.date_creation DESC
LIMIT 100;

-- =============================================================================
-- DONNÉES DE TEST (Développement uniquement)
-- =============================================================================

-- Sources de donnees utilisees par le projet (3 types distincts pour
-- satisfaire le critere E1 / C1 du referentiel de certification : une API
-- REST/JSON, un scraping hybride, et un dataset volumineux traite avec Spark).
INSERT INTO sources (nom, type, url_base, description, est_active) VALUES
    ('The Guardian', 'api', 'https://content.guardianapis.com',
     'API REST/JSON du journal The Guardian (tier Developer gratuit, 5000 req/jour, 12 req/s). Source principale REST/JSON depuis avril 2026 : contenu integral garanti (bodyText), sections environment/technology/sustainable-business.',
     true),
    ('Dev.to', 'api', 'https://dev.to/api',
     'API publique Dev.to / Forem (aucune cle requise). Source REST/JSON complementaire apportant un registre technique/developpeur (tags greenit, sustainability, climatechange, webperf, environment).',
     true),
    ('TechCrunch Climate', 'scraping', 'https://techcrunch.com/category/climate/',
     'Scraping hybride RSS + HTML Scrapy + Playwright. Section Climate de TechCrunch, articles complets extraits depuis le DOM rendu.',
     true),
    ('arXiv Dataset', 'file', 'https://www.kaggle.com/datasets/Cornell-University/arxiv',
     'Dataset scientifique Cornell University (~1.7M articles, 3.6 Go JSON). Traite avec Apache Spark pour filtrer les categories cs.AI/cs.LG/cs.CL/cs.CV/cs.SE.',
     true),
    -- NewsData.io : conservee comme source historique mais desactivee par
    -- defaut depuis avril 2026 a cause de la troncature systematique du
    -- contenu en free tier (message "ONLY AVAILABLE IN PAID PLANS").
    ('NewsData.io', 'api', 'https://newsdata.io/api/1/latest',
     'LEGACY - Source API REST/JSON desactivee en avril 2026 (free tier tronque le contenu au placeholder "ONLY AVAILABLE IN PAID PLANS", dataset inexploitable). Remplacee par The Guardian.',
     false),
    -- === arXiv API (nouvel ajout B2.2) ===
    ('arXiv API', 'api', 'https://export.arxiv.org/api/query',
     'API Atom XML arXiv pour recuperer les abstracts de publications scientifiques en lien avec le Green IT. Complementaire du dataset Kaggle historique (source arXiv Dataset). Categories ciblees : cs.*, eess.*, stat.ML.',
     true),
    -- === Crossref (nouvel ajout B2.2) ===
    ('Crossref', 'api', 'https://api.crossref.org/works',
     'API JSON Crossref pour publications editoriales peer-reviewed. Filtre sur has-abstract:true + journal-article + from-pub-date:2020. Polite Pool active via CROSSREF_MAILTO.',
     true),
    -- === Spiders Scrapy statiques (nouvel ajout B2.3) ===
    ('GreenIT.fr', 'scraping', 'https://www.greenit.fr',
     'Blog francophone reference sur le Green IT (1 001 posts). Ecrit par Frederic Bordage et la communaute, couvre eco-conception web, impact environnemental du numerique, etudes Green IT. Scraping via sitemap WordPress, HTML statique, langue FR.',
     true),
    ('Green Software Foundation', 'scraping', 'https://greensoftware.foundation',
     'Fondation tech dediee au green software engineering (170 articles). Contenu anglophone sur SCI standard, carbon-aware computing, mesure d''emissions logicielles. Scraping via pagination HTML statique.',
     true),
    ('Sustainable Web Design', 'scraping', 'https://sustainablewebdesign.org',
     'Reference WSDG (Web Sustainability Design Guidelines). 131 items : 50 posts + 81 guidelines rediges en prose. Scraping via 2 sitemaps WordPress (post-sitemap + guidelines-sitemap).',
     true),
    ('Climate Action Tech', 'scraping', 'https://climateaction.tech',
     'Communaute de professionnels tech engages sur le climat (71 posts). CAT Salons, retours d''experience, pratiques industrie. Scraping via sitemap WordPress, theme Neve.',
     true)
ON CONFLICT (nom) DO NOTHING;

-- =============================================================================
-- UTILISATEUR APPLICATIF (Droits Restreints)
-- =============================================================================

-- Créer le rôle applicatif (lecture/écriture seulement, pas de DDL)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'greentech_app') THEN
        CREATE ROLE greentech_app WITH LOGIN PASSWORD 'greentech_app_password';
    END IF;
END $$;

GRANT CONNECT ON DATABASE greentech_db TO greentech_app;
GRANT USAGE ON SCHEMA public TO greentech_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO greentech_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO greentech_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO greentech_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO greentech_app;

-- Message de confirmation
DO $$
BEGIN
    RAISE NOTICE 'Base de données GreenTech Intelligence initialisée avec succès !';
    RAISE NOTICE 'Utilisateur applicatif greentech_app créé avec droits restreints.';
END $$;
