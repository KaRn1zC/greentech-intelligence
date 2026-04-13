-- =============================================================================
-- GreenTech Intelligence - Script d'initialisation de la base de données
-- PostgreSQL 15+
-- =============================================================================

-- Extension pour UUID
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- TABLES DE CONFIGURATION
-- =============================================================================

-- Table de configuration des recherches (Source SQL dynamique)
CREATE TABLE IF NOT EXISTS search_config (
    id_config SERIAL PRIMARY KEY,
    mot_cle VARCHAR(100) NOT NULL,
    url_source TEXT,
    type_source VARCHAR(20) CHECK (type_source IN ('api', 'scraping', 'file')),
    priorite INTEGER DEFAULT 1,
    actif BOOLEAN DEFAULT true,
    date_creation TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    date_modification TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Données initiales de configuration
INSERT INTO search_config (mot_cle, type_source, priorite) VALUES
    ('Green IT', 'api', 1),
    ('Sustainable AI', 'api', 1),
    ('Eco-friendly Tech', 'api', 2),
    ('Carbon Footprint Software', 'scraping', 2),
    ('Energy Efficient Computing', 'scraping', 2),
    ('Sustainable Software Development', 'api', 3),
    ('Data Center Sustainability', 'scraping', 3)
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

-- Sources de test
INSERT INTO sources (nom, type, url_base, description) VALUES
    ('NewsData.io', 'api', 'https://newsdata.io/api/1/latest', 'API REST actualites technologiques'),
    ('TechCrunch Climate', 'scraping', 'https://techcrunch.com/category/climate/', 'Blog tech - section Climate (Scraping Playwright)'),
    ('arXiv Dataset', 'file', 'https://www.kaggle.com/datasets/Cornell-University/arxiv', 'Dataset scientifique Cornell University (1.7M+ articles)')
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
