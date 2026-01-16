-- =============================================================================
-- GreenTech Intelligence - Database Initialization Script
-- PostgreSQL 15+
-- =============================================================================

-- Extension pour UUID
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- TABLES DE CONFIGURATION
-- =============================================================================

-- Table de configuration des recherches (Source SQL dynamique)
CREATE TABLE IF NOT EXISTS search_config (
    id SERIAL PRIMARY KEY,
    keyword VARCHAR(100) NOT NULL,
    source_url TEXT,
    source_type VARCHAR(20) CHECK (source_type IN ('api', 'scraping', 'file')),
    priority INTEGER DEFAULT 1,
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Données initiales de configuration
INSERT INTO search_config (keyword, source_type, priority) VALUES
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
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    type VARCHAR(20) NOT NULL CHECK (type IN ('api', 'scraping', 'file')),
    base_url TEXT,
    description TEXT,
    is_active BOOLEAN DEFAULT true,
    last_fetched_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Articles collectés
CREATE TABLE IF NOT EXISTS articles (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE,
    source_id INTEGER REFERENCES sources(id) ON DELETE SET NULL,
    
    -- Contenu
    title VARCHAR(500) NOT NULL,
    url TEXT UNIQUE NOT NULL,
    content TEXT,
    summary TEXT,
    
    -- Métadonnées
    author VARCHAR(200),
    published_at TIMESTAMP WITH TIME ZONE,
    language VARCHAR(10) DEFAULT 'en',
    
    -- Résultats IA
    is_green_it BOOLEAN,
    confidence_score FLOAT CHECK (confidence_score >= 0 AND confidence_score <= 1),
    classification_model VARCHAR(100),
    
    -- Audit
    raw_data_path TEXT,
    analyzed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index pour les recherches fréquentes
CREATE INDEX IF NOT EXISTS idx_articles_is_green_it ON articles(is_green_it);
CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_articles_source_id ON articles(source_id);
CREATE INDEX IF NOT EXISTS idx_articles_analyzed ON articles(analyzed_at) WHERE analyzed_at IS NOT NULL;

-- =============================================================================
-- TABLES UTILISATEURS (FastAPI Users)
-- =============================================================================

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(320) UNIQUE NOT NULL,
    hashed_password VARCHAR(1024) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    is_superuser BOOLEAN DEFAULT false,
    is_verified BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================================================
-- TABLES ANALYTICS
-- =============================================================================

-- Statistiques quotidiennes
CREATE TABLE IF NOT EXISTS daily_stats (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    total_articles INTEGER DEFAULT 0,
    green_it_articles INTEGER DEFAULT 0,
    non_green_it_articles INTEGER DEFAULT 0,
    avg_confidence_score FLOAT,
    articles_by_source JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Logs d'analyse IA
CREATE TABLE IF NOT EXISTS analysis_logs (
    id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    inference_time_ms INTEGER,
    carbon_emissions_kg FLOAT,
    prediction BOOLEAN,
    confidence FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================================================
-- FONCTIONS & TRIGGERS
-- =============================================================================

-- Fonction pour mettre à jour updated_at automatiquement
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers pour updated_at
DROP TRIGGER IF EXISTS update_articles_updated_at ON articles;
CREATE TRIGGER update_articles_updated_at
    BEFORE UPDATE ON articles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_search_config_updated_at ON search_config;
CREATE TRIGGER update_search_config_updated_at
    BEFORE UPDATE ON search_config
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- VUES
-- =============================================================================

-- Vue des statistiques globales
CREATE OR REPLACE VIEW v_global_stats AS
SELECT 
    COUNT(*) as total_articles,
    COUNT(*) FILTER (WHERE is_green_it = true) as green_it_count,
    COUNT(*) FILTER (WHERE is_green_it = false) as non_green_it_count,
    COUNT(*) FILTER (WHERE is_green_it IS NULL) as pending_analysis,
    ROUND(AVG(confidence_score)::numeric, 3) as avg_confidence,
    ROUND((COUNT(*) FILTER (WHERE is_green_it = true)::float / NULLIF(COUNT(*) FILTER (WHERE is_green_it IS NOT NULL), 0) * 100)::numeric, 2) as green_it_percentage
FROM articles;

-- Vue des articles récents
CREATE OR REPLACE VIEW v_recent_articles AS
SELECT 
    a.id,
    a.uuid,
    a.title,
    a.url,
    a.summary,
    a.author,
    a.published_at,
    a.is_green_it,
    a.confidence_score,
    s.name as source_name,
    a.analyzed_at
FROM articles a
LEFT JOIN sources s ON a.source_id = s.id
ORDER BY a.created_at DESC
LIMIT 100;

-- =============================================================================
-- DONNÉES DE TEST (Développement uniquement)
-- =============================================================================

-- Sources de test
INSERT INTO sources (name, type, base_url, description) VALUES
    ('NewsAPI', 'api', 'https://newsapi.org/v2', 'News aggregator API'),
    ('Dev.to', 'scraping', 'https://dev.to', 'Developer community blog'),
    ('Medium Tech', 'scraping', 'https://medium.com/tag/technology', 'Medium technology articles'),
    ('Historical CSV', 'file', NULL, 'Historical dataset from CSV import')
ON CONFLICT (name) DO NOTHING;

-- Message de confirmation
DO $$
BEGIN
    RAISE NOTICE 'GreenTech Intelligence database initialized successfully!';
END $$;
