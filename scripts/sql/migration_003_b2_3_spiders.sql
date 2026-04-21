-- Migration 003 : ajout des 4 sources spiders Green IT (B2.3)
--
-- Complete la migration 002 (arxiv_api + crossref) avec les 4 sources
-- scraping statique introduites en B2.3 :
--
-- - GreenIT.fr               : 1 001 posts FR, 100% Green IT
-- - Green Software Foundation : 170 articles EN, 100% Green IT
-- - Sustainable Web Design    : 131 items EN (posts + guidelines)
-- - Climate Action Tech       : 71 posts EN, tech + climat
--
-- Idempotent : peut etre rejoue sans creer de doublons.
--
-- Date : 2026-04-19
-- Auteur : KaRn1zC

-- ============================================================================
-- Nouvelles sources (type=scraping, complement de TechCrunch Climate)
-- ============================================================================

INSERT INTO sources (nom, type, url_base, description, est_active) VALUES
    ('GreenIT.fr', 'scraping', 'https://www.greenit.fr',
     'Blog francophone reference sur le Green IT (1 001 posts). Ecrit par '
     'Frederic Bordage et la communaute, couvre eco-conception web, impact '
     'environnemental du numerique, etudes Green IT. Scraping via sitemap '
     'WordPress, HTML statique, langue FR.',
     true),
    ('Green Software Foundation', 'scraping', 'https://greensoftware.foundation',
     'Fondation tech dediee au green software engineering (170 articles). '
     'Contenu anglophone sur SCI standard, carbon-aware computing, mesure '
     'd''emissions logicielles. Scraping via pagination HTML statique.',
     true),
    ('Sustainable Web Design', 'scraping', 'https://sustainablewebdesign.org',
     'Reference WSDG (Web Sustainability Design Guidelines). 131 items : '
     '50 posts + 81 guidelines rediges en prose. Scraping via 2 sitemaps '
     'WordPress (post-sitemap + guidelines-sitemap).',
     true),
    ('Climate Action Tech', 'scraping', 'https://climateaction.tech',
     'Communaute de professionnels tech engages sur le climat (71 posts). '
     'CAT Salons, retours d''experience, pratiques industrie. Scraping via '
     'sitemap WordPress, theme Neve.',
     true)
ON CONFLICT (nom) DO NOTHING;

-- ============================================================================
-- Mots-cles search_config : scraping filtre OFF (sites 100% Green IT)
-- ============================================================================
-- Les 4 nouveaux sites sont a haute densite Green IT (contenu deja
-- thematique). Contrairement a TechCrunch dont on filtrait les URLs RSS,
-- ici on scrape integralement chaque site. On n'ajoute donc PAS de
-- mots-cles dedies : les spiders tournent sans filtrage par mot-cle.
--
-- Le contenu global est qualifie ensuite par le LLM judge (etage 2 de
-- la classification hybride).

-- ============================================================================
-- Verification post-migration
-- ============================================================================

DO $$
DECLARE
    new_sources INT;
BEGIN
    SELECT COUNT(*) INTO new_sources
    FROM sources
    WHERE nom IN (
        'GreenIT.fr',
        'Green Software Foundation',
        'Sustainable Web Design',
        'Climate Action Tech'
    );

    RAISE NOTICE 'Migration 003 : nouvelles sources scraping = %', new_sources;
END $$;
