-- Migration 001 : ajout de la colonne resume_ecologique
--
-- Cette colonne stocke le resume oriente "aspects ecologiques" genere
-- par un LLM instructif (HF SaaS, par defaut Llama-3.2-3B-Instruct)
-- lorsque l'article est classifie Green IT.
--
-- Compatible avec les deploiements existants : la colonne est ajoutee
-- uniquement si elle n'existe pas deja (idempotent).
--
-- Date : 2026-04-13
-- Auteur : KaRn1zC

ALTER TABLE articles
    ADD COLUMN IF NOT EXISTS resume_ecologique TEXT;

COMMENT ON COLUMN articles.resume_ecologique IS
    'Resume des aspects ecologiques (LLM instructif HF SaaS, rempli si Green IT)';
