"""Spiders Scrapy pour le scraping de sites Green IT statiques.

Package introduit en B2.3 : quatre spiders dedies aux sites Green IT
identifies en validation prealable (B2.1) :

- ``GreenItFrSpider`` : scraping du blog francophone greenit.fr
- ``GreenSoftwareSpider`` : articles de la Green Software Foundation
- ``SustainableWebSpider`` : sustainablewebdesign.org (posts + guidelines)
- ``ClimateActionTechSpider`` : climateaction.tech

Ces spiders heritent de ``StaticArticleSpider`` (base.py) qui centralise
la logique commune (decouverte d'URLs, extraction titre/contenu via
chaine de selecteurs CSS, filtrage MIN_CONTENT_LENGTH, dedup par URL).

Architecture Scrapy HTTP (pas Playwright)
-----------------------------------------

Contrairement a ``TechCrunchArticleSpider`` qui utilise Playwright pour
attendre le rendu JS de TechCrunch (site React), ces 4 spiders operent
sur du HTML statique : Scrapy + httpx suffisent. Cela donne :

- ~5x plus rapide (pas de Chromium a lancer par page)
- ~5x moins energivore (coherent avec positionnement Green IT du projet)
- ~5x moins de RAM (50 MB vs 300 MB par contexte)
- Moins de modes de defaillance (pas de crash Chromium, pas de recyclage)

Le design garde toutefois les hooks pour activer Playwright par site si
un des sites ajoute du JS critique plus tard (voir ``StaticArticleSpider.
enable_playwright``).
"""
