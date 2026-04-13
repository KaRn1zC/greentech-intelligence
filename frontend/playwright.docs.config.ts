import { defineConfig } from "@playwright/test"

/**
 * Configuration Playwright dediee a l'audit d'accessibilite de la
 * documentation Sphinx.
 *
 * Cette configuration :
 *   1. Lance un serveur HTTP statique servant `docs/_build/html` sur le port 8800.
 *   2. N'execute que les tests presents dans `tests/docs-accessibility.spec.ts`.
 *   3. Genere un rapport HTML distinct (`a11y-docs-report/`) pour ne pas
 *      ecraser celui de l'audit du frontend applicatif.
 *
 * Prerequis : avoir build la doc avant (`sphinx-build -b html docs docs/_build/html`).
 *
 * Execution locale :
 *   `npm run test:a11y:docs`
 */
export default defineConfig({
  testDir: "./tests",
  testMatch: "**/docs-accessibility.spec.ts",
  timeout: 30000,
  use: {
    baseURL: "http://localhost:8800",
    headless: true,
  },
  webServer: {
    // Sert le contenu HTML buildé par Sphinx via le module http.server de Python
    // (disponible partout, pas besoin de dependance Node supplementaire).
    command: "python -m http.server 8800 --directory ../docs/_build/html",
    port: 8800,
    reuseExistingServer: true,
    timeout: 30000,
  },
  projects: [
    {
      name: "chromium",
      use: { browserName: "chromium" },
    },
  ],
  reporter: [["html", { outputFolder: "a11y-docs-report", open: "never" }]],
})
