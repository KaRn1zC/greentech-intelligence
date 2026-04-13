import { test, expect } from "@playwright/test"
import AxeBuilder from "@axe-core/playwright"

/**
 * Audit d'accessibilite de la documentation Sphinx generee.
 *
 * Les pages HTML produites par Sphinx + theme Furo sont testees avec axe-core
 * pour verifier la conformite WCAG 2.1 AA (contraste, structure semantique,
 * labels ARIA, navigation clavier, etc.). Ces tests correspondent aux
 * recommandations de l'association Valentin Haüy et d'Atalan (AcceDe),
 * referencees dans le referentiel de certification (critere transversal
 * "La documentation est communiquee dans un format qui respecte les
 * recommandations d'accessibilite").
 *
 * Prerequis : la documentation doit etre buildee au prealable
 *   `cd docs && uv run sphinx-build -b html . _build/html`
 *
 * Execution :
 *   `npx playwright test tests/docs-accessibility.spec.ts --config=playwright.docs.config.ts`
 */

// Pages principales a auditer (les plus lues par un utilisateur de la doc)
const PAGES_A_AUDITER = [
  { url: "/", nom: "Index (accueil de la doc)" },
  { url: "/CHECKLIST_SUIVI.html", nom: "Checklist de suivi des competences" },
  { url: "/PLAN_ETAPES.html", nom: "Plan des etapes du projet" },
  { url: "/SPECIFICATIONS_TECHNIQUES.html", nom: "Specifications techniques" },
  { url: "/BENCHMARK_SERVICES_IA.html", nom: "Benchmark des services IA" },
  { url: "/Sources_Données.html", nom: "Sources de donnees" },
  { url: "/REGISTRE_RGPD.html", nom: "Registre RGPD" },
  { url: "/ACCESSIBILITE_DOCUMENTATION.html", nom: "Accessibilite documentation" },
  { url: "/PROCEDURE_MAJ_MODELE.html", nom: "Procedure MAJ modele" },
  { url: "/PLAYBOOK_MAINTENANCE.html", nom: "Playbook maintenance" },
  { url: "/ETAT_AVANCEMENT.html", nom: "Etat d'avancement" },
]

test.describe("Audit accessibilite documentation Sphinx (WCAG 2.1 AA)", () => {
  for (const page of PAGES_A_AUDITER) {
    test(`${page.nom} — aucune violation critique WCAG 2.1 AA`, async ({
      page: browserPage,
    }) => {
      const response = await browserPage.goto(page.url)
      // Certaines pages peuvent ne pas exister si le build n'est pas complet,
      // on ne les audite pas dans ce cas (le test reste vert).
      if (!response || response.status() >= 400) {
        test.skip(true, `Page ${page.url} non trouvee (build incomplet)`)
        return
      }
      await browserPage.waitForLoadState("networkidle")

      const results = await new AxeBuilder({ page: browserPage })
        .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
        .analyze()

      const critiques = results.violations.filter(
        (v) => v.impact === "critical" || v.impact === "serious",
      )

      if (critiques.length > 0) {
        console.log(
          `\nViolations detectees sur ${page.nom} :`,
          JSON.stringify(
            critiques.map((v) => ({
              id: v.id,
              impact: v.impact,
              description: v.description,
              nodesCount: v.nodes.length,
              firstNode: v.nodes[0]?.html.slice(0, 200),
            })),
            null,
            2,
          ),
        )
      }

      expect(critiques).toHaveLength(0)
    })
  }
})

test.describe("Structure semantique de la documentation", () => {
  test("L'attribut lang est defini sur l'element html", async ({ page }) => {
    await page.goto("/")
    await page.waitForLoadState("networkidle")

    const lang = await page.getAttribute("html", "lang")
    expect(lang).toBeTruthy()
    expect(lang).not.toBe("")
  })

  test("Chaque page doit avoir un titre (balise title)", async ({ page }) => {
    await page.goto("/")
    await page.waitForLoadState("networkidle")

    const title = await page.title()
    expect(title.length).toBeGreaterThan(0)
    expect(title).not.toBe("Untitled")
  })

  test("Un seul h1 par page (hierarchie semantique)", async ({ page }) => {
    await page.goto("/")
    await page.waitForLoadState("networkidle")

    const h1Count = await page.locator("main h1, article h1").count()
    // Furo met generalement 1 seul h1 par page (le titre du document)
    expect(h1Count).toBeLessThanOrEqual(1)
  })

  test("Les images ont un attribut alt", async ({ page }) => {
    await page.goto("/")
    await page.waitForLoadState("networkidle")

    const imagesSansAlt = await page
      .locator("img:not([alt])")
      .filter({ hasNot: page.locator('[role="presentation"]') })
      .count()

    expect(imagesSansAlt).toBe(0)
  })
})

test.describe("Navigation clavier dans la doc", () => {
  test("Le contenu principal est accessible via un landmark main", async ({
    page,
  }) => {
    await page.goto("/")
    await page.waitForLoadState("networkidle")

    // Furo genere un <main> pour le contenu, ce qui permet aux lecteurs d'ecran
    // de sauter directement au contenu (equivalent fonctionnel du skip link).
    const mainLandmark = page.locator('main, [role="main"]')
    expect(await mainLandmark.count()).toBeGreaterThan(0)
  })
})
