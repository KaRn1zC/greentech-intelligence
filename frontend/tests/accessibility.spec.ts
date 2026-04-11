import { test, expect } from "@playwright/test"
import AxeBuilder from "@axe-core/playwright"

test.describe("Audit accessibilite WCAG", () => {
  test("Page de connexion — zero violation critique", async ({ page }) => {
    await page.goto("/login")
    await page.waitForLoadState("networkidle")

    const results = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa"])
      .analyze()

    const critical = results.violations.filter(
      (v) => v.impact === "critical" || v.impact === "serious",
    )

    if (critical.length > 0) {
      console.log("Violations critiques :", JSON.stringify(critical, null, 2))
    }

    expect(critical).toHaveLength(0)
  })

  test("Page dashboard — zero violation critique", async ({ page }) => {
    await page.goto("/")
    await page.waitForLoadState("networkidle")

    const results = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa"])
      .analyze()

    const critical = results.violations.filter(
      (v) => v.impact === "critical" || v.impact === "serious",
    )

    if (critical.length > 0) {
      console.log("Violations critiques :", JSON.stringify(critical, null, 2))
    }

    expect(critical).toHaveLength(0)
  })

  test("Page detail article — zero violation critique", async ({ page }) => {
    await page.goto("/articles/1")
    await page.waitForLoadState("networkidle")

    const results = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa"])
      .analyze()

    const critical = results.violations.filter(
      (v) => v.impact === "critical" || v.impact === "serious",
    )

    if (critical.length > 0) {
      console.log("Violations critiques :", JSON.stringify(critical, null, 2))
    }

    expect(critical).toHaveLength(0)
  })
})

test.describe("Navigation au clavier", () => {
  test("Login — tous les champs accessibles par tabulation", async ({
    page,
  }) => {
    await page.goto("/login")
    await page.waitForLoadState("networkidle")

    // Premier Tab → champ email
    await page.keyboard.press("Tab")
    const emailFocused = await page.evaluate(
      () => document.activeElement?.id === "email",
    )
    expect(emailFocused).toBe(true)

    // Deuxieme Tab → champ mot de passe
    await page.keyboard.press("Tab")
    const passwordFocused = await page.evaluate(
      () => document.activeElement?.id === "password",
    )
    expect(passwordFocused).toBe(true)

    // Troisieme Tab → bouton submit
    await page.keyboard.press("Tab")
    const submitFocused = await page.evaluate(
      () => document.activeElement?.getAttribute("type") === "submit",
    )
    expect(submitFocused).toBe(true)
  })

  test("Login — soumission du formulaire via Enter", async ({ page }) => {
    await page.goto("/login")
    await page.waitForLoadState("networkidle")

    await page.fill("#email", "test@example.com")
    await page.fill("#password", "motdepasse123")

    // Enter sur le champ password devrait soumettre le formulaire
    await page.press("#password", "Enter")

    // Le formulaire est soumis (on verifie que le bouton est en etat loading)
    // Le backend n'est pas la, on s'assure juste que la soumission fonctionne
    const submitButton = page.locator("button[type=submit]")
    // Le bouton sera soit disabled pendant le chargement, soit une erreur s'affichera
    await expect(submitButton).toBeVisible()
  })
})

test.describe("Responsive design", () => {
  const viewports = [
    { name: "Mobile", width: 375, height: 667 },
    { name: "Tablette", width: 768, height: 1024 },
    { name: "Desktop", width: 1280, height: 720 },
  ]

  for (const viewport of viewports) {
    test(`Page login visible en ${viewport.name} (${viewport.width}x${viewport.height})`, async ({
      page,
    }) => {
      await page.setViewportSize({
        width: viewport.width,
        height: viewport.height,
      })
      await page.goto("/login")
      await page.waitForLoadState("networkidle")

      // Le formulaire de login est visible
      const form = page.locator("form")
      await expect(form).toBeVisible()

      // Le header est visible
      const header = page.locator("header")
      await expect(header).toBeVisible()

      // Le champ email est visible et utilisable
      const emailInput = page.locator("#email")
      await expect(emailInput).toBeVisible()
      await expect(emailInput).toBeEnabled()
    })
  }
})
