/**
 * Corrections d'accessibilite pour la documentation Sphinx + theme Furo.
 *
 * 1. Le theme Furo applique role="heading" sur les captions de la sidebar
 *    mais n'ajoute pas aria-level (viole WCAG 4.1.2, axe-core
 *    "aria-required-attr").
 * 2. L'extension MyST-Parser "tasklist" genere des <input type="checkbox">
 *    sans label associe (viole WCAG 3.3.2 / 4.1.2, axe-core "label").
 *    On ajoute un aria-label deduit du texte qui suit la checkbox.
 *
 * Ces corrections s'appliquent au chargement de la page.
 */
(function () {
  "use strict"

  function fixAriaLevels() {
    // Captions de la sidebar (role="heading" sans aria-level)
    var captions = document.querySelectorAll('.caption[role="heading"]')
    captions.forEach(function (caption) {
      if (!caption.hasAttribute("aria-level")) {
        caption.setAttribute("aria-level", "3")
      }
    })
  }

  function fixTasklistLabels() {
    // Checkboxes issues de l'extension tasklist de MyST-Parser
    var checkboxes = document.querySelectorAll(
      'input[type="checkbox"].task-list-item-checkbox, ' +
      'li.task-list-item > input[type="checkbox"]'
    )
    checkboxes.forEach(function (checkbox) {
      if (checkbox.hasAttribute("aria-label") || checkbox.labels.length > 0) {
        return
      }
      // Le texte associe est generalement dans le li parent
      var parent = checkbox.closest("li")
      var texte = parent ? parent.textContent.trim().slice(0, 150) : "Tache"
      checkbox.setAttribute("aria-label", texte || "Tache")
      // Les cases sont en lecture seule dans la doc statique
      if (!checkbox.hasAttribute("aria-readonly")) {
        checkbox.setAttribute("aria-readonly", "true")
      }
    })
  }

  function fixScrollablePre() {
    // Les blocs <pre> contenant du code peuvent avoir du scroll horizontal
    // sans etre focusables. On leur ajoute tabindex="0" pour permettre la
    // navigation clavier (WCAG 2.1.1, axe-core "scrollable-region-focusable").
    var pres = document.querySelectorAll(".highlight pre")
    pres.forEach(function (pre) {
      if (!pre.hasAttribute("tabindex")) {
        pre.setAttribute("tabindex", "0")
      }
    })
  }

  function applyFixes() {
    fixAriaLevels()
    fixTasklistLabels()
    fixScrollablePre()
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", applyFixes)
  } else {
    applyFixes()
  }
})()
