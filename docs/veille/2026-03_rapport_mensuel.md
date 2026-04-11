# Rapport Mensuel de Veille - Mars 2026

> **Periode** : Du 1er mars au 31 mars 2026
> **Projet** : GreenTech Intelligence

---

## Vue d'Ensemble du Mois

Mars 2026 marque un mois charniere pour l'ecosysteme Green IT et les outils MLOps utilises dans le projet GreenTech Intelligence. Trois tendances majeures se degagent :

1. **Maturation du support AMD ROCm sur Windows** : ROCm 7.2 stabilise le support PyTorch natif pour les GPU Radeon et APU Ryzen AI, validant definitivement notre choix d'infrastructure AMD. Le passage du stade "preview" a un support officiel leve un risque technique majeur du projet.

2. **Evolution reglementaire europeenne** : L'adoption de la directive Omnibus (CSRD simplifiee) et la preparation du Data Centre Energy Efficiency Package par la Commission europeenne redessinent le cadre reglementaire. Le projet reste pertinent meme avec le relevement des seuils CSRD, car la demande de transparence environnementale dans le numerique s'intensifie.

3. **Convergence MLOps/GenAI** : MLflow 3.10 unifie le suivi ML classique et GenAI dans un meme outil, refletant exactement la dualite de notre projet (classification DeBERTa + resume BART). Les outils de suivi carbone (CodeCarbon 3.2.3) et d'evaluation energetique (AI Energy Score) gagnent en maturite.

### Statistiques

- **Syntheses hebdomadaires realisees** : 4
- **Articles Inoreader consultes** : 156
- **Recherches approfondies** : 14
- **Issues GitHub creees** : 5
- **Bibliotheques mises a jour** : 3 (CodeCarbon, MLflow, PyTorch/ROCm)

---

## Top 10 des Actualites du Mois

### 1. ROCm 7.2 : support PyTorch natif stable sur Windows pour GPU Radeon

- **Date** : Mars 2026
- **Source** : [AMD Blog](https://www.amd.com/en/blogs/2025/the-road-to-rocm-on-radeon-for-windows-and-linux.html)
- Support des RX 7000/9000 et APU Ryzen AI. Modeles LLM et Diffusion valides. Driver 26.1.1 requis.
- Impact : 🔴 Majeur — Valide notre architecture materielle (RX 7900 XTX + Ryzen AI 9 HX 370).

### 2. MLflow 3.10.0 : multi-workspace, trace cost tracking et separation GenAI/ML

- **Date** : Fevrier-Mars 2026
- **Source** : [MLflow Releases](https://mlflow.org/releases)
- Organisations multi-workspace, calcul automatique des couts d'inference LLM, redesign UI GenAI vs ML classique.
- Impact : 🔴 Majeur — Impacte directement notre pipeline MLOps (`ai/mlops/tracking.py`).

### 3. Directive Omnibus UE : simplification de la CSRD adoptee

- **Date** : 24 fevrier 2026 (impact mars)
- **Source** : [Gibson Dunn](https://www.gibsondunn.com/omnibus-simplification-of-eu-sustainability-rules-csrd-and-csddd-enacted/)
- Seuil releve a 1000 employes et 450M EUR CA. PME exclues. Report des echeances "vague 2/3" a 2028.
- Impact : 🟡 Modere — Le cadre evolue mais ne remet pas en cause la pertinence du projet.

### 4. CodeCarbon 3.2.3 : ameliorations continues du suivi carbone ML

- **Date** : 22 fevrier 2026
- **Source** : [CodeCarbon PyPI](https://pypi.org/project/codecarbon/) | [GitHub](https://github.com/mlco2/codecarbon)
- Quatrieme release en 4 mois (3.2.0 a 3.2.3). Precision amelioree des mesures. Integration MLflow via `mlflow-emissions-sdk`.
- Impact : 🔴 Majeur — Composant central du module `ai/mlops/carbon.py`.

### 5. Hugging Face AI Energy Score : notation energetique des modeles IA

- **Date** : Actualise en continu (reconnu AI Action Summit Paris 2025)
- **Source** : [Hugging Face AI Energy Score](https://huggingface.github.io/AIEnergyScore/)
- Systeme de notation par etoiles de l'efficacite energetique des modeles par tache. Pilote par Sasha Luccioni et Margaret Mitchell.
- Impact : 🟡 Modere — Outil d'aide a la selection de modeles pour notre argumentaire Green IT.

### 6. EU Data Centre Energy Efficiency Package prevu au T1 2026

- **Date** : T1 2026
- **Source** : [White & Case](https://www.whitecase.com/insight-alert/data-centres-and-energy-consumption-evolving-eu-regulatory-landscape-and-outlook-2026)
- La Commission europeenne prepare un paquet legislatif pour integrer durablement la consommation electrique des data centres dans le systeme energetique europeen. Un Cloud and AI Development Act est egalement attendu.
- Impact : 🟡 Modere — Renforce le contexte reglementaire justifiant le projet.

### 7. Empreinte carbone de l'IA : 32-80 Mt CO2 en 2025 selon les estimations

- **Date** : 2025-2026
- **Source** : [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2666389925002788) | [Cornell](https://news.cornell.edu/stories/2025/11/roadmap-shows-environmental-impact-ai-data-center-boom)
- Empreinte carbone IA estimee entre 32,6 et 79,7 Mt CO2. Consommation electrique des data centres projetee a 1000+ TWh (vs 460 TWh en 2022). Feuille de route pour reduire l'impact de 73% (CO2) et 86% (eau).
- Impact : 🟢 Mineur — Chiffres cles pour la documentation et la soutenance.

### 8. Transformers v5 : refonte de l'inference avec kernels specialises

- **Date** : 2025-2026
- **Source** : [Hugging Face Blog](https://huggingface.co/blog/transformers-v5)
- Kernels specialises, batching continu, attention paginee, `transformers serve` compatible OpenAI API. TGI passe en maintenance.
- Impact : 🟡 Modere — Migration a planifier pour optimiser l'inference BART et DeBERTa.

### 9. LoRAFusion et optimisations QLoRA pour GPU consumer

- **Date** : 2025-2026
- **Source** : [arXiv - LoRAFusion](https://arxiv.org/html/2510.00206v1) | [arXiv - Profiling Consumer GPUs](https://arxiv.org/abs/2509.12229)
- Kernels LoRAFusion pour poids 4-bit. Paged optimizers : +25% throughput. fp16 > bf16 sur GPU consumer. Pattern enterprise : base unique + adaptateurs LoRA echangeables.
- Impact : 🟡 Modere — Optimisations directement applicables a notre fine-tuning DeBERTa.

### 10. FastAPI 0.135.x : SSE, streaming et Starlette 1.0

- **Date** : Mars 2026
- **Source** : [FastAPI Release Notes](https://fastapi.tiangolo.com/release-notes/)
- Support Server-Sent Events, streaming JSON Lines, strict Content-Type checking, support Starlette 1.0.0+. Python 3.8 abandonne.
- Impact : 🟡 Modere — Le support SSE ouvre la possibilite de streaming temps reel pour les resultats d'analyse (`api/routes/analyze.py`).

---

## Analyses Approfondies

### Sujet 1 : ROCm 7.2 et la maturite de l'ecosysteme PyTorch sur AMD

**Contexte**

L'ecosysteme de calcul GPU pour le deep learning a longtemps ete domine par NVIDIA avec CUDA. AMD a progressivement developpe ROCm comme alternative open-source, mais le support Windows restait experimental. L'annee 2025-2026 marque un tournant :

- **ROCm 6.4.4** (2025) : Premier preview public de PyTorch sur Windows pour Radeon RX 7000/9000.
- **ROCm 7.1** (2025) : Elargissement du support avec des guides communautaires pour les GPU non officiellement supportes.
- **ROCm 7.2** (mars 2026) : Support officiel stable. Driver 26.1.1 integre. Validation de modeles LLM (DeepSeek, Llama) et Diffusion (Flux, SD3.5).

**Analyse technique pour le projet**

Notre infrastructure repose sur deux machines AMD :

| Machine | GPU/APU | VRAM | Usage prevu | Statut ROCm 7.2 |
|---------|---------|------|-------------|------------------|
| PC Fixe | RX 7900 XTX (RDNA 3) | 24 Go | Entrainement, fine-tuning LoRA | Supporte nativement |
| PC Portable | Ryzen AI 9 HX 370 (NPU integre) | Memoire partagee | Inference, tests | Support APU confirme |

Points cles pour notre usage :
- **Fine-tuning DeBERTa-v3-base** (~86M parametres) : Le budget VRAM de 24 Go est largement suffisant pour LoRA (rank 16-32) en fp16.
- **Inference BART-large-CNN** (~406M parametres) : Possible en local sur la RX 7900 XTX, alternative a l'API Hugging Face pour reduire la latence et les couts.
- **NPU Ryzen AI** : Potentiel inexploite pour l'acceleration de l'inference en edge computing. A explorer dans un second temps.

**Risques identifies et mitigation** :
- Certains operateurs PyTorch peuvent ne pas etre optimises pour ROCm — prevoir des tests exhaustifs sur nos modeles specifiques.
- Le support ROCm evolue vite, rester en veille sur les release notes AMD pour les regressions potentielles.

**Ressources** :
- [AMD ROCm Documentation](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/index.html)
- [PyTorch Installation ROCm Windows](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html)
- [AMD PyTorch on Windows Release Notes](https://www.amd.com/en/resources/support-articles/release-notes/RN-AMDGPU-WINDOWS-PYTORCH-7-1-1.html)

---

### Sujet 2 : Evolution du cadre reglementaire Green IT en Europe

**Contexte**

Le paysage reglementaire europeen en matiere de durabilite numerique evolue sur plusieurs fronts en 2026 :

**1. Directive Omnibus (CSRD simplifiee)**

Le Conseil de l'UE a adopte le texte final le 24 fevrier 2026. Changements majeurs :
- Seuil d'applicabilite releve : >1000 employes ET >450M EUR de CA (vs >250 employes auparavant)
- PME exclues du perimetre
- Report des echeances "vague 2/3" a 2028 (exercice 2027)
- Standards ESRS revises attendus pour l'exercice 2027

Source : [EU CSRD Omnibus - Official Journal](https://www.insideenergyandenvironment.com/2026/02/eu-csddd-csrd-omnibus-published-in-official-journal-transposition-delegated-acts-and-guidelines-are-next/)

**2. Data Centre Energy Efficiency Package**

La Commission europeenne prepare au T1 2026 :
- Un paquet legislatif pour integrer durablement les data centres dans le systeme energetique
- Un Strategic Roadmap on Digitalisation and AI for the Energy Sector
- Un Cloud and AI Development Act prevu T4 2025/T1 2026

Source : [White & Case - Data Centres EU Regulatory Landscape](https://www.whitecase.com/insight-alert/data-centres-and-energy-consumption-evolving-eu-regulatory-landscape-and-outlook-2026)

**3. Loi REEN et RGESN en France**

Le cadre francais continue de se renforcer :
- La loi REEN (Reduction de l'Empreinte Environnementale du Numerique) encadre l'eco-conception des services numeriques
- Le RGESN (Referentiel General d'Ecoconception de Services Numeriques) fournit 79 criteres d'evaluation
- Le barometre 2025 d'eco-conception web revele une urgence ecologique dans le secteur

Sources : [RGESN Gouvernement](https://ecoresponsable.numerique.gouv.fr/publications/referentiel-general-ecoconception/) | [Barometre eco-conception 2025](https://www.natural-net.fr/blog-agence-web/2025/11/05/eco-conception-web-et-digitale-le-barometre-2025-revele-une-urgence-ecologique.html)

**4. Directive EmpCo (Green Claims)**

A partir du 27 septembre 2026, les affirmations environnementales generiques ("ecologique", "durable", "vert") devront etre etayees par des preuves concretes.

Source : [GFAW - 2026 Sustainability Regulations](https://gfaw.eu/en/2026-a-year-of-sustainability-regulations-what-companies-need-to-know/)

**Implications pour GreenTech Intelligence**

Le projet se positionne a la croisee de ces reglementations :
- **Classification d'articles Green IT** : Aide les entreprises a identifier le contenu reglementaire pertinent
- **Eco-conception native** : Le projet applique les principes du RGESN (mesure d'empreinte via CodeCarbon, optimisation des modeles via LoRA, choix de modeles efficaces)
- **Transparence** : Le tracking MLflow + CodeCarbon documente l'empreinte carbone du pipeline IA

---

### Sujet 3 : Suivi carbone ML — CodeCarbon, AI Energy Score et bonnes pratiques

**Contexte**

Le suivi de l'empreinte carbone du machine learning est passe d'une pratique de niche a une exigence de plus en plus attendue. Trois outils/initiatives structurent ce domaine en 2026 :

**1. CodeCarbon 3.2.x**

CodeCarbon mesure les emissions CO2 en estimant la consommation electrique (GPU + CPU + RAM) et en appliquant l'intensite carbone de la region de calcul. Les versions recentes (3.2.0 a 3.2.3 entre novembre 2025 et fevrier 2026) montrent un rythme de developpement soutenu.

Points forts pour notre projet :
- Integration Python native (`from codecarbon import EmissionsTracker`)
- Metriques contextualisees (equivalents concrets : km en voiture, heures de TV, etc.)
- Integration MLflow via `mlflow-emissions-sdk` de Dataroots
- Dashboard web pour visualiser les emissions

Source : [CodeCarbon GitHub](https://github.com/mlco2/codecarbon)

**2. AI Energy Score (Hugging Face)**

L'initiative de Sasha Luccioni et Margaret Mitchell cree un systeme de notation par etoiles pour l'efficacite energetique des modeles par tache. Points cles :
- Notation comparative entre modeles pour une meme tache (ex : resume de texte)
- Reconnu par le gouvernement francais (AI Action Summit Paris 2025)
- Donnees ouvertes et reproductibles

Source : [AI Energy Score](https://huggingface.github.io/AIEnergyScore/)

**3. Impact macro de l'IA**

Les chiffres 2025-2026 sont eloquents :
- 32,6 a 79,7 Mt CO2 pour les systemes IA en 2025
- Consommation electrique des data centres : 1000+ TWh projetee (vs 460 TWh en 2022)
- 56% de l'electricite des data centres provient d'energies fossiles
- Une feuille de route optimiste montre qu'on pourrait reduire l'impact de 73% (CO2) avec un deploiement intelligent

Sources : [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2666389925002788) | [Carbon Brief](https://www.carbonbrief.org/ai-five-charts-that-put-data-centre-energy-use-and-emissions-into-context/)

**Implementation dans GreenTech Intelligence**

Notre approche multi-niveaux :

```
Pipeline MLOps
├── CodeCarbon (3.2.3) ─── Mesure emissions par run/experience
├── MLflow (3.10.1) ────── Centralisation metriques + couts
├── AI Energy Score ────── Selection modeles (BART, DeBERTa)
└── Prometheus/Grafana ─── Monitoring production temps reel
```

Actions concretes :
- Chaque experience de training/fine-tuning declenche un `EmissionsTracker`
- Les emissions sont loggees comme metrics MLflow
- Le dashboard Grafana affiche les emissions cumulees en production
- Le rapport de soutenance inclura un bilan carbone complet du projet

---

## Bilan des Actions Entreprises

### Actions Realisees

| Action | Origine veille | Module | Statut |
|--------|---------------|--------|--------|
| Mise a jour CodeCarbon vers 3.2.3 | Synthese S1 mars | `ai/mlops/carbon.py` | 🚧 En cours |
| Mise a jour MLflow vers 3.10.1 | Synthese S1 mars | `ai/mlops/tracking.py` | 🚧 En cours |
| Verification compatibilite ROCm 7.2 | Synthese S1 mars | Infrastructure | ⏳ En attente |
| Consultation AI Energy Score pour BART/DeBERTa | Synthese S1 mars | Documentation | ⏳ En attente |
| Veille RGESN pour eco-conception de l'API | Synthese S2 fevrier | `api/` | ✅ Termine |

### Integrations Concretes au Projet

- **CodeCarbon** : Integre dans le module `ai/mlops/carbon.py` pour le suivi des emissions lors de l'entrainement et de l'inference. Mesure automatique GPU + CPU + RAM.
- **MLflow** : Module `ai/mlops/tracking.py` pour le tracking des experiences. Configuration des experiments "classification-deberta" et "summarization-bart".
- **LoRA/PEFT** : Module `ai/models/training.py` pour le fine-tuning efficient de DeBERTa-v3. Strategie LoRA avec rank configurable.
- **BART-large-CNN** : Module `ai/services/summarizer.py` pour le resume d'articles via l'API Hugging Face Inference. Choix valide par les benchmarks ROUGE (score 43.98%).
- **DeBERTa-v3** : Module `ai/models/classifier.py` pour la classification Green IT. Confirme comme SOTA pour la classification de texte par les etudes recentes (fevrier 2026).

### Choix techniques valides par la veille

| Choix | Validation |
|-------|-----------|
| BART-large-CNN pour le resume | Meilleur ROUGE score (43.98%) parmi BART/T5/PEGASUS selon [etude comparative MDPI](https://www.mdpi.com/1999-5903/17/9/389) |
| DeBERTa-v3 pour la classification | "Fine-tuned transformer language models define the state of the art" — BERT variants (RoBERTa, DeBERTa) restent les meilleurs selon [revue comparative 2025](https://arxiv.org/html/2204.03954v6) |
| CodeCarbon pour le suivi carbone | Outil de reference, maintenu activement (4 releases en 4 mois), reconnu par Mila et HuggingFace |
| LoRA pour le fine-tuning | "90-95% full fine-tuning quality" avec 10-20x moins de memoire — [benchmarks 2025](https://introl.com/blog/fine-tuning-infrastructure-lora-qlora-peft-scale-guide-2025) |
| AMD ROCm pour le GPU | Support officiel stable Windows + PyTorch en mars 2026 — risque technique leve |

---

## Axes d'Amelioration pour le Mois Suivant

### Thematiques a Approfondir

- **Quantization post-training** : Explorer GPTQ et AWQ pour optimiser l'inference DeBERTa en production
- **Transformers v5 migration** : Planifier la migration pour beneficier des kernels specialises et du nouveau systeme de serving
- **Prometheus 3.x native histograms** : Exploiter les histogrammes natifs pour des metriques plus fines avec moins de series temporelles
- **DVC + lakeFS** : Surveiller l'evolution post-acquisition de DVC par lakeFS (novembre 2025) et evaluer l'impact sur notre pipeline de versioning de donnees
- **Green claims EmpCo** : Preparer notre documentation pour la conformite avec la directive sur les affirmations environnementales (applicable septembre 2026)

### Actions Prioritaires

- [ ] Finaliser la mise a jour ROCm 7.2 et valider le fine-tuning DeBERTa sur la RX 7900 XTX
- [ ] Configurer les workspaces MLflow 3.10 (classification / summarization)
- [ ] Benchmarker fp16 vs bf16 pour LoRA DeBERTa sur ROCm 7.2
- [ ] Evaluer Transformers v5 pour l'inference BART-large-CNN
- [ ] Integrer le cost tracking MLflow pour l'API Hugging Face Inference
- [ ] Documenter le bilan carbone cumule du projet avec CodeCarbon 3.2.3
- [ ] Creer un dashboard Grafana dedie aux metriques Green IT du pipeline

---

## Ressources Cles du Mois

### Articles de Reference

- [Data Centres and Energy Consumption: EU Regulatory Landscape 2026](https://www.whitecase.com/insight-alert/data-centres-and-energy-consumption-evolving-eu-regulatory-landscape-and-outlook-2026) — White & Case LLP
- [2026: A Year of Sustainability Regulations](https://gfaw.eu/en/2026-a-year-of-sustainability-regulations-what-companies-need-to-know/) — GFAW
- [Omnibus CSRD Simplification](https://www.gibsondunn.com/omnibus-simplification-of-eu-sustainability-rules-csrd-and-csddd-enacted/) — Gibson Dunn
- [The Carbon and Water Footprints of Data Centers](https://www.sciencedirect.com/science/article/pii/S2666389925002788) — ScienceDirect
- [AI: Five Charts on Data Centre Energy Use](https://www.carbonbrief.org/ai-five-charts-that-put-data-centre-energy-use-and-emissions-into-context/) — Carbon Brief
- [Roadmap on AI Data Center Environmental Impact](https://news.cornell.edu/stories/2025/11/roadmap-shows-environmental-impact-ai-data-center-boom) — Cornell University
- [Small is Sufficient: Reducing World AI Energy Consumption](https://arxiv.org/html/2510.01889v1) — arXiv
- [Eco-conception web : Barometre 2025](https://www.natural-net.fr/blog-agence-web/2025/11/05/eco-conception-web-et-digitale-le-barometre-2025-revele-une-urgence-ecologique.html) — Natural-Net
- [Comparative Study: PEGASUS, BART, T5 for Summarization](https://www.mdpi.com/1999-5903/17/9/389) — MDPI Future Internet
- [DeBERTa for Stance Detection (Feb 2026)](https://doi.org/10.62411/faith.3048-3719-168) — Journal of Future AI Technologies
- [Fine-Grained Human-AI Text Classification with DeBERTa](https://ceur-ws.org/Vol-4038/paper_300.pdf) — CEUR Workshop Proceedings

### Repositories GitHub

- [CodeCarbon](https://github.com/mlco2/codecarbon) — v3.2.3, suivi emissions CO2
- [MLflow](https://github.com/mlflow/mlflow) — v3.10.1, tracking ML/GenAI
- [Microsoft DeBERTa](https://github.com/microsoft/DeBERTa) — Architecture classifieur
- [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn) — Modele de resume
- [Hugging Face AI Energy Score](https://huggingface.github.io/AIEnergyScore/) — Notation energetique
- [BART Summarization Implementation](https://github.com/carlosrod723/BART-Abstract-Text-Summarization) — Exemple d'implementation

### Documentations Officielles

- [AMD ROCm on Radeon and Ryzen](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/index.html) — Installation et configuration
- [PyTorch via PIP sur Windows (ROCm)](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html) — Guide d'installation
- [MLflow 3.0 Documentation](https://mlflow.org/docs/latest/mlflow-3/) — Nouvelles fonctionnalites
- [MLflow New Features](https://mlflow.org/docs/latest/new-features/index.html) — Changelog detaille
- [RGESN](https://ecoresponsable.numerique.gouv.fr/publications/referentiel-general-ecoconception/) — Referentiel eco-conception (79 criteres)
- [Transformers v5](https://huggingface.co/blog/transformers-v5) — Migration et nouvelles API
- [FastAPI Release Notes](https://fastapi.tiangolo.com/release-notes/) — Dernieres fonctionnalites
- [Grafana Observability Survey 2025](https://grafana.com/observability-survey/2025/) — Tendances monitoring
- [EU Green Digital Sector](https://digital-strategy.ec.europa.eu/en/policies/green-digital) — Strategie numerique europeenne
- [EU Green Cloud and Data Centres](https://digital-strategy.ec.europa.eu/en/policies/green-cloud) — Politique cloud durable

---

**Prochain rapport mensuel** : 30 avril 2026

