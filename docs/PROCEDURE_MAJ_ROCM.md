# Procedure de Mise a Jour ROCm sur Windows 11

> Derniere migration executee : **7.1 (HIP SDK MSI) + 7.2 pre-release (wheels) -> 7.2.1 stable (wheels-only)** le **2026-04-18**.
> Projet : `greentech-intelligence` - Section BONUS B1 de `docs/CHECKLIST_SUIVI.md`.

---

## Contexte

Ce document capture la procedure complete de mise a jour de la stack ROCm sur un poste de developpement Windows 11 equipe d'une GPU AMD Radeon RX 7900 XTX (architecture RDNA3, gfx1100). Il a ete redige a l'issue de la migration vers ROCm 7.2.1 et contient les lecons apprises, notamment la transition majeure d'AMD vers une distribution Windows **wheels-only** (sans installeur MSI natif).

Le journal horodate de la migration est conserve dans `docs/ROCM_MIGRATION_LOG.md`. Le present document est la procedure reutilisable pour les migrations futures.

---

## Decouverte cle 2026 : ROCm Windows devient wheels-only

Jusqu'a ROCm 7.1, la distribution Windows comportait un installeur MSI HIP SDK qui deposait les bibliotheques HIP natives dans `C:/Program Files/AMD/ROCm/<version>/` et mettait a jour les variables d'environnement systeme (`HIP_PATH`, `HIP_PATH_<ver>`, `PATH`). Les wheels PyTorch ROCm se contentaient alors de lier dynamiquement ces bibliotheques natives.

**A partir de ROCm 7.2.1, AMD a pivote** : le repertoire `https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/` ne contient plus d'installeur MSI, uniquement des wheels Python qui embarquent leur propre runtime HIP :

- `rocm_sdk_core-7.2.1-py3-none-win_amd64.whl`
- `rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl`
- `rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl`
- `torch-2.9.1+rocm7.2.1-cp312-cp312-win_amd64.whl`
- `torchvision-0.24.1+rocm7.2.1-cp312-cp312-win_amd64.whl`
- `torchaudio-2.9.1+rocm7.2.1-cp312-cp312-win_amd64.whl`

**Implication pratique** : la migration d'une version ROCm a une autre se reduit a une edition de `pyproject.toml` et un `uv sync`. Pas d'installeur MSI, pas de reboot, pas de modification de PATH systeme.

**Test de validation** : apres desinstallation complete du HIP SDK MSI 7.1, `torch.cuda.is_available()` retourne toujours `True` avec les wheels 7.2 pre-release encore installes. Les wheels sont donc self-contained.

---

## Prerequis

- **OS** : Windows 11 Pro (64 bits)
- **GPU** : AMD Radeon (RDNA2/RDNA3 supportee, RX 7900 XTX testee)
- **Python** : 3.12 gere par UV (Astral)
- **Driver Adrenalin** : version >= 26.2.2 (requis pour ROCm 7.2.x)
- **Espace disque** : ~5 Go pour les wheels + cache UV
- **Permissions** : acces en ecriture sur le projet et le registre utilisateur (aucun droit admin Windows necessaire pour la migration wheels-only)

---

## Procedure

### Etape 1 - Verification de la version actuelle

```bash
# Version PyTorch/ROCm installee
uv run python -c "import torch; print(torch.__version__)"

# Version HIP SDK native (si MSI encore present)
& "C:\Program Files\AMD\ROCm\<version>\bin\hipconfig.exe" --version

# Detection GPU
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Etape 2 - Consultation de la version cible

- Page officielle AMD : https://rocm.docs.amd.com/projects/install-on-windows/en/latest/
- Release notes ROCm : https://rocm.docs.amd.com/en/latest/about/release-notes.html
- Repository des wheels : https://repo.radeon.com/rocm/windows/rocm-rel-<VERSION>/
- Matrice de compatibilite PyTorch : https://pytorch.org/get-started/locally/

Verifier :
- Fixes de stabilite interessants dans les release notes
- Absence (ou mesurabilite) de regressions connues
- Disponibilite des wheels pour la version Python cible (cp312 pour nous)

### Etape 3 - Sauvegarde de l'environnement

```bash
cd /path/to/greentech-intelligence

# Snapshot des packages installes
uv pip list > requirements.snapshot.$(date +%Y%m%d).txt

# Backup des fichiers de configuration
cp pyproject.toml pyproject.toml.$(date +%Y%m%d).backup
cp uv.lock uv.lock.$(date +%Y%m%d).backup
```

Ces backups permettent un rollback via :

```bash
cp pyproject.toml.<date>.backup pyproject.toml
cp uv.lock.<date>.backup uv.lock
uv sync --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio
```

### Etape 4 - Desinstallation d'un ancien HIP SDK MSI (si present)

Uniquement necessaire si une version precedente a installe le HIP SDK via MSI (<= 7.1). Les versions plus recentes n'utilisent plus de MSI.

1. Fermer tous les programmes utilisant le GPU (IDE, MLflow UI, scripts en cours).
2. Ouvrir une **PowerShell Administrateur**.
3. Executer (remplacer le GUID par celui de la version installee) :

```powershell
# Lister les HIP SDK installes
Get-WmiObject -Class Win32_Product | Where-Object { $_.Name -like "*HIP*" -or $_.Name -like "*ROCm*" } | Select-Object Name, IdentifyingNumber

# Desinstaller (exemple GUID a adapter)
msiexec /x "{GUID-HERE}" /qn
```

4. Verifier la suppression :

```powershell
# Registre
Get-ItemProperty HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\* | `
  Where-Object { $_.DisplayName -match 'HIP SDK|ROCm(?! Libraries)' } | `
  Select-Object DisplayName, DisplayVersion

# Dossier
Get-ChildItem 'C:\Program Files\AMD\ROCm\' -ErrorAction SilentlyContinue

# Variables d'environnement
[Environment]::GetEnvironmentVariable('HIP_PATH','Machine')
[Environment]::GetEnvironmentVariable('PATH','Machine') -split ';' | Where-Object { $_ -like '*ROCm*' }
```

5. **Rebooter le PC** pour que Windows propage la suppression des variables d'environnement au niveau systeme.

### Etape 5 - Mise a jour de `pyproject.toml`

Editer les 3 URLs de wheels dans `[tool.uv.sources]` + le `find-links` de `[tool.uv]`. Exemple pour une migration vers 7.2.1 :

```toml
[tool.uv.sources]
torch = [
    { url = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl", marker = "sys_platform == 'win32'" },
]
torchvision = [
    { url = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl", marker = "sys_platform == 'win32'" },
]
torchaudio = [
    { url = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl", marker = "sys_platform == 'win32'" },
]

[tool.uv]
find-links = ["https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/"]
prerelease = "allow"  # Garder par securite pour d'eventuelles deps transitives en .dev
```

**Point d'attention sur le schema de versioning** :
- Les wheels de ROCm 7.2 pre-release portent le tag `rocmsdk20260116` (date du build).
- Les wheels stables 7.2.1 portent le tag `rocm7.2.1`.
- Le format de tag change souvent entre releases : toujours verifier le nom exact des fichiers dans le repository.

### Etape 6 - Reinstallation des wheels

```bash
uv sync --reinstall-package torch \
        --reinstall-package torchvision \
        --reinstall-package torchaudio \
        --reinstall-package rocm-sdk-core \
        --reinstall-package rocm-sdk-devel \
        --reinstall-package rocm-sdk-libraries-custom
```

UV :
1. Resout les nouvelles URLs.
2. Desinstalle les anciennes versions.
3. Telecharge les nouveaux wheels (~2 Go pour torch + rocm_sdk_*).
4. Installe les nouvelles versions.

Verifier le resume final : les versions ROCm doivent avoir perdu leur suffixe `.dev0` et les versions torch leur tag `rocmsdk*` au profit du tag stable `rocm<X.Y.Z>`.

### Etape 7 - Tests de validation

**Test 1 : Sanity check de base**

```bash
uv run python -c "
import torch
print('version    :', torch.__version__)
print('cuda avail :', torch.cuda.is_available())
print('device     :', torch.cuda.get_device_name(0))
print('capability :', torch.cuda.get_device_capability(0))
"
```

Attendu : `version : 2.9.1+rocm<X.Y.Z>`, `cuda avail : True`, device name correct.

**Test 2 : Operation GPU**

```python
import torch
x = torch.randn(1024, 1024, device='cuda')
y = torch.randn(1024, 1024, device='cuda')
z = x @ y
assert z.shape == (1024, 1024)
```

**Test 3 : Inference LLM courte (Qwen3-4B pour valider bf16 sur gfx1100)**

Voir `scripts/benchmark_baseline.py` pour un test representatif du classifieur de production.

**Test 4 : Dispatcher local (chaine complete asyncio + transformers + ROCm)**

```python
import asyncio
from greentech.ai.services.llm_local import LocalQwenClient

async def t():
    c = LocalQwenClient()
    r = await c.chat_completion(
        [{'role': 'user', 'content': 'Dis bonjour en trois mots.'}],
        max_tokens=20,
    )
    print(r.choices[0].message.content)

asyncio.run(t())
```

**Test 5 : Stabilite sustained load**

Pour les migrations sensibles, lancer un entrainement LoRA court (1 epoch, 100 steps) sur un petit dataset pour verifier :
- Pas de freeze systeme
- VRAM stable
- Pas d'erreur `hipBLASLt`
- Pas de degradation de performance notable

Pour les migrations mineures, on peut s'en remettre au prochain entrainement de production reel.

### Etape 8 - Mise a jour de la documentation

- `documentation interne` section "Tech Stack" : mettre a jour la version ROCm
- `docs/PLAN_ETAPES.md` section 1.1 : torch/torchvision/torchaudio versions
- `docs/CHECKLIST_SUIVI.md` section bonus B1 : cocher les cases effectuees
- `docs/ROCM_MIGRATION_LOG.md` : ajouter une entree datee avec le diff de versions

---

## Rollback en cas d'echec

En cas de probleme (GPU non detecte, inference instable, crash), le rollback est trivial grace aux backups de l'Etape 3 :

```bash
cp pyproject.toml.<date>.backup pyproject.toml
cp uv.lock.<date>.backup uv.lock
uv sync --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio
```

Tant qu'on reste en wheels-only, aucune operation systeme (MSI, registre, PATH) n'est a inverser. Le rollback est instantane une fois les wheels redescendus.

---

## Lecons apprises

1. **Tester la desinstallation MSI avant de conclure qu'elle est necessaire.** Dans notre cas, la desinstallation complete du HIP SDK 7.1 n'a pas empeche `torch.cuda.is_available()` de retourner `True`. Les wheels `rocm_sdk_*` contiennent leur propre runtime HIP, rendant le MSI obsolete. Verifier cette hypothese en amont aurait permis de sauter les Etapes 4 et le reboot associe.
2. **Les tags de version des wheels changent entre releases.** `rocmsdk20260116` (pre-release) -> `rocm7.2.1` (stable). Toujours verifier les noms de fichiers exacts dans `rocm-rel-<version>/` avant de coder les URLs.
3. **Les release notes mentionnent parfois des regressions known.** Pour ROCm 7.2.1, une regression hipBLASLt sur certaines configs GEMM pour LLMs etait documentee. Dans notre cas, les tests d'inference bf16 sur Qwen3-4B et Qwen2.5 n'ont montre aucune degradation (28,7 tok/s). Mesurer reste necessaire pour chaque cas d'usage.
4. **Conserver `prerelease = "allow"` dans `[tool.uv]`.** Meme apres migration vers une version stable, des dependances transitives peuvent rester en `.dev0`. Supprimer cette option peut casser la resolution UV.
5. **Le HIP_PATH residuel dans l'environnement utilisateur n'est pas nettoye par Windows apres desinstallation du MSI.** Il faut soit le supprimer manuellement, soit compter sur le reboot pour sa propagation au niveau Machine. Ce nettoyage n'est pas bloquant tant que `torch.cuda.is_available() == True`.

---

## References

- [AMD HIP SDK for Windows](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html)
- [HIP SDK for Windows - ROCm Documentation](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/)
- [PyTorch via PIP installation (ROCm Windows)](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html)
- [Repository des wheels ROCm](https://repo.radeon.com/rocm/windows/)
- [ROCm Release Notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html)
