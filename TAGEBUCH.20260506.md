# TAGEBUCH 2026-05-06

## Tagesziel (ursprünglich)

vLLM 0.17.1 → 0.20.1 Upgrade auf branch `multiquant-vllm-0.20`. Ziel:
- DeepSeek-V4-Flash lauffähig (nur in vllm 0.20+)
- 3 Minor-Versions Upstream-Bugfixes
- Bessere Base für nächste Upgrades

## Stand (Abbruch)

**Branch `multiquant-vllm-0.20` ist committed/gepusht** (`57beba5a3`),
aber das resultierende Image bringt 35B XFP-V2a nicht zum Laufen.
Stop-Entscheidung gefallen, Branch wird als Lerngrund konserviert.

## Verlauf

### 1. Audit — `MULTIQUANT_VLLM_COUPLING.md`

51 multiquant-eigene files (12,153 LOC) + 28 vllm-core files mit
echtem multiquant-Coupling. 7 HIGH-Risk Hot-Spots. Empirische Daten
für Decoupling-Strategie.

### 2. Phase 1 Merge-Probe

Branch `multiquant-vllm-0.20` von `da5818c03`, tag `pre_vllm20` als
Rollback-Anker. Erste merge-probe: 1333 auto-merged, **21 conflicts**
— deutlich besser als die 28 erwarteten.

### 3. Hunk-Verteilung — `CONFLICT_v20.md`

Pro Conflict-File: Hunk-Lokalität + Decoupling-Bewertung.
- 8 LOKAL (1-Hunk-Files, leicht extract-able)
- 4 MODERAT (kohärenter Zweck, mehrere hunks)
- 4 DURCHWACHSEN (5+ verstreute hunks, bleiben Patches)

### 4. Tier-A Refactor — gpu_worker extract

`vllm/v1/worker/gpu_worker.py` hatte 159 LOC inline-Patches für
shutdown-Cleanup. Extracted zu `vllm/multiquant/worker_hooks.py`,
gpu_worker.py jetzt 5 LOC dispatch. **End-to-end validiert** auf
v0.17 image (logs zeigen worker_hooks.py:56/108/181 + 5.5 GiB freed).

### 5. Phase 2 — 21 Conflict-Resolutions

| Strategie | Files | Outcome |
|---|---|---|
| Union (beide Seiten koexistieren) | 8 | OK |
| Upstream genommen | 4 | OK (alte multiquant-Patches obsolet, function moved, etc.) |
| Ours (multiquant) genommen | 6 | **PROBLEM** für HIGH-Files |
| Refactored merge (gpu_worker) | 1 | OK |
| AA / rename | 2 | OK |

### 6. Image Build — 4 Versuche, alle mit lessons

| Build | Dauer | Outcome | Lehre |
|---|---|---|---|
| #1 | 30 min | ✅ Image, aber alter commit (`1ca0804c3`) — Layer-Cache trotz `--no-cache` | Eigenes Dockerfile.<variant> pro branch |
| #2 | 5 min | ❌ STEP 19: `requirements/build.txt` weg → `build/cuda.txt` | Pfad-Renames upstream |
| #3 | 7h 52min | ❌ STEP 23 wheel: `vllm/vllm_flash_attn/cute` directory error | FA4 CuteDSL skip für SM12x nötig |
| #4 | 2h 50min | ✅ Image, commit `57beba5a3` korrekt | — |

### 7. Smoke-Test #2 — und der Pessimismus-Moment

Smoke crashed: `vllm_is_batch_invariant` → in v0.20 renamed zu
`addmm_batch_invariant`. **Mein Bug, nicht multiquant**: ich habe für
6 HIGH-Files `git checkout --ours` genommen statt selektiv die
multiquant-deltas auf upstream-base zu legen. → upstream's
import-update verworfen.

User-Diagnose: "wir hatten doch nur weniger echte patches, 95% war
auto-merge. hat sich was fundamental geändert oder sind die patches
scheiße?"

Antwort: Patches sind nicht scheiße. **`--ours`-Strategie für HIGH-
Files war falsch.** Saubere Strategie: upstream-base + multiquant-
delta selektiv re-applyen (1-2 Tage statt 5-10).

### 8. Stop-Entscheidung

**Heute Engineering-Aufwand:** ~9h für 0/4 lauffähige 0.20-Smoke,
0 funktionale Wins, DSV4 weiterhin nicht serv-bar.

**Was wir auf 0.17 haben (bleibt unangetastet):**
- 4/4 Modelle V2a-Quality validiert (35B/122B/Q3.6/GLM)
- mq-serve seit 4 Tagen stabil produktiv
- Alle 5 V2-aot_compile-Fixes committed
- gpu_worker Refactor (Coupling-Reduktion)

**Branch `multiquant-vllm-0.20` bleibt** als:
- Lerngrund für nächsten 0.X-Upgrade
- Alle Conflicts vermessen + dokumentiert
- Image `vllm-multiquant-v020` auf RTX (44.2 GB) bleibt für späteren retry
- Tag `pre_vllm20` für Rollback

## Lessons learned (für nächsten 0.X-Upgrade)

1. **Eigenes Dockerfile pro branch** — Layer-Cache trickst auch bei `--no-cache`
2. **Patch-Targets aktuell halten** — `ModuleName` ist jetzt `LayerName` (NGC-Patches müssen pro version updated werden)
3. **`--ours` ist falsch für HIGH-Files** mit upstream-API-Drift — selektive deltas sind nötig
4. **Auto-merge funktioniert** für 95%+ der files — Coupling-Audit sollte sich auf die echten 5% konzentrieren
5. **Realistic timeline 0.X-Upgrade**: ~2 Tage wenn man die HIGH-Strategie richtig macht, bei 21 conflicts
6. **DSV4-Verlockung war zu früh** — vllm 0.20.1 ist 1 Tag alt, NGC hat es nicht im base-image. Warten auf 0.22+ mit NGC-image-support spart Re-Discovery-Zeit

## Was bleibt offen (deferred)

- Tier-B-Refactor: `base_loader.py` (194 LOC) + `attention.py` (89 LOC) zu Hook-Modules extracten — wäre Tier-A++ für nächstes Upgrade
- xfp_moe_gemm_v17_lib.cu Wiring in JIT-Loader (Task #62, war salvaged + committed aber Performance-Win nicht aktiviert)
- DSV4-Flash test (warten auf vllm-stabilität + NGC-image-bump)
- 122B Throughput-Regression untersuchen (138 → 107 tok/s nach V2a)

## Production-Status (RTX, unangetastet)

- mq-serve: 122B XFP-V2 PreV2a, 4 Tage up, GPU 0 — produktiv
- xfp35b-bf16-serve, glm47-bf16, q36-27b-bf16: alle gestoppt nach Bench
- Branch `multiquant` (0.17 baseline): committed bis `1ca0804c3` (Refactor #1)
- Tag `pre_vllm20` (`da5818c03`): Pre-Upgrade-Rollback-Anker

## Commits heute

- `82045efad` merge: vllm v0.20.1 — 21 conflicts resolved
- `df1ab7d65` build: add Dockerfile.multiquant-v020 + build.v020.sh
- `89150410e` fix(xfp): commit salvaged xfp_moe_gemm_v17_lib.cu
- `9f3f4103a` fix(build): requirements/build.txt → build/cuda.txt
- `57beba5a3` fix(build): skip FA4 CuteDSL extension on SM12x
