# vLLM 0.17.1 → 0.20.1 Merge-Conflict Report

> Stand: 2026-05-05, Branch `multiquant-vllm-0.20`
> 21 Conflicts vermessen, klassifiziert, Decoupling-Potenzial bewertet.
> Datenquelle: `git diff :1:<f> :2:<f>` (multiquant-Δ) und `git diff :1:<f> :3:<f>` (upstream-Δ).

## Master-Statistik

| File | mq+ | mq− | us-Δ | Klasse | Decoupling-Potenzial |
|---|---:|---:|---:|---|---|
| `csrc/torch_bindings.cpp` | 5 | 1 | 310 | LOW (pybind add) | **Hoch** — pybind-add via plugin |
| `tests/v1/core/test_kv_cache_utils.py` | 146 | 6 | 83 | LOW (test-only) | n/a (test) |
| `vllm/config/cache.py` | 11 | 1 | 83 | LOW (config field) | **Mittel** — eigener config-class möglich |
| `vllm/model_executor/layers/attention/attention.py` | 89 | 1 | **320** | **HIGH** | **Niedrig** — KV-quant tief in forward |
| `vllm/model_executor/layers/conv.py` | 13 | 2 | 9 | LOW | **Hoch** — minor patch entkoppelbar |
| `vllm/model_executor/layers/fused_moe/fused_moe.py` | 42 | 12 | **266** | **HIGH** | **Mittel** — Hooks könnten via subclass |
| `vllm/model_executor/layers/fused_moe/layer.py` | 87 | 1 | **396** | **HIGH** | **Niedrig** — FusedMoE base ist die API |
| `vllm/model_executor/layers/fused_moe/router/base_router.py` | 97 | 1 | 159 | **HIGH** (RIY) | **Mittel** — RIY als optional router-stage |
| `vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py` | 3 | 1 | 108 | LOW | **Hoch** — minor RIY-hook |
| `vllm/model_executor/layers/fused_moe/router/router_factory.py` | 8 | 1 | 55 | LOW | **Hoch** — registry-add |
| `vllm/model_executor/layers/quantization/__init__.py` | 21 | 1 | 44 | MED | **Hoch** — Quant-Plugin-API existiert |
| `vllm/model_executor/layers/vocab_parallel_embedding.py` | 15 | 1 | 21 | LOW | **Mittel** — embed-quant-aware Hook |
| `vllm/model_executor/model_loader/base_loader.py` | **194** | 4 | 23 | **HIGH** | **Niedrig** — process_weights-Sequence intrinsisch |
| `vllm/model_executor/models/qwen3_5_mtp.py` | 36 | 122 | 26 | MED (own model) | n/a — eigene model-class |
| `vllm/model_executor/models/qwen3_next.py` | 18 | 1 | **1062** | MED-LOW | **Hoch** — NVTX wrapper, separable |
| `vllm/utils/torch_utils.py` | 11 | 1 | 235 | MED | **Hoch** — direct_register_custom_op upstream stable |
| `vllm/v1/attention/backend.py` | 4 | 2 | 92 | LOW | **Hoch** — backend-base 4-line patch |
| `vllm/v1/attention/backends/registry.py` | 10 | 1 | 3 | LOW | **Hoch** — registry-add 1-liner |
| `vllm/v1/attention/backends/turboquant_attn.py` | AA | (new) | (new) | n/a | own file (no patch) |
| `vllm/v1/core/kv_cache_utils.py` | 139 | 36 | **533** | **HIGH** | **Niedrig** — KV-cache-utils intrinsisch |
| `vllm/v1/worker/gpu_worker.py` | **159** | 1 | 70 | **HIGH** | **Niedrig** — Shutdown-Lifecycle intrinsisch |

## Sub-Reports per File

### 1. `csrc/torch_bindings.cpp` (LOW)
- **multiquant**: 5 lines added — pybind-Bindings für `turboquant_round_trip`
- **upstream**: 310 lines geändert — neue ops registriert
- **Decoupling**: ✅ **Hoch** — vllm hat plugin-mechanism für custom ops via `torch.library.custom_op`. Statt `torch_bindings.cpp`-Patch könnte `vllm.multiquant._cpp_ops` als eigenes pybind-Modul kommen.
- **Aufwand:** 2-3h für Migration zu standalone-pybind

### 2. `vllm/v1/attention/backends/registry.py` (LOW, 1-liner)
- **multiquant**: 10 lines = 2-3 Backend-Eintragungen
- **upstream**: 3 lines (minimal change)
- **Decoupling**: ✅ **Hoch** — vllm hat schon `register_attention_backend()` plugin-API. Multiquant-backends könnten sich via `vllm.attention.backends.register("multiquant", ...)` aus `vllm/multiquant/__init__.py` selbst eintragen.
- **Aufwand:** 1h

### 3. `vllm/v1/attention/backend.py` (LOW)
- **multiquant**: 4 lines — vermutlich Hook-Punkt
- **upstream**: 92 lines change — base-class refactor
- **Decoupling**: ✅ **Hoch** wenn der 4-line-Hook sich als subclass-override darstellen lässt
- **Aufwand:** 1h

### 4. `vllm/model_executor/layers/conv.py` (LOW)
- **multiquant**: 13 lines, **upstream**: 9 lines — beide klein
- **Decoupling**: ✅ **Hoch** — minor patch wahrscheinlich für TQ/MultiQuant-conv. Re-applicable als monkey-patch oder subclass.

### 5. `vllm/model_executor/layers/fused_moe/router/router_factory.py` (LOW)
- **multiquant**: 8 lines — vermutlich Registry-Add für RIY-router
- **Decoupling**: ✅ **Hoch** — falls Factory-API `register(name, factory)` hat. Sonst monkey-patch.

### 6. `vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py` (LOW)
- **multiquant**: 3 lines — RIY-Trigger?
- **Decoupling**: ✅ **Hoch** — sehr kleiner Hook, einfach in router-base subclass abzuhandeln.

### 7. `vllm/v1/core/kv_cache_utils.py` (HIGH)
- **multiquant**: 139+/-36 lines — substantielle Patches
- **upstream**: 533 lines change — massive refactor
- **Coupling**: KV-Cache-Utils sind tief in der request-scheduler-Pipeline.
- **Decoupling**: ❌ **Niedrig** — multiquant braucht spezielle KV-cache-Berechnung für FP8/INT8/RIY. KV-cache-utils ist nicht plugin-fähig in vllm.
- **Strategie:** Bei jedem upstream-refactor neu portieren. Pro Release ~2-4h.

### 8. `tests/v1/core/test_kv_cache_utils.py` (LOW, test)
- **multiquant**: 146 lines neue test cases
- **Decoupling**: ✅ — Tests gehören sowieso in `tests/multiquant/` statt `tests/v1/core/`. Move ist ein einfacher Refactor.

### 9. `vllm/config/cache.py` (LOW)
- **multiquant**: 11 lines — neue config-fields (vermutlich `xfp_*`, `riy_*`)
- **Decoupling**: ✅ **Mittel** — vllm-config ist erweiterbar via dataclass. Eigene Config-Class `MultiquantCacheConfig` möglich, in `cache.py` nur 1 line "if multiquant: extra_config = ...".

### 10. `vllm/model_executor/layers/quantization/__init__.py` (MED, registry)
- **multiquant**: 21 lines — Registry-Add für `autoround_rtn`, `xfp`, `tq*`, etc.
- **upstream**: 44 lines — neue quant-methods upstream
- **Decoupling**: ✅ **Hoch** — vllm hat `register_quantization_config()` API. Multiquant-quants registrieren sich aus `vllm/multiquant/__init__.py` selbst. **Eliminiert diesen Patch komplett.**
- **Aufwand:** 1-2h, **Top-Priorität-Refactor**

### 11. `vllm/model_executor/layers/vocab_parallel_embedding.py` (LOW)
- **multiquant**: 15 lines — vermutlich embed-quant-Attribute oder XFP-Skip
- **Decoupling**: ✅ **Mittel** — könnte als layer-attribute-check abstrahiert werden.

### 12. `vllm/utils/torch_utils.py` (MED)
- **multiquant**: 11 lines — vermutlich `direct_register_custom_op`-Verwandtes oder torch-helper
- **upstream**: 235 lines change — diese Util-Datei ist hot-zone bei vllm
- **Decoupling**: ✅ **Hoch** — vllm.utils ist generisch, multiquant-Helper können nach `vllm.multiquant.utils.torch_helpers` migrieren.

### 13. `vllm/model_executor/models/qwen3_next.py` (MED-LOW)
- **multiquant**: 18 lines — `VLLM_SKIP_GDN_WARMUP` env-flag (siehe gerade gesehen)
- **upstream**: **1062 lines change** — komplettes refactor des qwen3_next-Modells (multiquant-Code zwischen line 197-918 überlappt mit upstream-Blöcken)
- **Decoupling**: ✅ **Hoch** — die 18 Zeilen für VLLM_SKIP_GDN_WARMUP sind:
  - 1 if-block in `_warmup_prefill_kernels()`
  - Nichts anderes
  Dieses if-block könnte über monkey-patch oder via Subklassen-Override implementiert werden, ohne core-Patch.
- **Aufwand:** 30 min — **Top-Priorität-Refactor**, eliminiert massive Re-Application bei jedem qwen3_next-upstream-update.

### 14. `vllm/model_executor/models/qwen3_5_mtp.py` (eigene Datei)
- **multiquant**: 36+/-122 = NETZ verkleinert
- **upstream**: 26 lines change
- **Status**: Multiquant hat eigene Qwen3_5MTP-Implementation. Falls upstream das Modell-class ändert, brauchts Re-Sync.
- **Decoupling**: n/a — eigene model-class, kein "patch in vllm/" sondern eigener File.

### 15. `vllm/model_executor/layers/fused_moe/router/base_router.py` (HIGH, RIY)
- **multiquant**: 97 lines — RIY% expert masking infrastructure
- **upstream**: 159 lines change — router-base-class refactor
- **Coupling**: RIY% (Reduced Inference Y%) braucht Hook IN den Routing-Forward — kann Experts maskieren bevor allokiert wird.
- **Decoupling**: ✅ **Mittel** — könnte als optional `router_pre_dispatch_hook` API in vllm sein (upstream PR notwendig). Sonst lokal als BaseRouter-Subclass.
- **Aufwand:** 4-8h für sauberen Subclass-Pattern oder upstream PR.

### 16. `vllm/model_executor/layers/fused_moe/fused_moe.py` (HIGH)
- **multiquant**: 42+/-12 lines
- **upstream**: 266 lines — fused_moe wird oft refactored
- **Decoupling**: ✅ **Mittel** — wenn die Hooks sich als override in `MultiquantFusedMoE` darstellen lassen (subclass), sonst patch.

### 17. `vllm/model_executor/layers/fused_moe/layer.py` (HIGH)
- **multiquant**: 87 lines — FusedMoE.apply(), forward, etc. patches
- **upstream**: 396 lines — größte Änderung in MoE
- **Coupling**: Multiquant injiziert XFPMoEMethod in den FusedMoE-Apply-Pfad. Apply ist die HOT-API.
- **Decoupling**: ❌ **Niedrig** — apply() ist DIE Quantization-Method-API. Patches sind hier minimal nötig wenn auch unangenehm. Subclass ginge nur wenn die quant_method-API selbst genug Hooks anbietet.

### 18. `vllm/model_executor/layers/attention/attention.py` (HIGH)
- **multiquant**: 89+/-1 lines — attention-Hooks (vermutlich KV-quant init + apply)
- **upstream**: 320 lines — attention-base oft refactored
- **Coupling**: Attention.weight setup + KV cache-allocation tangiert multiquant.
- **Decoupling**: ❌ **Niedrig** — KV-Cache-Allocation ist core, kein Plugin-Punkt.

### 19. `vllm/model_executor/model_loader/base_loader.py` (HIGH)
- **multiquant**: **194 lines added** — größter Single-File-Patch
- **upstream**: 23 lines change — relativ stabil
- **Coupling**: process_weights_after_loading-Sequenz wird multiquant-aware orchestriert (Cache-Load, Dispatch zu xfp/marlin/autoround).
- **Decoupling**: ❌ **Niedrig** — Loader-Lifecycle intrinsisch. Aber: 194 lines könnten in `vllm/multiquant/loader_hooks.py` extrahiert werden + nur 5-10 line dispatch-call in `base_loader.py`. **Reduktion 194 → 10 Lines coupling.**
- **Aufwand:** 4-8h **Top-Priorität-Refactor**.

### 20. `vllm/v1/worker/gpu_worker.py` (HIGH, shutdown)
- **multiquant**: **159 lines added** — Shutdown patches (UniProc cleanup, MoE-meta-state)
- **upstream**: 70 lines change
- **Coupling**: Worker-Lifecycle intrinsisch.
- **Decoupling**: ❌ **Niedrig** für Shutdown-Hooks (kein Plugin-Punkt). ✅ **Mittel** für CleanUp-Logic (kann nach `vllm/multiquant/worker_hooks.py` extrahiert werden + 5-10 line dispatch).
- **Aufwand:** 4-8h für Extract-Refactor.

### 21. `vllm/v1/attention/backends/turboquant_attn.py` (AA — both added)
- **Status**: Beide Branches haben den File neu hinzugefügt. Multiquant hat eigene TQ-Backend, upstream hat... was Anderes mit dem gleichen Namen.
- **Decoupling**: Datei umbenennen auf `multiquant_tq_attn.py`, dann conflict-free.
- **Aufwand:** 5 min.

## Decoupling-Empfehlungen — Priorisiert

### TIER A (sofort, hohe Wirkung) — schätzungsweise -50% Coupling

| Refactor | Effekt | Aufwand |
|---|---|---|
| `quantization/__init__.py` → Plugin-Registry | -21 Lines coupling, eliminiert File komplett | 1-2h |
| `qwen3_next.py` VLLM_SKIP_GDN_WARMUP → monkey-patch | eliminiert massive Re-Application bei upstream-refactors | 30 min |
| `attention/backends/registry.py` → self-register | eliminiert File | 1h |
| `attention/backend.py` → subclass-Hook | eliminiert 4-line patch | 1h |
| `turboquant_attn.py` → rename `multiquant_tq_attn.py` | conflict-free | 5 min |
| `tests/v1/core/test_kv_cache_utils.py` → move zu `tests/multiquant/` | Test-conflict eliminiert | 30 min |

**Tier A total**: ~5h Aufwand, eliminiert 6 Files (von 21 Conflicts).

### TIER B (refactor in Zwischenzeit) — schätzungsweise weitere -25% Coupling

| Refactor | Effekt | Aufwand |
|---|---|---|
| `model_loader/base_loader.py` → extract zu `vllm/multiquant/loader_hooks.py` | 194 lines → ~10 lines coupling | 4-8h |
| `v1/worker/gpu_worker.py` → extract Cleanup zu `vllm/multiquant/worker_hooks.py` | 159 → ~10 lines coupling | 4-8h |
| `csrc/torch_bindings.cpp` → eigenes pybind-Modul | eliminiert File-conflict | 2-3h |
| `vllm/utils/torch_utils.py` → multiquant-helpers nach `vllm/multiquant/utils.py` | eliminiert File | 1-2h |
| `vllm/config/cache.py` → MultiquantCacheConfig-Subclass | -10 lines | 2h |
| `vllm/model_executor/layers/conv.py` → monkey-patch | eliminiert | 1h |
| Router-Patches (`router_factory`, `fused_topk_bias_router`) → factory-register-API | -3-8 lines coupling | 2h |
| `vocab_parallel_embedding.py` → attribute-check | -15 lines | 2h |

**Tier B total**: ~20-30h Aufwand, eliminiert weitere 8 Files.

### TIER C (intrinsisch, NICHT sinnvoll zu entkoppeln)

Diese Files bleiben Patches. Coupling-Punkt ist intrinsisch zu vllm-Architektur:

| File | Warum |
|---|---|
| `model_executor/layers/attention/attention.py` | KV-Allocation + Dequant tief in forward |
| `model_executor/layers/fused_moe/fused_moe.py` | apply()-Hot-Path |
| `model_executor/layers/fused_moe/layer.py` | FusedMoE-Apply ist die API |
| `model_executor/layers/fused_moe/router/base_router.py` (RIY) | Routing-Forward intrinsisch ohne upstream-Plugin-API |
| `v1/core/kv_cache_utils.py` | KV-cache-Berechnung tief im Scheduler |
| `model_executor/models/qwen3_5_mtp.py` | Eigene MTP-Modell-Class — kein Patch sondern eigener File |

**Diese 6 Files bleiben Patches.** Bei jedem upstream-refactor neu portieren (~1-4h pro File).

## Ergebnis-Zusammenfassung

**Aktuell (vor Refactor):** 21 Conflicts, 1-2 Tage Aufwand pro vLLM-Upgrade.

**Nach Tier A (~5h):** ~14 Conflicts, ~1 Tag.

**Nach Tier A+B (~25-35h):** ~6 Conflicts, ~3-6h pro Upgrade.

**Tier C** (6 Files) bleibt unvermeidbar.

## Konkreter Vorschlag

**Vor 0.20-Upgrade:** Tier A in `multiquant`-Branch durchführen (5h). Spart bei diesem Upgrade direkt 6 Files Conflict-Auflösung.

**Nach 0.20-Upgrade:** Tier B in eigenen PRs auf `multiquant` durchführen (20-30h verteilt über Tage).

**Bei 0.21-Upgrade:** Test ob Refactor-Effekt sich materializiert (sollte 6 → 14 → 21 Conflicts statistisch nachweisbar sein).

## Hunk-Verteilung pro File (lokal vs. durchwachsen)

Daten aus `git diff :1:<f> :2:<f> | grep '^@@'`:

### LOKAL (1 zusammenhängender Hunk — perfekt extract-able)

| File | Hunk-Position | LOC | Refactor-Strategie |
|---|---|---|---|
| `v1/worker/gpu_worker.py` | **Line 1005**, 164 LOC | 164 | **Move kompletter Block** zu `vllm/multiquant/worker_hooks.py`, in core 1 Zeile `register_cleanup_hooks(self)`. **Top-Kandidat** weil 164 LOC in 1 Block. |
| `models/qwen3_next.py` | Line 734, 23 LOC | 23 | Monkey-patch via `_warmup_prefill_kernels` override |
| `attention/backend.py` | Line 937, 9 LOC | 9 | subclass-override |
| `attention/backends/registry.py` | Line 82, 15 LOC | 15 | self-register aus `vllm.multiquant.__init__` |
| `config/cache.py` | Line 20, 16 LOC | 16 | MultiquantCacheConfig-Subclass |
| `conv.py` | Line 257, 17 LOC | 17 | Monkey-patch oder subclass |
| `utils/torch_utils.py` | Line 41, 16 LOC | 16 | move zu `vllm.multiquant.utils` |
| `vocab_parallel_embedding.py` | Line 66, 23 LOC | 23 | attribute-check abstrahieren |

### MODERAT (2-6 hunks, kohärenter Zweck)

| File | Hunks | Spread | Strategie |
|---|---|---|---|
| `quantization/__init__.py` | 2 | Lines 33+161 | **Plugin-API** (Tier A) |
| `fused_topk_bias_router.py` | 2 | Lines 184+192 (eng) | router-subclass |
| `router_factory.py` | 6 | Lines 49-172 (klein) | factory-register-API |
| `base_router.py` | 6 | Lines 6-303 | RIY als optional router-stage |
| `model_loader/base_loader.py` | 4 | Lines 24-157 | **Extract zu `multiquant/loader_hooks.py`** (Tier B) — 194 LOC over 4 hunks → ~10 lines coupling |
| `v1/core/kv_cache_utils.py` | 4 | Lines 1078-1439 | KV-cache-helpers — schwer zu extrahieren |
| `csrc/torch_bindings.cpp` | 2 | Lines 2+774 | eigenes pybind-Modul |
| `tests/v1/core/test_kv_cache_utils.py` | 2 | Lines 1742+2142 | move zu `tests/multiquant/` |

### DURCHWACHSEN (5+ verstreute hunks — bleiben Patches)

| File | Hunks | Spread | Warum nicht extract-bar |
|---|---|---|---|
| `fused_moe/fused_moe.py` | **12** | Lines 125-1695 | quant_method-calls überall im forward — Hot-Path-Hooks intrinsisch |
| `attention/attention.py` | 4 | Lines 241-736 | KV+attention-hooks an verschiedenen forward-Punkten |
| `fused_moe/layer.py` | 4 | Lines 448-1547 | FusedMoE-apply() + andere Methods |
| `models/qwen3_5_mtp.py` | 5 | Lines 2-198 | eigene MTP-class — kein Patch sondern eigener File. Move-able zu `vllm/multiquant/models/qwen3_5_mtp.py`. |

## Refactor-Effekt — Quantifizierung

**Aktueller Stand:** 21 Conflicts pro Upgrade.

**Nach Tier A (alle 8 LOKALE + `quantization/__init__.py`):**
- 9 Files entkoppelt
- ~12 Conflicts pro Upgrade (-43%)
- Aufwand: ~5h einmalig

**Nach Tier A+B (zusätzlich `base_loader.py`, `gpu_worker.py`, `kv_cache_utils.py` extract + Router-Plugin):**
- 13-15 Files entkoppelt
- ~6-8 Conflicts pro Upgrade (-65%)
- Aufwand: ~25-35h einmalig

**Tier C bleibt:** 4-5 Files mit durchwachsenen hunks (`fused_moe.py`, `attention.py`, `layer.py`, `qwen3_5_mtp.py`). Diese müssen pro Upgrade per Hand portiert werden, ~1-3h pro File.

**Break-Even-Punkt:** Tier A spart ab dem 2. Upgrade Zeit. Tier B amortisiert sich nach 4-5 Upgrades (vllm hat ~17 Upgrades pro Jahr).

## Lessons learned aus diesem Audit

1. **`qwen3_next.py` 1062 LOC upstream-Δ vs 18 LOC multiquant** ist das schmerzhafteste Verhältnis — kleiner Patch in massiv geändertem File → Refactor unbedingt.
2. **`base_loader.py` 194 LOC multiquant vs 23 LOC upstream** — multiquant ist hier die Mehrheit, aber Logik gehört in eigenen module.
3. **Tier-A-Refactors sind alle <2h** und eliminieren je 1 File. ROI ist exzellent.
4. **Tier-C-Patches** sollten **detailliert dokumentiert** sein (warum genau diese Lines), damit Re-Application nach 6 Monaten verständlich ist.
