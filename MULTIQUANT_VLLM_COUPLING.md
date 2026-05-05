# MultiQuant ↔ vLLM Coupling-Audit

> Stand: 2026-05-05. Vorbereitung für vLLM 0.17.1 → 0.20.1 Upgrade.
> Ziel: Coupling-Fläche identifizieren, damit zukünftige Upgrades
> (jede 2-3 Wochen ein neuer vLLM-release) chirurgisch portierbar sind.

## TL;DR

| Bereich | Files | LOC | Coupling-Klasse |
|---|---|---|---|
| Multiquant own (`vllm/multiquant/`) | 51 | 12,153 | self-contained |
| **Patched vLLM core mit multiquant-refs** | **28** | (line-deltas in commits) | **siehe unten** |
| Total vllm-files diff zu v0.17.1 (incl. cherry-picks) | 757 | — | nicht alles multiquant |

## Coupling-Hot-Spots (HIGH RISK bei vLLM-Upgrade)

Files mit >10 multiquant-references und tief in vLLM-Lifecycle:

| File | Hits | Zone | Was bricht potentiell |
|---|---|---|---|
| `model_executor/layers/fused_moe/riy.py` | 25 | RIY MoE | RIY% expert-masking |
| `model_executor/model_loader/multiquant_loader.py` | 20 | weight-loader | Cache-load + dispatch quant |
| `model_executor/model_loader/base_loader.py` | 17 | Loader-Lifecycle | process_weights_after_loading hook |
| `model_executor/layers/fused_moe/router/base_router.py` | 17 | MoE routing | RIY+TopK |
| `v1/attention/backends/multiquant_attn.py` | 16 | Attention | Attention KV-quant hooks |
| `model_executor/layers/attention/mla_attention.py` | 11 | MLA | get_and_maybe_dequant_weights |
| `model_executor/layers/attention/attention.py` | 11 | Attention | KV-quant hooks |

## MID-COUPLING (API-Verträge)

| File | Hits | Zone |
|---|---|---|
| `model_executor/layers/fused_moe/layer.py` | 9 | FusedMoE base |
| `v1/attention/backends/mla/multiquant_mla.py` | 7 | MLA backend |
| `engine/arg_utils.py` | 7 | CLI flags |
| `model_executor/layers/quantization/__init__.py` | 6 | Quant-method registration |
| `model_executor/layers/quantization/inc.py` | 6 | INC quant |
| `entrypoints/serve/riy_api.py` | 5 | API server |
| `v1/worker/gpu_worker.py` | 4 | Worker-Lifecycle (shutdown patches) |
| `v1/attention/ops/triton_mq_fused_decode.py` | 4 | Triton MQ decode |
| `model_executor/model_loader/utils.py` | 4 | _moe_meta_active hook |
| `v1/worker/gpu_model_runner.py` | 3 | model-runner |
| `turboquant/__init__.py` | 3 | TurboQuant |
| `model_executor/model_loader/__init__.py` | 2 | loader registry |
| `config/parallel.py` | 2 | TP config |

## LOW-COUPLING (loose, fast pluginable)

| File | Hits |
|---|---|
| `v1/attention/backends/registry.py` | 1 |
| `v1/attention/ops/triton_mq_decode.py` | 1 |
| `platforms/cuda.py` | 1 |
| `model_executor/models/qwen3_next.py` | 1 |
| `model_executor/model_loader/default_loader.py` | 1 |
| `entrypoints/serve/__init__.py` | 1 |

## vLLM-API Konsumption (was multiquant aus vLLM importiert)

```
vllm.distributed                         (TP comm)
vllm.logger.init_logger                  (logger)
vllm.envs                                (env-var registry)
vllm._custom_ops.turboquant_round_trip   (low-level kernel)
vllm.utils.torch_utils.direct_register_custom_op   (custom_op API)

vllm.model_executor.parameter            (ModelWeightParameter, BasevLLMParameter)
vllm.model_executor.layers.linear        (LinearBase)
vllm.model_executor.layers.vocab_parallel_embedding
vllm.model_executor.layers.fused_moe     (FusedMoE, fused_experts, FusedMoEConfig)
vllm.model_executor.layers.fused_moe.fused_moe_method_base
vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method
vllm.model_executor.layers.quantization.base_config
vllm.model_executor.layers.quantization.gptq        (GPTQ-Re-Use)
vllm.model_executor.layers.quantization.gptq_marlin (Marlin-Re-Use)
vllm.model_executor.kernels.linear.mixed_precision.marlin
vllm.model_executor.model_loader.utils._moe_meta_active
```

15 verschiedene Module — alle Kandidaten für API-Drift bei minor-version-Upgrade.

## vLLM 0.17 → 0.20 Risiko-Bewertung

vLLM macht ~alle 2-3 Wochen einen Minor-Release. 3 Minor-Versions
(0.18, 0.19, 0.20) zwischen unserem Stand und v0.20.1 — typische
Breaking-Change-Bereiche:

| Bereich | Risiko 0.17 → 0.20 |
|---|---|
| `compilation/` (aot_compile, cuda graph) | **HIGH** — heute schon 5 Crashes wegen aot_compile-API. 0.20 hat wohl weitere Änderungen. |
| `v1/worker/` (Worker-Lifecycle) | **HIGH** — UniProc/MultiProc transitions, shutdown-Hooks |
| `model_loader/{base, default}_loader.py` | **HIGH** — process_weights_after_loading-Sequence |
| `model_executor/layers/fused_moe/` | **MEDIUM-HIGH** — MoE-Runner-Refactors üblich |
| `model_executor/layers/quantization/` (registry) | MEDIUM — neue Quant-Schemes hinzu |
| `model_executor/layers/attention/{mla,}attention.py` | MEDIUM — MLA-Absorption-Patterns |
| `model_executor/parameter.py` | LOW — stable API |
| `direct_register_custom_op` | LOW — stabilisiert in 0.16+ |

## Empfehlungen für sauberes Decoupling

### 1. **Patch-Stack als sicherbare Artefakte**
Aktuell sind ~28 patches in core verstreut. Als formales `multiquant-patches/`-Verzeichnis mit `.patch`-Files würden Re-Applications auf neuer base mechanisch:

```bash
# Bei vllm-Upgrade:
cd /tmp/vllm-build && git checkout v0.20.1
for p in multiquant-patches/*.patch; do
  git am < "$p" || git am --abort && echo "BREAK: $p needs port"
done
```

Patch-Files dokumentieren genau **welche Änderung in welchem File** für **welchen Zweck** — vs. heute nur durch git-blame rekonstruierbar.

### 2. **Plugin-Pfade nutzen wo möglich**
vLLM 0.18+ hat erweitertes Plugin-System. Bereiche die als Plugin gehen:

- **Quant-Methods**: `XFPLinearMethod`, `XFPMoEMethod` als plugin-Registrierung statt Patch in `quantization/__init__.py`
- **Attention-Backends**: `multiquant_attn`, `multiquant_mla` registrieren via `vllm.attention.backends.registry` API
- **Model-Loader**: `multiquant_loader.py` als load-format-plugin

Geschätzt: 3-5 der 28 patches könnten so eliminiert werden, je 30-100 LOC weniger Coupling.

### 3. **Adapter-Patterns für unvermeidbare Hooks**
Hooks die in core bleiben müssen (process_weights_after_loading, MLA absorption):

- Single-line dispatch in core: `multiquant.hooks.on_process_weights(layer)`
- Logik in `vllm/multiquant/hooks.py` (ohne core-Patches an logik-position)
- Bei vllm-API-Wandel: nur die single-line dispatch in core re-applyen

### 4. **Versionierungs-Tagging**
Branch-Namensschema: `multiquant-vllm-0.20`, `multiquant-vllm-0.21`, etc.
Master = letzter erfolgreich getesteter `multiquant-vllm-X.Y` re-tagged.

## Konkreter Plan für 0.20-Upgrade

### Phase 1: Vorbereitung (DGX, ~halber Tag)
1. Branch `multiquant-vllm-0.20` von HEAD (`cb60a6770`)
2. `git remote add upstream https://github.com/vllm-project/vllm.git && git fetch upstream`
3. Identifizieren: welche der 28 modified-files haben **upstream-Conflicts** mit v0.20.1

### Phase 2: Merge / Re-Apply (DGX, 1-2 Tage)
1. `git merge upstream/v0.20.1` — Conflicts in den 28 hot-spot-files erwartet
2. Pro Conflict-File:
   - vllm-side reviewen (was hat sich geändert in API)
   - multiquant-Patch portieren (kann komplettes Rewrite sein bei Compilation/Worker)
   - Test: import + AST-check
3. Multiquant own files (51 files in `vllm/multiquant/`) sollten unverändert weiterlaufen

### Phase 3: Test (DGX dann RTX, 1-2 Tage)
1. DGX: 35B XFP-V2a Smoke + GSM8K-Probe
2. DGX: 122B/Q3.6/GLM Smoke + GSM8K-Probe
3. RTX: gleiche Reihe + DSV4-Flash Smoke (das ist der Pay-Off)
4. Falls cascade-bugs (siehe heute): pro Bug Fix-Commit + Doku

### Phase 4: Merge zurück (1h)
- Wenn alles grün: PR `multiquant-vllm-0.20 → multiquant`
- Tag Snapshot: `multiquant-on-v0.17.1-snapshot` für Rollback

## Lessons Learned heute (2026-05-04/05)

1. **Patch-Salvage VOR `git reset --hard`** — 9 modified-file-hunks verloren = 2-3h Re-Discovery von 5 V2-aot_compile-Fixes
2. **Layer-Cache invalidiert nicht durch git-push** — Image-rebuild kann mit `--use-layer-cache` alten code drin haben
3. **File-Mount-Overlay** schneller als Image-Rebuild für single-file-fixes
4. **MLA + XFP** brauchen eigenen Ausnahme-Mechanismus (`XFP_SKIP_LAYERS`)
5. **`@torch.compiler.disable`** ist VERBOTEN in fullgraph_capture (gb0098) — nicht naive verwenden
6. **vllm 0.17 aot_compile** triggert graph-breaks bei JIT-load + logger-calls in dispatch — alle V2-paths müssen custom_op-wrapped sein (gb0291, gb0007)

## Zukünftige Coupling-Risiken im Auge behalten

- vllm 0.21+ wird vermutlich neue Compilation-Pipelines bringen (TF32-default-changes etc.)
- Worker-Lifecycle bleibt Hot-Spot (UniProc → MultiProc transitions)
- DSPy/structured-output integration könnte multiquant-Hook-Punkte stören
