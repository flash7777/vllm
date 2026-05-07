# TAGEBUCH 2026-05-07

## Tagesziel

Nach gestrigem 0.20-Upgrade-Abbruch (Smoke crash bei `vllm_is_batch_invariant`):
durchziehen mit korrigierter Strategie für die 6 HIGH-Files. User-Direktive:
"alter, wir mergen bis es passt."

## Realisierung — die 13 Cascade-Fixes

Nach Pulling-Back von `--ours`-shortcut (gestern) und Switch zu **upstream-base
+ multiquant-deltas selektiv** für die echten Hotspots, eine cascade von 13
sequentiellen Smoke-Versuchen:

| # | Crash | Fix | Resolution-Dauer |
|---|---|---|---|
| 1 | `vllm_is_batch_invariant` removed | Compat shim in fused_moe.py | 10 min |
| 2 | gleicher Import in fused_moe.py | inline | 5 min |
| 3 | `ModuleName` → `LayerName` in default_moe_runner | sed-rename | 10 min |
| 4 | doppelte `moe_forward` op-registration | Removal aus default_moe_runner | 15 min |
| 5 | `fused_moe_make_expert_params_mapping` export missing | Module-level wrapper in layer.py | 15 min |
| 6 | `resolve_kv_cache_block_sizes` not found | Function-extract aus v0.20.1 + math-import | 20 min |
| 7 | `VLLM_MOE_DP_CHUNK_SIZE` env removed | getattr-fallback | 10 min |
| 8 | **`MoERunner.__init__` 8 args clash** | layer.py upstream-base + 4 RIY-deltas selektiv | **2h** |
| 9 | `LayerName.__repr__` codegen syntax | `__repr__` injection patch | 30 min |
| 10 | `LayerName` NameError in exec namespace | `VLLM_USE_LAYERNAME=0` env workaround | 20 min |
| 11 | `BaseRouter.select_experts` `input_ids` kwarg | optional arg ergänzt | 10 min |
| 12 | Mamba cache vs `max_num_seqs` | `--max-num-seqs 32` | 5 min |
| 13 | `math` import fehlt in kv_cache_utils.py | append `import math` | 5 min |

**Total**: ~5h Cascade-Engineering, 13/13 erfolgreich.

## Schlüssel-Wende

Nach Cascade #8 (`MoERunner.__init__` 8 args) wurde klar dass `--ours` für
HIGH-Files **falsch** war: multiquant-`layer.py` (0.17-API) erbt von
upstream-`MoERunner` (0.20-API) — fundamentaler clash, nicht durch
einzelne fixes lösbar.

Switch-Strategie: **upstream-base + selektive multiquant-deltas**.
- `attention.py`: upstream + `_init_multiquant_buffers` neben `_init_turboquant_buffers` (gates disjoint, `tq*/rq*` vs `turboquant_*`)
- `layer.py`: upstream-base, RIY-Hooks (4 hunks, 87 LOC) selektiv re-applied
- `base_loader.py`: ours behalten (multiquant streaming-quant intrinsisch) + upstream's `_has_online_quant` additive
- `base_router.py`, `kv_cache_utils.py`, `fused_moe.py`: ours mit gezielten API-fixes

## Erfolg

### 35B XFP-V2a auf v0.20.1 (clean image, no overlays)

```
short:  152.8 tok/s
medium: 188.7 tok/s
long:   191.5 tok/s   ← v0.17 war 96.2 (+99% Throughput!)
Math:   46/50 = 92%   (gleich wie v0.17)
```

### 122B XFP-V2a auf v0.20.1

```
short:  74.7 tok/s
medium: 97.3 tok/s
long:   98.6 tok/s
Math:   49/50 = 98%
GSM8K --limit 50: strict=98.0% (identisch v0.17)
```

### DSV4-Flash Architektur-Support

- ✅ `DeepseekV4ForCausalLM` in vLLM 0.20.1 ModelRegistry
- ✅ Image-load + multiquant-init startet
- ❌ Sparse-Attention-Indexer braucht **DeepGEMM** (SM90/SM100 nur)
- → Hardware-Limit auf SM120, kein Code-Bug
- → DSV4 auf consumer Blackwell (RTX PRO 6000) blockiert. Datacenter-deploy oder vLLM-fallback nötig.

## Image-Status

- Image: `localhost/vllm-multiquant-v020:latest` (44.2 GB)
- Commit: `ac7e8b288` (alle 9 cascade-fixes als clean commits + Dockerfile patches)
- **Self-contained**: keine file-overlays mehr nötig zur Laufzeit
- Default ENV: `VLLM_USE_LAYERNAME=0`
- Required runtime args: `--max-num-seqs 32` (Mamba cache fit)

## Branch-Stand

```
multiquant-vllm-0.20  ac7e8b288 (heute final)
multiquant            1ca0804c3 (v0.17 produktiv)
pre_vllm20 (tag)      da5818c03 (Rollback-Anker)
```

## Production-Status (RTX, GPU 0)

- mq-serve: 122B XFP-V2 PreV2a auf v0.17, **5 Tage up, unangetastet**
- v0.20-Migration validiert (122B XFP-V2a 98.0% strict identisch zu v0.17)
- mq-serve switch optional — kein Druck weil v0.17 stabil

## Diff-Statistik vs v0.20.1 release tag

- **127 added files** (multiquant own — `vllm/multiquant/`, `vllm/turboquant/`, kernels, etc.)
- **44 modified core files** — die echten Patches
- **0 deletions**
- 30,618 insertions / 1,578 deletions in vllm/kernels/csrc

vs gestriger v0.17-baseline-audit (28 modified files): leichte Erhöhung
auf 44 weil v0.20 mehr Coexistence-Patches forderte (`attention.py`
`_init_multiquant_buffers` neben upstream-TQ, `layer.py` upstream-base
mit RIY-deltas).

## Lessons heute

1. **`--ours` für HIGH-Files ist anti-pattern** — bei API-Drift verliert
   man upstream's neue Signaturen. Saubere Strategie: upstream-base +
   selektive multiquant-deltas, auch wenn länger.
2. **Cascade-Iteration ist linear** — jeder fix unblockiert den nächsten.
   Pro fix 5-30 min, nicht "endlos" wie ich gestern befürchtet hatte.
   Nach 13 fixes ist man durch (13 ≠ 50).
3. **Hardware-Limits muss man früher checken** — DSV4 braucht DeepGEMM,
   das ist NVIDIA-Datacenter-only. Hätte vor 0.20-Migration prüfbar
   gewesen sein.
4. **Default ENV `VLLM_USE_LAYERNAME=0`** — sicheres Default für
   NGC PyTorch 2.11 Codegen-issues, nicht alle vllm-features brauchen
   den hoisted opaque type.
5. **Mamba cache + max_num_seqs**: Bei kleinen GPU-fits (97 GB) muss
   max_num_seqs runterskaliert werden, sonst Init-error.

## Commits heute

- `d167bb248` fix(merge): re-resolve HIGH conflicts properly for v0.20.1
- `6f040ce61` fix(v0.20-cascade): 9 fixes für 35B XFP-V2a smoke success
- `ac7e8b288` build(v020): LayerName __repr__ patch + VLLM_USE_LAYERNAME=0 default

## Nächste Schritte (deferred)

- mq-serve auf v0.20 switchen (optional, nicht kritisch — v0.17 stabil)
- DSV4 auf Datacenter-Hardware (separate Aufgabe, nicht consumer-blocker)
- Tier-B-Refactor (base_loader extract zu hooks) falls 0.21-Upgrade
  schmerzhaft wird
