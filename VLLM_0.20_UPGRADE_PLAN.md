# vLLM 0.17.1 → 0.20.1 Upgrade Plan

> Branch: `multiquant-vllm-0.20` (von `da5818c03`)
> Rollback-Tag: `pre_vllm20`
> Common ancestor: `a5e9d511d` (vor 517 multiquant + 998 upstream Commits)

## Phase 1 — Merge-Probe (DONE 2026-05-05)

Dry-run `git merge --no-commit --no-ff v0.20.1`:
- **Auto-merged:** 1333 files
- **Conflicts:** 21 files (deutlich besser als 28 erwartet)

## Conflict-Klassifikation

### HIGH-IMPACT (6 Files) — multiquant-core-hooks

Werden bei Upgrade fast immer breaking sein, brauchen API-Verständnis:

| File | Was multiquant macht | 0.20-Risiko |
|---|---|---|
| `model_executor/layers/attention/attention.py` | KV-Cache + Attention-hooks | KV-Quantization-API in 0.18+ refactored |
| `model_executor/layers/fused_moe/fused_moe.py` | MoE forward dispatch | MoE-Runner geändert in 0.19 |
| `model_executor/layers/fused_moe/layer.py` | FusedMoE base + apply | breaking expected |
| `model_executor/layers/fused_moe/router/base_router.py` | RIY% expert masking | Router refactor in 0.20 |
| `model_executor/layers/quantization/__init__.py` | XFP/AutoRound registry | quant-registry-API in 0.18 plugin-fähig |
| `v1/worker/gpu_worker.py` | Shutdown-cleanup hooks | UniProc/MultiProc transition in 0.20 |

### MEDIUM (7 Files)

| File | Risiko |
|---|---|
| `v1/attention/backends/{registry,backend}.py` + `turboquant_attn.py` | Backend-Register-API stabil, aber line-number-shifts |
| `model_executor/model_loader/base_loader.py` | process_weights_after_loading-Sequenz |
| `fused_moe/router/{fused_topk_bias_router,router_factory}.py` | Router refactor follow-on |
| `utils/torch_utils.py` | direct_register_custom_op API stabil seit 0.16 |

### LOW (8 Files) — schnell

| File | Erwartung |
|---|---|
| `csrc/torch_bindings.cpp` | pybind add/remove ops |
| `config/cache.py` | KV-Cache-Config-Felder |
| `layers/conv.py` | minor patch |
| `layers/vocab_parallel_embedding.py` | embed-multiquant-attribute |
| `models/qwen3_5_mtp.py` | MTP packed_modules_mapping fix |
| `models/qwen3_next.py` | NVTX wrapper |
| `v1/core/kv_cache_utils.py` + test | minor signatures |

## Phase 2 — Conflict Resolution Plan

**Reihenfolge: LOW → MEDIUM → HIGH** (sammelt Vertrauen, früh Fehler-Detection)

### Schritt 2.1 — Real Merge starten + LOW Conflicts (~30-60 min)
```bash
git checkout multiquant-vllm-0.20
git merge --no-ff v0.20.1   # Conflicts erscheinen
# Pro LOW-File: git diff <file>, manuell mergen, git add <file>
```

### Schritt 2.2 — MEDIUM (~1-3 h)
- Backend-Registry, Loader-Sequenz: API-Verträge prüfen via `git log v0.17.1..v0.20.1 -- <file>`
- bei Doubt: `git checkout v0.20.1 -- <file>` als Basis nehmen + multiquant-Patch händisch re-apply

### Schritt 2.3 — HIGH (~4-8 h, evtl. Cascade-Fixes wie heute)
- Pro File:
  1. `git log v0.17.1..v0.20.1 -- <file>` lesen
  2. multiquant-Patch verstehen (`git log multiquant.. -- <file>`)
  3. neu schreiben gegen 0.20-API
  4. AST-Check + Import-Check

### Schritt 2.4 — Commit + Push
```bash
git commit -m "merge: vllm v0.20.1 — XX conflicts resolved (HIGH/MED/LOW)"
git push origin multiquant-vllm-0.20
```

## Phase 3 — Build & Smoke (~halber Tag)

1. Image-Rebuild auf DGX (oder RTX) mit branch `multiquant-vllm-0.20`
2. 35B XFP-V2a Smoke + GSM8K --limit 50 → Δ vs `pre_vllm20` baseline
3. 122B XFP-V2a Smoke + GSM8K
4. Q3.6-27B XFP-V2a Smoke
5. GLM XFP-V2a Smoke (mit XFP_SKIP_LAYERS)

Falls cascade-bugs (siehe heute): pro Bug Fix-Commit, Doku, weiter.

## Phase 4 — DSV4-Flash Test

1. RTX: vllm-multiquant-0.20 Image deployen
2. DSV4-Flash Container starten (BF16 zunächst — kein XFP-V2 für DSV4)
3. Smoke + GSM8K --limit 50
4. Bei Erfolg: dokumentieren, evtl. später XFP-V2 für DSV4 wenn Architektur passt

## Phase 5 — Merge zurück (1 h)

```bash
git checkout multiquant
git merge --no-ff multiquant-vllm-0.20    # fast-forward weil keine multiquant-Drift
git tag post_vllm20-base
git push origin multiquant --tags
```

## Rollback-Strategie

Falls Phase 2 oder 3 in einer Sackgasse landet:
```bash
git checkout multiquant
git reset --hard pre_vllm20
# Branch multiquant-vllm-0.20 bleibt für späteres Studium
```

## Erwartete Outcomes

- ✅ DSV4-Flash lauffähig (vLLM 0.20 hat `DeepseekV4ForCausalLM`)
- ✅ Performance-Optimierungen aus 3 Minor-Versions
- ✅ Bessere base für nächsten Upgrade (0.21, 0.22)
- ⚠️ V2a-Bench-Werte könnten leicht abweichen (compiler-changes)
- ⚠️ Custom_op-API könnte sich evolved haben — möglicherweise neue cascade

## Lessons heute → Schutz dieses Mal

1. **vor Hard-Reset Patch-Sicherung** — auf RTX VOR git pull `git diff > .salvage/...patch`
2. **File-Mount-Overlay** für Crash-Cycle-Iteration statt Image-Rebuild
3. **AST-Check nach jedem Edit** — `python3 -c "import ast; ast.parse(open(f).read())"`
4. **Custom_op vor Logger-Calls** in dispatch — Lehre aus 5 V2-aot_compile-Crashes
