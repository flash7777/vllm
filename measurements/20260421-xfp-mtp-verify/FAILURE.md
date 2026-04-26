# MTP Phase A Verify — Ergebnis: Fall B (Fehler reproduziert)

**Date:** 2026-04-22 (Start 23:40 CEST, Fehler ~00:01 CEST nach 21 min Load)
**Image:** `localhost/vllm-multiquant:xfp_speed` (commit `683d80d8b`,
linear_attn-Fix live, bf16 + XFP auto)
**Command:**
```bash
./start.multiquant --model Qwen3.5-122B-A10B \
  --weight-dtype xfp --kv fp8 --weight-dtype-lm-head fp8 \
  --spec 1 --spec-method mtp
```
→ `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`

## Ergebnis

**Fall B** aus dem Plan: Fehler reproduziert exakt wie im ersten Versuch
(vor vielen Image-Rebuilds und Code-Änderungen). Die ursprüngliche
Behauptung in TASK_testrunde_xfp.md war **korrekt**. Meine
Plan-Mode-Analyse, die die Existenz von `packed_modules_mapping` als
hinreichend annahm, war **falsch** — `packed_modules_mapping` alleine
bewirkt die Fusion hier nicht.

## Exakter Fehler

```
ValueError: There is no module or parameter named
'model.layers.0.self_attn.q_proj' in Qwen3_5MoeMTP.
The available parameters belonging to model.layers.0.self_attn
(Qwen3NextAttention) are:
  {'model.layers.0.self_attn.qkv_proj.weight',
   'model.layers.0.self_attn.o_proj.weight',
   'model.layers.0.self_attn.q_norm.weight',
   'model.layers.0.self_attn.k_norm.weight'}
```

Ort: `vllm/model_executor/models/utils.py:342` in `_load_module`, aus
`Qwen3_5MoeMTP.load_weights` → `AutoWeightsLoader.load_weights` →
rekursivem `_load_module("model", …)` → `_load_module("model.layers", …)`
→ `_load_module("model.layers.0", …)` → `_load_module("model.layers.0.self_attn", …)`.
Volle Stack in `fail-trace.log`.

## Was *erst nach Phase A* verstanden ist

Das Modul-Setup:

| Level | Klasse | hat `packed_modules_mapping`? |
|---|---|---|
| Root `self` | `Qwen3_5MoeMTP(Qwen3_5MTP, QwenNextMixtureOfExperts)` | **ja** (via `Qwen3_5MTP` line 166-173) |
| `self.model` | `Qwen3_5MultiTokenPredictor` (line 56-106) | **nein** |
| `self.model.layers[0]` | `Qwen3_5DecoderLayer(..., layer_type="full_attention")` | nein |
| `self.model.layers[0].self_attn` | `Qwen3NextAttention` | nein (hat schon fused `qkv_proj`) |

**Hypothese:** `AutoWeightsLoader(self)` (Zeile 266 in `qwen3_5_mtp.py`)
bekommt `self = Qwen3_5MoeMTP`. Beim Rekursionsdurchlauf über
`_load_module` wandert der Loader in `self.model`
(`Qwen3_5MultiTokenPredictor`) — **und dieser Submodul hat keine
`packed_modules_mapping`**. Die q/k/v-Fusion wird dort nicht ausgelöst,
weil der Loader die mapping-Tabelle nicht den Weg nach unten trägt.

Zum Vergleich: Die Main-Model-Klasse
`Qwen3_5MoeForConditionalGeneration` hat `packed_modules_mapping`
**direkt auf dem Submodul**, das `self_attn` enthält (wahrscheinlich auf
dem Decoder-Root oder einem Zwischenmodul, das bei der Rekursion
aufgelöst wird) — deshalb funktioniert der Haupt-Load, aber der MTP-Load
nicht.

**Fix-Optionen** (nicht in diesem Plan-Scope):

1. **Klasse `Qwen3_5MultiTokenPredictor` bekommt `packed_modules_mapping`**
   (line ~56 ergänzen):
   ```python
   class Qwen3_5MultiTokenPredictor(nn.Module):
       packed_modules_mapping = {
           "qkv_proj": ["q_proj", "k_proj", "v_proj"],
           "gate_up_proj": ["gate_proj", "up_proj"],
       }
   ```
   Einfachster Fix, folgt dem bestehenden Muster.

2. **In `remap_weight_names` manuell fusen** (qwen3_5_mtp.py:255-264):
   Die drei Tensoren `q_proj.weight`, `k_proj.weight`, `v_proj.weight`
   aus dem Generator sammeln und als ein `qkv_proj.weight` mit
   `shard_id`-Info an AutoWeightsLoader geben. Invasiver, braucht
   Sharding-Awareness.

Option 1 ist der korrekte Fix in vLLM-Upstream-Stil. **Separater Task**.

## Konsequenz für TASK/Messungen

1. Die **NST=1..5-Matrix in `measurements/20260420-xfp-mtp/`** ist
   reiner **ngram**-Lauf — kein mtp-Test. Das Label
   "MTP / ngram Speculative Decoding Matrix" im COMPARISON.md-Header ist
   irreführend. Korrigieren zu "ngram-Only Speculative Decoding Matrix
   (MTP blockiert durch qwen3_5_mtp-Loader-Bug)".

2. Die Aussage in TASK_testrunde_xfp.md §2026-04-21 MTP Matrix
   ("`qwen3_5_mtp.py:267 remap_weight_names` fusioniert q/k/v nicht in
   qkv") bleibt **sachlich korrekt**, aber ist präziser formulierbar:
   die Root-Cause liegt darin, dass `Qwen3_5MultiTokenPredictor` keine
   `packed_modules_mapping` hat — der Loader sieht sie nicht rekursiv
   aus der Parent-Klasse durchgereicht.

3. Die "open bugs"-Section in TASK bleibt gültig. Fix ist ein separater
   Task (1–2 Zeilen, fast trivial wenn Hypothese Option 1 stimmt).

## Phase B übersprungen

Phase B wird nicht ausgeführt — Fall B aus dem Plan. Die NST-Matrix
existiert bereits als ngram-only Daten und ist für die paper-relevanten
Aussagen ausreichend ("ngram hilft nicht bei GSM8K").

## Phase C — Empfohlene Doku-Updates (nach Nutzer-Zustimmung)

- `measurements/20260420-xfp-mtp/COMPARISON.md` Header + Interpretation:
  Label-Korrektur "MTP blockiert, gemessen wurde ngram"
- `TASK_testrunde_xfp.md`: "MTP scheiterte mit ValueError" bleibt, aber
  Ergänzung "Fehler **erneut verifiziert 2026-04-22 00:01 CEST unter
  xfp_speed** — Bug liegt in fehlender `packed_modules_mapping` auf
  `Qwen3_5MultiTokenPredictor`"

## Artefakte

- `fail-trace.log` — vollständiger Stack (mehrere MiB, vollständiger
  Load-Log inkl. 1800 Layer Quantisierung + Fehler)
- `load.log` — leer (podman rm hat container-logs gelöscht, Stack
  sitzt nur in fail-trace.log)
- `FAILURE.md` — diese Datei
