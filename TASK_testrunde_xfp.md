# TASK — XFP Test-Runde (2026-04-19 bis 2026-04-21)

Session-safe Referenz: bei Kontext-Kompaktierung hier aufgreifen.

## Mission

Profile XFP vs Marlin-INT4 auf Qwen3.5-MoE-Modellen (DGX Spark, GB10,
SM121a), verstehen wo XFP langsamer ist, optimieren — am Ende mehrere
Modelle sauber messen und Paper/RESULTS aktualisieren.

## Zeitleiste + Ergebnisse

### 2026-04-19/20 — Qwen3.5-122B-A10B Matrix

**Setup durchgehend:** `Qwen3.5-122B-A10B` BF16-Source, fp8 KV + fp8
LM-head, `bench.py seed=42 n=5`. Image
`localhost/vllm-multiquant:xfp_speed` (NGC 26.03-py3 + vLLM 0.17.1).

| Run | long | medium | short | Math | Notes |
|---|---:|---:|---:|---:|---|
| XFP v1 (pre linear_attn fix) | 17.3 | 18.9 | 2.5 | 98% | 108 GatedDeltaNet-Projektionen in BF16 |
| Marlin INT4 AutoRound | 25.8 | 29.6 | 2.5 | 94% | Intel auto_round:auto_gptq, fused_marlin_moe |
| **XFP v2 (linear_attn fix)** | **29.9** | **34.8** | **2.6** | **98%** | **+73% long, +4pp math vs Marlin** |

**Der Hebel:** `vllm/multiquant/policy.py:141` — `classify_layer()`
matched nur `self_attn`/`attention`, nicht `linear_attn`. 1-Zeilen-Fix.
Commit `683d80d8b` (branch `multiquant`, pushed to
`flash7777/vllm:multiquant`).

### 2026-04-21 — ngram Matrix (MTP damals blockiert)

**NICHT INS PAPER.** NgramSpec verschlechtert durchgehend, Math drops.

| NST | long | medium | Math |
|---:|---:|---:|---:|
| 0 (no spec) | 29.9 | 34.8 | 98% |
| 1 | 29.2 | 34.3 | 96% |
| 2 | 26.6 | 30.8 | 94% |
| 3 | 27.6 | 31.2 | 90% |
| 4 | 23.2 | 26.9 | 90% |
| 5 | 23.1 | 27.4 | 90% |

### 2026-04-22 — MTP Matrix (nach Loader-Fix)

**INS PAPER.** Echter Speedup bei NST=3, Math erhalten.

| NST | long | medium | Math | Δ vs no-spec |
|---:|---:|---:|---:|---:|
| 0 (no spec) | 29.9 | 34.8 | 98% | — |
| 1 | 29.3 | 35.9 | 98% | flat |
| 2 | 29.1 | 33.7 | 92% | math drop |
| **3** | **32.7** | **37.2** | **98%** | **+9.4% long** |
| 4 | 24.0 | 26.3 | 98% | overhead dominiert |
| 5 | 25.0 | 26.5 | 96% | — |

Fix: `Qwen3_5MultiTokenPredictor.load_weights` (qwen3_5_mtp.py:154-232)
nach dem Muster von `Qwen3NextMultiTokenPredictor` — manuelles
`stacked_params_mapping` Iteration mit `weight_loader(param, w, shard_id)`
Dispatch. Plus Mount in `start.multiquant:290`. Details + Fail-Historie:
`measurements/20260421-xfp-mtp-verify/COMPARISON-MTP.md`.

MTP head path (`--spec-method mtp`) scheiterte mit:
`ValueError: no module model.layers.0.self_attn.q_proj in Qwen3_5MoeMTP`.

**Verify-Run 2026-04-22 00:01 CEST (xfp_speed, commit 683d80d8b):** Fehler
reproduziert exakt. Root-Cause präziser: `Qwen3_5MultiTokenPredictor`
(qwen3_5_mtp.py:56) hat **keine** `packed_modules_mapping` — die auf
`Qwen3_5MTP` (line 166-173) definierte Tabelle wird von
`AutoWeightsLoader` nicht rekursiv in den Predictor-Submodul
durchgereicht, deshalb wird die q/k/v→qkv-Fusion am `self_attn` nicht
ausgelöst. Fix-Empfehlung: `packed_modules_mapping` direkt auf
`Qwen3_5MultiTokenPredictor` duplizieren (triviale 4 Zeilen).

Volle Analyse + Stack: `measurements/20260421-xfp-mtp-verify/FAILURE.md`.
Fallback auf ngram für die NST-Matrix.

Dokumentiert in `measurements/20260420-xfp-mtp/COMPARISON.md` (Label
„MTP / ngram" ist irreführend — es wurde nur ngram gemessen).

### 2026-04-21/22 — MoE 4-Expert-Sample Validation (fertig)

**Ziel:** Bestätigen (oder widerlegen), dass XFP's `sample_experts=4` in
`online_moe.py:255` die Verteilungscharakteristik der restlichen 252
Experten repräsentativ erfasst.

Script: `tools/validate_moe_sample.py`. Volle Auswertung:
`measurements/20260421-moe-sample-validation/VALIDATION_REPORT.md`.

| Modell | Experten | MoE-Blöcke | first-4 ≠ full | Fall A | Fall B |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-122B-A10B | 256 | 96 | **11 (11.5 %)** | 10 | 1 |
| Qwen3.5-35B-A3B | 256 | 80 | 0 (0 %) | 0 | 0 |
| GLM-4.7-Flash | 64 | 94 | 0 (0 %) | 0 | 0 |

**Fazit:** für 35B und GLM ist `sample_experts=4` provably ausreichend.
Für 122B gibt es 10 Fall-A Fälle (first-4 pickt xfp3 wo full xfp4 pickte).
Produktiv **ohne Math-Regression** (98 % auf GSM8K), weil Outlier-Pfad
den Bit-Verlust absorbiert. Weitere Empfehlungen (sample_experts=16 oder
stratified sampling) im VALIDATION_REPORT.md §Schlussfolgerungen.

Layer-0-Anomalie `gate_up_proj min/med cos=0.0 below_gate=255/256`:
geklärt — der Script interpretiert den Dense-MLP-Layer als stacked MoE;
live wird er korrekt über Dense-MLP-Pfad geroutet.

## Geänderte/neue Dateien

```
M  vllm/multiquant/policy.py               # :141 linear_attn classifier fix
M  vllm/multiquant/xfp/online_moe.py       # silu_and_mul fused + topk_weights in kernel
M  start.multiquant                         # --nvtx, --profile, --cpu-offload-gb flags, live-mounts erweitert
A  vllm/multiquant/_profiler.py             # NVTX helper stub (unused — dynamo rejects custom CMs)
A  tools/validate_moe_sample.py             # 4-expert-sample validation
A  PAPER_XFP.md                             # vollständig überarbeitet mit 122B results
A  RESULTS.xfp.v12.md                       # 122B ausgefüllt, 35B + GLM TBD
A  XFP_COS.md                               # doku: wie/wieviel/warum 0.98 cos-gate
A  TASK_testrunde_xfp.md                    # dieses dokument
A  measurements/20260419-xfp-vs-marlin/     # COMPARISON + traces + summaries
A  measurements/20260420-xfp-mtp/           # COMPARISON + 5× bench.txt
A  measurements/20260421-moe-sample-validation/  # läuft
```

## ⚠ IMMER BEI JEDEM XFP-Start mitloggen: Distributions-Table

Das Load-Log hat per-Layer-Zeilen in der Form

```
INFO [online_linear.py:350] XFP ? [20480x3072] -> xfp3(auto) | mse=5.77e-06 cos=0.984 | 3sigma=0.6% | outliers=0.163% (k=4.0)
INFO [online_moe.py:264]    XFP MoE auto-select: bits=4 (from 4 expert sample, lloyd=5)
INFO [online_moe.py:353]    XFP MoE: 256 experts w13[2048x3072] + w2[3072x1024] -> xfp (fused, fpe=786432/393216, lloyd=5)
```

Diese **immer** direkt nach dem Start abgreifen und nach
`measurements/<datum>-xfp-distributions/<model>.log` ablegen:

```bash
podman logs mq-serve 2>&1 | grep -iE '[online_linear|online_moe]\.py.*XFP' \
  > measurements/20260421-xfp-distributions/<model>.log
```

Wir haben sie früher immer vergessen einzusammeln — enthält die einzigen
Beweise für per-Layer cos/mse/sigma/outlier-Distribution unterm
live-geflossenen 0.98-cos-gate. Ohne diese Zeilen sind Behauptungen zu
Auto-Mode-Bit-Selection ("99% der Layer landen auf xfp3") nicht belegbar.

Bereits gesichert:
- `measurements/20260421-xfp-distributions/qwen3.5-122b-a10b-xfp-auto.log`
  (324 Zeilen, 48 layers × ~6 linears + 47 MoE-Gruppen)

Noch zu sichern:
- Qwen3.5-35B-A3B XFP auto (beim nächsten Start)
- GLM-4.7-Flash XFP auto v12 (beim nächsten Start)

## Noch zu tun

1. **Validation-Ergebnisse** auswerten sobald `pgrep -af validate_moe_sample`
   leer ist. Report in `RESULTS.xfp.v12.md §4` + XFP_COS.md §4 integrieren.
2. **Qwen3.5-35B-A3B E2E Bench**: XFP auto vs Marlin INT4 AutoRound,
   identisch wie 122B. Start:
   ```bash
   ./start.multiquant --model Qwen3.5-35B-A3B \
     --weight-dtype xfp --kv fp8 --weight-dtype-lm-head fp8
   # warten, bench.py, dann:
   ./start.multiquant --model Qwen3.5-35B-A3B-int4-AutoRound \
     --kv fp8 --weight-dtype-lm-head fp8
   ```
   Erwartet: XFP ~100 tok/s long (Skalierung 3.33× vs 122B via active-params);
   Albond's Marlin-single-step 65-67 tok/s → XFP sollte +50% darüber liegen.
3. **GLM-4.7-Flash v12 Re-run**: aktuelles XFP auto + fp8 KV + fp8 LMH
   gegen Marlin auf GLM. Die historischen v8-Zahlen im Paper (XFP4 32.7
   vs Marlin 53.6) sind veraltet.
4. **MTP-Head remap fix** in `vllm/model_executor/models/qwen3_5_mtp.py`
   Zeile 267 `remap_weight_names`: q/k/v Tensoren bei MTP-Head zu
   `qkv_proj` fusionieren. Dann MTP-Speedup vs albond's 127 tok/s peak
   messbar (nur dann wird's Apples-to-Apples).
5. **Paper **: wenn 35B+GLM-Zahlen da sind, PAPER_XFP.md §7.5 ergänzen,
   §8 Limitations (Punkt 2) "Single-model E2E" streichen.
6. **Layer-0 XFP-auto-Anomalie** (0 cos auf layer0.gate_up_proj): im
   Validation-Ergebnis nachgehen. Hypothese: layer 0 ist der
   Dense-MLP-Layer und das stacked-Tensor-Format dort ist nicht
   [E,N,K] sondern [1,N*?,K]. Script müsste das erkennen.
7. **Tagebuch** (Task #4 seit Tagen pending) — bei Gelegenheit.

## Open bugs & Workarounds in Place

- **`Qwen3_5MoeMTP` qkv-fusion**: ~~blockt MTP-Speedup~~ **GEFIXT
  2026-04-22**. Die Lösung war doch nicht „4 Zeilen" — `packed_modules_mapping`
  alleine reicht nicht, AutoWeightsLoader konsultiert die Tabelle nicht
  direkt. Nötig: manuelles `load_weights` auf
  `Qwen3_5MultiTokenPredictor` nach Muster `qwen3_next_mtp.py` (~80
  Zeilen stacked_params_mapping + expert_params_mapping Iteration,
  Zeilen 154-232 in qwen3_5_mtp.py). MTP NST=3 liefert +9.4 % long,
  Math 98 % erhalten. Fix einzureichen in vLLM upstream.
- **vLLM hook-NVTX + CUDA Graph**: `--enable-layerwise-nvtx-tracing`
  produziert nur top-level ranges, nicht per-Layer (weil inner forward
  in compiled region). Reicht aber für unsere Analyse.
- **custom `with _nvtx(...)` CMs**: reverted aus qwen3_next.py/online_moe/fused_marlin.
  Dynamo bricht (gb0208 + gb0142). `_profiler.py` Stub bleibt.
- **torch.profiler `active_iterations`**: der `/start_profile`-Pfad
  braucht `--profiler-config '{"profiler":"torch",...,"active_iterations":N}'`
  sonst Pydantic wirft. `output_dir` heißt `torch_profiler_dir`. In
  start.multiquant `--profile`-flag setzt das korrekt.

## Albond-Benchmark zum Abgleich (externe Quelle)

`albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4` (GitHub, Stand 2026-04-14)
auf 35B-A3B adaptiert, reportete:

| Config | 35B tok/s | Notiz |
|---|---:|---|
| BF16 baseline | 30.7 | FlashInfer, fp8 KV |
| INT4 AutoRound + FlashInfer | 66.8 | single-step peak vs XFP |
| + MTP-2 + INT8 LMH v2 | 113–127 | peak mit spec decoding |

Unser Single-Step-Ziel auf 35B: **über 67 tok/s XFP** (skalierte 122B
→ 35B-Erwartung ~100 tok/s). Dessen Peak 127 können wir nur schlagen
wenn MTP-Head-Fix greift.

## Image-Referenz

- `localhost/vllm-multiquant:xfp_speed` (42 GB, 2026-04-21)
- `localhost/vllm-multiquant:latest` (identisch)
- `localhost/vllm-multiquant:mq_2603_quantcache` (älter, vor linear_attn-Fix)

Git: branch `multiquant`, Commit `683d80d8b`, remote
`https://github.com/flash7777/vllm`.

## Benchmark-Referenz (Single-Step Decode, no spec)

| Model | Config | long | medium | Math | Eff. bits |
|---|---|---:|---:|---:|---:|
| Qwen3.5-122B-A10B | **XFP v2** | **29.9** | **34.8** | **98%** | ~3.97 |
| Qwen3.5-122B-A10B | Marlin INT4 AutoRound | 25.8 | 29.6 | 94% | 4.00 |
| Qwen3.5-35B-A3B | TBD | — | — | — | — |
| GLM-4.7-Flash | TBD (v12) | — | — | — | — |
| GLM-4.7-Flash | v8 XFP4 (historisch) | 32.7 | — | 66% | 4.0 |
| GLM-4.7-Flash | Marlin INT4 (historisch v8) | 53.6 | — | 54% | 4.0 |

---
Letzte Aktualisierung: 2026-04-21 ~22:00 CEST
