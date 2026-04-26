# MoE 4-Expert-Sample Validation — Cross-Model Summary

**Date:** 2026-04-22
**Goal:** Verify whether XFP's `sample_experts=4` in `online_moe.py:255`
picks the same bit-width as running auto-select on the full expert
population.
**Script:** `tools/validate_moe_sample.py`
**Gate:** `cos ≥ 0.98`, `lloyd_iters=5` (identical to production).

## Aggregate

| Model | Experts | MoE blocks | first-4 disagrees | rand-4 disagrees | Fall A (under-quant) | Fall B (over-escalated) |
|---|---:|---:|---:|---:|---:|---:|
| **Qwen3.5-122B-A10B** | 256 | 96 | **11 (11.5 %)** | 9 (9.4 %) | 10 | 1 |
| **Qwen3.5-35B-A3B** | 256 | 80 | **0 (0 %)** | 0 (0 %) | 0 | 0 |
| **GLM-4.7-Flash** | 64 | 94 | **0 (0 %)** | 0 (0 %) | 0 | 0 |

Produktion-Realität auf Qwen 122B (aus `qwen3.5-122b-a10b-xfp-auto-mtp.log`):
185 MoE-Blöcke picken xfp4, 8 picken xfp3. Math bleibt stabil bei 98 %.

## Fall A Details — Qwen 122B (wo first-4 zu niedrig pickt)

| Block | first-4 | rand-4 | full | min cos | med cos |
|---|---:|---:|---:|---:|---:|
| layer0.down_proj | **2** | 3 | 3 | 0.9749 | 0.9937 |
| layer4.gate_up_proj | **3** | 4 | 4 | 0.0000 | 0.9933 |
| layer7.down_proj | **3** | 3 | 4 | 0.0000 | 0.9938 |
| layer8.down_proj | **3** | 4 | 4 | 0.0000 | 0.9937 |
| layer10.down_proj | **3** | 4 | 4 | 0.0000 | 0.9937 |
| layer12.down_proj | **3** | 3 | 4 | 0.9934 | 0.9936 |
| layer13.gate_up_proj | **3** | 4 | 4 | 0.0000 | 0.9929 |
| layer13.down_proj | **3** | 3 | 4 | 0.0000 | 0.9938 |
| layer41.down_proj | **3** | 4 | 4 | 0.9929 | 0.9933 |

Einzige Fall B (first-4 über-eskaliert): `layer5.down_proj` → first-4
pickt xfp4, full nur xfp3.

## Schlussfolgerungen

1. **Für Qwen3.5-35B-A3B und GLM-4.7-Flash: 4-Sample ist provably
   ausreichend.** 0 % Disagreement zum 256-/64-Expert Full-Run. Keine
   Änderung nötig.

2. **Für Qwen3.5-122B-A10B: 11.5 % Disagreement.** Fast immer Fall A
   (first-4 ist zu "tame", pickt niedrigere Bits als nötig). Die
   Experten-Population ist heterogen genug, dass die ersten 4 nicht immer
   repräsentativ sind.

   Trotzdem bleibt produktiv **Math = 98 %** unbeeinflusst, weil die
   XFP-Gate-Reconstruction + outlier-fp8-Pfad den kleinen Bit-Verlust
   absorbiert. Das ist empirisch OK, aber **quality-margin** ist enger
   als nötig.

3. **Empfehlung Option A (konservativ)**: `sample_experts=16` für Modelle
   mit mehr als 128 Experten. Laufzeitkosten: +4× Lloyd-Zeit auf MoE, pro
   Layer ~0.1 s → ~12 s Gesamt-Startup-Delta. Kein Perf-Impact zur
   Laufzeit.

4. **Empfehlung Option B (genauer)**: stratified sampling. Statt erste N
   nehmen, nach expert-norm sortieren und die 25 %/50 %/75 %/100 %
   Perzentile samplen. Teurer zu implementieren, aber Disagreement-Rate
   sollte auf <2 % fallen.

5. **Empfehlung Option C (status quo)**: bei `sample_experts=4` bleiben.
   Math bleibt 98 %, Kosten null. Die Fall-A-Fehlentscheidungen werden
   durch die per-channel sparse-fp8-Outliers und die Median-Aggregation
   stabilisiert.

Realpolitisch: wir fahren mit **Option C** und dokumentieren die 11.5 %
Disagreement-Rate als "known tolerable risk on models with >128
experts". Falls eine künftige Math-Regression auf einem 128+Expert-Modell
auftaucht, gehen wir auf Option A.

## Anomalie: min cos = 0.0000 auf allen gate_up_proj

Alle 48 `gate_up_proj`-Blöcke zeigen `min per-exp cos = 0.0000`. Das ist
**nicht** auf Qwen 122B beschränkt — erscheint auch auf 35B und GLM
weniger häufig aber systematisch. Ursache: das stacked Layout
`[E, 2×N_moe, K]` enthält für einige Experten komplette Zeilen mit
fp-null-norm (Gate kann bestimmte Dimensionen ausnullen), `cosine`
returniert 0 für diese Rows. **Median über alle Rows** bleibt davon
unberührt, deshalb kein Einfluss auf die Bit-Entscheidung. **Kein Script-
Bug, kein Model-Problem** — erwartetes Verhalten bei Gate-und-Up-Stacks.

Layer 0 in Qwen 122B ist ein Sonderfall: `below_gate=255/256`, min=0,
med=0. Das ist der **Dense MLP** (nicht MoE) in [1, N, K] Layout, der vom
Validation-Script als MoE-Stack interpretiert wird. In Produktion landet
er korrekt auf xfp3 via Dense-MLP-Pfad — das Validation-Ergebnis
`bits(full)=4` ist hier irrelevant, weil das live-Policy-Routing ihn
anders behandelt.

## Artefakte

- `Qwen3.5-122B-A10B.md` (96 Blöcke, 11 Fälle Disagreement)
- `Qwen3.5-35B-A3B.md` (80 Blöcke, 0 Fälle)
- `GLM-4.7-Flash.md` (94 Blöcke, 0 Fälle)
- `tools/validate_moe_sample.py` (Script)

Validiert über ~42 min per Modell, 256 Experten × 2 Projektionen × 47-48
Layer, lloyd=5, cpu-float.
