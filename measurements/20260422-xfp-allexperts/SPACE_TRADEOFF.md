# XFP Speichertradeoff: bits+1 vs mehr Outliers

Paper-relevante Frage: **wieviel kostet es, einen Layer von xfp3 auf
xfp4 zu heben, und wann wäre es billiger die gleiche Qualität über mehr
Outliers statt mehr Bits zu kaufen?**

Antwort hängt vom Pfad ab — MoE-Pfad und Linear-Pfad haben **verschiedene
Speicherformate** in unserer XFP-Implementierung.

## MoE-Pfad (routed experts)

**Formatierung** — `online_moe.py:287-308`, `xfp_moe_kernel`:
- Gepacktes Index-Tensor `w_packed [E × fpe]` (uint32), enthält
  `bits`-Bit Codebook-Index pro Gewicht.
- Codebook `w_codebook [E × N × 2^bits]` (fp16).
- **Keine separaten Outliers**. Das Flag `outlier_sigma=None` in
  `_expertwise_pack_and_repack()` (Zeile 306) stellt sicher, dass der
  Pack rein codebook-quantisiert ist.
- Outlier-Handling existiert nur im **Auto-Select-Gate** (gate-decide),
  nicht im finalen Weight-Format.

**Speicher pro Gewicht bei MoE**:

| bits | dense bits/w | codebook amortized | gesamt bytes/w |
|---:|---:|---:|---:|
| 3 | 0.375 | ~5e-5 | **~0.375** |
| 4 | 0.5 | ~1e-4 | **~0.5** |

(Codebook ist vernachlässigbar: Qwen 122B routed N13=2048, lut=16 bei
bits=4 → 64 KB/Expert × 256 Experten = 16 MB/Layer auf ~1200 MB → <2 %
Anteil).

**Fazit MoE**: bits→bits+1 kostet **0.125 B/w** pauschal (+25 %
Delta-Memory pro Layer). Kein Outlier-Hebel verfügbar — die einzige
Alternative ist, das Gate strenger zu setzen (`XFP_MIN_COS=0.975`), um
xfp3 auch bei grenzwertigen Layern zu akzeptieren, und die
Qualitätsdegradation hinzunehmen.

### Konkrete Zahlen für Qwen3.5-122B-A10B

Pro Layer (E=256, w13=2048×3072, w2=3072×1024) = 2.4 Mrd. MoE-Weights:

| Config | Storage pro Layer | Delta vs xfp3 |
|---|---:|---:|
| xfp3 | 2.36 MB dense × 256 + 1.18 × 256 + cb = **884 MB** | — |
| xfp4 | 3.14 × 256 + 1.57 × 256 + cb = **1192 MB** | **+308 MB (+35 %)** |

Auf 122B mit 47 Routed-MoE-Layern: Layer-weite Escalation xfp3 → xfp4
kostet **14.5 GB** insgesamt. Empirisch (offline + 32-Layer live) flippen
aber nur **2 von 47** Layer bei all-experts — Kosten also **616 MB** auf
ein 60 GB-Modell = **~1 % Mehr-Footprint**.

## Linear-Pfad (Attention, shared expert, dense MLP)

**Formatierung** — `online_linear.py:125-142`, `xfp_outlier_scatter`:
- Gepacktes Index-Tensor `w_packed` bei `bits`-Bit pro Gewicht.
- Codebook `w_codebook` (fp16).
- **Separate Outlier-Trippels**: `outlier_row [int64]`, `outlier_col [int64]`,
  `outlier_val [fp16]` → **18 Bytes pro Outlier**.
- Outliers werden beim `xfp_outlier_scatter`-Kernel als `scatter_add`
  am Ende des GEMM-Outputs addiert.

**Speicher pro Gewicht bei Linear** (mit Outlier-Fraction `x`):

`bytes_per_weight(bits, x) = bits/8 + x × 18`

Tabelle bei üblichen Outlier-Caps:

| bits | x=0 % | x=1 % | x=2 % (default) | x=5 % | x=10 % |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.25 | 0.43 | 0.61 | 1.15 | 2.05 |
| 3 | 0.375 | 0.555 | 0.735 | 1.275 | 2.175 |
| 4 | 0.5 | 0.68 | 0.86 | 1.4 | 2.3 |

### Break-Even: xfp3+mehr-Outliers vs xfp4+weniger-Outliers

Gesucht: Outlier-Rate `y` bei xfp3 so dass Gesamtstorage = xfp4 bei
`x=2 %` (default).

`0.375 + y × 18 = 0.5 + 0.02 × 18`
`0.375 + 18y = 0.86`
`y = 0.027 = 2.7 %`

**Interpretation**: Wenn die Quality-Threshold bei xfp3 erreicht werden
kann mit **≤2.7 % Outliers**, ist xfp3+extra-outliers **gleich teuer
oder billiger** als xfp4-bei-default-2 %. Darüber gewinnt xfp4.

Anders herum: um den Memory-Vorteil von xfp3 zu erhalten (statt auf
xfp4 zu eskalieren), müssen die problematischen Channels **binnen +0.7
pp extra-Outlier-Budget** einfangbar sein. Das ist knapp.

### Konkrete Zahlen für Qwen3.5-122B-A10B Attention

Pro Attention-Layer (12 full_attn + 36 GatedDeltaNet):
- `qkv_proj [3072, 17408]`: 53.5 M weights × `bits/8 + x × 18` bytes
- `in_proj_qkvz [3072, 20480]` (GDN): 62.9 M weights

Delta xfp3 → xfp4 bei x=2 %:
- qkv_proj: 53.5 M × (0.86 - 0.735) = 6.7 MB pro Layer
- 48 Layer: ~320 MB
- Relativ zum 60 GB-Modell: **~0.5 %**

Delta xfp3 → xfp3 @ x=5 %:
- qkv_proj: 53.5 M × (1.275 - 0.735) = 28.9 MB pro Layer → wäre 4×
  teurer als die direkte xfp3→xfp4 Escalation!

**Operationelle Aussage**: im Linear-Pfad ist der Outlier-Hebel **nur
unterhalb 2.7 %** ein Gewinn. Unser Produktiv-Default ist bei 2 % Cap
(`_DEFAULT_OUTLIER_MAX_FRACTION = 0.02`) — sehr nah an der Break-Even.
Experimentell wäre ein 3 % Cap bei xfp3 ein winziger Gewinn, 5 %+ ist
schlechter als direkt xfp4 zu picken.

## Gesamtbild

| Pfad | Hebel verfügbar | Kosten xfp3→xfp4 | Kosten xfp3+5 %-outl | Paper-Takeaway |
|---|---|---:|---:|---|
| **MoE (routed)** | nur bits | +25 % (~616 MB auf 122B) | n/a | xfp4-Escalation bei 2 von 47 Layern → **+1 % Modellgröße**, 0 Math-Loss |
| **Linear (attn+dense)** | bits + outliers | +17 % | +73 % | Break-Even bei ~2.7 % Outlier-Rate → **xfp3+2 % ≈ xfp4** in memory |

**Konkrete Empfehlung für's Paper**:

1. MoE-Pfad: `XFP_MOE_SAMPLE_EXPERTS=0` (all-experts) erhöht für Qwen
   122B den Modell-Footprint um **~1 %** (2 Layer flippen xfp3→xfp4).
   Für 35B + GLM: **0 Delta** (offline-Validation zeigt 0 Disagreement).
2. Linear-Pfad: der 2 %-Outlier-Cap ist quasi-optimal. Höherer Cap
   (z.B. 5 %) ist speichertechnisch schlechter als auf xfp4 zu
   eskalieren.
3. Kein "secret memory win" durch Outlier-Magic — die Cos-Gate-Logik
   operiert bereits nahe Pareto-Optimum in Memory × Quality.

## Quellen

- `vllm/multiquant/xfp/online_moe.py:287-308` (MoE packing, keine
  Outliers)
- `vllm/multiquant/xfp/online_linear.py:125-142` (Linear scatter,
  18 bytes/outlier)
- `vllm/multiquant/xfp/xfp_pack.py:30` (v1 scope no outlier extraction
  in MoE bulk-pack)
- `xfp_pack.py:372` (`_DEFAULT_OUTLIER_MAX_FRACTION = 0.02`)
- Manifests `/data/tensordata/mq-cache/Qwen3.5-122B-A10B/{c7ba,e320}/`
  (all-experts rescued data)
