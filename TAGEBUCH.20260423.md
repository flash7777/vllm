# Tagebuch 2026-04-23: Quant-only + Cache-only + Eval-Harness

## Ausgangslage

- 397B BF16 Download läuft (pid 96327, 214G/800G ≈ 26%, seit ~3,5 h)
- 35B XFP cache-only Server seit 37 min auf :8011 (serving `/data/tensordata/Qwen3.5-35B-A3B-BF16`)
- Plan `pr-fe-die-speicherbereinugng-bei-elegant-goose` (Teil 1 + Teil 2) wartete
  auf Umsetzung: Pack-Stats persistieren + Eval-Container bauen.

## Erledigt heute

### Teil 1: Quant-only + Cache-only + Pack-Stats (Commit f4e7d7fda)

- `MULTIQUANT_QUANT_ONLY=1` Pfad in `online_linear.py` / `online_moe.py`:
  Tensoren werden nach Pack via `p.data = torch.empty(0, ...)` freigegeben,
  damit kein Gewicht im VRAM/UMA persistiert. `base_loader._finalize_multiquant_cache`
  schreibt `xfp_summary.json` und `sys.exit(0)` vor Graph-Capture.
- `--quant-only` / `--load-format` Flag in `start.multiquant` durchgereicht.
- `weight_cache.py` Cache-Key: `tp_size`/`ep_size` standardmäßig raus
  (opt-in via `MULTIQUANT_CACHE_TP_SPECIFIC=1`). Ziel: ein Pack für beliebiges TP.
- Env-Snapshot in `write_manifest()` iteriert jetzt alle `XFP_*` + `MULTIQUANT_*`
  Env-Vars statt zwei hartcodierter Strings.
- `XFPPackStats` erweitert: `cos_hist` (20 Bins [0,1]), `outlier_hist` (5 σ-Bänder),
  `bits_survived_gate`, plus `to_dict()` Serializer.
- `save_linear` / `save_moe` in `xfp_weight_cache.py` nehmen Stats über
  `layer._xfp_stats` / `layer._xfp_moe_stats{13,2}` ab und mergen in
  `_manifest.json` Metadata.
- `policy.py` `build_summary_json()` → `{by_layer_class, totals, hyperparams}`.
- `tools/pack_report.py` (~200 LOC): Cross-Shard Aggregator für Paper-Tabellen.

### Teil 2: Eval-Harness Container (Commit f21d8a4fe)

- `eval-harness/Dockerfile` — `python:3.12-slim-bookworm` +
  `lm-eval[api]==0.4.11`, numpy/scipy/datasets/transformers/tokenizers →
  918 MB `localhost/vllm-eval-harness:lm-0.4.11`.
- `bench.lm-eval` Wrapper: Preflight `/v1/models` → `.data[0].id` +
  `.data[0].root`, pro Task × Seed podman run mit `--network host` und
  `/data/tensordata:/data/tensordata:ro` Mount für lokale Tokenizer-Auflösung.
- `bench_lm_eval/aggregate.py` — per-Seed `results.json` → mean/std/CI95
  (scipy t-Interval) in `<task>.summary.json`.
- `bench_lm_eval/render_table.py` — `<date-dir>/*/*.summary.json` →
  cross-label Markdown mit CI95-Footnote.

### Smoke-Test

```
./bench.lm-eval --label smoke-35b-xfp --tasks gsm8k --seeds 1 --limit 10
```

Output: `flex-extract 0.2, strict-match 0.0` in 90 s. Layout + Pipeline OK.
Aggregator + render_table erzeugten valide `gsm8k.summary.json` + `REPORT.md`.

## Aktuelle Beobachtungen

- Existierender 35B XFP Cache unter
  `/data/tensordata/mq-cache/Qwen3.5-35B-A3B-BF16/83af0af92eb02e9c/xfp_summary.json`
  hat `by_layer_class: {}` und `hyperparams: {MULTIQUANT_CACHE_DIR: …}` — wurde
  vor heutigem Stats-Commit gepackt. Für Pack-Report-Demo müssen wir re-packen.
- Server läuft auf altem Cache stabil (81 tok/s avg). Kein Re-pack nötig für
  Eval-Harness Matrix; Bench läuft HTTP-only.

## Nachmittag: RTX Rollout (Spiegel 2, 2× RTX PRO 6000)

Da DGX 397B BF16 Download bei 27 % stagniert, Strategie-Umschwung: Spiegel 2
hat 397B BF16 schon komplett (`/data/tensordata/Qwen3.5-397B-A17B`, 752 GB).

- Git-Repo auf Spiegel 2 von `2344912` auf `142739edf` gezogen. Symlink
  `/root/vllm-riy → vllm-mq` durch echtes Verzeichnis ersetzt (podman statfs
  deref den Symlink nicht). Umbenennung `mv /root/vllm-mq /root/vllm-riy`.
- **Bug gefunden**: Commit f4e7d7fda hat
  `vllm/model_executor/model_loader/__init__.py` auf
  `from .multiquant_loader import MultiQuantCacheOnlyLoader` erweitert, aber
  die neue Datei `multiquant_loader.py` nicht mit `git add` gemacht. Auf DGX
  lief's (untracked Datei), auf Spiegel 2 fehlte sie → podman statfs fail.
  Fix: Nachcommit `142739edf`.
- **397B Quant-only-Pack gestartet** auf Spiegel 2 (TP=1, GPU 0):
  Policy: 512 experts × 59 MoE-Layer + 60 Attn-Layer + Dense Layer 0.
  Erwartete Pack-Dauer ~3-4 h.
- **Serve-Plan nach Pack** (TP=2, MIT RIY 24 %, 128K Kontext):
  INT4 AutoRound + RIY 24 % hat auf 2× 97 GB schon gepasst → XFP
  (avg ~3.3 bits < INT4) passt erst recht, kein CPU-Offload. KV-Budget
  für 2× parallele 128K-Sessions wird validiert sobald Pack steht.

## Abend: 397B Pack fertig, Serve OOM, zwei Folge-Bugs gefixt

### 397B XFP Pack done (Spiegel 2, 71 min, 17:00 UTC+2)

- Cache-Hash: `edcc981f77c55a20` bei `/data/tensordata/mq-cache/Qwen3.5-397B-A17B/`
- **eff_bits/param: 3.11** (5.2× compress vs BF16), cache **175 GB + 4 GB residuals**
- By class: attn 3.01 (165 layers, 154×xfp3 + 11×xfp4, cos 0.983) / routed_expert 4.0
  (alle xfp4, cos 0.993) / shared_expert 3.14 (cos 0.984)
- Total: 405 layers, 8.39B XFP-packed params, 0.1% outliers
- Exit war kosmetisch buggy: `sys.exit(0)` in `_finalize_multiquant_cache` wird
  vom vLLM APIServer als "Engine init failed" geloggt, aber alle Artefakte sind
  geschrieben. Cache funktioniert.

### Serve TP=2 (ohne RIY): OOM 94.65 GB/GPU

Vollständig gescheitert: weights belegen 94.65 GB pro GPU auf 2× 97 GB, KV + activations
passen nicht drauf. Meine Vorab-Schätzung (87 GB/GPU) war zu optimistisch — XFP-Codebooks
(BF16) + Outlier-Sparse (BF16) + dequant workspace + CUDA graph preallocation machen
~7 GB extra pro GPU.

### Serve TP=2 mit RIY 24%: immer noch OOM (94.17 GB/GPU)

RIY 24 % sparte 0.5 GB VRAM statt der erwarteten ~15 GB. **Root-cause gefunden**:
`multiquant/xfp/xfp_weight_cache.py::load_moe` ignorierte `layer._expert_map` und
lud unverändert alle 512 experts. Der classic-Pfad in `fused_moe/layer.py:1130` skippt
korrekt via `expert_id == -1 → return False`, aber unser MultiQuantCacheOnlyLoader
geht nicht durch `weight_loader`. Feature-Lücke im neuen cache-only-Path.

**Fix** (Commit `18844bd58`): `load_moe` detektiert `local_E < global_E`, stageed
auf CPU, gathered nur kept experts, moved zum Device. Cache-Layout bleibt
RIY-agnostisch (pack schreibt alle 512), damit ein Cache-Shard jedes RIY-Profile
serven kann.

### Parallel: Pack 0.95 (XFP_MIN_COS aggressiv) — ENV-BUG entdeckt

User wollte zweiten Pack mit `XFP_MIN_COS=0.95` für aggressiveres xfp2/3. Erster
Versuch: `start.multiquant` reichte die Env-Var nicht an podman durch. Pack lief
mit default 0.98 und **überschrieb den bestehenden 0.98-Cache (gleicher Hash)**.
Glück im Unglück: gleiche Inputs → deterministic identische Outputs → kein Schaden.

**Fix** (Commit `2ea29c57e`): `start.multiquant` reicht jetzt `XFP_MIN_COS`,
`XFP_MOE_LLOYD_ITERS`, `MULTIQUANT_CACHE_TP_SPECIFIC` durch.

### Status Ende 18:00

- **DGX 35B XFP bench**: seed 0 done (flex 8.49 %), seed 1 done (9.70 %),
  seed 2 bei 40 %. Base-Modell, erwartungsgemäß niedrig.
- **Spiegel 2 397B Pack 0.95**: gerade gestartet mit korrekter env
- **DGX 397B Download**: 347 G / 800 G (43 %), User hat Platz gemacht, läuft weiter.
- **397B-Serve TP=2 mit RIY-Fix**: noch nicht getestet (wartet auf freien GPU)

## Next

1. 0.95-Pack abwarten (~1-1.5 h). Dann zwei Configs zum Vergleich: 0.98 + RIY-Fix vs. 0.95 ohne RIY.
2. 35B Bench seed 2 abwarten (~75 min).
3. 397B TP=2 Serve testen, bench gegen lm-eval harness.
