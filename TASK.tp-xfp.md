# TASK — XFP TP=2 via Cache-Only Load (2026-04-22+)

Ziel: **397B XFP auf DGX+PGX (je 128 GB UMA) via TP=2 ausführbar machen**,
ohne dass die ~800 GB BF16-Source auf beiden Nodes gleichzeitig Disk
belegen muss und ohne dass das quantisierte Modell gleichzeitig in einem
Node-VRAM-Budget Platz finden muss.

## Kontext

- **XFP braucht BF16 Source** zum Lloyd-fit (single-node bereits etabliert,
  siehe `PAPER_EVIDENCE.md` + `MATRIX_RESULTS.md`).
- **397B hat keine lokale BF16** — nur Intel/AutoRound-INT4 (auf DGX+PGX)
  + GPTQ-Int4 (nur DGX). BF16-Repo `Qwen/Qwen3.5-397B-A17B` existiert
  auf HuggingFace (~800 GB).
- **PGX Disk-Limit**: NVMe-Root nur 916 GB total. 800 GB BF16 passt
  nicht + OS + Logs + Cache.
- **DGX UMA-Limit**: 128 GB für CUDA + Host — 397B **quantisiert**
  (~200 GB) passt nicht einmal als Serving-Footprint, geschweige denn
  als Serving + Lloyd-Staging zusammen.
- Der laufende Marlin-397B-TP=2 Run (ohne XFP) demonstriert zwar, dass
  die Ray/NCCL-Orchestrierung via `start.multiquant.tp2` auf der
  192.168.0.x Direktlink funktioniert — aber XFP ist das eigentliche
  Paper-Ziel.
- **Ray-TP=2 XFP-Streaming hat bekannten Bug**: `w2_weight` wird auf
  Ray-Workern nicht aus meta → CUDA materialisiert, `process_weights_after_loading`
  crasht mit `w2.shape=(0,)`. Diagnostic-Log im online_moe.py:245-262
  bestätigt das — beide Ranks betroffen. Fix benötigt tieferes
  Verständnis der Ray-Weight-Distribution; **nicht Blocker für 397B wenn
  wir Phase 2 (quant-to-cache-only) einführen**, weil damit TP=2 nie
  einen Pack-Durchlauf macht sondern nur cache-only-Load.

## Gesamtplan: 3 Phasen (refined 2026-04-23)

### Architektur — "Quant-to-Cache-Only" Modus

**Kernidee**: Pack-Schritt entkoppelt vom Serve-Schritt. Während des
Packs werden die quantisierten Tensoren **nur auf Disk geschrieben**,
nicht im VRAM als `nn.Parameter` auf den Layern zurückgehalten. Das
erlaubt:

1. **Peak-UMA-Budget** während Pack: ~1 Layer BF16 + Lloyd-Intermediate
   + 1 Layer packed-transient (alles vor cache.save wieder freigegeben)
   — schätzungsweise ~15 GB UMA pro Layer. 397B mit 48 MoE-Layern
   sequentiell: Peak bleibt **immer unter 20 GB UMA**, statt ~210 GB
   (wenn alle packed Parameter auf Layer blieben).
2. **Pack ist komplett single-node** auf DGX. Kein Ray, kein TP>1,
   kein Multi-Node-Orchestrierung. Bug #43-w2.shape wird umgangen statt
   gefixt.
3. **Cache ist TP-blind** (siehe Phase 2b) → einmal packen, überall
   serven.

Pseudo-Code-Skizze:

```python
# Bestehender Pfad in online_moe.py process_weights_after_loading:
...
p13, cb13, fpe13, stats13 = _expertwise_pack_and_repack(w13)
layer.w13_weight.data = torch.empty(0)
p2, cb2, fpe2, stats2 = _expertwise_pack_and_repack(w2)
layer.w2_weight.data = torch.empty(0)
if reg is not None: reg.record_stats(...)

# Cache IS already saved (existierender Pfad)
...

# NEU: im quant-only-Modus keine Parameter-Zuweisung auf dem Layer
if _env_truthy("MULTIQUANT_QUANT_ONLY"):
    del p13, cb13, p2, cb2
    torch.cuda.empty_cache()
    layer._xfp_moe_packed = True   # für Idempotenz, falls mehrfach aufgerufen
    return  # skip nn.Parameter Zuweisungen unterhalb

# Normal serving path (unverändert):
layer.w13_xfp_packed = nn.Parameter(p13.to(device), ...)
...
```

Analog in `online_linear.py`.

Nach letztem Layer: `_finalize_multiquant_cache` (schreibt Manifest +
Residuals). Dann `sys.exit(0)` im quant-only-Modus um CUDA-Graph-Capture
+ profile_run zu skippen (würden OOMen ohne Parameter).

### Phase 1 — Single-Node Quant-to-Cache-Only auf DGX

Zweck: einmalig den XFP-Cache für 397B auf DGX generieren, **ohne dass
das quantisierte Modell jemals in VRAM steht**.
Voraussetzung: genug Disk-Platz auf DGX, keine PGX-Änderungen nötig.

1. **DGX Disk-Cleanup** (bereits teilweise durch, 2026-04-22):
   - ✓ `Qwen3.5-REAP-262B-*` (263 GB) gelöscht
   - ✓ `Qwen3.5-397B-A17B-GPTQ-Int4/` (220 GB) gelöscht
   - ✓ `glm-47-NVPF4/` (188 GB) gelöscht
   - ✓ `deepseek-coder-v2-instruct-awq/` (120 GB) gelöscht
   - Frei: **1.2 TB** vor Download-Start

2. **BF16 397B download** (läuft seit 2026-04-22, resumed 2026-04-23):
   ```
   hf download Qwen/Qwen3.5-397B-A17B \
     --local-dir /data/tensordata/Qwen3.5-397B-A17B-BF16 \
     --max-workers 8
   ```
   Stand 2026-04-23 morgens: ~120 GB / ~800 GB geschrieben, 13 von 107
   safetensors-Shards. Download muss nach reboot/timeout neu angeschoben
   werden (`nohup` + `disown` empfohlen).

3. **Quant-to-Cache-Only Run** (neu):
   ```
   MULTIQUANT_QUANT_ONLY=1 \
   ./start.multiquant --model Qwen3.5-397B-A17B-BF16 \
     --weight-dtype xfp --kv fp8 --weight-dtype-lm-head fp8 --tp 1
   ```
   streaming_loader arbeitet weiter layer-by-layer (BF16 → CUDA →
   Lloyd → cache.save → free). Unterschied zum alten Pfad: keine
   nn.Parameter-Zuweisung der packed tensors an den Layern, VRAM bleibt
   pro Layer unter ~15 GB Peak, steady-state nahe 0 nach letztem Layer.
   `sys.exit(0)` nach `_finalize_multiquant_cache` um Graph-Capture/
   profile_run zu skippen.
   Ergebnis: `mq-cache/Qwen3.5-397B-A17B-BF16/<hash>/` ~200 GB +
   `residuals.safetensors` ~3 GB.

4. **Sanity-Check auf 35B nachziehen** bevor 397B pack läuft:
   Dasselbe Flag `MULTIQUANT_QUANT_ONLY=1` auf 35B (BF16 haben wir,
   Cache haben wir, Mem-Footprint kennen wir) → Cache muss byte-
   identisch zum bestehenden sein, und der Run muss sauber mit
   Exit-Code 0 enden ohne Graph-Capture-OOM. **Erst danach 397B.**

5. **Single-Node Bench** auf 397B — hier brauchen wir **einen separaten
   Serve-Run** (ohne `MULTIQUANT_QUANT_ONLY`). Aber: 397B quantisiert
   ~200 GB passt nicht in 128 GB UMA single-node → Bench-Run ist nur
   via TP=2-cache-only (siehe Phase 3) möglich. Kein single-node 397B-
   Sanity-Bench. Stattdessen: 35B + 122B XFP-Paper-Zahlen sind unsere
   Sanity, Phase 3 liefert dann direkt den 397B TP=2 Bench.

6. **BF16 bleibt auf DGX** für Re-Pack-Experimente (verschiedene
   XFP-Varianten werden kommen: andere `XFP_MIN_COS`, andere
   `XFP_MOE_SAMPLE_EXPERTS`, andere `lloyd_iters`, andere
   `outlier_sigma` — jede Variante = neuer Cache-Shard, BF16 als
   shared source). Platzbudget auf DGX post-Phase-1:
   - 800 GB BF16 source
   - 200 GB XFP cache (erste Variante)
   - zusätzliche ~200 GB pro weitere Variante
   → mit 600 GB Cleanup vor Phase 1 sind ~1.4 TB frei → Raum für
   BF16 + 3 Cache-Varianten parallel.

### Phase 2 — Cache-only Load + Quant-only + TP-blindes Cache (Code)

Zweck: vLLM soll (a) in einem dedizierten Quant-Only-Run nur Cache
schreiben ohne Serving, (b) später von diesem Cache ohne BF16 laden
und (c) dieser Cache ist TP-blind (Pack unter TP=1 funktioniert für
beliebige TP=N serve).

**Design-Entscheidungen (Standard bis anders beschlossen):**

- **F1 Residuals**: separate `residuals.safetensors` ✓ schon implementiert
  (2026-04-22).
- **F2 Aktivierung**: `--load-format multiquant` ✓ schon implementiert.
- **F3 Cache-Miss-Verhalten**: hart-error ✓ schon implementiert.
- **F4 Scope für 1. PR**: **nur XFP**. Marlin/AutoRound-Cache-only
  kommt später.
- **F5 Quant-only-Aktivierung**: env `MULTIQUANT_QUANT_ONLY=1` (neu,
  siehe unten). Einfacher als LoadFormat-Enum weil es ein orthogonaler
  Runtime-Mode ist, kein separater Loader-Pfad.
- **F6 Cache-TP-Blindheit**: `tp_size` + `ep_size` aus Cache-Key-Hash
  entfernen. Die gepackten Tensoren sind pro-Layer-vollständig
  `[E, N, K]` und werden beim Load-Assignment gesliced (von vllm oder
  unserem cache-load-Code).

**Implementations-Schritte (Status 2026-04-23):**

1. ✓ **Residual-Dump beim Pack** — in `weight_cache.py:save_residuals()`
   implementiert, automatisch am Ende von `_finalize_multiquant_cache`.

2. ✓ **`MultiQuantCacheOnlyLoader`** — in
   `vllm/model_executor/model_loader/multiquant_loader.py` implementiert,
   registriert als `"multiquant"` LoadFormat.

3. ✓ **`--load-format multiquant`** pass-through in `start.multiquant`
   und `start.multiquant.tp2`.

4. **Quant-to-Cache-Only Mode** (NEU, ~15 LOC in online_moe +
   online_linear + base_loader):
   ```python
   # online_moe.py / online_linear.py am Ende von process_weights_after_loading:
   import os
   if os.environ.get("MULTIQUANT_QUANT_ONLY", "").lower() in ("1", "true", "yes"):
       # Cache already saved above. Skip nn.Parameter assignment → VRAM
       # retention stays zero. Mark idempotent for potential re-call.
       layer._xfp_moe_packed = True  # oder _xfp_packed_done für Linear
       del p13, cb13, p2, cb2  # oder packed, codebook, outlier_*
       import torch
       if torch.cuda.is_available():
           torch.cuda.empty_cache()
       return
   # Normal serving path (unverändert):
   layer.w13_xfp_packed = nn.Parameter(...)
   ...
   ```
   Plus in `base_loader._finalize_multiquant_cache(model)`: nach allen
   Phasen (Manifest + Residuals-Dump + log_summary), wenn Flag gesetzt,
   `sys.exit(0)` bevor `process_weights_after_loading` → Graph-capture
   → profile_run → OOM.

5. **Cache-TP-Blindheit** (NEU, ~2 LOC in `weight_cache.py`):
   In `compute_cache_key()` die Zeile `f"|tp={tp_size}|ep={ep_size}"`
   streichen. Alternativ: nur behalten wenn env
   `MULTIQUANT_CACHE_TP_SPECIFIC=1`. Default = TP-blind.

6. **Verifikation Cache-TP-Blindheit**:
   - Bestehenden 35B TP=1 cache-Hash notieren (z.B. `f7e92ebafdf406c7`)
   - Nach F6-Änderung TP=1 starten — Hash muss unverändert bleiben
     (oder sich ändern, aber dann einmalig repacken)
   - Cache-Shard zu PGX kopieren
   - TP=2 mit `--load-format multiquant` starten — Hash muss derselbe
     sein wie TP=1 und cache.load_moe muss die full-[E,N,K]-Tensoren
     korrekt auf beide Ranks splitten

7. **streaming_loader Graceful-Skip** (bereits umgesetzt):
   - `MultiQuantCacheOnlyLoader.load_weights` stubs meta-Params auf
     size-0 real-device, `process_weights_after_loading` greift dann
     den cache.load_moe / cache.load_linear Pfad ab.

8. **`start.multiquant` — `--quant-only` Flag**:
   ```bash
   --quant-only)  QUANT_ONLY=true; shift ;;
   ```
   setzt `MULTIQUANT_QUANT_ONLY=1` env im podman run.
   `start.multiquant.tp2` braucht diesen Pfad NICHT (pack ist per
   Design single-node). Der Flag existiert also nur in `start.multiquant`.

### Phase 2b — Cache-Transfer DGX→PGX (pro Variante)

Nach jedem erfolgreichen Pack einer neuen XFP-Variante den zugehörigen
Shard nach PGX spiegeln (BF16 bleibt ausschließlich auf DGX):
```
# Shard-Hash z.B. via ls identifizieren (letzter Timestamp)
SHARD=<hash>  # z.B. c7ba067a3c20ecce
rsync -av --progress --info=progress2 \
  -e "ssh" /data/tensordata/mq-cache/Qwen3.5-397B-A17B-BF16/$SHARD/ \
  flash@192.168.0.116:/data/tensordata/mq-cache/Qwen3.5-397B-A17B-BF16/$SHARD/
```
~200 GB über 200 Gb/s direct link = ~2 min ideal, ~10-15 min real.
Mehrere Shards (eine pro XFP-Variante) können koexistieren — PGX hat
nach Cleanup ~700 GB frei, also Raum für mind. 3 parallele Varianten.

### Phase 3 — TP=2 XFP-Run (cache-only)

Nachdem Cache + Residuals auf beiden Nodes stehen, der Cache TP-blind
ist, und der cache-only Loader auf TP=2 Ray-Workern funktioniert:

```
./start.multiquant.tp2 --model Qwen3.5-397B-A17B-BF16 \
  --weight-dtype xfp --kv fp8 --weight-dtype-lm-head fp8 \
  --load-format multiquant
```

Wichtig: **`--load-format multiquant` umgeht den Ray-TP-Streaming-Bug
komplett**, weil beim cache-only-Load keine meta→material-Transition
per streaming_loader nötig ist — `MultiQuantCacheOnlyLoader` füllt die
meta-Tensoren direkt aus cache.load_moe / cache.load_linear.

Peak-UMA pro Node: ~100 GB (split-share der 200 GB Cache + 10 GB KV +
Activations). Passt locker auf 128 GB UMA.

Bench-Erwartungen:
- Long ~5-8 tok/s (397B active-params + NCCL-AllReduce-Overhead über
  Ethernet). Albond berichtete ~10 tok/s single-node RTX — auf GB10 TP=2
  lower wegen cross-node AllReduces.
- Math: unabhängig von TP, sollte single-node-XFP-Ergebnis
  reproduzieren (XFP-bits sind byte-identisch).

## Verification

### Phase 1 (Quant-Only)
- 35B Sanity: `MULTIQUANT_QUANT_ONLY=1 ./start.multiquant --model Qwen3.5-35B-A3B-BF16 ...`
  → exit 0 nach letztem Layer-Pack, kein `Graph-Capture`, kein `profile_run`
- Cache-Shard-Inhalt byte-identisch zum vorher unter normalem Flow gepackten
- Peak-UMA im Monitor < 30 GB (nicht der volle Serving-Footprint)
- Dann dasselbe auf 397B-BF16: Cache 200 ± 20 GB, Manifest, Residuals

### Phase 2 (Loader + TP-Blindheit)
- `./start.multiquant --load-format multiquant` startet Server ohne
  BF16-Source präsent (BF16 dir `mv` oder `--undo`-NFS-mount)  ✓ bereits
  auf 35B single-node verifiziert
- **Neu zu verifizieren**: Cache-Hash ändert sich NICHT zwischen
  TP=1-pack und TP=2-load — TP-Blindheit bestätigen per
  `cat /data/tensordata/mq-cache/<model>/*/manifest.json | jq .cache_key`
- Produziert identische Bit-Decisions wie der erste Pack
- Math-Output identisch single-node vs cache-only (innerhalb Lloyd-Noise)

### Phase 3 (TP=2 cache-only)
- 2 Ray-Nodes ALIVE (checked via `ray status` im head-container)
- NCCL-AllReduces über `enp1s0f0np0` (Trace: `NCCL_DEBUG=INFO` zeigt
  socket-if)
- Bench läuft durch, tok/s > 0, Math > Baseline-Rausch-Grenze
- Ray-TP-w2.shape-Bug wird **umgangen** (cache-only-load → keine
  streaming_loader meta→material Transition für MoE-Weights)

## Artefakt-Struktur

```
measurements/20260423-397b-xfp/
├─ PHASE1_single_node.md      # Pack + single-node bench
├─ PHASE2_loader_design.md    # Code-Walk-Through multiquant_loader.py
├─ PHASE3_tp2_bench.md        # TP=2 numbers + NCCL traces
├─ bench-397b-singlenode.txt
├─ bench-397b-tp2.txt
└─ distributions/
   └─ qwen3.5-397b-a17b-xfp-auto.log
```

Plus code:
```
vllm/multiquant/weight_cache.py            # + residuals dump
vllm/multiquant/cache_only_loader.py       # NEW, oder unter model_loader/
vllm/model_executor/model_loader/
  └─ multiquant_loader.py                  # NEW
  └─ __init__.py                           # + registration
vllm/model_executor/model_loader/utils.py  # streaming_loader graceful-skip
vllm/config/load.py                        # LoadFormat enum
start.multiquant, start.multiquant.tp2     # --load-format pass-through
```

## Aufwandsschätzung (aktualisiert 2026-04-23)

| Phase | Status | Aufwand | Risiken |
|---|---|---|---|
| 1. Cleanup (~600 GB) | ✓ 791 GB gelöscht | — | — |
| 1. BF16 Download (800 GB von HF) | ⏳ 120/800 GB, resumed | ~10 h wallclock | HF throttle (800 sec/file) |
| 1. Quant-Only Mode (~15 LOC) | ⏳ offen | ~1 h dev + 30 min test | Edge-Cases bei Linear vs MoE Pfad |
| 1. Sanity-Pack 35B mit Quant-Only | ⏳ offen | ~20 min | muss cache-byte-identisch sein |
| 1. 397B Pack quant-only | ⏳ offen | ~30 min wallclock (nach Download) | Peak-UMA darf nicht explodieren |
| 2. Residuals-Dump | ✓ done | — | — |
| 2. MultiQuantCacheOnlyLoader | ✓ done, single-node verified | — | — |
| 2. `--load-format multiquant` | ✓ done | — | — |
| 2. Cache-TP-Blindheit (~5 LOC) | ⏳ offen | ~15 min dev + 30 min test | TP=2 slicing am load-Zeit untested |
| 3. Cache-Transfer | ~15 min wallclock | — | — |
| 3. TP=2 Bench | ⏳ offen | ~30 min | Ray bootstrap Edge-Cases |
| **Gesamt Dev (restlich)** | | **~2-3 h reine Codearbeit** | |
| **Gesamt wallclock** | | **~15 h** | + Download parallel |

## Nicht-Ziele / später

- Cache-only für Marlin/AutoRound-Pfade (nicht in 1. PR)
- LoRA-Support in cache-only mode
- Adapter-loading (bleibt bf16-basiert)
- Multi-Rank-Sharding des Cache (für TP>2, momentan speichern wir
  pro-Layer komplette Tensoren, Rank liest seinen Slice raus)

## Offene Fragen

1. Wie groß sind die Residuals für 397B in der Praxis?
   Erste Schätzung: embed_tokens = 6144 × 152064 × 2 = 1.9 GB +
   48 RMSNorm × 6144 × 2 = 600 KB +
   e_score_correction_bias = 256 × 48 × 4 = 50 KB +
   MTP-Head-Layer falls nicht fp8'd = ~150 MB.
   **Total ~2 GB**. Fits easy.

2. Kann `process_weights_after_loading` outside of streaming_loader
   sicher aufgerufen werden? → das testen wir mit dem Loader.

3. Was passiert wenn Config-Hash sich ändert (z.B. neue vLLM-Version)?
   → Cache-Miss, hard-error, PR-Beschreibung muss das erklären.

## Status

- **2026-04-22 abend**: Task erstellt nach 397B-TP=2-Startversuch der
  am PGX-Disk-Limit scheiterte. Phase 1 noch nicht begonnen.
- **2026-04-22 nacht**: Cleanup durch, BF16-Download gestartet (107
  files, ~800 GB). Cache-only Code-Stack komplett:
  Residuals-Dump + `MultiQuantCacheOnlyLoader` + `--load-format
  multiquant`. Single-Node verifiziert auf 35B (cache-only lädt ohne
  BF16-Source + antwortet sensibel "Paris" / "2+2=4").
- **2026-04-23 früh morgens**: NFSv4 DGX→PGX live (1 GB/s dd-read über
  Direktlink), BF16-Symlink auf PGX. TP=2-pack-Versuch auf 35B via
  NFS-BF16 scheiterte: Ray-Worker material w2_weight nicht aus meta
  → `w2.shape=(0,)` auf beiden Ranks, `IndexError` in
  online_moe.py:281. Diagnostic-Warning eingebaut, Root-Cause
  lokalisiert in Ray-Meta→Material-Pfad. Download pausiert bei
  120/800 GB.
- **2026-04-23 frühmorgen refine**: Plan umgestellt auf
  **Quant-to-Cache-Only + TP-blinde Caches**. Damit Ray-TP-Bug wird
  umgangen statt gefixt: pack läuft ausschließlich single-node TP=1
  (wo alles stabil ist), TP=2-serve nutzt cache-only-Load (nie in
  streaming_loader meta-Transition).
- **Parallel laufend**: 397B-BF16 Download resumed mit 8 workers +
  nohup. ETA ~10 h bis komplett.
- **Nächster Step (morgen)**: 1. `MULTIQUANT_QUANT_ONLY` Flag
  einbauen (~15 LOC), 2. Cache-Hash TP-blind machen (~5 LOC),
  3. Sanity-Check mit 35B, 4. Nach Download-Complete → 397B pack
  quant-only, 5. Cache→PGX rsync, 6. TP=2 bench cache-only.
