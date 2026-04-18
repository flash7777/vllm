# Tagebuch 2026-04-18: Qwen 122B XFP Cache — sauberer Cold-Start

## Ausgangslage (Fortsetzung von 04-17)

Gestern abend: Weight-Cache-Code war in f492530e9 gecommittet, aber das
Image war 6 Tage alt (vor 3139c06a2 gebaut) — hatte weder Meta-Swap-Fix
noch Cache-Code. Alle Cold-Start-Versuche scheiterten:

1. OOM beim initialize_model (kein Meta-Swap) → Kernel-OOM-Killer → UMA-Leak → Reboot
2. Falscher `--gpu-memory-utilization 0.33` Default in start.multiquant
3. `--weight-dtype xfp` wurde nicht an vllm durchgereicht (Script-Bug)

Heute: Image neu bauen, alle Bugs fixen, Cold-Start durchziehen, einchecken.

## Bug-Fixes (in Reihenfolge gefunden)

### 1. start.multiquant: UMA-Defaults

Der CUDA-Profiler lügt auf GB10 UMA (meldet immer `<9 GiB frei`).
Default auf `--gpu-memory-utilization 0.05 --kv-cache-memory-bytes 10G`
umgestellt; für discrete GPUs bleibt `0.95`.

### 2. start.multiquant: WDTYPE und WLMHEAD wurden verschluckt

Das Script parste `--weight-dtype xfp` aber fügte es nie an `QUANT_ARGS`
an. Policy zeigte `bf16 → bf16` obwohl `xfp` gesetzt war. Fix:

```bash
[[ -n "$WDTYPE" ]]  && QUANT_ARGS+=("--weight-dtype" "$WDTYPE")
[[ -n "$WLMHEAD" ]] && QUANT_ARGS+=("--weight-dtype-lm-head" "$WLMHEAD")
```

### 3. Image-Neubau (commit f492530e9)

`./build.sh` dank ccache in ~20 min durch. Commit stimmt mit HEAD überein.
Dockerfile kopiert jetzt auch `kernels/multiquant` nach `/opt/mq_kernels`
(sonst fehlt das Verzeichnis im Image).

### 4. KV-dtype `tq3` ist kaputt — Code-Level-Rejection

Plain `tq3`/`tq4` (ohne Walsh-Hadamard oder block-rotation) degeneriert
mathematisch auf der long-tailed KV-Verteilung → Kauderwelsch-Output.
Nur `tq3w`/`tq4w`/`tq3r`/`tq4r` sind brauchbar. Habe ich heute zum zweiten
Mal verwechselt — User hatte es schon mehrmals gesagt, Memory war da aber
nicht prominent genug. Fix:

- `vllm/multiquant/policy.py`: `_reject_broken_kv_dtype()` raised
  ValueError wenn jemand `tq2/tq3/tq4` ohne Suffix nutzt.
- `start.multiquant` Default auf `tq3w`.
- Memory `feedback_tq_kv_dtype.md` angelegt.

### 5. `/opt/mq_kernels` fehlte im Image (kritisch)

Erster Cold-Start-Versuch: Pack + Cache-Write lief durch, aber beim
`profile_run` crashte der Forward-Pass mit

```
RuntimeError: xfp custom op called before xfp_gemm kernel was loaded.
```

Root cause: `xfp_kernel._resolve_kernel_dir()` sucht relativ zum Install-
Pfad oder unter `/opt/mq_kernels` — beides existierte im Image nicht.
Der erste `_load_xfp_gemm()` setzte `_load_attempted = True` nach dem
stummen Fail, alle weiteren Calls returnten None, der custom op flog
beim ersten Forward.

Fix:
- Dockerfile: `cp -r kernels/multiquant /opt/mq_kernels` (für den nächsten
  Build drin; für heute live-gemountet in start.multiquant).

### 6. Cache-Hit-Path: BF16-Leak über `layer._parameters`

Warm-Start-Crash im vorigen Versuch: CUDA alloc wuchs linear ~6 GiB/Layer
bis zum OOM bei Layer ~17. Cause: streaming_loader materialisiert BF16
weights auf CUDA, der Cache-Hit-Path in `process_weights_after_loading`
lädt die XFP-Tensoren aus dem Cache und macht danach `del layer.w<N>_weight`.
Das entfernt nur das Attribut — der `nn.Parameter` in `layer._parameters`
hält die Storage am Leben (~5 GiB pro MoE-Layer). Über 48 Layer: 240 GB
Ghost-BF16.

Fix in `online_moe.py` und `online_linear.py` Cache-Hit-Paths:

```python
for attr in ("w13_weight", "w2_weight"):
    p = layer._parameters.get(attr)
    if p is not None:
        p.data = torch.empty(0, device=p.data.device, dtype=p.data.dtype)
try:
    del layer.w<N>_weight, ...
except AttributeError:
    pass
```

Das schrumpft die Storage sofort zu 0 Bytes. Die `del` danach entfernt
das Attribut wie vorher. Analog zu der bereits korrekten Cold-Start-
Logik (`layer.w13_weight.data = torch.empty(0)` vor dem Pack).

## Cold-Start Erfolg (Run #3, 14:35–15:21)

Nach Reboot und allen Fixes:

```
./start.multiquant --model Qwen3.5-122B-A10B --weight-dtype xfp \
    --max-model-len 32768 --kv tq3w
```

### Timeline
- 14:37:17  Erstes MoE auto-select (bits=4)
- 14:37:48  `XFP GEMM kernel compiled (xfp_gemm_v12) from /opt/mq_kernels` ← Kernel-Mount greift
- 14:39:10  MoE #1 fertig gepackt
- 15:13:46  **Cache summary**: `saves=168 (592.4s) save_fail=0 load_fail=0`
            `misses={xfp_linear=120, xfp_moe=48}` — alle 48 MoE + 120 Linear
- 15:15:03  `torch.compile took 73.40 s in total`
- 15:16:09  KV-Cache reserved 10 GiB, initial free memory 113.02 GiB
- 15:21:?   `Application startup complete`
- **Gesamtdauer: 2779 s = 46:19 min**

### Cache auf Disk
- `/data/tensordata/mq-cache/Qwen3.5-122B-A10B/1846a13e35dde785/` (neuer
  Key weil `policy.py` geändert wurde — alter Key `7d579eb7c2686224` ist
  invalid)
- 57 GB, 168 safetensors-files (flach per Layer-Prefix)

### Memory nach Startup (92 GiB "used")

| Quelle | Größe |
|---|---:|
| XFP Weights auf CUDA | ~57 GiB |
| KV-Cache reserved | 10 GiB |
| Page Cache (vom BF16-Read, reclaimable) | 16 GiB |
| AnonPages/Slab/Driver | ~9 GiB |

Peak CUDA alloc im Verlauf: **70.2 GiB** (Linear-pack-Phase).

## Kapazitäts-Hochrechnung für 200-GB-Klasse

100 GB XFP + 10 GB KV + ~9 GB driver = 119 GB hart auf 120 GiB UMA.
Kein Puffer für torch.compile-Allokationen, CUDA-Graph-Captures o.ä.
Stellschrauben für später:

- `cudagraph_capture_sizes` reduzieren (aktuell 51 sizes 1..512)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `kv_cache_memory_bytes` dynamisch je nach Modellgröße

## Offen

- **Warm-Start-Test** aus dem geschriebenen Cache — sollte jetzt (mit
  BF16-Leak-Fix) in ca. 2 min durchgehen. Das ist der eigentliche Zweck
  des Caches.
- Tagebuch 20260417 war nur halb-fertig und sagt im Text "cp.async v11
  als nächster Schritt" — dieser Pfad ist seit Weight-Cache-Commit
  (f492530e9) parallel, kein akuter Drop.
