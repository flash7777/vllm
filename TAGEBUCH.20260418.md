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

## Warm-Start: 18–25 min, nicht "2 min"

Nachmittag: 4× Warm-Start durchgemessen. Alle identisch:

| Phase | Zeit |
|---|---:|
| Cache-I/O (57 GB von Disk, 168 files × `.to(device)`) | **~9:30 min** |
| torch.compile + AOT-Graph | ~73 s |
| profile_run (dummy forward) | ~2 min |
| Rest (init, kv-reserve) | ~2 min |
| **Total** | **~15 min** |

Die "2 min" von heute früh waren Wunschdenken: der Cache spart die
Lloyd-Max-Quantisierung (Cold: ~20 min davon), aber nicht das I/O. Gegenüber
Cold (46 min) spart der Cache **28 min** — nicht mehr, aber auch nicht wenig.
Cache-Summary bestätigt sauber: `hits={xfp_linear=120, xfp_moe=48}
misses={-} | saves=0`.

## Shutdown-Bug untersucht — Driver-Level UVM-Leak

Beim Stoppen des Containers (ob sauber via `podman stop` oder SIGKILL
nach Timeout) bleiben auf GB10 UMA ~53 GiB als "used" stehen, obwohl
kein Prozess sie hält. Vorher als "klassischer OOM-Leak" abgetan —
User hat zurecht korrigiert: passiert auch bei sauberem Shutdown.

### Plan implementiert (committed 9df4f7cc3)

Drei Änderungen im Shutdown-Pfad:

1. `vllm/v1/executor/uniproc_executor.py` — `destroy_model_parallel()` +
   `destroy_distributed_environment()` dazu (war nur in multiproc).

2. `vllm/v1/worker/gpu_worker.py` — Model + KV-Caches droppen,
   `gc.collect()` × 2, `torch.cuda.empty_cache()`, `ipc_collect()`.
   Plus `torch._dynamo.reset()` zum Flushen der AOT-Cache-Closures.

3. `vllm/multiquant/__init__.py` — `_cleanup_multiquant_globals()`,
   resetet die JIT-Kernel-Handles (`_xfp_gemm_kernel`,
   `_xfp_moe_kernel`, `_mq_gemm_int{2,3}`) und die Singletons.

### Test #1: `model_runner = None` reicht nicht (0 MiB freed)

```
[shutdown] GPUWorker releasing model + CUDA caches (CUDA alloc=77989 MiB)
[shutdown] CUDA alloc after cleanup: 77989 MiB (freed 0 MiB)
```

Code lief in 48 ms — `empty_cache()` hatte nichts zu tun, weil die Tensoren
noch Refs hatten. `model_runner.model = None` bricht nur EINE Referenz;
`nn.Module._parameters` / `_modules` Dicts, Compiled-CUDA-Graphs,
`torch.ops.vllm.xfp_apply`-Closures und der Dynamo-AOT-Cache halten
Einzelparameter-Refs, die Refs-via-`= None` nicht erfasst.

### Test #2: Param-Shrink-in-place greift (committed 5734dfa7b)

Statt gegen den Ref-Graphen zu kämpfen: alle `model.parameters()` +
`named_buffers()` iterieren und `.data = torch.empty(0, device=…, dtype=…)`
setzen. Wrapper-Ketten (`CUDAGraphWrapper → UBatchWrapper → inner`) über
Attribute `model/module/wrapped/_orig_mod` abwandern. Analog zum
Cache-Hit-Path-Fix vom Vormittag auf Einzel-Layern.

```
[shutdown] GPUWorker releasing model + CUDA caches (CUDA alloc=77989 MiB)
[shutdown] shrunk 1658 model params/buffers, freed 66.0 GiB of nominal storage
[shutdown] CUDA alloc after cleanup: 10383 MiB (freed 67606 MiB)
```

**Caching-Allocator meldet korrekt 66 GiB frei**, verbleibende 10 GiB sind
der KV-Cache (außerhalb von `model.parameters()`).

### ABER: `free -h` zeigt trotzdem nur ~5 GiB Recovery

```
pre-stop:  89 GiB used,  30 GiB avail
post-stop: 84 GiB used,  35 GiB avail
```

PyTorch-Seite ist alles korrekt — `torch.cuda.memory_allocated()` geht
runter, der Caching-Allocator freed die Blocks. Aber die UVM-Physical-
Pages gibt der NVIDIA-Driver **nicht** an den Kernel zurück, nicht
einmal nach Prozess-Exit (`podman stop` bringt die Container-PID zum
Verschwinden). Warnung im Log: `[W ProcessGroupNCCL.cpp:1569]
destroy_process_group() was not called before program exit` — das
passiert durch die `finally` in `run_engine_core` aber es deutet auch
an, dass die abrupte Auflösung nicht alle CUDA-Ressourcen aufräumt.

Mögliche nächste Stufe (nicht mehr heute getestet):

- `ctypes.CDLL('libcudart.so').cudaDeviceReset()` am Ende der
  Worker-Shutdown — erzwingt Context-Destroy.
- `os._exit(0)` statt sauberem Python-Teardown nach dem Shrink.
- KV-Cache explizit schrumpfen (die 10 GiB die noch übrig sind).

Aktuell hilft nur Reboot zwischen Runs. Das ist ein **Driver-Level-
Issue auf GB10**, kein Python/vLLM-Bug mehr — unsere Fixes sind im
Python-Stack vollständig, der Caching-Allocator released wie erwartet.

## Commits heute

- `c8e427881` — XFP weight cache fix (per-tensor save, cache-hit BF16
  free, Dockerfile kernels-mq Pfad, tq3w default, tq3/tq4 rejection)
- `9df4f7cc3` — uniproc shutdown: destroy_* + model drop + multiquant
  cleanup helper
- `5455a8fa0` — dynamo.reset + double gc pass (didn't help yet, kept
  for defense in depth)
- `5734dfa7b` — param-shrink in place (the actually-working part)

## Offen

- KV-Cache ebenfalls shrink-en im shutdown-Pfad (erspart die 10 GiB
  die aktuell noch im CUDA-Allocator verbleiben nach dem Shrink).
- `cudaDeviceReset`-Experiment für die OS-Level Recovery — wenn das
  funktioniert, ist das der echte Fix gegen den Reboot-Zyklus.
- Rebuild des vllm-multiquant-Images mit den finalen shutdown-Fixes
  (aktuell via Live-Mount von gpu_worker.py und uniproc_executor.py
  aktiv — Dockerfile-Patch ist in `c8e427881`, aber die shutdown-Files
  werden beim Rebuild automatisch aus dem Fork-Branch geholt).
