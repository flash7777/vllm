# TASK.fix.md — Testgetriebene Entwicklungsstrategie MultiQuant

## Ausgangslage

- Unit Tests (cos 0.86-0.95) bestehen — Algorithmus ist mathematisch korrekt
- Live-Serving produziert Müll (`15+27=54`, Mozart="(the")
- Auch der "stabile" Commit `8d373f2ba` hat das Math-Problem
- **Lücke:** Unit Tests decken den Live-Pfad nicht ab

## Root Cause Hypothese

Der Unit Test testet **1 Layer, eager, kleine Sequenz**.
Live GLM-4.7 hat **47 Layer, CUDA Graphs, variable Sequenzen**.
Die `_python_decode_reference()` in test_fused_decode.py (Zeile 54) ist die
einzige korrekte Referenz — sie existiert NUR im Testcode, nicht im Serving.

## Strategie: Von innen nach außen testen

### Stufe 1: Reinen Python-Algorithmus verifizieren (KEIN CUDA)

**Ziel:** Stimmt die Mathematik ohne jeden Kernel?

Test: Pure Python Pack→Decode Round-Trip mit GLM-4.7 Dimensionen:
- D=256, num_heads=20, num_kv_heads=20, seq_len=100, 47 Layer
- Pack: `_pack_batch()` (Python bitpack)
- Decode: `_python_decode_reference()` aus test_fused_decode.py
- Referenz: naive BF16 bmm Attention
- Vergleich: cos_sim pro Layer + akkumuliert

**Keine CUDA Kernel, kein JIT, kein Image nötig.**
Kann im Container mit gemounteten Dateien laufen.

### Stufe 2: CUDA Kernel vs Python Referenz

**Ziel:** Produziert der CUDA Kernel identische Ergebnisse wie Python?

Test: Gleiche Daten wie Stufe 1, aber:
- Pack: Python `_pack_batch()`
- Decode: CUDA `tq_fused_decode_attention()` 
- Vergleich: byte-für-byte gegen Python Decode

**JIT Cache löschen → Kernel wird frisch kompiliert.**
Mount: `kernels/turboquant/` + `triton_mq_fused_decode.py`

### Stufe 3: vLLM Forward-Simulation (1 Layer)

**Ziel:** Stimmt `forward()` mit Prefill + Decode + KV Cache Update?

Test:
- Erstelle MultiQuantImpl mit GLM-4.7 Dimensionen
- Prefill: 10 Tokens → `_forward_prefill()` + `do_kv_cache_update()`
- Decode: 20 Tokens sequentiell → `forward()` pro Token
- Referenz: gleiche Tokens durch naive BF16 Attention
- Vergleich: cos_sim pro Decode-Step

**1 Layer, eager, keine CUDA Graphs.**
Mount: nur `multiquant_attn.py`

### Stufe 4: Multi-Layer Forward (47 Layer Simulation)

**Ziel:** Akkumuliert der Fehler über die Layer?

Test:
- 47× Layer-Forward hintereinander (wie vLLM Model-Forward)
- Jeder Layer hat eigene Pi/S/centroids (verschiedene Seeds)
- Input Layer N+1 = Output Layer N (echte Residual-Kette)
- Vergleich: cos_sim pro Layer vs BF16 Referenz

### Stufe 5: vLLM Integration (Serve + Request)

**Ziel:** Funktioniert das Gesamtsystem?

Test: Container mit gemounteten Dateien, `--enforce-eager`:
- BF16 Modell + TQ3, Mozart-Frage + Math
- Erst wenn Stufe 1-4 bestehen

## Test-Infrastruktur

### Dateien mounten (kein Build nötig)

```bash
podman run --rm \
  -v multiquant_attn.py:/.../ multiquant_attn.py:ro \
  -v triton_mq_fused_decode.py:/.../ triton_mq_fused_decode.py:ro \
  -v kernels/turboquant:/opt/tq_build:ro \
  -v tests/multiquant:/opt/tests/multiquant:ro \
  -v torch-extensions:/root/.cache/torch_extensions:rw \
  vllm-multiquant python3 -m pytest ...
```

### JIT Cache löschen (Kernel-Neukompilierung erzwingen)

```bash
rm -rf /data/sources/torch-extensions/py312_cu*/tq_fused_decode/
rm -rf /data/sources/torch-extensions/py312_cu*/tq_pack_kv/
rm -rf /data/sources/torch-extensions/py312_cu*/clifford_sandwich/
```

### Test-Datei

Alle Tests in `tests/multiquant/test_fix_math.py`:
- `TestPurePythonRoundTrip` (Stufe 1)
- `TestCUDAvsReference` (Stufe 2)
- `TestSingleLayerForward` (Stufe 3)
- `TestMultiLayerAccumulation` (Stufe 4)

Parametrisiert: D=[128,256], dtype=[tq3,tq4], heads=[4,20]

## Reihenfolge

1. Schreibe `test_fix_math.py` mit Stufe 1
2. Laufe im Container → bestanden? → weiter
3. Stufe 2 hinzufügen → bestanden? → weiter
4. Stufe 3 → wenn hier fehlschlägt: Bug ist im forward/kv_update Pfad
5. Stufe 4 → wenn hier fehlschlägt: Fehler-Akkumulation über Layer

## Erwartete Ergebnisse

- Stufe 1: PASS (Python Algo ist korrekt, Unit Test bestätigt)
- Stufe 2: PASS (CUDA Kernel matched Python bei D=128, fraglich bei D=256)
- Stufe 3: HIER erwarte ich den Bug — forward() mit Prefill+Decode+KV
- Stufe 4: Akkumulation zeigt ob cos 0.86 pro Layer über 47 Layer degradiert
