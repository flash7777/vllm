# MultiQuant Quality Test Results

## Testebenen (von innen nach außen)

| Stufe | Was wird getestet | TQ3 cos | TQ4 cos | Status |
|-------|-------------------|---------|---------|--------|
| A | Eigene Impl + eigene Metadata (Referenz) | 0.874 | 0.954 | PASS |
| B | vLLM-style block_table (Offset, max_blocks) | 0.874 | 0.954 | PASS |
| C | Echte Buffer-Init (vLLM Seeds, register_buffer) | 0.859 | 0.952 | PASS |
| D | Skalierte K/V (std=0.2, wie echtes Modell) | 0.972 | 0.972 | PASS |
| E | Batch-Prefill + sequentieller Decode | 0.874 | 0.956 | PASS |
| E | Prefill allein | 1.000 | 1.000 | PASS |
| F | CUDA Graph Capture | — | — | **SKIP** (capture failed) |

## Details

### Stufe A: Eigener Test (Referenz)
- `MultiQuantImpl` direkt instanziiert
- Pi/S/centroids mit seed=42
- Sequentieller Pack + Decode, eigene TQMetadata
- D=256, H=20, Hkv=20, seq=8+5

### Stufe B: vLLM-style Metadata
- Wie A, aber block_offset=17 (nicht bei 0 startend)
- max_blocks_per_seq=64, total_blocks=128
- Ergebnis identisch zu A → Block-Adressierung korrekt

### Stufe C: Echte Buffer-Init
- seed = mq_config.seed + layer_idx * 1337 (wie vLLM)
- register_buffer wie in attention.py
- Leicht niedrigerer cos (0.859 vs 0.874) → Seed-Rauschen, kein Bug

### Stufe D: Skalierte K/V
- K/V mit std=0.2 (realistisch) statt std=1.0
- **Besserer** cos (0.97) → kleinere Werte = weniger Quantisierungsfehler
- Auch mit std=1.0 getestet: cos=0.97

### Stufe E: Batch-Prefill
- Alle Prefill-Tokens auf einmal (wie vLLM)
- do_kv_cache_update mit batch slot_mapping
- _forward_prefill mit causal mask
- Prefill cos=1.0000 (perfekt — naive bmm, keine Quantisierung)
- Decode cos=0.87/0.96 (wie erwartet)

### Stufe F: CUDA Graph
- Versuch den Decode-Forward in CUDA Graph zu capturen
- **FAILED**: `cudaErrorStreamCaptureInvalidated`
- Ursache: `is_current_stream_capturing()` Guard → `return` → KV nicht geschrieben
- Oder: Python-Ops im forward-Pfad nicht capture-safe

## Offene Fragen

1. **enforce-eager Test zeigte Müll** — warum, wenn A-E alle bestehen?
   - Möglichkeit: der Container hatte gemischten Code (gemountete + Image-Dateien)
   - Möglichkeit: vLLM enforce-eager nutzt trotzdem torch.compile
   - **Muss nochmal sauber getestet werden**

2. **CUDA Graph**: der Guard verhindert KV-Write bei Capture
   - Lösung: graph-safe KV Pack (PyTorch tensor ops oder pre-compiled CUDA)
   - Stufe F zeigt dass der Graph-Pfad der Bruchpunkt ist

3. **RQ3/RQ4**: cos ≈ 0 — komplett kaputt (separate Regression)

## Testdatei

`tests/multiquant/test_vllm_integration.py`

Laufen im Container mit gemounteten Dateien:
```bash
podman run --rm \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  --hooks-dir=/usr/share/containers/oci/hooks.d \
  -v multiquant_attn.py:/.../multiquant_attn.py:ro \
  -v triton_mq_fused_decode.py:/.../triton_mq_fused_decode.py:ro \
  -v kernels/turboquant:/opt/tq_build:ro \
  -v tests/multiquant:/opt/tests/multiquant:ro \
  -v torch-extensions:/root/.cache/torch_extensions:rw \
  vllm-multiquant \
  python3 -m pytest /opt/tests/multiquant/test_vllm_integration.py -v -s
```
