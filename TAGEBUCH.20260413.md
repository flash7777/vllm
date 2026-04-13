# Tagebuch 2026-04-13: Fused XFP MoE Kernel

## Fortführung von gestern

### Memory-Fix: BF16 sofort freigeben
- `_batched_pack_and_repack`: `del W_stack`, `del W_flat`, etc. nach Benutzung
- `layer.w13_weight.data = torch.empty(0)` nach Pack
- Ergebnis: 78 GB frei während Packing (vorher System-Hang bei Layer 44)

### Lloyd-Iterationen: 20 → 5 für MoE
- MoE-Experts homogen → 5 Iterationen reichen
- Parametrisierbar via `XFP_MOE_LLOYD_ITERS` Env-Var
- Packing-Zeit: 15 Min → 5 Min (4× schneller)

### Fused MoE Kernel Tests
- Single Expert: cos=1.0 PASS
- Multi-Expert (4 Experts, 2 topk): cos=1.0 PASS  
- Full MoE Pfad (gate_up → SiLU → down → reduce): cos=1.0 PASS
- Bug gefunden und gefixt: Down-GEMM muss top_k=1 + identity_ids nutzen
  (activated ist pro sorted-Entry, nicht pro Original-Token)

### E2E Problem
- Server startet, aber Modell gibt Müll aus (0% Math)
- Kernel isoliert korrekt — Problem ist in der vLLM-Integration
- Debug-Logging eingebaut, nächster Server-Start läuft

## E2E Ergebnisse

### Bug gefunden: Kernel schreibt C[token_id] in Original-topk-Order
Der Kernel schreibt `C[token_id]` wobei token_id aus sorted_token_ids kommt.
Das platziert Ergebnisse an der ORIGINALEN topk-Position, nicht der sortierten.
Python-Seite las es als sorted order → falsche Expert-Zuordnung.

Fix: gate_up/down als [B*topk, N] allokieren, Down-GEMM mit topk_ids direkt
als expert_ids, Scatter-Reduce über [0..BT) mit topk_weights.

### Fused MoE XFP E2E Benchmark (enforce-eager)

| Config | tok/s (long) | Math |
|--------|-------------|------|
| XFP4 attn+shared only, BF16 MoE (eager) | 24.6 | 66% |
| XFP4 attn+shared only, BF16 MoE (CUDA Graphs) | 32.5 | 66% |
| **XFP4 ALL mit fused MoE Kernel (eager)** | **29.5** | **56%** |
| XFP4 ALL mit fused MoE Kernel (CUDA Graphs) | CRASH | - |

### CUDA Graphs Crash
`cudaErrorStreamCaptureUnsupported` — moe_align_block_size oder Python-seitige
Tensor-Allokationen im apply() nicht Graph-kompatibel. Braucht custom op Wrapper
wie bei xfp_apply/xfp_outlier_scatter.

### Nächste Schritte
1. Custom op für MoE apply (torch.compile + CUDA Graphs kompatibel)
2. Dann Bench mit CUDA Graphs → erwartung ~35+ tok/s
3. Profiling des fused MoE Kernels vs Marlin
