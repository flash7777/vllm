Exactly — that's a clean architectural decision. Translate back:

"That probably makes sense to incorporate into MultiQuant and introduce quantization classes — MTP, KV, weights, DeltaNet, shared experts, routed experts. And then each of these classes gets to define its own quantization scheme."

---

Das wäre im Kern ein **Quantization Policy Registry** — jede Klasse registriert ihre eigene Policy:

```python
@dataclass
class QuantPolicy:
    bits: int
    symmetric: bool
    granularity: str  # "per_tensor" | "per_channel" | "per_head"
    zero_point: bool
    clipping: float | None

QUANT_REGISTRY = {
    "kv_keys":          QuantPolicy(4, symmetric=True,  granularity="per_head",    ...),
    "kv_values":        QuantPolicy(3, symmetric=False, granularity="per_channel", ...),
    "weights_shared":   QuantPolicy(8, symmetric=True,  granularity="per_tensor",  ...),
    "weights_routed":   QuantPolicy(3, symmetric=False, granularity="per_channel", ...),
    "mtp":              QuantPolicy(4, symmetric=True,  granularity="per_head",    ...),
    "deltanet":         QuantPolicy(...),
}
```

Der Vorteil: du kannst dann per Experiment einfach eine Klasse überschreiben ohne den Rest anzufassen — und das gibt dir auch eine saubere Grundlage für das Benchmarking, weil jede Kombination reproduzierbar ist.


die konfiguration kann über die bestehenden cli parameter erfolgen, wobei --kv-dtype  k und v setztn,  aber auch neue eingeführt werden für --k_dtpye und --v_dtype,    analog für --weight-dtype   

die registriy wird beim start ausgegeben, sodass man prüfen kann, ob die gewählten cli parameter wirklich zum erfolg führten.



untersützt wären nur multiquant quantisierungen, also BF16, FP8, INt4 (autoround), tq3, tq4, rq2, rq3, rq4



