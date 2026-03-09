# Pitfalls Research

**Domain:** Full fixed-point LLM inference infra (W8A8 + static quantized serving)
**Researched:** 2026-03-09
**Confidence:** MEDIUM-HIGH

## Critical Pitfalls

### Pitfall 1: Calibration dataset mismatch ("calibrated on easy text, served on hard traffic")

**What goes wrong:**
Static activation scales look fine offline but production prompts (longer context, code/math, multilingual, tool-heavy chat) trigger clipping and quality collapse, often breaching the <5% quality-loss gate.

**Why it happens:**
Teams calibrate with too few samples, wrong sequence-length distribution, or generic corpora that do not match serving traffic. SmoothQuant/W8A8 is very sensitive to activation outliers.

**How to avoid:**
- Build a **traffic-shaped calibration set** (domain mix + realistic prompt lengths + system prompts + chat formatting).
- Track and version calibration artifacts (dataset hash, tokenizer version, model revision, quant config).
- Add **calibration coverage checks**: per-layer activation max/percentiles on calibration vs replay traffic.
- Recalibrate whenever tokenizer/template/model revision changes.

**Warning signs:**
- Offline perplexity looks acceptable, but online task metrics regress sharply.
- Regressions concentrated on long prompts / specific domains (e.g., math/code).
- Layer-wise activation clipping ratio spikes on replay traffic.

**Phase to address:**
Phase 1 (evaluation harness + traffic replay) and Phase 2 (calibration pipeline).

---

### Pitfall 2: Assuming W8A8 means "fully integer path" while hidden float islands remain

**What goes wrong:**
Project claims full fixed-point inference, but critical ops (norm, rotary, attention internals, residual paths, softmax-adjacent flows, KV cache handling) silently run in fp16/bf16. Performance and numerical behavior are inconsistent, and comparisons become misleading.

**Why it happens:**
Most frameworks quantize linear/matmul first; non-linear and attention paths are harder. Kernel availability drives fallback behavior.

**How to avoid:**
- Define and enforce an **integerization contract** per op class (Linear, RMSNorm, RoPE, attention, MLP activation, KV cache, sampling path).
- Emit runtime instrumentation: per-op dtype counters and fallback logs per request.
- Treat any unplanned float fallback as failing CI for "full-integer" milestones.

**Warning signs:**
- Good memory wins but disappointing latency/token improvements.
- Inference logs show mixed dtypes despite integer mode enabled.
- Different hosts produce different fallback patterns.

**Phase to address:**
Phase 2 (integer kernel integration) and Phase 3 (full-path integerization hardening).

---

### Pitfall 3: Per-token dynamic activation quantization sneaks into a "static" serving target

**What goes wrong:**
Serving path unintentionally uses dynamic per-token activation scales (runtime-dependent), reducing determinism/reproducibility and making static-quality claims hard to trust.

**Why it happens:**
Many W8A8 implementations default to dynamic activation quantization when static input scales are absent; this is easy to miss.

**How to avoid:**
- Enforce explicit mode selection: `static` vs `dynamic` activation quantization as a required launch/weight metadata field.
- Validate checkpoint completeness (`input_scale`, `weight_scale`, zero-point policy) at load time.
- Fail-fast if static mode is requested but static scales are missing.

**Warning signs:**
- Identical prompt + seed gives unstable token-level outputs across runs.
- Quantized checkpoint loads without static scale tensors but still serves.
- "Static" eval artifacts cannot be exactly reproduced on another node.

**Phase to address:**
Phase 2 (quantized checkpoint schema + loader validation).

---

### Pitfall 4: Saturation/clipping blind spots in int8 matmul pipelines

**What goes wrong:**
Model quality degrades due to systematic clipping/saturation in activation or intermediate integer math, especially on outlier-heavy layers.

**Why it happens:**
Teams monitor only end metrics and ignore low-level quant stats (clamp counts, range occupancy, per-layer error).

**How to avoid:**
- Add quant observability: per-layer clip ratio, int8 histogram occupancy, max-abs before/after quant.
- Introduce automatic guardrails: reject artifacts with clip ratio above thresholds.
- Keep a dequantized shadow-run debug mode for selected batches to localize loss.

**Warning signs:**
- Degradation appears suddenly after one quant config tweak.
- A few layers dominate total quantization error.
- Long-context tasks regress more than short-context tasks.

**Phase to address:**
Phase 2 (quant debug tooling) and Phase 3 (acceptance gates).

---

### Pitfall 5: Wrong scale granularity choices (per-tensor where per-channel/per-token needed)

**What goes wrong:**
Using coarse scales where fine granularity is needed causes excessive error; using overly fine granularity where unsupported hurts throughput and portability.

**Why it happens:**
Granularity is chosen for convenience, not hardware+model compatibility.

**How to avoid:**
- Standardize quant policy matrix by op type: e.g., linear weights per-channel static; activations explicitly static-per-tensor (if true static target) or per-token dynamic (if not).
- Benchmark each policy on **quality + throughput** jointly before locking defaults.
- Keep hardware-specific policy profiles (Hopper/Ampere/etc.).

**Warning signs:**
- Acceptable latency but poor quality, or vice versa.
- Quantized model works on one GPU generation but regresses on another.

**Phase to address:**
Phase 2 (policy search) and Phase 4 (hardware profile tuning).

---

### Pitfall 6: KV-cache precision strategy not treated as first-class

**What goes wrong:**
Even if linear path is W8A8, leaving KV cache precision unmanaged can erase memory/latency gains or cause long-context quality instability.

**Why it happens:**
Teams focus on GEMM quantization and postpone KV cache decisions.

**How to avoid:**
- Define KV precision policy early (bf16/fp8/int8) and test long-context quality separately.
- Add KV memory pressure tests (context-length sweeps) to acceptance.
- Include KV precision in model artifact metadata and runtime config.

**Warning signs:**
- Throughput drops sharply with context growth despite W8A8 linear path.
- OOM/eviction behavior inconsistent across workloads.

**Phase to address:**
Phase 3 (end-to-end path) and Phase 4 (scaling/long-context hardening).

---

### Pitfall 7: Tensor-parallel scale/metadata inconsistency across ranks

**What goes wrong:**
Different TP ranks use mismatched scales or quant metadata interpretation, creating silent divergence and non-deterministic outputs.

**Why it happens:**
Quant metadata sharding/loading logic is less mature than weight sharding logic; rank-local assumptions leak in.

**How to avoid:**
- Add distributed consistency checks: hash scales/metadata per rank after load.
- Unit-test quantized TP load for merged/split projections (QKV/O-proj/MLP split points).
- Include cross-rank parity test in CI (single-GPU vs TP outputs on fixed prompts).

**Warning signs:**
- Single-GPU quality passes, TP quality fails.
- Rare token glitches only in multi-GPU deployment.

**Phase to address:**
Phase 3 (distributed integer serving).

---

### Pitfall 8: Quantized checkpoint contract drift

**What goes wrong:**
Model loads but semantics are wrong because naming/schema drifted (`weight_scale`, `input_scale`, zero-point assumptions, transpose conventions). Failures surface as subtle quality loss instead of hard errors.

**Why it happens:**
No strict schema/versioning; loader tries to be permissive.

**How to avoid:**
- Version quantized checkpoint schema explicitly.
- Validate required tensors + shapes + dtypes + axis semantics before serving.
- Add compatibility tests for each produced artifact version.

**Warning signs:**
- New quantized artifact version "loads fine" but quality shifts unexpectedly.
- Different exporters produce same tensor names with different semantics.

**Phase to address:**
Phase 2 (artifact schema) and Phase 3 (serving validation gate).

---

### Pitfall 9: Backend capability assumptions (kernel exists ≠ kernel selected)

**What goes wrong:**
Deployment falls back to slower/non-integer path due to kernel availability, shape constraints, or backend incompatibilities; SLOs miss despite "quantized" build.

**Why it happens:**
Capability checks are done at install time only, not at runtime per-shape/per-op.

**How to avoid:**
- Add startup capability report + runtime fallback counters.
- Define "supported serving envelope" (batch/seq/head dims, dtype, GPU arch).
- Gate production rollout on fallback rate thresholds from replay traffic.

**Warning signs:**
- Performance highly sensitive to batch shape.
- Different pods with same image show different throughput.

**Phase to address:**
Phase 3 (runtime observability) and Phase 4 (production readiness).

---

### Pitfall 10: Quality gate defined too narrowly (single benchmark overfitting)

**What goes wrong:**
System passes one benchmark (e.g., GSM8K) but fails real production tasks; project ships "successful" quantization with hidden regressions.

**Why it happens:**
Narrow acceptance criteria chosen for speed.

**How to avoid:**
- Use a **multi-metric gate**: perplexity + instruction following + long-context + latency/throughput + stability.
- Keep bf16 shadow baseline and automated relative-delta reporting.
- Separate smoke metrics (fast CI) from release metrics (representative evaluation suite).

**Warning signs:**
- Metric parity on one task, user complaints on others.
- Regressions only visible in multi-turn or tool-augmented flows.

**Phase to address:**
Phase 1 (evaluation strategy) and Phase 4 (release gating).

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Hardcode one global scale policy for all layers | Faster implementation | Hidden quality cliffs on specific layers/models | MVP only, with explicit TODO + telemetry |
| Silent float fallback when int kernel unavailable | "It still works" demos | Invalid full-integer claims, unpredictable perf | Never for milestone sign-off |
| Reusing stale calibration artifacts across model revisions | Saves calibration time | Undetected quality drift | Only for local experiments, never release |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| OpenAI-compatible serving + evalscope | Comparing across runs with drifted prompts/templates/tokenizer | Freeze chat template/tokenizer/system prompts in eval manifest |
| Quantized artifact loader | Accepting partial metadata (`weight_scale` present, `input_scale` missing) | Strict schema validation by serving mode (static/dynamic) |
| Multi-GPU TP launch | Assuming scale metadata shards align like weights | Add rank-wise metadata hash/parity checks at startup |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Per-token dynamic quant in "static" deployment | Extra runtime overhead, unstable latency | Enforce static activation scales for static mode | High-QPS online serving |
| Frequent dtype conversions (int8↔bf16↔int32) | Kernel timeline dominated by cast ops | Fuse casts and audit per-op dtype transitions | Medium batch + long decode |
| Quantized math but unfixed scheduler settings | No throughput gain despite lower precision | Tune scheduling jointly with quant path; profile end-to-end | As concurrency increases |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Trusting external quantized model artifacts without validation | Malformed or poisoned checkpoints causing crashes or silent corruption | Verify artifact signatures/checksums and schema before load |
| Exposing debug telemetry with internal tensors in prod endpoints | Potential leakage of model internals/user prompt fragments | Restrict debug APIs, redact/scope telemetry |

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| No explicit model precision metadata in API responses/health | Users cannot reason about quality/perf changes after rollout | Return serving precision profile/version in status endpoints |
| Quantized rollouts without canary comparison | Sudden quality regressions for all tenants | Canary with bf16 shadow + automatic rollback thresholds |

## "Looks Done But Isn't" Checklist

- [ ] **W8A8 enabled:** Verify per-op dtype/fallback report, not just env var.
- [ ] **Static quantization complete:** Verify static activation scales are present and used.
- [ ] **<5% quality drop achieved:** Verify on representative multi-task suite, not one benchmark.
- [ ] **TP support ready:** Verify single-GPU and TP parity on fixed seeds/prompts.
- [ ] **Production-ready:** Verify fallback rate, clipping telemetry, and rollback plan exist.

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Calibration mismatch | MEDIUM | Rebuild traffic-shaped calibration set, regenerate scales, rerun full eval gates |
| Hidden float fallbacks | MEDIUM | Enable dtype/fallback tracing, patch unsupported ops, lock CI guardrails |
| Checkpoint schema drift | HIGH | Freeze exporter/loader versions, add schema migration + compatibility tests, re-export artifacts |
| TP metadata inconsistency | HIGH | Add rank-consistency validators, patch sharding rules, revalidate single-vs-TP parity |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Calibration dataset mismatch | Phase 1-2 | Replay traffic eval + layer clipping dashboards |
| Hidden float islands | Phase 2-3 | Per-op dtype report must match integerization contract |
| Dynamic quant in static target | Phase 2 | Loader fails without static scales in static mode |
| Saturation/clipping blind spots | Phase 2-3 | Clip ratio thresholds enforced in CI/release |
| Scale granularity mismatch | Phase 2 | Policy benchmark matrix (quality+latency) signed off |
| KV-cache precision neglect | Phase 3-4 | Long-context quality+memory sweeps pass |
| TP scale inconsistency | Phase 3 | Single-GPU vs TP parity tests pass |
| Checkpoint contract drift | Phase 2-3 | Schema validator + compatibility tests pass |
| Backend capability assumptions | Phase 3-4 | Fallback rate under threshold on canary workload |
| Narrow quality gate | Phase 1 & 4 | Multi-task release gate with bf16-relative deltas |

## Sources

- ONNX Runtime quantization docs (static vs dynamic, debugging APIs, saturation caveats, GPU constraints): https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html (**HIGH**)
- TensorRT-LLM numerical precision / quantization docs (W8A8 SmoothQuant, scaling modes, model/hardware support): https://nvidia.github.io/TensorRT-LLM/reference/precision.html and https://nvidia.github.io/TensorRT-LLM/features/quantization.html (**HIGH**)
- SmoothQuant official repository (activation outliers, calibration scales, integrations): https://github.com/mit-han-lab/smoothquant (**MEDIUM-HIGH**, official repo + paper links)
- vLLM INT8 W8A8 docs (ecosystem serving pattern, calibration workflow emphasis): https://docs.vllm.ai/en/stable/features/quantization/int8/ (**MEDIUM**, page content partially truncated)
- Mini-SGLang project code (current failure surfaces: mode switching, quant scale loading, quantized linear behavior):
  - `python/minisgl/layers/quantization/w8a8_int8.py`
  - `python/minisgl/layers/base.py`
  - `python/minisgl/layers/linear.py`
  - `python/minisgl/models/register.py`
  - `python/minisgl/server/args.py` (**HIGH**, first-party code)

---
*Pitfalls research for: quantized/fixed-point LLM serving infrastructure*
*Researched: 2026-03-09*
