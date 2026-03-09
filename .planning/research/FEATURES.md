# Feature Research

**Domain:** Quantized LLM inference infrastructure (production API serving)
**Researched:** 2026-03-09
**Confidence:** HIGH

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist. Missing these = product feels incomplete.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| OpenAI-compatible API serving (`/v1/chat/completions`, streaming) | Every modern inference engine exposes OpenAI-compatible endpoints for drop-in integration | LOW | Already aligned with Mini-SGLang direction; must remain stable under integer mode. |
| Pre-quantized checkpoint loading (AWQ/GPTQ/FP8/W8A8 metadata-aware) | Users expect to run existing quantized artifacts from HF/ModelOpt-style toolchains | MEDIUM | Must parse quant config safely and fail fast on unsupported formats. |
| Quantization mode selection + clear fallback behavior | vLLM/SGLang/TGI ecosystems expose quant modes and users expect deterministic startup behavior | MEDIUM | Explicitly expose `integer_mode`, quant scheme, and whether fallback to bf16 is allowed. |
| Quality regression gating vs bf16 baseline | For quantized infra, "works" means quality loss is controlled, not just serving starts | MEDIUM | Hard gate against <5% drop (project core value); include task-level metrics and threshold checks. |
| Throughput/latency observability (TTFT, TPOT, toks/s, memory) | Production infra buyers require perf visibility and reproducible measurements | MEDIUM | Must split prefill vs decode metrics because quantization effects differ across phases. |
| Multi-GPU tensor parallel serving | Standard expectation for >7B production serving | MEDIUM | Integer kernels must behave consistently across TP ranks and NCCL paths. |
| KV-cache controls (size policy + quantized KV option roadmap) | Quantized serving is usually memory-constrained; cache behavior is a first-order feature | HIGH | Start with deterministic cache accounting; add quantized KV once integer path is stable. |
| Reproducible eval + launch workflow (one-command runbook) | Teams expect CI-like reproducibility for model launch and acceptance checks | LOW | Include pinned command templates for Qwen3-8B quantized + evalscope API-mode eval. |

### Differentiators (Competitive Advantage)

Features that set the product apart. Not required, but valuable.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| End-to-end **full fixed-point** path (not only weight-only) | Strong differentiation: many stacks support mixed precision quantization, fewer deliver stable full integer inference path | HIGH | Stage rollout: W8A8 linear -> expand to broader operator path with explicit exclusions. |
| Numerical stability guardrails (outlier handling + per-layer failover policies) | Prevents "fast but wrong" deployments; directly protects <5% quality target | HIGH | Add layer-wise diagnostics and selective precision escapes rather than global fallback. |
| Quantization acceptance dashboard (quality/perf/cost triplet) | Makes go/no-go decisions objective for infra teams | MEDIUM | Report: quality delta vs bf16, TTFT/TPOT delta, VRAM/tokens-per-dollar delta. |
| Quantization-aware scheduler policies | Better real-world efficiency: schedule decisions tuned for integer kernel behavior and memory profile | HIGH | Integrate with chunked prefill, overlap scheduling, CUDA graph capture constraints. |
| "Golden model" compatibility pack (Qwen3-8B first) | Faster adoption: opinionated, validated defaults beat generic knobs | MEDIUM | Ship known-good profiles (kernel backend, batch limits, prefill chunk) tied to hardware class. |
| Failure-mode introspection tooling | Speeds debugging of accuracy cliffs and kernel regressions | MEDIUM | Expose per-layer activation range stats / saturation indicators in debug mode. |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem good but create problems.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| "Support every quant format in v1" | Users want universal compatibility immediately | Explodes validation matrix; destabilizes quality target and delays launch | Support a strict, high-confidence subset first (W8A8 static path + selected pre-quantized formats). |
| Silent automatic fallback to bf16 on quantization mismatch | Feels convenient and avoids startup failures | Hides real failures; can fake quality/perf success and break cost assumptions | Default to fail-fast with explicit opt-in fallback flag and audit log entry. |
| Aggressive auto-tuning at runtime for all kernels | Promises peak perf | Adds startup instability, non-determinism, and hard-to-reproduce regressions | Provide curated hardware profiles + offline benchmark-based tuning artifacts. |
| Premature multi-tenant quota/billing layer | Stakeholders often ask for "platform completeness" | Distracts from core quantized inference quality/perf objective | Keep infra single-tenant focused first; export metrics needed for later platformization. |

## Feature Dependencies

```
[OpenAI-compatible API serving]
    └──requires──> [Stable integer execution path (W8A8 linear)]
                        └──requires──> [Quantized checkpoint loading + config parsing]

[Quality regression gating vs bf16]
    └──requires──> [Reproducible eval + launch workflow]
                        └──requires──> [Golden baseline artifacts + fixed prompts/tasks]

[Full fixed-point end-to-end path]
    └──requires──> [Numerical stability guardrails]
                        └──requires──> [Failure-mode introspection tooling]

[Quantization-aware scheduler policies] ──enhances──> [Throughput/latency observability]

[Silent bf16 fallback] ──conflicts──> [Quality regression gating vs bf16]
```

### Dependency Notes

- **Stable integer execution path requires quantized checkpoint loading:** without reliable quant metadata parsing and mapping, runtime correctness is random.
- **Quality gating requires reproducible workflow:** if launch/eval are not deterministic, <5% quality claims are not trustworthy.
- **Full fixed-point path requires stability guardrails:** integerization widens numerical risk surface (especially activations/KV path), so guardrails are prerequisite, not polish.
- **Quantization-aware scheduling enhances observability:** scheduler/prefill/decode behavior changes metric distributions; both must evolve together.
- **Silent fallback conflicts with quality gating:** hidden fallback invalidates acceptance metrics and masks regressions.

## MVP Definition

### Launch With (v1)

Minimum viable product — what's needed to validate the concept.

- [ ] Stable W8A8 integer linear serving path in OpenAI-compatible API mode — core technical proof.
- [ ] Quantized checkpoint loading for selected formats + fail-fast validation — operational reliability.
- [ ] bf16-relative quality gate (<5% degradation) with repeatable evalscope workflow — acceptance criterion.
- [ ] Core observability (TTFT/TPOT/toks/s/VRAM) with quantized-vs-bf16 comparison reports — decision support.

### Add After Validation (v1.x)

Features to add once core is working.

- [ ] Layer-level stability diagnostics (activation range/saturation tracing) — trigger: first unexplained quality cliff.
- [ ] Quantization-aware scheduler tuning presets by GPU generation — trigger: multiple production hardware targets.
- [ ] Quantized KV-cache support path — trigger: memory ceiling reached on long-context workloads.

### Future Consideration (v2+)

Features to defer until product-market fit is established.

- [ ] Broader quantization matrix (many methods/vendors) — defer until v1 quality/perf is repeatably strong.
- [ ] Multi-tenant governance/billing/productization layer — defer until core inference substrate is proven.

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Stable W8A8 integer linear serving | HIGH | HIGH | P1 |
| Quantized checkpoint loading + strict validation | HIGH | MEDIUM | P1 |
| bf16-relative quality regression gate | HIGH | MEDIUM | P1 |
| OpenAI-compatible API stability under integer mode | HIGH | LOW | P1 |
| Throughput/latency/memory observability | HIGH | MEDIUM | P1 |
| Numerical stability guardrails (selective failover) | HIGH | HIGH | P2 |
| Quantization-aware scheduler policies | MEDIUM | HIGH | P2 |
| Quantized KV cache | MEDIUM | HIGH | P2 |
| Broad "all formats" compatibility | MEDIUM | HIGH | P3 |
| Multi-tenant platform features | LOW | HIGH | P3 |

**Priority key:**
- P1: Must have for launch
- P2: Should have, add when possible
- P3: Nice to have, future consideration

## Competitor Feature Analysis

| Feature | Competitor A (vLLM/SGLang/TGI/TensorRT-LLM ecosystem) | Competitor B (llama.cpp-style local serving) | Our Approach |
|---------|--------------|--------------|--------------|
| Quantization method breadth | Broad method menus are common (AWQ/GPTQ/FP8/INT8 variants), often mixed offline/online support | Strong GGUF-centric local quantized deployment | Start narrower and validated: prioritize W8A8 + full fixed-point roadmap with quality gate discipline. |
| OpenAI-compatible serving | Standard expectation and widely implemented | Also present, usually single-node/local-first | Match parity for API compatibility; compete on quantized correctness and reproducibility. |
| Quantization quality controls | Often documented as recommendation; strict project-level quality SLO varies | Usually user-managed quality/perf tradeoff | Make <5% bf16 regression a first-class product contract with automated acceptance checks. |
| Hardware/platform scope | Broad but complex support matrices | Strong CPU/local portability | Focus initial scope on high-confidence GPU path, then expand with validation budget. |

## Sources

- Mini-SGLang project context and requirements: `.planning/PROJECT.md` (HIGH)
- Mini-SGLang current feature baseline: `docs/features.md` (HIGH)
- SGLang quantization docs (offline/online modes, platform matrix, caveats): https://docs.sglang.ai/advanced_features/quantization.html (HIGH)
- TensorRT-LLM precision + quantization matrix: https://nvidia.github.io/TensorRT-LLM/reference/precision.html (HIGH)
- TGI quantization conceptual docs: https://github.com/huggingface/text-generation-inference/blob/main/docs/source/conceptual/quantization.md (MEDIUM; GitHub-rendered doc)
- vLLM quantization feature area (stable docs): https://docs.vllm.ai/en/stable/features/quantization/ (MEDIUM; page content large/truncated during fetch)
- ONNX Runtime quantization guidance (static/dynamic, debugging, int4): https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html (HIGH)

---
*Feature research for: Full fixed-point LLM inference infrastructure*
*Researched: 2026-03-09*
