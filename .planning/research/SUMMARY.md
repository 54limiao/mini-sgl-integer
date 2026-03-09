# Project Research Summary

**Project:** Mini-SGLang Full Integer Inference Infra
**Domain:** Quantized LLM inference infrastructure (production API serving)
**Researched:** 2026-03-09
**Confidence:** MEDIUM-HIGH

## Executive Summary

This project is a production inference infrastructure effort, not a model research project: the goal is to serve quantized LLMs through OpenAI-compatible APIs while enforcing a hard quality contract (<5% degradation vs bf16). The research converges on a practical expert pattern: keep serving deterministic, move quantization/calibration offline, and treat evaluation as a release gate rather than an afterthought. The recommended baseline is a pinned stack around `torch==2.9.1`, `transformers==4.57.3`, `flashinfer==0.6.5`, plus TorchAO/ModelOpt for quantization and EvalScope for API-mode regression.

The strongest recommendation is phased delivery: establish reproducible bf16 baseline + evaluation first, then land stable W8A8 linear execution with strict artifact/loader validation, then expand toward full integer coverage with explicit precision-island policy and telemetry, and finally harden TP/long-context/runtime envelope for production rollouts. This order is dependency-driven: without baseline/eval infra, quality claims are unreliable; without strict quant artifact contracts, integer runtime correctness is non-deterministic.

The dominant risks are silent drift and hidden fallbacks: calibration mismatch, schema drift, dynamic-vs-static quant ambiguity, and untracked float islands can all produce “looks good in demo, fails in production” outcomes. Mitigation is clear across all research files: fail-fast loader validation, per-op dtype/fallback counters, traffic-shaped calibration, multi-task quality gates, and canary rollout with bf16 shadow comparison.

## Key Findings

### Recommended Stack

The stack research is unusually crisp and high-confidence: pin core runtime versions for ABI stability, especially around kernel-sensitive quantized paths. Avoid floating latest dependencies and postpone ecosystem migrations (e.g., Transformers 5.x) until the W8A8 path is stable.

**Core technologies:**
- **Python 3.12 (3.10–3.12 supported):** runtime/tooling baseline — stable ecosystem without 3.13+ churn.
- **PyTorch 2.9.1:** execution runtime — aligns with `sgl-kernel` requirement and avoids ABI drift.
- **Mini-SGLang (in-repo):** serving/scheduler substrate — keeps architecture coherent and avoids sidecar runtime drift.
- **FlashInfer 0.6.5:** high-performance attention/sampling kernels — first-class fit for SGLang-style serving.
- **Transformers 4.57.3:** model/tokenizer loading — matches current project pin and avoids migration risk.
- **TorchAO 0.16.0 + NVIDIA ModelOpt 0.41.0:** two-track quantization workflow — TorchAO for integration velocity, ModelOpt for hardened offline artifacts.
- **EvalScope 1.4.2 (+ lm-eval 0.4.11 cross-check):** bf16-relative quality regression gating.

Critical version requirements: `torch==2.9.1`, `transformers==4.57.3`, and pinned quant/kernel deps are non-negotiable for reproducibility.

### Expected Features

The feature set is well-prioritized around correctness and reproducibility over breadth. v1 should be intentionally narrow and high-confidence.

**Must have (table stakes):**
- OpenAI-compatible API serving stability in integer mode.
- Stable W8A8 integer linear serving path.
- Quantized checkpoint loading with strict/fail-fast metadata validation.
- bf16-relative quality gate (<5% loss) with reproducible eval workflow.
- Core observability (TTFT, TPOT, toks/s, VRAM; prefill/decode split).
- Multi-GPU TP serving parity for supported scope.

**Should have (competitive):**
- Numerical stability guardrails (layer-wise diagnostics, selective precision escapes).
- Quantization-aware scheduler policies (with chunked prefill/overlap/CUDA graph constraints).
- Quantization acceptance dashboard (quality/perf/cost triplet).
- Golden compatibility profile (Qwen3-8B first) for fast adoption.

**Defer (v2+):**
- “Support every quant format” breadth expansion.
- Multi-tenant quota/billing/platform features.

### Architecture Approach

Architecture research recommends four hard separations: (1) serving control plane (API/tokenizer/scheduler), (2) quantized execution plane (engine/model runner/kernels/KV cache), (3) offline quantization artifact plane, and (4) online black-box evaluation gate. The key pattern is **integer-first with explicit precision islands** governed by policy, not accidental fallback. Runtime should consume immutable versioned artifacts (`weights + scales + quant_config + calibration_meta`), and every promotion should pass black-box API eval against bf16 baselines.

**Major components:**
1. **Serving Control Plane (API + tokenizer + scheduler):** request handling, batching, TP coordination; remains precision-agnostic.
2. **Quantized Model Runner + Kernel Layer:** executes integer-aware graph, manages precision policy, dispatches W8A8/Q15 kernels.
3. **KV Cache Manager:** prefix reuse/memory lifecycle with explicit KV precision policy.
4. **Offline Quant Pipeline + Artifact Registry:** calibrate/quantize/package immutable artifacts.
5. **Eval Runner + Quality Gate Service:** enforce <5% bf16-relative acceptance before release.

### Critical Pitfalls

1. **Calibration dataset mismatch** — build traffic-shaped calibration sets, version artifacts, and monitor layer clipping against replay traffic.
2. **Hidden float islands in “integer mode”** — enforce per-op integerization contract and fail CI on unplanned float fallbacks.
3. **Static-vs-dynamic quantization ambiguity** — require explicit mode, validate required scales at load, fail-fast on incomplete static artifacts.
4. **Quantized checkpoint schema drift** — version schema, validate tensors/shapes/semantics strictly, maintain compatibility tests.
5. **Narrow quality gate overfitting** — use multi-task/multi-metric bf16-relative gates, not single benchmark pass/fail.

## Implications for Roadmap

Based on dependency structure across stack/features/architecture/pitfalls, the roadmap should use **5 phases**.

### Phase 1: Baseline & Evaluation Contract
**Rationale:** Everything downstream depends on trustworthy bf16-relative measurement.
**Delivers:** Reproducible bf16 serving runbook, frozen prompts/templates/tokenizer, API-mode eval harness, initial multi-metric gate.
**Addresses:** OpenAI-compatible serving reliability, reproducible eval workflow, quality gate foundation.
**Avoids:** Pitfall 1 (calibration mismatch blind start), Pitfall 10 (single-benchmark overfitting).

### Phase 2: W8A8 Runtime + Artifact Contract
**Rationale:** Core MVP proof is stable integer linear execution with deterministic loading semantics.
**Delivers:** W8A8 linear path, quantized checkpoint schema/versioning, strict loader validation (static/dynamic explicit), base quant telemetry.
**Uses:** Pinned torch/transformers/flashinfer stack; TorchAO/ModelOpt artifact generation.
**Implements:** Model-runner precision policy + kernel dispatch boundary.
**Avoids:** Pitfall 2, 3, 4, 8.

### Phase 3: Full-Path Integerization Hardening
**Rationale:** Move from “quantized linear” to credible fixed-point system behavior.
**Delivers:** Incremental integer ops (RMSNorm/activation/RoPE/etc.), explicit precision islands, per-op dtype/fallback counters, TP parity tests, KV precision policy v1.
**Addresses:** Full fixed-point differentiator, numerical stability guardrails.
**Avoids:** Pitfall 2, 6, 7, 9.

### Phase 4: Performance/Scale Tuning & Production Envelope
**Rationale:** After correctness, optimize scheduler+kernel behavior in realistic traffic envelopes.
**Delivers:** Quantization-aware scheduler presets, hardware profile tuning, long-context KV sweeps, startup capability reports, fallback-rate SLOs, canary+rollback playbook.
**Addresses:** Observability + throughput/latency goals, production readiness.
**Avoids:** Pitfall 5, 6, 9.

### Phase 5: Release Automation & Expansion Gate
**Rationale:** Lock repeatability before broad format/platform expansion.
**Delivers:** Automated release gates (quality/perf/stability), artifact promotion workflow, Qwen3-8B golden profile pack, clear criteria for v2 expansion.
**Addresses:** Sustainable delivery cadence with quality contract enforcement.
**Avoids:** Anti-features (silent fallback, broad format explosion too early).

### Phase Ordering Rationale

- Evaluation-first is mandatory because project success is quality-constrained, not throughput-first.
- Artifact contract and loader strictness must precede broad integerization to prevent non-deterministic failures.
- Scheduler/hardware tuning should follow numerical correctness to avoid optimizing unstable behavior.
- Expansion (formats/platformization) is last to protect launch confidence and validation bandwidth.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 3:** Integer operator coverage boundaries and precision-island policy by op/model family need targeted validation.
- **Phase 4:** Hardware-specific quant policy profiles (Ampere/Hopper and backend envelope constraints) need dedicated benchmarking research.
- **Phase 5 (partial):** Release gate metric mix/thresholds for real workloads may require additional domain-task curation.

Phases with standard patterns (can likely skip deeper research):
- **Phase 1:** OpenAI-compatible serving + baseline eval runbook are established patterns.
- **Phase 2 (core pieces):** Schema-versioned artifact loading and fail-fast validation are well-documented best practices.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Version pins are concrete, internally consistent, and backed by official package/project constraints. |
| Features | HIGH | Prioritization is coherent and closely aligned to explicit project acceptance criteria. |
| Architecture | MEDIUM-HIGH | Core boundaries are clear; some guidance is pattern-derived from broader ecosystem rather than direct Mini-SGLang precedent. |
| Pitfalls | MEDIUM-HIGH | Risk catalog is strong and actionable; some competitor-doc evidence had partial/truncated retrieval. |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

- **EvalScope endpoint/runbook granularity:** exact API-eval integration details should be validated in early Phase 1 implementation.
- **Operator-by-operator full-integer coverage map:** needs explicit tracking matrix per model architecture before committing “full fixed-point” milestone dates.
- **KV-cache precision strategy choice (bf16/fp8/int8):** requires empirical long-context tradeoff data before defaulting.
- **Cross-GPU generation default policies:** performance-quality policy defaults must be benchmarked per hardware class.

## Sources

### Primary (HIGH confidence)
- `.planning/PROJECT.md` — project goals, constraints, acceptance criteria.
- `/pyproject.toml` (referenced by STACK.md) — Mini-SGLang dependency constraints.
- Mini-SGLang local docs/code cited in research (`docs/features.md`, `docs/structures.md`, integer/quantization modules) — architecture/runtime reality.
- PyTorch versions: https://pytorch.org/get-started/previous-versions/
- SGLang quantization guidance: https://docs.sglang.ai/advanced_features/quantization.html
- FlashInfer docs/package: https://docs.flashinfer.ai, https://pypi.org/project/flashinfer-python/
- EvalScope package: https://pypi.org/project/evalscope/
- TensorRT-LLM precision/quantization docs: https://nvidia.github.io/TensorRT-LLM/reference/precision.html, https://nvidia.github.io/TensorRT-LLM/features/quantization.html
- ONNX Runtime quantization docs: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html

### Secondary (MEDIUM confidence)
- LM Evaluation Harness package/docs: https://pypi.org/project/lm-eval/
- vLLM quantization docs: https://docs.vllm.ai/en/stable/features/quantization/
- TGI quantization conceptual docs: https://github.com/huggingface/text-generation-inference/blob/main/docs/source/conceptual/quantization.md
- SmoothQuant repo: https://github.com/mit-han-lab/smoothquant

### Tertiary (LOW confidence)
- EvalScope OpenAI-subpage specifics were not fully retrievable during research run; validate exact workflow details in implementation.

---
*Research completed: 2026-03-09*
*Ready for roadmap: yes*
