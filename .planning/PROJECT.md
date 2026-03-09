# Mini-SGLang Full Integer Inference Infra

## What This Is

A full fixed-point LLM inference infrastructure built on Mini-SGLang, focused on serving quantized models reliably in production-style API mode. The system targets W8A8 integer execution for linear layers first, then drives toward full static integer quantization across the LLM inference path. It is for model infra engineers who need bf16-level quality with integer efficiency.

## Core Value

Serve LLMs in full fixed-point static quantization mode with less than 5% quality drop versus bf16 full-precision baselines.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Build stable integer-mode serving path in Mini-SGLang with W8A8 linear execution
- [ ] Resolve quantization failure modes so integer inference remains numerically stable
- [ ] Achieve end-to-end static quantized LLM inference with quality loss under 5% versus bf16
- [ ] Provide repeatable evaluation and launch workflow for Qwen3-8B in OpenAI-compatible serving mode
- [ ] Integrate paper-backed outlier mitigation for prefix-quantized checkpoints, with focus on down-projection outlier behavior

### Out of Scope

- Training new base models from scratch — this project is inference-focused
- Building a generic benchmark platform beyond required quantization validation — benchmark tooling only supports acceptance criteria

## Context

- Project anchor: Mini-SGLang inference framework in this repository
- Quantization direction: W8A8 integer mode for linear layers, then full fixed-point static quantization
- Quality target: less than 5% degradation compared with bf16 floating-point baseline
- Evaluation path currently demonstrated with evalscope OpenAI API evaluation and gsm8k
- Serving path currently demonstrated with `MINISGL_INTEGER_MODE=1` and Mini-SGLang API server startup against a quantized Qwen3-8B artifact
- Existing example runtime snippets indicate focus on practical online serving and measurable eval parity
- Reference technical background includes arXiv:2411.07191, arXiv:2410.05265, and arXiv:2603.05498
- Prepared prefix-quantized model artifact: `/workspace/lim42@xiaopeng.com/github/quant_bench/data/models/Qwen/Qwen3-8B-r1r2-r4-gptq-w4a8-static-ignore-prefix-640`
- Current debugging hypothesis: token outliers are concentrated around down projection path and should be treated as first-class stabilization target
- Baseline non-prefix artifact metadata: `/workspace/lim42@xiaopeng.com/github/quant_bench/data/models/Qwen/Qwen3-8B-r1r2-gptq-w8a8-static/down_proj_scales.csv`, with obvious outliers around layers 6 and 16
- Current observed quality: prefix-quantized artifact without runtime prefix-KV capability gives gsm8k around `0.4761`, which is considered too low
- Immediate development focus: validate the prepared prefix-quantized model in Mini-SGLang and verify whether enabling prefix KV cache materially improves quality

## Constraints

- **Quality**: Loss versus bf16 must stay below 5% — this is the primary acceptance gate
- **Quantization**: Linear layers must run in W8A8 integer mode — core technical requirement
- **Architecture**: Built inside Mini-SGLang serving/runtime flow — avoid sidecar inference stack drift
- **Evaluation**: Must support reproducible OpenAI-compatible evaluation flow (evalscope API mode) — enables objective regression checks
- **Model scope**: Initial validation anchored on Qwen3-8B quantized artifacts — keeps early milestones focused

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Use Mini-SGLang as the execution and serving substrate | Existing codebase already provides high-performance inference primitives and API surface | — Pending |
| Prioritize W8A8 integer linear path before broader full-path integerization | Linear layers are the dominant compute path and highest leverage for quantization gains | — Pending |
| Gate progress with bf16-relative quality target (<5% drop) instead of raw throughput-only metrics | Prevents shipping fast but unusable quantized models | — Pending |
| Validate with OpenAI-compatible evalscope flow and production-like service launch commands | Ensures reproducibility and operational realism | — Pending |
| Anchor quantization stabilization design to selected paper techniques and verified outlier locations | Reduces trial-and-error and directly targets observed failure modes | — Pending |

---
*Last updated: 2026-03-09 after initialization*
