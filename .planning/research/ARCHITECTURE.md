# Architecture Research

**Domain:** Full fixed-point quantized LLM inference infrastructure (serving + evaluation)
**Researched:** 2026-03-09
**Confidence:** MEDIUM-HIGH

## Standard Architecture

### System Overview

```text
┌───────────────────────────────────────────────────────────────────────────────┐
│                            Serving Control Plane                             │
├───────────────────────────────────────────────────────────────────────────────┤
│  API Server (OpenAI) → Tokenizer → Scheduler(Rank0) → Scheduler(RankN)      │
│                                      │                     │                 │
│                                      └────── NCCL ─────────┘                 │
├───────────────────────────────────────────────────────────────────────────────┤
│                        Quantized Execution Data Plane                         │
├───────────────────────────────────────────────────────────────────────────────┤
│  Engine → Quantized Model Runner → INT Kernels (W8A8 linear + Q15 ops)      │
│            │                             │                                    │
│            └──────────── KV Cache Manager (radix/naive, later int KV)        │
├───────────────────────────────────────────────────────────────────────────────┤
│                   Model & Quantization Artifact Plane                         │
├───────────────────────────────────────────────────────────────────────────────┤
│  BF16 Checkpoint → Quantizer/Calibrator → Quantized Checkpoint + Manifest    │
│                                    (weight_scale/input_scale/rules/version)   │
├───────────────────────────────────────────────────────────────────────────────┤
│                            Evaluation & Regression                            │
├───────────────────────────────────────────────────────────────────────────────┤
│  Eval Runner (OpenAI-compatible) → Dataset tasks (e.g., GSM8K) → Metrics     │
│  BF16 baseline store ↔ quality gate (<5% drop) ↔ release decision            │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities (with boundaries)

| Component | Responsibility | Communicates With | Boundary Rule |
|---|---|---|---|
| API Server | OpenAI-compatible ingress/streaming, auth, request normalization | Tokenizer, clients | No quantization logic here |
| Tokenizer/Detokenizer workers | Text↔token conversion | API Server, Scheduler rank0 (ZMQ) | Pure data conversion; no model math |
| Scheduler | Continuous batching, prefill/decode orchestration, TP coordination | Tokenizer, Engine, peer schedulers (NCCL/ZMQ) | Never owns quant params |
| Quantized Model Runner | Executes model graph with integer-aware layers and dispatch | Scheduler, kernel layer, KV cache | Owns precision policy (int vs fallback islands) |
| Kernel Layer | W8A8/int8 GEMM, Q15 ops (RMSNorm/SiLU/RoPE), CUDA/Triton kernels | Model Runner only | No request semantics |
| KV Cache Manager | Prefix reuse, memory/page lifecycle, cache eviction | Scheduler, Model Runner | Isolated from quantization calibration logic |
| Quantization Pipeline | Offline calibration, quantization, artifact packaging | Model source, artifact registry | No runtime serving dependencies |
| Artifact Registry | Stores quantized checkpoints + manifest + version | Quant pipeline, runtime loader, eval runner | Immutable versioned artifacts only |
| Eval Runner | Runs standardized tasks against OpenAI API endpoint | Serving API, metrics store | Black-box evaluation (no in-process hacks) |
| Quality Gate Service | Compares INT run vs BF16 baseline; enforces <5% drop | Eval runner, CI/CD | Release gate, not in request path |

## Recommended Project Structure

```text
python/minisgl/
├── server/                   # OpenAI-compatible frontend + launch
├── scheduler/                # Batching, prefill/decode, TP coordination
├── engine/                   # Runtime orchestration on each rank
├── layers/
│   ├── quantization/         # W8A8 methods, scale handling, dispatch glue
│   └── *_integer.py          # Integer operators (RMSNorm/activation/etc.)
├── kernel/
│   └── fixed_point/          # Q15 kernels and integer math primitives
├── kvcache/                  # Cache manager implementations
├── models/
│   ├── integer/              # Integer-first model variants
│   └── weight.py             # Artifact loading and parameter wiring
├── quant_pipeline/           # (add) offline quantize/calibrate/manifest tools
└── eval/
    ├── online_eval.py        # (add) OpenAI API eval runner wrappers
    └── quality_gate.py       # (add) bf16 vs int acceptance checks
```

### Structure Rationale

- **Hard separate runtime vs offline quantization.** Serving stays deterministic and simple; quantization iteration stays offline and reproducible.
- **Kernel layer behind model-runner API.** Makes it possible to swap/upgrade kernels without touching scheduler/API.
- **Evaluation as first-class module.** Required because your project success criteria is quality drop, not only speed.

## Architectural Patterns

### Pattern 1: Integer-First with Explicit Precision Islands

**What:** Default execution in integer path; allow clearly marked float islands where integer kernels are not production-ready yet.
**When to use:** Transitioning from W8A8 linear-only to full fixed-point.
**Trade-offs:** Faster migration and safer correctness; temporary complexity from conversions.

### Pattern 2: Quantization Artifact Contract

**What:** Runtime consumes only versioned artifact bundles: `{weights, scales, quantization_config, calibration_meta}`.
**When to use:** Always; prevents "works on one machine" quantization drift.
**Trade-offs:** More upfront pipeline work, much lower runtime ambiguity.

### Pattern 3: Black-Box Online Evaluation Gate

**What:** Evaluate quantized model through the exact OpenAI-compatible API path used in production.
**When to use:** Every model/cuda/kernel release.
**Trade-offs:** Slower CI than unit tests, but catches real serving regressions.

## Data Flow

### Request Flow (serving path)

```text
Client Request
  ↓
API Server
  ↓
Tokenizer Worker
  ↓
Scheduler Rank0 ──broadcast──> Scheduler RankN
  ↓                               ↓
Engine/Model Runner (per rank) -> Integer kernels + KV cache
  ↓
Scheduler Rank0
  ↓
Detokenizer
  ↓
API Server Stream Response
```

### Quantized Model Onboarding Flow

```text
BF16 model checkpoint
  ↓
Calibration dataset + quantization recipe (W8A8 first)
  ↓
Quantized artifact build (weights + scales + manifest)
  ↓
Offline sanity checks (tensor-level + short generation)
  ↓
Staging serve
  ↓
Online eval vs BF16 baseline
  ↓
Pass gate (<5% drop) -> promote artifact
```

### Key Data Flows

1. **Control flow:** API/Tokenizer/Scheduler messages over ZMQ + NCCL coordination across TP ranks.
2. **Numeric flow:** Token embeddings and hidden states through integer-first operators, with explicit conversion boundaries.
3. **Governance flow:** Evaluation metrics back into release gate before artifact promotion.

## Suggested Build Order (dependency-aware)

1. **Baseline parity harness (must build first)**
   - Stand up repeatable BF16 serving + eval path (OpenAI API + evalscope-style workflow).
   - Why first: you cannot measure "<5% drop" without trusted baseline infrastructure.

2. **W8A8 linear runtime path**
   - Land int8 weight/activation linear path and scale loading contract in runtime.
   - Keep other ops in float initially.
   - Dependency: baseline harness.

3. **Integer operator stack (RMSNorm/activation/RoPE, etc.)**
   - Move non-linear and normalization operators to fixed-point kernels incrementally.
   - Dependency: stable W8A8 linear + conversion boundary utilities.

4. **Full fixed-point graph hardening**
   - Remove ad-hoc float fallbacks, enforce precision-island policy and telemetry.
   - Dependency: operator coverage and kernel stability.

5. **Production quality gate + release automation**
   - Automate bf16-vs-int regression checks on every artifact/runtime change.
   - Dependency: stable runtime + repeatable evaluation.

## Scaling Considerations

| Scale | Architecture Adjustments |
|---|---|
| 0-1 model / low QPS | Single deployment, local artifact store, manual eval trigger |
| Multi-model / medium QPS | Versioned artifact registry, canary serving, scheduled eval runs |
| High QPS / multi-cluster | Disaggregated prefill/decode, centralized quality gate, rollout controller by artifact version |

## Anti-Patterns

### Anti-Pattern 1: Coupling quantization pipeline into live scheduler

**What people do:** Calibrate/requantize inside serving process.
**Why it's wrong:** Non-deterministic latency, hard-to-reproduce quality, operational risk.
**Do this instead:** Offline quantization pipeline + immutable artifact handoff.

### Anti-Pattern 2: No explicit precision boundary ownership

**What people do:** Silent int↔float conversions spread across many layers.
**Why it's wrong:** Hidden quality regressions and debugging dead-ends.
**Do this instead:** Single precision policy module + explicit conversion points + counters.

### Anti-Pattern 3: Throughput-only optimization

**What people do:** Ship fastest kernels without bf16-relative quality gates.
**Why it's wrong:** Violates project’s primary acceptance criteria (<5% drop).
**Do this instead:** Promote artifacts only through quality-gated evaluation.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---|---|---|
| Hugging Face / ModelScope | Pull source or quantized model artifacts | Keep model+quant manifest version-locked |
| EvalScope (or equivalent) | OpenAI API black-box evaluation against served endpoint | Matches real serving behavior |
| Observability stack (Prometheus/Grafana) | Emit latency/throughput + quantization health metrics | Track quality and perf regressions together |

### Internal Boundaries

| Boundary | Communication | Notes |
|---|---|---|
| server ↔ tokenizer/scheduler | ZMQ messages | Keep protocol stable and versioned |
| scheduler ↔ engine | In-process API (per rank) | Scheduler remains precision-agnostic |
| model runner ↔ kernels | Strict operator interface | Kernel swaps should not change scheduler/api behavior |
| quant pipeline ↔ runtime loader | Artifact manifest contract | Critical for reproducibility |
| eval runner ↔ serving | OpenAI-compatible HTTP API | Ensures true end-to-end validation |

## Confidence Notes

- **HIGH:** Mini-SGLang process boundaries and flow (repo docs + code), integer mode controls (`MINISGL_INTEGER_MODE`) and current W8A8/int kernel implementation.
- **MEDIUM:** Industry pattern recommendations (artifact contract, quality gate automation) derived from TensorRT-LLM/vLLM ecosystem docs and common serving practice.
- **LOW/unknown:** Exact EvalScope OpenAI API subpage URL was not retrievable in this run; integration guidance uses EvalScope repo docs and known CLI patterns.

## Sources

- Mini-SGLang architecture docs: https://github.com/sgl-project/mini-sglang/blob/main/docs/structures.md (repo-local mirror reviewed)
- Mini-SGLang features docs: https://github.com/sgl-project/mini-sglang/blob/main/docs/features.md (repo-local mirror reviewed)
- Mini-SGLang project context: `.planning/PROJECT.md`
- Mini-SGLang integer and quantization code (local):
  - `python/minisgl/models/register.py`
  - `python/minisgl/models/integer/qwen3_integer.py`
  - `python/minisgl/layers/quantization/w8a8_int8.py`
  - `python/minisgl/server/args.py`
- TensorRT-LLM quantization docs (updated Feb 27, 2026): https://nvidia.github.io/TensorRT-LLM/features/quantization.html
- TensorRT-LLM numerical precision docs (updated Sep 15, 2025): https://nvidia.github.io/TensorRT-LLM/reference/precision.html
- vLLM docs (stable): https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html and quantization sections
- EvalScope repository + usage examples (latest): https://github.com/modelscope/evalscope

---
*Architecture research for: Mini-SGLang full fixed-point inference infra*
*Researched: 2026-03-09*
