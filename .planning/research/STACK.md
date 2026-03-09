# Stack Research

**Domain:** Full fixed-point quantized LLM inference infra (Mini-SGLang)
**Researched:** 2026-03-09
**Confidence:** HIGH

## Recommended Stack (2025 standard, pinned for 2026 reproducibility)

### Core Technologies

| Technology | Version | Purpose | Why Recommended | Confidence |
|------------|---------|---------|-----------------|------------|
| Python | 3.10–3.12 (recommend 3.12) | Runtime + tooling baseline | Mini-SGLang and key deps (torch, flashinfer, evalscope) all support Python 3.10+; 3.12 gives modern runtime without bleeding-edge 3.13/3.14 ecosystem churn. | HIGH |
| PyTorch | **2.9.1** | Execution runtime for kernels, quantized ops, CUDA graph, tensor parallel | This is the safest anchor because `sgl-kernel` explicitly requires torch 2.9.1, and Mini-SGLang already constrains torch `<2.10.0`. Pinning avoids ABI drift. | HIGH |
| Mini-SGLang (this repo) | current mainline in-repo | Serving runtime and scheduler substrate | Project constraint is “build inside Mini-SGLang, no sidecar runtime drift.” Keep the core inference/runtime in one codebase. | HIGH |
| FlashInfer | **0.6.5** | High-performance attention/GEMM/sampling kernels | Active serving kernel stack with modern GPU support and fast release cadence; already first-class in SGLang ecosystem and used by Mini-SGLang deps. | HIGH |
| Hugging Face Transformers | **4.57.3** (pin) | Model/config/tokenizer loading | Mini-SGLang currently pins `>=4.56.0, <=4.57.3`; stay on this line until explicit migration plan to Transformers 5.x. | HIGH |

### Quantization & Validation Stack

| Library | Version | Purpose | When to Use | Confidence |
|---------|---------|---------|-------------|------------|
| TorchAO | **0.16.0** | Integer/low-bit quantization workflows inside PyTorch | Use for online/prototyping quantization and integration experiments in the Mini-SGLang path; especially useful while stabilizing W8A8 linear execution behavior. | HIGH |
| NVIDIA ModelOpt | **0.41.0** | Offline production quantization/export flows (NVIDIA-first) | Use when you need pre-quantized production artifacts and reproducible deployment (especially on NVIDIA hardware). | HIGH |
| Apache TVM FFI | **0.1.9** | Stable ABI/FFI layer for kernel packaging/JIT bindings | Use where custom kernel exposure/packaging is required (already in Mini-SGLang dependencies). | HIGH |
| EvalScope | **1.4.2** | API-mode benchmark/eval regression harness | Primary automated acceptance gate for “<5% quality drop vs bf16” using OpenAI-compatible serving endpoints. | HIGH |
| LM Evaluation Harness (`lm-eval`) | **0.4.11** | Secondary independent quality validation | Use as cross-check to ensure EvalScope-only pipeline does not mask scoring/prompting artifacts. | MEDIUM |

### Development Tools

| Tool | Version | Purpose | Notes |
|------|---------|---------|-------|
| pytest | 8.x (project currently `>=6.0`) | Unit/integration regression tests | Add quantization-specific golden tests (layer parity and end-to-end decode invariants). |
| Ruff | 0.11.x+ | Linting/consistency | Keep strict linting; quantization codepaths are brittle and benefit from disciplined reviews. |
| mypy | 1.x (project currently `>=0.950`) | Type safety on config/runtime interfaces | Strongly type quantization configs and fallback policies to avoid silent behavior drift. |

## Installation (recommended pinned baseline)

```bash
# Core runtime
uv pip install "torch==2.9.1" "transformers==4.57.3" "flashinfer-python==0.6.5"

# Quantization + validation
uv pip install "torchao==0.16.0" "nvidia-modelopt==0.41.0" "evalscope==1.4.2" "lm-eval==0.4.11"

# Kernel/ABI dependencies used by Mini-SGLang
uv pip install "sgl-kernel==0.3.21" "apache-tvm-ffi==0.1.9"
```

## Alternatives Considered

| Recommended | Alternative | Why Not Primary |
|-------------|-------------|-----------------|
| Mini-SGLang runtime as serving core | vLLM or TGI as serving core | Violates project constraint (avoid sidecar stack drift); use them only as external baseline comparators. |
| Offline quantization for release artifacts | Online quantization only | SGLang docs explicitly recommend offline quantization for better performance/usability/convenience. |
| TorchAO + ModelOpt combination | Single-tool quantization strategy | In practice you need both: TorchAO for rapid integration loops, ModelOpt for hardened offline deployment artifacts. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Unpinned Torch/Transformers (floating latest) | Breaks kernel/model compatibility; high risk with quantized runtimes and ABI-sensitive paths | Pin `torch==2.9.1`, `transformers==4.57.3` and upgrade deliberately in milestones |
| Silent bf16 fallback as default behavior | Masks quantization failures and invalidates “<5% vs bf16” acceptance logic | Fail-fast startup + explicit opt-in fallback flag with audit logs |
| Online quantization as the only production path | Higher startup/instability risk; harder reproducibility | Pre-quantized offline artifacts + fixed evaluation gate |
| Immediate migration to Transformers 5.x in this project phase | Mini-SGLang currently pinned to 4.57.x; forced upgrade adds migration risk unrelated to W8A8 goals | Complete W8A8 stability first, then run dedicated compatibility migration |

## Stack Patterns by Variant

**If you are in R&D bring-up (W8A8 linear kernel stabilization):**
- Use TorchAO + Mini-SGLang + EvalScope
- Because iteration speed and instrumentation matter more than broad deployment matrix

**If you are in release gating (static quantized serving):**
- Use offline quantized artifacts (ModelOpt or validated pre-quantized checkpoints) + fixed eval runbook
- Because reproducibility and quality contract enforcement matter more than on-the-fly convenience

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| `torch==2.9.1` | `sgl-kernel==0.3.21` | `sgl-kernel` docs explicitly require torch 2.9.1 |
| `transformers==4.57.3` | Mini-SGLang current dependency constraints | Project pin is `>=4.56.0, <=4.57.3` |
| `flashinfer-python==0.6.5` | Python >=3.10 and modern CUDA stack | Keep CUDA version aligned with FlashInfer supported matrix |
| `torchao==0.16.0` | PyTorch 2.9.x line | Confirm against torchao compatibility table before major upgrades |

## Sources

- Mini-SGLang project goals and constraints: `.planning/PROJECT.md` (HIGH)
- Mini-SGLang dependency pins: `/pyproject.toml` (HIGH)
- PyTorch official versions page (shows 2.9.1): https://pytorch.org/get-started/previous-versions/ (HIGH)
- TorchAO official repo/docs and release references: https://github.com/pytorch/ao, https://pypi.org/project/torchao/ (HIGH)
- FlashInfer official package/docs: https://pypi.org/project/flashinfer-python/, https://docs.flashinfer.ai (HIGH)
- SGLang quantization guidance (offline vs online, torchao/modelopt methods): https://docs.sglang.ai/advanced_features/quantization.html (HIGH)
- EvalScope package (API eval workflow): https://pypi.org/project/evalscope/ (HIGH)
- LM Eval Harness package: https://pypi.org/project/lm-eval/ (MEDIUM)
- NVIDIA ModelOpt package: https://pypi.org/project/nvidia-modelopt/ (HIGH)
- Apache TVM FFI package/docs: https://pypi.org/project/apache-tvm-ffi/ (HIGH)
- SGL kernel package metadata (notably archived marker + torch requirement): https://pypi.org/project/sgl-kernel/ (HIGH)

---
*Stack research for: Full fixed-point LLM inference infrastructure on Mini-SGLang*
*Researched: 2026-03-09*
