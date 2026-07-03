# XP5 Hybrid MiniSGL Development Plan

## Baseline

- Branch: `xp5-dev`
- Code base: latest official `sgl-project/mini-sglang` `main`
- Push target: `origin` = `https://github.com/54limiao/mini-sgl-integer.git`
- First product target: MiniSGL serving with a hybrid static-quant path for Qwen3.

Hybrid means linear layers use static W8A8 integer kernels and quantized packed weights, while attention, norm, activation, sampling, tokenizer, scheduler, and OpenAI API behavior remain the official MiniSGL runtime unless a specific precision island is explicitly replaced later.

## Product Goal

Turn the current TileLang Qwen3 int-only demo into a deployable model-serving workflow:

1. Convert or load a static quantized Qwen3 artifact.
2. Start MiniSGL with an explicit XP5 hybrid mode.
3. Serve OpenAI-compatible `/v1/chat/completions` requests.
4. Validate quality with API-level GSM8K and later MMLU/CMMLU.
5. Report serving metrics and kernel-level performance for bf16 versus hybrid.

The selling point is not peak A100 INT8 TOPS. The selling point is a reproducible static-quant deployment path that can move from CUDA/TileLang validation toward XP5 fixed-point hardware.

## Milestone 0: Clean Branch And Planning

**Status**: in progress

**Deliverables**

- Local `xp5-dev` branch based on official MiniSGL `upstream/main`.
- `origin` remains the fork for pushing development work.
- This milestone plan is checked into `.planning/`.

**Exit Criteria**

- `git log -1` matches official MiniSGL main at branch creation time.
- `git remote -v` shows both `origin` fork and `upstream` official repo.
- `git push -u origin xp5-dev` creates the remote development branch.

## Milestone 1: Hybrid Runtime Interface

**Goal**: make hybrid mode a first-class runtime mode without changing normal MiniSGL behavior.

**Implementation Scope**

- Add CLI options in `python/minisgl/server/args.py`:
  - `--quant-mode {none,xp5-hybrid}`
  - `--quant-artifact PATH`
  - `--quant-kernel {torch,sgl-kernel,tilelang}`
- Add corresponding fields to `python/minisgl/engine/config.py`.
- Keep `quant_mode=none` as default and behavior-identical to official MiniSGL.
- Log the selected quant mode during server startup.

**Exit Criteria**

- Official bf16 launch still works unchanged.
- `python -m minisgl --model /code/Qwen3-0.6B --quant-mode xp5-hybrid --quant-artifact ...` reaches model construction or fails fast with a clear artifact error.

## Milestone 2: Static Quant Artifact Contract

**Goal**: define the file format MiniSGL expects from the TileLang quantization pipeline.

**Implementation Scope**

- Add a small artifact reader module, for example `python/minisgl/quant/xp5_artifact.py`.
- Validate:
  - model family and hidden sizes match HF config
  - required packed linear weights exist
  - weight scales and optional activation scales exist
  - tensor-parallel sharding policy is declared
  - artifact version is supported
- Start with Qwen3 dense models only.

**Proposed Artifact Layout**

```text
model.xp5/
  xp5_quant_config.json
  tensors/
    model.layers.0.self_attn.qkv_proj.weight.int8.safetensors
    model.layers.0.self_attn.qkv_proj.weight_scale.safetensors
    ...
  reports/
    calibration.json
    offline_accuracy.json
    kernel_profile.json
```

**Exit Criteria**

- Missing or mismatched artifact fields fail before serving starts.
- A valid artifact maps all Qwen3 linear modules to quantized tensor metadata.

## Milestone 3: Hybrid W8A8 Linear In MiniSGL

**Goal**: replace selected Qwen3 linear layers with W8A8 execution while leaving the rest of the runtime intact.

**Implementation Scope**

- Add quantized linear wrappers compatible with current `python/minisgl/layers/linear.py` classes:
  - `LinearQKVMerged`
  - `LinearOProj`
  - `LinearColParallelMerged`
  - `LinearRowParallel`
- Load int8 weights and scales from the static artifact.
- Support an initial backend priority:
  1. TileLang Qwen3 linear kernel if available.
  2. `sgl_kernel.int8_scaled_mm` for quick bring-up.
  3. Torch fallback only for correctness debugging, never silent production fallback.
- Preserve tensor-parallel sharding semantics.

**Exit Criteria**

- Qwen3-0.6B server returns valid chat completions in `xp5-hybrid` mode.
- Runtime logs prove linear layers are using hybrid W8A8 execution.
- bf16 path remains unchanged.

## Milestone 4: Qwen3 Serve-Through Validation

**Goal**: prove the quantized path works through real MiniSGL serving, not only offline scripts.

**Implementation Scope**

- Add launch scripts under `scripts/xp5/`:
  - bf16 server
  - hybrid server
  - smoke chat request
- Add API-level eval runner for:
  - GSM8K exact match
  - a small curated chat smoke set
- Track invalid answer rate, latency, and throughput.

**Exit Criteria**

- GSM8K can run through `/v1/chat/completions` against bf16 and hybrid servers.
- Report contains bf16 score, hybrid score, and delta.
- Hybrid quality delta is acceptable for the artifact under test or the failure is localized to artifact/kernel issues.

## Milestone 5: Performance Report

**Goal**: make performance claims concrete and reproducible.

**Implementation Scope**

- Collect serving metrics:
  - TTFT
  - TPOT
  - request throughput
  - output tokens/s
  - peak memory
- Collect kernel metrics:
  - linear_i8 time and TOPS
  - attention time
  - decode versus prefill breakdown
- Compare:
  - official bf16 MiniSGL
  - XP5 hybrid
  - optional vLLM/SGLang reference if needed for sales material

**Exit Criteria**

- One command generates a bf16-vs-hybrid report for Qwen3-0.6B.
- Qwen3-14B report is added after the 0.6B path is stable.

## Milestone 6: Accuracy And Autotuning Loop

**Goal**: turn hybrid deployment into an optimization loop.

**Implementation Scope**

- Add shape-based linear autotune cache for TileLang kernels.
- Add static quant parameter sweep hooks:
  - activation scale policy
  - clipping ratio
  - per-layer overrides
  - outlier handling
- Add MMLU/CMMLU after GSM8K is stable.

**Exit Criteria**

- Quantization parameters and kernel configs are stored with the artifact.
- A failed quality run can be traced to layer-level or benchmark-level evidence.

## Immediate Next Tasks

1. Push the clean `xp5-dev` branch to the fork.
2. Implement `--quant-mode`, `--quant-artifact`, and `--quant-kernel`.
3. Add artifact schema validation with clear errors.
4. Bring up Qwen3-0.6B hybrid using the fastest available int8 scaled matmul path.
5. Replace the bring-up kernel with TileLang linear kernels once serving correctness is stable.
