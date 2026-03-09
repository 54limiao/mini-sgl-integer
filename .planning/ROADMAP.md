# Roadmap: Mini-SGLang Full Integer Inference Infra

## Overview

This roadmap delivers a production-ready integer inference path in Mini-SGLang by moving from usable W8A8 serving, to strict quantization safety/stability, to enforceable quality gates, and finally to operator-grade observability and repeatable launch/eval operations. It is sequenced to protect the core acceptance contract: <5% quality loss versus bf16.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Integer Serving MVP** - Deliver a working OpenAI-compatible integer-mode serving path with W8A8 linear execution.
- [ ] **Phase 2: Quantization Safety & Stability** - Enforce strict artifact contracts, fail-fast behavior, and reproducible stable runtime policy.
- [ ] **Phase 3: Evaluation Quality Gate** - Make bf16-relative quality checks reproducible and release-blocking.
- [ ] **Phase 4: Operational Observability & Runbook** - Provide repeatable command flow and quantized-vs-bf16 performance reporting.

## Phase Details

### Phase 1: Integer Serving MVP
**Goal**: Users can call a Mini-SGLang OpenAI-compatible endpoint in integer mode and receive valid responses from a W8A8-enabled model path.
**Depends on**: Nothing (first phase)
**Requirements**: SERV-01, SERV-02, SERV-03, QRT-01
**Success Criteria** (what must be TRUE):
  1. Operator can start a quantized model service in Mini-SGLang and the `/v1/chat/completions` endpoint is reachable.
  2. User can submit chat completion requests in integer mode and receive valid completion responses.
  3. Operator can explicitly enable integer mode at launch and can observe from runtime output/state that integer mode is active.
  4. In normal inference flow for the target model path, linear layers execute in W8A8 integer mode.
**Plans**: TBD

### Phase 2: Quantization Safety & Stability
**Goal**: Integer inference is trustworthy under strict artifact validation, controlled precision behavior, and reproducible execution.
**Depends on**: Phase 1
**Requirements**: QRT-02, QRT-03, QRT-04, QST-01, QST-02, QST-03
**Success Criteria** (what must be TRUE):
  1. Operator can load static quantized checkpoints only when required schema and metadata are valid; invalid artifacts are rejected.
  2. When quantization metadata is missing or malformed, the system stops early and reports actionable error messages.
  3. Integer-mode serving does not silently fall back to bf16 unless an explicit override is turned on by the operator.
  4. Operator can inspect layer-level instability signals and confirm documented precision policy (integer path plus explicit precision islands).
  5. Re-running the same model artifact and serving config produces reproducible runtime behavior.
**Plans**: TBD

### Phase 3: Evaluation Quality Gate
**Goal**: Release readiness is objectively judged by reproducible bf16-relative quality evaluation with an enforced <5% degradation threshold.
**Depends on**: Phase 2
**Requirements**: EVAL-01, EVAL-02, EVAL-03
**Success Criteria** (what must be TRUE):
  1. Operator can run a documented evalscope OpenAI API evaluation flow against the served quantized model and reproduce results.
  2. Evaluation output includes a quantized-versus-bf16 quality delta report from the predefined benchmark workflow.
  3. Release gate automatically fails when quality degradation exceeds 5% versus the bf16 baseline.
**Plans**: TBD

### Phase 4: Operational Observability & Runbook
**Goal**: Operators can run the full launch/eval workflow end-to-end with repeatable performance visibility for quantized and bf16 modes.
**Depends on**: Phase 3
**Requirements**: OBS-01, OBS-02, OBS-03
**Success Criteria** (what must be TRUE):
  1. Operator can view baseline serving metrics (TTFT, TPOT, throughput, memory usage) for production-style runs.
  2. Operator can generate a repeatable comparison report between quantized and bf16 serving metrics.
  3. A new operator can execute the documented launch and evaluation commands end-to-end without undocumented manual intervention.
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Integer Serving MVP | 0/TBD | Not started | - |
| 2. Quantization Safety & Stability | 0/TBD | Not started | - |
| 3. Evaluation Quality Gate | 0/TBD | Not started | - |
| 4. Operational Observability & Runbook | 0/TBD | Not started | - |
