# Requirements: Mini-SGLang Full Integer Inference Infra

**Defined:** 2026-03-09
**Core Value:** Serve LLMs in full fixed-point static quantization mode with less than 5% quality drop versus bf16 full-precision baselines.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Serving API

- [ ] **SERV-01**: Operator can start quantized model service via Mini-SGLang with OpenAI-compatible endpoint support.
- [ ] **SERV-02**: User can request chat completions from `/v1/chat/completions` in integer mode and receive valid responses.
- [ ] **SERV-03**: Operator can run integer mode with explicit runtime flag and observe whether integer mode is active.

### Quantized Runtime

- [ ] **QRT-01**: System can execute linear layers in W8A8 integer mode for target model path.
- [ ] **QRT-02**: System can load static quantized checkpoint artifacts with strict schema validation.
- [ ] **QRT-03**: System fails fast with actionable errors when required quantization metadata is missing or invalid.
- [ ] **QRT-04**: System prevents silent fallback to bf16 unless explicit override is enabled.

### Quantization Stability

- [ ] **QST-01**: Operator can detect layer-level numeric instability signals relevant to quantization failures.
- [ ] **QST-02**: System can enforce documented precision policy (integer path plus explicit precision islands only where required).
- [ ] **QST-03**: System behavior is reproducible across repeated runs with the same model artifact and serving configuration.

### Evaluation & Quality Gate

- [ ] **EVAL-01**: Operator can run reproducible evalscope OpenAI API evaluation against served quantized model.
- [ ] **EVAL-02**: System reports quantized-vs-bf16 quality delta using predefined benchmark workflow.
- [ ] **EVAL-03**: Release gate fails when quality degradation exceeds 5% versus bf16 baseline.

### Performance Observability

- [ ] **OBS-01**: Operator can view baseline serving metrics including TTFT, TPOT, throughput, and memory usage.
- [ ] **OBS-02**: Operator can compare quantized and bf16 serving metrics in a repeatable report.
- [ ] **OBS-03**: Operator can run the documented launch and evaluation commands end-to-end without manual undocumented steps.

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Quantization Coverage

- **QCV-01**: System supports expanded quantization format matrix beyond initial validated subset.
- **QCV-02**: System supports wider model-family compatibility profiles beyond Qwen3-8B baseline.

### Runtime Optimization

- **OPT-01**: System supports quantization-aware scheduler presets across multiple GPU generations.
- **OPT-02**: System supports quantized KV cache with validated long-context tradeoff envelope.

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Training or finetuning new base models | Project scope is inference infrastructure and quantized serving, not training pipeline development. |
| Broad all-format quantization support in v1 | Expands validation matrix too early and risks missing the core <5% quality objective. |
| Multi-tenant billing/quota platform layer | Not required to validate core integer inference quality and serving goals. |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| SERV-01 | Phase 1 | Pending |
| SERV-02 | Phase 1 | Pending |
| SERV-03 | Phase 1 | Pending |
| QRT-01 | Phase 1 | Pending |
| QRT-02 | Phase 2 | Pending |
| QRT-03 | Phase 2 | Pending |
| QRT-04 | Phase 2 | Pending |
| QST-01 | Phase 2 | Pending |
| QST-02 | Phase 2 | Pending |
| QST-03 | Phase 2 | Pending |
| EVAL-01 | Phase 3 | Pending |
| EVAL-02 | Phase 3 | Pending |
| EVAL-03 | Phase 3 | Pending |
| OBS-01 | Phase 4 | Pending |
| OBS-02 | Phase 4 | Pending |
| OBS-03 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 16 total
- Mapped to phases: 16
- Unmapped: 0 ✅

---
*Requirements defined: 2026-03-09*
*Last updated: 2026-03-09 after roadmap mapping*
