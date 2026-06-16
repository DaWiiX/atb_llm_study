# Cosine Similarity Threshold Standards

This document defines the standard cosine similarity thresholds used across all
test files in this directory.  Every test must reference the appropriate level
and the rationale must be documented in-test.

## Threshold Table

| Test class                               | Cosine  | Rationale                                                                 |
|------------------------------------------|---------|---------------------------------------------------------------------------|
| Identity / data-check (fp32 or exact)    | 0.9999  | Deterministic comparison — same weights, same indices, same mask pattern. Any deviation indicates a loading bug or data conversion error. |
| Single fp16 operator (0–1 layers)        | 0.999   | One fp16 matmul / attention / MLP call. Rounding error is ~1e-6 per element; cosine drift < 1e-3. |
| Moderate fp16 accumulation (2–5 layers)  | 0.99    | 2–5 stacked fp16 layers. Each layer adds ~1e-3 to ~2e-3 cosine drift due to matmul rounding. Total drift stays within 0.01–0.02. |
| Full 28-layer text model                 | 0.95    | Full Qwen3VL-Embedding-2B text decoder depth. Over 28 layers, fp16 matmul rounding can compound to ~0.03–0.05 cosine reduction. If cosine falls below 0.95, run per-layer diagnostics to determine if it is genuine fp16 drift or a code bug. |

### Reserved levels (not currently used)

| Level                                    | Cosine  | Status                                                                     |
|------------------------------------------|---------|----------------------------------------------------------------------------|
| Deep fp16 accumulation (6+ layers)       | 0.98    | Reserved for future 6–27 layer tests. No test currently uses this level.   |
| Cross-framework (ATB vs Transformers)    | 0.97    | Reserved, not currently used. All cross-framework tests in practice use 0.99 (moderate accumulation) because ATB and TF paths are close enough that 0.99 is achievable. |

## Why fp16 accumulation reduces cosine

Each fp16 matrix multiply rounds its result to 16-bit mantissa (~1e-4 relative
error).  When N layers are stacked, the error in each layer is amplified by the
weights of all subsequent layers:

```
cosine_drop ≈ N × (1e-3 to 2e-3)
```

- **1 layer**:   drift < 0.001  → cosine stays ≥ 0.999
- **2 layers**:  drift < 0.003  → cosine stays ≥ 0.997   (we allow 0.99)
- **28 layers**: drift ≈ 0.03–0.05 → cosine ≥ 0.95 expected

This is *not* a bug — it is the expected behaviour of fp16 arithmetic when
matrix multiplications are chained without intermediate upcasting.

## Standard threshold levels

### 0.9999 — Identity check

**Use for**: Loading the same safetensors file in two frameworks; generating
identical integer position_ids; constructing a causal mask with identical
values and dtypes.

**Typical tests**: `test_text_diagnostics.py` (embed_tokens weight,
position_ids, RoPE cos/sin, causal mask).

If a 0.9999 check fails, there is a **data bug** — a weight loading mismatch,
an indexing off-by-one, or a dtype conversion error.  Do not relax this
threshold; fix the root cause.

### 0.999 — Single fp16 operator

**Use for**: One isolated ATB graph op vs its TF equivalent — single matmul,
single attention head, single MLP block, patch embedding, position embedding,
or image preprocessing.

**Typical tests**: `test_vision_attention.py`, `test_vision_mlp.py`,
`test_vision_block.py`, `test_vision_patch_embed.py`,
`test_vision_pos_embed.py`, `test_preprocess.py`,
`test_deepstack_merger_precision.py`, `test_text_model.py` (1-layer),
`test_vision_diagnostics.py` (pixel_values), `test_vision_model.py`.

### 0.99 — Moderate accumulation

**Use for**: 2–5 stacked layers, full end-to-end pipelines with real model
weights, or deepstack feature extraction/injection across multiple vision
blocks.  Also used as the default for cross-framework comparisons where minor
implementation differences may exist (e.g. ATB's GQA vs TF's MHA expansion on
310P).

**Typical tests**: `test_text_model.py` (2-layer),
`test_embedder_e2e.py`, `test_deepstack_integration.py`,
`test_pipeline_trace.py`, `test_310p_diag.py`, `test_310p_combinations.py`,
`test_text_diagnostics.py` (single text layer comparison),
`test_vision_diagnostics.py` (default report).

### 0.95 — Full 28-layer model

**Use for**: The full Qwen3VL-Embedding-2B text decoder (28 layers, GQA,
head_dim=128).  This is the deepest cumulative fp16 path in the entire project.

**Typical tests**: `test_text_model.py` (28-layer `test_text_model_28_layers`), `test_e2e.py`.

If cosine drops below 0.95, do NOT relax this threshold.  Instead, run per-layer
diagnostics (`test_text_diagnostics.py`) to isolate whether the drop is genuine
fp16 accumulation or a code bug in a specific layer.

## How to choose a threshold for a new test

1. **Count the number of fp16 matmul/attention/MLP calls** in the ATB graph
   path vs the reference path.
2. **Match the closest level** from the table above.  When in doubt, use the
   stricter level and relax only if the test consistently fails with cosines
   just below the threshold.
3. **Document in the test**:
   - Which threshold level applies and why (number of layers, ops).
   - A link to this file (`THRESHOLDS.md`).
   - The line comment `# 0.XXX: <level description> — see THRESHOLDS.md`.

## Threshold deviation approval process

If a test needs a threshold that is NOT one of the standard levels above:

1. **Document the exact cosine value observed** in the test docstring.
2. **Explain why the standard level is insufficient** (e.g. a specific
   operator has known precision limitations on a particular platform).
3. **Add a `# THRESHOLD-DEVIATION:` comment** at the threshold definition
   line, citing the reason.
4. **Update this file** to add the deviation with a `[DEVIATION]` marker.

Deviations should be rare.  If multiple tests need the same deviation, that
deviation becomes a new standard level and is moved to the main table.
