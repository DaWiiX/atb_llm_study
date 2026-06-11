# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Pure ATB graph implementations for Qwen3VL-Embedding-2B inference on Huawei Ascend NPU. The Python package (`atb_python_qwen3vl_embedding`) provides a **zero-dependency inference engine** — no transformers library in the hot path. It loads weights directly from safetensors and builds ATB computation graphs for every model component.

The `atb_cpp_llm` directory contains a C++ multi-model LLM engine built on ATB, supporting Qwen3, Qwen3VL, DeepSeek-V2/V3, Mixtral and other architectures.

## Hardware requirements

- Huawei Ascend NPU (tested on 910B and 310P)
- `torch_npu` (NPU runtime)
- `torch_atb` (Ascend Transformer Boost graph compiler)

## Running tests

Tests live in `atb_python_qwen3vl_embedding/tests/`. Each test file is a standalone script that can be run directly:

```bash
cd atb_python_qwen3vl_embedding
python tests/test_text_attention.py
python tests/test_text_mlp.py
python tests/test_text_decoder_layer.py
python tests/test_text_model.py
python tests/test_vision_attention.py
python tests/test_vision_mlp.py
python tests/test_vision_block.py
python tests/test_vision_model.py
python tests/test_vision_patch_embed.py
python tests/test_engine.py           # E2E: text-only, image-only, image+text
python tests/test_e2e_full_pipeline.py  # Full pipeline vs transformers reference
```

Performance benchmark:
```bash
python tests/benchmark.py  # --iter N --warmup M
```

Tests require a model checkpoint at `/mnt/workspace/gitCode/models/Qwen3-VL-Embedding-2B/` (containing `config.json`, `preprocessor_config.json`, and `model.safetensors`).

All unit tests validate ATB output against the transformers reference implementation using cosine similarity (threshold typically 0.99).

## 测试精度原则

**绝不通过降低验收标准来"通过"测试。** 如果 C++ 和 Python 在相同输入下余弦相似度低于 0.99，说明存在 bug，必须定位并修复根因，而不是放宽阈值。在有 Python 参考实现的情况下，C++ 应严格对齐 Python 的计算逻辑，逐阶段排查差异直到余弦相似度 ≥ 0.99。

## Architecture

### Split-graph strategy

To avoid OOM from building a giant 28-layer graph, the model uses **split graphs** — individual ATB graphs for each component type are built once and reused in Python-level loops with per-layer weights:

- **Text path**: one `DecoderLayer` graph (built lazily for sequence length `S`), looped 28x with per-layer weights and cosine/sine embeddings. A separate small `FinalNorm` graph handles output normalization.
- **Vision path**: one `FirstLayer` graph (patch_embed + pos_embed + block 0), one `VisionBlock` graph (looped for blocks 1-23), one `Merger` graph, one `DeepstackMerger` graph.

### Pipeline flow (`engine.py:Qwen3VLEngine`)

```
input_ids ──→ embed_tokens (CPU) ──→ inputs_embeds
                                        │
pixel_values ──→ VisionModel (NPU) ────→ masked_scatter (image tokens injected)
                                        │
                              get_rope_index (CPU) → MRoPE cos/sin
                                        │
                              TextModel (NPU, 28-layer loop) → norm → hidden_states
```

1. Text token embedding via `F.embedding` (CPU lookup table)
2. Vision features computed on NPU and injected into embedding positions matching `image_token_id`
3. MRoPE (Multi-dimensional RoPE) position IDs computed on CPU
4. Text decoder layers run on NPU via the shared DecoderLayer graph

### Deepstack features

Intermediate vision block outputs at specific layer indices (e.g. [5, 11, 17]) are fed through deepstack merger MLPs and added into the corresponding early text decoder layers. This is the Qwen3VL cross-modal fusion mechanism.

### NPU memory strategy (`engine.py:92-119`)

All weights are pre-loaded to NPU as float16 at `__init__` time:
- `t_layer_weights`: list of 28 lists, each with 11 NPU-resident float16 tensors
- `v_block_weights`: list of 24 lists, each with 12 NPU-resident float16 tensors
- `embed_w`: token embedding stays on CPU (accessed via `F.embedding`)
- `norm_w`, `v_pe_w/b`, `v_merger_w`, `v_ds_w`: all NPU-resident

### Key files

| File | Role |
|---|---|
| `engine.py` | Top-level `Qwen3VLEngine` — orchestrates full pipeline |
| `engine_utils.py` | Weight loading from safetensors, MRoPE, RoPE indices, vision position embeddings |
| `utils.py` | ATB parameter factories (Linear, RMSNorm, SelfAttention, RopeOp, etc.) and `set_atb_buffer_size` |
| `preprocess.py` | CPU-side image preprocessing (resize, normalize, patch extraction) |
| `text_decoder_layer.py` | Single decoder layer: input_norm → attention(+residual) → post_norm → MLP(+residual) |
| `text_attention.py` | Text self-attention: Q/K projections → RMSNorm → RoPE → SelfAttention → O-projection |
| `text_mlp.py` | SwiGLU MLP: gate_proj → SiLU, up_proj, elementwise multiply, down_proj |
| `text_model.py` | Text model runner: builds decoder layer graph once, loops with per-layer weights |
| `vision_block.py` | Single vision block: norm1 → attention(+residual) → norm2 → MLP(+residual) |
| `vision_attention.py` | Vision attention: QKV combined linear → split → RoPE → SelfAttention → projection |
| `vision_mlp.py` | Vision MLP: fc1 → GELU → fc2 |
| `vision_patch_embed.py` | Patch embedding (Conv3d → Linear equivalent) |
| `vision_model.py` | Vision model runner: first_layer → loop blocks → merger, with deepstack extraction |
| `data_utils.py` | Test config/data generation (uses transformers for reference data only) |
| `transformers_runner.py` | Reference transformers implementations for comparison testing |

### Critical constraint

**`set_atb_buffer_size()` must be called exactly once before any ATB graph build.** Calling it twice can corrupt graph outputs. The buffer size depends on the workload — unit tests use ~2GB, E2E tests use 5-20GB.

### Weight naming convention

Weights follow HuggingFace key naming: `model.language_model.*` for text, `model.visual.*` for vision. The engine's `load_weights` loads from `model.safetensors` and converts bf16 → float32. Individual weight extractors in `engine_utils.py` pull specific layers by key prefix.

### How to Debug

when you debug python code, remember to check atb log `cat /root/ascend/log/atb/$(ls -rt /root/ascend/log/atb/ | tail -n 1)`

when you debug cpp code, remember to check atb log `cat /root/atb/log/$(ls -rt /root/atb/log/ | tail -n 1)`

when you trying to make atb cpp code, remember to execute blow code first:
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann/set_env.sh

source /usr/local/Ascend/nnal/atb/latest/atb/set_env.sh --cxx_abi=1
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
export ATB_BUILD_DEPENDENCY_PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1
```

### Python 参考数据生成器 import 规范

`atb_cpp_llm/tests/python_reference/` 下的 Python 脚本通过 `gen_all.py` 编排生成 C++ 测试用的 `.bin` 参考数据。所有引用 `atb_python_qwen3vl_embedding` 包的 import **必须使用全限定路径**：

```python
# ✅ 正确（所有生成器统一风格）
from atb_python_qwen3vl_embedding.preprocess import smart_resize
from atb_python_qwen3vl_embedding.engine_utils import get_rope_index

# ❌ 错误（裸 import 在 gen_all.py 编排下会 ModuleNotFoundError）
from preprocess import smart_resize
from engine_utils import get_rope_index
```

这是因为 `gen_all.py` 从 repo root 调用子进程，`sys.path` 里只有 repo root，没有 `atb_python_qwen3vl_embedding/` 子目录。验收：`python tests/python_reference/gen_all.py` 5/5 生成器全部 OK。

### 查阅 ATB 文档

**环境**: Linux aarch64，Playwright + Chromium（已装于 `/tmp/node_modules/playwright`，浏览器在 `~/.cache/ms-playwright/chromium-1223/`）。

**脚本模板**（`cd /tmp && node playwright_test.js`）：
```js
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  await page.goto('目标URL', { timeout: 30000, waitUntil: 'networkidle' });
  await page.waitForTimeout(3000);
  const content = await page.evaluate(() => {
    const main = document.querySelector('.document-main');
    return main ? main.innerText : document.body.innerText;
  });
  console.log(content);
  await browser.close();
})();
```

**搜索算子/类名**（过滤 ALL_META.TXT.json）：
```js
const metaText = await page.evaluate(async () => {
  const resp = await fetch('/doc_center/source/zh/CANNCommunityEdition/900/API/ascendtb/ALL_META.TXT.json');
  const data = await resp.json();
  return data.filter(item => item.kw && item.kw.includes('目标关键词'))
    .map(item => ({ uri: item.uri, title: item.title, des: (item.des || '').substring(0, 200) }));
});
```

**入口 & URL**: 参考主页 `ascendtb_01_0098.html`，C++ 算子 `01_0{0100~0320}`，Python API `01_0{0330+}`。基 URL: `https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/ascendtb/`。

**排查指引**: 参数→"参数列表"，硬件→"产品支持"，报错→"约束说明"，Python/C++ API 参数一一对应（`torch_atb.XxxParam` vs `atb::infer::XxxParam`）。

### 310P 平台适配

310P 上 SelfAttention 不支持 GQA 模式（`kv_head_num < head_num`）。解决方案是在权重加载时将 GQA 展开为 MHA（数学精确变换）。详见 [`atb_cpp_llm/docs/platform-310p.md`](./atb_cpp_llm/docs/platform-310p.md)。

关键要点：
- Python: `is_310p()` 检测平台 → engine 层自动展开 K/V 权重
- C++: `Is310P()` 检测平台 → `Qwen3VLModel::Load()` 中自动展开
- 新增 GQA 测试必须加 `Is310P()` 守卫（910B 可运行 310P 路径，310P 不可运行原生 GQA）
- 平台配置：`.env` 中 `ASCEND_PLATFORM=310P`（默认 910B）
# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
