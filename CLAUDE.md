# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Pure ATB graph implementations for Qwen3VL-Embedding-2B inference on Huawei Ascend NPU. The Python package (`atb_python_qwen3vl_embedding`) provides a **zero-dependency inference engine** — no transformers library in the hot path. It loads weights directly from safetensors and builds ATB computation graphs for every model component.

The `atb_cpp_llm` directory contains a C++ multi-model LLM engine built on ATB, supporting Qwen3, Qwen3VL, DeepSeek-V2/V3, Mixtral and other architectures.

## Hardware requirements

- Huawei Ascend NPU (tested on 910B)
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

### 查阅昇腾/CANN/ATB 文档的标准方法

**环境说明**: Linux Arm64 (aarch64)，Chrome 不可用。Playwright 使用 Firefox。需要先 `cd /tmp && npm install playwright`（已有则跳过），然后将脚本写入 `/tmp/playwright_test.js` 用 `node` 执行。

**1. 文档首页入口**

ATB API 参考主页（变更声明）: `https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/ascendtb/ascendtb_01_0098.html`

**2. 快速查某个算子/类名的所有页面（通过元数据）**

```js
// 在 page.evaluate 中 fetch ALL_META.TXT.json，按关键词过滤
const metaText = await page.evaluate(async () => {
  const resp = await fetch('/doc_center/source/zh/CANNCommunityEdition/900/API/ascendtb/ALL_META.TXT.json');
  const data = await resp.json();
  return data.filter(item => item.kw && item.kw.includes('目标关键词'))
    .map(item => ({ uri: item.uri, code: item.code, title: item.title, des: (item.des || '').substring(0, 200) }));
});
```

每个条目包含 `uri`（如 `ascendtb_01_0262.html`），拼接 `https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/ascendtb/{uri}` 即可直接访问。

**3. 获取文档完整目录树（含所有子页面）**

API: `GET https://www.hiascend.com/ascendgateway/ascendservice/doc/node/tree/zh/CANNCommunityEdition/900/API/ascendtb/ascendtb_01_0098_90x_html`（需带 Referer header）

返回的 `directory` 中包含完整的层级结构，每个节点有 `nodeName`、`nodeHtml`（HTML 文件名）、`nodeId`。

**4. Playwright 脚本模板**

```js
const { firefox } = require('playwright');
(async () => {
  const browser = await firefox.launch({ headless: true });
  const page = await browser.newPage();
  await page.goto('目标URL', { timeout: 30000, waitUntil: 'networkidle' });
  await page.waitForTimeout(3000); // 等 SPA 渲染

  // 提取主内容区的文本
  const content = await page.evaluate(() => {
    const main = document.querySelector('.document-main');
    return main ? main.innerText : 'no content';
  });
  console.log(content);

  await browser.close();
})();
```

**5. API 页面 URL 结构规律**

- 变更声明: `ascendtb_01_0098.html`
- C++ 算子定义/说明子页面: `ascendtb_01_0{编号}.html`（编号范围 0100~0320 左右）
- Python API（OpParam/SelfAttentionParam 等）: `ascendtb_01_0{编号}.html`（编号范围 0330+）
- 头文件索引: `ascendtb_0012.html`
- 所有页面都在 `https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/ascendtb/` 下

**6. 常见问题排查思路**

- 算子参数不确定 → 查该算子的"参数列表"子页面（定义/默认值/是否必选）
- 某个硬件是否支持 → 查该算子的"产品支持情况"子页面
- 报错/约束 → 查"约束说明"子页面
- Python API vs C++ API → 两者参数一一对应，类型名略有不同（`torch_atb.XxxParam` vs `atb::infer::XxxParam`）
- 不确定算子有哪些功能 → 查"功能列表"及"各功能共存情况"子页面