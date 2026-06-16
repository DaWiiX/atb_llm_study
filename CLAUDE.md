# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

## Project overview

注意我们适配的模型是 Qwen3VL-Embedding-2B 而不是 Qwen3VL

Pure ATB graph implementations for Qwen3VL-Embedding-2B inference on Huawei Ascend NPU. The Python package (`atb_python_qwen3vl_embedding`) provides a **zero-dependency inference engine** — no transformers library in the hot path. It loads weights directly from safetensors and builds ATB computation graphs for every model component.

The `atb_cpp_llm` directory contains a C++ multi-model LLM engine built on ATB, supporting Qwen3, Qwen3VL, DeepSeek-V2/V3, Mixtral and other architectures.

## Hardware requirements

- Huawei Ascend NPU (tested on 910B and 310P)
- `torch_npu` (NPU runtime)
- `torch_atb` (Ascend Transformer Boost graph compiler)


Tests require a model checkpoint at `{somewhere}/Qwen3-VL-Embedding-2B/` (containing `config.json`, `preprocessor_config.json`, and `model.safetensors`).

All unit tests validate ATB output against the transformers reference implementation using cosine similarity (threshold typically 0.99).

## 测试精度原则

**绝不通过降低验收标准来"通过"测试。** 如果 C++ 和 Python 在相同输入下余弦相似度低于 0.99，说明存在 bug，必须定位并修复根因，而不是放宽阈值。在有 Python 参考实现的情况下，C++ 应严格对齐 Python 的计算逻辑，逐阶段排查差异直到余弦相似度 ≥ 0.99。

## Architecture

### Split-graph strategy

To avoid OOM from building a giant 28-layer graph, the model uses **split graphs** — individual ATB graphs for each component type are built once and reused in Python-level loops with per-layer weights:

- **Text path**: one `DecoderLayer` graph (built lazily for sequence length `S`), looped 28x with per-layer weights and cosine/sine embeddings. A separate small `FinalNorm` graph handles output normalization.
- **Vision path**: one `FirstLayer` graph (patch_embed + pos_embed + block 0), one `VisionBlock` graph (looped for blocks 1-23), one `Merger` graph, one `DeepstackMerger` graph.


### How to Debug

when you debug python code, remember to check atb log `cat /root/ascend/log/atb/$(ls -rt /root/ascend/log/atb/ | tail -n 1)`

when you debug cpp code, remember to check atb log `cat /root/atb/log/$(ls -rt /root/atb/log/ | tail -n 1)`

when you trying to make atb cpp code, remember to execute blow code first:

if you are root:
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann/set_env.sh

source /usr/local/Ascend/nnal/atb/latest/atb/set_env.sh --cxx_abi=1
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
export ATB_BUILD_DEPENDENCY_PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1
```
if you are normal user:
```bash
source ~/Ascend/ascend-toolkit/set_env.sh
source ~/Ascend/cann/set_env.sh

source ~/Ascend/nnal/atb/latest/atb/set_env.sh --cxx_abi=1
source ~/Ascend/nnal/atb/set_env.sh --cxx_abi=1
export ATB_BUILD_DEPENDENCY_PATH=~/Ascend/nnal/atb/latest/atb/cxx_abi_1
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

310P 上 SelfAttention PA_ENCODER 的 mask 格式要求为 NZ (FRACTAL_NZ)，不同于 910B 的 ND 格式。GQA 模式（`kv_head_num < head_num`）在 310P 上原生支持（实测 cos=1.0），无需展开为 MHA。详见 [`atb_cpp_llm/docs/platform-310p.md`](./atb_cpp_llm/docs/platform-310p.md)。

关键要点：
- Python: `is_310p()` 检测平台 → `make_causal_mask_nz_npu()` 生成 NZ 格式 mask
- C++: `Is310P()` 检测平台 → `MakeCausalMaskNzFp16()` 生成 NZ 格式 mask
- GQA 测试无需 `Is310P()` 守卫（310P 原生支持 GQA）
- 平台配置：`.env` 中 `ASCEND_PLATFORM=310P`（默认 910B）