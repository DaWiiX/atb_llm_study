# E2E Image Precision Bug: Packed-Sequence Mask Misfire

## 症状

`test_e2e.py` 对比 ATB 引擎与 transformers 参考实现的 `last_hidden_state` 余弦相似度：

| 测试用例      | 余弦相似度 |
|-------------|----------|
| Text-Only   | 0.9996   |
| Image-Only  | 0.48     |
| Image+Text  | 0.80     |

但使用 `Qwen3VLEmbedder.process()`（pooling + L2 归一化后）对比时，三种场景全部 > 0.999。
说明 per-token 层面存在差异，被 pooling 掩盖。

## 根因

### 因果链

```
Qwen3VLModel.forward(attention_mask=None)
    ↓
Qwen3VLTextModel.forward(attention_mask=None)
    ↓
create_causal_mask()
    ↓
_preprocess_mask_arguments():
    position_ids ≠ None  AND  attention_mask is None  AND  past_key_values is None
    ↓
触发 find_packed_sequence_indices(position_ids[0])   ← 只看 temporal 维度
    ↓
图像 token 的 temporal position_ids 全是同一个值
    ↓
diff = 0 ≠ 1  →  每个 token 被分配到不同的"序列"
    ↓
packed_sequence_mask_function: 只有同一序列索引的 token 才能 attend
    ↓
图像 token 之间互相隔离，无法 attention  →  mask 完全错误
```

### 为什么纯文本不受影响

MRoPE 的 temporal position_ids 在纯文本中连续递增 `[0, 1, 2, 3, ...]`，
`diff` 全为 1，不触发 packed-sequence 检测。

但在图文混合输入中，同一张图的 token 共享同一个 temporal position：

```
[0, 1, 2, ..., N-1, N, N, N, ..., N, N+1, ...]
                    ↑  ↑  ↑        ↑
                  24个图像token全在位置N
```

`diff` 出现 0（≠ 1），被误判为多个独立 packed sequence。

### 为什么 Qwen3VLEmbedder 不受影响

`Qwen3VLEmbedder.process()` 通过 `self.processor()` 获取 `attention_mask`（全 1 张量）
并传给模型，绕过了 `_preprocess_mask_arguments` 中的 packed-sequence 检测分支。

### 为什么 use_cache=True 也会导致精度差异

`Qwen3VLTextModel.forward()` 中 `use_cache=True` 且 `past_key_values is None` 时，
会创建 `DynamicCache` 对象。此时 `past_key_values is not None`，
同样跳过 packed-sequence 检测，但走了一条不同的 attention 计算路径
（cache-aware vs cache-free），输出数值不同。

## 修复

### 1. `tests/test_e2e.py` — 传递 `attention_mask`

```python
# prepare_inputs(): 保存 attention_mask
'attention_mask': tf_in.get('attention_mask',
                            torch.ones_like(tf_in['input_ids']))

# run_tf_phase(): 传给 TF reference
attn_mask = inputs['attention_mask'].npu()
kwargs = {'input_ids': input_ids, 'attention_mask': attn_mask}
ref(use_cache=False, **kwargs)
```

两个要点：
- **`attention_mask`**：避免 packed-sequence 误判
- **`use_cache=False`**：避免 cache-aware/cache-free 数值差异

### 2. `preprocess.py` — bicubic 插值

```python
# 修复前
mode='bilinear'
# 修复后
mode='bicubic'
```

transformers 的 `Qwen2VLImageProcessor` 默认使用 `BICUBIC` 采样，
ATB 用 `bilinear` 会导致 pixel_values 不同，污染所有视觉 token 输入。

### 3. `engine.py` — deepstack 注入方式

```python
# 修复前（原地 += 可能导致 ATB 图的 output tensor 写冲突）
hidden[0, visual_mask, :] += deepstack_features[li]

# 修复后（clone + add + writeback，匹配 TF _deepstack_process）
local = hidden[0, visual_mask, :].clone() + deepstack_features[li]
hidden[0, visual_mask, :] = local
```

## 经验教训

1. **传递 `attention_mask` 给 transformers 模型**：当输入包含视觉 token 时，
   MRoPE 的 temporal position_ids 不连续递增，`attention_mask=None` 会触发
   packed-sequence 误判。始终显式传递 `attention_mask`（哪怕全 1）。

2. **`use_cache` 影响计算路径**：即使只是 prefill 阶段，
   `use_cache=True` 也会走 cache-aware attention 路径，产生数值差异。
   做 reference 对比时应用 `use_cache=False`。

3. **插值算法必须一致**：bilinear vs bicubic 是完全不同的采样核，
   会导致 pixel_values 产生系统性偏差。

4. **pooling 掩盖问题**：per-token 精度问题可能被 last-token pooling + L2 归一化
   洗掉，需要在 `last_hidden_state` 层面做全序列对比。
