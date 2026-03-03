# Triton Attention Backend 技术解析

## 概览

mini-sglang 的 Triton 注意力后端是一个**纯 Python / Triton** 实现的 Paged Attention 引擎，不依赖 FlashAttention 或 FlashInfer 等预编译二进制，完全通过 Triton JIT 在运行时编译 CUDA kernel。整个后端由三层组成：

```
python/minisgl/attention/triton.py          ← 后端接口层（Backend & Metadata）
python/minisgl/kernel/triton/
    attention_kernels.py                    ← 适配层（wrapper）
    decode_attention.py                     ← 解码 Triton kernel（两阶段 Flash Decoding）
    extend_attention.py                     ← 预填充/扩展 Triton kernel（统一单阶段）
    prefill_attention.py                    ← 纯预填充辅助 kernel（无 prefix 缓存）
```

---

## 核心数据结构

### KV Cache 内存布局

KV Cache 由 `MHAKVCache` 管理，原始存储形状为：

```
[2, num_layers, num_pages, page_size, num_kv_heads, head_dim]
 ↑K/V  ↑层        ↑物理页     ↑页大小     ↑KV头数       ↑头维度
```

Triton kernel 要求**平坦的 3-D 布局**才能正确计算 stride：

```
[num_pages × page_size, num_kv_heads, head_dim]
    stride(0) = num_kv_heads × head_dim   ← token 步长
    stride(1) = head_dim                  ← head 步长
    stride(2) = 1
```

因此 `forward()` 中执行一次无拷贝的 view：

```python
k_cache = k_cache.view(-1, k_cache.shape[-2], k_cache.shape[-1])
```

若不做此 reshape，`cur_kv_head × stride(1)` 会计算为 `cur_kv_head × (num_kv_heads × head_dim)`，造成所有 head 读错内存地址（产生乱码输出）。

### TritonMetadata

每个 batch 携带的注意力元数据：

| 字段 | 形状 | 含义 |
|------|------|------|
| `kv_indptr` | `[batch+1]` | 每条序列在 `kv_indices` 中的起止偏移（累积和） |
| `kv_indices` | `[Σ device_len_i]` | 每个 KV token 对应的**物理页地址**（int64） |
| `qo_indptr` | `[batch+1]` or `None` | 每条序列新增 token（extend）的起止偏移，解码时为 None |
| `max_seqlen_q/k` | int | 当前 batch 中最长的 Q/K 序列长度 |
| `causal` | bool | 是否对 extend 区域施加因果掩码 |

---

## 解码阶段：两阶段 Flash Decoding

### 算法原理

解码时每条序列只有 **1 个新 token** 作为 Query，但 KV 序列可能很长。若直接在单个 CUDA block 内串行归约，会造成严重的内存带宽瓶颈。**Flash Decoding** 的做法是将每条序列的 KV token 分成 $S$ 个 split，并行计算局部 softmax，再合并：

**Stage 1**（`_fwd_kernel_stage1` / `_fwd_grouped_kernel_stage1`）

每个 CUDA block 处理 `(batch_i, head_j, split_k)` 三元组，计算该 split 内的局部注意力输出和 log-sum-exp：

$$
\text{att\_out}[i,j,k] = \frac{\sum_{t \in \text{split}_k} \exp(q_j \cdot K_t / \sqrt{d} - m_k) \cdot V_t}{Z_k}, \quad
\text{lse}[i,j,k] = m_k + \log Z_k
$$

**Stage 2**（`_fwd_kernel_stage2`）

跨 split 合并，使用 online softmax 免去显式归一化分母：

$$
o_j = \frac{\sum_k \exp(\text{lse}_k - m^*) \cdot \text{att\_out}[i,j,k]}{\sum_k \exp(\text{lse}_k - m^*)}
$$

### GQA/MQA 支持

当 `kv_group_num = num_qo_heads / num_kv_heads > 1` 时（GQA/MQA），启用 grouped kernel（`BLOCK_H=16`），每个 block 同时处理 16 个共享同一 KV head 的 Q heads，通过 `tl.dot` 矩阵乘法批量计算 QK，比逐 head 循环快约 `min(BLOCK_H, kv_group_num)` 倍。

### 中间缓冲区 & KV Splits

```python
attn_logits  # [batch, num_heads, max_kv_splits, v_head_dim]  fp32
attn_lse     # [batch, num_heads, max_kv_splits]               fp32
num_kv_splits  # [batch]  每条序列实际使用的 split 数（默认全部 = 8）
```

`max_kv_splits=8` 是静态默认值，即每条序列的 KV 最多被分为 8 段并行处理。对于超长上下文可适当调大。

---

## 预填充阶段：统一单阶段 Extend Kernel

### Extend 与纯 Prefill 的区别

| 场景 | Q tokens | KV tokens |
|------|----------|-----------|
| 纯 Prefill（无缓存） | 全部新 token | 全部新 token（= Q） |
| Extend（RadixCache 命中） | 仅新增 token | 缓存 prefix + 新增 token |

Triton 后端使用**统一 extend kernel**（`extend_attention_fwd_unified` / `_fwd_kernel_unified`）同时处理两种场景，通过 `prefix_lens[i]` 区分每条序列的 prefix 边界。

### 统一 KV 索引构造

`kv_indices` 将 prefix 和 extend 的物理页地址**拼接**为单一数组：

```
kv_indices = [prefix_pages(seq0), extend_pages(seq0),
              prefix_pages(seq1), extend_pages(seq1), ...]
```

对应的 `kv_indptr[i]` 指向 seq_i 在该数组中的起始位置，`kv_indptr[i+1] - kv_indptr[i] = device_len_i`（prefix + extend 总长）。

在 `prefill_attention_fwd` 包装函数中，`prefix_lens` 由差分计算得到：

```python
prefix_lens = (kv_indptr[1:] - kv_indptr[:-1]) - (qo_indptr[1:] - qo_indptr[:-1])
#              ↑ total KV len（device_len）        ↑ extend len
```

### 因果掩码策略

kernel 内使用**分区因果掩码**，不是简单的全局下三角：

```
KV 索引范围:  [0 .. prefix_len)   [prefix_len .. prefix_len+extend_len)
Q 索引范围:   (始终在 extend 区)
因果规则:     prefix 区 → 全部可见（无掩码）
              extend 区 → 仅 Q_idx >= K_idx_in_extend 时可见
```

这保证了前缀复用时注意力的正确性：新 token 可以看到所有缓存的 prefix，但在 extend 序列内维持因果顺序。

---

## CUDA Graph 支持

CUDA Graph 通过**固定地址缓冲区**实现 kernel 重放（replay）：

```
TritonCaptureData
├── cu_seqlens_k  [max_bs+1]   ← 捕获时预填 i * max_seq_len
└── page_table    [max_bs * max_seq_len]  ← 展平为 1-D，用作 kv_indices
```

**捕获（Capture）阶段**：以最大序列长度 `max_seq_len` 构造 `kv_indptr` 和 `kv_indices`，kernel 绑定到这些固定缓冲区的地址。

**重放（Replay）阶段**：仅需将真实运行时数据 in-place 写入同一缓冲区，kernel 自动使用更新后的数据：

```python
# prepare_for_replay 的核心操作：
self.capture.cu_seqlens_k[: bs + 1].copy_(metadata.kv_indptr)
self.capture.page_table[:total_tokens].copy_(metadata.kv_indices)
```

CUDA Graph 消除了解码阶段的 CPU kernel launch overhead，对 `bs=1` 这类小 batch 尤其显著（一般可降低约 30-50% 的端到端延迟）。

---

## 数据流总结

```
AttentionLayer.forward(qkv)
  │
  ├─ split q, k, v
  ├─ q_norm / k_norm（可选）
  ├─ RoPE rotary embedding
  │
  └─ TritonBackend.forward(q, k, v, layer_id, batch)
       │
       ├─ kvcache.store_kv(k, v, out_loc, layer_id)   ← 写入物理页
       ├─ k_cache / v_cache = kvcache.k_cache(layer_id).view(-1, H, D)
       │
       ├─ [is_decode] decode_attention_fwd(...)
       │     ├─ Stage 1: _fwd_kernel_stage1  (batch × heads × splits 并行)
       │     └─ Stage 2: _fwd_kernel_stage2  (batch × heads 归并)
       │
       └─ [is_prefill] prefill_attention_fwd(...)
             └─ extend_attention_fwd_unified(...)
                   └─ _fwd_kernel_unified  (batch × heads × q_blocks 并行)
```

---

## 与其他后端的对比

| 特性 | Triton | FlashAttention (FA) | FlashInfer (FI) |
|------|--------|---------------------|-----------------|
| 依赖 | 仅 `triton` | `sgl-kernel` 预编译库 | `flashinfer` 预编译库 |
| Paged KV | ✅ page_size=1 | ✅ paged | ✅ paged |
| GQA 支持 | ✅ grouped kernel | ✅ | ✅ |
| RadixCache Extend | ✅ unified kernel | ✅ | ✅ |
| CUDA Graph | ✅ 固定缓冲区 replay | ✅ | ✅ CUDAGraph wrapper |
| 可调试性 | ⭐⭐⭐ 源码可读 | ⭐⭐ | ⭐⭐ |
| 峰值性能 | 略低于 FI | 接近 FI | 最高 |
