# Mini-SGLang Attention 模块深度分析

## 概述

Attention 模块是 Mini-SGLang 中最复杂、最核心的组件之一，与普通的 data-independent 算子（如线性层、激活函数等）有本质区别。本文档将深入分析 attention 模块的设计方式、功能特点以及其特殊之处。

---

## 1. Attention 与普通算子的核心区别

### 1.1 Data-Independent vs Data-Dependent

| 特性 | Data-Independent 算子 | Attention 算子 |
|------|---------------------|---------------|
| **输入依赖性** | 仅依赖当前输入张量 | 依赖历史序列（KV Cache） |
| **内存模式** | 输入-输出直接映射 | 输出依赖累积的 KV Cache |
| **状态管理** | 无状态（纯函数） | 有状态（需要维护 KV Cache） |
| **计算复杂度** | O(N) | O(N²) 或 O(N) with PagedAttention |
| **并行性** | 易于并行 | 受限于序列依赖和内存带宽 |

### 1.2 核心差异点

1. **历史依赖性**：
   - 普通算子：`y = f(x)`，输出仅由当前输入决定
   - Attention：`output = Attention(Q, K_cache, V_cache)`，K_cache 和 V_cache 是历史所有 token 的累积

2. **内存管理**：
   - 普通算子：计算完成后即可释放中间张量
   - Attention：需要持久化存储 K、V 值，支持增量更新

3. **序列长度动态性**：
   - 普通算子：batch size 和 sequence length 预先确定
   - Attention：处理变长序列，需要动态内存分配和高效的 KV Cache 管理

---

## 2. Attention 模块的核心架构

### 2.1 三层设计架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Scheduler (调度层)                       │
│  - 请求调度和批处理                                           │
│  - Prefill/Decode 模式管理                                   │
│  - Radix Cache 管理（前缀共享）                               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Engine (引擎层)                           │
│  - Attention Backend 创建和管理                               │
│  - KV Cache 初始化和页面管理                                   │
│  - CUDA Graph 管理                                           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              Attention Backend (后端实现层)                   │
│  - 基类: BaseAttnBackend                                      │
│  - 具体实现: FlashInferBackend, FlashAttentionBackend        │
│  - 元数据管理: BaseAttnMetadata                               │
│  - CUDA Graph 支持                                           │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心组件交互

```python
# Scheduler 中的调用流程
scheduler._prepare_batch(batch)
    ↓
scheduler.attn_backend.prepare_metadata(batch)  # 准备 attention 元数据
    ↓
scheduler.engine.forward_batch(batch, sample_args)
    ↓
engine.attn_backend.forward(q, k, v, layer_id, batch)  # 执行 attention
    ↓
attn_backend.kvcache.store_kv(k, v, batch.out_loc, layer_id)  # 存储 KV
```

---

## 3. Attention Backend 的接口设计

### 3.1 BaseAttnBackend 抽象类

```python
class BaseAttnBackend(ABC):
    @abstractmethod
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        layer_id: int, batch: Batch
    ) -> torch.Tensor:
        """执行 attention 计算"""
        pass

    @abstractmethod
    def prepare_metadata(self, batch: Batch) -> None:
        """准备 attention 元数据（序列长度、位置等）"""
        pass

    @abstractmethod
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        """初始化 CUDA Graph capture"""
        pass

    @abstractmethod
    def prepare_for_capture(self, batch: Batch) -> None:
        """为 graph capture 准备"""
        pass

    @abstractmethod
    def prepare_for_replay(self, batch: Batch) -> None:
        """为 graph replay 准备"""
        pass
```

### 3.2 设计理念

1. **分离元数据准备和计算**：
   - `prepare_metadata()` 在调度阶段调用，在 CPU 上处理
   - `forward()` 在 GPU 上执行实际计算
   - 允许调度和计算重叠（overlap scheduling）

2. **CUDA Graph 支持**：
   - Decode 阶段使用 CUDA Graph 减少启动开销
   - 需要预先 capture 固定 batch size 的计算图
   - `prepare_for_capture()` 和 `prepare_for_replay()` 处理 graph 特定逻辑

3. **Hybrid Backend**：
   - Prefill 和 Decode 使用不同的后端
   - Prefill: 处理长序列，使用不同的优化策略
   - Decode: 处理单个 token，使用 CUDA Graph

---

## 4. Attention 的特殊挑战

### 4.1 KV Cache 管理

#### 4.1.1 为什么需要 KV Cache？

在自回归生成中，每个新 token 都需要与所有历史 token 进行 attention：

```
Token 1: attention(1, {})
Token 2: attention(2, {1})
Token 3: attention(3, {1, 2})
...
Token N: attention(N, {1, 2, ..., N-1})
```

如果每次都重新计算所有历史 K、V，计算复杂度为 O(N²)。通过缓存 K、V，复杂度降为 O(N)。

#### 4.1.2 Radix Cache (前缀共享)

**问题**：多个请求可能有相同的输入前缀，如果每个请求都独立存储 KV，会有大量冗余。

**解决方案**：使用 Radix Tree 结构共享前缀的 KV Cache。

```
请求 A: "The quick brown fox"
请求 B: "The quick brown dog"
请求 C: "The quick red fox"

共享前缀: "The quick"  (只存储一次)
```

**实现**：
- `RadixCacheManager` 管理 Radix Tree
- `match_prefix()` 查找最长匹配前缀
- `insert_prefix()` 插入新前缀
- `evict()` 当内存不足时驱逐旧缓存

### 4.2 Paged KV Cache (分页缓存)

**问题**：传统 KV Cache 需要连续内存，难以支持变长序列和动态扩展。

**解决方案**：借鉴操作系统的分页机制，将 KV Cache 划分为固定大小的页面。

```python
# 页表结构
page_table[req_id][seq_pos] = page_id

# KV Cache 结构
k_cache[page_id][page_size, num_kv_heads, head_dim]
v_cache[page_id][page_size, num_kv_heads, head_dim]
```

**优势**：
- 支持非连续内存分配
- 易于扩展（动态添加页面）
- 支持缓存共享和回收

### 4.3 GQA/MQA 支持

**问题**：为了减少 KV Cache 大小，现代模型使用 Grouped Query Attention (GQA) 或 Multi-Query Attention (MQA)。

```
MHA: 32 Q heads, 32 K heads, 32 V heads
GQA: 32 Q heads, 8 K heads, 8 V heads (每组 4 个 Q 共享 1 个 K/V)
MQA: 32 Q heads, 1 K head, 1 V head
```

**实现挑战**：
- K/V 维度小于 Q 维度，需要广播
- Scale 计算需要考虑 GQA 因子
- Triton/FlashInfer 内核需要支持 GQA

---

## 5. CUDA Graph 优化

### 5.1 为什么使用 CUDA Graph？

**问题**：Decode 阶段每次只生成一个 token，CPU 启动开销相对较大。

**解决方案**：使用 CUDA Graph 将多次 kernel 启动合并为一次。

```python
# 传统方式
for layer in layers:
    qkv = layer.qkv_proj(x)
    q, k, v = split_qkv(qkv)
    o = attention(q, k, v)  # 每次都有 CPU 开销
    x = layer.o_proj(o)

# CUDA Graph 方式
with torch.cuda.graph(graph):
    for layer in layers:
        qkv = layer.qkv_proj(x)
        q, k, v = split_qkv(qkv)
        o = attention(q, k, v)  # 所有 kernel 在图中预启动
        x = layer.o_proj(o)

graph.replay()  # 一次调用执行整个图
```

### 5.2 Graph Capture 的挑战

**挑战 1**：动态内存分配
- CUDA Graph 要求内存布局固定
- 解决方案：预先分配固定大小的 buffer

**挑战 2**：变长序列
- 不同的 batch size 需要不同的 graph
- 解决方案：为每个可能的 batch size 预先 capture graph

**挑战 3**：KV Cache 索引
- KV Cache 的页表索引是动态的
- 解决方案：使用固定索引，在 replay 时更新数据

### 5.3 Mini-SGLang 的实现

```python
# Graph capture 流程
for bs in [1, 2, 4]:  # 为不同 batch size capture
    batch = Batch(reqs=[dummy_req] * bs, phase="decode")
    attn_backend.prepare_for_capture(batch)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        model.forward()
    graph_map[bs] = graph

# Graph replay
if batch.size <= max_graph_bs:
    graph = graph_map[batch.padded_size]
    attn_backend.prepare_for_replay(batch)
    graph.replay()
```

---

## 6. FlashInfer vs FlashAttention

### 6.1 FlashAttention

- **特点**：IO 感知的 attention 实现，减少 HBM 访问
- **优势**：理论最优的 FLOPs 和带宽利用
- **限制**：主要优化单一请求的 attention

### 6.2 FlashInfer

- **特点**：专为 LLM 推理优化的 attention 库
- **优势**：
  - 原生支持 Paged KV Cache
  - 支持 Batched Prefill 和 Decode
  - 支持 CUDA Graph
  - 支持 GQA/MQA
- **Mini-SGLang 的选择**：使用 FlashInfer 作为主要 backend

---

## 7. Attention 模块的工作流程

### 7.1 Prefill 阶段

```python
# 1. 调度器准备 batch
batch = prefill_manager.schedule_next_batch()

# 2. 准备 metadata（CPU 端）
attn_backend.prepare_metadata(batch)
# 构建 cu_seqlens_q, cu_seqlens_k, indices 等

# 3. 执行 forward（GPU 端）
o = attn_backend.forward(q, k, v, layer_id, batch)
# 使用 BatchPrefillWithPagedKVCacheWrapper
# K/V 存储到 paged cache

# 4. 更新 cache handle
cache_manager.insert_prefix(input_ids, indices)
```

### 7.2 Decode 阶段

```python
# 1. 调度器准备 batch
batch = decode_manager.schedule_next_batch()

# 2. 准备 metadata（CPU 端）
attn_backend.prepare_metadata(batch)
# 构建 indptr, indices, last_page_len 等

# 3. 执行 forward（GPU 端）
o = attn_backend.forward(q, k, v, layer_id, batch)
# 使用 BatchDecodeWithPagedKVCacheWrapper
# K/V 从 paged cache 读取

# 4. 更新 page table
page_table[req_id][new_pos] = new_page_id
```

### 7.3 CUDA Graph Replay 阶段

```python
# 1. 检查是否可以使用 graph
if can_use_cuda_graph(batch):
    # 2. 准备 replay
    attn_backend.prepare_for_replay(batch)

    # 3. 复制数据到 graph buffer
    graph_runner.buffer.copy_from(batch)

    # 4. Replay graph
    graph = graph_runner.graph_map[batch.padded_size]
    graph.replay()

    # 5. 获取输出
    output = graph_runner.buffer.logits[:batch.size]
```

---

## 8. Int8 Attention 量化

### 8.1 量化的动机

**问题**：Attention 计算是内存带宽受限的，降低数据类型可以提升吞吐量。

**解决方案**：将 Q、K、V 量化为 int8。

### 8.2 量化策略

```python
# Dynamic Per-Head Quantization
def _quantize_to_int8(x):
    # 计算每个 head 的最大绝对值
    x_abs_max = torch.abs(x).amax(dim=-1, keepdim=True)
    # 计算缩放因子
    scale = x_abs_max.clamp(min=1e-5) / 127.0
    # 量化
    x_quantized = torch.clamp(torch.round(x / scale), -128, 127).to(torch.int8)
    return x_quantized, scale
```

### 8.3 注意事项

1. **Scale 管理**：需要保存每个 head 的 scale，用于反量化
2. **GQA/MQA 广播**：K/V 的 scale 需要广播到 QO heads
3. **数值稳定性**：需要确保量化误差不会影响最终输出

---

## 9. 总结：为什么 Attention 如此特殊

### 9.1 复杂性来源

1. **状态依赖**：需要维护历史 KV Cache
2. **内存管理**：需要高效的分页和共享机制
3. **性能优化**：需要 CUDA Graph、PagedAttention 等高级优化
4. **功能需求**：需要支持 GQA/MQA、变长序列、批量处理

### 9.2 与普通算子的对比

| 方面 | 普通算子 (如 Linear) | Attention |
|------|---------------------|-----------|
| 状态 | 无状态 | 有状态 (KV Cache) |
| 内存 | 临时 | 持久 |
| 优化重点 | 计算效率 | 内存带宽 + 计算 |
| 实现复杂度 | 简单 | 复杂 |
| 依赖关系 | 层内独立 | 跨层共享 KV Cache |

### 9.3 设计原则

1. **分离关注点**：将调度、计算、缓存管理分离
2. **抽象接口**：提供统一的 backend 接口，支持多种实现
3. **性能优先**：通过 CUDA Graph、PagedAttention 等技术优化性能
4. **灵活性**：支持多种 attention backend (FlashInfer, FlashAttention, Triton)

---

## 10. 相关文件

- `attention/base.py`: Attention backend 基类
- `attention/fi.py`: FlashInfer backend 实现
- `attention/fa.py`: FlashAttention backend 实现
- `attention/triton_int8.py`: Triton Int8 backend 实现
- `kvcache/base.py`: KV Cache 基类
- `kvcache/radix_manager.py`: Radix Cache 管理
- `scheduler/scheduler.py`: 调度器
- `engine/engine.py`: 引擎主类
- `engine/graph.py`: CUDA Graph 管理