# Mini-SGLang 项目指南

## 项目概述

Mini-SGLang 是一个**轻量级但高性能**的大型语言模型（LLM）推理框架，是 [SGLang](https://github.com/sgl-project/sglang) 的精简实现。整个代码库仅约 **5,000 行 Python 代码**，既是一个功能完善的推理引擎，也是研究人员和开发者的透明参考实现。

### 核心技术栈

- **Python 3.10+**: 主要开发语言
- **PyTorch**: 深度学习框架
- **CUDA**: GPU 加速计算
- **FlashAttention / FlashInfer**: 高性能注意力计算内核
- **ZeroMQ (ZMQ)**: 进程间通信
- **NCCL**: GPU 间张量并行通信
- **FastAPI**: API 服务器框架
- **Apache TVM FFI**: CUDA 内核 JIT 编译绑定

### 主要特性

- **Radix Cache**: 跨请求共享前缀的 KV 缓存复用
- **Chunked Prefill**: 长上下文分块预填充，降低峰值内存
- **Overlap Scheduling**: CPU 调度与 GPU 计算重叠
- **Tensor Parallelism**: 多 GPU 张量并行扩展
- **CUDA Graph**: 最小化解码阶段的 CPU 启动开销
- **OpenAI 兼容 API**: 提供 `/v1/chat/completions` 端点

---

## 项目结构

```
/workspace/lim42@xiaopeng.com/github/mini-sglang/
├── python/minisgl/          # 主源代码目录
│   ├── core.py              # 核心数据类 (Req, Batch, Context, SamplingParams)
│   ├── server/              # 服务器相关
│   │   ├── args.py          # 命令行参数解析
│   │   ├── launch.py        # 服务器启动逻辑
│   │   └── api_server.py    # FastAPI API 服务器
│   ├── scheduler/           # 调度器实现
│   │   ├── scheduler.py     # 主调度器类
│   │   ├── config.py        # 调度器配置
│   │   ├── prefill.py       # 预填充逻辑
│   │   ├── decode.py        # 解码逻辑
│   │   └── cache.py         # 缓存管理
│   ├── engine/              # 推理引擎
│   │   ├── engine.py        # 引擎主类
│   │   ├── graph.py         # CUDA 图管理
│   │   └── sample.py        # 采样逻辑
│   ├── models/              # 模型实现
│   │   ├── llama.py         # Llama-3 模型
│   │   ├── qwen3.py         # Qwen-3 模型
│   │   ├── qwen3_moe.py     # Qwen-3 MoE 模型
│   │   ├── base.py          # 模型基类
│   │   └── weight.py        # 权重加载
│   ├── layers/              # 神经网络层
│   │   ├── attention.py     # 注意力层
│   │   ├── linear.py        # 线性层
│   │   ├── norm.py          # 归一化层
│   │   ├── rotary.py        # RoPE 位置编码
│   │   └── embedding.py     # 嵌入层
│   ├── attention/           # 注意力后端
│   │   ├── base.py          # 注意力基类
│   │   ├── fa.py            # FlashAttention 后端
│   │   └── fi.py            # FlashInfer 后端
│   ├── kvcache/             # KV 缓存管理
│   │   ├── base.py          # 缓存基类
│   │   ├── radix_manager.py # Radix 缓存管理器
│   │   └── naive_manager.py # 简单缓存管理器
│   ├── distributed/         # 分布式/张量并行
│   │   ├── impl.py          # 并行实现
│   │   └── info.py          # 分布式信息
│   ├── tokenizer/           # 分词器
│   │   ├── tokenize.py      # 分词
│   │   └── detokenize.py    # 反分词
│   ├── message/             # 消息传递
│   │   ├── frontend.py      # 前端消息
│   │   └── backend.py       # 后端消息
│   ├── kernel/              # 自定义 CUDA 内核
│   │   ├── csrc/            # C++/CUDA 源码
│   │   └── triton/          # Triton 内核
│   ├── utils/               # 工具函数
│   ├── llm/                 # Python LLM 接口
│   ├── moe/                 # MoE 实现
│   ├── benchmark/           # 基准测试工具
│   └── shell.py             # 交互式 shell
├── tests/                   # 测试目录
├── benchmark/               # 基准测试脚本
│   ├── offline/             # 离线推理测试
│   └── online/              # 在线服务测试
├── docs/                    # 文档
│   ├── features.md          # 特性文档
│   └── structures.md        # 架构文档
├── pyproject.toml           # Python 项目配置
├── Dockerfile               # Docker 构建文件
└── README.md                # 项目说明
```

---

## 构建和运行

### 环境要求

- **操作系统**: Linux (x86_64 或 aarch64)
- **Python**: 3.10 或更高版本
- **CUDA Toolkit**: 与驱动版本匹配
- **GPU**: NVIDIA GPU（支持 CUDA）

### 安装

使用 `uv` 进行快速安装：

```bash
# 克隆仓库
git clone https://github.com/sgl-project/mini-sglang.git
cd mini-sglang

# 创建虚拟环境并安装
uv venv --python=3.12
source .venv/bin/activate
uv pip install -e .
```

### 运行服务器

**在线服务模式**（OpenAI 兼容 API）：

```bash
# 单 GPU 部署
python -m minisgl --model "Qwen/Qwen3-0.6B"

# 多 GPU 张量并行
python -m minisgl --model "meta-llama/Llama-3.1-70B-Instruct" --tp 4 --port 30000

# 使用 ModelScope 下载模型
python -m minisgl --model "Qwen/Qwen3-0.6B" --model-source modelscope
```

**交互式 Shell 模式**：

```bash
python -m minisgl --model "Qwen/Qwen3-0.6B" --shell
```

### Docker 运行

```bash
# 构建镜像
docker build -t minisgl .

# 运行服务
docker run --gpus all -p 1919:1919 minisgl --model Qwen/Qwen3-0.6B --host 0.0.0.0

# 交互式 shell
docker run -it --gpus all minisgl --model Qwen/Qwen3-0.6B --shell
```

---

## 开发规范

### 代码风格

项目使用以下工具保持代码质量：

- **Black**: 代码格式化（行宽 100）
- **Ruff**: 快速 Python 代码检查
- **mypy**: 静态类型检查
- **pre-commit**: 提交前自动检查

### 提交前检查

```bash
# 安装 pre-commit 钩子
pre-commit install

# 手动运行检查
pre-commit run --all-files
```

### 测试

```bash
# 运行测试
pytest

# 带覆盖率报告
pytest --cov=minisgl --cov-report=html
```

### 类型注解

项目要求完整的类型注解：

- 所有函数必须添加类型注解
- 使用 `from __future__ import annotations` 启用 postponed evaluation
- 复杂类型使用 `typing` 模块

---

## 关键命令参考

| 命令 | 说明 |
|------|------|
| `python -m minisgl --help` | 查看所有命令行参数 |
| `python -m minisgl --model "MODEL"` | 启动 API 服务器 |
| `python -m minisgl --model "MODEL" --shell` | 启动交互式 shell |
| `python -m minisgl --model "MODEL" --tp 4` | 4 GPU 张量并行 |
| `python -m minisgl --model "MODEL" --cache naive` | 使用简单缓存 |
| `python -m minisgl --model "MODEL" --attn fa,fi` | 指定注意力后端 |
| `pytest` | 运行测试套件 |
| `pre-commit run --all-files` | 运行代码检查 |

### 常用参数

- `--model-path, --model`: 模型路径或 Hugging Face ID
- `--tensor-parallel-size, --tp`: 张量并行大小
- `--dtype`: 数据类型 (auto/float16/bfloat16/float32)
- `--max-running-requests`: 最大并发请求数
- `--cuda-graph-max-bs`: CUDA 图最大 batch size
- `--max-prefill-length`: 分块预填充最大长度
- `--attention-backend, --attn`: 注意力后端 (fa/fi)
- `--cache-type`: 缓存类型 (radix/naive)

---

## 系统架构

### 组件通信

```
用户请求 → API Server → Tokenizer → Scheduler (Rank 0)
                                              ↓
Scheduler (Rank 0) ← NCCL Broadcast → Scheduler (Other Ranks)
                                              ↓
Scheduler (Rank 0) → Detokenizer → API Server → 用户
```

### 数据流

1. **API Server**: 接收用户请求（OpenAI 兼容格式）
2. **Tokenizer**: 文本 → Token IDs
3. **Scheduler**: 管理请求调度、批处理、KV 缓存
4. **Engine**: 执行实际推理计算
5. **Detokenizer**: Token IDs → 文本

### 支持的模型

- **Llama-3** 系列
- **Qwen-3** 系列（包括 Dense 和 MoE 变体）

---

## 环境变量

| 变量 | 说明 |
|------|------|
| `MINISGL_DISABLE_OVERLAP_SCHEDULING` | 禁用重叠调度（用于消融研究） |
| `CUDA_HOME` | CUDA 安装路径 |
| `HF_HOME` | Hugging Face 缓存目录 |
| `TVM_FFI_CACHE_DIR` | TVM FFI 缓存目录 |
| `FLASHINFER_WORKSPACE_BASE` | FlashInfer 工作空间 |

---

## 参考资料

- [SGLang 官方仓库](https://github.com/sgl-project/sglang)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
- [项目文档 - Features](./docs/features.md)
- [项目文档 - Structures](./docs/structures.md)
