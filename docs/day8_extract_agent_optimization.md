# Extract Agent 优化：MLX Python API 集成

## 一、优化前后对比

### 优化前（subprocess 方案）

```
每篇论文:
  subprocess.run(["python", "-m", "mlx_lm.generate", "--model", ..., "--prompt", ...])
    → 启动Python子进程
    → 从磁盘加载2.9GB safetensors模型文件  ← 重复N次！
    → Tokenize + 推理生成
    → 进程退出，模型从内存释放
```

问题：100篇论文需要100次模型加载，每次~5秒，100次=500秒纯浪费在I/O上。

### 优化后（MLX Python API）

```
首篇论文:
  mlx_lm.load() → 加载2.9GB模型到M4统一内存  ← 仅一次
  mlx_lm.generate() → 推理生成

后续论文:
  mlx_lm.generate() → 直接推理（模型已在内存）  ← 零I/O开销
```

### 性能对比

| 指标 | subprocess | Python API | 改进 |
|------|-----------|-----------|------|
| 首篇延迟 | ~7s | 6.5s | 持平 |
| 后续每篇延迟 | ~7s | **1.85s** | **3.8x** |
| 100篇总耗时 | ~700s (12min) | **~190s (3min)** | **3.7x** |
| 模型加载次数 | 100次 | 1次 | 100x减少 |
| API费用 | $0 | $0 | 不变 |

---

## 二、技术设计细节

### 核心类：MLXExtractor

```
┌─────────────────────────────────┐
│         MLXExtractor            │
│                                 │
│  _model: nn.Module   ← 模型权重 │
│  _tokenizer: Tokenizer          │
│  _lock: threading.Lock          │
│  _loaded: bool                  │
│                                 │
│  + extract(abstract) → PICOS    │
│  + is_available → bool          │
│  + unload()                     │
└─────────────────────────────────┘
```

### 设计决策

**1. 懒加载（Lazy Loading）**

不在 `__init__` 时加载模型，而是在首次调用 `extract()` 时才加载。理由：模块导入时不应该阻塞，只有真正需要推理时才占用显存。

**2. 双重检查锁定（Double-Checked Locking）**

```python
def _ensure_loaded(self) -> bool:
    if self._loaded:          # 第一次检查（无锁，快速路径）
        return True
    with self._lock:           # 获取锁
        if self._loaded:       # 第二次检查（持锁，防止重复加载）
            return True
        self._model, self._tokenizer = mlx_lm.load(...)
        self._loaded = True
```

理由：多个线程可能同时触发首次加载。双重检查锁保证模型只被加载一次，且后续调用无锁开销。

**3. 模块级单例**

```python
_mlx_extractor: Optional[MLXExtractor] = None

def _get_mlx_extractor() -> MLXExtractor:
    global _mlx_extractor
    if _mlx_extractor is None:
        _mlx_extractor = MLXExtractor()
    return _mlx_extractor
```

理由：整个 extract_agent 生命周期内只有一个模型实例，避免重复加载。

**4. 与 subprocess 方案的 fallback 兼容**

MLXExtractor 的 `extract()` 返回 `Optional[Dict]`——加载失败或推理失败时返回 `None`，调用方无需改动，fallback 链自动生效：
- MLX 失败 → 自动回退 DeepSeek API
- DeepSeek 失败 → 自动回退 heuristic 规则

---

## 三、与 LoRA 训练的衔接

Extract Agent 是 LoRA 训练的**下游消费者**：

```
LoRA训练产出                       Extract Agent消费
─────────────                      ─────────────────
adapters_v3/                       
  → mlx_lm fuse                    
  → models/Qwen2.5-1.5B-Med-       MLXExtractor 加载此路径
    PICOS_v3/model.safetensors      mlx_lm.load("models/..._v3")
                                    mlx_lm.generate(model, tokenizer, prompt)
```

环境变量 `MLX_PICOS_MODEL_PATH` 控制加载哪个版本的模型，可以随时切换到 V2/V3 或未来的版本。

---

## 四、面试 Q&A

### Q1: Extract Agent 为什么要用本地小模型而不是直接用 DeepSeek？

> 三个原因。首先是**成本**：一次 Meta 分析可能筛选上百篇论文，如果用 DeepSeek API 每篇 $0.0002，一次 $0.02 不算贵，但如果频繁重跑或上千篇，成本就累积了。本地模型一次加载后无限调用，零费用。其次是**延迟**：本地推理 <2s/篇，API 调用加上网络往返 >1s/篇，大量论文时差异明显。最重要的是**架构意义**——Small-to-Large 的核心思想是把高频、格式固定的任务交给小模型，把低频、需要复杂推理的任务交给大模型。PICOS 提取是结构化 JSON 输出，不需要 GPT-4 级别的推理能力，LoRA 微调后的 1.5B 模型完全胜任。

### Q2: 你用 subprocess 调用 MLX 有什么问题？怎么优化的？

> subprocess 方案的问题是每次调用都启动一个新 Python 子进程，然后从磁盘加载 2.9GB 的 safetensors 模型文件。100 篇论文就要加载 100 次，每次加载 5 秒左右，总共浪费 500 秒在 I/O 上。优化方案是用 MLX Python API 直接加载模型：`mlx_lm.load()` 一次，返回 model 和 tokenizer 对象，然后对每篇论文调用 `mlx_lm.generate(model, tokenizer, prompt)`。模型常驻在 M4 的统一内存中，后续推理不再有 I/O 开销。100 篇从 12 分钟降到 3 分钟。

### Q3: 你的 MLXExtractor 是怎么保证线程安全的？

> 用双重检查锁定（Double-Checked Locking）。场景是多个线程可能同时触发首次 `extract()` 调用。第一个检查 `self._loaded` 是无锁的快速路径——一旦模型加载完成，所有后续调用直接通过，没有锁竞争。如果尚未加载，进入锁保护的代码块后再检查一次，防止多个线程同时执行 `mlx_lm.load()` 导致重复加载或内存问题。这是经典的 lazy initialization 线程安全模式。

### Q4: 为什么不直接在 __init__ 里加载模型？

> 懒加载（Lazy Loading）的好处是：import 模块时不会阻塞，不占用显存。extract_agent 可能被导入但不一定被调用（比如用户在 UI 里只做了检索和筛选，没到提取那一步）。如果 `__init__` 就加载，2.9GB 显存就被白白占用了。懒加载把资源消耗推迟到真正需要时。

### Q5: 如果本地 MLX 模型提取失败怎么办？

> 三级 fallback 链：1) 本地 MLX LoRA 模型提取（快、免费、首选）；2) DeepSeek API 提取（慢一些、有费用、但能力强）；3) heuristic 规则提取（最快、无依赖、但质量最低——只是把摘要前 120 字符当 Population、后 120 字符当 Outcome）。三级 fallback 保证了系统的鲁棒性，即使 MLX 模型文件损毁或 DeepSeek API 挂了，系统不会崩溃，只是输出质量逐级下降。

### Q6: 为什么不把 MLX 模型用 FastAPI 包成一个微服务？

> 当前阶段不需要。微服务引入额外的网络开销、序列化开销和运维复杂度。直接用 Python API 在同一进程内调用，延迟最低、无网络故障风险。如果未来需要多客户端共享同一个模型实例（比如 Web 服务有多个 worker），那再考虑微服务化——但 MLX 的模型对象不能跨进程共享，所以会是一组同质负载均衡实例，而不是简单的 `mlx_lm serve`。

### Q7: 你怎么衡量这个优化的效果？除了延迟还有什么？

> 三个维度。第一，**延迟**——从每次 7s 降到首篇 6.5s + 后续 1.85s。第二，**资源利用率**——M4 的 16GB 统一内存始终有 2.9GB 被模型占用，但这是有意为之的 trade-off（用空间换时间）。旧方案模型反复加载/卸载，看起来省显存，但对频繁调用场景是无效的资源抖动。第三，**代码简洁性**——删掉了 subprocess、字符串拼接、timeout 处理等 20 行代码，替换为 1 行 `extractor.extract(abstract)`。

### Q8: 这个优化在整个 Small-to-Large 架构里的意义是什么？

> Small-to-Large 架构的前提是 small model 必须足够快，否则用户会问"为什么不直接用大模型"。旧方案 subprocess 每次 7s 的开销让这个前提受到质疑——如果每次调用都这么慢，零费用的优势就被延迟劣势抵消了。优化到 1.85s/篇后，small model 真正体现出了"高频、低延迟、零费用"的价值，和 large model 形成了清晰的互补关系：small model 负责 90% 的结构化提取任务，large model 负责 10% 需要深度推理的 GRADE 评估。
