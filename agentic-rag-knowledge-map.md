# Agentic RAG 知识地图 v1 — 完整 Agentic RAG 体系深度版

> **专题深度文档**: 完全面向 Agentic RAG (智能体增强检索, RAG Gen 4 范式) 的知识体系.
>
> 跟 [RAG 知识地图通用版](./rag-knowledge-map.html) 互补 — 通用版讲 Naive/Advanced/Modular/Agent 4 代全栈, 本文档专精 Agent 一代深度.
>
> 适合: Agent 开发者 / Agent 架构师 / 想深入 Agentic RAG 落地的工程师 / 面试 Agent RAG 相关岗位.
>
> 来源参考: Anthropic "Building Effective Agents" (2024.12) + Anthropic Claude Code SDK + 业界 2024-2026 最新实践.

---

## 〇. 速览 + 4 个核心决策 + 学习路径

### 0.0 Agentic RAG 速览思维导图 ⭐

> 进入本章前先看这张思维导图建立全章认知.

### 0.1 一句话定义 Agentic RAG

- **Agentic RAG** (智能体增强检索) = LLM 在循环里**自主决定**"下一步检什么 / 调什么 / 何时停"的多步 RAG
- 把"一次检索 → 一次回答"的固定管道, 升级为"LLM 在循环里自己开车"的动态决策过程
- **核心标志**: 不是步数多 (固定脚本调 5 次 LLM 也不算), 而是**LLM 看上一步结果决定下一步**这个反馈环
- 类比: 普通 RAG = 预先写好菜谱按步骤做菜; Agentic RAG = 厨师根据冰箱里有什么 + 客人口味临场决定怎么做

### 0.2 跟普通 RAG 的本质差别 (一图)

| 维度 | 普通 RAG (Modular) | Agentic RAG |
|---|---|---|
| 路径决定者 | 工程师写好的代码 | LLM 在运行时决定 |
| 流程图 | 能预先画出来 | 取决于运行时 LLM 输出 |
| 可预测性 | 高 (相同输入相同流程) | 低 (LLM 决策有方差) |
| LLM 调用次数 | 1-3 次 | 5-50 次 |
| 单 query 成本 | $0.005-0.05 | $0.05-1 |
| 单 query 延迟 | 1-3s | 5-30s |
| 适用流量比例 | 80-95% | 5-20% |
| 调试复杂度 | 中 | 高 (必须 LangSmith 追踪) |

### 0.3 Anthropic 三层模型 (业界统一框架)

Anthropic "Building Effective Agents" (2024.12) 推出的三层架构, 是当前业界共识:

- **层次 1 — Augmented LLM** (90% 场景): 单次 LLM 调用 + 检索 + 工具调用 + 记忆. 即标准 Modular RAG, 一次进出.
- **层次 2 — Workflow** (8-15% 场景): 工程师写死多步流程, LLM 在固定节点上工作 (路径预先确定). 5 种主流 Pattern: Prompt Chaining / Routing / Parallelization / Orchestrator-Workers / Evaluator-Optimizer.
- **层次 3 — Agent** (2-5% 场景): LLM 在循环里自主决策, 路径运行时生成. 5 种主流形态: Plan-and-Execute / ReAct / Multi-Agent / Self-Reflection / Iterative.

**核心原则**: **优先用层次 1, 失败再上层次 2, 再失败才上层次 3.** 跳层是 RAG 项目失败首因.

### 0.4 4 个核心决策

| 决策点 | 选项 | 决策依据 | 详见 |
|---|---|---|---|
| **决策 1 — 该用哪一层?** | Augmented LLM / Workflow / Agent | 单次 RAG 能解 → 层 1; 步骤可预先固定 → 层 2; 步骤需 LLM 决定 → 层 3 | §1.4 决策树 |
| **决策 2 — 选哪种 Workflow Pattern?** | Prompt Chaining / Routing / Parallelization / Orchestrator-Workers / Evaluator-Optimizer | 任务能拆线性 → Chaining; 有类别 → Routing; 子任务独立 → Parallel; 子任务运行时知 → Orchestrator; 高质量 + 评估标准 → Evaluator | §4 |
| **决策 3 — 选哪种 Agent 形态?** | Plan-and-Execute / ReAct / Multi-Agent / Self-Reflection / Iterative | 可预先分解 → P&E; 不可预测 → ReAct; 多角色 → Multi-Agent; 输出后自评 → Self-Reflection; 检索不全循环 → Iterative | §2 |
| **决策 4 — 选哪个 Agent 框架?** | LangGraph / LlamaIndex / AutoGen / CrewAI / OpenAI Agents SDK / Anthropic Claude Agent SDK | 已用 LangChain → LangGraph; 多 Agent 协作 → AutoGen; 极简 → CrewAI; 单 LLM 厂 → OpenAI/Anthropic SDK | §9 |

### 0.5 全文 14 章地图

| 章 | 核心内容 | 适合谁 |
|---|---|---|
| §0 | 速览 + 决策框架 | 所有人 |
| §1 | Anthropic 三层模型 (Agent vs Workflow vs Augmented LLM) | 想理解概念边界 |
| §2 | Agent 5 大形态深度 (Plan-and-Execute / ReAct / Multi-Agent / Self-Reflection / Iterative) | Agent 设计者 |
| §3 | 7 层架构 + 决策循环 (Layer 1-7 + Cost Controller 横切) | 架构师 |
| §4 | Workflow 5 Pattern 深度 (Anthropic 官方框架) | Workflow 设计者 |
| §5 | Tool Calling 深度 (三家 API + MCP + Computer Use + Browser Use) | Tool 工程师 |
| §6 | Memory 深度 (三层 + Episodic/Semantic/Procedural/Skill) | Memory 设计 |
| §7 | Multi-Agent 系统 (Orchestrator/Hierarchical/Swarm/Magentic-One) | Multi-Agent 设计 |
| §8 | 高级 RAG-Agent 模式 (Self-RAG/CRAG/GraphRAG/LightRAG/Adaptive/Reflexion/ToT) | 算法研究 |
| §9 | Agent 框架对比 (8 主流 + 选型决策) | 框架选型 |
| §10 | 死循环防御 + FinOps + 评估 | SRE / 运维 |
| §11 | 真实落地案例深度 (8 个标杆) | 学经验 |
| §12 | 失败模式 + 安全 (Tool Misuse / Memory Leak / Prompt Injection) | 安全 / 运维 |
| §13 | 落地路径 + 最佳实践 (4 阶段 + 团队 + 国产化) | PM / 项目经理 |
| §14 | 未来趋势 (2026-2027) | 战略思考 |

### 0.6 学习路径 (按角色)

| 角色 | 推荐路径 | 跳过 |
|---|---|---|
| **Agent 新手** | §0 → §1 → §3 → §4 → §13 | §7 §12 §14 |
| **Agent 架构师** | §0 → §1 → §3 → §5 → §6 → §7 → §10 → §11 | §13 §14 |
| **Agent 工程师** | 全文 (按 §0 → §14 顺序) | / |
| **算法研究** | §0 → §2 → §8 → §14 | §10 §13 |
| **PM / 业务** | §0 → §1 → §11 → §13 → §14 | §3-§7 §9-§10 |
| **面试准备** | §0 → §1 → §2 → §3 → §11 → §12 | §13 §14 |

### 0.7 跟通用 RAG 文档的关系

- [RAG 通用知识地图](./rag-knowledge-map.html) — 18,941 行, 覆盖 4 代 RAG (Naive/Advanced/Modular/Agent) 全栈 + 5 层企业架构 + 60+ 面试题
- 本文档 — 8000-12000 行, 完全聚焦 Gen 4 Agentic RAG, 深度专精
- 内容关系: 通用版 §20 章是本文档的"超级浓缩版", 本文档是 §20 的"专题展开版" (10× 深度)

---

## 一. Anthropic 三层模型 — Agent vs Workflow vs Augmented LLM

### 1.0 Anthropic 三层模型思维导图 ⭐

> 进入本章前先看这张思维导图建立全章认知.

### 1.1 三层模型来源

- **出处**: Anthropic "Building Effective Agents" (2024.12 engineering blog)
- **作者**: Anthropic 团队, 总结跟数十个客户合作经验
- **核心观点**: 成功的 agentic 实现不依赖复杂框架, 而是 "simple, composable patterns" (简单可组合的模式)
- **影响**: 业界 (OpenAI / Google / 国内大厂) 都接受这个三层分类

### 1.2 三层完整定义

#### 1.2.1 层次 1 — Augmented LLM (增强型 LLM)

##### 是什么
- 单次 LLM 调用, 输入是 system + 检索回来的资料 + 用户 query, 输出答案
- 即标准的 Modular RAG (RAG 通用文档 §19), 一次进出
- 是所有 agentic 系统的**原子单位**

##### 适用 (80-95% 场景)
- 简单问答 / FAQ / 单点查询
- 标准客服: "退款政策是什么?"
- 知识检索: "去年 Q3 销售额"
- 任何"单次 RAG 能解决"的 query

##### 工程实现
- 1 次 Embedder 调用 (query embed)
- 1 次向量库 + 倒排索引检索 (Hybrid)
- 1 次 LLM 调用 (生成答案)
- 总延迟 1-3s, 总成本 $0.005-0.05/query

##### 决策: 永远先尝试这层
- 单次 RAG 能解决的, 永远用这层
- 跳过这层直接上 Agent 是 RAG 项目失败首因 (§13.X 反模式)

#### 1.2.2 层次 2 — Workflow (工作流)

##### 是什么
- 工程师写死多步流程, LLM 在每个固定节点上做事 (路径预先确定)
- **关键特征**: 路径固定, 流程图能预先画出来
- 5 种主流 Pattern (§4 详讲)

##### 适用 (8-15% 场景)
- 任务可预先分解 (e.g. "分类 → 路由 → 答 → 校验")
- 子任务独立可并行 (e.g. "10 个文档每个独立摘要")
- 多角色协作但角色固定 (e.g. Researcher + Writer + Critic)

##### 工程实现
- 几十行代码 + 标准 LLM API (不需要 LangGraph 等重框架)
- 总延迟 3-8s, 总成本 $0.02-0.10/query

##### 决策: 任务可预先固定 → 用 Workflow, 不要上 Agent
- Workflow 比 Agent 简单 5-10×
- Workflow 可预测 + 可调试 + 不会死循环

#### 1.2.3 层次 3 — Agent (智能体)

##### 是什么
- LLM 在循环里自己决定下一步, 路径运行时生成
- **关键特征**: 路径动态, 流程图取决于运行时 LLM 输出, 不能预先画
- 5 种主流形态 (§2 详讲)

##### 适用 (2-5% 场景)
- 任务无法预先分解 (e.g. Coding / 跨多源诊断 / 探索性研究)
- 需要 LLM 自己判断"何时停"
- 跨 3+ 数据源的诊断
- 需要执行操作 (创建工单 / 修代码 / 提 PR)

##### 工程实现
- 必须用 Agent 框架 (LangGraph / OpenAI Agents SDK 等)
- 必须上追踪 (LangSmith / Phoenix / Langfuse)
- 必须设防线 (max_steps / budget cap / repeat 检测)
- 总延迟 5-30s, 总成本 $0.05-1/query

##### 决策: 只有前两层都不够时才上
- 跳层 (从 Augmented LLM 直接到 Agent) 是 RAG 项目失败首因
- Agent 调试成本 5-10× Workflow, 能不上就不上

### 1.3 三层选型决策树

#### 1.3.1 决策树 (3 个问题)

- **Q1**: 单次 RAG 能解决吗 (一次检索 + 一次 LLM 调用)?
  - YES → **层次 1 Augmented LLM** (停, 别上 Agent)
  - NO → 进 Q2
- **Q2**: 步骤可预先固定写脚本吗 (能画出流程图)?
  - YES → **层次 2 Workflow** (5 种 Pattern 选一, §4)
  - NO → 进 Q3
- **Q3**: 步骤需要运行时 LLM 自己决定?
  - YES → **层次 3 Agent** (5 种形态选一, §2)
  - NO → 重新审视, 大概率回到 Q1 / Q2

#### 1.3.2 三层量化对比 (Klarna 实测)

| 流量比例 | 层次 | 单次成本 | 单次延迟 | 满意度 |
|---|---|---|---|---|
| 80% | 层次 1 (普通 RAG) | $0.008 | 1.2s | 4.5/5 |
| 15% | 层次 2 (Workflow + 增强) | $0.02 | 2-3s | 4.6/5 |
| 5% | 层次 3 (Agent Plan-and-Execute) | $0.42 | 8.3s | 4.7/5 |

数据来源: Klarna 2024 Q1 公开年报 + Glean 内部分享.

### 1.4 三大常见误区 (面试高频, 完整深度版)

#### 1.4.1 误区 1 — "Agent 替代 RAG"

##### 错在哪
- 听到 "Agent" 觉得是更高级的东西, 以为有了 Agent 就不需要 RAG 管道了
- 部分团队 PoC 时直接上 LangGraph 重写整个系统, 把原来 Modular RAG 删掉
- 误以为 "Agent + LLM" 就是完整方案, 不需要"老的"检索流水线

##### 真相
- Agent 内部 80-90% 时间还在调 RAG (检索是 Agent 工具池里的核心工具之一)
- Agent 没有取代 RAG, 而是把 RAG 当工具用
- 所以 "Agent vs RAG" 是错误问题, 正确问题是 "Agent + RAG vs 单 RAG"

##### 量化证据
- **Klarna 公开年报数字**: 95% query 走纯 Modular RAG, 5% 走 Agent (而 Agent 内部仍多次调 RAG)
- **Glean 内部分享**: 90% query 走纯 RAG (Confluence 检索), 10% 走 Agent (跨多源诊断)
- **Cursor 例外**: 几乎全 Agent (Coding 任务无法预先分解), 但 Agent 内部仍频繁 grep 代码 (本质是检索)

##### 正解
- Agent 是 RAG 的**上层调度**, 是叠加不是替代
- Agent 5 部件公式: Agent RAG = Modular RAG (基座) + Planner + Tool Calling + Memory + 多步推理
- 没 Modular RAG 这个基座, Agent 检出来的全是垃圾, 循环报错

##### 反模式
- ❌ 删除原 Modular RAG 改全 Agent: 月成本翻 50× (Klarna 早期就栽过)
- ❌ Agent 不调 RAG 工具, 让 LLM 凭训练知识答: 知识过时 + 幻觉爆
- ❌ 部分团队上 Agent 时把 Embedder / Reranker / Hybrid 这些 Modular RAG 组件全删: 一定召不准

##### 决策建议
- 永远先做 Modular RAG (Recall@10 ≥ 0.85) 才考虑上 Agent
- Agent 是"在 Modular RAG 之上加智能调度", 不是替代
- 详见 §13 落地路径 (待写) 4 阶段渐进

#### 1.4.2 误区 2 — "多步 LLM 调用 = Agent"

##### 错在哪
- 看到 "Agent" 概念, 以为多步 LLM 调用 = Agent
- 工程师写死的 5 步固定脚本调 5 次 LLM, 自称 "我的 Agent"
- 把 Workflow Pattern 1 (Prompt Chaining) 当 Agent 卖

##### 真相
- 多步 ≠ Agent. **关键看路径是不是 LLM 决定的, 不是步数**
- 5 步固定脚本是 Workflow (Anthropic 三层模型层次 2)
- Agent 的核心标志是 "LLM 看上一步结果决定下一步" 这个**反馈环**

##### 鉴别口诀
- **能预先画出流程图** (即使有 if/else 分支) → Workflow
- **流程图取决于运行时 LLM 输出, 不能预先画** → Agent
- 例子: 客服分类 → 走对应路径 (3 路分类预先固定) = Workflow Pattern 2 Routing
- 例子: Cursor 改 bug (LLM 自己决定 read 哪个文件 / 改哪一行 / 跑哪个测试) = Agent ReAct

##### 量化证据 (架构对比)
- Workflow 单 query LLM 调用次数: 1-3 (含 Query Transform / Validator)
- Agent 单 query LLM 调用次数: 5-50 (循环动态)
- Workflow 同输入相同流程 (确定性): YES
- Agent 同输入流程不同 (LLM 决策有方差): YES

##### 正解
- 多步 LLM 调用 ≠ Agent
- 评估标准: 路径是不是 LLM 在运行时决定
- Workflow + Agent 都属于"超越单次 LLM 调用", 但路径决定者不同

##### 反模式
- ❌ 把 Prompt Chaining (Pattern 1) 包装成"我的 Agent"卖: 客户被坑, 真正需要 Agent 时上不去
- ❌ 用 LangGraph 写 5 步固定脚本: overkill, 用 LangChain SequentialChain 几行搞定
- ❌ 用 Agent 框架做 Workflow 任务: 调试成本 5-10×, 不值

##### 决策建议
- 任务能预先画出流程图 → Workflow (5 Pattern 选一)
- 任务路径必须 LLM 运行时决定 → Agent (5 形态选一)
- 详见 §1.3 三层选型决策树

#### 1.4.3 误区 3 — "上 Agent 就解决质量问题"

##### 错在哪
- RAG 召回质量差 (Recall@10 = 0.5), 想用 Agent 抢救
- 觉得 Agent "更智能", 应该能弥补检索的不足
- 期待 Agent "自动找资料, 自动判断, 自动答对"

##### 真相
- Agent 解决的是 "**一次性管道解不了**" 的问题, 不解决 "检索本身差" 的问题
- Recall@10 < 0.7 时上 Agent 只会**循环报错** (LLM 反复检索同一查询拿到相同的差结果)
- max_steps 限制再严, Agent 也只是"循环烧钱", 答案质量不会变好

##### 量化证据 (反模式真实事故)
- 某厂 Modular RAG Recall@10 = 0.5, 上 LangGraph 期待 Agent 救场
- 结果: Agent 循环 8 步, 每步检索拿到相同的差 chunk, LLM 综合答案仍差
- 单 query 成本 $0.40 (vs 普通 RAG $0.005), 月预算翻 80×, 满意度反而降 5pt
- 修复: 砍掉 Agent, 回头治 L1 数据治理 + L2 索引 + L3 检索, Recall@10 0.5 → 0.85, 用户满意度回升

##### 正解
- Agent 跟检索质量是**正交**关系, 不是补救
- 必须先把 **Modular RAG 调到 Recall@10 ≥ 0.85** 才考虑上 Agent
- 顺序: L1 数据治理 → L2 索引 → L3 检索 → L4 Router → L5 Agent (5 层渐进, 详见通用 RAG §3)

##### 量化对比 (Klarna 实测)
- Recall@10 < 0.7 时上 Agent: ROI 负 (成本翻 50× + 满意度降)
- Recall@10 ≥ 0.85 时上 Agent: ROI 正 (5% 复杂 query 满意度 +5pt, 解决跨系统诊断)

##### 反模式
- ❌ Modular RAG 没调好就上 Agent: 必栽
- ❌ 期待 Agent "自动学会"修复检索质量: LLM 再强也救不了垃圾 chunk
- ❌ 把 Agent 当"质量提升神器"卖给业务方: 上线后体验崩, 业务方失去信任

##### 决策建议
- **正确顺序**: 先治 L1 数据治理 + L2 索引 + L3 检索, 再上 L5 Agent
- 检验标准: Modular RAG 在 Golden Set 上 Recall@10 ≥ 0.85 + Faithfulness ≥ 0.90 才能上 Agent
- 详见 §13 落地路径 4 阶段渐进 (待写)

### 1.5 跟其它框架的关系 (完整深度版)

#### 1.5.1 跟 OpenAI 的 GPT 模式 / Agents SDK

##### OpenAI 立场
- OpenAI 没有公开等价的"三层模型"分类
- 实践中 OpenAI 把 Agent 等所有"超越单次 GPT 调用"的统称 "Agentic" 或 "Function Calling"

##### 实质对应关系
- **OpenAI Function Calling** ≈ Anthropic 的 Workflow + Agent (取决于工程师怎么用)
  - 单步 Function Calling (GPT 决定调一个工具就完事) ≈ Augmented LLM (层次 1)
  - 多步 Function Calling (GPT 决定调多个工具串起来) ≈ Workflow / Agent
- **OpenAI Agents SDK** (前 Swarm, 2025.03 升级) ≈ Anthropic 的 Multi-Agent Agent
  - 内置 handoff (Agent 之间转交) + MCP client
  - 主推 Multi-Agent 模式

##### 关键差异
- Anthropic 偏概念化 (三层模型), OpenAI 偏实操工具 (SDK + API)
- Anthropic 推 "simple, composable patterns", OpenAI 推完整 SDK
- 但底层概念可互换, 工程师跨平台都能用

##### 跨平台兼容
- LangGraph / LlamaIndex 跨 Anthropic + OpenAI 都能跑
- MCP (Model Context Protocol) 是跨平台标准, Anthropic 主导, OpenAI 2025 接入

#### 1.5.2 跟 LangChain / LangGraph 的 Chain / Agent 对应

##### LangChain Chain (经典)
- **SimpleSequentialChain** ≈ Workflow Pattern 1 Prompt Chaining
- **RouterChain** ≈ Workflow Pattern 2 Routing
- **MapReduceChain** ≈ Workflow Pattern 3 Parallelization (sectioning)
- **RefineChain** ≈ Workflow Pattern 5 Evaluator-Optimizer

##### LangChain Agent (经典)
- **ZeroShotAgent / ReActAgent** ≈ Anthropic ReAct Agent (5 形态形态 2)
- **OpenAIFunctionsAgent** ≈ ReAct Agent 但用 Function Calling
- **SelfAskWithSearchAgent** ≈ Iterative RAG (5 形态形态 5)

##### LangGraph (跨两层)
- **graph + node + edge** 抽象, 任意编排 Workflow + Agent
- 既能写 Workflow (固定 graph) 也能写 Agent (动态决策的 graph)
- 业界主流的"二选一框架": 简单用 LangChain, 复杂用 LangGraph

##### LlamaIndex Agents
- **ReActAgent** ≈ Anthropic ReAct
- **SubQuestionQueryEngine** ≈ Workflow Pattern 4 Orchestrator-Workers
- **OpenAIAgent** ≈ OpenAI Function Calling 风格
- 跟 LlamaIndex RAG 检索深度集成

#### 1.5.3 跟 Modular RAG (Yunfan Gao 2024) 的关系

##### Modular RAG 是什么
- Yunfan Gao 2024.07 综述 "Modular RAG: Transforming RAG Systems" (arXiv:2407.21059)
- 把 RAG 系统重构为 7 个模块化组件 (Indexing / Pre-Retrieval / Retrieval / Post-Retrieval / Generation / Routing / Orchestration)
- 每模块独立可替换, 接口标准化

##### 跟 Anthropic 三层模型对应
- **Modular RAG = 层次 1 Augmented LLM 的工程化实现**
- 7 模块化的 Augmented LLM, 主推理仍是 1 次 LLM 调用 (Generator)
- 周边 Pre-Retrieval / Post-Retrieval / Validator 是辅助调用

##### 跟 Agent 的关系
- **Agent = Modular RAG (基座) + Planner + Tool Calling + Memory + 多步推理**
- Modular RAG 是 Agent 的"内部工具", 即 Agent Layer 4 Tool Loop 里的"RAG 工具" 就是 Modular RAG 完整管道
- 没 Modular RAG, Agent 检出来的全是垃圾, 循环报错

##### 演化路径
- Naive RAG (2020-2022, Gen 1) — 3 段固定管道
- Advanced RAG (2023, Gen 2) — Naive + 增强 (HyDE / Multi-Query / Reranker / Hybrid)
- **Modular RAG (2024, Gen 3) — 7 模块化, 是 Anthropic 层次 1 Augmented LLM 工程实现**
- **Agent RAG (2024-2025, Gen 4) — 在 Modular RAG 之上加 Planner / Tool Loop / Memory**

#### 1.5.4 跟 Google Gemini 生态

##### Google ADK (Agent Development Kit, 2025)
- Google 2025 推出的 Agent 开发框架
- 跟 Vertex AI Agent Builder 集成
- 类似 OpenAI Agents SDK + Anthropic Claude Agent SDK 的角色

##### Project Mariner (Browser Agent, 2024.12)
- Google 浏览器 Agent, 类似 Anthropic Computer Use
- 但聚焦 Chrome 浏览器场景

##### 三家差异
- Anthropic: 偏概念化 + 主导 MCP 协议 + Computer Use
- OpenAI: 偏 SDK + Operator (浏览器 Agent)
- Google: 偏 enterprise + Vertex AI 集成 + Mariner

#### 1.5.5 跟国内 Agent 生态

##### 字节: Coze
- 飞书生态内的 Agent 平台, 拖拽式构建
- 偏 no-code, 适合 PM / 业务方
- 跟 Anthropic 三层模型对应: Workflow + Agent 都支持

##### 阿里: 百炼 + Spring AI Alibaba
- 通义千问 + 工具调用 + Multi-Agent
- Spring AI Alibaba 是 Java 生态 Agent 框架
- 适合 Java 重度企业

##### 智谱: GLM-4 系列
- GLM-4 / GLM-4.5 / GLM-Z1 系列
- Agent 能力跟 Sonnet 4.5 类似
- 国产化合规首选

### 1.6 关键金句 + 解读 (Anthropic 原话深度展开)

#### 1.6.1 金句 1 — 设计哲学

> "成功不在于构建最复杂的系统, 而在于为你的需求构建正确的系统"

##### 解读
- Anthropic 反复强调"反过度工程化"
- 业界常见错误: 看到 LangGraph 就想用, 看到 Multi-Agent 就跟风
- 正确做法: **从最简单方案开始, 验证不够才升级**

##### 实操建议
- PoC 阶段: 用 Augmented LLM (单 LLM + 检索) 跑通
- 验证不够: 加 Workflow (5 Pattern 选一)
- 仍不够: 才上 Agent (单 Agent ReAct 起步)
- 最后: Multi-Agent (绝对必要时)

#### 1.6.2 金句 2 — 演化路径

> "从简单提示开始, 用全面的评估优化它们, 仅在简单方案不足时添加多步 agentic 系统"

##### 解读
- Anthropic 推荐"Eval-driven"开发
- 不要凭感觉决定是否上 Agent, 要看 evaluation 数据
- "全面的评估" = RAGAS 4 指标 + Golden Set + 用户反馈

##### 实操建议
- 上线前必跑 evaluation (Recall@10 / Faithfulness / 用户满意度)
- 单 LLM 调用 evaluation 不达标 → 加 Workflow Pattern
- Workflow evaluation 不达标 → 才考虑 Agent
- 详见 §10 评估 (待写) + 通用 RAG §14

#### 1.6.3 金句 3 — Agent 的代价

> "agentic 系统通常会用延迟和成本换取更好的任务性能"

##### 解读
- Agent 不是"免费午餐", 必须明确知道 trade-off
- 延迟 5-30s vs Modular RAG 1-3s (慢 5-10×)
- 成本 $0.05-1/query vs Modular RAG $0.005-0.05 (贵 10-50×)
- "更好的任务性能" 必须能量化, 否则不值

##### 实操建议
- 上 Agent 前算 ROI: (满意度提升) × (复杂 query 数) > (成本增量)?
- Klarna 案例: 5% Agent 流量解决"跨系统诊断", 满意度 +5pt, ROI 正
- 反例: 简单 FAQ 上 Agent, 满意度无提升 + 成本翻 50×, ROI 负

#### 1.6.4 金句 4 — 框架陷阱

> "对框架的错误假设是客户错误的常见来源"

##### 解读
- 框架 (LangGraph / AutoGen / CrewAI) 创建抽象层, 隐藏底层 prompt + LLM 调用
- 工程师容易"信任框架不读源码", 出错时不知道为什么
- Anthropic 建议: 从直接调用 LLM API 开始, 理解原理再用框架

##### 实操建议
- PoC 阶段: 直接用 anthropic SDK / openai SDK 写, 看清楚每个 prompt 长啥样
- 验证 OK 再迁移到框架 (LangChain / LangGraph)
- 用框架时**必读源码** (尤其错误处理 / retry / cache 部分)
- 反模式: 上线后出 bug, 改完不验证, 凭运气

#### 1.6.5 金句 5 — 工具是核心 (新加, Anthropic 2024.12 后续 blog)

> "我们在 SWE-bench Agent 中花在工具优化上的时间, 比花在 prompt 优化上还多"

##### 解读
- Tool Calling 是 Agent 的核心, 但常被低估
- 工具描述 (description) + 输入输出 schema 决定 LLM 选错率
- Anthropic 实测: 优化工具描述 (清晰 / 含 few-shot) 比改 prompt 收益高 2-3×

##### 实操建议
- 工具描述要写清"什么场景用 / 输入输出 / 失败行为"
- 工具池 5-12 个 (太多 LLM 选错率塌)
- 详见 §5 Tool Calling 深度 (待写)


## 二. Agent 5 大形态深度 (Plan-and-Execute / ReAct / Multi-Agent / Self-Reflection / Iterative)

### 2.0 Agent 5 大形态思维导图 ⭐

> 进入本章前先看这张思维导图建立全章认知.

### 2.1 5 大形态总览

| 形态 | 一句话 | 适用 | 代表系统 |
|---|---|---|---|
| **Plan-and-Execute** | 开局先全规划 N 步, 再串行执行 | 任务可预先分解 (退款诊断) | Klarna 客服 / Anthropic Computer Use |
| **ReAct** | 每步规划下一步, 边推理边行动 | 任务不可预测 (Coding) | Cursor / Devin / Claude Code |
| **Multi-Agent** | 多角色协作 (Researcher/Writer/Critic) | 内容创作 / 跨域协作 | Microsoft Copilot Workspace / Magentic-One |
| **Self-Reflection** | 输出后自评, 不满意则重做 | 高质量需求 + 自带评估 | Self-RAG / Reflexion |
| **Iterative** | 检索→评估→不够则重检, 直到信息充分 | 跨多源诊断 / 多跳推理 | CRAG / Iterative RAG |

### 2.2 形态 1: Plan-and-Execute (规划-执行解耦)

#### 2.2.1 解决什么问题
- 复杂任务一次 LLM 输出不准, 容易乱跳乱抓
- ReAct 每步重新思考慢且贵 (每步都用 frontier LLM)
- Plan-and-Execute 解法: **开局先用强 LLM 一次出完整 N 步 plan, 再用便宜 LLM 按 plan 串行执行**, 一次复杂规划摊销到多步执行

#### 2.2.2 算法 (核心循环)
- 步 1 — Planner LLM (Sonnet 4.5 / GPT-5 / o3) 接 query 生成结构化 plan:
  - plan = [step_1: 调 tool_A(args), step_2: 调 tool_B(args), step_3: 综合, ...]
- 步 2 — 进入 Executor 循环 (按 plan 串行执行):
  - for step in plan: result = execute(step); state.append(result)
- 步 3 — Synthesizer LLM (Haiku 4.5 / GPT-5-mini) 综合所有 step 结果生成最终答案

#### 2.2.3 完整执行流程 — 退款诊断案例

- 用户 query: "用户 U123 反馈未收到退款"
- 步 1 — Planner 输出 plan:
  - step1: 查订单系统 — 用户 U123 最近 30 天订单
  - step2: 检查退款 API — 哪些订单已发起退款
  - step3: 检查支付网关 — 退款是否到账
  - step4: 综合给出诊断
- 步 2 — Executor 按 plan 串行执行:
  - step1: 调 order_api(user="U123", days=30) → 3 单
  - step2: 调 refund_api(order_ids=[...]) → 1 单已退
  - step3: 调 payment_gateway_api(refund_id=...) → 状态: pending
  - step4: 综合 → "退款已发起但银行处理中, 预计 3-5 工作日"
- 步 3 — 输出最终答案 + 引用所有调用记录

#### 2.2.4 关键设计: Planner / Executor / Synthesizer 模型分级

| 角色 | 推荐 LLM | 为什么 | 单次成本 |
|---|---|---|---|
| **Planner** | Sonnet 4.5 / GPT-5 / o3 / DeepSeek-R1 | 必须强推理, plan 错则全错 | $0.05-0.15 |
| **Executor** | Haiku 4.5 / GPT-5-mini / Gemini 2.0 Flash | 执行简单 (调 tool), 用便宜 LLM 省 5-10× | $0.001-0.005 × N steps |
| **Synthesizer** | Haiku 4.5 / GPT-5-mini | 综合不吃推理力, Haiku 够用 | $0.005-0.02 |

**总成本**: 1 × $0.10 (Sonnet plan) + 8 × $0.005 (Haiku execute) + 1 × $0.01 (Haiku synth) = $0.15
**vs ReAct**: 8 × $0.10 (Sonnet 每步) = $0.80 (省 5×)

#### 2.2.5 优势
- 步骤清晰可解释 (Plan 是显式的, 用户能看到全流程)
- 减少 LLM 调用 (规划只算 1 次, vs ReAct 每步重新规划)
- 易调试 (Plan 错可单步重试, 不用全部重跑)
- 成本低 (Executor 用 Haiku, 单 query 比 ReAct 省 5-10×)
- 可缓存 (相似 query 的 plan 可复用)

#### 2.2.6 劣势
- Planner 错则全错 (无 mid-correction, 一旦 plan 偏就废)
- 不适合探索性任务 (Plan 时未知信息)
- Plan 跟 reality 偏差大时只能重 plan
- 适合"已知怎么解决"的任务, 不适合"边探索边解决"

#### 2.2.7 真实采用
- **Klarna AI 客服** (5% Agent 流量走 Plan-and-Execute)
- **Anthropic Computer Use** (内部架构, 推测)
- **Microsoft Copilot Workspace** (Plan-Implement-Review 三步, 公开博客)
- **LangGraph 官方 Plan-and-Execute 模板**

#### 2.2.8 反模式
- ❌ 用 Plan-and-Execute 做开放探索 (e.g. "帮我研究这个新领域") — 应该用 ReAct
- ❌ Planner 用便宜 LLM (Plan 错全盘崩) — 必须用强推理 LLM
- ❌ 不允许 mid-correction — 复杂任务总有意外, 必须支持 plan 调整
- ❌ Plan 步数 > 10 — 拆得太细, LLM 规划质量塌

### 2.3 形态 2: ReAct (Reasoning + Acting, Yao et al. 2022)

#### 2.3.1 解决什么问题
- 复杂任务无法预先全规划 (e.g. Coding 任务, 改 bug 需要边读代码边发现问题)
- Plan-and-Execute 假设"开局能想清楚", 但探索性任务必失败
- ReAct 解法: **LLM 在每一步交替输出"思考 (Thought)" + "行动 (Action)", 接收"观察 (Observation)"**, 形成 Thought→Action→Observation→Thought→... 单步循环

#### 2.3.2 算法 (核心循环)
- 步 1 — LLM 看当前 state (含 query + history + 上一步 observation), 输出:
  - Thought: "我需要先查 X..."
  - Action: tool_call(tool_name, args)
- 步 2 — 执行器调用 tool, 拿到结果
- 步 3 — 把 tool 结果作为 Observation 加入 state
- 步 4 — 回步 1 继续 (直到 LLM 输出 Final Answer 或 max_steps)

#### 2.3.3 完整执行流程 — Cursor 修 bug 案例

- 用户 query: "fix the bug in user_service.py"
- 循环步 1:
  - Thought: 我需要先看 user_service.py 内容
  - Action: read_file("user_service.py")
  - Observation: <文件内容 200 行>
- 循环步 2:
  - Thought: 看起来 line 45 的 if 判断错了, 但我得看 caller
  - Action: grep("user_service.create_user")
  - Observation: <3 处 caller>
- 循环步 3:
  - Thought: caller 都期待返回 User 对象, 但当前返回 dict
  - Action: edit_file("user_service.py", line=45, change="return User(**data)")
  - Observation: <编辑成功>
- 循环步 4:
  - Thought: 跑测试验证
  - Action: run_test("test_user_service.py")
  - Observation: <测试通过>
- 循环步 5:
  - Thought: 修好了
  - Final Answer: "已修复 user_service.py:45 的 bug, 改为返回 User 对象, 测试通过"

#### 2.3.4 优势
- 灵活 (任务边走边定, 适合探索性)
- 透明 (每步 Thought 可见, 易理解 LLM 推理过程)
- 错误恢复 (一步错可下一步纠正)
- 通用 (任何 Tool Calling 场景都能用)

#### 2.3.5 劣势
- 慢 (每步都要 frontier LLM 思考)
- 贵 (每步 $0.05-0.15, 8 步 = $0.4-1.2)
- 容易迷路 (无全局视野, 长任务易忘记目标)
- 容易死循环 (没有 Plan 约束, 可能反复同样的事)

#### 2.3.6 真实采用
- **Cursor Composer** (代码 Agent, 用 Sonnet 4.5 + ReAct)
- **Devin** (Cognition, Software Engineer Agent)
- **Anthropic Claude Code** (终端 Agent)
- **LlamaIndex ReAct Agent** (默认)
- **LangChain ReAct Agent** (经典)

#### 2.3.7 ReAct vs Plan-and-Execute 决策

| 任务类型 | 选哪个 | 理由 |
|---|---|---|
| 退款诊断 (步骤可预知) | Plan-and-Execute | 省钱 + 可解释 |
| Coding 修 bug | ReAct | 探索性, 边读边改 |
| 写竞品分析 | Plan-and-Execute | 框架可预先定 |
| 法律研究 | ReAct + Self-Reflection | 边检索边判断够不够 |
| 报表生成 | Plan-and-Execute | 步骤固定 |
| 数据分析探索 | ReAct | 不知道结论在哪 |

#### 2.3.8 反模式
- ❌ 用 ReAct 做退款诊断 — 太贵, Plan-and-Execute 省 5×
- ❌ 不限 max_steps — Cursor 类任务可 50 步, 但客服必须 ≤ 8
- ❌ Thought 不写出来 (省 token) — 失去 ReAct 透明优势
- ❌ Observation 太长 (整个文件 5000 行) — 超 context, 必须摘要

### 2.4 形态 3: Multi-Agent 协作 (AutoGen / CrewAI / Magentic-One)

#### 2.4.1 解决什么问题
- 复杂任务需要多角色 (Researcher + Writer + Critic), 单 Agent 难胜任
- 角色分工 + 互相 review 提升质量
- Multi-Agent 解法: **多个独立 Agent 各自专精, 通过对话 / handoff / shared state 协作**

#### 2.4.2 主流模式

##### 模式 A: Orchestrator + Workers (协调员 + 工人)
- 1 个 Orchestrator Agent 接 query, 拆任务分给 N 个 Worker Agent
- Worker 执行后, Orchestrator 综合
- 代表: Microsoft Magentic-One (1 Orchestrator + 4 Workers: WebSurfer / FileSurfer / Coder / ComputerTerminal)

##### 模式 B: 角色化对话 (Conversable Agents)
- 每个 Agent 一个角色 (UserProxy / Assistant / Critic / Executor)
- Agents 互相对话, 形成 group chat
- 代表: Microsoft AutoGen (GroupChat 模式)

##### 模式 C: Sequential Pipeline
- Agent A 输出给 Agent B, B 输出给 C, ...
- 形成流水线, 但每个 Agent 内部仍可能是 Agent (嵌套)
- 代表: CrewAI (sequential process)

##### 模式 D: Hierarchical (分层)
- 顶层 Manager Agent 拆任务给中层 Lead Agent
- 中层 Lead Agent 再拆给底层 Worker Agent
- 代表: Anthropic Multi-Agent 架构 (内部使用)

#### 2.4.3 完整执行流程 — Microsoft Copilot Workspace 案例

- 用户 query: "Fix the bug in issue #42"
- Architect Agent: 分析 issue + 设计修复方案 → 输出 plan
- Coder Agent: 按 plan 写代码 + 跑测试 → 输出 PR
- Reviewer Agent: 审查 PR (代码质量 / 测试覆盖 / 风格) → 输出 approve / changes_requested
- 如果 changes_requested: 回 Coder Agent 改, 再回 Reviewer
- 通过后: 自动提 PR + 等用户最终确认

#### 2.4.4 优势
- 任务自然分解 (按角色)
- 并行执行 (多 Worker 同时跑)
- 专业化角色 (Researcher 只做研究, Writer 只做写)
- 互相 review 提质量

#### 2.4.5 劣势
- **协作开销大** (5 Agent × 5 轮 = 25 次 LLM 调用)
- **调试复杂** (多个 Agent 互相影响, 出错难定位)
- 容易过度工程 (单 Agent 能解决的硬上 multi-agent)
- 成本高 (每 Agent 都需要 LLM)

#### 2.4.6 真实采用
- **Microsoft Magentic-One** (5 Agent: Orchestrator + WebSurfer + FileSurfer + Coder + Terminal)
- **Microsoft Copilot Workspace** (Architect + Coder + Reviewer)
- **AutoGen** (内置 GroupChat 多 Agent 协作)
- **CrewAI** (角色化 Agent + sequential)
- **CAMEL Society** (学术 Multi-Agent 框架)

#### 2.4.7 反模式
- ❌ 单 Agent 任务硬上 multi-agent (overkill, 成本翻 5-10×)
- ❌ Agent 数量 > 10 (协作开销爆炸)
- ❌ Agent 之间没明确的角色边界 (互相覆盖)
- ❌ 不限循环 (Agent 互相 ping pong 死循环)

#### 2.4.8 跟单 Agent 决策
- 单一明确任务 → 单 Agent (用 ReAct 或 Plan-and-Execute)
- 多角色 + 角色明确 + 输出质量高 → Multi-Agent
- 探索性研究 → 单 Agent ReAct (Multi-Agent 协调成本太高)

### 2.5 形态 4: Self-Reflection (Self-RAG / Reflexion)

#### 2.5.1 解决什么问题
- LLM 第一次输出可能不完整 / 不准确, 需要"自检 + 改进"
- Self-Reflection 解法: **LLM 输出后, 让另一个 LLM (或同一个 LLM 不同 prompt) 评估, 不满意则重做**

#### 2.5.2 跟 Workflow Pattern 5 Evaluator-Optimizer 区别 (易混淆)

| 维度 | Evaluator-Optimizer (Workflow) | Self-Reflection (Agent) |
|---|---|---|
| 循环数 | **预先写死** (e.g. 3 轮) | **LLM 决定何时停** (信息够 / 质量到) |
| 路径 | 工程师定 | LLM 决定 |
| 适合 | 翻译 / 写作 / 代码 | 复杂推理 / 检索 / 反思 |
| 工具 | LangChain Chain | Self-RAG / Reflexion / LangGraph |

#### 2.5.3 算法 (核心循环)
- 步 1 — LLM 接 query, 生成初版输出
- 步 2 — Reflector LLM (或 Critic Agent) 评估输出:
  - 是否回答了问题?
  - 信息是否完整?
  - 是否准确?
  - 输出: SCORE (1-5) + 改进建议
- 步 3 — 如果 SCORE < 阈值: LLM 接收建议 + 改进重做, 回步 2
- 步 4 — 否则输出最终答案
- 终止条件: SCORE ≥ 阈值 OR max_iterations OR 改进不显著

#### 2.5.4 主流实现

##### Self-RAG (Asai et al. 2023, arXiv:2310.11511)
- 把"何时检索 / 检索结果是否相关 / 答案是否被支撑 / 答案是否有用" 4 个判断, 训练成 LLM 自己输出的 reflection token
- LLM 在生成过程中插入这些 token, 实现自我反思
- 优势: 端到端 (不需要外部 Critic)
- 劣势: 需要 fine-tune LLM (普通用户用不了)

##### Reflexion (Shinn et al. 2023, arXiv:2303.11366)
- LLM 完成任务后, 写"反思日志" (我哪里错了 / 下次应该怎么做)
- 反思存到 episodic memory, 下次类似任务时复用
- 优势: 越用越强 (vs 传统 Agent 每次从 0 开始)
- 落地: 学术阶段, 工业生产应用少

#### 2.5.5 真实采用
- **Self-RAG** (论文标杆, 学术为主)
- **Reflexion** (论文标杆 + 部分研究 Agent)
- **LangGraph Reflection 模板**
- **CrewAI** 内置 review 角色

#### 2.5.6 反模式
- ❌ 反思循环不限次数 — 可能反复反思不收敛, 必须 max_iterations
- ❌ Critic 跟 Generator 同一 LLM 同 prompt — 容易自圆其说, 改进有限
- ❌ Critic 阈值过高 — 永远不满意, 死循环

### 2.6 形态 5: Iterative RAG (CRAG / Iterative Retrieval)

#### 2.6.1 解决什么问题
- 复杂多跳问题单次检索答不全 (e.g. "Klarna 退款流程跟 PayPal 比有什么差异?")
- 单次检索 Recall 低时, 需要"改 query 重检"
- Iterative 解法: **检索 → 评估完整性 → 不够则改 query 重检 → 直到信息充分**

#### 2.6.2 跟 Self-Reflection 区别
- Self-Reflection: 评估**输出质量**, 不满意重做
- Iterative: 评估**检索完整性**, 不全则补检

#### 2.6.3 算法 (核心循环)
- 步 1 — 接 query, 第一次检索
- 步 2 — Evaluator 评估:
  - 检索结果是否覆盖 query 全部信息需求?
  - 是否有遗漏关键 entity / aspect?
- 步 3 — 如果不全:
  - LLM 改写 query (e.g. 拆子问题, 加同义词, Step-Back 抽象)
  - 重检索
- 步 4 — 循环到信息充分
- 步 5 — 综合所有检索结果生成答案

#### 2.6.4 CRAG (Corrective-RAG, Yan et al. 2024.01) 完整流程

- 步 1: 检索 → top-K chunks
- 步 2: Evaluator 给每 chunk 打 confidence (高 / 中 / 低)
- 步 3: 三档分类:
  - 全部高: 直接生成答案 (跳过其它步)
  - 有低: 把低 chunks 标 "ambiguous", 触发 web_search 兜底
  - 全部低: 完全 web_search, 不用 KB
- 步 4: 用最终 chunks (KB + web) 生成答案

#### 2.6.5 真实采用
- **CRAG (Yan et al. 2024)** — 工业落地最广
- **Self-Ask** (Press et al. 2022) — 自问自答多跳
- **IRCoT** (Trivedi et al. 2022) — Iterative + Chain of Thought
- **LangGraph Iterative RAG 模板**

#### 2.6.6 反模式
- ❌ 不限循环 — 可能反复检索不收敛
- ❌ 阈值不调优 — 默认值不一定适合业务
- ❌ Evaluator 用便宜 LLM — 评估错就循环错

### 2.7 5 形态选型决策树

#### 2.7.1 决策树 (从问题特征到形态)

- Q1: 任务是 Coding / 探索类?
  - YES → ReAct (Cursor / Devin 验证)
  - NO → 进 Q2
- Q2: 任务可预先分解 N 步?
  - YES → Plan-and-Execute (Klarna 客服验证)
  - NO → 进 Q3
- Q3: 任务需要多角色协作 (Researcher + Writer + Critic)?
  - YES → Multi-Agent (Magentic-One / CrewAI)
  - NO → 进 Q4
- Q4: 任务输出质量需求高 + 自带评估标准?
  - YES → Self-Reflection (Self-RAG / Reflexion)
  - NO → 进 Q5
- Q5: 任务是检索类, 单次检索可能不全?
  - YES → Iterative RAG (CRAG)
  - NO → 回 Q1 重新审视

#### 2.7.2 真实流程对照表

| 形态 | 真实采用 | 单 query 成本 | 单 query 延迟 |
|---|---|---|---|
| Plan-and-Execute | Klarna / Anthropic Computer Use / Copilot Workspace | $0.10-0.20 | 5-10s |
| ReAct | Cursor / Devin / Claude Code | $0.50-5.00 (5-50 步) | 10-300s |
| Multi-Agent | Magentic-One / Copilot Workspace | $0.30-2.00 | 15-60s |
| Self-Reflection | Self-RAG (学术) | $0.05-0.30 | 5-20s |
| Iterative RAG | CRAG (Yan 2024) | $0.05-0.20 | 5-15s |

---

## 三. 7 层架构 + 决策循环 (Agentic RAG 解剖图)

### 3.0 Agentic 7 层架构思维导图 ⭐

> 进入本章前先看这张思维导图建立全章认知.

### 3.1 7 层架构总览 — 为什么是这 7 层

#### 3.1.1 一句话
- Agentic RAG 7 层 = 把 Agent 内部"从 query 到 answer"的全过程切成 7 个职责清晰的层
- 每层独立可替换, 上下层接口契约明确
- 加上 1 个横切 (Cost Controller) 监控全程, 总共 7+1 架构

#### 3.1.2 架构演进 (为什么从 5 层到 7 层)
- 早期 Modular RAG (§19 通用版) 是 7 模块 (Indexing/Pre-Retrieval/Retrieval/Post-Retrieval/Generation/Routing/Orchestration), 偏检索流水线
- Anthropic Building Effective Agents (2024.12) 出后, 业界把 Agent 抽象成"决策循环 + 工具池 + 记忆 + 校验"的 7 层
- 7 层是 5 层企业架构 (L1 数据治理 / L2 索引 / L3 检索 / L4 Router / L5 Agent) 中 **L5 Agent 的 zoom-in 视图**
- 即: 通用 RAG L5 Agent 内部展开就是这 7 层

#### 3.1.3 7 层职责一句话表 (速记)

| Layer | 一句话 | 类比人脑 |
|---|---|---|
| L1 Query Understanding | 看懂用户问什么 | 视听理解 |
| L2 Router | 决定走简单路径还是 Agent 路径 | 决策中枢 |
| L3 Planner | 拆解任务成 N 步 | 前额叶规划 |
| L4 Tool Loop | 调工具 → 看结果 → 决定下一步循环 | 双手 + 反馈环 |
| L5 Memory | 跨步 / 跨会话 / 跨用户保存状态 | 海马体 + 长期记忆 |
| L6 Synthesizer | 综合所有 tool 结果生成最终答案 | 综合表达 |
| L7 Validator | 校验答案是否准确可信 | 自我审查 |
| ⊥ Cost Controller | 全程监控 token / cost / 步数 | 体能管理 |

### 3.2 7 层职责完整详解 (每层按"是什么 / 为什么需要 / 怎么实现 / 反模式 / 真实采用")

#### 3.2.1 Layer 1 — Query Understanding (入口理解层)

##### 是什么
- Agent 接到用户 query 后的第一步, 解析 query 的"意图 + 复杂度 + 用户上下文"
- 输出供 Layer 2 Router 决策用的标签

##### 为什么需要
- 不做意图理解, Router 没法精准切流, 90% 简单 query 可能误进 Agent 烧钱
- 用户 query 形态多样 (自然语言 / 含编号 / 含模糊词), 必须先标准化
- Memory L2 用户偏好需要在这一层注入 (e.g. 用户偏好简短答案 vs 详细)

##### 输入输出
- 输入: 用户原始 query (str) + user_id + session_id
- 输出: 含意图 (intent) + 复杂度评分 (1-5) + 关键 entity + 用户偏好 的 enriched query

##### 关键技术 + 工程实现
- **意图分类** (3 种实现):
  - 起步: 关键词正则 (含 "退款" → refund / 含 "RF\\d+" → order_lookup)
  - 中级: kNN on intent embeddings (准备 100-500 标注样本, query embed → cosine 找最近邻意图)
  - 高级: Haiku 4.5 LLM-as-judge (single API call 给意图标签, 准确率 95%+)
- **复杂度评分** (5 档):
  - 1: 单点查询 (直接 RAG)
  - 2-3: 多步骤但可固定 (Workflow)
  - 4-5: 跨多源 / 探索性 (Agent)
- **Entity 抽取**:
  - 用 spaCy NER 抽人名 / 地名 / 时间 / 编号
  - 高精度场景用 LLM 抽

##### 关键参数
- 意图分类阈值: top-1 confidence ≥ 0.85 走对应分支, 否则进 LLM 兜底
- 复杂度阈值: ≥ 3 进 Agent, < 3 进 Workflow / Augmented LLM
- Memory L2 注入: 单次取最近 30 天用户偏好

##### 反模式
- ❌ 跳过这层直接进 Router — Router 只看 query 字符串, 没意图标签效果差
- ❌ 用 Sonnet / GPT-5 做意图分类 — overkill, Haiku 已够
- ❌ 不抽 entity — 后续 Tool Calling 没参数

##### 真实采用
- Klarna 客服: 起步规则 + 升级 Haiku, 意图分类准确率 96%
- Glean 内部 KB: BERT-class classifier + 用户角色映射 (PM/Eng/Sales 看不同源)

#### 3.2.2 Layer 2 — Router (路由决策层)

##### 是什么
- 接收 Layer 1 的 enriched query, 决定 query 走哪条路径
- 是 Agentic RAG 跟 Augmented LLM 的"分流闸门"
- 决定 80%-95% query 走简单路径 (省钱省时), 5%-20% 走 Agent (高质量)

##### 为什么需要
- 全 Agent 是反模式 (成本 50× / 延迟 10×, Klarna 早期栽过)
- 全 Augmented LLM 是另一反模式 (跨源诊断答不出, 用户失望转人工)
- Router 让 5%-20% 高价值 query 走 Agent, 95% 简单 query 走便宜路径, ROI 最优

##### 输入输出
- 输入: enriched query + 意图 + 复杂度
- 输出: 路径标签 (one of: simple_rag / enhanced_rag / agent / sql / clarification / refusal)

##### 关键技术 — 三层混合路由 (业界标配)

| 层 | 技术 | 占比 | 延迟 | 成本 |
|---|---|---|---|---|
| 第 1 层规则 | 关键词正则 / 长度 / 数字编号 | 70% | < 1ms | 0 |
| 第 2 层语义 | kNN on intent embeddings (cosine) | 20% | ~10ms | 0 (本地) |
| 第 3 层 LLM 兜底 | Haiku 4.5 LLM-as-judge | 10% | 200-500ms | $0.001 |

##### 关键参数
- Router 准确率必须 ≥ 0.95 才上线 (低于这个误分流频繁, 用户骂街)
- 切流比例 (工业典型): simple 80-95% / agent 5-20% / 其它 < 5%
- 失败 fallback: 任一层匹配失败 → fallback simple_rag (硬错就算了)

##### 反模式
- ❌ 全 LLM 路由 — 单 query 多 0.5-1s, 高 QPS 时 LLM 排队拖死全链
- ❌ 全 Agent 一刀切 — 简单 FAQ 也走 Agent, 平均 $0.4/query, 是普通 50× 成本
- ❌ 不监控 path_distribution — Router 退化没人发现 (上线 1 个月后 50% fallback agent, 月底账单出来才知道)

##### 真实采用
- Klarna: 80/15/5 分流 (公开年报数字)
- Cursor: 几乎全 agent (Coding 任务无法预先分解, Router 几乎不分流)
- Snowflake Cortex: text2sql vs simple_rag 二分

#### 3.2.3 Layer 3 — Planner (Agent 大脑层)

##### 是什么
- Agent 形态的核心 — 接到 query 后用强 LLM 生成"下一步要做什么"的计划
- 是普通 RAG (单次调用) 跟 Agent (多步循环) 的本质区别
- Planner 错了, 后面所有步都白费 (反模式 1)

##### 为什么需要
- 简单 query 不需要 Planner (单次检索 + 综合就够)
- 复杂 query (跨多源诊断) 必须先规划 N 步, 不规划乱抓乱跳
- Planner 是 Agent 的"智商上限" — 用 Haiku 做 Planner 等于 Agent 智商塌

##### 输入输出
- 输入: query + Memory (L1+L2+L3) + 工具描述列表 + 历史 plan (如果是 mid-correction)
- 输出: 结构化 Plan (JSON 数组), 每个 step 含 step_id / tool_name / params / expected_output / fallback

##### 关键技术 — 两种 Planning 模式 (详见 §2)
- **Plan-and-Execute** (开局全规划): 1 次 Sonnet 出完整 N 步, N 次 Haiku 执行, 省钱适合可预测任务
- **ReAct** (每步规划下一步): N 次 Sonnet 思考 + 行动, 灵活适合不可预测任务 (Coding)

##### 关键参数
- max_plan_depth: 5-8 步 (超过 LLM 规划质量塌, 步骤之间逻辑断裂)
- Planner LLM 选型: Sonnet 4.5 / GPT-5 / o3 / DeepSeek-R1 (frontier 必需)
- Plan JSON Schema 必须严格 (否则 Executor 解析失败)
- Replanner 触发条件: 实际执行偏离 plan > 30% 时触发 mid-correction

##### Plan JSON 示例 (退款诊断)
- step_id: 1, tool_name: order_api, params: {user_id: "U123", days: 30}, expected: "订单列表"
- step_id: 2, tool_name: refund_api, params: {order_ids: "$step1.output"}, expected: "退款状态"
- step_id: 3, tool_name: payment_gateway_api, params: {refund_id: "$step2.output"}, expected: "支付状态"
- step_id: 4, tool_name: synthesize, params: {context: "$step1+2+3"}, expected: "诊断答案"

##### 反模式
- ❌ Planner 用 Haiku / GPT-3.5 — 规划质量塌, 步骤之间逻辑断裂
- ❌ max_plan_depth > 10 — LLM 规划"幻觉", 后期 step 跟前期脱节
- ❌ Plan 没 fallback 字段 — 单步失败全盘崩
- ❌ 不允许 mid-correction — Plan 偏离 reality 时只能重来

##### 真实采用
- Klarna 客服: Sonnet 4.5 Planner + Haiku 4.5 Executor (Plan-and-Execute)
- Cursor: Sonnet 4.5 + extended thinking (ReAct, 每步规划)
- Devin: 不公开但 Sonnet 级 + ACE 框架

#### 3.2.4 Layer 4 — Tool Execution Loop (双手循环层)

##### 是什么
- Agent 的核心循环 — "调工具 → 看结果 → 决定下一步" 反复执行
- 是 Agent 跟 Workflow 的本质区别 (Workflow 路径预先固定, Agent 这层动态决定下一步)
- 包含 4 个核心组件: Tool Registry / Tool Executor / Loop Controller / State Manager

##### 为什么需要
- LLM 自己不能"做"事 (查 DB / 调 API / 改文件), 必须通过工具
- 单次工具调用不够, 需要循环 (e.g. 第一次查订单, 第二次查支付, ...)
- 必须有终止条件防死循环 (4 终止条件)

##### 关键组件

###### Tool Registry (工具池)
- 存所有可用工具的注册表
- 每工具含: name + description + JSON Schema (参数定义) + handler 函数
- 工具池规模 5-12 个 (太少 LLM 没选择, 太多选错率塌 30-50%)
- 工具描述工程: 必须写清"什么场景用 / 输入输出 / 失败行为"

###### Tool Executor (执行器)
- 解析 LLM 输出的 tool_call (含 name + arguments)
- 调真实 API / 函数 (e.g. HTTP request / DB query / file read)
- 序列化结果回传给 LLM (作为下一轮的 observation)
- 处理超时 / 失败 / rate limit

###### Loop Controller (循环控制器)
- 4 终止条件 (任一即退出):
  - LLM 主动声明 "已收集到足够信息, 进入综合"
  - max_steps 触发 (默认 8, 客服场景)
  - 同一工具连续重复 3 次 (LLM 卡住信号)
  - 累计 cost / token 超预算 (单 query $1)

###### State Manager (状态管理)
- 维护循环中的 state: query / history / tool_results / cost / step_count
- 每步都更新 state, 写到 Memory L1 Session
- 终止后 state 摘要进 Memory L2 (用户偏好) 和 L3 (业务记忆)

##### 关键参数 (按场景)

| 场景 | max_steps | max_cost | timeout/step |
|---|---|---|---|
| 客服 / FAQ | 8 | $1 | 30s |
| 通用 RAG | 12 | $2 | 30s |
| Coding (Cursor/Devin) | 50+ | $5-10 | 60s |
| 科研 / 长任务 | 100+ | $50 | 300s |

##### 反模式
- ❌ 工具池 > 20 个 — LLM 选错率塌 30-50%, 必须分层路由 (上层选大类, 下层选具体工具)
- ❌ 工具描述模糊 ("search the database") — LLM 选不准
- ❌ 不限 max_steps — 死循环烧钱事故 (1 query 烧 $200)
- ❌ 没有 same_tool_repeat 检测 — LLM 反复调同一工具
- ❌ Tool Executor 不处理 timeout — 单工具 hang 拖死全链

##### 真实采用
- Klarna: 5 个核心工具 (订单 / 退款 / 支付 / 物流 / 客户)
- Cursor: read_file / grep / edit_file / run_test (Coding 4 大工具)
- Anthropic Computer Use: screenshot / click / type / scroll (GUI 4 工具)
- 详见 §5 Tool Calling 深度

#### 3.2.5 Layer 5 — Memory (脊髓三层记忆层)

##### 是什么
- LLM 是 stateless 的 (每次 API 调用都是独立的), Memory 提供跨步 / 跨会话 / 跨用户的状态
- 三层架构: L1 Session (短期) / L2 User Pref (长期) / L3 Business (跨用户)
- 是 Agent 多步循环的"必需基础", 没 Memory Agent 第 5 步看不到第 1 步结果

##### 为什么需要
- Agent 多步: 第 5 步 LLM 调用必须能看到前 4 步的 tool_results
- 跨会话: 用户上次问过的偏好, 这次还应记得
- 跨用户: 业务知识 / 团队约定应跨用户共享 (但 ACL 隔离)

##### 三层完整对比

| 层 | 用途 | DB | TTL | 单 user 容量 | 跨用户? |
|---|---|---|---|---|---|
| L1 Session | 本次对话 + tool_results | Redis | 6h | 5KB | ❌ |
| L2 User Pref | 跨会话用户偏好 | Postgres JSONB | 永久 (90 天 archive) | 10-50KB | ❌ |
| L3 Business | 跨用户业务知识 | Vector DB | 永久 + 版本 | 全公司 GB 级 | ✅ (按 ACL) |

##### 关键技术细节
- **L1 Redis schema**: messages: List[{role, content, ts}] + tool_calls: List[{name, args, result}] + cumulative_cost
- **L2 Postgres JSONB schema**: user_id PK + preferences JSONB + interaction_history JSONB + last_active
- **L3 Vector DB schema**: memory_id PK + content + embedding[1024] + acl_tags + sensitivity + canonical_id + version
- **容量约束**: Memory 总占 context window ≤ 6K (16K budget 内), 否则挤掉检索结果空间
- **摘要策略**: 超容量时调 LLM 把旧 message 摘成 200 字 Conversation Summary

##### 反模式
- ❌ L1 永久不清理 — 100 query 后 Memory 占满拖慢链路
- ❌ L2 用 Redis (全永久) — 内存爆 + JSONB 复杂查询慢
- ❌ L3 不带 user_id / tenant_id 隔离 — 跨用户 PII 泄露 (§13.27 MS Recall 类事故)
- ❌ 不做 Conversation Summary — 多步 Agent context 爆 16K
- ❌ 用 LLM context 当跨用户共享 cache (Anthropic Prompt Caching 必须按 user 分 prefix)

##### 真实采用
- 详见 §6 Memory 深度 (待写)
- Anthropic Claude Memory tool (2025) / OpenAI Memory in ChatGPT 都是工业实现

#### 3.2.6 Layer 6 — Synthesizer (综合答案层)

##### 是什么
- 接收 Layer 4 循环结束后的所有 tool_results, 综合成最终答案给用户
- 是 Agent 的"出口", 把多步的中间结果整合成自然语言答案 + 引用

##### 为什么需要
- Tool 返回的是结构化数据 (JSON / 表格 / 文件), 用户看不懂
- 多个 tool_results 可能矛盾 / 冗余, 需要综合判断
- 必须生成"含引用"的答案 (用户能追溯每条事实的来源)

##### 输入输出
- 输入: 所有 tool_results (List[Dict]) + query + Memory (相关 snippets)
- 输出: 最终答案 (str, 含 [chunk_X] / [tool_call_X] 引用编号)

##### 关键技术
- **LLM 选型**: Haiku 4.5 / GPT-5-mini / Gemini 2.0 Flash (便宜模型够用)
- **Prompt 模板**: system 写"基于以下 tool_results 综合答案 + 必须引用"
- **引用机制**: tool_results 里每条加 `[tool_call_X]` 编号, LLM 输出时引用, 后处理反查 source

##### 关键参数
- Synthesizer LLM: 永远用便宜的 (综合不吃推理力)
- max_output_tokens: 2048 (普通客服) / 4096 (法律 / 长答案)
- temperature: 0.3 (综合要稳, 不要发散)

##### 反模式
- ❌ 用 Sonnet 4.5 做 Synthesizer — 单 query 成本翻 5-10×, 综合任务不吃推理力
- ❌ 不引用 — 用户没法验证, RAG 失去"可溯源"核心价值
- ❌ Synthesizer prompt 模糊 — LLM 跳过 tool_results 凭训练知识答 (幻觉)
- ❌ 综合时丢失关键数字 — 必须 prompt 强调"涉及金额/时间必须从 tool_results 精确抄"

##### 真实采用
- Klarna: Haiku 4.5 综合, 单次 $0.005, 1.2s
- Cursor: Sonnet 4.5 综合代码 (代码场景需要推理力, 例外)
- Glean: Haiku 4.5 综合检索结果

#### 3.2.7 Layer 7 — Validator (质量校验闸门)

##### 是什么
- Agent 出口的最后一道闸门, 校验 Synthesizer 输出的答案是否准确 / 可信 / 安全
- 通过则放行给用户, 不通过则拒答 / 重试 / 转人工
- 是 Agent 防"答错被截图传播"的命脉

##### 为什么需要
- LLM 可能编造 (幻觉), 答案看似合理但实际错
- LLM 可能输出敏感信息 (PII / 内部数据)
- LLM 可能被 prompt injection 欺骗 (输出违规内容)
- 没 Validator 上线 = Air Canada 类法律事故 (LLM 编造退款政策, 法院判公司赔)

##### 4 种检查 (并行执行)

###### 检查 1: Faithfulness (忠实度)
- 答案中每个事实声明是否被 tool_results 支撑?
- 实现: RAGAS faithfulness 公式 (LLM-as-judge 拆答案成 claims, 每 claim 跟 tool_results 对照)
- 阈值: ≥ 0.85 通过, < 0.85 拒答
- 单次成本: $0.002 (Haiku 调用)

###### 检查 2: Citation (引用校验)
- 答案中的 [chunk_X] / [tool_call_X] 引用是否真实存在?
- 实现: 正则提取引用编号, 对照 Layer 4 的 tool_results / Layer 6 的 chunks
- 阈值: 100% 引用必须真实

###### 检查 3: PII (敏感信息过滤)
- 答案是否含 SSN / 信用卡 / 个人邮箱 / 电话?
- 实现: Microsoft Presidio + 中文 NER fine-tune
- 处理: 检测到 PII 直接 mask 或拒答

###### 检查 4: Guardrail (内容安全)
- 答案是否违反公司政策 (色情 / 暴力 / 歧视 / 商业敏感)?
- 实现: LlamaGuard / Anthropic Constitutional AI / OpenAI Moderation
- 处理: 触发即拒答 + 写 audit log

##### 决策路径

| Validator 结果 | 处理 |
|---|---|
| 4 检查全过 | 放行答案给用户 |
| Faithfulness < 0.85 | 回 Layer 3 改 Plan 重试 (max 2 次) |
| Citation 失效 | 回 Layer 6 重新综合 |
| PII 检出 | 拒答 + 写 audit log |
| Guardrail 触发 | 拒答 + 告警 + 写 audit log |

##### 反模式
- ❌ 跳过 Validator 直接返回 — Klarna 早期就栽过, 答错被截图传播
- ❌ Faithfulness 阈值过低 (0.5) — 漏掉幻觉
- ❌ Faithfulness 阈值过高 (0.95) — 拒答率塌 30%, 用户体验崩
- ❌ 不写 audit log — 出事故没法追溯

##### 真实采用
- Klarna: 4 检查全上, Faithfulness 0.85
- Anthropic Computer Use: 加额外 "危险操作必须用户确认" Guardrail
- Glean: PII + Guardrail 严格 (内部 KB 含敏感数据)

#### 3.2.8 横切 — Cost Controller (FinOps 监控全程)

##### 是什么
- 跨所有 7 层的横切组件, 监控 Agent 全程的 token / cost / latency / step_count
- 超预算硬熔断 (任何一层都可触发)
- 是 Agent 项目防"边缘 query 烧 $200"的命脉

##### 为什么需要
- Agent 单 query 成本变化大 (简单 $0.05, 复杂 $1+)
- 没 Cost Controller, 边缘 query 单次烧 $200 (真实事故 1h 烧 $5000)
- 月底账单出来才发现成本翻 10× = 项目被砍

##### 监控指标 (实时)

| 指标 | 含义 | 告警阈值 |
|---|---|---|
| token_used | 累计 LLM token | per query > 50K |
| cost_so_far | 累计美元成本 | > $0.5 review |
| latency | 已用时间 | > 30s 告警 |
| step_count | 已执行步数 | 超 max_steps × 0.8 警告 |
| tool_call_count | 工具调用次数 | per query > 20 |

##### 硬熔断 4 阈值 (任一触发立即退出)

| 阈值 | 客服 | 通用 | Coding | 科研 |
|---|---|---|---|---|
| max_cost_per_query | $1 | $2 | $5 | $50 |
| max_steps | 8 | 12 | 50+ | 100+ |
| max_same_tool_repeat | 3 | 3 | 5 | 5 |
| timeout_per_step | 30s | 30s | 60s | 300s |

##### 反模式
- ❌ 不设熔断 — 真实事故 (1h 烧 $5000)
- ❌ 阈值统一所有场景 (客服跟 Coding 一刀切) — 客服设 50 步浪费, Coding 设 8 步不够
- ❌ 不告警 — 单 query > $0.5 是异常信号, 必须 review

##### 真实采用
- 详见 §10 死循环防御 (待写)
- 业界标配: LangSmith / Phoenix / Langfuse 全链路监控 + Datadog 告警

### 3.3 决策循环 (5 部件如何串起来) — 完整数据流

#### 3.3.1 完整在线流程 (一次 query 全过程)

- 输入: 用户 query (str)
- → **Layer 1 入口**: 解析意图 + 复杂度 + 取 Memory L2 用户偏好
- → **Layer 2 Router** 判路径:
  - **简单路径 (80-95% 流量)**: → Modular RAG 单次调用 → Layer 7 Validator → 答案
  - **Agent 路径 (5-20% 流量)**: ↓
    - → **步 A — Layer 3 Planner**: 调 frontier LLM 生成 N 步执行 Plan
    - → **步 B — 进入 Layer 4 Loop** (max_steps 内反复):
      - B.1 — 从 Layer 5 Memory 取 state (query + history + tool_results)
      - B.2 — LLM 看 state + 工具描述, 决定下一步调哪个 Tool 和参数
      - B.3 — Tool Executor 执行 (Modular RAG / SQL / Web Search / Function Call)
      - B.4 — 工具结果写回 Layer 5 Memory + Cost Controller 累计 cost
      - B.5 — 判 4 终止条件 (LLM 主动 / max_steps / 重复 / 超预算), 任一满足则退出 Loop
    - → **步 C — Layer 6 Synthesizer**: 综合所有 tool_results, 生成最终答案 + 引用
    - → **步 D — Layer 7 Validator** 校验 (4 检查并行):
      - 通过 → 放行给用户
      - 不通过 → 拒答 / 回步 A 改 Plan 重试
    - → **步 E — 返回**: 答案 + 完整 trace (LangSmith / Phoenix / Langfuse 调试用)
- 全程 **Cost Controller** 监控 token/cost/latency, 超预算硬熔断退出

#### 3.3.2 7 层之间的接口契约

| 接口 | 输入 | 输出 |
|---|---|---|
| User → L1 | raw query (str) | enriched query (含 intent + complexity + user_pref) |
| L1 → L2 | enriched query | path_label (simple/agent/sql/clarification) |
| L2 → L3 (Agent 路径) | enriched query + path_label | Plan (JSON) |
| L3 → L4 | Plan + Memory | state (含 step_id + tool_call) |
| L4 ↔ Tool Registry | tool_call | tool_result |
| L4 ↔ L5 Memory | state read/write | updated state |
| L4 → L6 (终止后) | 所有 tool_results | candidate answer |
| L6 → L7 | candidate answer + tool_results | validated answer / 重试信号 |
| L7 → User | final answer + citations + trace | (展示给用户) |

#### 3.3.3 异常路径 (出错时怎么办)

| 场景 | 处理 |
|---|---|
| L1 意图分类失败 | fallback simple_rag (硬错就算了) |
| L2 Router 不确定 | fallback simple_rag |
| L3 Planner 输出 JSON 解析失败 | 重试 1 次 → 仍失败拒答 |
| L4 工具调用 timeout | 单工具熔断, Loop 继续下一步 |
| L4 同工具重复 3 次 | 触发死循环熔断, 进 L6 综合现有结果 |
| L5 Memory 读写失败 | 降级 (用 in-memory dict) + 告警 |
| L6 Synthesizer 输出空 | 重试 1 次 → 仍空拒答 |
| L7 Validator 不通过 (Faithfulness < 0.85) | 回 L3 改 Plan, max 2 次重试 → 仍不过拒答 |
| Cost 超预算 | 立即停止 + 用现有 tool_results 综合 + 标 "answer truncated" |

### 3.4 7 层架构 vs 5 层企业架构 (跟通用 RAG 文档关系)

#### 3.4.1 关系映射
- 通用 RAG 5 层 (L1 数据治理 / L2 索引 / L3 检索 / L4 Router / L5 Agent) 是建材
- Agent RAG 7 层 = 5 层架构中 **L5 Agent** 这一层的 zoom-in 视图

#### 3.4.2 关键映射
- Agent RAG **Layer 4 (Tool Loop) 内的 "Modular RAG" 工具** = 5 层架构的 L1+L2+L3 完整管道
- Agent RAG **Layer 2 (Router)** = 5 层架构 **L4 Router** (实际是同一个东西)
- Agent RAG Layer 5 (Memory) 是 5 层架构没有的 (Modular RAG 不需要)
- 即: Agent RAG 在 5 层架构 L5 内部展开成 7 层细化结构

#### 3.4.3 看图顺序
- **想知道企业全栈** → 用 5 层架构图 (通用 RAG §3 / §0.3)
- **想知道 Agent 内部** → 用 Agent RAG 7 层图 (本节)
- **想知道写流程** → 看 5 层架构 L1+L2 (通用 RAG §4 + §5)
- **想知道读流程** → 看 5 层架构 L3+L4+L5 (通用 RAG §6 + §7 + §8)
- **想知道 Agent 决策细节** → 看本章 7 层


## 四. Workflow 5 Pattern 深度 (Anthropic 2024.12 官方框架)

### 4.0 Workflow 5 Pattern 思维导图 ⭐

> 进入本章前先看这张思维导图建立全章认知.

### 4.1 5 Pattern 总览 — 在考虑 Agent 前先看这 5 种

#### 4.1.1 一句话
- Anthropic 2024.12 "Building Effective Agents" 提出的 5 种 Workflow Pattern
- 是层次 2 (Workflow) 的具体实现模式
- **90% 业务用其中一种就解决, 不需要上 Agent (层次 3)**

#### 4.1.2 5 Pattern 速记表

| Pattern | 一句话 | 适合 | 实现复杂度 | 真实采用 |
|---|---|---|---|---|
| **Prompt Chaining** | 任务拆成线性 N 步, 每步 LLM 处理上一步输出 | 任务能拆清晰串行步骤 | 低 (几行代码) | Anthropic 文档 pipeline / LangChain SequentialChain |
| **Routing** | 分类输入后转发到不同的专门处理分支 | 有明显类别区分 | 低 | 任何 RAG L4 Router |
| **Parallelization** | 同时跑多个独立 LLM 任务后聚合 | 子任务独立 / 需要投票 | 中 | Constitutional AI / Multi-Query |
| **Orchestrator-Workers** | 中枢 LLM 动态拆任务给 Worker LLM | 子任务运行时才知数量 | 中 | 写竞品分析 / 多视角生成 |
| **Evaluator-Optimizer** | 一 LLM 生成 + 一 LLM 评估的迭代循环 | 输出质量高 + 有评估标准 | 中-高 | Anthropic 翻译 / CodeT5 |

#### 4.1.3 关键认知
- ✅ 这 5 种都属于 Workflow (Anthropic 三层模型层次 2), **不是 Agent** — 因为路径都是工程师预先写好的
- ✅ 90% 业务能用其中一种解决, 不需要上 Agent
- ✅ 实现都是几十行代码 + 标准 LLM API, **不需要 LangGraph / AutoGen 等重框架**
- ✅ 跟 Anthropic Prompt Caching 配合可省 35-49% 成本

### 4.2 Pattern 1 — Prompt Chaining (链式调用)

#### 4.2.1 解决什么问题
- 复杂任务一次 LLM 输出做不到 (e.g. 文档摘要 + 关键词提取 + 标签分类)
- 单 prompt 塞 3 个任务质量塌, 拆开成 3 个 prompt 串行做更好
- Prompt Chaining 解法: **把任务拆成线性 N 步, 每步独立 prompt, 上一步输出作为下一步输入**

#### 4.2.2 算法步骤
- 步 1 — 把任务拆成 N 个独立 step (e.g. step 1: 翻译, step 2: 审校, step 3: 输出格式化)
- 步 2 — 每 step 写独立 prompt (聚焦单一任务, 易调优)
- 步 3 — 串行执行: output_1 = LLM(prompt_1, input)
- 步 4 — output_2 = LLM(prompt_2, output_1)
- 步 5 — ... 直到 final output
- 可选: 每步加 gate check (规则验证), 失败重试或终止

#### 4.2.3 Python 伪代码示例
- def prompt_chain(input_text):
- &nbsp;&nbsp;# Step 1: 翻译
- &nbsp;&nbsp;translated = llm.complete(prompt="把以下中文翻译为英文:" + input_text)
- &nbsp;&nbsp;# Step 2: 审校
- &nbsp;&nbsp;reviewed = llm.complete(prompt="检查并修正翻译错误:" + translated)
- &nbsp;&nbsp;# Step 3: 输出格式化
- &nbsp;&nbsp;final = llm.complete(prompt="按学术风格 reformat:" + reviewed)
- &nbsp;&nbsp;return final

#### 4.2.4 业务场景

##### ✅ 适合
- **文档处理**: 摘要 → 关键词 → 标签 → 入库 (4 步流水线)
- **翻译质量**: 翻译 → 审校 → 改进 → 输出 (3 步迭代)
- **内容生成**: 大纲 → 段落 → 优化 → 校对 (4 步)
- **数据清洗**: 抽取 → 规范化 → 去重 → 入库

##### ❌ 不适合
- 任务无法清晰拆步骤 (用 Agent ReAct)
- 步骤之间高度耦合 (拆了反而错)
- 单 LLM 调用够 (overkill, 浪费成本)
- 步骤可并行 (用 Pattern 3 Parallelization)

#### 4.2.5 关键设计原则

##### 原则 1: 每步单一职责
- Step 1 只做"翻译", 不要混"翻译+审校"
- 单一职责 prompt 比"all-in-one" 质量高 30-50%

##### 原则 2: Gate Check (中间校验)
- 每步输出后加规则校验 (e.g. step 1 必须输出英文 100+ 字)
- 失败立即终止, 不传错给下一步

##### 原则 3: 模型分级 (省钱)
- 简单步 (格式化) 用 Haiku 4.5
- 复杂步 (推理) 用 Sonnet 4.5
- 别每步都用 Sonnet, 浪费 5-10×

#### 4.2.6 真实采用

##### Anthropic 官方文档处理 pipeline
- 公开博客提到的 pipeline: parse → summarize → tag → store
- 4 步 chain, 总耗时 5-10s, 单文档 $0.005-0.02

##### LangChain SequentialChain
- 经典实现, `from langchain.chains import SequentialChain`
- 内置 chain 组合机制 + 中间变量传递

##### LlamaIndex Pipeline
- `from llama_index.core.workflow import Workflow`
- 类似 LangChain 但偏 RAG 场景

##### 业界小公司案例
- 大量 SaaS 内部用 Prompt Chaining 做 ETL (extract → transform → load)
- 简单可靠, 是上 Agent 之前的标配

#### 4.2.7 反模式 + 真实事故

- ❌ **Step 拆得太碎 (10+ step)**: 每步 LLM 调用累积成本爆 (10 步 × $0.005 = $0.05/query, 比单次 Sonnet 还贵)
- ❌ **Step 之间没有 gate check**: 错误一路传, 最后才发现整个 chain 失败 (排查痛苦)
- ❌ **每步都用强 LLM (Sonnet)**: 简单 step 用 Haiku 够, 不分级浪费 5-10×
- ❌ **把可并行的强行串行**: 失去并行优化机会 (用 Pattern 3)
- ✅ 标配: 3-5 step + 每步 gate check + 模型分级

#### 4.2.8 性能 / 成本

| 场景 | step 数 | 耗时 | 成本 |
|---|---|---|---|
| 简单 chain (翻译+审校) | 2 | 3-5s | $0.005-0.01 |
| 标准 chain (4 步文档处理) | 4 | 5-10s | $0.01-0.03 |
| 复杂 chain (8 步含 LLM judge) | 8 | 15-30s | $0.05-0.10 |

### 4.3 Pattern 2 — Routing (路由分流)

#### 4.3.1 解决什么问题
- 用户 query 有不同类别 (FAQ / 编号 / 复杂诊断), 用同一个 prompt 处理质量塌
- e.g. 用户问 "RF12345 退款进度" 跟 "退款政策是什么" 用完全不同的 prompt 模板
- Routing 解法: **先分类输入, 再转发到不同的专门处理分支**

#### 4.3.2 算法步骤
- 步 1 — Router LLM (或 classifier) 接 query, 输出类别标签
- 步 2 — 根据标签路由到对应处理分支:
  - FAQ → 简单 RAG (单次检索 + Haiku 综合)
  - 编号 → BM25 字面检索 + 直接返结构化数据
  - 复杂诊断 → Agent (多步规划)
- 步 3 — 各分支独立处理, 用专门 prompt + 专门 LLM

#### 4.3.3 Python 伪代码
- def routing_workflow(query):
- &nbsp;&nbsp;label = classifier(query)  # rule / kNN / LLM
- &nbsp;&nbsp;if label == "faq":
- &nbsp;&nbsp;&nbsp;&nbsp;return simple_rag(query)
- &nbsp;&nbsp;elif label == "lookup":
- &nbsp;&nbsp;&nbsp;&nbsp;return bm25_search(query)
- &nbsp;&nbsp;elif label == "complex":
- &nbsp;&nbsp;&nbsp;&nbsp;return agent(query)
- &nbsp;&nbsp;else:
- &nbsp;&nbsp;&nbsp;&nbsp;return clarify(query)

#### 4.3.4 三层混合路由 (业界标配, Klarna 实战)

##### 第 1 层 — 规则路由 (最快, 占 70%)
- 关键词正则 / query 长度 / 包含数字编号
- e.g. 含 "RF\\d{5}" → 必走 BM25 字面检索
- 延迟 < 1ms, 成本 0
- 准确率: 在已知模式上 99%+

##### 第 2 层 — 语义路由 (中等, 占 20%)
- query embedding 跟预先标注的"意图样本" cosine 相似度 (kNN)
- top-1 类别即标签
- 延迟 ~10ms, 成本 0 (本地推理)
- 准确率: 0.85-0.95 (取决于标注质量)

##### 第 3 层 — LLM 兜底 (最贵, 占 10%)
- 前两层都没匹配, 用 Haiku 4.5 LLM-as-judge 给意图标签
- 延迟 200-500ms, 成本 $0.001
- 准确率: 0.95+ (但慢且贵)

#### 4.3.5 业务场景

##### ✅ 适合
- **客服 query 分类** (退款 / 物流 / 账户 / 技术) — 走不同 prompt + 不同 LLM
- **内容分类后选不同 LLM** (复杂 → Sonnet, 简单 → Haiku)
- **多语言路由** (中 → 中文 LLM, 英 → 英文 LLM)
- **任何企业 RAG L4 Modular Router** (这就是 Routing Pattern)

##### ❌ 不适合
- 类别边界模糊 (用户 query 跨多类别)
- 类别少且每类处理同质 (用统一 prompt 即可)
- 高 QPS + LLM 兜底比例高 (LLM 排队拖死全链)

#### 4.3.6 关键参数
- Router 准确率必须 ≥ 0.95 才上线 (低于这个误分流频繁, 用户骂街)
- 切流比例 (工业典型): simple 80-95% / agent 5-20% / 其它 < 5%
- 失败 fallback: 任一层匹配失败 → fallback simple_rag

#### 4.3.7 真实采用
- **Klarna**: 三层混合路由, 80/15/5 分流 (公开年报)
- **Glean 内部 KB**: 多源路由 (Confluence / Slack / Salesforce / Email 各自 retriever)
- **Snowflake Cortex**: text2sql vs simple_rag 二分

#### 4.3.8 反模式

- ❌ **全 LLM 路由**: 单 query 多 0.5-1s, 高 QPS 时 LLM 排队拖死全链
- ❌ **Router 准确率 < 0.95 上线**: 误分流频繁, 复杂 query 走 simple_rag 答错
- ❌ **不监控 path_distribution**: Router 退化没人发现 (上线 1 个月后 50% fallback agent, 月底账单出来才知)
- ❌ **类别太多 (> 10 类)**: classifier 准确率塌, 应合并到 5-7 类

#### 4.3.9 性能 / 成本

| 场景 | 准确率 | 单 query 延迟 | 单 query 成本 |
|---|---|---|---|
| 纯规则 | 99% (在已知模式上) | < 1ms | 0 |
| 规则 + 语义 | 0.92-0.95 | ~10ms | 0 |
| 规则 + 语义 + LLM 兜底 | 0.95-0.98 | 50ms (P95) | $0.0001 |

### 4.4 Pattern 3 — Parallelization (并行化)

#### 4.4.1 解决什么问题
- 子任务相互独立但顺序执行慢, 浪费时间
- e.g. 10 个文档每个独立摘要, 串行 10 × 2s = 20s; 并行只需 2s
- Parallelization 解法: **同时跑多个独立 LLM 任务, 然后聚合结果**

#### 4.4.2 两个子变体

##### 子变体 A: Sectioning (分段并行)
- **算法**: 把大任务切成独立子任务并行
- **聚合**: 拼接 / 求和 / 排序
- **典型场景**:
  - 10 个文档每个独立摘要 → 并行 10 路, 总耗时 = 单路时间 (而非 × 10)
  - 长文档切段并行 embed
  - 多用户批量处理

##### 子变体 B: Voting (投票并行)
- **算法**: 同一任务跑 N 次, 取多数票
- **聚合**: 多数投票 / 加权平均 / LLM judge 综合
- **典型场景**:
  - 安全检测跑 3 次取一致结果 → 提质量 (vs 单次随机)
  - 多模型投票 (3 个 LLM 投票判断输出是否安全)
  - 测试代码 N 次取通过率

#### 4.4.3 Python 伪代码

##### Sectioning
- import asyncio
- async def parallel_summarize(docs):
- &nbsp;&nbsp;tasks = [llm.complete(prompt=f"摘要: {doc}") for doc in docs]
- &nbsp;&nbsp;summaries = await asyncio.gather(*tasks)
- &nbsp;&nbsp;return summaries

##### Voting
- async def voting_check(content):
- &nbsp;&nbsp;tasks = [llm.complete(prompt=f"是否违规: {content}") for _ in range(3)]
- &nbsp;&nbsp;results = await asyncio.gather(*tasks)
- &nbsp;&nbsp;return Counter(results).most_common(1)[0][0]  # 多数票

#### 4.4.4 业务场景

##### ✅ Sectioning 适合
- 长文档处理 (切段并行 embed + 摘要)
- 批量数据处理 (1000 个 query 并行)
- Multi-Query 检索 (4 个 query 变体并行, RAG-Fusion 基础)
- 跨多源同时查 (Confluence / Slack / Salesforce 并行)

##### ✅ Voting 适合
- 内容审核 (3 个 LLM 投票判断违规, 提稳定性)
- 关键决策双校验 (高风险场景多模型投票)
- 测试结果聚合 (跑 5 次取通过率)

##### ❌ 不适合
- 子任务有依赖 (前后顺序) → 用 Pattern 1 Prompt Chaining
- 子任务结果不能聚合 (无 voting 标准)
- 任务量小 (并行收益不明显, 且并行有 overhead)

#### 4.4.5 关键参数
- 并发数: 限制 max concurrent (避免打爆 API rate limit), 通常 10-50
- 超时: 单子任务超时 30s, 超时归为失败
- 重试: 失败子任务重试 1-2 次, 超过则放弃
- Voting 数量: 至少 3 (2 投票 50/50 时无法决定), 推荐 3-5

#### 4.4.6 真实采用

- **Anthropic Constitutional AI**: 3 个 LLM 投票判断输出是否符合 constitution
- **LangChain MapReduceChain**: sectioning 实现 (大文档 map + reduce 聚合)
- **LlamaIndex 异步 batch retrieval**: asyncio.gather 并行检索
- **OpenAI Moderation 多次投票**: 高风险内容 3 次校验

#### 4.4.7 反模式

- ❌ **子任务有依赖硬并行**: 结果不一致, 必须 Pattern 1 串行
- ❌ **Voting 只用 2 个 LLM**: 50/50 时无法决定, 至少 3 个
- ❌ **不限并发数**: 100 路并行打爆 LLM API rate limit
- ❌ **不处理子任务失败**: 1 个失败拖垮整个聚合

#### 4.4.8 性能 / 成本

| 场景 | 串行耗时 | 并行耗时 | 加速比 | 成本 |
|---|---|---|---|---|
| 10 文档摘要 | 20s | 2s | 10× | 同 (10 × $0.005 = $0.05) |
| 安全 3 投票 | 6s | 2s | 3× | 3× ($0.015 vs $0.005) |
| 100 query 批量 | 200s | 10s | 20× | 同 |

### 4.5 Pattern 4 — Orchestrator-Workers (协调员 + 工人)

#### 4.5.1 解决什么问题
- 任务结构复杂, 子任务**数量 / 内容运行时才知道**
- Pattern 3 Parallelization 假设子任务预先固定, 但有些场景 Orchestrator 要在运行时拆
- e.g. 写竞品分析: 不知道用户想分析几家竞品, Orchestrator 看 query 后才决定拆 5 家还是 10 家
- Orchestrator-Workers 解法: **一个中枢 LLM (Orchestrator) 动态拆解任务 → 分配给多个 Worker LLM → 综合结果**

#### 4.5.2 跟 Multi-Agent 的核心区别 (易混淆)

| 维度 | Orchestrator-Workers (Workflow) | Multi-Agent (Agent) |
|---|---|---|
| **Worker 路径** | 工程师写死 (Worker 内部按固定逻辑跑) | LLM 决定 (Worker 自己用 LLM 决策) |
| **鉴别口诀** | Worker 内部不是 Agent | Worker 内部是 Agent |
| **例子** | 写竞品分析 (Orchestrator 拆 5 家, 每家 Worker 跑相同 prompt) | Magentic-One (Worker 各自是独立 Agent, 比如 WebSurfer/FileSurfer/Coder) |
| **复杂度** | 中 (一个 LLM 拆任务即可) | 高 (多个 Agent 互相协作) |

#### 4.5.3 算法步骤
- 步 1 — Orchestrator LLM 接 query, 动态拆任务 (输出 N 个子任务的描述)
- 步 2 — Worker LLM 并行处理每个子任务 (每个 Worker 跑相同 prompt 不同输入, 或不同 prompt)
- 步 3 — Orchestrator 接收 Worker 输出, 综合成最终答案

#### 4.5.4 Python 伪代码
- async def orchestrator_workers(query):
- &nbsp;&nbsp;# Step 1: Orchestrator 动态拆任务
- &nbsp;&nbsp;subtasks = orchestrator_llm.complete(
- &nbsp;&nbsp;&nbsp;&nbsp;prompt=f"拆解以下任务为独立子任务:{query}"
- &nbsp;&nbsp;)  # 输出: ["调研 Klarna", "调研 PayPal", "调研 Stripe"]
- &nbsp;&nbsp;# Step 2: Worker 并行
- &nbsp;&nbsp;tasks = [worker_llm.complete(prompt=f"调研: {st}") for st in subtasks]
- &nbsp;&nbsp;outputs = await asyncio.gather(*tasks)
- &nbsp;&nbsp;# Step 3: Orchestrator 综合
- &nbsp;&nbsp;final = orchestrator_llm.complete(
- &nbsp;&nbsp;&nbsp;&nbsp;prompt=f"综合以下子任务结果:{outputs}"
- &nbsp;&nbsp;)
- &nbsp;&nbsp;return final

#### 4.5.5 业务场景

##### ✅ 适合
- **写竞品分析**: Orchestrator 拆 5 家竞品, 每家 Worker 跑调研
- **多视角生成**: Orchestrator 让 Worker 1 写正面 / Worker 2 写反面 / Worker 3 写中立
- **文档批处理**: Orchestrator 决定每文档用哪个 prompt (依据文档类型)
- **数据分析探索**: Orchestrator 拆数据集成多个分析角度

##### ❌ 不适合
- 简单单任务 (用 Pattern 1 或 Pattern 2 即可)
- 子任务相同且数量预先知道 (用 Pattern 3 Parallelization)
- Worker 之间需要通信协作 (用 Multi-Agent)

#### 4.5.6 关键设计

##### Orchestrator 模型选型
- 必须强 LLM (Sonnet 4.5 / GPT-5 / o3): 拆任务质量决定全流程

##### Worker 模型选型
- 简单任务: Haiku 4.5 (省钱)
- 复杂任务: Sonnet 4.5 (但少用)

##### 子任务数量限制
- 5-10 个最优 (太少 overhead 不值, 太多 Orchestrator 综合质量塌)
- > 15 子任务时 Orchestrator 综合容易遗漏

#### 4.5.7 真实采用

- **Anthropic 内部研究 Agent** (公开博客提到, 用于撰写技术分析报告)
- **LangGraph Plan-and-Execute 模板** (Plan 阶段 = Orchestrator)
- **LlamaIndex Sub-Question Engine** (子问题拆解)
- **写代码 review**: Orchestrator 拆"代码风格 / 安全 / 性能 / 可读性" 4 维, Worker 各查

#### 4.5.8 反模式

- ❌ **Orchestrator 拆 20+ 子任务**: Worker 调用爆 + 综合质量塌
- ❌ **Worker 用 Sonnet**: 简单子任务用 Haiku, 省 5-10×
- ❌ **Worker 之间需要通信**: Pattern 4 假设独立, 有通信用 Multi-Agent
- ❌ **不限 Worker 失败比例**: 50% Worker 失败仍综合, 答案质量塌

#### 4.5.9 性能 / 成本

| 场景 | Orchestrator | Workers | 总耗时 | 总成本 |
|---|---|---|---|---|
| 5 家竞品分析 | 1 × Sonnet ($0.05) | 5 × Sonnet 并行 ($0.25) | 10-15s | $0.30 |
| 10 视角生成 | 1 × Sonnet | 10 × Haiku 并行 | 5-10s | $0.10 |

### 4.6 Pattern 5 — Evaluator-Optimizer (评估 + 优化)

#### 4.6.1 解决什么问题
- 输出质量需要"先生成再评估再改进"循环
- 单次 LLM 输出不够好, 但又不需要 Agent 那么复杂
- e.g. 翻译初稿可能词不达意, 让另一个 LLM 评估 + 改进, 循环 2-3 轮质量飞跃
- Evaluator-Optimizer 解法: **一个 LLM 生成初稿 → 另一个 LLM 评估并反馈 → 形成迭代改进循环**

#### 4.6.2 跟 Self-Reflection (Agent 形态) 的核心区别 (易混淆)

| 维度 | Pattern 5 Evaluator-Optimizer | Self-Reflection (Agent) |
|---|---|---|
| **循环数** | 预先写死 (e.g. 3 轮) | LLM 决定何时停 |
| **路径** | 工程师定 | LLM 决定 |
| **是不是 Agent** | 否 (Workflow) | 是 (Agent) |
| **代表实现** | LangChain refine chain | Self-RAG / Reflexion |

#### 4.6.3 算法步骤
- 步 1 — Generator LLM 接 query, 生成初版输出
- 步 2 — Evaluator LLM 接收输出, 给评分 + 改进建议:
  - 5 维度评分 (准确性 / 完整性 / 流畅度 / 简洁 / 风格)
  - 改进建议 (具体到段落 / 句子)
- 步 3 — Optimizer LLM 接收建议 + 原输出, 生成改进版
- 步 4 — 回步 2 (循环 N 次, 通常 2-3 轮)
- 步 5 — 最后一轮输出作为最终结果

#### 4.6.4 Python 伪代码
- def evaluator_optimizer(query, max_rounds=3):
- &nbsp;&nbsp;output = generator_llm.complete(prompt=f"生成: {query}")
- &nbsp;&nbsp;for round in range(max_rounds):
- &nbsp;&nbsp;&nbsp;&nbsp;eval_result = evaluator_llm.complete(
- &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;prompt=f"评估并给改进建议: {output}"
- &nbsp;&nbsp;&nbsp;&nbsp;)
- &nbsp;&nbsp;&nbsp;&nbsp;if eval_result.score >= 4.5: break  # 已达标
- &nbsp;&nbsp;&nbsp;&nbsp;output = optimizer_llm.complete(
- &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;prompt=f"按建议改进: 原文 {output}, 建议 {eval_result.suggestions}"
- &nbsp;&nbsp;&nbsp;&nbsp;)
- &nbsp;&nbsp;return output

#### 4.6.5 业务场景

##### ✅ 适合
- **翻译质量**: 翻译 → Evaluator 5 维度打分 → Optimizer 重写 → 循环 3 轮 (Anthropic 内部用)
- **代码生成**: 生成 → Evaluator 跑测试 / lint → Optimizer 修 → 循环
- **摘要质量**: 生成 → Evaluator 评信息密度 → Optimizer 改 → 循环
- **学术写作**: 生成段落 → Evaluator 评论证 → Optimizer 改 → 循环

##### ❌ 不适合
- 输出无法评估 (创意写作没标准) — Evaluator 不准
- 任务太简单 (FAQ 客服) — overkill, 单次生成就够
- 实时性要求高 (循环增加 3-5× 延迟)

#### 4.6.6 关键设计

##### Evaluator 必须独立于 Generator
- 同 LLM 同 prompt — 容易自圆其说, 改进有限
- 推荐: Generator 用 Haiku, Evaluator 用 Sonnet (反过来也行)
- 或: 不同 prompt 不同 temperature

##### Evaluator 阈值调优
- 阈值过低 (3.0): 容易满意, 改进效果有限
- 阈值过高 (4.8): 永远不满意, 浪费 LLM 调用
- 标配: 4.0-4.5 (5 分制)

##### 循环数量限制
- max_rounds = 3 (主流): 平衡质量跟成本
- > 5 轮: 边际收益递减, 浪费

#### 4.6.7 真实采用

- **Anthropic 内部翻译 pipeline** (官方 blog 提到, 翻译质量提升显著)
- **CodeT5 / Self-Edit**: 代码生成评估
- **Reflexion 灵感来源** (但 Reflexion 是 Agent 不是 Workflow)
- **LangChain refine chain**: 经典实现

#### 4.6.8 反模式

- ❌ **不限循环次数 (变 Agent Self-Reflection)**: 失去 Workflow 可预测性
- ❌ **Evaluator 跟 Generator 同 LLM 同 prompt**: 容易自圆其说
- ❌ **Evaluator 阈值过高 (4.8)**: 永远不满意, 浪费 LLM 调用
- ❌ **不存中间版本**: 万一最后一轮变差, 没法回滚

#### 4.6.9 性能 / 成本

| 场景 | 轮数 | 总耗时 | 总成本 | 质量提升 |
|---|---|---|---|---|
| 翻译 (3 轮) | 3 | 10-15s | $0.05-0.10 | +30-50% (vs 单次) |
| 代码 (5 轮) | 5 | 20-30s | $0.10-0.20 | +20-40% |

### 4.7 5 Pattern 选型决策树

#### 4.7.1 决策树 (从问题特征到 Pattern)

- Q1: 任务能拆成清晰串行步骤?
  - YES → **Prompt Chaining** (Pattern 1)
  - NO → 进 Q2
- Q2: 输入有明显类别区分?
  - YES → **Routing** (Pattern 2)
  - NO → 进 Q3
- Q3: 子任务相互独立可并行?
  - YES (数量预知) → **Parallelization** (Pattern 3)
  - YES (数量运行时定) → **Orchestrator-Workers** (Pattern 4)
  - NO → 进 Q4
- Q4: 输出质量需要先生成再评估?
  - YES → **Evaluator-Optimizer** (Pattern 5)
  - NO → 重新审视, 大概率回到 Q1 / Q2

#### 4.7.2 5 Pattern 真实采用对照表

| Pattern | 真实采用 | 单 query 成本 | 实现复杂度 |
|---|---|---|---|
| Prompt Chaining | Anthropic 文档 pipeline / LangChain SequentialChain | $0.005-0.05 | 低 (几行代码) |
| Routing | Klarna 三层混合 / Glean 多源路由 | $0.0001 (规则) - $0.005 (LLM) | 低 |
| Parallelization | Constitutional AI (3 投票) / Multi-Query 4 变体 | 同步行 × N (并行加速) | 中 |
| Orchestrator-Workers | Anthropic 研究 Agent / 写竞品分析 | $0.10-0.50 | 中 |
| Evaluator-Optimizer | Anthropic 翻译 / CodeT5 | $0.05-0.20 | 中-高 |

#### 4.7.3 Pattern 组合 (高级用法)

##### 组合 1: Routing → Pattern 1/3/5
- e.g. 客服: Router 分类 → FAQ 走 Pattern 1 chain / 复杂走 Pattern 5 evaluator
- 多数生产系统都是 Routing 包外层 + 内部其它 Pattern

##### 组合 2: Pattern 4 嵌套 Pattern 3
- Orchestrator 拆任务 → Worker 内部用 Parallelization 并行子子任务
- e.g. 5 家竞品分析, 每家 Worker 内部并行抓 (产品 + 价格 + 评论 + 财报)

##### 组合 3: Pattern 1 末尾接 Pattern 5
- Chain 完成后用 Evaluator 验证质量, 不达标重跑 chain

#### 4.7.4 跟 Agent 的边界 — 何时升级到 Agent

- 5 Pattern 全部不够时才上 Agent (层次 3)
- 信号: 任务真正"无法预先固定路径", 需要 LLM 运行时决定
- 典型: Coding (无法预先知道改哪个文件) / 探索性研究 / 跨多源诊断
- 详见 §1.3 三层选型决策树


## 五. Tool Calling 深度 (待写, 阶段 2 完成)

> 阶段 2 内容: 三家 API 完整对比 + MCP 协议 + Computer Use + Browser Use + 工具池设计 + 工具描述工程

## 六. Memory 深度 (待写, 阶段 2 完成)

> 阶段 2 内容: 三层 Memory + Episodic / Semantic / Procedural / Skill Memory + 摘要策略 + 跨用户隔离

## 七. Multi-Agent 系统 (待写, 阶段 2 完成)

> 阶段 2 内容: Orchestrator + Hierarchical + Swarm + CAMEL + Magentic-One + 通信协议

## 八. 高级 RAG-Agent 模式 (待写, 阶段 2 完成)

> 阶段 2 内容: Self-RAG / CRAG / GraphRAG / LightRAG / Adaptive RAG / Reflexion / Tree of Thoughts

## 九. Agent 框架对比 (待写, 阶段 2 完成)

> 阶段 2 内容: 8 主流框架深度对比 + 选型决策

## 十. 死循环防御 + FinOps + 评估 (待写, 阶段 3 完成)

## 十一. 真实落地案例深度 (待写, 阶段 3 完成)

## 十二. 失败模式 + 安全 (待写, 阶段 3 完成)

## 十三. 落地路径 + 最佳实践 (待写, 阶段 3 完成)

## 十四. 未来趋势 (2026-2027) (待写, 阶段 3 完成)

---

## 附录

### 附录 A: 跟通用 RAG 文档的对应关系

| 本文档章节 | 通用 RAG 文档对应 | 关系 |
|---|---|---|
| §0-§4 | §20.1 (核心讲透) | 本文档是 §20.1 的 10× 深度展开 |
| §5 Tool Calling | §20.4 + §8.3 | 本文档加 MCP / Computer Use / Browser Use 深度 |
| §6 Memory | §20.5 + §8.4 | 本文档加 Episodic / Semantic / Procedural |
| §7 Multi-Agent | §20.2.3 | 本文档加 Magentic-One / Hierarchical / Swarm |
| §8 高级 RAG-Agent | §8.5 5 模式 | 本文档加 Reflexion / Tree of Thoughts / Plan-and-Solve |
| §9 框架 | §20.3 + §8.2 | 本文档加 Pydantic AI / Mastra / Smolagents 等新框架 |
| §11 案例 | §13 + §20.6 | 本文档加更深 walkthrough |

### 附录 B: 参考资料

#### B.1 必读官方博客
- **Anthropic — Building Effective Agents** (2024.12) — anthropic.com/engineering/building-effective-agents — 本文档 §1 §4 出处
- **Anthropic — Contextual Retrieval** (2024.09) — RAG 性能提升 49% 的方法
- **Anthropic Claude Agent SDK 文档** (2025) — Tool use + MCP 官方实现
- **OpenAI — Practical Guide to Building Agents** (2025) — OpenAI 官方 Agent 指南

#### B.2 必读论文
- **ReAct** (Yao et al. 2022) — arXiv:2210.03629 — Agent 推理-行动循环始祖
- **Reflexion** (Shinn et al. 2023) — arXiv:2303.11366 — Self-Reflection Agent
- **Plan-and-Solve** (Wang et al. 2023) — arXiv:2305.04091 — Plan-and-Execute 雏形
- **Self-RAG** (Asai et al. 2023) — arXiv:2310.11511 — 自反思 RAG
- **CRAG** (Yan et al. 2024) — arXiv:2401.15884 — Corrective RAG
- **GraphRAG** (Microsoft 2024) — github.com/microsoft/graphrag
- **Modular RAG** (Yunfan Gao 2024) — arXiv:2407.21059 — 模块化 RAG 综述
- **Magentic-One** (Microsoft 2024.11) — Multi-Agent 5 角色编排

### 附录 C: 文档版本历史

- v1.0 (2026.04) — 初版, §0-§4 (基础 + 三层模型 + 5 形态 + 7 层架构 + Workflow 5 Pattern), 阶段 1 完成
- v2.0 (规划) — §5-§9 (Tool Calling + Memory + Multi-Agent + 高级模式 + 框架), 阶段 2
- v3.0 (规划) — §10-§14 (评估 + 案例 + 安全 + 落地 + 未来), 阶段 3

