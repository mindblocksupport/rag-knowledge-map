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

### 1.4 三大常见误区 (面试高频)

#### 1.4.1 误区 1 — "Agent 替代 RAG"

- **错在哪**: 以为有了 Agent 就不需要 RAG 管道了
- **真相**: Agent 内部 80-90% 时间还在调 RAG (检索是 Agent 工具池里的核心工具之一)
- **正解**: Agent 是 RAG 的上层调度, 是叠加不是替代
- **量化证据**: Klarna 95% query 走纯 Modular RAG, 5% 走 Agent (而 Agent 内部仍多次调 RAG)

#### 1.4.2 误区 2 — "多步 LLM 调用 = Agent"

- **错在哪**: 把固定脚本调多次 LLM 也叫 Agent
- **真相**: 多步 ≠ Agent. 工程师写死的 5 步管道不算 Agent, 算 Workflow
- **正解**: Agent 的核心标志是 "LLM 看上一步结果决定下一步" 这个反馈环
- **鉴别口诀**: 流程图能预先画出来 → Workflow; 流程图取决于运行时 LLM 输出 → Agent

#### 1.4.3 误区 3 — "上 Agent 就解决质量问题"

- **错在哪**: 检索召回差, 想用 Agent 抢救
- **真相**: Agent 解决 "一次性管道解不了" 的问题, 不解决 "检索本身差" 的问题
- **正解**: Recall@10 < 0.7 时上 Agent 只会循环报错, 必须先治 L1+L2+L3
- **正确顺序**: 先把 Modular RAG 调到 Recall@10 ≥ 0.85, 再上 Agent

### 1.5 跟其它框架的关系

#### 1.5.1 跟 OpenAI 的 GPT 模式
- OpenAI 没有公开等价的"三层模型"分类, 但 Function Calling / Agents SDK 实质对应 Anthropic 的 Workflow + Agent

#### 1.5.2 跟 LangChain 的 Chain / Agent 对应
- LangChain Chain ≈ Workflow Pattern (Prompt Chaining / Sequential Chain)
- LangChain Agent ≈ Anthropic Agent (ReAct Agent / OpenAI Function Agent)
- LangGraph 跨两层 (graph 任意编排, 既支持 Workflow 也支持 Agent)

#### 1.5.3 跟 Modular RAG (Yunfan Gao 2024) 关系
- Modular RAG 7 模块 = Anthropic 三层中的"层次 1 Augmented LLM" 工程化实现
- Modular RAG 是"模块化的 Augmented LLM"
- Agent 是 Modular RAG + Planner + Tool Loop + Memory + 多步推理

### 1.6 关键金句 (Anthropic 原话)

- > "成功不在于构建最复杂的系统, 而在于为你的需求构建正确的系统"
- > "从简单提示开始, 用全面的评估优化它们, 仅在简单方案不足时添加多步 agentic 系统"
- > "agentic 系统通常会用延迟和成本换取更好的任务性能"
- > "对框架的错误假设是客户错误的常见来源"

---

## 二. Agent 5 大形态深度 (Plan-and-Execute / ReAct / Multi-Agent / Self-Reflection / Iterative)

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

## 三. 7 层架构 + 决策循环

### 3.1 7 层架构总图

> 这是 Agentic RAG 的"解剖图". 7 个核心层 + 1 个横切 (Cost Controller), 每层职责清晰. 配合下方 7 层职责详解 + 决策循环看.

### 3.2 7 层职责详解 (每层一句话讲清)

#### 3.2.1 Layer 1 — Query Understanding (入口)
- 输入: 用户原始 query (str)
- 输出: 意图标签 (intent) + 复杂度评分 (1-5) + 取 Memory L2 用户偏好
- 关键技术: 意图多分类 (kNN on intent embeddings 起步, LLM judge 进阶) + 复杂度评分
- 选型: 起步 BERT-class classifier + 关键词规则; 高级 Haiku 4.5 做 LLM-as-judge
- 反模式: 跳过这层, Router 没法做精准切流, 90% 简单 query 误进 Agent 烧钱

#### 3.2.2 Layer 2 — Router (路由决策)
- 输入: query + 意图 + 复杂度
- 输出: 路径标签 (simple / agent / sql / clarification)
- 关键技术: 三层混合 — 规则 (先) → 语义匹配 (中) → LLM 兜底 (后)
- 切流比例 (工业典型): 简单 80-95% / Agent 5-20% / 其它 < 5%
- 反模式: 全用 LLM 路由 — 单 query 多花 0.5-1s, 高 QPS 时 LLM 排队拖死全链

#### 3.2.3 Layer 3 — Planner (Agent 大脑)
- 输入: query + Memory + 工具描述列表
- 输出: 结构化 Plan (JSON, 含 step_id / tool_name / params / fallback)
- 关键技术: 强推理 LLM (Claude Sonnet 4.5 / GPT-5 / o3 / DeepSeek-R1)
- 两种实现: Plan-and-Execute (开局全规划, 省钱) / ReAct (每步规划, 灵活)
- 反模式: 用 Haiku / GPT-3.5 做 Planner — 规划质量塌, 步骤之间逻辑断裂

#### 3.2.4 Layer 4 — Tool Execution Loop (双手循环)
- 输入: Plan 中的下一步动作 + 当前 state
- 输出: tool_results 列表 + 新 state
- 关键组件:
  - Tool Registry (5-12 工具池, 每个含 name + description + JSON Schema)
  - Tool Executor (解析 LLM tool_call → 调真实 API → 序列化结果回传)
  - Loop Controller (判断终止)
- 4 终止条件 (任一即退出):
  - LLM 主动声明 "已收集到足够信息"
  - max_steps 触发 (默认 8)
  - 同一工具连续重复 3 次
  - 累计 cost / token 超预算
- 反模式: 工具池 > 20 个 — LLM 选错率塌 30-50%

#### 3.2.5 Layer 5 — Memory (脊髓三层)
- L1 Session Memory (Redis, TTL 6h): 本次对话最近 20 message + 已调工具结果 — Agent 多步必需
- L2 User Preference (Postgres JSONB): 用户偏好 / 角色 / 历史购买 / 历史问题 — 跨会话累积
- L3 Business Memory (Vector DB): 重要决策 / 客户画像 / 团队约定 — 跨用户共享
- 容量约束: Memory 占 context window ≤ 6K (16K budget 内), 否则挤掉检索结果空间
- 摘要策略: 超容量时调 LLM 摘旧 message 成 200 字 (Conversation Summary)
- 反模式: L1 永久不清理 — 100 query 后 Memory 占满拖慢链路

#### 3.2.6 Layer 6 — Synthesizer (综合答案)
- 输入: 所有 tool_results + query + Memory
- 输出: 最终答案 + 引用 (citation 含 chunk_id + source URL)
- 关键技术: 便宜 LLM (Claude Haiku 4.5 / GPT-5-mini) 做综合, 不必用强模型
- 选型: 90% 场景 Haiku 够用; 综合涉及深度推理时升级 Sonnet (但 Plan 阶段已分解, 大概率不需要)
- 反模式: 用强模型 (Sonnet) 综合 — 单 query 成本翻 5-10x, 综合任务不吃推理力

#### 3.2.7 Layer 7 — Validator (质量校验闸门)
- 输入: 候选答案 + 引用 + tool_results
- 输出: 通过 (放行) / 拒答 (返回兜底) / 重试 (回 Layer 3 改 Plan)
- 4 种检查 (并行):
  - Faithfulness: 答案是否被 tool_results 支撑 (RAGAS faithfulness 公式)
  - Citation: 每个事实声明是否有引用 + 引用是否真实存在
  - PII: 答案是否含 SSN / 信用卡 / 个人邮箱 (Presidio + 中文 NER)
  - Guardrail: 是否触发 LlamaGuard / Constitutional AI 红线
- 反模式: 跳过 Validator 直接返回 — Klarna 早期就栽过, 答错被截图传播

#### 3.2.8 横切 — Cost Controller (FinOps 监控)
- 监控指标: token / cost / latency / step_count / tool_call_count
- 硬熔断 4 阈值 (任一触发即退出):
  - max_cost_per_query: 客服 $1 / Coding $5 / 科研 $50
  - max_steps: 客服 8 / 通用 12 / Coding 50+
  - max_same_tool_repeat: 3 (LLM 卡住信号)
  - timeout_per_step: 30s (单步工具 hang)
- 真实事故: 没设熔断的项目, 边缘 query 单次烧 $200 (详见 §10.X)

### 3.3 决策循环 (5 部件如何串起来)

完整数据流 (一次 Agent query 的全过程):
- 输入: 用户 query → Layer 1 入口 → Layer 2 Router 判路径
- 简单路径 (80-95% 流量): → Modular RAG 单次调用 → Layer 7 Validator → 答案
- Agent 路径 (5-20% 流量):
  - 步 A — Layer 3 Planner 接 query, 调 frontier LLM 生成 N 步执行 Plan
  - 步 B — 进入 Layer 4 Loop (max_steps 内反复):
    - B.1 — 从 Layer 5 Memory 取 state (query + history + tool_results)
    - B.2 — LLM 看 state + 工具描述, 决定下一步调哪个 Tool 和参数
    - B.3 — Tool Executor 执行 (Modular RAG / SQL / Web Search / Function Call)
    - B.4 — 工具结果写回 Layer 5 Memory + Cost Controller 累计 cost
    - B.5 — 判 4 终止条件, 任一满足则退出 Loop
  - 步 C — Layer 6 Synthesizer 综合所有 tool_results, 生成最终答案 + 引用
  - 步 D — Layer 7 Validator 校验, 不通过则拒答或回步 A 改 Plan 重试
  - 步 E — 返回答案 + 完整 trace (LangSmith / Phoenix / Langfuse 调试用)
- 全程 Cost Controller 监控, 超预算硬熔断退出

### 3.4 7 层架构 vs 5 层企业架构 (跟通用 RAG 文档的关系)

- 通用 RAG 5 层 (L1 数据治理 / L2 索引 / L3 检索 / L4 Router / L5 Agent) 是建材
- Agent RAG 7 层 = 5 层架构中 L5 (Agent) 这一层的 zoom-in 视图
- 关键映射:
  - Agent RAG Layer 4 (Tool Loop) 内的 "Modular RAG" 工具 = 5 层架构的 L1+L2+L3 完整管道
  - Agent RAG Layer 2 (Router) = 5 层架构 L4 Router
  - 即: Agent RAG 在 5 层架构 L5 内部展开成 7 层细化结构

---

## 四. Workflow 5 Pattern 深度 (Anthropic 2024.12)

### 4.1 5 Pattern 总览 (在考虑 Agent 前先看这 5 种)

> Anthropic 总结的 5 种 Workflow Pattern, 90% 业务用其中一种就解决, 不需要上 Agent.

| Pattern | 一句话 | 适合 | 实现复杂度 |
|---|---|---|---|
| **Prompt Chaining** | 任务拆成线性 N 步, 每步 LLM 处理上一步输出 | 任务能拆清晰串行步骤 | 低 (几行代码) |
| **Routing** | 分类输入后转发到不同的专门处理分支 | 有明显类别区分 | 低 |
| **Parallelization** | 同时跑多个独立 LLM 任务后聚合 | 子任务独立 / 需要投票 | 中 |
| **Orchestrator-Workers** | 中枢 LLM 动态拆任务给 Worker LLM | 子任务运行时才知数量 | 中 |
| **Evaluator-Optimizer** | 一 LLM 生成 + 一 LLM 评估的迭代循环 | 输出质量高 + 有评估标准 | 中-高 |

### 4.2 Pattern 1 — Prompt Chaining (链式调用)

#### 4.2.1 解决什么问题
- 复杂任务一次 LLM 输出做不到 (e.g. 文档摘要 + 关键词提取 + 标签分类)
- 单 prompt 塞 3 个任务质量塌, 拆开成 3 个 prompt 串行做更好
- Prompt Chaining 解法: **把任务拆成线性 N 步, 每步独立 prompt, 上一步输出作为下一步输入**

#### 4.2.2 算法
- 步 1 — 把任务拆成 N 个独立 step (e.g. step 1: 翻译, step 2: 审校, step 3: 输出格式化)
- 步 2 — 每 step 写独立 prompt
- 步 3 — 串行执行: output_1 = LLM(prompt_1, input)
- 步 4 — output_2 = LLM(prompt_2, output_1)
- 步 5 — ... 直到 final output
- 可选: 每步加 gate check (规则验证), 失败重试或终止

#### 4.2.3 业务场景
- ✅ 文档处理: 摘要 → 关键词 → 标签 → 入库 (4 步流水线)
- ✅ 翻译质量: 翻译 → 审校 → 改进 → 输出 (3 步迭代)
- ✅ 内容生成: 大纲 → 段落 → 优化 → 校对 (4 步)
- ❌ 任务无法清晰拆步骤 (用 Agent ReAct)
- ❌ 步骤之间高度耦合 (拆了反而错)

#### 4.2.4 真实采用
- **Anthropic 官方文档处理 pipeline** (Anthropic blog 公开)
- **LangChain SequentialChain** (经典实现)
- **LlamaIndex Pipeline** (内置)

#### 4.2.5 反模式
- ❌ Step 拆得太碎 (10+ step) — 每步 LLM 调用累积成本爆
- ❌ Step 之间没有 gate check — 错误一路传, 最后才发现
- ❌ 每步都用强 LLM (Sonnet) — 简单 step 用 Haiku 够, 省 5-10×
- ❌ 把可并行的强行串行 — 失去并行优化机会 (用 Pattern 3)

### 4.3 Pattern 2 — Routing (路由分流)

#### 4.3.1 解决什么问题
- 用户 query 有不同类别 (FAQ / 编号 / 复杂诊断), 用同一个 prompt 处理质量塌
- Routing 解法: **先分类输入, 再转发到不同的专门处理分支**

#### 4.3.2 算法
- 步 1 — Router LLM (或 classifier) 接 query, 输出类别标签
- 步 2 — 根据标签路由到对应处理分支:
  - FAQ → 简单 RAG
  - 编号 → BM25 字面检索
  - 复杂诊断 → Agent
- 步 3 — 各分支独立处理, 用专门 prompt + 专门 LLM

#### 4.3.3 业务场景
- ✅ 客服 query 分类 (退款 / 物流 / 账户 / 技术)
- ✅ 内容分类后选不同 LLM (复杂 → Sonnet, 简单 → Haiku)
- ✅ 多语言路由 (中 → 中文 LLM, 英 → 英文 LLM)
- ✅ 这就是企业 RAG L4 Modular Router 的 Workflow 实现 (详见通用 RAG 文档 §7)

#### 4.3.4 三层混合路由 (业界标配)
- **第 1 层 — 规则路由** (最快, 占 70%): 关键词正则 / query 长度 / 包含数字编号
- **第 2 层 — 语义路由** (中等, 占 20%): query embedding 跟预先标注的"意图样本" cosine 相似度 (kNN), top-1 类别即标签
- **第 3 层 — LLM 兜底** (最贵, 占 10%): 前两层都没匹配, 用 Haiku 4.5 LLM-as-judge 给意图标签

#### 4.3.5 反模式
- ❌ 全 LLM 路由 — 单 query 多花 0.5-1s, 高 QPS 拖死
- ❌ Router 准确率 < 0.95 上线 — 误分流, 用户体验崩
- ❌ 不监控分流分布 — Router 退化没人发现 (90% fallback agent 月底发现)

### 4.4 Pattern 3 — Parallelization (并行化)

#### 4.4.1 解决什么问题
- 子任务相互独立但顺序执行慢, 浪费时间
- Parallelization 解法: **同时跑多个独立 LLM 任务, 然后聚合结果**

#### 4.4.2 两个子变体

##### 子变体 A: Sectioning (分段)
- 把大任务切成独立子任务并行
- e.g. 10 个文档每个独立摘要 → 并行 10 路, 总耗时 = 单路时间 (而非 × 10)

##### 子变体 B: Voting (投票)
- 同一任务跑 N 次, 取多数票
- e.g. 安全检测跑 3 次取一致结果 → 提质量 (vs 单次随机)

#### 4.4.3 业务场景
- ✅ 长文档处理: 切段并行 embed + 摘要
- ✅ 内容审核投票: 3 个 LLM 投票判断是否违规
- ✅ Multi-Query 检索: 4 个 query 变体并行 (RAG-Fusion 基础)
- ❌ 子任务有依赖 (前后顺序) — 必须串行, 用 Pattern 1
- ❌ 子任务结果不能聚合 — 没法 Voting

#### 4.4.4 真实采用
- **Anthropic Constitutional AI** (3 个 LLM 投票判断输出)
- **LangChain MapReduceChain** (sectioning 实现)
- **LlamaIndex 异步 batch retrieval**

#### 4.4.5 反模式
- ❌ 子任务有依赖硬并行 — 结果不一致
- ❌ Voting 只用 2 个 LLM — 50/50 时无法决定, 至少 3 个
- ❌ 不限并发数 — 100 路并行打爆 LLM API rate limit

### 4.5 Pattern 4 — Orchestrator-Workers (协调员 + 工人)

#### 4.5.1 解决什么问题
- 任务结构复杂, 子任务**数量 / 内容运行时才知道**
- Pattern 3 Parallelization 假设子任务预先固定, 但有些场景 Orchestrator 要在运行时拆
- Orchestrator-Workers 解法: **一个中枢 LLM (Orchestrator) 动态拆解任务 → 分配给多个 Worker LLM → 综合结果**

#### 4.5.2 跟 Multi-Agent 区别 (易混淆)

| 维度 | Orchestrator-Workers (Workflow) | Multi-Agent (Agent) |
|---|---|---|
| Worker 路径 | **工程师写死** (Worker 内部按固定逻辑跑) | **LLM 决定** (Worker 自己用 LLM 决策) |
| 鉴别 | Worker 内部不是 Agent | Worker 内部是 Agent |
| 例子 | 写竞品分析 (Orchestrator 拆 5 家, 每家 Worker 跑相同 prompt) | Magentic-One (Worker 各自是独立 Agent) |

#### 4.5.3 算法
- 步 1 — Orchestrator LLM 接 query, 动态拆任务 (输出 N 个子任务的描述)
- 步 2 — Worker LLM 并行处理每个子任务 (每个 Worker 跑相同 prompt 不同输入, 或不同 prompt)
- 步 3 — Orchestrator 接收 Worker 输出, 综合成最终答案

#### 4.5.4 业务场景
- ✅ 写竞品分析: Orchestrator 拆 5 家竞品, 每家 Worker 跑调研
- ✅ 多视角生成: Orchestrator 让 Worker 1 写正面 / Worker 2 写反面 / Worker 3 写中立
- ✅ 文档批处理: Orchestrator 决定每文档用哪个 prompt (依据文档类型)
- ❌ 简单单任务 (用 Pattern 1 或 Pattern 2 即可)

#### 4.5.5 反模式
- ❌ Orchestrator 拆 20+ 子任务 — Worker 调用爆, 拆得太细质量塌
- ❌ Worker 用 Sonnet — 简单子任务用 Haiku, 省 5-10×
- ❌ Worker 之间需要通信 — Pattern 4 假设独立, 有通信用 Multi-Agent

### 4.6 Pattern 5 — Evaluator-Optimizer (评估 + 优化)

#### 4.6.1 解决什么问题
- 输出质量需要"先生成再评估再改进"循环
- 单次 LLM 输出不够好, 但又不需要 Agent 那么复杂
- Evaluator-Optimizer 解法: **一个 LLM 生成初稿 → 另一个 LLM 评估并反馈 → 形成迭代改进循环**

#### 4.6.2 跟 Self-Reflection (Agent) 区别
- Pattern 5: **循环数预先写死** (e.g. 3 轮)
- Self-Reflection: **LLM 决定何时停**
- 鉴别: 循环数定不定

#### 4.6.3 算法
- 步 1 — Generator LLM 接 query, 生成初版输出
- 步 2 — Evaluator LLM 接收输出, 给评分 + 改进建议:
  - 5 维度评分 (准确性 / 完整性 / 流畅度 / 简洁 / 风格)
  - 改进建议 (具体到段落 / 句子)
- 步 3 — Optimizer LLM 接收建议 + 原输出, 生成改进版
- 步 4 — 回步 2 (循环 N 次, 通常 2-3 轮)
- 步 5 — 最后一轮输出作为最终结果

#### 4.6.4 业务场景
- ✅ 翻译质量: 翻译 → Evaluator 给 5 维度打分 + 改进 → Optimizer 重写 → 循环 3 轮
- ✅ 代码生成: 生成 → Evaluator 跑测试 / lint → Optimizer 修 → 循环
- ✅ 摘要质量: 生成 → Evaluator 评信息密度 → Optimizer 改 → 循环
- ❌ 输出无法评估 (创意写作没标准) — Evaluator 不准
- ❌ 任务太简单 (FAQ 客服) — overkill, 单次生成就够

#### 4.6.5 真实采用
- **Anthropic 内部翻译质量 pipeline** (官方 blog 提到)
- **CodeT5 / Self-Edit** (代码生成评估)
- **Reflexion 灵感来源** (但 Reflexion 是 Agent 不是 Workflow)

#### 4.6.6 反模式
- ❌ 不限循环次数 (变 Agent Self-Reflection)
- ❌ Evaluator 跟 Generator 同 LLM — 容易自圆其说
- ❌ Evaluator 阈值过高 — 永远不满意, 浪费 LLM 调用

### 4.7 5 Pattern 选型决策

#### 4.7.1 决策表

| Pattern | 何时选 | 实现复杂度 | 真实采用 |
|---|---|---|---|
| Prompt Chaining | 任务能拆成清晰串行步骤 | 低 (几行代码) | Anthropic 文档 pipeline / LangChain SequentialChain |
| Routing | 有明显类别区分 (FAQ/编号/复杂) | 低 | 任何 RAG L4 Router |
| Parallelization | 子任务独立 / 需要投票 | 中 | Constitutional AI / Multi-Query |
| Orchestrator-Workers | 子任务运行时才知数量 | 中 | 写竞品分析 / 多视角生成 |
| Evaluator-Optimizer | 输出质量高 + 有评估标准 | 中-高 | Anthropic 翻译 / CodeT5 |

#### 4.7.2 关键认知

- ✅ 这 5 种都属于 Workflow (Anthropic 三层模型层次 2), **不是 Agent** — 因为路径都是工程师预先写好的
- ✅ 90% 业务能用其中一种解决, 不需要上 Agent
- ✅ 实现都是几十行代码 + 标准 LLM API, **不需要 LangGraph / AutoGen 等重框架**
- ✅ 跟 Anthropic Prompt Caching 配合好可省 35-49% 成本

---

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

