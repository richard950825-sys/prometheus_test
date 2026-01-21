

# PROMETHEUS-AGI：AI 协作 + 进化闭环（State Machine + Shared Blackboard + Structured Contract）

## 0. 目标与交付

**目标**：不是一次性生成“看起来完成”的方案，而是通过 *生成—审计—演练—归纳—回注入* 的闭环，让 Workflow 在证据驱动下持续迭代收敛。

**最终交付**：

1. **最终 Workflow（可执行图）**：节点/边/条件/预算/权限/终止条件
2. **迭代证据摘要**：为何选择该版本（关键指标变化、主要失败修复）
3. **进化资产**：新增 WDR（规则）与 Anti-Pattern（反模式），带适用条件与证据引用

---

## 1. 三个“硬约束”机制（贯穿全流程）

### 1.1 状态机编排（LangGraph）

智能体不自由聊天，所有动作都必须发生在**显式节点**中，并由**可审计的边条件**驱动转移。
核心思想：把协作过程做成“可执行的控制系统”，而不是“随缘对话”。

### 1.2 共享黑板（ForgeState）

所有节点共享一个中心状态对象 **ForgeState**，用于：

* 传递“需求→草案→审计反馈→演练日志→知识规则→迭代计数”
* 为自纠正与收敛判据提供“可计算的状态变量”

**ForgeState（最小必需字段）**：

* `user_requirements`: 需求与约束（预算/安全/交付形式）
* `draft_workflows`: 候选草案（A/B/C…，版本号）
* `current_workflow`: 当前待审/待测版本
* `audit_report`: Miser 输出（Pass/Reject + reason + required_fixes + risks）
* `simulation_summary`: Oracle 输出（通过率/失败簇/成本/延迟等摘要）
* `failure_cases`: 可复现失败样本（输入+关键日志指纹）
* `wdr_updates`: Synthesis 生成/更新的规则与反模式
* `iteration_count`: 迭代次数
* `stop_reason`: 收敛/终止原因（成功、预算耗尽、无进展等）

### 1.3 自纠正路由 + 结构化契约（Pydantic JSON）

* **结构化契约**：Visionary 的输出必须是可解析 JSON，严格符合 Pydantic Schema，确保 Miser 能做“代码级逻辑计算”，而非读散文。
* **自纠正路由**：当 Miser `Reject`，路由强制回到 Visionary，并把 `audit_report.required_fixes + audit_report.reason` 注入下一轮输入，促使“定向修复”，而不是盲目重试。

> 这三件事共同保证：流程可控、可测、可收敛、可复盘。

---

## 2. 结构化契约：WorkflowDraft Schema（协作协议）

Visionary 必须输出 `AgentWorkflowDraft`（示例字段结构如下，具体可按你现有 schema 扩展）：

* `name`: workflow 名称
* `nodes[]`: 节点定义（角色、模型、工具、system_prompt）
* `edges[]`: 控制流（source/target/condition/max_retries）
* `budgets`: 全局/节点级预算（token、工具调用、超时、最大回合）
* `capabilities`: 节点允许的工具权限（最小权限原则）
* `stop_conditions`: 明确终止条件（成功/失败/升级人工/澄清）
* `risk_hypotheses`: 预期失败模式（用于后续演练验证）
* （可选）`assumptions`: 性能/数据/工具可靠性假设（作为“可被证伪”的输入）

**要点**：避免让 Visionary 输出“拍脑袋成功率”，改为输出“风险假设与验证点”。

---

## 3. 完整闭环流程（可执行状态机视角）

下面用“状态机节点（State Nodes）”方式描述全流程，每个节点都写清楚 **输入→动作→输出→路由条件**。

---

### Node A — Ingest & Normalize（需求归一）

**输入**：用户需求（自然语言）
**动作**：抽取约束（预算/安全/交付形式/工具可用性）写入 `ForgeState.user_requirements`
**输出**：更新 ForgeState
**路由**：→ Node B

---

### Node B — Generate Variants（多方案发散）

**执行者**：Visionary（Cognitive Architect）
**输入**：`user_requirements + wdr_updates(历史规则)`
**动作**：生成 2~3 个 WorkflowDraft 变体（不同拓扑/拆分/模型/工具链），全部必须符合 Schema
**输出**：`draft_workflows`，并选定一个 `current_workflow` 进入审计
**路由**：→ Node C

---

### Node C — Static Audit（静态审计）

**执行者**：Miser（Logic & Token Auditor）
**输入**：`current_workflow`
**动作**（必须可计算）：

* 预算与成本风险（上下文膨胀点、重型模型滥用、缺失预算字段）
* 死循环风险（缺失 max_retries/stop_conditions，边条件不可判定）
* 安全风险（prompt 注入面、工具权限过大、敏感操作无 guardrail）
* 可维护性风险（节点职责混乱、状态接口不一致）

**输出**：`audit_report`：

* `decision`: Pass/Reject
* `required_fixes`: 必须修改项（结构化列表）
* `reason`: 拒绝理由（可注入下一轮）
* `risks_to_validate`: 需要动态演练验证的假设

**路由**：

* `Reject` → Node B（自纠正回路：注入 required_fixes + reason）
* `Pass` → Node D

---

### Node D — Arena Run（动态演练）

**执行者**：Oracle（Synthetic Arena）
**输入**：`current_workflow + risks_to_validate`
**动作**：执行多会话演练，记录轨迹（对话、工具调用、状态转移、token、延迟、错误）
**输出**：

* `simulation_summary`（分层指标：成功/失败、成本、延迟、错误率、循环迹象）
* `failure_cases`（可复现样本包：输入 + 日志指纹 + 触发路径）
  **路由**：→ Node E

---

### Node E — Synthesis & Knowledge Update（归因与知识提炼）

**执行者**：Synthesis（规则归纳器）
**输入**：`simulation_summary + failure_cases + current_workflow`
**动作**：

* 失败聚类（常见失败模式 Top-N）
* 根因归因（prompt/拓扑/边条件/权限/预算）
* 生成可复用资产：

  * `WDR`：规则（适用条件 + 证据来源 + 推荐修复）
  * `Anti-Pattern`：高风险结构模式（触发即预警/否决）
    **输出**：`wdr_updates` 写回 ForgeState
    **路由**：→ Node F

---

### Node F — Revise Draft（基于证据改写）

**执行者**：Visionary（再次上场）
**输入**：`audit_report（若有） + simulation_summary + wdr_updates + failure_cases`
**动作**：做“最小改动修复”或“策略切换”，典型动作：

* 增加澄清节点、降级路径、明确 stop_conditions
* 收紧工具权限、移除高风险工具组合
* 调整拓扑（拆分/合并/加 Router）、替换模型分配
* 调整预算与重试策略（max_retries/backoff）

**输出**：新 `current_workflow`（版本号 + diff 摘要）
**路由**：→ Node C（再次审计）

---

## 4. 收敛与终止条件（避免无限循环）

在每轮结束时（尤其 Node F 后）更新 `iteration_count`，并检查：

* **成功收敛**：关键指标达到阈值（质量/成本/安全）→ 进入 Node G（交付）
* **预算/次数耗尽**：`iteration_count` 超上限或成本超预算 → 进入 Node G（带失败原因/降级交付）
* **无进展收敛**：连续 N 轮关键失败模式未减少（可用失败簇签名判断）→ 强制策略切换或停止

---

### Node G — Package Delivery（打包交付）

**输入**：最终 `current_workflow + wdr_updates + 关键评测摘要`
**输出**：

* Workflow（可执行图 + Schema 合规）
* 迭代证据摘要（版本对比、主要修复点、剩余风险）
* 新增 WDR/反模式（可复用进化资产）
* `stop_reason`（本次为何停止：已达标/预算耗尽/无进展）

---

