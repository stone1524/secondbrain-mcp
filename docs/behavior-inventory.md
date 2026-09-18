# 行为清单（W1 盘点 · 2026-09-19）

> 来源：生产网关单文件实现（367 行）+ RAG 三件套（索引器 379 行 / 嵌入助手 61 行 / 评测器 53 行）。
> 本清单是 v0.1 重写的**行为基线**：开源版按「行为对拍」验收——同名工具在生产版与开源版上对同一输入的行为必须一致。

## 1. 工具面（17 个）

| 工具 | 安全轨道 | v0.1 取舍 |
|------|----------|-----------|
| read_note | 路径收容 → 隐私硬拒 → 审计；返回全文 sha256 + 分页窗口 | ✅ 重写 |
| list_notes | 遍历过滤私有区；mtime 倒序；pattern/limit | ✅ 重写 |
| search_notes | 索引在→纯词法模式；索引空→线性扫描兜底（跳私有） | ✅ 重写 |
| rag_search | scope 白名单校验；审计记录 scope_decision；只回证据候选（要求 read_note 取证） | ✅ 重写 |
| rag_reindex | 只动可重建索引；审计含索引状态 | ✅ 重写 |
| rag_status | 仅元数据，不暴露内容 | ✅ 重写 |
| create_inbox_note | 目录白名单（3 个 inbox）+ 文件名净化 + `.md` 强制 + **wx flag 存在即败** | ✅ 重写 |
| append_note | 锁 + 哈希锁 + 审计 | ✅ 重写 |
| update_metadata | **字段白名单**（status/tags/updated_at/agent_generated/confidence/state/last_verified）+ 锁 + 哈希锁 | ✅ 重写 |
| move_note | 目的地禁入私有区 + 目的存在检查 + 锁 + 哈希锁 | ✅ 重写 |
| list_cleanup_candidates | 只读夜间产物，零变更 | ⏩ v0.2（依赖 collector） |
| archive_notes | **候选白名单**（仅夜间清单内的笔记）+ 每次移动前哈希复验 + dry_run + 冲突哈希后缀重命名 + 批次回滚清单 + restore 命令回显 | ⏩ v0.2（v0.1 先交付 move_note + 回滚清单 + restore 脚本） |
| conversation_route | 纯配置驱动分类（quick/context/workflow），零 Vault I/O | ✅ 重写（配置文件通用化） |
| personality_read_context | 仅 confirmed + 非敏感条目；维度过滤；条数上限；原始 JSON 永不经通用工具暴露 | ⏩ v0.2（propose-only 模式保留，作为差异点） |
| personality_propose_candidate | **只能提交 pending 候选**；锁 + 内联快照（留 30 份）+ 审计含 agent 标签 | ⏩ v0.2 |
| personality_list_candidates | pending + 非敏感过滤 | ⏩ v0.2 |
| MCP 壳（stdio JSON-RPC） | 协议版本协商；notifications 静默；错误统一 isError 包裹 | ✅ 重写 |

## 2. 横切安全机制（重写为独立模块）

| 机制 | 生产行为 | 测试要点 |
|------|----------|----------|
| 路径收容 | 拒绝绝对路径、拒绝逃逸 vault 的相对路径 | 8 种逃逸用例全拒 |
| 隐私硬边界 | 私有区在「读 / move 目的地 / 索引排除 / 检索过滤」四层全部拒绝 | 每层一条探针；评测器的泄漏门必须是 0 命中 |
| 文件锁 | lock 文件 `wx` 原子创建；>5 分钟视为陈旧并轮转到 history 目录；finally 释放 | 并发双写一方被拒；陈旧锁自恢复 |
| 哈希锁 | 写前校验全文 sha256，不匹配即拒并提示带新哈希重试 | 读→外部改→写 被拒 |
| 审计 | JSONL 追加；记录 agent 标签（多客户端归因）+ scope_decision | 每个工具一条断言 |
| 无覆盖创建 | inbox 创建用 `wx`，已存在即失败 | 覆盖尝试被拒 |
| 无删除 | 全网关无 unlink/rm；唯一移除路径是可回滚的归档 | 静态断言 + 行为测试 |

## 3. RAG 管线行为

**索引（三阶段）**：只读扫描（哈希比对 + 自愈：documents 无 chunks 视为中断遗留并重索引）→ 纯计算（分块按标题 + 1800 字符上限；CJK bigram 分词；面包屑嵌入输入=路径+标题+正文）→ **单事务提交**（BEGIN IMMEDIATE，中断不留半写状态）。 vanished 文件 ENOENT 容忍并停用旧路径；schema 版本变更强制重建 + VACUUM 回收。

**排除与分区**：硬排除 5 类（私有区 / .obsidian / 归档区 / 人格审计与候选视图 / 再生聚合清单）；辅助区（自动生成的文件索引、agent 运维笔记）默认跳过、显式 `paths` 才启用；文件名存根区免嵌入省成本。

**检索**：BM25(k1=1.2, b=0.75) + 精确短语奖励 + 余弦×2 融合 + 0.35 余弦地板 + 每笔记取最高分块去重。两个关键修正必须保留：
- BM25 长度归一化只用散文行计算 avgLength（存根行会压碎真实分块的 tf 权重）；
- 双语查询扩展**只增不删**——原词永远保留，扩展词只是增加召回竞争者，最坏不伤害精确命中。

**嵌入子进程**：常驻 python 助手（NDJSON 协议、懒加载模型、readline 循环而非管道迭代）；事件循环引用计数区分「一次性 CLI」（要能退出）与「常驻网关」（要保活）；stderr 持续排空防 OS 管道缓冲死锁；win32 用 taskkill /T 杀进程树。模型：fastembed 本地 ONNX 推理，bge-small-zh-v1.5，L2 归一化，E5 系列自动加 query/passage 前缀，HF 默认离线，单文本 6000 字符截断。

**评测门**（发布与夜间必过）：Recall@5 ≥ 0.8；**隐私泄漏探针 = 0**（直接搜私有区不得有任何命中）；来源可追溯（每个命中必须带路径）；索引空心预检（chunks/documents < 0.5 判为完整性失败而非质量问题）。退出码 0/10/20 供流水线分支。

## 4. 配置面（开源版前缀 `OBSIDIAN_*` → `SB_*`）

`SB_HUB` / `SB_VAULT_ROOT` / `SB_REVIEW_LOG_ROOT` / `SB_LOCK_ROOT` / `SB_AGENT_LABEL`（审计归因）/ `SB_RAG_DB` / `SB_RAG_PYTHON` / `SB_RAG_SEMANTIC=0`（纯词法模式）/ `SB_RAG_INDEX_TIMEOUT_MS` / `SB_RAG_MODEL` / `SB_RAG_MODEL_CACHE` / `SB_RAG_HF_OFFLINE` / `SB_RAG_EMBED_BATCH`。**代码中无任何凭据**——嵌入模型本地推理，无 API 依赖。

## 5. 相对抽取计划的修订

1. `queryGlossary` 双语扩展表含个人领域条目 → 开源版改为「通用默认（几个示例）+ 用户配置文件加载」，机制不变。
2. `archive_notes` 依赖夜间 collector 产物 → v0.1 交付「move_note + 回滚清单写入 + restore 脚本」自洽闭环，candidate 白名单流放 v0.2 与 collector 一起交付。
3. `personality_*` 三工具 → v0.2 独立模块（propose-only + 快照 + 过滤视图是一个独立卖点，不挤 v0.1 工期）。
4. 评测用例集 → 生产用例指向真实笔记，开源版换合成 demo vault 用例，**评测器机制原样保留**（含隐私泄漏探针——这是 README 差异化主张的自动化证明）。
5. 前端壳 server.mjs 的 MCP 协议部分极薄（20 行）→ 直接重写，工具注册表改声明式。

## 6. 生产教训 → 直接转为测试用例

- [ ] 双客户端并发写同一笔记：文件锁一方拒绝
- [ ] 读后外部修改再写：哈希锁拒绝
- [ ] 一次性 CLI 跑完必须退出（子进程不挂事件循环）
- [ ] stderr 大量输出不死锁（ONNX 日志场景）
- [ ] 存根密集语料下 BM25 归一化（散文 avgLength）
- [ ] 索引中断后下次运行自愈（documents 有、chunks 无）
- [ ] 归档移动与夜间扫描之间文件被改：跳过而非错归
- [ ] schema 升级强制重建且新旧向量不混
- [ ] 私有区四层探针全部为拒/零命中
