# 架构总览

## 分层

依赖只向下。`core` 与其中的 `policy` 不 import 包内任何其他模块，因此任何层都可以依赖它，它不能反向依赖。

```
adapters/   mcp_stdio.py, cli.py        只做协议转换
    ↓
tools/      registry.py                 参数 schema、错误映射
    ↓
services/   notes, archive, review      用例编排
    ↓
index/      indexer, searcher, text, embed, schema
    ↓
core/       policy, locking, atomic, audit, errors, config
```

## 单一策略源

`core/policy.py` 是唯一做分区判定的地方。生产系统里同一条隐私规则被写了四遍——网关、RAG 索引器、清理收集器、维护脚本——其中三份用原始字符串做大小写敏感比较。在 Windows 上，当目录拼写与硬编码前缀不一致时，私有区会被静默关闭。这是本次重构的第一个动机。

## 分区表

| 分区 | 可读 | 可写 | 进索引 | 默认结果 |
|---|---|---|---|---|
| `PRIVATE`（95_Private, .obsidian） | 否 | 否 | 否 | 否 |
| `ARCHIVE`（98_Archive） | 是 | 是 | 否 | 否 |
| `AUXILIARY`（90_Agent） | 是 | 是 | 是 | 否，需传 `paths` |
| `LEXICAL_ONLY`（30_Resources/FileIndex） | 是 | 是 | 只建词法 | 是 |
| `GENERATED`（audit、candidates、聚合索引） | 是 | 是 | 否 | 否 |
| `STANDARD` | 是 | 是 | 是 | 是 |

两条需要说明理由：

**`ARCHIVE` 不进索引。** 真实 vault 里有 860 篇归档笔记，绝大多数是机器生成的会话产物。让它们进索引，会在工作流类关键词上压过真实笔记——这正是生产系统排除它们的原因。

**`GENERATED` 完全不索引**，而不是「索引但排后面」。聚合知识索引把每一篇其他笔记的标题都重复了一遍，所以对几乎任何查询都有词法命中；做成可检索文档时，它在影子比对里大量占据 rank 1。它是索引，不是文档。

## 索引结构

```sql
documents(path PK, content_hash, size, mtime, updated_at, embed_state, active)
chunks(id PK, path, heading, ordinal, content, token_count, body_chars, active)
postings(chunk_id, term, tf)                    -- 倒排索引
embeddings(chunk_id PK, fingerprint, dim, vector)
model_meta(fingerprint PK, model_name, dim, revision, prefix_scheme, created_at)
meta(key PK, value)
```

`documents.embed_state` 取 `EMBEDDED` / `LEXICAL_ONLY` / `FAILED`。它存在的理由：生产索引只看内容哈希判断「未变更」，一次嵌入失败提交了空向量之后就永远认为这篇不用再处理。实测库里 74/228 个 chunk 永久没有向量，包括整个 Inbox。

`embeddings.fingerprint` 覆盖模型名、维度、库版本、前缀方案。换模型会触发全量向量重建，而不是用同一个模型名把两个向量空间混在一起。

## 检索

**词法**：`postings` 上的 BM25，`k1=1.2`、`b=0.75`。长度归一化只用真实散文——文件桩每个只有几个 token，否则会把平均值压到接近零。

**语义**：每个 generation 把全部向量拼成 `(n_chunks, dim)` 矩阵，查询做一次矩阵乘。

**融合**：RRF，`score = Σ 1/(60 + rank)`，每路深度 50。

生产用的公式是 `lexical + cosine * 2`。实测词法分 3–35、余弦 0–1，向量最多贡献 2 分，而词法经常已经 14 分。后果可量化：11 条两种引擎都命中的查询里，混合检索与纯关键词检索的排名**完全一致**。RRF 比较名次而不是量纲，任一路都无法靠数值尺度压过另一路。

语义通道有两道门禁：

- 只靠语义命中的条目必须同时越过相似度下限（0.55）和最小 token 数（20）；
- 正文不足 120 字符的文档得分减半。

在 22 篇的活跃语料上，近空的 persona 占位文件靠近语料质心，会仅凭余弦就在无关查询上排到前列，而词法分是 0。

## 错误契约

所有可恢复失败都是 `GatewayError`，带稳定的 `code`、`message` 和 `recovery` 指令，以结构化 payload 返回。MCP SDK 会把普通异常压成 `"Error executing tool X"`，消息直接丢失——所以生产系统里那句精心写的「用当前哈希重试」从没到达过模型，乐观并发对调用方是不可见的。

## 并发与持久性

- **锁带归属令牌。** 原先，一个锁被当作 stale 抢走后，原持有者会在 `finally` 里把**新持有者的锁**改名移走，互斥直接失效。
- **`expected_hash` 必填。** 可选哈希意味着调用方只要不传参数就能完全绕过并发控制。
- **写入原子化**（`temp → fsync → os.replace`）。原先全部原地写，进程被杀就留下半截笔记。
- **审计记录 intent 与 outcome**，并带哈希链。原先只在变更前写一行、变更后不写，被拒绝的写和成功的写长得一模一样。
- **manifest 路径视为不可信输入。** 恢复时每条 `from`/`to` 都重新走路径策略；`pathlib` 的 `/` 运算符在右操作数是绝对路径时会丢弃左边，`..` 则能直接走出 vault。

## 设计原则

| 原则 | 含义 |
|---|---|
| 文件是真相 | Vault 永远是纯 Markdown；受控写只走网关 |
| 乐观并发 | 写前必读，`expected_hash` 不匹配即拒绝 |
| 无删除 | 物理删除永远是人的动作；网关只提供归档 + 回滚清单 |
| 索引可弃 | 索引是派生物，可随时全量重建 |
| 隐私默认拒绝 | 私有分区不进读取、不进索引、不进检索 |

## 与同类项目的差异

多数 Obsidian MCP 工具只做「读」。本项目的重心是**受控的写**：哈希锁、审计、无删除归档，让 agent 能长期协作而不毁库。另一个差异是检索质量用可失败的评测门禁验收，而不是靠体感——见 `docs/evaluation.md`。
