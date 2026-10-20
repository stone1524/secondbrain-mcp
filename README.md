# secondbrain-mcp

> Local-first personal knowledge base as MCP: Markdown vault + hybrid RAG retrieval + a safety-first gateway with hash locks, audit trail, privacy zones and rollback — so that AI agents can *use* your knowledge base without being able to wreck it.

> 状态：✅ **v0.2** —— 13 个工具 · 95 个测试（含真实 stdio 协议端到端）· 检索门禁全过 ｜ License: MIT

## 为什么做这个

2026 年的 AI agent 已经很会「读」，但很少被允许「写」用户的长期知识库——因为写没有安全边界。

本项目把一套**在个人生产环境真实运行**的知识库网关重写为单一 Python 核心，核心是四件事：

1. **哈希锁写入**：所有更新必须携带 `expected_hash`，且在锁内复验；冲突即拒绝并给出恢复指令。
2. **无删除设计**：归档 = 移动到 Archive 区并生成可回滚清单，清单在移动**之前**写入。
3. **隐私分区**：`95_Private` 在读取、写入、索引、检索四层拒绝，分区判定大小写无关。
4. **全量审计**：intent 与 outcome 分开记录，每条带前一条的哈希，形成可校验的链。

## 功能

- [x] MCP server：13 个窄工具（read / list / search / index / review / create / append / patch / meta / move / archive / restore）
- [x] 混合检索：BM25 倒排 + 向量余弦 + RRF 融合，两道防噪门禁
- [x] 增量索引：`embed_state` 显式跟踪，失败在下一轮自动重试
- [x] 模型指纹：换模型强制重建向量，杜绝向量空间混用
- [x] 正文编辑：`note_patch` 两阶段确认，先出 diff 再落盘
- [x] 隐私分区与待确认队列
- [x] 合成演示知识库（demo vault，无任何真实个人内容）
- [x] 单元测试 + 对抗性测试 + MCP stdio 端到端 + 可失败的检索门禁
- [x] 部署健康检查（暂存库跑全契约 + live 索引只读快照探针）

## 文档

- [架构总览](docs/architecture.md) — 分层、分区表、索引结构、检索设计
- [评测方法](docs/evaluation.md) — 为什么旧评测报 1.00 而真实只有 60%，以及现在的四道门禁
- [威胁模型](docs/threat-model.md) — 防什么、不防什么、已知限制
- [迁移记录](docs/migration.md) — 从 Node 网关迁到 Python 核心的行为变化与回滚方式
- [脱敏抽取计划](docs/extract-plan.md)
- [路线图](docs/roadmap.md)

## 快速开始

要求 Python 3.11+。

```bash
git clone https://github.com/stone1524/secondbrain-mcp
cd secondbrain-mcp
pip install -e .
```

**试用合成 demo vault**（不接你自己的数据，仓库自带、全部为合成内容）：

```bash
secondbrain-mcp          # 不设 SB_VAULT_ROOT 时默认服务 demo-vault/
```

**接入自己的知识库 / MCP 客户端**（Claude Code、Cursor 等任何支持 MCP 的客户端）：

```json
{
  "mcpServers": {
    "secondbrain": {
      "command": "python",
      "args": ["-m", "secondbrain_mcp.adapters.mcp_stdio"],
      "env": {
        "SB_VAULT_ROOT": "C:/path/to/your/markdown-vault",
        "SB_STATE_ROOT": "C:/path/to/state",
        "SB_AGENT_LABEL": "claude"
      }
    }
  }
}
```

环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `SB_VAULT_ROOT` | 仓库内 `demo-vault/` | vault 目录 |
| `SB_STATE_ROOT` | `<vault>/.secondbrain` | 索引、审计、锁的根目录 |
| `SB_RAG_DB` | `<state>/index.db` | 索引库路径 |
| `SB_AGENT_LABEL` | 无 | 审计归因标签，多客户端共用时区分写入者 |
| `SB_RAG_SEMANTIC` | `1` | 设为 `0` 走纯词法 |
| `SB_RAG_MODEL` | `BAAI/bge-small-zh-v1.5` | 嵌入模型 |
| `SB_RAG_MODEL_CACHE` | 无 | 模型缓存目录（离线场景必填） |

首次调用 `index_rebuild` 建索引，之后 `search` 即可检索；`note_read` 才是取证。

## 安全模型

- **写**：`note_create` 只进 inbox 且永不覆盖（重名加后缀）；`note_append` / `note_patch` / `note_meta` / `note_move` 全部要求 `expected_hash`，并在文件锁内复验
- **删**：不存在。唯一移除路径是带批次回滚清单的归档，restore 可整批或按条恢复
- **隐私**：私有前缀在读、写目的地、索引、搜索四层拒绝，且大小写无关；泄漏探针在单元与线上两层测试
- **审计**：每次写落 `intent` 与 `outcome` 两条，带哈希链；`AuditLog.verify()` 可校验
- **内容即数据**：所有读取与检索结果带 `trust: untrusted_note`，提示调用方不得把笔记内容当作指令

## 评测

```bash
python -m secondbrain_mcp.eval.run --cases eval/cases-dev.json --semantic
```

四道门禁：`recall@5 ≥ 0.85`、`semantic rank-1 wins ≥ 1`、`null vectors == 0`、`privacy leakage == 0`。
当前结果与旧系统的对照见 [docs/evaluation.md](docs/evaluation.md)。

## License

MIT
