# secondbrain-mcp

> Local-first personal knowledge base as MCP: Markdown vault + hybrid RAG retrieval + a safety-first gateway with hash locks, audit trail, privacy zones and rollback — so that AI agents can *use* your knowledge base without being able to wreck it.
>
> 状态：✅ **v0.1** —— 安全网关 + 关键词检索 + MCP stdio server（13 个工具）+ 71 个测试 ｜ License: MIT

## 为什么做这个

2026 年的 AI agent 已经很会「读」，但很少被允许「写」用户的长期知识库——因为写没有安全边界。
本项目把一套**在个人生产环境真实运行**的知识库网关抽成开源框架，核心是四件事：

1. **哈希锁写入**：所有更新携带 `expected_hash`，防止 agent 并发覆盖；冲突即拒绝。
2. **无删除设计**：归档 = 移动到 Archive 区并生成可回滚清单，回收站操作永远留给人类。
3. **隐私分区**：`95_Private` 等私有区对索引和读取工具不可见，人格数据只经过滤视图输出。
4. **全量审计**：每次写操作留痕，agent 生成的内容带来源标签，便于事后区分人写与 AI 写。

## 功能

- [x] MCP server：标准工具面（read / search / rag_search / create / append / update_metadata / move / archive / restore）
- [x] 关键词检索：CJK bigram + BM25（语义向量路见路线图）
- [x] 增量索引：只重建变更部分，索引可随时推倒重建（derived artifact）
- [x] 隐私分区与人格数据过滤视图
- [x] 合成演示知识库（demo vault，无任何真实个人内容）
- [x] 安全与检索的单元测试 + MCP stdio 端到端测试（官方 SDK 客户端）
- [ ] 一键部署：`uvx` / `npx` 可跑（待发布 PyPI 后提供）

## 文档

- [架构总览](docs/architecture.md)
- [脱敏抽取计划](docs/extract-plan.md) ← 从个人系统到开源的迁移红线与顺序
- [路线图](docs/roadmap.md)

## 快速开始

要求 Python 3.10+。

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
      "args": ["-m", "secondbrain_mcp.server"],
      "env": {
        "SB_VAULT_ROOT": "C:/path/to/your/markdown-vault",
        "SB_AGENT_LABEL": "claude"
      }
    }
  }
}
```

环境变量：`SB_VAULT_ROOT`（vault 目录）、`SB_RAG_DB`（索引库路径，默认 `<vault>/.secondbrain/index.db`）、`SB_AGENT_LABEL`（审计归因标签，多客户端共用时区分写入者）。

首次调用 `rag_reindex` 建索引，之后 `rag_search` 即可检索；`read_note` 才是取证。

## 安全模型

- **写**：`create_inbox_note` 只进 inbox 且永不覆盖；`append_note` / `update_metadata` / `move_note` 全部带 `expected_hash` 乐观并发校验与每文件锁
- **删**：不存在。唯一移除路径是带批次回滚清单的归档，restore 可整批或按条恢复
- **隐私**：私有前缀在读、写目的地、索引、搜索四层硬拒绝，泄漏探针测试常绿
- **审计**：每次读写落 JSONL 日志，`SB_AGENT_LABEL` 标注调用方

## License

MIT
