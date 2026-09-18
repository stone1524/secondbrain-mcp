# secondbrain-mcp

> Local-first personal knowledge base as MCP: Markdown vault + hybrid RAG retrieval + a safety-first gateway with hash locks, audit trail, privacy zones and rollback — so that AI agents can *use* your knowledge base without being able to wreck it.
>
> 状态：🚧 v0.1 抽取中（从个人生产系统脱敏重构，目标 2026-10-20 release）｜ License: MIT

## 为什么做这个

2026 年的 AI agent 已经很会「读」，但很少被允许「写」用户的长期知识库——因为写没有安全边界。
本项目把一套**在个人生产环境真实运行**的知识库网关抽成开源框架，核心是四件事：

1. **哈希锁写入**：所有更新携带 `expected_hash`，防止 agent 并发覆盖；冲突即拒绝。
2. **无删除设计**：归档 = 移动到 Archive 区并生成可回滚清单，回收站操作永远留给人类。
3. **隐私分区**：`95_Private` 等私有区对索引和读取工具不可见，人格数据只经过滤视图输出。
4. **全量审计**：每次写操作留痕，agent 生成的内容带来源标签，便于事后区分人写与 AI 写。

## 功能

- [ ] MCP server：标准工具面（read / search / rag_search / create / append / update_metadata / move / archive）
- [ ] 混合检索：语义（bge-small-zh-v1.5）+ 关键词双路召回
- [ ] 增量索引：只重建变更部分，索引可随时推倒重建（derived artifact）
- [ ] 隐私分区与人格数据过滤视图
- [ ] 合成演示知识库（demo vault，无任何真实个人内容）
- [ ] 每个 gateway 工具的单元测试 + 集成测试
- [ ] 一键部署：`uvx` / `npx` 可跑的 MCP 配置示例

## 文档

- [架构总览](docs/architecture.md)
- [脱敏抽取计划](docs/extract-plan.md) ← 从个人系统到开源的迁移红线与顺序
- [路线图](docs/roadmap.md)

## 快速开始

（v0.1 后补全：安装、demo vault 体验、接入 Claude/Cursor 等 MCP 客户端的配置片段）

## License

MIT
