# Bear MCP Agent 使用说明（基于当前代码）

## 运行前置
- 数据源：直接读取 Bear 官方 SQLite，默认路径 `~/Library/Group Containers/9K33E3U3T4.net.shinyfrog.bear/Application Data/database.sqlite`，可用 `BEAR_DATABASE_PATH` 覆盖。
- 语义索引：依赖 `note_vectors.index` + `note_vectors.json`（默认目录 `data/neo4j/`，可用 `NOTE_VECTORS_DIR` 覆盖）。首次或笔记有变动先运行 `node src/create-index.js` 生成。
- 模型：`@xenova/transformers` 加载 `all-MiniLM-L6-v2`，首次运行会自动下载缓存；索引加载失败时会降级为关键词搜索。

## 启动方式
- MCP（stdio）：`node src/bear-mcp-server.js`。输出日志走 stderr，注册的工具见下。适用于 ChatGPT / MCP 客户端。
- HTTP 路由：`npm start`（对应 `src/server.js`），所有工具通过 `POST /<tool>` 调用，请求体形如 `{"args":{...}}`。

## 工具清单
### MCP 中的工具（src/bear-mcp-server.js）
- `search_notes(query, limit=10, semantic=true)`：有索引时走语义检索，否则自动回退关键词；返回 `notes[]`（含 tags/creation_date/score）。
- `get_note(id)`：按 Bear 的 `ZUNIQUEIDENTIFIER` 取单条笔记。
- `get_tags()`：返回 Bear 内所有标签名。
- `retrieve_for_rag(query, limit=5)`：仅当向量索引加载成功时暴露；返回可直接用于 RAG 的简化上下文。

### HTTP 额外工具（src/handle.js）
- `daily_insight_context(hours=24, limit=5)`：聚合“今日笔记”与语义相关笔记，返回带好格式提示词的上下文。
- `find_notes_by_tag(tag, limit=20)`：基于 tags 表的精确标签查询。
- 标签归一化与合并：`normalize_tag_case(style)`, `unify_tag_prefix(mappings)`, `merge_tags(fromTags, toTag)`, `modify_note_tag(note_id, remove_tag, add_tag)`。
- `create_note(title, content)`：写入一条新笔记（时间戳按 Bear 2001 基准计算）。
- ⚠️ 默认 DB 连接以只读方式打开，且部分写操作依赖简化的 `notes/tags` 表结构；在直接连 Bear 原库时这些写类接口会失败，仅在自建可写的分析库场景下使用。

## 调用示例
- 语义搜索（HTTP）：`curl -X POST http://localhost:8000/search_notes -H "Content-Type: application/json" -d '{"args":{"query":"SOP","semantic":true,"limit":5}}'`
- RAG 取数（MCP/HTTP 同参）：`{"tool":"retrieve_for_rag","args":{"query":"Second Brain 落地","limit":5}}`
- 每日洞察上下文：`curl -X POST http://localhost:8000/daily_insight_context -H "Content-Type: application/json" -d '{"args":{"hours":24,"limit":5}}'`

## 使用建议
- 先确保索引存在且加载成功，否则不要调用 `retrieve_for_rag` 和依赖语义排序的场景。
- 读取类工具可直接连 Bear 原生库；写类操作请切换到可写的分析库副本，并调整连接参数后再用。

## 注释约定
- chunk 相关代码（chunker、chunk 预览/对比/索引脚本）统一使用中文注释，避免重复约定和混用中英。
