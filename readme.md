Harley Peng，下面这份 **README.md** 是为你未来的 bear-mcp / personal-knowledge-hub 项目准备的。
内容经过精炼、结构化、专业化，可以直接放到 GitHub 作为项目首页。

全中文（简体）。
风格偏产品化 + 工程化，让潜在贡献者 / 用户一眼理解核心价值。

你可以直接复制粘贴到 GitHub。

---

# 📘 Lumi Bear-MCP

**让 Bear 成为可计算、可分析、可对话的「第二大脑中台」。**
这是一个基于 SQLite + LlamaIndex + MCP 的个人知识分析系统，支持语义检索、主题聚簇、标签漂移分析、PARA 动态追踪与可视化探索。

> 目标不是替代 Bear，而是构建一个与 Bear 共生的「语义智能层」。

---

## 🧭 背景与理念

Bear 是一个优秀的个人知识管理工具，但它缺少：

* 数据分析能力
* 趋势洞察能力
* 聚类与语义“主题发现”能力
* 与智能体（Agent）的深层协作能力

**bear-mcp 的使命：**
为 Bear 提供一个可扩展的 AI 中台，使笔记可以被计算、理解、链接和对话。

---

# ✨ 核心功能

## 1. 📊 分析数据库（Analytics DB）

将 Bear 的内部 SQLite（只读提取）转换成结构化的分析数据库，包括：

| 表名              | 内容                                        |
| --------------- | ----------------------------------------- |
| `notes`         | 标题、正文、创建时间、PARA 类型等                       |
| `tags`          | 标签名称、路径、类型（area/project/resource/archive） |
| `note_tags`     | 笔记 ↔ 标签映射                                 |
| `embeddings`    | 语义向量（持久化）                                 |
| `clusters`      | 聚类主题快照（按月/按范围）                            |
| `note_clusters` | 笔记在主题中的归属与分数                              |

📌 **统一数据层**：Metabase（可视化）与 LlamaIndex（语义层）共用同一份 DB。

---

## 2. 🧠 语义检索 & LlamaIndex 主题聚簇

基于 embedding 与时间维度实现：

* 高质量语义检索（semantic_search）
* 笔记相似度分析（similar_notes）
* 月度主题聚簇（HDBSCAN）
* 自动主题命名（LLM）
* 新兴主题识别（Emerging Topics）
* 主题演化（Topic Drift）

---

## 3. 🔍 PARA 标签漂移分析

分析：

* Resource → Project → Archive 的自然流动
* 某 Area 的主题强度变化
* 标签含义随时间的语义漂移（Tag Semantic Drift）
* 任务/项目的热度上升与冷却

这是一个基于 **向量 + 时间序列** 的动态知识监控系统。

---

## 4. 🧩 MCP 工具集

项目提供可被 ChatGPT / LumiAgent 调用的工具，包括：

* `semantic_search(query, top_k)`
* `similar_notes(note_id)`
* `get_trending_topics(time_range)`
* `note_timeline(topic_or_cluster)`
* `sync_analytics(mode)`（全量/增量 ETL）
* `export_graph`（将笔记/标签/向量写入 Neo4j）

MCP 是整个系统的“对话接口”，让智能体可以自然访问你的第二大脑。

## 5. 🧱 Chunk 管道与相似图

长文/多主题笔记的分块、索引与相似图：

* 切分与索引：`src/chunk/chunker.js`（标题感知分块），`src/chunk/create-index-chunk.js`（生成 meta_chunks.json + chunk_embeddings.json + FAISS 索引），`src/chunk/import_to_neo4j_brew.py`（导入 Chunk 节点、HAS_CHUNK 关系，兜底补 Note）。
* 相似边：`scripts/chunk/build-chunk-sim.js`（基于 chunk_embeddings.json 计算 TopK，相似度=1/(1+d)，输出 chunk_sim_edges.json + chunk_full.json），`scripts/chunk/import-chunk-sim.js`（UNWIND 批量写入 `:SIMILAR_CHUNK`，按 chunk_id 匹配）。
* 调试/对比：`scripts/chunk/chunk-preview.js`（按 note_id 预览分块），`scripts/chunk/compare-recall.js`（整篇 vs 分块召回对比）。

---

# 💬 语义对话 / 查询示例

常见问法如“我的项目有哪些沉淀 SOP 相关的笔记”，可以直接用语义检索完成：

1) 先生成/更新索引（首次会下载模型）：`node src/create-index.js`
2) 启动服务：HTTP（`npm start`）或 MCP（`node src/bear-mcp-server.js`）
3) HTTP 调用示例：
   ```bash
   curl -X POST http://localhost:8000/search_notes \
     -H "Content-Type: application/json" \
     -d '{"args":{"query":"项目有哪些沉淀的 SOP 笔记","semantic":true,"limit":10}}'
   ```
   会返回语义匹配的笔记，即使没有显式“SOP”标签也能命中。
   在 MCP 客户端则调用 `search_notes` 工具并设置 `semantic: true`。

---

# 🎯 架构设计（Architecture）

```
           ┌───────────────────────────────┐
           │          Bear (SQLite)        │
           │ Reminders / Calendar (API)    │
           └───────────────────────────────┘
                          │ 只读 ETL
                          ▼
             ┌──────────────────────────┐
             │     Analytics DB          │
             │  notes / tags / clusters  │
             │  embeddings / timelines   │
             └──────────────────────────┘
              │              │
     (SQL Dashboard)         │ (Semantic Layer)
         Metabase            ▼
                          LlamaIndex
                          HNSW Index
                          HDBSCAN Clusters
                          ▼
            ┌───────────────────────────┐
            │           MCP Tools       │
            │ semantic_search / trends  │
            │ timeline / sync_etl       │
            └───────────────────────────┘
                          ▼
                  LumiAgent / ChatGPT
```

---

# 🏗 目录结构（建议）

```
bear-mcp/
│
├── src/
│   ├── connectors/           # Bear / Reminders / Calendar 提取
│   ├── etl/                  # ETL 管道（notes/tags/para 推断）
│   ├── embeddings/           # 向量生成与落库
│   ├── clustering/           # 主题聚簇 & Drift 分析
│   ├── tools/                # MCP 工具
│   └── api/                  # MCP Server
│
├── analytics_db.sqlite       # 分析数据库
├── raw/                      # 原始 sqlite 快照
└── README.md
```

---

# ⚡ 启动方式（示例）

## Neo4j 语义图层（已验证连通）

1) 启动 Neo4j（本地 brew）：`neo4j start`，浏览器 <http://localhost:7474>，账号密码按你实际设置（示例 neo4j/neo4j123）。
2) 设置环境变量（按需）：`export NEO4J_PASSWORD=neo4j123`，如有自定义端口/账号可设 `NEO4J_URI bolt://localhost:7687`、`NEO4J_USER neo4j`。
3) 运行导出（直接 Node）：`NEO4J_PASSWORD=neo4j123 node -e "import { exportGraph } from './src/export-graph.js'; await exportGraph();"`；或启动 HTTP `npm start` 后 `curl -X POST http://localhost:8000/export_graph`.
4) 使用 v2 文件导入/验证：
   - 生成 v2 图文件：`node src/export-graph-v2.js`（输出到 `note_vectors/graph.json`/`embeddings.json`/`meta.json`）
   - 导入 Neo4j（清空旧节点后导入 + Concept 边支持）：`NEO4J_PASS=neo4j123 node scripts/import-graph-v2.js`
     * 可选：提供 `note_vectors/concepts.json`，结构示例：
       ```json
       [
         {
           "noteId": "NOTE_ID_1",
           "concepts": [
             {"name": "Second Brain", "type": "topic", "score": 0.86, "source": "llm"},
             {"name": "PKM", "type": "area", "score": 0.74, "source": "tag"}
           ]
       }
       ]
       ```
       也支持 map 形式：`{ "NOTE_ID_1": [ {name, type, score, source}, ... ] }`
       自动生成（基于标签/顶级标签）：`node scripts/build-concepts-from-tags.js`，会在 `note_vectors/` 输出 concepts.json
       * Concept 类型枚举可用：topic/entity/method/area/idea/resource（默认 topic）
       * 边标签：`:MENTIONS`，保留 score/source
   - 验证图数据与 Neo4j 计数：`NEO4J_PASS=neo4j123 node src/verify-cypher-v2.js`（结果写入 `note_vectors/verify.log`）

**Concept 支撑（2025-02-01）**
- 诉求：在 SIG 图谱体现“自由生长”的概念层，Note 指向 Concept，Concept 带类型（话题/实体/方法/领域/想法/资源）与来源。
- 生成：`node src/export-graph-v2.js` 会同时生成基于标签/顶级标签的 `note_vectors/concepts.json`；也可用 `node scripts/build-concepts-from-tags.js` 单独生成或手工编辑。
- 导入：`NEO4J_PASS=... node scripts/import-graph-v2.js`（清空后导入 Note/Tag/Concept + MENTIONS 边）。
- 验证：`MATCH (c:Concept) RETURN count(c);` 以及 `MATCH ()-[r:MENTIONS]->() RETURN count(r);`，或运行 `node src/verify-cypher-v2.js`。

**V4 摘要/概念策略（本地优先）**
- 笔记摘要：默认走 LLM + 本地兜底；如需全本地，运行时设 `OLLAMA_SUMMARY_DISABLE=1`（摘要仍写入 `note_vectors/meta_summary_part.json`，`summary_source`=local）。
- 概念生成：保持标签/聚簇来源；如想暂停 LLM 概念抽取，可跳过 `buildLLMConcepts_v4`（仅用 tag/cluster，保留合并逻辑）。
- 聚簇解释：可单独对聚簇标题集调用 LLM 做说明，不影响笔记级摘要/概念。
- Sankey 分析：`meta.json` 提供 note→area（top_tag），`concepts.json` 提供 note→concept；可拼成 `[{noteId,noteTitle,concept,area}]` JSON，在 Metabase/Sankey 中查看 Note→Concept→Area 流向。

---

## 🔧 脚本速查（导出/导入/可视化）
- `src/create-index.js`：从 Bear SQLite 生成本地向量索引（FAISS），输出到 `data/neo4j/note_vectors*`（可用 `NOTE_VECTORS_DIR` 覆盖）。
- `src/export-graph-v2.js`：基础导出，生成 `graph.json`/`embeddings.json`/`meta.json` + 标签派生的 `concepts.json`（输出到 `data/neo4j/`）。
- `src/export-graph-v3.js`：在 v2 基础上做概念合并（词法/语义阈值），仍输出到 `data/neo4j/`。
- `src/export-graph-v4.js`：带 LLM 摘要/概念的增强导出，写入 `meta_summary_part.json` 与 `concepts.json`；`OLLAMA_*` 控制模型/超时。
- `scripts/import-graph-v2.js`：从 `data/neo4j/` 读取导出文件，清空后导入 Neo4j（Note/Tag/Concept + 边）；支持 `NOTE_VECTORS_DIR`/`NEO4J_*`。
- `scripts/import-graph-v2-backlog.js`：早期精简导入（无概念边），仅作兜底参考。
- `scripts/export-graph.js`：旧版导出脚本（基础 Note/Tag/边），保留兼容。
- `scripts/build-llm-concepts.js`：调用 Ollama 批量生成概念，产出 `out/llm_concepts.txt` + 失败列表。
- `scripts/build_metabase_json.js` / `scripts/export_metabase_notes_db.js`：将 `meta_v2.json` + `llm_concepts.txt` 组合为 Metabase 可用的 JSON/SQLite。
- `scripts/test_summaries.js`：快速抽样生成/刷新笔记摘要缓存，不重建全量导出。
- `scripts/test_sig_cypher.js`：Neo4j 连接与写入的快速连通性测试。
- `scripts/animate_graph.py`：Manim 星图动画（读 `data/neo4j/graph.json`/`meta.json`），输出到 `media/`/`out/`。
- `scripts/animate_knowledge_graph.R`：R/ggraph 版本的图可视化/动画，供对比展示。

**V3 概念生成（伪 LLM）**
- 概念来源：标签/顶级标签推断 + 简单规则从标题/summary 抽取 1–3 个短语（非真正 LLM）。
- 合并：词法相似（Levenshtein/Jaccard/子串）+ 语义相似（embedding）去重，融合来源/类型/score。
- 输出：`concepts.json` 中 `noteConcepts`（每笔记的概念列表）和 `canonicalConcepts`（合并后概念）。
5) 验证（cypher-shell/Browser）：
   - `MATCH (n:Note) RETURN count(n);`
   - `MATCH (t:Tag) RETURN count(t);`
   - `MATCH (n:Note) WHERE n.embedding IS NOT NULL RETURN count(n);`

### CLI 说明

当前仓库尚未发布全局 `bear-mcp` 命令，以下为可用脚本：

- `npm start`（或 `node src/bear-mcp-server.js`）：启动 HTTP/MCP 服务端口 8000，提供工具路由。
- `node src/export-graph.js`：导出 v1 图文件到 `src/exports/`，包含节点、边、向量。
- `node src/export-graph-v2.js`：导出 v2 图文件到 `note_vectors/`，供 `scripts/import-graph-v2.js` 导入 Neo4j。

```bash
# 启动 HTTP/MCP 服务
npm start
# 或直接运行 MCP 入口
node src/bear-mcp-server.js

# 导出 Neo4j 图谱（v1）
node src/export-graph.js
# 导出 v2 文件，用于 scripts/import-graph-v2.js
node src/export-graph-v2.js
```

---

# 📚 时间语义能力（Temporal Intelligence）

具备以下高阶分析能力：

### ✔ 最近主题变化

过去 7 天 / 30 天 / 90 天的语义主题变化趋势。

### ✔ PARA 漂移

自动检测：

* Project 升温趋势
* Area 新出现的关注点
* Resource 进入 Archive 的生命周期

### ✔ 标签语义漂移

同一标签随时间在向量空间的移动轨迹。

### ✔ 主题演化（Topic Evolution）

同一 cluster 在不同月份的语义中心偏移量。

---

# 🚀 路线图（Roadmap）

### **M0 – 最小可视化**

* Metabase 连接 Bear 原生 SQLite
* 基础仪表盘（笔记数 / 增长趋势）

### **M1 – Analytics DB + ETL**

* notes/tags/note_tags 全量提取
* PARA 推断器
* 向量落库

### **M2 – 语义层**

* semantic_search
* similar_notes
* 初版聚簇
* 主题命名

### **M3 – LumiAgent 协作**

* MCP 工具
* 趋势洞察
* PARA 漂移
* 主题演化可视化

### One Click & 可视化控制台（设计草案）

* 一键启动：脚本/按钮串联“增量索引 → 启动 MCP/HTTP → 启动隧道(可选) → 动态刷新 .well-known 地址 → 打开控制台”。
* 增量索引：记录 `meta.json`（lastIndexedAt + hash/mtime），仅重算改动笔记并更新 FAISS；无索引则全量。
* 控制台视图：对话区（检索上下文可溯源）、分析页（时间/标签过滤的数量与趋势）、语义聚类页（时间窗口 + 标签过滤，簇代表笔记）、洞察页（基于 daily_insight_context 可调窗口/返回条数），支持结果导出/复制。
* 后端 API 规划：`/stats?from=&to=&tag=` 返回计数/时间序列；`/cluster` 返回聚类结果；`/insight` 包装 daily_insight_context。
* 前端交互：时间范围选择器、标签多选、簇切换；状态栏展示“索引中/服务运行/隧道地址”，提供“重建索引”“复制插件 URL”。
* 模型/技术：继续用本地 embedding + FAISS；聚类可用 UMAP+KMeans/HDBSCAN（Node 或 Python 子进程）；对话模型保持可插拔（GPT/Claude/本地）。

## 现状快照 & 可落地改进

- **阶段判断**：已有语义检索、向量索引、daily_insight 等，分析/聚类/趋势 API 和前端尚未落地，处于“语义检索 + 基础洞察”阶段。
- **知识结构**：基于 Bear note/tags + 向量索引；缺少分析库/聚类结果的持久化表（time series、clusters、note_clusters）。
- **时间关系**：未存储主题/标签的时间演化，暂不能自动回答“过去经验与现在的关系”。需按时间窗口记录主题中心/频次。
- **自然聚类**：未有 UMAP/KMeans/HDBSCAN 的计算与缓存，无法直接回答“哪些笔记自然聚类”。需新增聚类计算并存储簇中心/代表笔记。
- **Metabase 参数化**：现有仪表盘疑似写死过滤值，应在 SQL/模型中加变量（from/to/tag/cluster），并绑定仪表盘参数；或在模型层暴露维度供 Metabase 自动生成筛选器。
- **建议落地顺序**：1) 增量 ETL + 分析库（notes/tags/embeddings/clusters/time_series）；2) 定期聚类，存 cluster_id/centroid/代表笔记/时间桶；3) API `/stats` `/cluster` `/insight` 并接前端；4) Metabase 改为参数化查询；5) “oneclick” 按钮脚本串联索引+服务+tunnel+控制台。

---

# 🔒 隐私声明

* 全部处理在本地完成，不上传到云端数据库。
* 若使用云端 embedding 服务，仅上传文本，不含敏感 metadata。
* 用户需手动授权访问 Bear 数据库，无隐式读取行为。

---

# 🤝 贡献

欢迎提出：

* Bug
* 需求建议
* 新的聚类算法
* 可视化插件（如星图）

---

# 📬 联系（未来可替换为你的信息）

LumiAgent / Harley Peng 第二大脑实验室

> 让知识自由生长，让思考可视化。

## LlamaIndex 小规模聚簇试验（最近 N 条笔记）

- 安装依赖：`pip install -r llamaindex/requirements.txt`
- 运行：`npm run cluster-llama`
- 可选环境变量：
  - `BEAR_DATABASE_PATH`：自定义 Bear DB 路径（默认系统路径）
  - `CLUSTER_RECENT_N`：聚类的最近笔记条数（默认 50）
  - `CLUSTER_K`：手动指定簇数（默认依据 sqrt(n) 取 2~10）
  - `EMBED_MODEL`：默认 `BAAI/bge-small-en-v1.5`；若用 Gemini，请填对应 embedding 模型名（如 `models/text-embedding-004`）
  - `USE_GEMINI`：设为 `1`/`true` 切换到 Gemini API 生成向量（需 `GOOGLE_API_KEY`）
  - `GOOGLE_API_KEY`：Gemini API Key
  - `CLUSTER_DAYS`：仅聚类最近 N 天内的笔记（按修改时间），可与 `CLUSTER_RECENT_N` 结合
- 输出：`vector_data/recent_clusters.json`，包含 note 元数据、聚类分配、每簇笔记列表、自动生成的 cluster 名称、embeddings。

### 落库到 Metabase 可读的 SQLite
- 依赖：同上（先跑一次聚簇生成 `recent_clusters.json`）
- 运行：`npm run load-cluster-sqlite`
- 可选环境变量：
  - `ANALYTICS_DB_PATH`：目标 SQLite 路径，默认 `/Users/yangdongpeng/projects/metabase_bear.sqlite`
- 落表：
  - `recent_notes(note_id, title, tags, updated_at, cluster_id, embedder, embed_model)`
  - `recent_clusters(cluster_id, note_count, note_ids, titles)`
  - `recent_note_tags(note_id, tag)`
