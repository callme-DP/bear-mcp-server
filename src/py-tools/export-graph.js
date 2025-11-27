// src/export-graph.js
// 导出全量知识星图 + 语义向量与元数据
import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";
import { createDb, getDbPath, initEmbedder, createEmbedding } from "../utils.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ===== 可调参数 =====
const OUTPUT_DIR = path.join(__dirname, "exports");       // 输出目录：exports/
const TOPK = Number(process.env.TOPK || 5);               // 每条笔记连到最相近的前 K 条
const COS_THRESHOLD = Number(process.env.TH || 0.58);     // 语义边阈值（向量已归一化时≈cosine）
const MAX_CONTENT_CHARS = 3000;                           // 过长笔记截断上限（节省算力）
const BATCH = 32;                                         // 计算 embedding 的 batch 大小

// Apple 2001 时间基准 → UNIX
const APPLE_EPOCH_OFFSET = 978307200;

// 工具：确保输出目录
async function ensureDir(dir) {
  try { await fs.mkdir(dir, { recursive: true }); } catch {}
}

// 工具：一次性查询所有笔记（含聚合标签）
async function fetchAllNotes(db) {
  const rows = await db.allAsync(`
    SELECT 
      n.ZUNIQUEIDENTIFIER AS id,
      n.ZTITLE AS title,
      n.ZTEXT  AS content,
      n.ZCREATIONDATE AS creation_date,
      n.ZMODIFICATIONDATE AS modification_date
    FROM ZSFNOTE n
    WHERE n.ZTRASHED = 0
  `);

  // 拉取所有 tag -> note 的映射
  const tagRows = await db.allAsync(`
    SELECT 
      ZN.ZUNIQUEIDENTIFIER AS id,
      ZT.ZTITLE AS tag
    FROM Z_5TAGS ZNT
    JOIN ZSFNOTETAG ZT ON ZT.Z_PK = ZNT.Z_13TAGS
    JOIN ZSFNOTE ZN     ON ZN.Z_PK = ZNT.Z_5NOTES
  `);

  const tagMap = new Map(); // id -> Set(tags)
  for (const r of tagRows) {
    if (!tagMap.has(r.id)) tagMap.set(r.id, new Set());
    tagMap.get(r.id).add(r.tag);
  }

  for (const n of rows) {
    n.tags = Array.from(tagMap.get(n.id) || []);
    if (n.creation_date) {
      n.creation_date = new Date((n.creation_date + APPLE_EPOCH_OFFSET) * 1000).toISOString();
    }
    if (n.modification_date) {
      n.modification_date = new Date((n.modification_date + APPLE_EPOCH_OFFSET) * 1000).toISOString();
    }
  }

  return rows;
}

// 按 batch 计算 embedding（使用你 utils.js 的 embedder）
async function embedAll(notes) {
  await initEmbedder(); // 确保已加载
  const vectors = new Array(notes.length);

  for (let i = 0; i < notes.length; i += BATCH) {
    const batch = notes.slice(i, i + BATCH);
    // 减少算力：content 过长截断；若为空则用 title 兜底
    const texts = batch.map(n => {
      const t = (n.content && n.content.trim().length > 0)
        ? n.content.slice(0, MAX_CONTENT_CHARS)
        : (n.title || "");
      return t || " "; // 保险
    });

    // 逐条算，简单稳妥（也可改为并发 Promise.all）
    for (let j = 0; j < texts.length; j++) {
      const vec = await createEmbedding(texts[j]); // 已 mean+normalize（见 utils.js）
      vectors[i + j] = vec;
    }
    process.stdout.write(`\r🔧 Embedding: ${Math.min(i + BATCH, notes.length)} / ${notes.length}`);
  }
  process.stdout.write("\n");
  return vectors;
}

// 计算语义 TopK 边（使用余弦≈点积；utils.createEmbedding 已 normalize:true）
function buildSemanticEdges(vectors, ids, topk = TOPK, th = COS_THRESHOLD) {
  const edges = [];
  const dim = vectors[0]?.length || 0;
  if (!dim) return edges;

  for (let i = 0; i < vectors.length; i++) {
    const vi = vectors[i];
    // 计算与其他所有的相似度（简洁起见 O(N^2)，数据量大时可后续接上 faiss/annoy/hnsw）
    const scores = [];
    for (let j = 0; j < vectors.length; j++) {
      if (i === j) continue;
      const vj = vectors[j];
      // 归一化后 dot 即 cosine
      let dot = 0;
      for (let k = 0; k < dim; k++) dot += vi[k] * vj[k];
      scores.push([j, dot]);
    }
    scores.sort((a, b) => b[1] - a[1]);
    let added = 0;
    for (const [j, s] of scores) {
      if (s < th) break;
      edges.push({ from: ids[i], to: ids[j], type: "semantic", weight: Number(s.toFixed(4)) });
      added++;
      if (added >= topk) break;
    }
  }
  return edges;
}

function buildTagEdges(notes) {
  const nodes = [];
  const edges = [];
  const tagSet = new Set();

  // 收集 tag 节点
  for (const n of notes) for (const t of (n.tags || [])) tagSet.add(t);
  for (const t of tagSet) nodes.push({ id: `tag:${t}`, type: "Tag", name: t });

  // 连接 Note -> Tag
  for (const n of notes) {
    for (const t of (n.tags || [])) {
      edges.push({ from: n.id, to: `tag:${t}`, type: "has_tag", weight: 1 });
    }
  }
  return { tagNodes: nodes, tagEdges: edges };
}

async function main() {
  await ensureDir(OUTPUT_DIR);

  const db = createDb(getDbPath());
  const notes = await fetchAllNotes(db);

  // 基础节点（Note）
  const noteNodes = notes.map(n => ({
    id: n.id,
    type: "Note",
    title: n.title || "",
    created: n.creation_date,
    modified: n.modification_date,
    tags: n.tags || []
  }));

  // 计算向量
  const vectors = await embedAll(notes);

  // 语义边（Note-Note）
  const ids = notes.map(n => n.id);
  const semEdges = buildSemanticEdges(vectors, ids);

  // Tag 相关
  const { tagNodes, tagEdges } = buildTagEdges(notes);

  // 汇总图
  const graph = {
    generated_at: new Date().toISOString(),
    stats: {
      notes: noteNodes.length,
      tags: tagNodes.length,
      semantic_edges: semEdges.length,
      tag_edges: tagEdges.length
    },
    nodes: [...noteNodes, ...tagNodes],
    edges: [...semEdges, ...tagEdges]
  };

  // 写出：graph.json（星图），embeddings.json（矩阵），meta.json（可视化需要的元信息）
  await fs.writeFile(path.join(OUTPUT_DIR, "graph.json"), JSON.stringify(graph, null, 2), "utf8");
  await fs.writeFile(path.join(OUTPUT_DIR, "embeddings.json"), JSON.stringify(vectors), "utf8");
  await fs.writeFile(
    path.join(OUTPUT_DIR, "meta.json"),
    JSON.stringify(
      notes.map((n, idx) => ({
        idx,
        id: n.id,
        title: n.title || "",
        top_tag: (n.tags && n.tags[0]) || null,
        tags: n.tags || []
      })),
      null,
      2
    ),
    "utf8"
  );

  console.log("✅ 导出完成：");
  console.log(`   - 星图:      ${path.join(OUTPUT_DIR, "graph.json")}`);
  console.log(`   - 向量矩阵:  ${path.join(OUTPUT_DIR, "embeddings.json")}`);
  console.log(`   - 元数据:    ${path.join(OUTPUT_DIR, "meta.json")}`);
  console.log(`   - 语义边阈值: TH=${COS_THRESHOLD}，TopK=${TOPK}`);
}

main().catch(err => {
  console.error("❌ 导出失败：", err);
  process.exit(1);
});
