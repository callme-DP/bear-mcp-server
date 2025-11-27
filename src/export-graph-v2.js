#!/usr/bin/env node
import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";
import { createDb, getDbPath, initEmbedder, createEmbedding } from "./utils.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ===== 可调参数 =====
// Allow override; default points to consolidated data/neo4j.
const OUTPUT_DIR = process.env.NOTE_VECTORS_DIR || path.resolve(__dirname, "../data/neo4j");
const TOPK = Number(process.env.TOPK || 5); // 每条笔记连到最相近的前 K 条
const COS_THRESHOLD = Number(process.env.TH || 0.58); // 语义边阈值
const MAX_CONTENT_CHARS = Number(process.env.MAX_CONTENT_CHARS || 3000);
const BATCH = Number(process.env.BATCH || 32);

const APPLE_EPOCH_OFFSET = 978307200;
const SUMMARY_CACHE = path.join(OUTPUT_DIR, "meta_summary_part.json");
const ZERO_VECTOR = new Array(384).fill(0);
const CONCEPT_PATH = path.join(OUTPUT_DIR, "concepts.json");

// Concept 类型映射（仅使用标签/顶层标签推断，不调用 LLM）
const TYPE_MAP = {
  concept: "topic",
  topic: "topic",
  entity: "entity",
  method: "method",
  area: "area",
  idea: "idea",
  resource: "resource",
};

function log(msg) {
  console.log(msg);
}

async function ensureDir(dir) {
  await fs.mkdir(dir, { recursive: true });
}

async function readJsonIfExists(filePath) {
  try {
    const data = await fs.readFile(filePath, "utf8");
    return JSON.parse(data);
  } catch {
    return null;
  }
}

async function writeJson(filePath, obj) {
  await fs.writeFile(filePath, JSON.stringify(obj, null, 2), "utf8");
}

function appleTsToIso(ts) {
  if (ts === null || ts === undefined) return null;
  return new Date((ts + APPLE_EPOCH_OFFSET) * 1000).toISOString();
}

function buildEmbeddingInput(note) {
  const title = note.title || "";
  const tags = Array.isArray(note.tags) ? note.tags.join(", ") : "";
  const summary = note.summary || "";
  const contentTrim = (note.content || "").slice(0, MAX_CONTENT_CHARS);
  const type = note.type || "";
  return `
[Title]: ${title}

[Tags]: ${tags}

[Type]: ${type}

[Summary]: ${summary}

[Content]:
${contentTrim}
`.trim();
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function createEmbedding_v2(text) {
  // const trimmed = (text || "").slice(0, MAX_CONTENT_CHARS);
  if (!text.trim()) return ZERO_VECTOR;
  try {
    return await createEmbedding(text);
  } catch (err) {
    console.error("Embedding failed, retry once:", err?.message || err);
    await sleep(80);
    try {
      return await createEmbedding(text);
    } catch (err2) {
      console.error("Embedding retry failed, fallback to zero vector:", err2?.message || err2);
      return ZERO_VECTOR;
    }
  }
}

async function fetchAllNotes_v2(db) {
  const notes = await db.allAsync(`
    SELECT 
      ZUNIQUEIDENTIFIER AS id,
      ZTITLE AS title,
      ZTEXT AS content,
      ZCREATIONDATE AS creation_date,
      ZMODIFICATIONDATE AS modification_date
    FROM ZSFNOTE
    WHERE ZTRASHED = 0
  `);

  const tagRows = await db.allAsync(`
    SELECT 
      ZN.ZUNIQUEIDENTIFIER AS id,
      ZT.ZTITLE AS tag
    FROM Z_5TAGS ZNT
    JOIN ZSFNOTETAG ZT ON ZT.Z_PK = ZNT.Z_13TAGS
    JOIN ZSFNOTE ZN     ON ZN.Z_PK = ZNT.Z_5NOTES
  `);

  const tagMap = new Map();
  for (const row of tagRows) {
    if (!tagMap.has(row.id)) tagMap.set(row.id, new Set());
    tagMap.get(row.id).add(row.tag);
  }

  for (const n of notes) {
    n.tags = Array.from(tagMap.get(n.id) || []);
    n.created = appleTsToIso(n.creation_date);
    n.modified = appleTsToIso(n.modification_date);
  }
  return notes;
}

function autoSummary(note) {
  const base = (note.content || note.title || "").replace(/\s+/g, " ").slice(0, 200);
  return base;
}

async function attachSummaries(notes) {
  const cache = (await readJsonIfExists(SUMMARY_CACHE)) || {};
  let updated = false;
  for (const n of notes) {
    if (cache[n.id]) {
      n.summary = cache[n.id];
      continue;
    }
    const sum = autoSummary(n);
    cache[n.id] = sum;
    n.summary = sum;
    updated = true;
  }
  if (updated) {
    await writeJson(SUMMARY_CACHE, cache);
  }
}

function normalizeConcept(tag) {
  if (!tag) return null;
  const parts = tag.split("/");
  const head = parts[0]?.toLowerCase() || "";
  const name = parts.length > 1 ? parts.slice(1).join("/") : tag;
  const type = TYPE_MAP[head] || "topic";
  return { name: name || tag, type, source: "tag", score: 1.0 };
}

function buildConcepts(notes) {
  const concepts = [];
  for (const n of notes) {
    const noteConcepts = [];
    const tagSet = new Set(n.tags || []);
    for (const t of tagSet) { // 直接用标签前缀推断 Concept/type/source=tag
      const c = normalizeConcept(t);
      if (c) noteConcepts.push(c);
    }
    if (n.tags?.length > 0 && n.tags[0]) {
      // 用首个标签兜底作为领域/方向（source=top_tag）
      noteConcepts.push({
        name: n.tags[0],
        type: "area",
        source: "top_tag",
        score: 0.8,
      });
    }
    if (noteConcepts.length > 0) {
      concepts.push({ noteId: n.id, concepts: noteConcepts });
    }
  }
  return concepts;
}

async function embedAll_v2(notes) {
  await initEmbedder();
  const vectors = new Array(notes.length);

  for (let i = 0; i < notes.length; i += BATCH) {
    const batch = notes.slice(i, i + BATCH);
    const embedPromises = batch.map((n) => createEmbedding_v2(buildEmbeddingInput(n)));
    const batchVecs = await Promise.all(embedPromises);
    for (let j = 0; j < batchVecs.length; j++) {
      vectors[i + j] = batchVecs[j];
    }
    log(`🔧 Embedding batch: ${Math.min(i + BATCH, notes.length)} / ${notes.length}`);
  }
  return vectors;
}

function buildSemanticEdges_v2(vectors, ids, topk = TOPK, th = COS_THRESHOLD) {
  const edges = [];
  const dim = vectors[0]?.length || 0;
  if (!dim) return edges;

  for (let i = 0; i < vectors.length; i++) {
    const vi = vectors[i];
    if (!vi || vi.every((x) => x === 0)) continue;
    const scores = [];
    for (let j = 0; j < vectors.length; j++) {
      if (i === j) continue;
      const vj = vectors[j];
      if (!vj || vj.length !== dim || vj.every((x) => x === 0)) continue;
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

  for (const n of notes) for (const t of n.tags || []) tagSet.add(t);
  for (const t of tagSet) nodes.push({ id: `tag:${t}`, type: "Tag", title: t, summary: "", tags: [], created: null, modified: null });

  for (const n of notes) {
    for (const t of n.tags || []) {
      edges.push({ from: n.id, to: `tag:${t}`, type: "has_tag", weight: 1 });
    }
  }
  return { tagNodes: nodes, tagEdges: edges };
}

async function main() {
  log("🚀 export-graph-v2 start");
  await ensureDir(OUTPUT_DIR);

  const db = createDb(getDbPath());
  const notes = await fetchAllNotes_v2(db);
  await attachSummaries(notes);
  log(`📒 Total notes: ${notes.length}`);

  const vectors = await embedAll_v2(notes);
  const ids = notes.map((n) => n.id);
  const semEdges = buildSemanticEdges_v2(vectors, ids);
  const { tagNodes, tagEdges } = buildTagEdges(notes);

  const noteNodes = notes.map((n) => ({
    id: n.id,
    type: "Note",
    title: n.title || "",
    summary: n.summary || "",
    tags: n.tags || [],
    created: n.created || null,
    modified: n.modified || null,
  }));

  const graph = {
    generated_at: new Date().toISOString(),
    stats: {
      notes: noteNodes.length,
      tags: tagNodes.length,
      semantic_edges: semEdges.length,
      tag_edges: tagEdges.length,
      topk: TOPK,
      threshold: COS_THRESHOLD,
    },
    nodes: [...noteNodes, ...tagNodes],
    edges: [...semEdges, ...tagEdges],
  };

  const meta = notes.map((n, idx) => ({
    idx,
    id: n.id,
    title: n.title || "",
    tags: n.tags || [],
    summary: n.summary || "",
    top_tag: (n.tags && n.tags[0]) || null,
  }));

  await writeJson(path.join(OUTPUT_DIR, "graph.json"), graph);
  await fs.writeFile(path.join(OUTPUT_DIR, "embeddings.json"), JSON.stringify(vectors), "utf8");
  await writeJson(path.join(OUTPUT_DIR, "meta.json"), meta);
  const concepts = buildConcepts(notes);
  await writeJson(CONCEPT_PATH, concepts);

  log("✅ export-graph-v2 complete");
  log(`   graph.json -> ${path.join(OUTPUT_DIR, "graph.json")}`);
  log(`   embeddings.json -> ${path.join(OUTPUT_DIR, "embeddings.json")}`);
  log(`   meta.json -> ${path.join(OUTPUT_DIR, "meta.json")}`);
  log(`   concepts.json -> ${CONCEPT_PATH} (items: ${concepts.length})`);
  log(`   Summary cache -> ${SUMMARY_CACHE}`);

  db.close();
}

main().catch((err) => {
  console.error("❌ export-graph-v2 failed:", err);
  process.exit(1);
});
