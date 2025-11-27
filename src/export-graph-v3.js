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
const CONCEPT_PATH = path.join(OUTPUT_DIR, "concepts.json");

// 概念合并阈值
const LEX_EDIT_TH = 0.85; // Levenshtein 相似阈值
const LEX_JACCARD_TH = 0.75; // 词集合 Jaccard 阈值
const SEM_CONCEPT_TH = 0.8; // 概念 embedding 语义阈值
const SCORE_BONUS_PER_SOURCE = 0.05;
const SCORE_BONUS_CAP = 0.15;

const APPLE_EPOCH_OFFSET = 978307200;
const SUMMARY_CACHE = path.join(OUTPUT_DIR, "meta_summary_part.json");
const ZERO_VECTOR = new Array(384).fill(0);

// Concept 类型映射（标签 / 顶层标签）
const TYPE_MAP = {
  concept: "topic",
  topic: "topic",
  entity: "entity",
  method: "method",
  area: "area",
  idea: "idea",
  resource: "resource",
};

const TYPE_PRIORITY = ["area", "idea", "topic", "method", "entity", "resource"];
const NAME_PRIORITY = ["semantic_cluster", "llm", "tag", "top_tag"];

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

function normalizeConceptName(name) {
  if (!name) return "";
  return name
    .toLowerCase()
    .replace(/[\s.,;:，。；：·]/g, "")
    .replace(/[()（）[\]{}]/g, "")
    .normalize("NFKC");
}

function levenshtein(a, b) {
  const m = a.length;
  const n = b.length;
  if (m === 0) return n;
  if (n === 0) return m;
  const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      dp[i][j] = Math.min(
        dp[i - 1][j] + 1,
        dp[i][j - 1] + 1,
        dp[i - 1][j - 1] + cost
      );
    }
  }
  return dp[m][n];
}

function tokenize(str) {
  return str
    .toLowerCase()
    .split(/[^a-z0-9\u4e00-\u9fa5]+/i)
    .filter(Boolean);
}

function jaccardSim(a, b) {
  const ta = new Set(tokenize(a));
  const tb = new Set(tokenize(b));
  const inter = new Set([...ta].filter((x) => tb.has(x)));
  const union = new Set([...ta, ...tb]);
  if (union.size === 0) return 0;
  return inter.size / union.size;
}

function isLexicallySimilar(a, b) {
  const na = normalizeConceptName(a);
  const nb = normalizeConceptName(b);
  if (!na || !nb) return false;
  const dist = levenshtein(na, nb);
  const editSim = 1 - dist / Math.max(na.length, nb.length);
  if (editSim >= LEX_EDIT_TH) return true;
  const jac = jaccardSim(na, nb);
  if (jac >= LEX_JACCARD_TH) return true;
  if (na.includes(nb) || nb.includes(na)) return true; // 子串吸收
  return false;
}

function cosineSim(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  let dot = 0;
  let sa = 0;
  let sb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    sa += a[i] * a[i];
    sb += b[i] * b[i];
  }
  if (sa === 0 || sb === 0) return 0;
  return dot / (Math.sqrt(sa) * Math.sqrt(sb));
}

const conceptEmbeddingCache = new Map();
async function getConceptEmbedding(name) {
  const key = normalizeConceptName(name);
  if (conceptEmbeddingCache.has(key)) return conceptEmbeddingCache.get(key);
  const vec = await createEmbedding_v2(name);
  conceptEmbeddingCache.set(key, vec);
  return vec;
}

function normalizeConcept(tag) {
  if (!tag) return null;
  const parts = tag.split("/");
  const head = parts[0]?.toLowerCase() || "";
  const name = parts.length > 1 ? parts.slice(1).join("/") : tag;
  const type = TYPE_MAP[head] || "topic";
  return { name: name || tag, type, source: "tag", score: 1.0 };
}

function buildTagConcepts(notes) {
  const noteConcepts = [];
  for (const n of notes) {
    const noteConceptList = [];
    const tagSet = new Set(n.tags || []);
    for (const t of tagSet) {
      const c = normalizeConcept(t);
      if (c) noteConceptList.push(c);
    }
    if (n.tags?.length > 0 && n.tags[0]) {
      noteConceptList.push({
        name: n.tags[0],
        type: "area",
        source: "top_tag",
        score: 0.8,
      });
    }
    noteConcepts.push({ noteId: n.id, concepts: noteConceptList });
  }
  return noteConcepts;
}

function simpleLLMConcepts(note) {
  const concepts = [];
  const candidateText = (note.title || "") + " " + (note.summary || "");
  const tokens = tokenize(candidateText)
    .filter((t) => t.length >= 2 && t.length <= 6)
    .slice(0, 6);
  const seen = new Set();
  for (const t of tokens) {
    if (seen.has(t)) continue;
    concepts.push({
      name: t,
      source: "llm",
      type: "topic",
      score: 0.65,
    });
    seen.add(t);
    if (concepts.length >= 3) break;
  }
  return concepts;
}

function buildLLMConcepts(notes) {
  return notes.map((n) => ({
    noteId: n.id,
    concepts: simpleLLMConcepts(n),
  }));
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

function buildClusterConcepts(notes, semEdges) {
  const noteMap = new Map(notes.map((n) => [n.id, n]));
  const adj = new Map();
  for (const e of semEdges) {
    if (!adj.has(e.from)) adj.set(e.from, new Set());
    if (!adj.has(e.to)) adj.set(e.to, new Set());
    adj.get(e.from).add(e.to);
    adj.get(e.to).add(e.from);
  }

  const visited = new Set();
  const clusters = [];
  for (const id of noteMap.keys()) {
    if (visited.has(id)) continue;
    const queue = [id];
    const cluster = [];
    visited.add(id);
    while (queue.length) {
      const cur = queue.shift();
      cluster.push(cur);
      for (const nxt of adj.get(cur) || []) {
        if (!visited.has(nxt)) {
          visited.add(nxt);
          queue.push(nxt);
        }
      }
    }
    if (cluster.length >= 2) clusters.push(cluster);
  }

  const concepts = [];
  for (const cluster of clusters) {
    const titles = cluster.map((id) => noteMap.get(id)?.title || "");
    const keywordScore = new Map();
    for (const t of titles) {
      for (const token of tokenize(t)) {
        if (!token) continue;
        keywordScore.set(token, (keywordScore.get(token) || 0) + 1);
      }
    }
    const sorted = [...keywordScore.entries()].sort((a, b) => b[1] - a[1]);
    const topTokens = sorted.slice(0, 2).map(([k]) => k);
    const name = topTokens.join(" ").trim() || "semantic cluster";

    concepts.push({
      concept: {
        name,
        source: "semantic_cluster",
        type: "topic",
        score: 0.9,
      },
      noteIds: cluster,
    });
  }
  return concepts;
}

function flattenRawConcepts(tagConcepts, llmConcepts, clusterConcepts) {
  const raw = [];
  for (const item of tagConcepts) {
    for (const c of item.concepts) {
      raw.push({ ...c, noteId: item.noteId });
    }
  }
  for (const item of llmConcepts) {
    for (const c of item.concepts) {
      raw.push({ ...c, noteId: item.noteId });
    }
  }
  for (const c of clusterConcepts) {
    for (const noteId of c.noteIds) {
      raw.push({ ...c.concept, noteId });
    }
  }
  return raw;
}

function pickType(types) {
  const byPriority = [...TYPE_PRIORITY, "topic"];
  for (const t of byPriority) {
    if (types.has(t)) return t;
  }
  // fallback: first
  return types.values().next().value || "topic";
}

function pickName(namesBySource) {
  for (const src of NAME_PRIORITY) {
    if (namesBySource[src]) return namesBySource[src][0];
  }
  const any = Object.values(namesBySource).flat();
  return any[0] || "";
}

async function mergeConcepts(rawConcepts) {
  const groups = [];

  for (const item of rawConcepts) {
    const { name, type = "topic", source = "tag", score = 1.0, noteId } = item;
    if (!name || !noteId) continue;
    let matched = null;

    for (const g of groups) {
      const candidateNames = [g.canonicalName, ...Object.values(g.namesBySource).flat()];
      if (candidateNames.some((cn) => isLexicallySimilar(name, cn))) {
        matched = g;
        break;
      }
      if (g.embedding) {
        const vec = await getConceptEmbedding(name);
        const sim = cosineSim(vec, g.embedding);
        if (sim >= SEM_CONCEPT_TH) {
          matched = g;
          break;
        }
      }
    }

    if (!matched) {
      const vec = await getConceptEmbedding(name);
      const g = {
        canonicalName: name,
        namesBySource: { [source]: [name] },
        types: new Set([type]),
        sources: new Set([source]),
        scores: [score],
        noteIds: new Set([noteId]),
        embedding: vec,
      };
      groups.push(g);
    } else {
      matched.noteIds.add(noteId);
      matched.sources.add(source);
      matched.types.add(type);
      matched.scores.push(score);
      matched.namesBySource[source] = matched.namesBySource[source] || [];
      matched.namesBySource[source].push(name);
      // 更新 embedding 为均值
      const vec = await getConceptEmbedding(name);
      if (matched.embedding && vec?.length === matched.embedding.length) {
        const merged = matched.embedding.map((v, idx) => (v + vec[idx]) / 2);
        matched.embedding = merged;
      }
    }
  }

  const canonicalConcepts = groups.map((g) => {
    const distinctSources = g.sources.size;
    const baseScore = Math.max(...g.scores, 0);
    const bonus = Math.min(SCORE_BONUS_CAP, SCORE_BONUS_PER_SOURCE * Math.max(0, distinctSources - 1));
    return {
      name: pickName(g.namesBySource),
      type: pickType(g.types),
      sources: Array.from(g.sources),
      score: Number((baseScore + bonus).toFixed(3)),
      noteIds: Array.from(g.noteIds),
    };
  });

  // 构建 note 层面的原始概念证据，便于导入和追溯
  const noteConcepts = new Map();
  for (const item of rawConcepts) {
    if (!noteConcepts.has(item.noteId)) noteConcepts.set(item.noteId, []);
    noteConcepts.get(item.noteId).push({
      name: item.name,
      type: item.type || "topic",
      source: item.source || "tag",
      score: item.score ?? 1.0,
    });
  }

  return {
    canonicalConcepts,
    noteConcepts: Array.from(noteConcepts.entries()).map(([noteId, concepts]) => ({
      noteId,
      concepts,
    })),
  };
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

async function main() {
  log("🚀 export-graph-v3 start");
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

  const tagConcepts = buildTagConcepts(notes);
  const llmConcepts = buildLLMConcepts(notes);
  const clusterConcepts = buildClusterConcepts(notes, semEdges);

  const rawConcepts = flattenRawConcepts(tagConcepts, llmConcepts, clusterConcepts);
  const merged = await mergeConcepts(rawConcepts);

  const conceptsPayload = {
    generated_at: new Date().toISOString(),
    params: {
      lexical: { LEX_EDIT_TH, LEX_JACCARD_TH },
      semantic: { SEM_CONCEPT_TH },
      score_bonus_per_source: SCORE_BONUS_PER_SOURCE,
      score_bonus_cap: SCORE_BONUS_CAP,
    },
    noteConcepts: merged.noteConcepts,
    canonicalConcepts: merged.canonicalConcepts,
  };

  await writeJson(path.join(OUTPUT_DIR, "graph.json"), graph);
  await fs.writeFile(path.join(OUTPUT_DIR, "embeddings.json"), JSON.stringify(vectors), "utf8");
  await writeJson(path.join(OUTPUT_DIR, "meta.json"), meta);
  await writeJson(CONCEPT_PATH, conceptsPayload);

  log("✅ export-graph-v3 complete");
  log(`   graph.json -> ${path.join(OUTPUT_DIR, "graph.json")}`);
  log(`   embeddings.json -> ${path.join(OUTPUT_DIR, "embeddings.json")}`);
  log(`   meta.json -> ${path.join(OUTPUT_DIR, "meta.json")}`);
  log(`   concepts.json -> ${CONCEPT_PATH}`);
  log(`   Summary cache -> ${SUMMARY_CACHE}`);

  db.close();
}

main().catch((err) => {
  console.error("❌ export-graph-v3 failed:", err);
  process.exit(1);
});
