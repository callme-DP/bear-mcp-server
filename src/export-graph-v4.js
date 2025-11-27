#!/usr/bin/env node
import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";
import { createDb, getDbPath, initEmbedder, createEmbedding } from "./utils.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ===== 可调参数 =====
// 允许通过 NOTE_VECTORS_DIR 覆盖，默认写入 data/neo4j
const OUTPUT_DIR = process.env.NOTE_VECTORS_DIR || path.resolve(__dirname, "../data/neo4j"); // 输出目录
const TOPK = Number(process.env.TOPK || 5); // 每条笔记连到最相近的前 K 条
const COS_THRESHOLD = Number(process.env.TH || 0.58); // 语义边阈值
const MAX_CONTENT_CHARS = Number(process.env.MAX_CONTENT_CHARS || 3000); // LLM 输入内容截断
const BATCH = Number(process.env.BATCH || 32); // 嵌入批大小
const CONCEPT_PATH = path.join(OUTPUT_DIR, "concepts.json");

// Ollama 调用配置
const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434/api/chat"; // Ollama 接口
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || "qwen2.5:7b"; // 概念抽取模型
// 默认 8s，避免大模型首 token 过慢。可用 env 调整。
const OLLAMA_TIMEOUT_MS = Number(process.env.OLLAMA_TIMEOUT_MS || 8000); // LLM 首次调用超时
const OLLAMA_RETRY_MS = Number(process.env.OLLAMA_RETRY_MS || 12000); // LLM 重试超时
const OLLAMA_LOG_SAMPLES = Number(process.env.OLLAMA_LOG_SAMPLES || 0); // 打印前 N 条概念抽取
const OLLAMA_LOG_CONTENT = ["1", "true", "yes"].includes(String(process.env.OLLAMA_LOG_CONTENT || "").toLowerCase()); // 打印 LLM 输入/调试
const OLLAMA_SUMMARY_MODEL_ENV = process.env.OLLAMA_SUMMARY_MODEL; // 摘要模型（不填则用 OLLAMA_MODEL）
const OLLAMA_SUMMARY_DISABLE = ["1", "true", "yes"].includes(String(process.env.OLLAMA_SUMMARY_DISABLE || "").toLowerCase()); // 关闭摘要 LLM
const OLLAMA_SUMMARY_NUM_PREDICT = Number(process.env.OLLAMA_SUMMARY_NUM_PREDICT || 64); // 摘要生成 token 上限
const OLLAMA_SUMMARY_STREAM = ["1", "true", "yes"].includes(String(process.env.OLLAMA_SUMMARY_STREAM || "false").toLowerCase()); // 摘要是否流式
const SUMMARY_MIN_LLM_CHARS = Number(process.env.SUMMARY_MIN_LLM_CHARS || 50); // 内容不足阈值则跳过 LLM
const SUMMARY_MIN_OUTPUT_LEN = Number(process.env.SUMMARY_MIN_OUTPUT_LEN || 15); // 摘要最小有效长度
const EMB_LOG_TIMING = ["1", "true", "yes"].includes(String(process.env.EMB_LOG_TIMING || "").toLowerCase()); // 输出嵌入耗时估计

// 概念合并阈值
const LEX_EDIT_TH = 0.85; // Levenshtein 相似阈值
const LEX_JACCARD_TH = 0.75; // 词集合 Jaccard 阈值
const SEM_CONCEPT_TH = 0.8; // 概念 embedding 语义阈值
const SCORE_BONUS_PER_SOURCE = 0.05;
const SCORE_BONUS_CAP = 0.15;

const APPLE_EPOCH_OFFSET = 978307200;
const SUMMARY_CACHE = path.join(OUTPUT_DIR, "meta_summary_part.json");
const ZERO_VECTOR = new Array(384).fill(0);
const SUMMARY_CONTENT_LIMIT = 2000; // 摘要输入截断

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
const BLACKLIST = new Set(["mj", "ll", "lm", "ai", "ok", "mm"]);

function log(msg) {
  console.log(msg);
}

function debugLog(msg) {
  if (OLLAMA_LOG_CONTENT) console.log(msg);
}

// Ensure output directory exists.
async function ensureDir(dir) {
  await fs.mkdir(dir, { recursive: true });
}

// Read JSON if present, else null.
async function readJsonIfExists(filePath) {
  try {
    const data = await fs.readFile(filePath, "utf8");
    return JSON.parse(data);
  } catch {
    return null;
  }
}

// Persist object as formatted JSON.
async function writeJson(filePath, obj) {
  await fs.writeFile(filePath, JSON.stringify(obj, null, 2), "utf8");
}

// Convert Bear Apple timestamp to ISO.
function appleTsToIso(ts) {
  if (ts === null || ts === undefined) return null;
  return new Date((ts + APPLE_EPOCH_OFFSET) * 1000).toISOString();
}

// Build embedding input template.
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

// Embedding with retry/fallback.
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

// ============= Note 基础数据 & Summary Pipeline =============
// Fetch Bear notes and tag mappings; attach created/modified ISO fields.
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

// Remove markdown/links/emoji/noise for summary inputs.
function cleanContent(text) {
  if (!text) return "";
  return text
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/`[^`]*`/g, " ")
    .replace(/https?:\/\/\S+/g, " ")
    .replace(/\!\[[^\]]*\]\([^\)]*\)/g, " ")
    .replace(/\[[^\]]*\]\([^\)]*\)/g, " ")
    .replace(/[#>*_~\-]{1,}/g, " ")
    .replace(/[^\w\s\u4e00-\u9fa5.,，。！？!?\n]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

// Split text into coarse sentences.
function splitSentences(text) {
  return text
    .split(/[\n。！？!?]/)
    .map((s) => s.trim())
    .filter(Boolean);
}

// Sanity-check LLM summary output.
function validSummaryText(text) {
  if (!text) return false;
  const t = text.trim();
  if (t.length < SUMMARY_MIN_OUTPUT_LEN) return false;
  if (/^\[?\s*\]?$/.test(t)) return false;
  if (/^(null|undefined|\{\}|\[\])$/i.test(t)) return false;
  if (/^[^a-zA-Z0-9\u4e00-\u9fa5]+$/.test(t)) return false;
  return true;
}

// Parse cached summary entry into normalized shape.
function parseCachedSummary(entry) {
  if (!entry) return null;
  if (typeof entry === "string") {
    return { summary: entry, source: "local", model: "none" };
  }
  if (entry.summary) {
    return {
      summary: entry.summary,
      source: entry.source || entry.summary_source || "local",
      model: entry.model || entry.summary_model || "none",
    };
  }
  return null;
}

// Attach summaries (LLM first, local fallback) and update cache.
async function attachSummaries(notes) {
  const cache = (await readJsonIfExists(SUMMARY_CACHE)) || {};
  let updated = false;
  for (const n of notes) {
    const cached = parseCachedSummary(cache[n.id]);
    if (cached) {
      n.summary = cached.summary;
      n.summary_source = cached.source;
      n.summary_model = cached.model;
      continue;
    }
    const res = await buildSummary(n);
    n.summary = res.summary || "";
    n.summary_source = res.source || "local";
    n.summary_model = res.model || "none";
    cache[n.id] = { summary: n.summary, source: n.summary_source, model: n.summary_model };
    updated = true;
  }
  if (updated) {
    await writeJson(SUMMARY_CACHE, cache);
  }
}

// ============= LLM 相关 =============
// Thin Ollama chat wrapper with timeout/abort + optional stream/num_predict.
async function ollamaChat(model, messages, timeoutMs, options = {}) {
  debugLog(
    `🛰️  Ollama request model=${model} stream=${options.stream ?? true} num_predict=${options.numPredict ?? "default"} timeout=${timeoutMs}ms; user_len=${(messages?.[1]?.content || "").length}`
  );
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(OLLAMA_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        messages,
        stream: options.stream ?? true,
        num_predict: options.numPredict,
      }),
      signal: controller.signal,
    });
    if (!res.ok) throw new Error(`Ollama HTTP ${res.status}`);
    const data = await res.json();
    return data?.message?.content || "";
  } finally {
    clearTimeout(timeout);
  }
}

// Drop noisy LLM concept candidates.
function filterLLMConcept(c) {
  if (!c || !c.name) return false;
  const raw = c.name.trim();
  if (!raw) return false;
  const cleaned = raw.replace(/[.,;:，。；：·、!！?？"'“”]/g, "").trim();
  if (cleaned.length <= 1) return false;
  const lower = cleaned.toLowerCase();
  if (BLACKLIST.has(lower)) return false;
  if (/^[0-9]+$/.test(cleaned)) return false;
  if (/^[^a-zA-Z0-9\u4e00-\u9fa5]+$/.test(cleaned)) return false;
  if (/^[a-z]+$/i.test(cleaned) && cleaned.length <= 3) return false; // 纯英文缩写
  return true;
}

async function extractConceptsWithLLM(note) {
  const conceptContent = `${note.title || ""}\n\n${(note.content || "").slice(0, MAX_CONTENT_CHARS)}`.trim();
  if (!conceptContent) return [];
  const prompt = `你是一个帮助构建我的自我洞察图谱（SIG）的概念抽取器。
请根据标题与摘要，提取 1-3 个概念。
要求：
- 抽象层次高
- 中文优先，不超过8字
- 不输出缩写（mj/ai/ll 等）
- 不输出乱码/符号
- 不输出事件描述，只输出“概念名称”
输出 JSON 数组：
[
  {"name":"xxx","type":"topic"},
  {"name":"yyy","type":"idea"}
]`;

  try {
    const messages = [
      { role: "system", content: prompt },
      { role: "user", content: conceptContent },
    ];

    let reply;
    try {
      reply = await ollamaChat(OLLAMA_MODEL, messages, OLLAMA_TIMEOUT_MS);
    } catch (err) {
      console.warn(
        `🔁 Ollama retry for note ${note.id} (concept) due to: ${err?.name || ""} ${err?.message || err} (len=${conceptContent.length}, timeout=${OLLAMA_TIMEOUT_MS}ms, model=${OLLAMA_MODEL})`
      );
      reply = await ollamaChat(OLLAMA_MODEL, messages, OLLAMA_RETRY_MS);
    }
    const match = reply.match(/\[[\s\S]*?\]/);
    const jsonStr = match ? match[0] : reply;
    const parsed = JSON.parse(jsonStr);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .map((c) => ({
        name: (c?.name || "").trim(),
        type: (c?.type || "topic").toLowerCase(),
        source: "llm",
        score: 0.7,
      }))
      .filter(filterLLMConcept);
  } catch (err) {
    console.error(`LLM extract failed for note ${note.id}:`, err?.message || err);
    if (OLLAMA_LOG_CONTENT) {
      console.error(`[conceptContent] ${conceptContent.slice(0, 200)}${conceptContent.length > 200 ? "..." : ""}`);
    }
    if (err?.responseText) {
      console.error(`[llm raw reply] ${String(err.responseText).slice(0, 200)}`);
    }
    return [];
  }
}

async function buildLLMConcepts_v4(notes) {
  const results = [];
  let logged = 0;
  for (const n of notes) {
    const concepts = await extractConceptsWithLLM(n);
    if (OLLAMA_LOG_SAMPLES > 0 && logged < OLLAMA_LOG_SAMPLES && concepts.length > 0) {
      console.log(
        `🧠 LLM concepts sample [${logged + 1}/${OLLAMA_LOG_SAMPLES}] note=${n.id} title="${(n.title || "").slice(0, 40)}":`,
        concepts.map((c) => `${c.name}(${c.type})`).join(", ")
      );
      logged++;
    }
    results.push({ noteId: n.id, concepts });
  }
  return results;
}

// ============= Concept 基础（Tag/Cluster） =============
// Normalize for lexical checks.
function normalizeConceptName(name) {
  if (!name) return "";
  return name
    .toLowerCase()
    .replace(/[\s.,;:，。；：·]/g, "")
    .replace(/[()（）[\]{}]/g, "")
    .normalize("NFKC");
}

// Basic edit distance.
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

// Loose tokenization for Jaccard.
function tokenize(str) {
  return str
    .toLowerCase()
    .split(/[^a-z0-9\u4e00-\u9fa5]+/i)
    .filter(Boolean);
}

// Token Jaccard similarity.
function jaccardSim(a, b) {
  const ta = new Set(tokenize(a));
  const tb = new Set(tokenize(b));
  const inter = new Set([...ta].filter((x) => tb.has(x)));
  const union = new Set([...ta, ...tb]);
  if (union.size === 0) return 0;
  return inter.size / union.size;
}

// Combined lexical check (edit/Jaccard/substring).
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

// Cosine similarity helper.
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

// Map tag to concept tuple.
function normalizeConcept(tag) {
  if (!tag) return null;
  const parts = tag.split("/");
  const head = parts[0]?.toLowerCase() || "";
  const name = parts.length > 1 ? parts.slice(1).join("/") : tag;
  const type = TYPE_MAP[head] || "topic";
  return { name: name || tag, type, source: "tag", score: 1.0 };
}

// Build tag+top_tag derived concepts per note.
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

// ============= 图构建（与 v3 相同） =============
// Build SEMANTIC edges via cosine similarity.
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

// Emit Tag nodes and HAS_TAG edges.
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

// Derive cluster-level concepts from connected components.
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

// Flatten three sources into a unified list.
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

// Pick canonical type using priority.
function pickType(types) {
  const byPriority = [...TYPE_PRIORITY, "topic"];
  for (const t of byPriority) {
    if (types.has(t)) return t;
  }
  return types.values().next().value || "topic";
}

// Pick canonical name using source priority.
function pickName(namesBySource) {
  for (const src of NAME_PRIORITY) {
    if (namesBySource[src]) return namesBySource[src][0];
  }
  const any = Object.values(namesBySource).flat();
  return any[0] || "";
}

// Merge raw concepts (lexical+semantic), fuse attributes, produce canonical + per-note evidence.
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

// Embed all notes in batches.
async function embedAll_v2(notes) {
  await initEmbedder();
  const vectors = new Array(notes.length);
  const t0 = Date.now();

  for (let i = 0; i < notes.length; i += BATCH) {
    const b0 = Date.now();
    const batch = notes.slice(i, i + BATCH);
    const embedPromises = batch.map((n) => createEmbedding_v2(buildEmbeddingInput(n)));
    const batchVecs = await Promise.all(embedPromises);
    for (let j = 0; j < batchVecs.length; j++) {
      vectors[i + j] = batchVecs[j];
    }
    const done = Math.min(i + BATCH, notes.length);
    const bElapsed = ((Date.now() - b0) / 1000).toFixed(2);
    const totalElapsed = ((Date.now() - t0) / 1000).toFixed(2);
    log(`🔧 Embedding batch: ${done} / ${notes.length} (batch ${bElapsed}s, total ${totalElapsed}s)`);
    if (EMB_LOG_TIMING && i === 0) {
      const est = (Number(bElapsed) * (notes.length / done)).toFixed(2);
      log(`⏱️  Estimated embedding time ~${est}s based on first batch`);
    }
  }
  return vectors;
}

// Main pipeline: fetch -> summarize -> embed -> edges -> concepts -> write artifacts.
async function main() {
  log("🚀 export-graph-v4 start");
  await ensureDir(OUTPUT_DIR);

  const db = createDb(getDbPath());
  const notes = await fetchAllNotes_v2(db);
  log(`📥 Notes fetched: ${notes.length}`);
  await attachSummaries(notes);
  log(`🧾 Summaries attached (source llm/local mixed)`);
  log(`📒 Total notes: ${notes.length}`);

  const vectors = await embedAll_v2(notes);
  log("🧠 Embeddings completed");
  const ids = notes.map((n) => n.id);
  const semEdges = buildSemanticEdges_v2(vectors, ids);
  log(`🌐 Semantic edges built: ${semEdges.length}`);
  const { tagNodes, tagEdges } = buildTagEdges(notes);
  log(`🏷️  Tag nodes: ${tagNodes.length}, tag edges: ${tagEdges.length}`);

  const noteNodes = notes.map((n) => ({
    id: n.id,
    type: "Note",
    title: n.title || "",
    summary: n.summary || "",
    summary_source: n.summary_source || "local",
    summary_model: n.summary_model || "none",
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
    summary_source: n.summary_source || "local",
    summary_model: n.summary_model || "none",
    top_tag: (n.tags && n.tags[0]) || null,
  }));

  const tagConcepts = buildTagConcepts(notes);
  log("🔖 Tag concepts built");
  const llmConcepts = await buildLLMConcepts_v4(notes); // 新的 LLM 概念来源
  log("🤖 LLM concepts built");
  const clusterConcepts = buildClusterConcepts(notes, semEdges);
  log("🧩 Cluster concepts built");

  const rawConcepts = flattenRawConcepts(tagConcepts, llmConcepts, clusterConcepts);
  const merged = await mergeConcepts(rawConcepts);
  log(`🌀 Concepts merged: canonical=${merged.canonicalConcepts.length}, noteConcepts=${merged.noteConcepts.length}`);

  const conceptsPayload = {
    generated_at: new Date().toISOString(),
    params: {
      lexical: { LEX_EDIT_TH, LEX_JACCARD_TH },
      semantic: { SEM_CONCEPT_TH },
      score_bonus_per_source: SCORE_BONUS_PER_SOURCE,
      score_bonus_cap: SCORE_BONUS_CAP,
      ollama: { url: OLLAMA_URL, model: OLLAMA_MODEL, timeout_ms: OLLAMA_TIMEOUT_MS },
    },
    noteConcepts: merged.noteConcepts,
    canonicalConcepts: merged.canonicalConcepts,
  };

  await writeJson(path.join(OUTPUT_DIR, "graph.json"), graph);
  await fs.writeFile(path.join(OUTPUT_DIR, "embeddings.json"), JSON.stringify(vectors), "utf8");
  await writeJson(path.join(OUTPUT_DIR, "meta.json"), meta);
  await writeJson(CONCEPT_PATH, conceptsPayload);

  log("✅ export-graph-v4 complete");
  log(`   graph.json -> ${path.join(OUTPUT_DIR, "graph.json")}`);
  log(`   embeddings.json -> ${path.join(OUTPUT_DIR, "embeddings.json")}`);
  log(`   meta.json -> ${path.join(OUTPUT_DIR, "meta.json")}`);
  log(`   concepts.json -> ${CONCEPT_PATH}`);
  log(`   Summary cache -> ${SUMMARY_CACHE}`);

  db.close();
}

main().catch((err) => {
  console.error("❌ export-graph-v4 failed:", err);
  process.exit(1);
});
// Preferred summary via Ollama; returns null on failure.
async function summarizeWithLLM(note) {
  if (OLLAMA_SUMMARY_DISABLE) return null;
  const contentClean = cleanContent((note.content || "").slice(0, SUMMARY_CONTENT_LIMIT));
  if (contentClean.length < SUMMARY_MIN_LLM_CHARS) {
    debugLog(`↩️  skip LLM (short content) note=${note.id} len=${contentClean.length}`);
    return null;
  }
  const title = note.title || "";
  const prompt = `你是一个专业的语义压缩助手。请将以下内容总结为不超过 120 字的中文摘要。

要求：
- 不加入主观推理
- 不生成新观点
- 不提问
- 仅保留核心语义
- 输出1行，不要换行

【内容】
${title ? `[标题] ${title}\n` : ""}${contentClean}
`;

  const messages = [
    { role: "system", content: prompt },
    { role: "user", content: contentClean || title },
  ];

  const model = OLLAMA_SUMMARY_MODEL_ENV || OLLAMA_MODEL;
  try {
    let reply;
    try {
      reply = await ollamaChat(model, messages, OLLAMA_TIMEOUT_MS, {
        stream: OLLAMA_SUMMARY_STREAM,
        numPredict: OLLAMA_SUMMARY_NUM_PREDICT,
      });
    } catch (err) {
      console.warn(
        `🔁 Ollama summary retry for note ${note.id} "${(title || "").slice(0, 30)}": ${err?.message || err} (len=${contentClean.length}, timeout=${OLLAMA_TIMEOUT_MS}ms, model=${model}, stream=${OLLAMA_SUMMARY_STREAM}, num_predict=${OLLAMA_SUMMARY_NUM_PREDICT})`
      );
      reply = await ollamaChat(model, messages, OLLAMA_RETRY_MS, {
        stream: OLLAMA_SUMMARY_STREAM,
        numPredict: OLLAMA_SUMMARY_NUM_PREDICT,
      });
    }
    const text = (reply || "").trim();
    if (validSummaryText(text)) {
      debugLog(`✅ LLM summary ok note=${note.id} model=${model} len=${text.length}`);
      return { summary: text, source: "llm", model };
    }
    console.warn(`⚠️ LLM summary rejected note ${note.id}: invalid output len=${text.length}`);
  } catch (err) {
    console.error(`LLM summary failed for note ${note.id}:`, err?.message || err);
    debugLog(`[summary debug] title="${title.slice(0, 80)}" content_len=${contentClean.length}`);
  }
  return null;
}

// Local fallback summary based on sentence similarity to title.
async function summarizeLocal(note) {
  const contentClean = cleanContent(note.content || "");
  const sentences = splitSentences(contentClean).filter((s) => s.length >= 4 && !(/^[a-z]{1,4}$/i.test(s)));
  if (sentences.length === 0) {
    const fallback = contentClean.slice(0, 180);
    return { summary: fallback, source: "local", model: "none" };
  }

  await initEmbedder();
  const titleText = note.title || "";
  const titleEmb = await createEmbedding_v2(titleText || sentences[0]);
  const scored = [];
  for (const s of sentences) {
    const emb = await createEmbedding_v2(s);
    scored.push({ s, score: cosineSim(titleEmb, emb) });
  }
  scored.sort((a, b) => b.score - a.score);
  const picked = scored.slice(0, 3).map((x) => x.s).join(" ").slice(0, 180);
  const finalText = picked || contentClean.slice(0, 180);
  return { summary: finalText, source: "local", model: "none" };
}

// Hybrid summary orchestrator.
async function buildSummary(note) {
  const llmRes = await summarizeWithLLM(note);
  if (llmRes) return llmRes;
  return summarizeLocal(note);
}
