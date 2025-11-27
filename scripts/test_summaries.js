#!/usr/bin/env node
// Quick-only summary test runner: pull recent N Bear notes, run hybrid summaries (LLM + local),
// report counts and sample outputs without rebuilding embeddings/graph.

import path from "path";
import fs from "fs/promises";
import { fileURLToPath } from "url";
import { createDb, getDbPath, initEmbedder, createEmbedding } from "../src/utils.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
// Allow override; default to consolidated data/neo4j directory.
const OUTPUT_DIR = process.env.NOTE_VECTORS_DIR || path.resolve(__dirname, "../data/neo4j");
const SUMMARY_CACHE = path.join(OUTPUT_DIR, "meta_summary_part.json");
const SUMMARY_CONTENT_LIMIT = 2000;

const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434/api/chat";
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || "qwen2.5:7b";
const OLLAMA_TIMEOUT_MS = Number(process.env.OLLAMA_TIMEOUT_MS || 8000);
const OLLAMA_RETRY_MS = Number(process.env.OLLAMA_RETRY_MS || 12000);
const OLLAMA_SUMMARY_MODEL_ENV = process.env.OLLAMA_SUMMARY_MODEL;
const OLLAMA_SUMMARY_DISABLE = ["1", "true", "yes"].includes(String(process.env.OLLAMA_SUMMARY_DISABLE || "").toLowerCase());
const OLLAMA_SUMMARY_NUM_PREDICT = Number(process.env.OLLAMA_SUMMARY_NUM_PREDICT || 64);
const OLLAMA_SUMMARY_STREAM = ["1", "true", "yes"].includes(String(process.env.OLLAMA_SUMMARY_STREAM || "false").toLowerCase());
const SUMMARY_DEBUG = ["1", "true", "yes"].includes(String(process.env.SUMMARY_DEBUG || "").toLowerCase());
// 临时开关：强制跳过缓存重算摘要。用完请删除/恢复默认。
const SUMMARY_FORCE_REBUILD = ["1", "true", "yes"].includes(String(process.env.SUMMARY_FORCE_REBUILD || "").toLowerCase());
const SUMMARY_MIN_LLM_CHARS = Number(process.env.SUMMARY_MIN_LLM_CHARS || 50);
const SUMMARY_MIN_OUTPUT_LEN = Number(process.env.SUMMARY_MIN_OUTPUT_LEN || 15);

const SAMPLE_SIZE = Number(process.env.SUMMARY_SAMPLE || 50);

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

function splitSentences(text) {
  return text
    .split(/[\n。！？!?]/)
    .map((s) => s.trim())
    .filter(Boolean);
}

function validSummaryText(text) {
  if (!text) return false;
  const t = text.trim();
  if (t.length < SUMMARY_MIN_OUTPUT_LEN) return false;
  if (/^\[?\s*\]?$/.test(t)) return false;
  if (/^(null|undefined|\{\}|\[\])$/i.test(t)) return false;
  if (/^[^a-zA-Z0-9\u4e00-\u9fa5]+$/.test(t)) return false;
  return true;
}

function parseCachedSummary(entry) {
  if (!entry) return null;
  if (typeof entry === "string") return { summary: entry, source: "local", model: "none" };
  if (entry.summary) {
    return {
      summary: entry.summary,
      source: entry.source || entry.summary_source || "local",
      model: entry.model || entry.summary_model || "none",
    };
  }
  return null;
}

function debugLog(msg) {
  if (SUMMARY_DEBUG) console.log(msg);
}

async function ollamaChat(model, messages, timeoutMs, options = {}) {
  debugLog(
    `🛰️  Ollama request model=${model} stream=${options.stream ?? false} num_predict=${options.numPredict ?? "default"} timeout=${timeoutMs}ms; user_len=${(messages?.[1]?.content || "").length}`
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
        stream: options.stream ?? false,
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

async function summarizeWithLLM(note) {
  if (OLLAMA_SUMMARY_DISABLE) return { status: "llm-disabled" };
  const contentClean = cleanContent((note.content || "").slice(0, SUMMARY_CONTENT_LIMIT));
  if (contentClean.length < SUMMARY_MIN_LLM_CHARS) {
    debugLog(`↩️  skip LLM (short content) note=${note.id} len=${contentClean.length}`);
    return { status: "llm-skipped-short" };
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
      console.warn(`🔁 Ollama summary retry for note ${note.id}: ${err?.message || err}`);
      reply = await ollamaChat(model, messages, OLLAMA_RETRY_MS, {
        stream: OLLAMA_SUMMARY_STREAM,
        numPredict: OLLAMA_SUMMARY_NUM_PREDICT,
      });
    }
    const text = (reply || "").trim();
    if (validSummaryText(text)) {
      debugLog(`✅ LLM summary ok note=${note.id} model=${model} len=${text.length}`);
      return { summary: text, source: "llm", model, status: "llm-ok" };
    }
    console.warn(`⚠️ LLM summary rejected note ${note.id}: invalid output len=${text.length}`);
    return { status: "llm-rejected" };
  } catch (err) {
    console.error(`LLM summary failed for note ${note.id}:`, err?.message || err);
    debugLog(`[summary debug] title="${title.slice(0, 80)}" content_len=${contentClean.length}`);
    return { status: "llm-error", error: err?.message || String(err) };
  }
}

async function summarizeLocal(note) {
  const contentClean = cleanContent(note.content || "");
  const sentences = splitSentences(contentClean).filter((s) => s.length >= 4 && !(/^[a-z]{1,4}$/i.test(s)));
  if (sentences.length === 0) {
    const fallback = contentClean.slice(0, 180);
    return { summary: fallback, source: "local", model: "none", status: "local" };
  }

  await initEmbedder();
  const titleText = note.title || "";
  const titleEmb = await createEmbedding(titleText || sentences[0]);
  const scored = [];
  for (const s of sentences) {
    const emb = await createEmbedding(s);
    scored.push({ s, score: cosineSim(titleEmb, emb) });
  }
  scored.sort((a, b) => b.score - a.score);
  const picked = scored.slice(0, 3).map((x) => x.s).join(" ").slice(0, 180);
  const finalText = picked || contentClean.slice(0, 180);
  return { summary: finalText, source: "local", model: "none", status: "local" };
}

async function buildSummary(note) {
  const llmRes = await summarizeWithLLM(note);
  if (llmRes?.summary) return llmRes;
  const local = await summarizeLocal(note);
  local.status = llmRes?.status || local.status;
  return local;
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

async function attachSummaries(notes) {
  const cache = (await readJsonIfExists(SUMMARY_CACHE)) || {};
  let updated = false;
  for (const n of notes) {
    const cached = parseCachedSummary(cache[n.id]);
    if (cached && !SUMMARY_FORCE_REBUILD) {
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
    if (SUMMARY_DEBUG && res.status && res.status !== "llm-ok") {
      console.log(
        `ℹ️  note=${n.id} title="${(n.title || "").slice(0, 40)}" used=${res.source} reason=${res.status}`
      );
    }
  }
  if (updated) await writeJson(SUMMARY_CACHE, cache);
}

async function fetchRecentNotes(limit = 50) {
  const db = createDb(getDbPath());
  try {
    const notes = await db.allAsync(
      `
      SELECT 
        ZUNIQUEIDENTIFIER AS id,
        ZTITLE AS title,
        ZTEXT AS content,
        ZMODIFICATIONDATE AS modification_date
      FROM ZSFNOTE
      WHERE ZTRASHED = 0
      ORDER BY ZMODIFICATIONDATE DESC
      LIMIT ?
    `,
      [limit]
    );
    return notes.map((n) => ({
      id: n.id,
      title: n.title || "",
      content: n.content || "",
      modified: n.modification_date,
    }));
  } finally {
    db.close();
  }
}

function summarizeLengths(notes) {
  const lens = notes.map((n) => (n.content || "").length).sort((a, b) => a - b);
  if (lens.length === 0) return null;
  const min = lens[0];
  const max = lens[lens.length - 1];
  const avg = lens.reduce((a, b) => a + b, 0) / lens.length;
  const pct = (p) => lens[Math.floor((lens.length - 1) * p)];
  return { min, max, avg: Number(avg.toFixed(1)), p25: pct(0.25), p50: pct(0.5), p75: pct(0.75) };
}

async function main() {
  log("🚀 summary test start");
  await ensureDir(OUTPUT_DIR);

  const notes = await fetchRecentNotes(SAMPLE_SIZE);
  log(`📒 Fetched recent notes: ${notes.length}`);
  const ls = summarizeLengths(notes);
  if (ls) {
    log(`📏 Content length stats (chars): min=${ls.min} p25=${ls.p25} median=${ls.p50} p75=${ls.p75} max=${ls.max} avg=${ls.avg}`);
    log(`   LLM min threshold: ${SUMMARY_MIN_LLM_CHARS}`);
  }

  await attachSummaries(notes);

  const llmCount = notes.filter((n) => n.summary_source === "llm").length;
  const localCount = notes.filter((n) => n.summary_source === "local").length;

  log(`✅ Summary done. llm: ${llmCount}, local: ${localCount}`);

  const samples = notes.slice(0, 5);
  for (const n of samples) {
    log(`--- ${n.id} | ${n.summary_source} | ${n.summary_model || ""}`);
    log(`title: ${n.title}`);
    log(`summary: ${n.summary}`);
  }
}

main().catch((err) => {
  console.error("❌ summary test failed:", err);
  process.exit(1);
});
