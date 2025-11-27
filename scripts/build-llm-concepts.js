#!/usr/bin/env node
// Batch generate llm_concepts.txt (and failed list) for offline/Metabase usage.
// Note: This is a standalone batch; not wired into graph export/import yet.

import path from "path";
import { fileURLToPath } from "url";
import fs from "fs";
import fetch from "node-fetch";
import { createDb, getDbPath } from "../src/utils.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const SAMPLE_SIZE = Number(process.env.LLM_NOTE_LIMIT || process.env.SUMMARY_SAMPLE || 50);
const MAX_CONTENT_CHARS = Number(process.env.MAX_CONTENT_CHARS || 3000);
const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434/api/chat";
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || "qwen2.5:7b";
const OLLAMA_TIMEOUT_MS = Number(process.env.OLLAMA_TIMEOUT_MS || 8000);
const OLLAMA_RETRY_MS = Number(process.env.OLLAMA_RETRY_MS || 12000);
const OLLAMA_STREAM = ["1", "true", "yes"].includes(String(process.env.OLLAMA_STREAM || "false").toLowerCase()) ? true : false;
const OLLAMA_NUM_PREDICT = Number(process.env.OLLAMA_NUM_PREDICT || 64);
const OUTPUT_FILE = process.env.CONCEPTS_OUT || path.join(__dirname, "..", "out", "llm_concepts.txt");
const FAILED_IDS_FILE = process.env.CONCEPTS_FAILED_OUT || path.join(__dirname, "..", "out", "llm_concepts_failed.txt");
const RERUN_FAILED_FROM = process.env.RERUN_FAILED_FROM || process.env.CONCEPTS_FAILED_IN;

function log(msg) {
  console.log(msg);
}

function cleanContent(text) {
  if (!text) return "";
  return text
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/`[^`]*`/g, " ")
    .replace(/https?:\/\/\S+/g, " ")
    .replace(/\!\[[^\]]*\]\([^\)]*\)/g, " ")
    .replace(/\[[^\]]*\]\([^\)]*\)/g, " ")
    .replace(/[#>*_~\\-]{1,}/g, " ")
    .replace(/[^\w\s\u4e00-\u9fa5.,，。！？!?\n]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

async function fetchRecentNotes(limit) {
  const db = createDb(getDbPath());
  try {
    const rows = await db.allAsync(
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
    return rows.map((n) => ({
      id: n.id,
      title: n.title || "",
      content: n.content || "",
    }));
  } finally {
    db.close();
  }
}

async function fetchNotesByIds(ids) {
  if (!ids?.length) return [];
  const unique = [...new Set(ids.filter(Boolean))];
  if (!unique.length) return [];
  const placeholders = unique.map(() => "?").join(",");
  const db = createDb(getDbPath());
  try {
    const rows = await db.allAsync(
      `
      SELECT
        ZUNIQUEIDENTIFIER AS id,
        ZTITLE AS title,
        ZTEXT AS content,
        ZMODIFICATIONDATE AS modification_date
      FROM ZSFNOTE
      WHERE ZTRASHED = 0
        AND ZUNIQUEIDENTIFIER IN (${placeholders})
    `,
      unique
    );
    return rows.map((n) => ({
      id: n.id,
      title: n.title || "",
      content: n.content || "",
    }));
  } finally {
    db.close();
  }
}

function loadFailedIds(filePath) {
  try {
    const raw = fs.readFileSync(filePath, "utf8");
    return raw
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);
  } catch (err) {
    log(`⚠️ Could not read failed IDs from ${filePath}: ${err?.message || err}`);
    return [];
  }
}

async function ollamaChat(model, messages, timeoutMs) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(OLLAMA_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        messages,
        stream: OLLAMA_STREAM,
        num_predict: OLLAMA_NUM_PREDICT,
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

async function extractConceptsWithLLM(note) {
  const conceptContent = `${note.title || ""}\n\n${(note.content || "").slice(0, MAX_CONTENT_CHARS)}`.trim();
  if (!conceptContent) return { status: "empty" };
  const prompt = `你是一个帮助构建自我洞察图谱（SIG）的概念抽取器。
请根据标题与正文摘取 1-3 个概念，返回 JSON 数组：
[
  {"name":"xxx","type":"topic"},
  {"name":"yyy","type":"idea"}
]`;

  const messages = [
    { role: "system", content: prompt },
    { role: "user", content: conceptContent },
  ];

  try {
    let reply;
    try {
      reply = await ollamaChat(OLLAMA_MODEL, messages, OLLAMA_TIMEOUT_MS);
    } catch (err) {
      log(`🔁 Retry note=${note.id} due to: ${err?.message || err}`);
      reply = await ollamaChat(OLLAMA_MODEL, messages, OLLAMA_RETRY_MS);
    }
    const match = reply.match(/\[[\s\S]*?\]/);
    const jsonStr = match ? match[0] : reply;
    const parsed = JSON.parse(jsonStr);
    if (!Array.isArray(parsed)) return { status: "invalid", raw: reply?.slice(0, 200) || "" };
    return {
      status: "ok",
      concepts: parsed.map((c) => ({
        name: (c?.name || "").trim(),
        type: (c?.type || "topic").toLowerCase(),
      })),
    };
  } catch (err) {
    return { status: "error", error: err?.message || String(err) };
  }
}

async function main() {
  log("🚀 build-llm-concepts start");
  let notes;
  if (RERUN_FAILED_FROM) {
    const ids = loadFailedIds(RERUN_FAILED_FROM);
    notes = await fetchNotesByIds(ids);
    log(`📒 Rerun failed IDs mode: loaded ${ids.length} ids, fetched ${notes.length} notes`);
  } else {
    notes = await fetchRecentNotes(SAMPLE_SIZE);
    log(`📒 Fetched recent notes: ${notes.length} (limit=${SAMPLE_SIZE})`);
  }

  let ok = 0;
  let error = 0;
  let invalid = 0;
  const failedIds = [];
  const rows = [];
  for (let i = 0; i < notes.length; i++) {
    const n = notes[i];
    const cleaned = cleanContent(n.content);
    const note = { ...n, content: cleaned };

    const res = await extractConceptsWithLLM(note);
    if (res.status === "ok" && res.concepts?.length) {
      rows.push(`${note.id}\t${JSON.stringify(res.concepts)}`);
      ok++;
    } else if (res.status === "invalid") {
      rows.push(`${note.id}\t[]`);
      invalid++;
    } else {
      failedIds.push(note.id);
      error++;
    }

    if ((i + 1) % 10 === 0) {
      log(`... processed ${i + 1}/${notes.length} (ok=${ok}, invalid=${invalid}, error=${error})`);
    }
  }

  fs.mkdirSync(path.join(__dirname, "..", "out"), { recursive: true });
  fs.writeFileSync(OUTPUT_FILE, rows.join("\n"), "utf8");
  fs.writeFileSync(FAILED_IDS_FILE, failedIds.join("\n"), "utf8");

  log(`✅ Done. ok=${ok}, invalid=${invalid}, error=${error}`);
  log(`   concepts -> ${OUTPUT_FILE}`);
  log(`   failed ids -> ${FAILED_IDS_FILE}`);
}

main().catch((err) => {
  console.error("❌ build-llm-concepts failed:", err);
  process.exit(1);
});
