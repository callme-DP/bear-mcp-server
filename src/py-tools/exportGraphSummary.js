// src/export-graph.js
// 导出全量知识星图 + 语义向量与元数据（含摘要）

import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";
import cliProgress from "cli-progress";
import pLimit from "p-limit";
import {
  createDb,
  getDbPath,
  initEmbedder,
  createEmbedding,
} from "../utils.js";
// 🟢 新增：导入 summarization 模型
import { pipeline, env } from "@xenova/transformers";
const bar = new cliProgress.SingleBar({}, cliProgress.Presets.shades_classic);

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ===== 可调参数 =====
// Allow override; default writes to consolidated data/neo4j.
const OUTPUT_DIR = process.env.NOTE_VECTORS_DIR || path.resolve(__dirname, "../data/neo4j");
const MODEL_DIR = path.resolve(__dirname, "../../models/distilbart-cnn-6-6");

// const OUTPUT_DIR = path.join(__dirname, "exports");
const TOPK = Number(process.env.TOPK || 5);
const COS_THRESHOLD = Number(process.env.TH || 0.58);
const MAX_CONTENT_CHARS = 3000;
const BATCH = 32;

// Apple 2001 时间基准 → UNIX
const APPLE_EPOCH_OFFSET = 978307200;

// 关键环境：只用本地，不走网络；把缓存目录指到你的 models/
env.allowLocalModels = true;
env.allowRemoteModels = false;
// 让模型生成到 node_modules/@xenova/transformers/.cache
env.cacheDir = path.resolve(
  __dirname,
  "../../node_modules/@xenova/transformers/models"
);
// env.cacheDir = path.resolve(__dirname, "../../models"); // 让它在此目录下查找/缓存

let summarizer = null;
async function initSummarizer() {
  console.log("🛠️ 准备加载 summarizer...", summarizer);

  // const modelFile = path.resolve(MODEL_DIR, "onnx/decoder_model.onnx");
  // const tokenizerFile = path.resolve(MODEL_DIR, "tokenizer.json");
  // const configFile = path.resolve(MODEL_DIR, "config.json");

  // console.log("📂 模型文件:", modelFile);
  // console.log("📂 tokenizer:", tokenizerFile);
  // console.log("📂 config:", configFile);

  if (!summarizer) {
    console.log("🧠 正在加载本地摘要模型 ...");
    summarizer = await pipeline("summarization", "Xenova/distilbart-cnn-6-6", {
      quantized: true,
      device: "cpu",
    });
    console.log("✅ 模型加载成功");
  }
  return summarizer;
}
async function ensureDir(dir) {
  try {
    await fs.mkdir(dir, { recursive: true });
  } catch {}
}

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

  const tagRows = await db.allAsync(`
    SELECT 
      ZN.ZUNIQUEIDENTIFIER AS id,
      ZT.ZTITLE AS tag
    FROM Z_5TAGS ZNT
    JOIN ZSFNOTETAG ZT ON ZT.Z_PK = ZNT.Z_13TAGS
    JOIN ZSFNOTE ZN     ON ZN.Z_PK = ZNT.Z_5NOTES
  `);

  const tagMap = new Map();
  for (const r of tagRows) {
    if (!tagMap.has(r.id)) tagMap.set(r.id, new Set());
    tagMap.get(r.id).add(r.tag);
  }

  for (const n of rows) {
    n.tags = Array.from(tagMap.get(n.id) || []);
    if (n.creation_date) {
      n.creation_date = new Date(
        (n.creation_date + APPLE_EPOCH_OFFSET) * 1000
      ).toISOString();
    }
    if (n.modification_date) {
      n.modification_date = new Date(
        (n.modification_date + APPLE_EPOCH_OFFSET) * 1000
      ).toISOString();
    }
  }

  return rows;
}

async function embedAll(notes) {
  await initEmbedder();
  const vectors = new Array(notes.length);

  for (let i = 0; i < notes.length; i += BATCH) {
    const batch = notes.slice(i, i + BATCH);
    const texts = batch.map((n) => {
      const t =
        n.content && n.content.trim().length > 0
          ? n.content.slice(0, MAX_CONTENT_CHARS)
          : n.title || "";
      return t || " ";
    });

    for (let j = 0; j < texts.length; j++) {
      const vec = await createEmbedding(texts[j]);
      vectors[i + j] = vec;
    }
    process.stdout.write(
      `\r🔧 Embedding: ${Math.min(i + BATCH, notes.length)} / ${notes.length}`
    );
  }
  process.stdout.write("\n");
  return vectors;
}

function buildSemanticEdges(vectors, ids, topk = TOPK, th = COS_THRESHOLD) {
  const edges = [];
  const dim = vectors[0]?.length || 0;
  if (!dim) return edges;

  for (let i = 0; i < vectors.length; i++) {
    const vi = vectors[i];
    const scores = [];
    for (let j = 0; j < vectors.length; j++) {
      if (i === j) continue;
      const vj = vectors[j];
      let dot = 0;
      for (let k = 0; k < dim; k++) dot += vi[k] * vj[k];
      scores.push([j, dot]);
    }
    scores.sort((a, b) => b[1] - a[1]);
    let added = 0;
    for (const [j, s] of scores) {
      if (s < th) break;
      edges.push({
        from: ids[i],
        to: ids[j],
        type: "semantic",
        weight: Number(s.toFixed(4)),
      });
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
  for (const t of tagSet) nodes.push({ id: `tag:${t}`, type: "Tag", name: t });
  for (const n of notes) {
    for (const t of n.tags || []) {
      edges.push({ from: n.id, to: `tag:${t}`, type: "has_tag", weight: 1 });
    }
  }
  return { tagNodes: nodes, tagEdges: edges };
}

// 🟢 新增：摘要生成

/**
 * 预处理笔记文本内容
 * @param {Object} note - 笔记对象
 * @returns {string} 处理后的文本内容，如果无内容则返回空字符串
 */
function preprocessNoteText(note) {
  const text =
    note.content && note.content.length > 0
      ? note.content.slice(0, 1000)
      : note.title;
  return text || "";
}

/**
 * 生成单条笔记的摘要
 * @param {string} text - 要摘要的文本内容
 * @param {Function} summarizer - 摘要生成器函数
 * @returns {Promise<string>} 生成的摘要文本，失败时返回空字符串
 */
async function generateNoteSummary(text, summarizer) {
  try {
    const res = await summarizer(text, { min_length: 10, max_length: 50 });
    return res[0].summary_text;
  } catch (err) {
    console.warn(`⚠️ 摘要失败: ${text.slice(0, 50)}...`);
    return "";
  }
}

/**
 * 处理单条笔记的摘要生成
 * @param {Object} note - 笔记对象
 * @param {Function} summarizer - 摘要生成器函数
 * @returns {Promise<void>}
 */
async function processNoteSummary(note, summarizer) {
  const text = preprocessNoteText(note);
  if (!text) {
    note.summary = "";
    return;
  }
  note.summary = await generateNoteSummary(text, summarizer);
}

const PARTIAL_FILE = path.resolve("src/note_vectors/meta_summary_part.json");

// 并发限制
const limit = pLimit(4);

// 断点续跑：加载已完成部分结果
async function loadCompletedSummaries() {
  try {
    const data = await fs.readFile(PARTIAL_FILE, "utf8");
    return JSON.parse(data);
  } catch {
    return {};
  }
}

// 保存部分结果（每100条）
async function savePartialSummaries(resultMap) {
  await fs.appendFile(PARTIAL_FILE, JSON.stringify(resultMap, null, 2), "utf8");
}

export async function summarizeNotes(notes, summarizer) {
  console.log(`🧠 开始生成摘要，总笔记数：${notes.length}`);

  const completed = await loadCompletedSummaries();
  const resultMap = { ...completed };

  // 过滤掉已完成的
  const pendingNotes = notes.filter((n) => !resultMap[n.id]);
  console.log(
    `⏩ 已完成 ${Object.keys(completed).length} 条，待处理 ${
      pendingNotes.length
    } 条`
  );

  if (pendingNotes.length === 0) {
    console.log("✅ 所有摘要已存在，跳过生成。");
    return Object.values(resultMap);
  }

  bar.start(notes.length, Object.keys(completed).length);
  let processed = 0;

  // 使用受控并发执行摘要
  const tasks = pendingNotes.map((n, i) =>
    limit(async () => {
      const text =
        n.content && n.content.length > 0 ? n.content.slice(0, 1000) : n.title;
      if (!text) {
        resultMap[n.id] = { id: n.id, title: n.title, summary: "" };
        bar.update(++processed + Object.keys(completed).length);
        return;
      }

      try {
        const res = await summarizer(text, { min_length: 10, max_length: 50 });
        resultMap[n.id] = {
          id: n.id,
          title: n.title,
          summary: res?.[0]?.summary_text || "",
        };
      } catch (err) {
        console.warn(`⚠️ 摘要失败: ${n.title}`);
        resultMap[n.id] = { id: n.id, title: n.title, summary: "" };
      }

      processed++;
      bar.update(processed + Object.keys(completed).length);

      // 每100条保存一次
      if (processed % 10 === 0) {
        await savePartialSummaries(resultMap);
        console.log(`💾 已保存 ${processed} 条中间结果`);
      }
    })
  );

  await Promise.all(tasks);
  bar.stop();

  // 保存最终结果
  await savePartialSummaries(resultMap);
  console.log("✅ 摘要生成完毕，已保存到：", PARTIAL_FILE);

  return Object.values(resultMap);
}

async function main() {
  console.log("🚀 main() 已启动");

  await ensureDir(OUTPUT_DIR);
  // 🧠 1️⃣ 初始化摘要模型
  const summarizer = await initSummarizer();

  // 📚 2️⃣ 加载笔记内容
  const db = createDb(getDbPath());
  const notes = await fetchAllNotes(db);
  console.log(`📄 笔记总数: ${notes.length}`);

  // 🧾 3️⃣ 生成摘要（自动断点续跑）
  const summaryResults = await summarizeNotes(notes, summarizer);

  // 🪶 4️⃣ 将摘要结果合并到 notes
  const summaryMap = new Map(summaryResults.map((r) => [r.id, r.summary]));
  for (const n of notes) {
    n.summary = summaryMap.get(n.id) || "";
  }

  const noteNodes = notes.map((n) => ({
    id: n.id,
    type: "Note",
    title: n.title || "",
    created: n.creation_date,
    modified: n.modification_date,
    tags: n.tags || [],
  }));

  const vectors = await embedAll(notes);
  const ids = notes.map((n) => n.id);
  const semEdges = buildSemanticEdges(vectors, ids);
  const { tagNodes, tagEdges } = buildTagEdges(notes);

  const graph = {
    generated_at: new Date().toISOString(),
    stats: {
      notes: noteNodes.length,
      tags: tagNodes.length,
      semantic_edges: semEdges.length,
      tag_edges: tagEdges.length,
    },
    nodes: [...noteNodes, ...tagNodes],
    edges: [...semEdges, ...tagEdges],
  };

  // 🟢 修改：在 meta.json 中增加 summary 字段
  await fs.writeFile(
    path.join(OUTPUT_DIR, "graph.json"),
    JSON.stringify(graph, null, 2),
    "utf8"
  );
  await fs.writeFile(
    path.join(OUTPUT_DIR, "embeddings.json"),
    JSON.stringify(vectors),
    "utf8"
  );
  await fs.writeFile(
    path.join(OUTPUT_DIR, "meta.json"),
    JSON.stringify(
      notes.map((n, idx) => ({
        idx,
        id: n.id,
        title: n.title || "",
        top_tag: (n.tags && n.tags[0]) || null,
        tags: n.tags || [],
        summary: n.summary || summaryMap.get(n.id) || "", // 🟢 新增
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

main().catch((err) => {
  console.error("❌ 导出失败：", err);
  process.exit(1);
});
