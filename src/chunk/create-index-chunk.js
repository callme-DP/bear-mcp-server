#!/usr/bin/env node
// 构建 Bear 笔记的分块向量索引（FAISS）。
// - 基于标题感知的 chunker（窗口 + 重叠）
// - 输出：note_vectors_chunk.index/json + meta_chunks.json + chunk_embeddings.json

import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";
import faissNode from "faiss-node";
import { getDbPath, createDb, initEmbedder, createEmbedding } from "../utils.js";
import { chunkNote } from "./chunker.js";

const { IndexFlatL2 } = faissNode;
const DIMENSION = 384; // all-MiniLM-L6-v2

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// 简易 .env 解析（无第三方依赖）
function loadEnv(envPath = ".env") {
  const full = path.resolve(process.cwd(), envPath);
  return fs
    .readFile(full, "utf8")
    .then((txt) => {
      txt
        .split(/\r?\n/)
        .map((l) => l.trim())
        .filter((l) => l && !l.startsWith("#"))
        .forEach((line) => {
          const eq = line.indexOf("=");
          if (eq > 0) {
            const k = line.slice(0, eq).trim();
            const v = line.slice(eq + 1).trim();
            if (!process.env[k]) process.env[k] = v;
          }
        });
    })
    .catch(() => {});
}

function getConfig() {
  // 默认输出到项目根的 src/note_vectors_chunk
  const outputDir = process.env.NOTE_VECTORS_DIR
    ? path.resolve(process.env.NOTE_VECTORS_DIR)
    : path.resolve(process.cwd(), "src/note_vectors_chunk");
  const indexPath = path.join(outputDir, "note_vectors_chunk");
  const windowTokens = Number(process.env.CHUNK_WINDOW || 500);
  const overlapTokens = Number(process.env.CHUNK_OVERLAP || 80);
  return { outputDir, indexPath, windowTokens, overlapTokens };
}

async function ensureDir(dir) {
  await fs.mkdir(dir, { recursive: true });
}

async function fetchNotes(db) {
  return db.allAsync(`
    SELECT 
      ZUNIQUEIDENTIFIER as id,
      ZTITLE as title,
      ZTEXT as content
    FROM ZSFNOTE
    WHERE ZTRASHED = 0
  `);
}

async function main() {
  // 先尝试加载 .env（如果存在）
  await loadEnv();

  const { outputDir, indexPath, windowTokens, overlapTokens } = getConfig();
  console.log("🚀 create-index-chunk start");
  await ensureDir(outputDir);

  const modelReady = await initEmbedder();
  if (!modelReady) {
    console.error("Failed to initialize embedding model");
    process.exit(1);
  }

  const db = createDb(getDbPath());
  try {
    const notes = await fetchNotes(db);
    const dbPath = getDbPath();
    console.log(`📒 Notes to index: ${notes.length}`);
    console.log(`📂 Bear DB: ${dbPath}`);

    // Chunk all notes
    const chunks = [];
    const chunkedNoteSet = new Set();
    for (const n of notes) {
      const cs = chunkNote(n, { windowTokens, overlapTokens });
      if (cs.length === 0) continue;
      chunks.push(...cs);
      chunkedNoteSet.add(n.id);
    }
    const skippedNotes = notes.length - chunkedNoteSet.size;
    console.log(`🧩 Total chunks: ${chunks.length}`);
    console.log(`🧾 Notes with chunks: ${chunkedNoteSet.size} / ${notes.length} (skipped=${skippedNotes})`);
    if (skippedNotes > 0) {
      console.log("⚠️ 部分笔记未生成 chunk，可能因内容过短或清洗后为空。");
    }
    if (chunks.length === 0) {
      console.error("No chunks generated; aborting.");
      process.exit(1);
    }

    const index = new IndexFlatL2(DIMENSION);
    const idMap = {};
    const meta = [];
    const embeddings = [];
    let row = 0;

    for (const c of chunks) {
      try {
        const emb = await createEmbedding(c.content);
        index.add(emb);
        embeddings.push(emb);
        idMap[row] = c.chunk_id;
        meta.push({
          chunk_id: c.chunk_id,
          note_id: c.note_id,
          heading: c.heading,
          order: c.order,
          content: c.content,
        });
        row += 1;
        if (row % 200 === 0) {
          process.stdout.write(`\r🔧 Embedded ${row}/${chunks.length}`);
        }
      } catch (err) {
        console.error(`Embedding failed for ${c.chunk_id}:`, err?.message || err);
      }
    }
    process.stdout.write("\r");
    console.log(`✅ Embeddings done: ${row} rows`);

    index.write(`${indexPath}.index`);
    await fs.writeFile(`${indexPath}.json`, JSON.stringify(idMap));
    await fs.writeFile(path.join(outputDir, "meta_chunks.json"), JSON.stringify(meta, null, 2));
    await fs.writeFile(path.join(outputDir, "chunk_embeddings.json"), JSON.stringify(embeddings));

    console.log("💾 Saved:");
    console.log(`   ${indexPath}.index`);
    console.log(`   ${indexPath}.json`);
    console.log(`   ${path.join(outputDir, "meta_chunks.json")}`);
    console.log(`   ${path.join(outputDir, "chunk_embeddings.json")}`);
    console.log(`   window=${windowTokens}, overlap=${overlapTokens}`);
  } catch (err) {
    console.error("Indexing failed:", err);
    process.exit(1);
  } finally {
    db.close();
  }
}

main().catch((err) => {
  console.error("create-index-chunk crashed:", err);
  process.exit(1);
});
