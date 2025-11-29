#!/usr/bin/env node
// 对比整篇向量与分块向量的召回效果。
// 用法：
//   node scripts/chunk/compare-recall.js "query text" \
//     --base-prefix ./src/note_vectors              # 非 chunk 索引
//     --chunk-prefix ./src/note_vectors_chunk       # chunk 索引
//
// If prefixes are omitted:
//   base  -> ./src/note_vectors
//   chunk -> ./src/note_vectors_chunk

import fs from "fs/promises";
import path from "path";
import faissNode from "faiss-node";
import { createEmbedding } from "../../src/utils.js";

const { IndexFlatL2 } = faissNode;

function parseArgs() {
  const args = process.argv.slice(2);
  const opts = { query: "", basePrefix: "./src/note_vectors", chunkPrefix: "./src/note_vectors_chunk", k: 5 };
  const rest = [];
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === "--base-prefix" && args[i + 1]) { opts.basePrefix = args[++i]; continue; }
    if (a === "--chunk-prefix" && args[i + 1]) { opts.chunkPrefix = args[++i]; continue; }
    if (a === "-k" && args[i + 1]) { opts.k = Number(args[++i]); continue; }
    rest.push(a);
  }
  opts.query = rest.join(" ").trim();
  return opts;
}

async function loadIndex(prefix) {
  // Allow prefix as file prefix or directory containing {note_vectors(.index|.json)}.
  let basePrefix = prefix;
  try {
    const st = await fs.stat(prefix);
    if (st.isDirectory()) {
      // Guess file prefix inside directory
      const cand = path.join(prefix, path.basename(prefix)); // e.g., dir=src/note_vectors_chunk -> src/note_vectors_chunk/note_vectors_chunk
      const idxGuess = `${cand}.index`;
      try { await fs.access(idxGuess); basePrefix = cand; } catch {
        // fallback to note_vectors
        basePrefix = path.join(prefix, "note_vectors");
      }
    }
  } catch {
    // prefix treated as file prefix (default)
  }

  const idxPath = path.resolve(`${basePrefix}.index`);
  const mapPath = path.resolve(`${basePrefix}.json`);
  const metaPath = path.resolve(basePrefix.endsWith("/") ? `${basePrefix}meta.json` : `${basePrefix}_meta.json`);
  const metaChunksPath = path.resolve(path.join(path.dirname(basePrefix), "meta_chunks.json"));
  const index = IndexFlatL2.read(idxPath);
  const idMap = JSON.parse(await fs.readFile(mapPath, "utf8"));
  let meta = [];
  for (const p of [metaPath, metaChunksPath]) {
    try {
      const m = JSON.parse(await fs.readFile(p, "utf8"));
      if (Array.isArray(m)) meta = m;
    } catch (_) { /* ignore */ }
  }
  return { index, idMap, meta };
}

async function search(index, idMap, query, k) {
  const emb = await createEmbedding(query);
  const { labels, distances } = index.search(emb, k);
  return labels.map((l, idx) => ({
    id: idMap[l],
    score: distances[idx],
  }));
}

function previewChunk(meta, cid, maxLen = 120) {
  const m = meta.find((x) => x.id === cid || x.chunk_id === cid);
  if (!m) return "";
  const heading = m.heading || "";
  const text = (m.content || m.summary || "").replace(/\s+/g, " ").trim();
  const snippet = text.slice(0, maxLen) + (text.length > maxLen ? "..." : "");
  return heading ? `[${heading}] ${snippet}` : snippet;
}

async function main() {
  const { query, basePrefix, chunkPrefix, k } = parseArgs();
  if (!query) {
    console.error("用法: node scripts/chunk/compare-recall.js \"your query\" [--base-prefix path] [--chunk-prefix path] [-k 5]");
    process.exit(1);
  }

  console.log(`🔍 Query: ${query}`);
  console.log(`   base index:  ${basePrefix}.index`);
  console.log(`   chunk index: ${chunkPrefix}.index`);

  const base = await loadIndex(basePrefix);
  const chunk = await loadIndex(chunkPrefix);

  const baseHits = await search(base.index, base.idMap, query, k);
  const chunkHits = await search(chunk.index, chunk.idMap, query, k);

  console.log("\nNon-chunk top hits (note_id):");
  baseHits.forEach((h, i) => console.log(`${i + 1}. ${h.id} (score=${h.score.toFixed(4)})`));

  console.log("\nChunked top hits (chunk_id + preview):");
  chunkHits.forEach((h, i) => {
    const preview = previewChunk(chunk.meta, h.id);
    console.log(`${i + 1}. ${h.id} (score=${h.score.toFixed(4)}) ${preview ? "-> " + preview : ""}`);
  });
}

main().catch((err) => {
  console.error("compare-recall failed:", err);
  process.exit(1);
});
