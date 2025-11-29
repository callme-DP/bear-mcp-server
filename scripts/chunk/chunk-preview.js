#!/usr/bin/env node
// 查看单条 Bear 笔记的分块结果。
// 用法：
//   node scripts/chunk/chunk-preview.js --id <note_id> [--window 500] [--overlap 80] [--maxLen 160]

import { createDb, getDbPath } from "../../src/utils.js";
import { chunkPreview } from "../../src/chunk/chunker.js";

function parseArgs() {
  const args = process.argv.slice(2);
  const opts = { window: 500, overlap: 80, maxLen: 160, id: null };
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === "--id" && args[i + 1]) opts.id = args[++i];
    else if (a === "--window" && args[i + 1]) opts.window = Number(args[++i]);
    else if (a === "--overlap" && args[i + 1]) opts.overlap = Number(args[++i]);
    else if (a === "--maxLen" && args[i + 1]) opts.maxLen = Number(args[++i]);
  }
  return opts;
}

async function fetchNote(db, id) {
  const row = await db.getAsync(
    `SELECT ZUNIQUEIDENTIFIER as id, ZTITLE as title, ZTEXT as content FROM ZSFNOTE WHERE ZUNIQUEIDENTIFIER = ? LIMIT 1`,
    [id]
  );
  return row || null;
}

async function main() {
  const { id, window, overlap, maxLen } = parseArgs();
  if (!id) {
    console.error("用法: node scripts/chunk/chunk-preview.js --id <note_id> [--window 500] [--overlap 80] [--maxLen 160]");
    process.exit(1);
  }

  const db = createDb(getDbPath());
  try {
    const note = await fetchNote(db, id);
    if (!note) {
      console.error(`Note not found: ${id}`);
      process.exit(1);
    }
    console.log(`🧩 Chunk preview for note=${id} title="${note.title || ""}" (window=${window}, overlap=${overlap})`);
    const previews = chunkPreview(
      note,
      { windowTokens: window, overlapTokens: overlap, maxLen, log: false }
    );
    if (previews.length === 0) {
      console.log("No chunks produced.");
      return;
    }
    for (const p of previews) {
      console.log(
        `${p.chunk_id} | heading="${p.heading}" | order=${p.order} | tokens=${p.tokens}\n  ${p.preview}`
      );
    }
  } finally {
    db.close();
  }
}

main().catch((err) => {
  console.error("chunk-preview failed:", err);
  process.exit(1);
});
