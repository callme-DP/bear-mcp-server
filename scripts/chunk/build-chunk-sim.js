#!/usr/bin/env node
// 构建 chunk 相似边（TopK）+ 合并 meta+embedding
import fs from "fs/promises";
import path from "path";
import faiss from "faiss-node";

const K = 10;
const DIM = 384;
const BASE = process.env.NOTE_VECTORS_DIR || "./src/note_vectors_chunk";
const META_PATH = path.join(BASE, "meta_chunks.json");
const EMB_PATH = path.join(BASE, "chunk_embeddings.json");
const EDGE_PATH = path.join(BASE, "chunk_sim_edges.json");
const FULL_PATH = path.join(BASE, "chunk_full.json");

function l2ToSim(d) {
  return 1 / (1 + d); // 需求中的公式
}

async function main() {
  console.log(`📂 META: ${META_PATH}`);
  console.log(`📂 EMB : ${EMB_PATH}`);
  const meta = JSON.parse(await fs.readFile(META_PATH, "utf8"));
  const embs = JSON.parse(await fs.readFile(EMB_PATH, "utf8"));
  if (meta.length !== embs.length) {
    console.warn(`⚠️ 数量不一致 meta=${meta.length} vs emb=${embs.length}，按最短处理`);
  }
  const N = Math.min(meta.length, embs.length);

  const index = new faiss.IndexFlatL2(DIM);
  for (let i = 0; i < N; i++) index.add(embs[i]);

  const edges = [];
  const full = [];
  for (let i = 0; i < N; i++) {
    const { labels, distances } = index.search(embs[i], K + 1); // +1 自身
    for (let j = 1; j < labels.length; j++) { // 跳过自身
      const nbr = labels[j];
      if (nbr >= N) continue;
      const src = meta[i].chunk_id;
      const dst = meta[nbr].chunk_id;
      const sim = l2ToSim(distances[j]);
      edges.push({ src, dst, score: Number(sim.toFixed(4)) });
    }
    full.push({ ...meta[i], embedding: embs[i] });
    if ((i + 1) % 500 === 0) process.stdout.write(`\rProcessed ${i + 1}/${N}`);
  }
  process.stdout.write("\n");

  await fs.writeFile(EDGE_PATH, JSON.stringify(edges, null, 2));
  await fs.writeFile(FULL_PATH, JSON.stringify(full));
  console.log(`✅ edges -> ${EDGE_PATH} (${edges.length})`);
  console.log(`✅ full  -> ${FULL_PATH} (${full.length})`);
}

main().catch((e) => { console.error(e); process.exit(1); });
