#!/usr/bin/env node
// 批量导入 chunk 相似边到 Neo4j
import fs from "fs/promises";
import path from "path";
import neo4j from "neo4j-driver";

// 简易 .env 解析（无第三方依赖）
async function loadEnv(envPath = ".env") {
  try {
    const txt = await fs.readFile(path.resolve(envPath), "utf8");
    txt
      .split(/\r?\n/)
      .map((l) => l.trim())
      .filter((l) => l && !l.startsWith("#") && l.includes("="))
      .forEach((line) => {
        const [k, v] = line.split("=", 2);
        if (k && !process.env[k]) process.env[k] = v;
      });
  } catch {
    /* ignore */
  }
}

await loadEnv();

const BASE = process.env.NOTE_VECTORS_DIR || "./src/note_vectors_chunk";
const EDGE_PATH = path.join(BASE, "chunk_sim_edges.json");
const URI = process.env.NEO4J_URI || "bolt://localhost:7687";
const USER = process.env.NEO4J_USER || "neo4j";
const PASS = process.env.NEO4J_PASS || process.env.NEO4J_PASSWORD || "neo4j";
const BATCH = 500;

async function main() {
  const edges = JSON.parse(await fs.readFile(EDGE_PATH, "utf8"));
  const driver = neo4j.driver(URI, neo4j.auth.basic(USER, PASS));
  const session = driver.session();
  console.log(`🔗 Import edges: ${edges.length} from ${EDGE_PATH}`);

  // 确保 chunk_id 存在（如果缺失则回填 id）
  const res = await session.run(
    "MATCH (c:Chunk) WHERE c.chunk_id IS NULL SET c.chunk_id = c.id RETURN count(c) AS updated"
  );
  const updated = res.records[0]?.get("updated")?.toNumber?.() ?? res.records[0]?.get("updated") ?? 0;
  if (updated > 0) console.log(`ℹ️ 填充 chunk_id: ${updated}`);

  for (let i = 0; i < edges.length; i += BATCH) {
    const slice = edges.slice(i, i + BATCH);
    await session.run(
      `
      UNWIND $rows AS row
      MATCH (a:Chunk {chunk_id: row.src})
      MATCH (b:Chunk {chunk_id: row.dst})
      MERGE (a)-[r:SIMILAR_CHUNK]->(b)
      SET r.score = row.score
      `,
      { rows: slice }
    );
    if ((i + BATCH) % 5000 === 0) console.log(`  inserted ${Math.min(i + BATCH, edges.length)}/${edges.length}`);
  }

  await session.close();
  await driver.close();
  console.log("✅ Import done");
}

main().catch((e) => { console.error(e); process.exit(1); });
