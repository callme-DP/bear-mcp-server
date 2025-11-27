#!/usr/bin/env node
// Import graph_v2 (graph.json + embeddings.json + meta.json) into Neo4j.
// Steps:
// 1) Delete existing nodes (detach delete) and verify empty.
// 2) Import Note/Tag nodes with properties and embeddings.
// 3) Import edges (semantic/has_tag).
// Env vars: NEO4J_URI (default bolt://localhost:7687), NEO4J_USER (default neo4j), NEO4J_PASS/NEO4J_PASSWORD.

import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";
import neo4j from "neo4j-driver";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const NEO4J_URI = process.env.NEO4J_URI || "bolt://localhost:7687";
const NEO4J_USER = process.env.NEO4J_USER || "neo4j";
const NEO4J_PASS = process.env.NEO4J_PASS || process.env.NEO4J_PASSWORD || "neo4j";
// Allow override, default to consolidated data/neo4j directory.
const DATA_DIR = process.env.NOTE_VECTORS_DIR || path.resolve(__dirname, "../data/neo4j");

const GRAPH_PATH = path.join(DATA_DIR, "graph.json");
const EMB_PATH = path.join(DATA_DIR, "embeddings.json");
const META_PATH = path.join(DATA_DIR, "meta.json");

function log(msg) {
  console.log(msg);
}

async function loadData() {
  const graph = JSON.parse(await fs.readFile(GRAPH_PATH, "utf8"));
  const embeddings = JSON.parse(await fs.readFile(EMB_PATH, "utf8"));
  const meta = JSON.parse(await fs.readFile(META_PATH, "utf8"));
  return { graph, embeddings, meta };
}

function buildEmbeddingMap(meta, embeddings) {
  const map = new Map();
  for (const m of meta) {
    if (embeddings[m.idx]) {
      map.set(m.id, embeddings[m.idx]);
    }
  }
  return map;
}

async function deleteAll(session) {
  log("🧹 Deleting all existing nodes/edges...");
  await session.run("MATCH (n) DETACH DELETE n");
  const res = await session.run("MATCH (n) RETURN count(n)");
  const cnt = res.records[0].get(0).toNumber ? res.records[0].get(0).toNumber() : res.records[0].get(0);
  log(`✅ After delete, node count = ${cnt}`);
}

async function importNodes(session, nodes, embeddingMap) {
  let idx = 0;
  for (const n of nodes) {
    idx++;
    if (idx % 200 === 0) log(`📥 Nodes imported: ${idx}/${nodes.length}`);

    if (n.type === "Note") {
      await session.run(
        `
        MERGE (x:Note {id: $id})
        SET x.title = $title,
            x.summary = $summary,
            x.tags = $tags,
            x.created = $created,
            x.modified = $modified
        `,
        {
          id: n.id,
          title: n.title || "",
          summary: n.summary || "",
          tags: n.tags || [],
          created: n.created || null,
          modified: n.modified || null,
        }
      );
      if (embeddingMap.has(n.id)) {
        await session.run(
          `MATCH (x:Note {id:$id}) SET x.embedding = $embedding`,
          { id: n.id, embedding: embeddingMap.get(n.id) }
        );
      }
    } else if (n.type === "Tag") {
      const name = n.title || n.name || n.id?.replace(/^tag:/, "") || "";
      await session.run(
        `
        MERGE (t:Tag {name: $name})
        SET t.title = $title
        `,
        { name, title: name }
      );
    }
  }
  log(`✅ Nodes imported total: ${nodes.length}`);
}

function relType(type) {
  if (!type) return "RELATED";
  const t = type.toLowerCase();
  if (t === "semantic") return "SEMANTIC";
  if (t === "has_tag") return "HAS_TAG";
  return "RELATED";
}

async function importEdges(session, edges) {
  let idx = 0;
  for (const e of edges) {
    idx++;
    if (idx % 500 === 0) log(`📡 Edges imported: ${idx}/${edges.length}`);
    const label = relType(e.type);
    await session.run(
      `
      MATCH (a {id:$from}), (b {id:$to})
      MERGE (a)-[r:${label}]->(b)
      SET r.weight = $weight, r.score = $weight
      `,
      { from: e.from, to: e.to, weight: e.weight ?? 1 }
    );
  }
  log(`✅ Edges imported total: ${edges.length}`);
}

async function verifyCounts(session) {
  const noteCnt = await session.run("MATCH (n:Note) RETURN count(n) AS c");
  const tagCnt = await session.run("MATCH (t:Tag) RETURN count(t) AS c");
  const semCnt = await session.run("MATCH ()-[r:SEMANTIC]->() RETURN count(r) AS c");
  const hasCnt = await session.run("MATCH ()-[r:HAS_TAG]->() RETURN count(r) AS c");
  const val = (res) => res.records[0].get("c").toNumber ? res.records[0].get("c").toNumber() : res.records[0].get("c");
  log(`📊 Verify counts -> Notes: ${val(noteCnt)}, Tags: ${val(tagCnt)}, SEMANTIC: ${val(semCnt)}, HAS_TAG: ${val(hasCnt)}`);
}

async function main() {
  log("🚀 Import graph_v2 -> Neo4j");
  const { graph, embeddings, meta } = await loadData();
  const embeddingMap = buildEmbeddingMap(meta, embeddings);

  const driver = neo4j.driver(NEO4J_URI, neo4j.auth.basic(NEO4J_USER, NEO4J_PASS));
  const session = driver.session();

  try {
    await deleteAll(session);
    await importNodes(session, graph.nodes || [], embeddingMap);
    await importEdges(session, graph.edges || []);
    await verifyCounts(session);
    log("🎉 Import finished");
  } catch (err) {
    console.error("❌ Import failed:", err);
    process.exit(1);
  } finally {
    await session.close();
    await driver.close();
  }
}

main();
