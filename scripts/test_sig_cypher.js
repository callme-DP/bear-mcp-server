#!/usr/bin/env node
import neo4j from "neo4j-driver";

const SUITE_LABEL = "SIG_TEST";
const SUITE_NAME = "sig-cypher-suite";
const KEEP_DATA = process.argv.includes("--keep-data");

const NEO4J_URI = process.env.NEO4J_URI || "bolt://localhost:7687";
const NEO4J_USER = process.env.NEO4J_USER || "neo4j";
const NEO4J_PASS = process.env.NEO4J_PASS || process.env.NEO4J_PASSWORD || "neo4j";

const now = new Date();
const daysAgo = (n) => new Date(now.getTime() - n * 24 * 60 * 60 * 1000).toISOString();

const DATASET = {
  tags: ["area/ML", "project/Atlas", "resource/Papers", "topic/Isolated"],
  notes: [
    {
      id: "N_HUB",
      title: "ML Hub Note",
      summary: "High-degree semantic hub inside area/ML.",
      tags: ["area/ML"],
      created: "2025-01-10T00:00:00.000Z",
      modified: "2025-01-15T00:00:00.000Z",
    },
    {
      id: "N_AREA_PEER",
      title: "Model Basics",
      summary: "Area peer connected to project.",
      tags: ["area/ML"],
      created: "2025-01-01T00:00:00.000Z",
      modified: "2025-01-01T00:00:00.000Z",
    },
    {
      id: "N_PROJECT",
      title: "Atlas Plan",
      summary: "Project node bridging to resources.",
      tags: ["project/Atlas"],
      created: "2025-02-05T00:00:00.000Z",
      modified: "2025-02-10T00:00:00.000Z",
    },
    {
      id: "N_RESOURCE",
      title: "Paper Digest",
      summary: "Resource node receiving dense links.",
      tags: ["resource/Papers"],
      created: "2025-01-20T00:00:00.000Z",
      modified: "2025-01-20T00:00:00.000Z",
    },
    {
      id: "N_DRIFT",
      title: "Recent Area Spike",
      summary: "Recent area/ML note to exercise recency and drift.",
      tags: ["area/ML"],
      created: daysAgo(2),
      modified: daysAgo(2),
    },
    {
      id: "N_ISOLATED",
      title: "Isolated Thought",
      summary: "No semantic edges; used to detect semantic islands.",
      tags: ["topic/Isolated"],
      created: "2024-12-01T00:00:00.000Z",
      modified: "2024-12-01T00:00:00.000Z",
    },
  ],
  semantic: [
    { from: "N_HUB", to: "N_AREA_PEER", weight: 0.9 },
    { from: "N_HUB", to: "N_PROJECT", weight: 0.85 },
    { from: "N_HUB", to: "N_RESOURCE", weight: 0.8 },
    { from: "N_AREA_PEER", to: "N_PROJECT", weight: 0.65 },
    { from: "N_PROJECT", to: "N_RESOURCE", weight: 0.92 },
    { from: "N_DRIFT", to: "N_RESOURCE", weight: 0.75 },
  ],
};

function toNative(value) {
  return neo4j.isInt(value) ? value.toNumber() : value;
}

function expect(condition, message) {
  if (!condition) throw new Error(message);
}

async function cleanup(session) {
  await session.executeWrite((tx) => tx.run(`MATCH (n:${SUITE_LABEL}) DETACH DELETE n`));
}

async function seed(session) {
  await cleanup(session);

  await session.executeWrite((tx) =>
    tx.run(
      `
      UNWIND $tags AS title
      CREATE (:Tag:${SUITE_LABEL} {title:title, testSuite:$suite});
    `,
      { tags: DATASET.tags, suite: SUITE_NAME }
    )
  );

  await session.executeWrite((tx) =>
    tx.run(
      `
      UNWIND $notes AS note
      CREATE (n:Note:${SUITE_LABEL} {
        id: note.id,
        title: note.title,
        summary: note.summary,
        created: datetime(note.created),
        modified: datetime(note.modified),
        testSuite: $suite
      })
      WITH n, note
      UNWIND note.tags AS tagTitle
      MATCH (t:Tag:${SUITE_LABEL} {title:tagTitle, testSuite:$suite})
      CREATE (n)-[:HAS_TAG {testSuite:$suite}]->(t);
    `,
      { notes: DATASET.notes, suite: SUITE_NAME }
    )
  );

  await session.executeWrite((tx) =>
    tx.run(
      `
      UNWIND $edges AS edge
      MATCH (s:Note:${SUITE_LABEL} {id:edge.from, testSuite:$suite})
      MATCH (d:Note:${SUITE_LABEL} {id:edge.to, testSuite:$suite})
      CREATE (s)-[:SEMANTIC {weight:edge.weight, testSuite:$suite}]->(d);
    `,
      { edges: DATASET.semantic, suite: SUITE_NAME }
    )
  );
}

async function runTest(session, label, cypher, params, verify) {
  process.stdout.write(`🧪 ${label} ... `);
  const records = await session.executeRead((tx) => tx.run(cypher, params).then((res) => res.records));
  await verify(records);
  console.log("ok");
}

async function main() {
  const driver = neo4j.driver(NEO4J_URI, neo4j.auth.basic(NEO4J_USER, NEO4J_PASS));
  const session = driver.session();

  try {
    await seed(session);

    await runTest(
      "1) Semantic cluster for hub (ordered by weight)",
      `
        MATCH (n:Note:${SUITE_LABEL} {id:$id})-[r:SEMANTIC]-(m:Note:${SUITE_LABEL})
        RETURN m.id AS id, m.title AS title, m.summary AS summary, r.weight AS weight
        ORDER BY weight DESC
        LIMIT 20
      `,
      { id: "N_HUB" },
      (records) => {
        expect(records.length === 3, "Expected 3 semantic neighbors for N_HUB");
        const expected = [
          { id: "N_AREA_PEER", weight: 0.9 },
          { id: "N_PROJECT", weight: 0.85 },
          { id: "N_RESOURCE", weight: 0.8 },
        ];
        records.forEach((rec, idx) => {
          const id = rec.get("id");
          const weight = toNative(rec.get("weight"));
          expect(id === expected[idx].id, `Neighbor ${idx} should be ${expected[idx].id}, got ${id}`);
          expect(weight === expected[idx].weight, `Weight for ${id} should be ${expected[idx].weight}, got ${weight}`);
        });
      }
    );

    await runTest(
      "2) High-degree semantic hub (degree >= 3)",
      `
        MATCH (n:Note:${SUITE_LABEL})-[r:SEMANTIC]->()
        WITH n, count(r) AS degree
        WHERE degree >= $minDegree
        RETURN n.id AS id, n.title AS title, degree
        ORDER BY degree DESC
      `,
      { minDegree: 3 },
      (records) => {
        expect(records.length === 1, "Expected exactly one hub with degree >= 3");
        const id = records[0].get("id");
        const degree = toNative(records[0].get("degree"));
        expect(id === "N_HUB", `Hub should be N_HUB, got ${id}`);
        expect(degree === 3, `N_HUB should have degree 3, got ${degree}`);
      }
    );

    await runTest(
      "3) Semantic islands (no SEMANTIC edges)",
      `
        MATCH (n:Note:${SUITE_LABEL})
        WHERE NOT (n)-[:SEMANTIC]-()
        RETURN n.id AS id, n.title AS title
        ORDER BY id
      `,
      {},
      (records) => {
        const ids = records.map((r) => r.get("id"));
        expect(ids.length === 1 && ids[0] === "N_ISOLATED", `Expected only N_ISOLATED, got ${ids.join(", ") || "none"}`);
      }
    );

    await runTest(
      "4) Tag anchor inside area/ML (semantic center)",
      `
        MATCH (t:Tag:${SUITE_LABEL} {title:$tag})<-[:HAS_TAG]-(n:Note:${SUITE_LABEL})-[r:SEMANTIC]->(m)
        RETURN n.id AS id, n.title AS title, count(r) AS connections
        ORDER BY connections DESC
        LIMIT 5
      `,
      { tag: "area/ML" },
      (records) => {
        expect(records.length >= 1, "Expected at least one note inside tag area/ML");
        const id = records[0].get("id");
        const connections = toNative(records[0].get("connections"));
        expect(id === "N_HUB", `Top semantic center should be N_HUB, got ${id}`);
        expect(connections === 3, `N_HUB semantic edges should be 3, got ${connections}`);
      }
    );

    await runTest(
      "5) Recent 14-day semantic additions under area/ML",
      `
        MATCH (t:Tag:${SUITE_LABEL} {title:$tag})<-[:HAS_TAG]-(n:Note:${SUITE_LABEL})
        WHERE n.created > datetime() - duration('P14D')
        WITH n
        MATCH (n)-[r:SEMANTIC]->(m:Note:${SUITE_LABEL})
        RETURN n.title AS source, m.title AS target, r.weight AS weight
        ORDER BY weight DESC
      `,
      { tag: "area/ML" },
      (records) => {
        expect(records.length === 1, `Expected only the recent area/ML note, got ${records.length}`);
        const source = records[0].get("source");
        const target = records[0].get("target");
        const weight = toNative(records[0].get("weight"));
        expect(source === "Recent Area Spike", `Recent node should be 'Recent Area Spike', got ${source}`);
        expect(target === "Paper Digest", `Recent edge target should be 'Paper Digest', got ${target}`);
        expect(weight === 0.75, `Edge weight should be 0.75, got ${weight}`);
      }
    );

    await runTest(
      "6) Cross-tag bridges",
      `
        MATCH (n:Note:${SUITE_LABEL})-[:HAS_TAG]->(t1:Tag:${SUITE_LABEL}),
              (n)-[:SEMANTIC]->(m:Note:${SUITE_LABEL})-[:HAS_TAG]->(t2:Tag:${SUITE_LABEL})
        WHERE t1 <> t2
        RETURN n.id AS id, n.title AS title, t1.title AS tag1, t2.title AS tag2
        ORDER BY id, tag2
      `,
      {},
      (records) => {
        const actual = records.map((r) => ({
          id: r.get("id"),
          tag1: r.get("tag1"),
          tag2: r.get("tag2"),
        }));
        const expected = [
          { id: "N_AREA_PEER", tag1: "area/ML", tag2: "project/Atlas" },
          { id: "N_DRIFT", tag1: "area/ML", tag2: "resource/Papers" },
          { id: "N_HUB", tag1: "area/ML", tag2: "project/Atlas" },
          { id: "N_HUB", tag1: "area/ML", tag2: "resource/Papers" },
          { id: "N_PROJECT", tag1: "project/Atlas", tag2: "resource/Papers" },
        ];
        expect(
          actual.length === expected.length,
          `Expected ${expected.length} bridge rows, got ${actual.length}`
        );
        expected.forEach((exp, idx) => {
          const act = actual[idx];
          expect(
            act.id === exp.id && act.tag1 === exp.tag1 && act.tag2 === exp.tag2,
            `Bridge row ${idx} mismatch: expected ${exp.id}/${exp.tag1}->${exp.tag2}, got ${act.id}/${act.tag1}->${act.tag2}`
          );
        });
      }
    );

    await runTest(
      "7) Structural holes (tag pairs with no semantic bridges)",
      `
        MATCH (t1:Tag:${SUITE_LABEL}), (t2:Tag:${SUITE_LABEL})
        WHERE t1 <> t2
        AND NOT EXISTS {
          MATCH (t1)<-[:HAS_TAG]-(n:Note:${SUITE_LABEL})-[:SEMANTIC]->(:Note:${SUITE_LABEL})-[:HAS_TAG]->(t2)
        }
        RETURN t1.title AS tag1, t2.title AS tag2
        ORDER BY tag1, tag2
      `,
      {},
      (records) => {
        const actual = records.map((r) => `${r.get("tag1")}::${r.get("tag2")}`);
        const expected = [
          "area/ML::topic/Isolated",
          "project/Atlas::topic/Isolated",
          "resource/Papers::topic/Isolated",
          "topic/Isolated::area/ML",
          "topic/Isolated::project/Atlas",
          "topic/Isolated::resource/Papers",
        ];
        expect(
          actual.length === expected.length,
          `Expected ${expected.length} structural holes, got ${actual.length}`
        );
        expected.forEach((pair, idx) => {
          expect(actual[idx] === pair, `Hole ${idx} should be ${pair}, got ${actual[idx]}`);
        });
      }
    );

    await runTest(
      "8) PARA intensity for area/",
      `
        MATCH (t:Tag:${SUITE_LABEL})
        WHERE t.title STARTS WITH $prefix
        WITH t
        MATCH (t)<-[:HAS_TAG]-(n:Note:${SUITE_LABEL})-[r:SEMANTIC]->(m:Note:${SUITE_LABEL})
        RETURN t.title AS para, count(r) AS intensity
        ORDER BY intensity DESC
      `,
      { prefix: "area/" },
      (records) => {
        expect(records.length === 1, `Expected one PARA bucket for prefix area/, got ${records.length}`);
        const para = records[0].get("para");
        const intensity = toNative(records[0].get("intensity"));
        expect(para === "area/ML", `PARA tag should be area/ML, got ${para}`);
        expect(intensity === 5, `Intensity for area/ML should be 5, got ${intensity}`);
      }
    );

    await runTest(
      "9) PARA neighbor drift (area/)",
      `
        MATCH (n:Note:${SUITE_LABEL})-[:HAS_TAG]->(t:Tag:${SUITE_LABEL})
        WHERE t.title STARTS WITH $prefix
        WITH n
        MATCH (n)-[r:SEMANTIC]->(m:Note:${SUITE_LABEL})
        RETURN n.id AS id, n.title AS title, collect(m.title)[0..8] AS neighbors
        ORDER BY id
        LIMIT 20
      `,
      { prefix: "area/" },
      (records) => {
        const map = Object.fromEntries(
          records.map((r) => [
            r.get("id"),
            r.get("neighbors").map((v) => v).sort(),
          ])
        );
        expect(
          map.N_HUB?.join(",") === ["Atlas Plan", "Model Basics", "Paper Digest"].sort().join(","),
          `Neighbors for N_HUB mismatch: ${map.N_HUB}`
        );
        expect(
          map.N_AREA_PEER?.join(",") === ["Atlas Plan"].join(","),
          `Neighbors for N_AREA_PEER mismatch: ${map.N_AREA_PEER}`
        );
        expect(
          map.N_DRIFT?.join(",") === ["Paper Digest"].join(","),
          `Neighbors for N_DRIFT mismatch: ${map.N_DRIFT}`
        );
      }
    );

    await runTest(
      "10) Top weighted semantic edges",
      `
        MATCH (a:Note:${SUITE_LABEL})-[r:SEMANTIC]->(b:Note:${SUITE_LABEL})
        RETURN a.title AS from, b.title AS to, r.weight AS weight
        ORDER BY r.weight DESC, from, to
        LIMIT 50
      `,
      {},
      (records) => {
        const expected = [
          { from: "Atlas Plan", to: "Paper Digest", weight: 0.92 },
          { from: "ML Hub Note", to: "Model Basics", weight: 0.9 },
          { from: "ML Hub Note", to: "Atlas Plan", weight: 0.85 },
          { from: "ML Hub Note", to: "Paper Digest", weight: 0.8 },
          { from: "Recent Area Spike", to: "Paper Digest", weight: 0.75 },
          { from: "Model Basics", to: "Atlas Plan", weight: 0.65 },
        ];
        expect(
          records.length === expected.length,
          `Expected ${expected.length} semantic edges, got ${records.length}`
        );
        expected.forEach((exp, idx) => {
          const rec = records[idx];
          const from = rec.get("from");
          const to = rec.get("to");
          const weight = toNative(rec.get("weight"));
          expect(
            from === exp.from && to === exp.to,
            `Edge ${idx} should be ${exp.from} -> ${exp.to}, got ${from} -> ${to}`
          );
          expect(weight === exp.weight, `Weight for ${from}->${to} should be ${exp.weight}, got ${weight}`);
        });
      }
    );

    console.log("✅ All Cypher query expectations passed");
  } catch (err) {
    console.error(`❌ Test run failed: ${err.message || err}`);
    process.exitCode = 1;
  } finally {
    if (!KEEP_DATA) {
      await cleanup(session);
    } else {
      console.log("⚠️ Keeping test data in Neo4j for inspection (--keep-data enabled)");
    }
    await session.close();
    await driver.close();
  }
}

main();
