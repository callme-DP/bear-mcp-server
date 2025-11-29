#!/usr/bin/env python3
# 将分块（Chunk）导入 Neo4j：
# 1) 为每个 chunk 创建节点，建立 (Note)-[:HAS_CHUNK]->(Chunk) 关系
# 2) 如有 chunk 级向量，写入 embedding 属性，并可生成 Chunk 相似边，供 Note<-Chunk->Chunk->Note 复排
# 3) 为后续 RAG/Chat-with-knowledge 预留：Chunk 节点带 content/heading，可直接按 chunk 召回上下文
#
# 环境变量：
#   NEO4J_URI     (默认 bolt://localhost:7687)
#   NEO4J_USER    (默认 neo4j)
#   NEO4J_PASS    / NEO4J_PASSWORD (默认 neo4j)
#   NOTE_VECTORS_DIR / CHUNK_DIR    (默认 ./src/note_vectors_chunk)
#   CHUNK_SIM_TOPK  (可选，生成 chunk 相似边的 TopK，默认 0=不生成)
#   CHUNK_SIM_TH    (相似度阈值，默认 0.58，需 embedding 可用且已归一化)
#
# 预期输入文件（在 CHUNK_DIR 内）：
#   meta_chunks.json   必需，含 chunk_id/note_id/heading/order/content
#   chunk_embeddings.json（可选，列表对齐 meta 顺序；若不存在则不写 embedding，也不生成相似边）

import json
import os
import sys
from pathlib import Path

import neo4j

# 简易 .env 解析（无第三方依赖），默认读取当前工作目录下 .env
def load_env(env_path=".env"):
  p = Path(env_path)
  if not p.exists():
    return
  try:
    for line in p.read_text(encoding="utf-8").splitlines():
      line = line.strip()
      if not line or line.startswith("#") or "=" not in line:
        continue
      k, v = line.split("=", 1)
      if k and k not in os.environ:
        os.environ[k] = v
  except Exception as e:
    print(f"⚠️  读取 {env_path} 失败: {e}", file=sys.stderr)

# 读取 .env（若有）
load_env()

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASS = os.environ.get("NEO4J_PASS") or os.environ.get("NEO4J_PASSWORD", "neo4j")

CHUNK_DIR = Path(os.environ.get("NOTE_VECTORS_DIR") or os.environ.get("CHUNK_DIR") or "./src/note_vectors_chunk")
META_PATH = CHUNK_DIR / "meta_chunks.json"
EMB_PATH = CHUNK_DIR / "chunk_embeddings.json"

CHUNK_SIM_TOPK = int(os.environ.get("CHUNK_SIM_TOPK", "0"))  # 0 表示不生成相似边
CHUNK_SIM_TH = float(os.environ.get("CHUNK_SIM_TH", "0.58"))  # 相似边阈值（近似余弦相似度，向量已归一化时 0.58≈较宽松，提升可调高）


def load_json(path: Path, required: bool = True):
  try:
    with open(path, "r", encoding="utf-8") as f:
      return json.load(f)
  except FileNotFoundError:
    if required:
      print(f"❌ 文件未找到: {path}", file=sys.stderr)
      sys.exit(1)
    return None


def to_embeddings(obj):
  if obj is None:
    return None
  if isinstance(obj, dict):
    # dict 场景：{chunk_id: [...], ...}
    return obj
  if isinstance(obj, list):
    # list 场景：与 meta_chunks 顺序对齐
    return obj
  return None


def cosine_sim(a, b):
  dot = sum(x * y for x, y in zip(a, b))
  sa = sum(x * x for x in a) ** 0.5
  sb = sum(y * y for y in b) ** 0.5
  if sa == 0 or sb == 0:
    return 0.0
  return dot / (sa * sb)


def build_topk_edges(chunks, emb_lookup, topk, th):
  """基于 embedding 构建 chunk 相似边：返回 [(id1, id2, score), ...]"""
  edges = []
  vecs = []
  ids = []
  for c in chunks:
    cid = c.get("chunk_id")
    vec = emb_lookup.get(cid)
    if cid and vec:
      ids.append(cid)
      vecs.append(vec)
  n = len(ids)
  if n == 0:
    return edges
  print(f"🧠 计算 chunk 相似 Top{topk}，有效向量 {n} 条")
  for i in range(n):
    vi = vecs[i]
    scores = []
    for j in range(n):
      if i == j:
        continue
      vj = vecs[j]
      sim = cosine_sim(vi, vj)
      if sim >= th:
        scores.append((j, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    for j, s in scores[:topk]:
      edges.append((ids[i], ids[j], round(float(s), 4)))
  print(f"✅ 相似边生成 {len(edges)} 条")
  return edges


def merge_chunks(session, chunks, emb_lookup):
  tx = session
  missing_ids = 0
  missing_note = 0
  for idx, c in enumerate(chunks, 1):
    cid = c.get("chunk_id")
    nid = c.get("note_id")
    if not cid:
      missing_ids += 1
      continue
    if not nid:
      missing_note += 1
      continue
    if idx % 200 == 0:
      print(f"📥 Chunk 导入进度: {idx}/{len(chunks)}")
    tx.run(
      """
      MERGE (c:Chunk {id:$cid})
      SET c.note_id = $nid,
          c.heading = $heading,
          c.order = $order,
          c.content = $content
      """,
      {
        "cid": cid,
        "nid": nid,
        "heading": c.get("heading") or "",
        "order": c.get("order") if c.get("order") is not None else -1,
        "content": c.get("content") or "",
      },
    )
    vec = emb_lookup.get(cid)
    if vec:
      tx.run("MATCH (c:Chunk {id:$cid}) SET c.embedding = $emb", {"cid": cid, "emb": vec})
    tx.run(
      """
      MERGE (n:Note {id:$nid})
      MERGE (n)-[:HAS_CHUNK]->(c)
      """,
      {"nid": nid, "cid": cid},
    )
  print(f"✅ Chunk 导入完成，共 {len(chunks)} 条；跳过 chunk_id 缺失 {missing_ids} 条，note_id 缺失 {missing_note} 条")


def import_chunk_sim_edges(session, edges):
  for idx, (c1, c2, score) in enumerate(edges, 1):
    if idx % 500 == 0:
      print(f"📡 相似边导入进度: {idx}/{len(edges)}")
    session.run(
      """
      MATCH (a:Chunk {id:$c1}), (b:Chunk {id:$c2})
      MERGE (a)-[r:SIMILAR_CHUNK]->(b)
      SET r.weight = $w
      """,
      {"c1": c1, "c2": c2, "w": score},
    )
  print(f"✅ 相似边导入完成，共 {len(edges)} 条")


def main():
  if not META_PATH.exists():
    print(f"❌ 未找到 meta_chunks.json: {META_PATH}", file=sys.stderr)
    sys.exit(1)

  chunks = load_json(META_PATH, required=True)
  print(f"📄 读取 chunk 元数据 {len(chunks)} 条，自 {META_PATH}")
  if chunks:
    sample = chunks[0]
    print(f"   样例 chunk: chunk_id={sample.get('chunk_id')}, note_id={sample.get('note_id')}, heading={sample.get('heading')}, order={sample.get('order')}")

  emb_raw = load_json(EMB_PATH, required=False)
  emb_lookup = {}
  if emb_raw is None:
    print("ℹ️ 未提供 chunk_embeddings.json，跳过 embedding 与相似边写入")
  else:
    data = to_embeddings(emb_raw)
    if isinstance(data, list):
      if len(data) != len(chunks):
        print(f"⚠️ embeddings 长度 {len(data)} 与 chunks {len(chunks)} 不一致，按顺序取较短部分")
      for i, c in enumerate(chunks):
        if i < len(data):
          emb_lookup[c.get("chunk_id")] = data[i]
    elif isinstance(data, dict):
      emb_lookup = {k: v for k, v in data.items() if isinstance(v, list)}
    print(f"🧠 载入 embedding {len(emb_lookup)} 条")

  print(f"🔧 配置: NEO4J_URI={NEO4J_URI}, NOTE_VECTORS_DIR={CHUNK_DIR}")

  driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=neo4j.basic_auth(NEO4J_USER, NEO4J_PASS))
  with driver.session() as session:
    merge_chunks(session, chunks, emb_lookup)
    # 兜底补齐 Note -> Chunk 关系，防止前面循环遗漏
    session.run(
      """
      MATCH (c:Chunk)
      MERGE (n:Note {id: c.note_id})
      MERGE (n)-[:HAS_CHUNK]->(c)
      """
    )
    if CHUNK_SIM_TOPK > 0 and emb_lookup:
      edges = build_topk_edges(chunks, emb_lookup, CHUNK_SIM_TOPK, CHUNK_SIM_TH)
      if edges:
        import_chunk_sim_edges(session, edges)
  driver.close()
  print("🎯 完成 Chunk 导入")


if __name__ == "__main__":
  main()
