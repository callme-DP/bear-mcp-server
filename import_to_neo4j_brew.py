from neo4j import GraphDatabase
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import os

# ========== Neo4j 连接配置 ==========
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password"  # ← 替换为你实际密码

EXPORT_PATH = "src/exports"

# ========== 读取数据 ==========
with open(os.path.join(EXPORT_PATH, "graph.json"), "r", encoding="utf-8") as f:
    graph_data = json.load(f)

with open(os.path.join(EXPORT_PATH, "meta.json"), "r", encoding="utf-8") as f:
    meta_raw = json.load(f)

# meta.json 可能是列表，也可能是字典，我们统一成字典格式
if isinstance(meta_raw, list):
    meta_data = {str(item.get("id")): item for item in meta_raw if item.get("id")}
else:
    meta_data = meta_raw

# 如果有 embeddings.json 就加载
embeddings = {}
embed_path = os.path.join(EXPORT_PATH, "embeddings.json")
if os.path.exists(embed_path):
    with open(embed_path, "r", encoding="utf-8") as f:
        embeddings_raw = json.load(f)

    # ✅ case1: list[list[float]]
    if isinstance(embeddings_raw, list):
        print(f"🧩 embeddings.json 检测为 list，共 {len(embeddings_raw)} 条记录")
        node_list = graph_data.get("nodes", [])
        valid_len = min(len(node_list), len(embeddings_raw))

        for i in range(valid_len):
            nid = str(node_list[i].get("id"))
            emb = embeddings_raw[i]
            if emb and isinstance(emb, list):
                embeddings[nid] = emb

        print(f"✅ 已成功绑定 {len(embeddings)} 条节点 embedding（自动按顺序对齐）")

    # ✅ case2: dict{id: [float,...]}
    elif isinstance(embeddings_raw, dict):
        print(f"🧩 embeddings.json 检测为 dict 格式，共 {len(embeddings_raw)} 条")
        embeddings = embeddings_raw

    else:
        print("⚠️ 无法识别 embeddings.json 格式，保持为空")

else:
    print("⚠️ 未找到 embeddings.json 文件，跳过")


nodes = graph_data["nodes"]
edges = graph_data["edges"]
# ========== 日志辅助函数 ==========
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
# ======= 智能检测输出日志 =======
def log_data_structure():
    log("\n📊 ====== 数据结构检测报告 ======")

    # 检查 graph.json
    node_count = len(nodes) if isinstance(nodes, list) else 0
    edge_count = len(edges) if isinstance(edges, list) else 0
    log(f"📘 graph.json: 节点 {node_count} 个，边 {edge_count} 条")

    # 检查 meta.json
    if isinstance(meta_raw, list):
        log(f"🟢 meta.json: list 格式 → 已自动转换为 dict（共 {len(meta_raw)} 条记录）")
    elif isinstance(meta_raw, dict):
        log(f"🟢 meta.json: dict 格式（共 {len(meta_raw)} 条记录）")
    else:
        log("⚠️ meta.json 格式异常，请检查内容")

    # 检查 embeddings.json
    if embeddings:
        log(f"🧠 embeddings.json: 含 {len(embeddings)} 条向量记录")
    else:
        log("⚠️ 未检测到 embeddings.json 或为空")

    log("=================================\n")

# 执行检测日志
log_data_structure()

# ========== 连接 Neo4j ==========
# auth=(NEO4J_USER, NEO4J_PASSWORD) -> NONE
driver = GraphDatabase.driver(NEO4J_URI, auth = None)

def detect_para_type(tags):
    """
    根据标签自动识别 PARA 类型（项目-领域-资源-归档系统）
    
    基于标签内容判断笔记所属的PARA分类，支持识别项目、领域、资源、归档、标签和普通笔记类型。

    Args:
        tags: 标签列表，包含笔记的所有标签信息

    Returns:
        str: 返回识别出的PARA类型，可能值为：
            - "Project": 项目类型
            - "Area": 领域类型  
            - "Resource": 资源类型
            - "Archive": 归档类型
            - "Tag": 标签类型
            - "Note": 普通笔记类型
    """
    """根据标签自动识别 PARA 类型"""
    if not tags:
        return "Note"
    tags_str = " ".join(tags).lower()
    if "project" in tags_str:
        return "Project"
    elif "area" in tags_str:
        return "Area"
    elif "resource" in tags_str:
        return "Resource"
    elif "archive" in tags_str:
        return "Archive"
    elif any(t.startswith("#") for t in tags) or "tag" in tags_str:   # ✅ 新增：识别 Tag
        return "Tag"
    return "Note"

# ========== 导入函数 ==========
def import_data(tx, nodes, edges, meta, embeds):
    # ======= 数据检测 =======
    invalid_edges = [e for e in edges if "source" not in e or "target" not in e]
    if invalid_edges:
        print("⚠️ 示例问题边：", invalid_edges[:3])

    # ======= 导入节点 =======
    for n in nodes:
        nid = str(n["id"])
        meta_info = meta.get(nid, {})
        embedding = embeds.get(nid)

        # 识别没有 meta 的标签节点
        is_tag = 0
        if not meta_info and nid.startswith("tag:"):
            is_tag = 1
            meta_info = {  # 给它一个最简默认meta
                "title": nid.replace("tag:", ""),
                "tags": ["tag"],
                "source": "graph_only"
            }
            
        # ✅ emoji标题 安全处理
        safe_title = meta_info.get("title") or n.get("label") or ""
        if not isinstance(safe_title, str):
            safe_title = str(safe_title)
        safe_title = safe_title.encode("utf-8", "surrogatepass").decode("utf-8")

        
        # ✅ PARA 分类检测
        tags = meta_info.get("tags", [])
        para_type = detect_para_type(tags)
        
        #✅ 添加url属性
        bear_url = f"bear://x-callback-url/open-note?id={nid}"

       # ✅ 改进版：让 PARA 类型成为标签
        tx.run(f"""
            MERGE (a:Note:{para_type} {{id: $id}})
            SET a.title = $title,
                a.path = $path,
                a.tags = $tags,
                a.source = $source,
                a.embedding = $embedding,
                a.para_type = $para_type,
                a.isTag = $isTag,              
                a.bear_url = $bear_url,
                a.createdAt = datetime($createdAt),
                a.updatedAt = datetime($updatedAt)
        """,
            id=nid,
            title=safe_title,
            path=meta_info.get("path"),
            tags=tags,
            source=meta_info.get("source"),
            embedding=embedding,
            para_type=para_type,
            isTag=is_tag,  
            bear_url=bear_url,
            createdAt=n.get("created"),
            updatedAt=n.get("modified")
        )

# ========== 导入函数 ==========
def import_data(tx, nodes, edges, meta, embeds):
    # ======= 示例打印 =======
    invalid_edges = [e for e in edges if "source" not in e or "target" not in e]
    if invalid_edges:
        print("⚠️ 示例问题边：", invalid_edges[:3])

    # ======= 导入节点 =======
    for n in nodes:
        nid = str(n["id"])
        meta_info = meta.get(nid, {})
        embedding = embeds.get(nid)

        # 识别没有 meta 的标签节点
        is_tag = 0
        if not meta_info and nid.startswith("tag:"):
            is_tag = 1
            meta_info = {  # 给它一个最简默认meta
                "title": nid.replace("tag:", ""),
                "tags": ["tag"],
                "source": "graph_only"
            }

            
        # ✅ emoji 安全处理
        safe_title = meta_info.get("title") or n.get("label") or ""
        if not isinstance(safe_title, str):
            safe_title = str(safe_title)
        safe_title = safe_title.encode("utf-8", "surrogatepass").decode("utf-8")
        
        # ✅ PARA 分类检测
        tags = meta_info.get("tags", [])
        para_type = detect_para_type(tags)
        
        #✅ 添加url属性
        bear_url = f"bear://x-callback-url/open-note?id={nid}"

       # ✅ 改进版：让 PARA 类型成为标签
        tx.run(f"""
            MERGE (a:Note:{para_type} {{id: $id}})
            SET a.title = $title,
                a.path = $path,
                a.tags = $tags,
                a.source = $source,
                a.embedding = $embedding,
                a.para_type = $para_type,
                a.isTag = $isTag,              
                a.bear_url = $bear_url,
                a.createdAt = datetime($createdAt),
                a.updatedAt = datetime($updatedAt)
        """,
            id=nid,
            title=safe_title,
            path=meta_info.get("path"),
            tags=tags,
            source=meta_info.get("source"),
            embedding=embedding,
            para_type=para_type,
            isTag=is_tag,  
            bear_url=bear_url,
            createdAt=n.get("created"),
            updatedAt=n.get("modified")
        )

    # ======= 导入边关系（含语义增强） =======
    for e in edges:
        # 基础验证
        if not isinstance(e, dict):
            print(f"⚠️ 跳过异常边（非 dict 类型）: {e}")
            continue
        if "from" not in e or "to" not in e:
            print(f"⚠️ 跳过缺少 from/to 字段的边: {e}")
            continue

        source_id = str(e["from"])
        target_id = str(e["to"])
        weight = float(e.get("weight", 1.0))

        # ====== 根据标签识别关系语义 ======
        source_meta = meta.get(source_id, {})
        target_meta = meta.get(target_id, {})
        source_tags = [t.lower() for t in meta.get(source_id, {}).get("tags", [])]
        target_tags = [t.lower() for t in meta.get(target_id, {}).get("tags", [])]
       
        # # 默认关系类型,默认灰色
        rel_type,color = "RELATED","#9ca3af"    

        if any("area" in t for t in target_tags):
            rel_type = "BELONGS_TO"
            color = "#3b82f6"  # 蓝
        elif any("resource" in t for t in target_tags):
            rel_type = "CITES"
            color = "#f59e0b"  # 橙
        elif any("concept" in t for t in target_tags):
            rel_type = "EXTENDS"
            color = "#8b5cf6"  # 紫
        elif target_meta.get("para_type") == "Tag" or any("tag" in t for t in target_tags):  # ✅ 新增：标签语义边
            rel_type = "HAS_TAG"
            color = "#10b981"  # 绿

        # ====== 创建语义关系 ======
        try:
            tx.run(f"""
                MATCH (a:Note {{id: $source}})
                MATCH (b:Note {{id: $target}})
                MERGE (a)-[r:{rel_type} {{
                    weight: $weight,
                    rel_type: $rel_type,
                    color: $color
                }}]->(b)
            """, 
                source=source_id,
                target=target_id,
                weight=weight,
                rel_type=rel_type,
                color=color
            )
        except Exception as ex:
            print(f"❌ 创建边失败: {e}, 错误信息: {ex}")

    print(f"\n✅ 成功导入 {len(edges)} 条边（异常条目已跳过）\n")
     # ======= 语义概念聚类（Concept 生成） =======
    try:
        print("\n🧠 开始基于节点 embedding 生成 Concept 聚类...")

        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import normalize
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np
        import re

        # 1️⃣ 过滤掉一级标签（area/project/reflect/concept/resource）
        exclude_tags = {"area", "project", "reflect", "concept", "resource"}
        filtered_embeds = {
            nid: emb for nid, emb in embeds.items()
            if emb is not None and all(
                t not in exclude_tags for t in meta.get(nid, {}).get("tags", [])
            )
        }

        if not filtered_embeds:
            print("⚠️ 无有效 embedding，可跳过 Concept 聚类")
            return

        node_ids = list(filtered_embeds.keys())
        vectors = np.array([np.array(filtered_embeds[i]) for i in node_ids])

        # 2️⃣ 归一化（防止 cosine 报 Negative values 错）
        vectors = normalize(vectors, norm='l2')

        # 3️⃣ DBSCAN 聚类
        clustering = DBSCAN(eps=0.15, min_samples=2, metric="cosine")
        clustering.fit(vectors)
        labels = clustering.labels_

        clusters = {}
        for nid, label in zip(node_ids, labels):
            if label == -1:
                continue
            clusters.setdefault(label, []).append(nid)

        print(f"🌟 共检测到 {len(clusters)} 个潜在 Concept 聚类")

        # 4️⃣ 计算每个簇的平均相似度
        sim_matrix = cosine_similarity(vectors)
        cluster_sims = {}
        for cid, ids in clusters.items():
            if len(ids) < 2:
                cluster_sims[cid] = 1.0
                continue
            idxs = [node_ids.index(i) for i in ids]
            avg_sim = float(np.mean(sim_matrix[np.ix_(idxs, idxs)]))
            cluster_sims[cid] = avg_sim
            print(f"🧩 Concept_{cid}: 平均相似度 = {avg_sim:.3f}")

        # 5️⃣ 构造 TF-IDF 语义命名 + 颜色 / type 标注
        for cid, ids in clusters.items():
            corpus = []
            for i in ids:
                mi = meta.get(i, {})
                title = str(mi.get("title", "") or "")
                tags_list = mi.get("tags", [])
                tags = " ".join([t for t in tags_list if isinstance(t, str)])
                text = f"{title} {tags}".strip()
                text = re.sub(r"[^a-zA-Z\u4e00-\u9fa5\s]+", " ", text)
                if text.strip():
                    corpus.append(text)

            # ---- TF-IDF ----
            top_words = []
            if corpus:
                try:
                    vec = TfidfVectorizer(max_features=50)
                    X = vec.fit_transform(corpus)
                    mean_tfidf = np.asarray(X.mean(axis=0)).ravel()
                    vocab = np.array(vec.get_feature_names_out())
                    top_idx = mean_tfidf.argsort()[::-1][:3]
                    top_words = [w for w in vocab[top_idx] if w]
                except Exception as ex:
                    print(f"⚠️ TF-IDF 命名失败 ({ex})，使用默认名。")

            # ---- 聚簇语义命名 ----
            if top_words:
                raw_name = f"Concept_{cid}_" + "_".join(top_words)
            else:
                raw_name = f"Concept_{cid}_semantic_cluster"
            concept_name = re.sub(r"[^a-zA-Z0-9_\u4e00-\u9fa5]+", "_", raw_name)[:120]

            # ---- 平均相似度评估 ----
            avg_sim = cluster_sims.get(cid, 0)
            if avg_sim >= 0.85:
                color = "#bdbdbd"       # 灰色：合理聚类
                semantic_type = "semantic_cluster"
            else:
                color = "#f6c344"       # 黄色：语义漂移
                semantic_type = "semantic_resource"

            # ---- 写入 Concept 节点 ----
            concept_emb = np.mean(
                [filtered_embeds[i] for i in ids if i in filtered_embeds], axis=0
            ).tolist()

            tx.run("""
                MERGE (c:Concept {name:$name})
                SET c.color=$color,
                    c.type=$type,
                    c.similarity=$sim,
                    c.embedding=$embedding,
                    c.createdAt=datetime()
            """, name=concept_name, color=color, type=semantic_type,
                sim=avg_sim, embedding=concept_emb)

            # ---- 建立 Note → Concept 关系 ----
            for nid in ids:
                tx.run("""
                    MATCH (n:Note {id:$nid}), (c:Concept {name:$cname})
                    MERGE (n)-[:EXTENDS {weight:1.0, color:$color}]->(c)
                """, nid=nid, cname=concept_name, color=color)

        print(f"\n✅ 成功生成 {len(clusters)} 个 Concept 节点并建立语义关联。\n")


    except Exception as ex:
        print(f"❌ Concept 聚类阶段出错: {ex}")


    print(f"\n✅ 成功导入 {len(edges)} 条边（异常条目已跳过）\n")



# ========== 执行导入 ==========
with driver.session() as session:
    session.execute_write(import_data, nodes, edges, meta_data, embeddings)

log("✅ 导入成功，所有笔记与向量已导入 Neo4j！")
