import json
import networkx as nx
import numpy as np
import umap
from manim import *
from tqdm import tqdm
import pandas as pd
from pathlib import Path

# =========================
# 配置 — 颜色定义
# =========================
COLOR_NOTE = BLUE
COLOR_CONCEPT = YELLOW
COLOR_RESOURCE = GREEN
COLOR_EDGE = GREY_B

NODE_SIZE = 0.10  # 点大小

# =========================
# 加载数据（默认读取 note_vectors/*）
# =========================
DATA_DIR = Path(__file__).resolve().parent.parent / "note_vectors"
GRAPH_PATH = DATA_DIR / "graph.json"
META_PATH = DATA_DIR / "meta.json"
EMB_PATH = DATA_DIR / "embeddings.json"


def load_graph():
    with open(GRAPH_PATH, "r") as f:
        graph_data = json.load(f)
    with open(META_PATH, "r") as f:
        meta_data = json.load(f)
    with open(EMB_PATH, "r") as f:
        embeddings = json.load(f)

    # idx -> embedding
    emb_map = {i: emb for i, emb in enumerate(embeddings)}
    meta_map = {m["id"]: m for m in meta_data}

    nodes = []
    edges = []

    for node in graph_data["nodes"]:
        nid = node["id"]
        title = node.get("title", nid)
        node_type = node.get("type", "Note")
        category = node_type.lower()
        meta = meta_map.get(nid, {})
        emb = emb_map.get(meta.get("idx", -1)) if isinstance(meta.get("idx", -1), int) else None

        nodes.append(
            {
                "id": nid,
                "title": title,
                "category": category,
                "timestamp": node.get("created") or None,
                "embedding": emb,
            }
        )

    for edge in graph_data["edges"]:
        s = edge.get("from") or edge.get("source")
        t = edge.get("to") or edge.get("target")
        if s and t:
            edges.append((s, t))

    df_nodes = pd.DataFrame(nodes)
    return df_nodes, edges


# =========================
# UMAP 2D 降维
# =========================
def compute_umap(df_nodes):
    if "embedding" in df_nodes.columns and df_nodes["embedding"].notna().any():
        valid = df_nodes["embedding"].apply(lambda x: isinstance(x, list) and len(x) > 0)
        X = np.vstack(df_nodes[valid]["embedding"].values) if valid.any() else np.random.randn(len(df_nodes), 64)
    else:
        X = np.random.randn(len(df_nodes), 64)

    reducer = umap.UMAP(n_neighbors=12, min_dist=0.3, random_state=42)
    coords = reducer.fit_transform(X)
    df_nodes["x"] = coords[:, 0]
    df_nodes["y"] = coords[:, 1]
    return df_nodes


# =========================
# Manim 星图动画主体
# =========================
class SemanticGraph(Scene):
    def construct(self):
        # 1) 加载数据
        df_nodes, edges = load_graph()
        # 1.1) 处理时间字段，避免 str/int 比较异常
        df_nodes["timestamp_parsed"] = (
            pd.to_datetime(df_nodes["timestamp"], errors="coerce")
            .view("int64")
            .fillna(0)
        )
        df_nodes = df_nodes.sort_values("timestamp_parsed")
        df_nodes = compute_umap(df_nodes)

        # 2) 构建 NetworkX 图
        G = nx.Graph()
        for _, row in df_nodes.iterrows():
            G.add_node(row["id"], **row.to_dict())
        G.add_edges_from(edges)

        # 3) 构建 Manim 点位
        node_mobs = {}
        for _, row in df_nodes.iterrows():
            pos = np.array([row["x"], row["y"], 0]) / 3
            cat = row["category"]
            if cat == "note":
                color = COLOR_NOTE
            elif cat == "concept":
                color = COLOR_CONCEPT
            else:
                color = COLOR_RESOURCE
            dot = Dot(point=pos, radius=NODE_SIZE, color=color)
            dot.set_opacity(0)
            node_mobs[row["id"]] = dot

        # 4) 边对象
        edge_mobs = []
        for s, t in edges:
            if s in node_mobs and t in node_mobs:
                line = Line(
                    node_mobs[s].get_center(),
                    node_mobs[t].get_center(),
                    stroke_color=COLOR_EDGE,
                    stroke_width=1.5,
                )
                line.set_opacity(0)
                edge_mobs.append(line)

        # 5) 节点渐显
        animations = [FadeIn(node_mobs[row["id"]], run_time=0.15) for _, row in df_nodes.iterrows()]
        self.play(*animations)

        # 6) 边渐显
        edge_anims = [Create(e, run_time=0.15) for e in edge_mobs]
        self.play(*edge_anims)

        # 7) 呼吸动画
        pulse = AnimationGroup(
            *[node_mobs[n].animate.scale(1.1).set_opacity(1) for n in node_mobs],
            lag_ratio=0.05,
            rate_func=there_and_back,
            run_time=2,
        )
        self.play(pulse)

        # 8) 停留
        self.wait(1)
