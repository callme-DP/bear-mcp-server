# -*- coding: utf-8 -*-
"""
语义星图 Dashboard（v2）
功能：
1) 加载向量与元数据（支持 src/exports/*.json 或 src/note_vectors/*.npy + meta.json）
2) 首次运行自动计算/缓存 UMAP 3D 坐标（exports/umap_coords_3d.npy）
3) 交互：查询框语义搜索、TopK 高亮、标签筛选、连线开关、点击节点显示详情
4) 未命中节点自动变灰，命中节点按相似度着色与放大
"""

import os
import json
import math
import pathlib
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

# 降维
try:
    import umap
except Exception:
    # 如未安装 umap-learn：pip install umap-learn
    raise

# 轻量可视化 & 交互
from dash import Dash, dcc, html, Input, Output, State, no_update
import plotly.graph_objects as go

# 句向量模型（离线友好）
from sentence_transformers import SentenceTransformer


# ----------------------------
# 路径与文件探测
# ----------------------------
ROOT = pathlib.Path(__file__).resolve().parent
EXPORTS_DIR = ROOT / "exports"
NV_DIR = ROOT / "note_vectors"

EMB_JSON = EXPORTS_DIR / "embeddings.json"        # [[...384...], ...]
META_JSON = EXPORTS_DIR / "meta.json"             # [{"id","title","tags",...}, ...]
GRAPH_JSON = EXPORTS_DIR / "graph.json"           # {"nodes":[...], "edges":[{"from","to",...}]}
UMAP_CACHE = EXPORTS_DIR / "umap_coords_3d.npy"   # (N, 3) 缓存

EMB_NPY = NV_DIR / "embeddings.npy"               # 备选 (N,384)
NV_META_JSON = NV_DIR / "meta.json"               # 备选（你之前导出的 meta 列表）
NV_IDS_JSON = NV_DIR / "note_vectors.json"        # 备选（索引->uuid 映射，必要时兜底）


# ----------------------------
# 数据加载
# ----------------------------
def load_embeddings_and_meta() -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    支持两种来源：
      A) exports/embeddings.json + exports/meta.json
      B) note_vectors/embeddings.npy + note_vectors/meta.json(可选) + note_vectors.json(兜底)
    """
    # 优先 A：JSON
    if EMB_JSON.exists():
        with open(EMB_JSON, "r", encoding="utf-8") as f:
            X = np.array(json.load(f), dtype=np.float32)
        if META_JSON.exists():
            meta = json.load(open(META_JSON, "r", encoding="utf-8"))
        else:
            # 构造最小元数据
            meta = [{"id": str(i), "title": f"Note {i}", "tags": []} for i in range(X.shape[0])]
        return X, meta

    # 其次 B：NPY
    if EMB_NPY.exists():
        X = np.load(str(EMB_NPY))
        # meta 优先取 NV_META_JSON，否则用 NV_IDS_JSON 兜底
        if NV_META_JSON.exists():
            meta = json.load(open(NV_META_JSON, "r", encoding="utf-8"))
            # 确保长度一致；不一致就截断/补全
            if len(meta) != X.shape[0]:
                meta = (meta + [{"id": str(i), "title": f"Note {i}", "tags": []}
                                for i in range(len(meta), X.shape[0])])[: X.shape[0]]
        else:
            ids_map = {}
            if NV_IDS_JSON.exists():
                ids_map = json.load(open(NV_IDS_JSON, "r", encoding="utf-8"))
            meta = []
            for i in range(X.shape[0]):
                nid = ids_map.get(str(i), str(i))
                meta.append({"id": nid, "title": f"Note {i}", "tags": []})
        return X, meta

    raise FileNotFoundError(
        "未找到向量文件。请确认以下其一存在：\n"
        f" - {EMB_JSON} (+ {META_JSON})\n"
        f" - {EMB_NPY} (+ {NV_META_JSON} 或 {NV_IDS_JSON})\n"
    )


def load_edges_id_pairs(meta: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """从 exports/graph.json 读取边，返回索引对 (i,j)。没有则返回空"""
    if not GRAPH_JSON.exists():
        return []
    g = json.load(open(GRAPH_JSON, "r", encoding="utf-8"))
    id2idx = {m.get("id", str(i)): i for i, m in enumerate(meta)}

    pairs = []
    for e in g.get("edges", []):
        s = id2idx.get(e.get("from"))
        t = id2idx.get(e.get("to"))
        if s is not None and t is not None and s != t:
            pairs.append((s, t))
    return pairs


# ----------------------------
# 降维（含缓存）
# ----------------------------
def get_or_fit_umap_3d(X: np.ndarray, cache_path: pathlib.Path = UMAP_CACHE) -> np.ndarray:
    if cache_path.exists():
        try:
            coords = np.load(str(cache_path))
            if coords.shape[0] == X.shape[0] and coords.shape[1] == 3:
                return coords
        except Exception:
            pass
    reducer = umap.UMAP(n_components=3, n_neighbors=20, min_dist=0.1, metric="cosine", random_state=42)
    coords = reducer.fit_transform(X)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), coords)
    return coords


# ----------------------------
# 颜色与标签
# ----------------------------
DEFAULT_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

def top_tag_of(tags) -> str:
    """你可以替换为自己的规则：例如第一个顶层标签、或统计权重最高的标签等"""
    if not tags:
        return "未分类"
    # 标准化成 str
    if isinstance(tags, (list, tuple)):
        t = tags[0] if tags else "未分类"
    else:
        t = str(tags)
    return t or "未分类"


def build_dataframe(coords: np.ndarray, meta: List[Dict[str, Any]]) -> pd.DataFrame:
    titles = [m.get("title", f"Note {i}") for i, m in enumerate(meta)]
    tags = [m.get("tags", []) for m in meta]
    top_tags = [top_tag_of(t) for t in tags]
    ids = [m.get("id", str(i)) for i, m in enumerate(meta)]
    df = pd.DataFrame({
        "id": ids,
        "title": titles,
        "tags": tags,
        "top_tag": top_tags,
        "x": coords[:, 0],
        "y": coords[:, 1],
        "z": coords[:, 2],
        "idx": list(range(len(meta)))
    })
    return df


# ----------------------------
# 模型（离线优先）
# ----------------------------
def get_encoder():
    # 首先尝试完全离线（若你已本地缓存 all-MiniLM-L6-v2）
    try:
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu", local_files_only=True)
    except Exception:
        # 回退：允许联网拉取（若你的环境已缓存，也不会联网）
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")


def normalize_rows(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    return a / n


# ----------------------------
# Plotly 绘图
# ----------------------------
def base_figure(df: pd.DataFrame, palette_map: Dict[str, str], edges: List[Tuple[int, int]], show_edges: bool) -> go.Figure:
    fig = go.Figure()
    # 各类别独立散点，利于图例与分类开关
    cats = sorted(df["top_tag"].astype(str).unique())
    for i, c in enumerate(cats):
        d = df[df["top_tag"].astype(str) == c]
        color = palette_map.get(c, DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)])
        fig.add_trace(go.Scatter3d(
            x=d["x"], y=d["y"], z=d["z"],
            mode="markers",
            name=str(c),
            marker=dict(size=4, color=color, opacity=0.9),
            hovertemplate="<b>%{customdata[0]}</b><br>Tag:%{customdata[1]}<extra></extra>",
            customdata=np.stack([d["title"].values, d["top_tag"].astype(str).values], axis=-1)
        ))

    # 可选连线：数量过多可能影响帧率，这里做个限制
    if show_edges and edges:
        # 限制 3000 条以内
        max_edges = min(3000, len(edges))
        xs, ys, zs = [], [], []
        for (s, t) in edges[:max_edges]:
            xs += [df.loc[s, "x"], df.loc[t, "x"], None]
            ys += [df.loc[s, "y"], df.loc[t, "y"], None]
            zs += [df.loc[s, "z"], df.loc[t, "z"], None]
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(color="rgba(120,120,120,0.25)", width=1),
            hoverinfo="skip",
            name="relations"
        ))

    fig.update_layout(
        title="语义知识星图（UMAP-3D）",
        scene=dict(aspectmode="data"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=0, r=0, t=40, b=0),
        template="plotly_white",
    )
    return fig


def recolor_for_query(
    df: pd.DataFrame,
    fig: go.Figure,
    top_idx: np.ndarray,
    sims: np.ndarray,
    dim_others: bool = True
) -> go.Figure:
    """
    将命中的节点（top_idx）放大、按相似度上色；其他节点降为灰/透明
    """
    # 构建一个 idx->(size, color, opacity) 的查表
    top_set = set(top_idx.tolist())
    sim_min, sim_max = float(np.min(sims)), float(np.max(sims)) if len(sims) else (0.0, 1.0)

    def sim_to_color(s: float) -> str:
        # Reds colorscale 手工取值（简单起见）
        # 从浅到深：#fee5d9 -> #a50f15
        stops = [
            (0.0, "#fee5d9"),
            (0.25, "#fcbba1"),
            (0.5, "#fc9272"),
            (0.75, "#ef3b2c"),
            (1.0, "#a50f15"),
        ]
        if sim_max - sim_min < 1e-9:
            t = 1.0
        else:
            t = (s - sim_min) / (sim_max - sim_min + 1e-12)
        for j in range(1, len(stops)):
            if t <= stops[j][0]:
                return stops[j][1]
        return stops[-1][1]

    size_map = np.full(len(df), 3.0)  # 默认大小
    color_map = np.array(["#BDBDBD"] * len(df), dtype=object)  # 默认灰
    alpha_map = np.full(len(df), 0.15) if dim_others else np.full(len(df), 0.9)

    for rank, (idx, sim) in enumerate(zip(top_idx, sims)):
        size_map[idx] = 7.5 if rank < 5 else 6.0   # Top5 更大
        color_map[idx] = sim_to_color(float(sim))
        alpha_map[idx] = 0.95

    # 应用到每个 trace（按分类拆分的 trace）
    # 注意：我们只改样式，不改数据点顺序
    pt_cursor = 0
    for tr in fig.data:
        if isinstance(tr, go.Scatter3d) and tr.mode == "markers":
            npts = len(tr.x)
            # 当前 trace 对应的全局 idx 范围
            xs = df.iloc[pt_cursor: pt_cursor + npts]
            loc_idx = xs["idx"].values
            # 组装新的样式
            tr.marker.size = [float(size_map[i]) for i in loc_idx]
            tr.marker.color = [str(color_map[i]) for i in loc_idx]
            tr.marker.opacity = [float(alpha_map[i]) for i in loc_idx]
            pt_cursor += npts
        elif isinstance(tr, go.Scatter3d) and tr.mode == "lines":
            # 线条整体降暗（命中节点占比越高越亮，你也可以做更复杂逻辑）
            tr.line.color = "rgba(120,120,120,0.12)" if dim_others else "rgba(120,120,120,0.25)"

    return fig


# ----------------------------
# 主程序：加载数据、建模、Dash 布局与回调
# ----------------------------
print("✅ 正在加载数据...")
X_raw, meta = load_embeddings_and_meta()
N, D = X_raw.shape
print(f"向量矩阵 shape: ({N}, {D})")

print("✅ 归一化向量...")
X = normalize_rows(X_raw)

print("✅ UMAP 降维/加载缓存...")
coords3d = get_or_fit_umap_3d(X)

print("✅ 准备 DataFrame...")
df = build_dataframe(coords3d, meta)

print("✅ 加载可选连线...")
edge_pairs = load_edges_id_pairs(meta)  # [(i,j), ...]

# 颜色映射表（类别->颜色）
cats = sorted(df["top_tag"].astype(str).unique())
palette_map = {c: DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)] for i, c in enumerate(cats)}

# 句向量编码器
print("✅ 加载查询编码器（如第一次会从本地缓存读取模型）...")
encoder = get_encoder()

# 初始图
SHOW_EDGES_DEFAULT = False
fig0 = base_figure(df, palette_map, edge_pairs, show_edges=SHOW_EDGES_DEFAULT)

# Dash 应用
app = Dash(__name__)
app.title = "语义星图 Dashboard"

app.layout = html.Div([
    html.H3("语义知识星图 · Dashboard", style={"margin": "6px 0 0 0"}),
    html.Div([
        dcc.Input(
            id="query-input",
            type="text",
            placeholder="输入关键词或短句进行语义高亮（Enter 或停顿触发）",
            debounce=True,
            style={"width": "52%", "padding": "8px", "fontSize": "16px"}
        ),
        html.Span("  Top-K: ", style={"marginLeft": "12px"}),
        dcc.Slider(id="topk", min=10, max=200, step=10, value=80,
                   tooltip={"always_visible": False}),
        html.Span("  标签筛选：", style={"marginLeft": "12px"}),
        dcc.Dropdown(
            id="tag-filter",
            options=[{"label": c, "value": c} for c in cats],
            value=None, clearable=True, placeholder="可选：只显示某一顶层标签"
        ),
        dcc.Checklist(
            id="edge-toggle",
            options=[{"label": "显示关系连线（多时可能影响帧率）", "value": "on"}],
            value=["on"] if SHOW_EDGES_DEFAULT else [],
            style={"marginLeft": "12px", "display": "inline-block"}
        ),
    ], style={"display": "grid", "gridTemplateColumns": "52% 10% 16% 18%", "alignItems": "center",
              "gap": "8px", "margin": "8px 0 8px 0"}),

    html.Div([
        dcc.Graph(id="semantic-graph", figure=fig0, clear_on_unhover=False, style={"height": "78vh"}),
        html.Div(id="side-info",
                 style={"width": "27%", "padding": "8px 12px", "borderLeft": "1px solid #eee",
                        "overflowY": "auto", "height": "78vh"})
    ], style={"display": "grid", "gridTemplateColumns": "73% 27%", "gap": "0px"}),

    dcc.Store(id="store-filter-mask")  # 存放标签筛选的布尔掩码
])


# 预计算：一个简单的标签筛选掩码
def mask_by_tag(tag: str) -> np.ndarray:
    if not tag:
        return np.ones(len(df), dtype=bool)
    return df["top_tag"].astype(str).values == str(tag)


@app.callback(
    Output("store-filter-mask", "data"),
    Input("tag-filter", "value")
)
def update_filter_mask(tag_value):
    mask = mask_by_tag(tag_value)
    return mask.tolist()


@app.callback(
    Output("semantic-graph", "figure"),
    Output("side-info", "children"),
    Input("query-input", "value"),
    Input("topk", "value"),
    Input("edge-toggle", "value"),
    Input("store-filter-mask", "data"),
    State("semantic-graph", "figure"),
    prevent_initial_call=False
)
def on_query(query, topk, edge_toggle, mask_list, cur_fig):
    show_edges = ("on" in (edge_toggle or []))
    mask = np.array(mask_list, dtype=bool) if mask_list is not None else np.ones(len(df), dtype=bool)

    # 先根据筛选重绘基图（提高交互连贯性）
    df_view = df[mask].copy()
    # 为了保持索引一致，这里不重建 df，只是让无关节点“看起来”被隐藏：通过置透明实现
    base = base_figure(df, palette_map, edge_pairs if show_edges else [], show_edges)

    # 若无查询，直接返回基图 + 简短提示
    if not query or not query.strip():
        # 将被过滤掉的节点置灰且透明
        if mask_list is not None:
            # 将 mask==False 的点做“隐藏效果”
            # 简单做法：整体降透明（也保持可选图例互斥）
            base = recolor_for_query(
                df, base,
                top_idx=np.where(mask)[0],         # 用“可见点”作为“高亮”
                sims=np.ones(int(mask.sum())),     # 无意义，只是给个非空
                dim_others=True
            )
        side = html.Div([
            html.H4("使用说明"),
            html.P("• 输入关键词或短句，系统会语义检索并高亮最相关的 Top-K 节点。"),
            html.P("• 可用标签筛选收敛范围；可打开『显示关系连线』辅助理解结构。"),
            html.P("• 点击节点右侧显示标题与标签。")
        ])
        return base, side

    # 语义查询
    qv = encoder.encode([query])
    qv = normalize_rows(qv)
    sims = cosine_similarity(qv, X)[0]  # shape=(N,)

    # 若存在标签筛选，只在 mask==True 的范围内取 Top-K
    idx_pool = np.where(mask)[0]
    sims_in = sims[idx_pool]
    k = int(max(10, min(int(topk or 80), len(idx_pool))))
    order = np.argsort(sims_in)[-k:][::-1]
    top_idx = idx_pool[order]
    top_sims = sims[top_idx]

    # 重着色
    fig = recolor_for_query(df, base, top_idx=top_idx, sims=top_sims, dim_others=True)

    # 侧栏信息：展示 Top-K 命中
    rows = []
    for i, (ti, ts) in enumerate(zip(top_idx[:30], top_sims[:30])):  # 侧栏显示最多 30 行
        m = meta[ti]
        row = html.Div([
            html.Div(f"#{i+1}  {m.get('title','无标题')}", style={"fontWeight": "600"}),
            html.Div("tags: " + ", ".join(map(str, m.get("tags", []))) if m.get("tags") else "tags: -",
                     style={"color": "#666", "fontSize": "12px"}),
            html.Div(f"score: {float(ts):.4f}", style={"color": "#999", "fontSize": "12px"}),
            html.Hr(style={"margin": "6px 0"})
        ])
        rows.append(row)

    side = html.Div([
        html.H4(f"查询：{query}"),
        html.Div(f"命中 Top-K={k}（仅对当前标签筛选范围）", style={"color": "#666", "marginBottom": "6px"}),
        html.Div(rows if rows else "（无结果）")
    ])
    return fig, side


if __name__ == "__main__":
    # 通过环境变量可快速改端口：PORT=8051 python src/semantic_dashboard_v2.py
    port = int(os.environ.get("PORT", "8050"))
    app.run(debug=False, host="127.0.0.1", port=port)
