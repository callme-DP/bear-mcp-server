# src/visualize_semantic_space.py
# 读取 exports/embeddings.json + meta.json，绘制 UMAP/TSNE 语义空间
import json, os, sys
import numpy as np
import plotly.express as px
import pandas as pd

EXPORT_DIR = os.path.join(os.path.dirname(__file__), "exports")
EMB_PATH   = os.path.join(EXPORT_DIR, "embeddings.json")
META_PATH  = os.path.join(EXPORT_DIR, "meta.json")
OUT_PNG    = os.path.join(EXPORT_DIR, "semantic_space.png")

def load_data():
    with open(EMB_PATH, "r", encoding="utf-8") as f:
        emb = np.array(json.load(f), dtype=np.float32)  # (N, 384)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return emb, meta

def reduce_2d(emb):
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
        Z = reducer.fit_transform(emb)
        method = "UMAP"
    except Exception:
        from sklearn.manifold import TSNE
        Z = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=42).fit_transform(emb)
        method = "TSNE"
    return Z, method

# ... 在 main() 函数 reduce_2d 之后添加：
emb, meta = load_data()
Z, method = reduce_2d(emb)

# 组装 DataFrame
df = pd.DataFrame({
    "x": Z[:, 0],
    "y": Z[:, 1],
    "title": [m["title"] for m in meta],
    "tag": [", ".join(m.get("tags", [])) for m in meta],
    "top_tag": [m.get("top_tag") for m in meta]
})

fig = px.scatter(
    df, x="x", y="y",
    color="top_tag",
    hover_data=["title", "tag"],  # ✅ 悬停显示 title 与 tags
    title=f"Semantic Space ({method}) · N={len(df)}",
    width=900, height=700
)

fig.update_traces(marker=dict(size=6, opacity=0.7))
fig.write_html(os.path.join(EXPORT_DIR, "semantic_space_interactive.html"))
print("✅ 已生成交互版 HTML：exports/semantic_space_interactive.html")

def pick_color(tags):
    # 简单地根据顶层 tag 上色（没有就用 "other"）
    if not tags:
        return "other"
    t = tags[0]
    return t.split("/")[0] if "/" in t else t

def main():
    if not os.path.exists(EMB_PATH):
        print("❌ 未找到 embeddings.json，请先运行：node src/export-graph.js")
        sys.exit(1)
    emb, meta = load_data()
    Z, method = reduce_2d(emb)

    # 颜色分组
    colors = list(map(lambda m: pick_color(m.get("tags") or []), meta))
    uniq = sorted(set(colors))
    palette = {c: i for i, c in enumerate(uniq)}
    cval = np.array([palette[c] for c in colors])

    cmap = plt.cm.tab10.colors  # ✅ 改为颜色列表

    plt.figure(figsize=(10, 8))
    for i, lbl in enumerate(uniq):
        idx = [j for j, c in enumerate(colors) if c == lbl]
        color = cmap[i % len(cmap)]
        plt.scatter(Z[idx, 0], Z[idx, 1], s=14, alpha=0.75, c=[color], label=lbl)

    plt.title(f"Semantic Space ({method})  ·  N={emb.shape[0]}")
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    plt.legend(title="Top Tag", loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=160)
    print("✅ 可视化完成：", OUT_PNG)

if __name__ == "__main__":
    main()
