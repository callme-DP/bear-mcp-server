import fs from 'fs/promises';
import faiss from 'faiss-node';
import { getLlamaIndex } from './loadChunkNodes.js';

export async function createFaissChunkVectorStore({
  indexPath,
  idMapPath,
  nodesByChunkId,
  dimension = 384,
}) {
  const llamaIndex = await getLlamaIndex();
  const { VectorStoreBase } = llamaIndex;
  const { IndexFlatL2 } = faiss;

  const noopEmbedModel = {
    // Prevents llamaindex from trying to auto-embed via OpenAI when hydrating an index.
    transform: async () => {},
  };

  class FaissChunkVectorStore extends VectorStoreBase {
    constructor() {
      super(noopEmbedModel);
      this.indexPath = indexPath;
      this.idMapPath = idMapPath;
      this.dimension = dimension;
      this.nodesByChunkId = nodesByChunkId;
      this.storesText = true;
      this.index = null;
      this.idMap = null;
    }

    async init() {
      if (this.index && this.idMap) return;
      console.info('[chunk-rag] loading FAISS index', {
        indexPath: this.indexPath,
        idMapPath: this.idMapPath,
      });
      this.index = IndexFlatL2.read(this.indexPath);
      this.idMap = JSON.parse(await fs.readFile(this.idMapPath, 'utf8'));
      const total = typeof this.index?.ntotal === 'function' ? this.index.ntotal() : this.index?.ntotal;
      const dim = this.index?.d || this.dimension;
      const idCount = this.idMap ? Object.keys(this.idMap).length : 0;
      console.info('[chunk-rag] FAISS index ready', {
        dimension: dim,
        totalVectors: total,
        idCount,
      });
    }

    async add(embeddingResults) {
      // The index is treated as read-only. We still accept nodes to populate lookups.
      const ids = [];
      for (const item of embeddingResults) {
        const chunkId = item.id ?? item.metadata?.chunk_id;
        if (chunkId && item.metadata && this.nodesByChunkId && !this.nodesByChunkId.has(chunkId)) {
          this.nodesByChunkId.set(chunkId, item.metadata?.node ?? item.node);
        }
        if (chunkId) ids.push(chunkId);
      }
      return ids;
    }

    async delete() {
      throw new Error('FaissChunkVectorStore is read-only; delete is not supported.');
    }

    async query(query) {
      await this.init();
      const topK = query.similarityTopK ?? query.topK ?? 8;
      const { labels, distances } = this.index.search(query.queryEmbedding, topK);

      const ids = [];
      const similarities = [];
      const nodes = [];

      labels.forEach((row, idx) => {
        const chunkId = this.idMap?.[row];
        if (!chunkId) return;
        const node = this.nodesByChunkId?.get(chunkId) || null;
        ids.push(chunkId);
        similarities.push(1 - distances[idx]);
        nodes.push(node);
      });

      console.info('[chunk-rag] vector search result', {
        requestedTopK: topK,
        returned: ids.length,
        nodesWithText: nodes.filter(Boolean).length,
      });

      return { ids, similarities, nodes };
    }
  }

  return new FaissChunkVectorStore();
}
