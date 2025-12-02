import { createEmbedding, tokenizeForEmbedding } from '../../utils.js';
import { buildNoteOrderKey } from './loadChunkNodes.js';

export async function createChunkContextRetriever({
  vectorStore,
  nodesByChunkId,
  nodesByNoteOrder,
  topK = 20,
  neighborWindow = 2,
}) {
  const tokenizeQuery = (text) =>
    (text || '')
      .toLowerCase()
      .split(/[\s,，。.!！？?；;、]+/)
      .filter((t) => t.length > 1);

  class ChunkContextRetriever {
    constructor() {
      this.vectorStore = vectorStore;
      this.nodesByChunkId = nodesByChunkId;
      this.nodesByNoteOrder = nodesByNoteOrder;
      this.topK = topK;
      this.neighborWindow = neighborWindow;
      this.serviceContext = undefined; // optional hook consumed by RetrieverQueryEngine
    }

    async retrieve(params) {
      const rawQuery =
        typeof params === 'string'
          ? params
          : params?.query ?? params?.queryStr ?? '';
      const query = rawQuery == null ? '' : String(rawQuery);
      const preview = query.replace(/\s+/g, ' ').slice(0, 120);
      console.info('[chunk-rag] _retrieve start', {
        topK: this.topK,
        neighborWindow: this.neighborWindow,
        queryPreview: preview,
      });
      const queryTokens = tokenizeQuery(query);
      const queryEmbedding = await createEmbedding(query);
      const result = await this.vectorStore.query({ queryEmbedding, similarityTopK: this.topK });
      const tokenInfo = await tokenizeForEmbedding(query);

      const collected = new Map();
      const similarities = result.similarities || [];
      const normalizedQuery = query.toLowerCase();

      (result.ids || []).forEach((id, idx) => {
        const node = result.nodes?.[idx] || this.nodesByChunkId.get(id);
        if (!node) return;

        const score = similarities[idx] ?? null;
        collected.set(id, { node, score });

        const noteId = node.metadata?.note_id;
        const order = node.metadata?.order;
        if (noteId && typeof order === 'number') {
          for (let offset = 1; offset <= this.neighborWindow; offset += 1) {
            for (const delta of [-offset, offset]) {
              const neighborKey = buildNoteOrderKey(noteId, order + delta);
              if (!neighborKey) continue;
              const neighborNode = this.nodesByNoteOrder.get(neighborKey);
              if (neighborNode && !collected.has(neighborNode.metadata?.chunk_id)) {
                collected.set(
                  neighborNode.metadata?.chunk_id,
                  { node: neighborNode, score: score ?? undefined },
                );
              }
            }
          }
        }
      });

      // Keyword fallback: scan all chunks for token matches if recall is empty/low.
      if (queryTokens.length) {
        const keywordHits = [];
        for (const [chunkId, node] of nodesByChunkId.entries()) {
          if (collected.has(chunkId)) continue;
          const text = (node?.text || '').toLowerCase();
          if (!text) continue;
          const hit = queryTokens.some((tok) => text.includes(tok));
          if (hit) {
            keywordHits.push({ node, score: undefined, keywordHit: true });
          }
        }
        // If没有任何向量召回，或召回过少(<3)，用关键词兜底前若干条。
        const needKeywords = result.ids?.length === 0 || collected.size < 3;
        if (needKeywords && keywordHits.length) {
          keywordHits.slice(0, this.topK).forEach((item) => {
            const cid = item.node?.metadata?.chunk_id;
            if (cid && !collected.has(cid)) {
              collected.set(cid, item);
            }
          });
        }
      }

      // Debug: check expected chunk presence in base hits (before neighbors/keyword)
      const baseNodes = (result.ids || [])
        .map((id, idx) => result.nodes?.[idx] || this.nodesByChunkId.get(id))
        .filter(Boolean);
      const expected = '00AD9C3C-AE81-4A25-BC6E-57CF781173F9::0';
      console.log('[debug] expected chunk in baseHits?', baseNodes.some((n) => n?.metadata?.chunk_id === expected));

      const prioritized = Array.from(collected.values()).sort((a, b) => {
        const aText = (a.node?.text || '').toLowerCase();
        const bText = (b.node?.text || '').toLowerCase();
        const aHit = aText.includes(normalizedQuery) ? 1 : 0;
        const bHit = bText.includes(normalizedQuery) ? 1 : 0;
        if (aHit !== bHit) return bHit - aHit; // exact/substring hit first
        const aKeyword = a.keywordHit ? 1 : 0;
        const bKeyword = b.keywordHit ? 1 : 0;
        if (aKeyword !== bKeyword) return bKeyword - aKeyword; // keyword fallback second
        return (b.score ?? 0) - (a.score ?? 0);
      });

      console.info('[chunk-rag] _retrieve collected chunks', {
        baseHits: result.ids?.length || 0,
        withNeighbors: collected.size,
        prioritizedTop: prioritized.slice(0, 3).map((item) => item.node?.metadata?.chunk_id),
        tokens: tokenInfo
          ? { count: tokenInfo.tokenCount, head: tokenInfo.tokens.slice(0, 32) }
          : undefined,
      });
      return prioritized;
    }
  }

  return new ChunkContextRetriever();
}
