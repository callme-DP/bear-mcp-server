import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

let cachedLlamaIndex = null;

async function getLlamaIndex() {
  if (!cachedLlamaIndex) {
    cachedLlamaIndex = await import('llamaindex');
  }
  return cachedLlamaIndex;
}

export async function loadChunkNodes(metaPath) {
  const llamaIndex = await getLlamaIndex();
  const { TextNode } = llamaIndex;
  const raw = JSON.parse(await fs.readFile(metaPath, 'utf8'));

  return raw.map((chunk) =>
    new TextNode({
      id_: chunk.chunk_id,
      text: chunk.content,
      metadata: {
        chunk_id: chunk.chunk_id,
        note_id: chunk.note_id,
        heading: chunk.heading,
        order: chunk.order,
      },
    }),
  );
}

export function buildChunkLookups(nodes) {
  const nodesByChunkId = new Map();
  const nodesByNoteOrder = new Map();

  for (const node of nodes) {
    const chunkId = node.metadata?.chunk_id ?? node.id_;
    nodesByChunkId.set(chunkId, node);

    const key = buildNoteOrderKey(node.metadata?.note_id, node.metadata?.order);
    if (key) {
      nodesByNoteOrder.set(key, node);
    }
  }

  return { nodesByChunkId, nodesByNoteOrder };
}

export function buildNoteOrderKey(noteId, order) {
  if (!noteId || typeof order !== 'number') {
    return null;
  }
  return `${noteId}:${order}`;
}

export function resolveChunkPaths(baseDir) {
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);
  const defaultDir = path.resolve(__dirname, '../../../data/neo4j');
  const rootDir = baseDir || process.env.CHUNK_VECTORS_DIR || process.env.NOTE_VECTORS_DIR || defaultDir;

  return {
    metaPath: path.join(rootDir, 'meta_chunks.json'),
    indexPath: path.join(rootDir, 'note_vectors_chunk.index'),
    idMapPath: path.join(rootDir, 'note_vectors_chunk.json'),
  };
}

export { getLlamaIndex };
