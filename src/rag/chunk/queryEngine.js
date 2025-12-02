import { createChunkContextRetriever } from './chunkRetriever.js';
import { createFaissChunkVectorStore } from './faissChunkVectorStore.js';
import { buildChunkLookups, loadChunkNodes, resolveChunkPaths, getLlamaIndex } from './loadChunkNodes.js';

async function buildVectorIndex(llamaIndex, vectorStore, nodes) {
  const { VectorStoreIndex, storageContextFromDefaults } = llamaIndex;
  if (!VectorStoreIndex || !storageContextFromDefaults) {
    return null;
  }
  const storageContext = await storageContextFromDefaults({ vectorStore });
  if (storageContext?.docStore) {
    await storageContext.docStore.addDocuments(nodes);
  }
  return VectorStoreIndex.fromVectorStore(vectorStore, { storageContext });
}

function buildFallbackQueryEngine(retriever) {
  return {
    async query(queryStr) {
      const sourceNodes = await retriever.retrieve(queryStr);
      const responseText = sourceNodes
        .map((item) => item.node?.getContent?.() ?? item.node?.text)
        .filter(Boolean)
        .join('\n\n---\n');

      return { response: responseText, sourceNodes };
    },
  };
}

function chooseLLM(llamaIndex) {
  const { OpenAI, Gemini, Ollama } = llamaIndex;
  if (process.env.OPENAI_API_KEY && OpenAI) {
    const model = process.env.OPENAI_MODEL;
    console.info('[chunk-rag] using OpenAI LLM', { model: model || '(default)' });
    return new OpenAI(model ? { model } : undefined);
  }
  if ((process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY) && Gemini) {
    const model = process.env.GEMINI_MODEL;
    console.info('[chunk-rag] using Gemini LLM', { model: model || '(default)' });
    return new Gemini(model ? { model } : undefined);
  }
  if (Ollama) {
    const model = process.env.OLLAMA_MODEL || 'qwen2.5:7b';
    console.info('[chunk-rag] using Ollama LLM', { model });
    return new Ollama({ model });
  }
  return null;
}

export async function buildChunkQueryEngine(options = {}) {
  const llamaIndex = await getLlamaIndex();
  const paths = resolveChunkPaths(options.baseDir);

  const metaPath = options.metaPath || paths.metaPath;
  const indexPath = options.indexPath || paths.indexPath;
  const idMapPath = options.idMapPath || paths.idMapPath;

  console.info('[chunk-rag] buildChunkQueryEngine: resolved paths', {
    baseDir: options.baseDir || process.env.CHUNK_VECTORS_DIR || process.env.NOTE_VECTORS_DIR || '(default)',
    metaPath,
    indexPath,
    idMapPath,
  });

  const nodes = await loadChunkNodes(metaPath);
  console.info('[chunk-rag] loaded chunk nodes', { count: nodes.length, metaPath });
  const { nodesByChunkId, nodesByNoteOrder } = buildChunkLookups(nodes);
  console.info('[chunk-rag] built lookups', {
    chunkCount: nodesByChunkId.size,
    noteOrderKeys: nodesByNoteOrder.size,
  });

  const vectorStore = await createFaissChunkVectorStore({
    indexPath,
    idMapPath,
    nodesByChunkId,
    dimension: options.dimension || 384,
  });

  const retriever = await createChunkContextRetriever({
    vectorStore,
    nodesByChunkId,
    nodesByNoteOrder,
    topK: options.topK || 70,
    neighborWindow: options.neighborWindow ?? 1,
  });
  console.info('[chunk-rag] retriever ready', {
    topK: options.topK || 70,
    neighborWindow: options.neighborWindow ?? 1,
    dimension: options.dimension || 384,
  });

  let index = null;
  try {
    index = await buildVectorIndex(llamaIndex, vectorStore, nodes);
    if (index) {
      console.info('[chunk-rag] VectorStoreIndex hydrated from FAISS index');
    }
  } catch (error) {
    // Index construction is optional; fall back to retriever-only mode.
    console.error('Failed to hydrate VectorStoreIndex from FAISS. Continuing with retriever-only mode.', error);
  }

  const { RetrieverQueryEngine, ResponseSynthesizer, Settings } = llamaIndex;
  const llmDisabled =
    String(process.env.CHUNK_LLM_DISABLE || '').toLowerCase() === '1' ||
    String(process.env.CHUNK_LLM_DISABLE || '').toLowerCase() === 'true';

  if (RetrieverQueryEngine && ResponseSynthesizer && !llmDisabled) {
    const llm = chooseLLM(llamaIndex);
    try {
      if (llm && Settings) {
        Settings.llm = llm;
      }
      const responseSynthesizer = new ResponseSynthesizer({});
      const queryEngine = new RetrieverQueryEngine(retriever, responseSynthesizer);
      return { queryEngine, retriever, vectorStore, index, nodesByChunkId, nodesByNoteOrder };
    } catch (error) {
      console.error('Failed to build RetrieverQueryEngine. Falling back to simple query engine.', error);
    }
  } else if (llmDisabled) {
    console.info('[chunk-rag] LLM disabled via CHUNK_LLM_DISABLE; using fallback query engine');
  }

  return {
    queryEngine: buildFallbackQueryEngine(retriever),
    retriever,
    vectorStore,
    index,
    nodesByChunkId,
    nodesByNoteOrder,
  };
}

export function summarizeChunkResult(sourceNodes) {
  return sourceNodes
    .map((item) => {
      const meta = item.node?.metadata || {};
      const heading = meta.heading ? `【${meta.heading}】` : '';
      return `${heading}${item.node?.text}`;
    })
    .join('\n\n');
}

export function defaultQueryExamplePrompt() {
  return '外界不给反馈时，我应该如何稳住自己？';
}
