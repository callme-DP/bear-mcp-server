import { getLlamaIndex } from './loadChunkNodes.js';

export async function createNeo4jGraphStore(options = {}) {
  const llamaIndex = await getLlamaIndex();
  const { Neo4jGraphStore } = llamaIndex;

  if (!Neo4jGraphStore) {
    throw new Error('Neo4jGraphStore is not available in this llamaindex build.');
  }

  return new Neo4jGraphStore({
    url: options.url || process.env.NEO4J_URI || 'bolt://localhost:7687',
    username: options.username || process.env.NEO4J_USER || 'neo4j',
    password: options.password || process.env.NEO4J_PASSWORD,
    database: options.database || process.env.NEO4J_DATABASE,
  });
}
