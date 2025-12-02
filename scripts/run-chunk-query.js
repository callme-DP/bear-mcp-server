#!/usr/bin/env node
import 'dotenv/config';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { buildChunkQueryEngine, summarizeChunkResult, defaultQueryExamplePrompt } from '../src/rag/chunk/queryEngine.js';

async function main() {
  const query = process.argv.slice(2).join(' ') || defaultQueryExamplePrompt();
  const { queryEngine } = await buildChunkQueryEngine();
  const result = await queryEngine.query(query);

  console.log('Q:', query);
  console.log('\nResponse:\n');
  console.log(result.response);

  if (result.sourceNodes?.length) {
    console.log('\n--- Source chunks with context ---');
    console.log(summarizeChunkResult(result.sourceNodes));
  }

  // Persist result to out/chunk-query/YYYYMMDD-HHmmss.json
  try {
    const __filename = fileURLToPath(import.meta.url);
    const __dirname = path.dirname(__filename);
    const outDir = path.resolve(__dirname, '../out/chunk-query');
    await fs.mkdir(outDir, { recursive: true });
    const ts = new Date().toISOString().replace(/[-:]/g, '').slice(0, 15); // YYYYMMDDTHHmm
    const filePath = path.join(outDir, `${ts}.json`);
    const payload = {
      query,
      response: result.response,
      sourceNodes: result.sourceNodes?.map((item) => ({
        score: item.score,
        metadata: item.node?.metadata,
        text: item.node?.text,
      })),
    };
    await fs.writeFile(filePath, JSON.stringify(payload, null, 2), 'utf8');
    console.log(`\n[saved] ${filePath}`);
  } catch (err) {
    console.error('Failed to persist chunk query output:', err);
  }
}

main().catch((error) => {
  console.error('Chunk-level query failed:', error);
  process.exit(1);
});
