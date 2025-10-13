import sqlite3 from 'sqlite3';
import path from 'path';
import os from 'os';
import { promisify } from 'util';
import { fileURLToPath } from 'url';
import fs from 'fs/promises';
import { pipeline } from '@xenova/transformers';
// Fix for CommonJS module import in ESM
import faissNode from 'faiss-node';
const { IndexFlatL2 } = faissNode;

// Get current file path for ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Setup SQLite with verbose mode
const sqlite = sqlite3.verbose();
const { Database } = sqlite;

// Default path to Bear's database
const defaultDBPath = path.join(
  os.homedir(),
  'Library/Group Containers/9K33E3U3T4.net.shinyfrog.bear/Application Data/database.sqlite'
);

// Path to the vector index - store in src directory
const INDEX_PATH = path.join(__dirname, 'note_vectors');

// Embedding model name
const EMBEDDING_MODEL = 'Xenova/all-MiniLM-L6-v2';

// Global variables for embedding model and vector index
let embedder = null;
let vectorIndex = null;
let noteIdMap = null;

// Get the database path from environment variable or use default
export const getDbPath = () => process.env.BEAR_DATABASE_PATH || defaultDBPath;

// Create and configure database connection
export const createDb = (dbPath) => {
  const db = new Database(dbPath, sqlite3.OPEN_READONLY, (err) => {
    if (err) {
      console.error('Error connecting to Bear database:', err.message);
      process.exit(1);
    }
    console.error('Connected to Bear Notes database at:', dbPath);
  });

  // Promisify database methods
  db.allAsync = promisify(db.all).bind(db);
  db.getAsync = promisify(db.get).bind(db);
  
  return db;
};

// Initialize the embedding model
export const initEmbedder = async () => {
  if (!embedder) {
    try {
      // Using Xenova's implementation of transformers
      console.error(`Initializing embedding model (${EMBEDDING_MODEL})...`);
      embedder = await pipeline('feature-extraction', EMBEDDING_MODEL);
      console.error('Embedding model initialized');
      return true;
    } catch (error) {
      console.error('Error initializing embedding model:', error);
      return false;
    }
  }
  return true;
};

// Load the vector index
export const loadVectorIndex = async () => {
  try {
    if (!vectorIndex) {
      // Check if index exists
      try {
      await fs.access(`${INDEX_PATH}.index`);
      
      // Load index using the direct file reading method
      vectorIndex = IndexFlatL2.read(`${INDEX_PATH}.index`);
        
        const idMapData = await fs.readFile(`${INDEX_PATH}.json`, 'utf8');
        noteIdMap = JSON.parse(idMapData);
        
        console.error(`Loaded vector index with ${vectorIndex.ntotal} vectors`);
        return true;
      } catch (error) {
        console.error('Vector index not found. Please run indexing first:', error.message);
        return false;
      }
    }
    return true;
  } catch (error) {
    console.error('Error loading vector index:', error);
    return false;
  }
};

// Create text embeddings
export const createEmbedding = async (text) => {
  if (!embedder) {
    const initialized = await initEmbedder();
    if (!initialized) {
      throw new Error('Failed to initialize embedding model');
    }
  }
  
  try {
    // Generate embeddings using Xenova transformers
    const result = await embedder(text, { 
      pooling: 'mean',
      normalize: true 
    });
    
    // Return the embedding as a regular array
    return Array.from(result.data);
  } catch (error) {
    console.error('Error creating embedding:', error);
    throw error;
  }
};

// Search for notes using semantic search
export const semanticSearch = async (db, query, limit = 10) => {
  try {
    // Ensure vector index is loaded
    if (!vectorIndex || !noteIdMap) {
      const loaded = await loadVectorIndex();
      if (!loaded) {
        throw new Error('Vector index not available. Please run indexing first.');
      }
    }
    
    // Create embedding for the query
    const queryEmbedding = await createEmbedding(query);
    
    // Search in vector index
    const { labels, distances } = vectorIndex.search(queryEmbedding, limit);
    
    // Get note IDs from the results
    const noteIds = labels.map(idx => noteIdMap[idx]).filter(id => id);
    
    if (noteIds.length === 0) {
      return [];
    }
    
    // Prepare placeholders for SQL query
    const placeholders = noteIds.map(() => '?').join(',');
    
    // Get full note details from database
    const notes = await db.allAsync(`
      SELECT 
        ZUNIQUEIDENTIFIER as id,
        ZTITLE as title,
        ZTEXT as content,
        ZSUBTITLE as subtitle,
        ZCREATIONDATE as creation_date
      FROM ZSFNOTE
      WHERE ZUNIQUEIDENTIFIER IN (${placeholders}) AND ZTRASHED = 0
      ORDER BY ZMODIFICATIONDATE DESC
    `, noteIds);
    
    // Get tags for each note
    for (const note of notes) {
      try {
        const tags = await db.allAsync(`
          SELECT ZT.ZTITLE as tag_name
          FROM Z_5TAGS ZNT
          JOIN ZSFNOTETAG ZT ON ZT.Z_PK = ZNT.Z_13TAGS
          JOIN ZSFNOTE ZN ON ZN.Z_PK = ZNT.Z_5NOTES
          WHERE ZN.ZUNIQUEIDENTIFIER = ?
        `, [note.id]);
        note.tags = tags.map(t => t.tag_name);
      } catch (tagError) {
        console.error(`Error fetching tags for note ${note.id}:`, tagError.message);
        note.tags = [];
      }
      
      // Convert Apple's timestamp (seconds since 2001-01-01) to standard timestamp
      if (note.creation_date) {
        // Apple's reference date is 2001-01-01, so add seconds to get UNIX timestamp
        note.creation_date = new Date((note.creation_date + 978307200) * 1000).toISOString();
      }
      
      // Store the semantic similarity score (lower distance is better)
      const idx = noteIds.indexOf(note.id);
      note.score = idx >= 0 ? 1 - distances[idx] : 0;
    }
    
    // Sort by similarity score
    return notes.sort((a, b) => b.score - a.score);
  } catch (error) {
    console.error('Semantic search error:', error);
    throw error;
  }
};

// Fallback to keyword search if vector search fails
export const searchNotes = async (db, query, limit = 10, useSemanticSearch = true) => {
  try {
    // Try semantic search first if enabled
    if (useSemanticSearch) {
      try {
        const semanticResults = await semanticSearch(db, query, limit);
        if (semanticResults && semanticResults.length > 0) {
          return semanticResults;
        }
      } catch (error) {
        console.error('Semantic search failed, falling back to keyword search:', error.message);
      }
    }
    
    // Fallback to keyword search
    const notes = await db.allAsync(`
      SELECT 
        ZUNIQUEIDENTIFIER as id,
        ZTITLE as title,
        ZTEXT as content,
        ZSUBTITLE as subtitle,
        ZCREATIONDATE as creation_date
      FROM ZSFNOTE
      WHERE ZTRASHED = 0 AND (ZTITLE LIKE ? OR ZTEXT LIKE ?)
      ORDER BY ZMODIFICATIONDATE DESC
      LIMIT ?
    `, [`%${query}%`, `%${query}%`, limit]);
    
    // Get tags for each note
    for (const note of notes) {
      try {
        const tags = await db.allAsync(`
          SELECT ZT.ZTITLE as tag_name
          FROM Z_5TAGS ZNT
          JOIN ZSFNOTETAG ZT ON ZT.Z_PK = ZNT.Z_13TAGS
          JOIN ZSFNOTE ZN ON ZN.Z_PK = ZNT.Z_5NOTES
          WHERE ZN.ZUNIQUEIDENTIFIER = ?
        `, [note.id]);
        
        note.tags = tags.map(t => t.tag_name);
      } catch (tagError) {
        console.error(`Error fetching tags for note ${note.id}:`, tagError.message);
        note.tags = [];
      }
      
      // Convert Apple's timestamp (seconds since 2001-01-01) to standard timestamp
      if (note.creation_date) {
        // Apple's reference date is 2001-01-01, so add seconds to get UNIX timestamp
        note.creation_date = new Date((note.creation_date + 978307200) * 1000).toISOString();
      }
    }
    
    return notes;
  } catch (error) {
    console.error('Search error:', error);
    throw error;
  }
};

// Retrieve a specific note by ID 按ID检索特定的笔记
export const retrieveNote = async (db, id) => {
  try {
    if (!id) {
      throw new Error('Note ID is required');
    }
    
    // Get the note by ID
    const note = await db.getAsync(`
      SELECT 
        ZUNIQUEIDENTIFIER as id,
        ZTITLE as title,
        ZTEXT as content,
        ZSUBTITLE as subtitle,
        ZCREATIONDATE as creation_date
      FROM ZSFNOTE
      WHERE ZUNIQUEIDENTIFIER = ? AND ZTRASHED = 0
    `, [id]);
    
    if (!note) {
      throw new Error('Note not found');
    }
    
    // Get tags for the note
    try {
      const tags = await db.allAsync(`
        SELECT ZT.ZTITLE as tag_name
        FROM Z_5TAGS ZNT
        JOIN ZSFNOTETAG ZT ON ZT.Z_PK = ZNT.Z_13TAGS
        JOIN ZSFNOTE ZN ON ZN.Z_PK = ZNT.Z_5NOTES
        WHERE ZN.ZUNIQUEIDENTIFIER = ?
      `, [note.id]);
      note.tags = tags.map(t => t.tag_name);
    } catch (tagError) {
      console.error(`Error fetching tags for note ${note.id}:`, tagError.message);
      note.tags = [];
    }
    
    // Convert Apple's timestamp (seconds since 2001-01-01) to standard timestamp
    if (note.creation_date) {
      // Apple's reference date is 2001-01-01, so add seconds to get UNIX timestamp
      note.creation_date = new Date((note.creation_date + 978307200) * 1000).toISOString();
    }
    
    return note;
  } catch (error) {
    console.error('Retrieve error:', error);
    throw error;
  }
};

// Get all tags
export const getAllTags = async (db) => {
  try {
    const tags = await db.allAsync('SELECT ZTITLE as name FROM ZSFNOTETAG');
    return tags.map(tag => tag.name);
  } catch (error) {
    console.error('Get tags error:', error);
    throw error;
  }
};

// RAG function to retrieve notes that are semantically similar to a query
export const retrieveForRAG = async (db, query, limit = 5) => {
  try {
    // Get semantically similar notes
    const notes = await semanticSearch(db, query, limit);
    
    // Format for RAG context
    return notes.map(note => ({
      id: note.id,
      title: note.title,
      content: note.content,
      tags: note.tags,
      score: note.score
    }));
  } catch (error) {
    console.error('RAG retrieval error:', error);
    // Fallback to keyword search
    const notes = await searchNotes(db, query, limit, false);
    return notes.map(note => ({
      id: note.id,
      title: note.title,
      content: note.content,
      tags: note.tags
    }));
  }
};

// 新增笔记
export async function insertNote(db, title, content) {
  try {
    const timestamp = Date.now() / 1000 - 978307200; // Bear 时间戳格式（2001 基准）
    const uuid = crypto.randomUUID();

    await db.runAsync(`
      INSERT INTO ZSFNOTE (
        Z_PK, Z_ENT, Z_OPT,
        ZUNIQUEIDENTIFIER, ZTITLE, ZTEXT, ZCREATIONDATE, ZMODIFICATIONDATE
      )
      VALUES (
        (SELECT MAX(Z_PK)+1 FROM ZSFNOTE), 4, 1,
        ?, ?, ?, ?, ?
      )
    `, [uuid, title, content, timestamp, timestamp]);

    console.log(`✅ 新建笔记成功: ${title}`);
    return { success: true, id: uuid };
  } catch (error) {
    console.error("❌ 插入笔记失败:", error);
    return { success: false, error: error.message };
  }
}

//今日洞察
export async function getTodayAndRelatedNotes(db, hours = 24, limit = 5) {
  // Step 1. 获取当天笔记
const localNow = new Date();
const todayStart = new Date(localNow.getFullYear(), localNow.getMonth(), localNow.getDate());
const startTimestamp = todayStart.getTime() / 1000 - 978307200; // Bear时间
  const todayNotes = await db.allAsync(`
    SELECT ZUNIQUEIDENTIFIER as id, ZTITLE as title, ZTEXT as content
    FROM ZSFNOTE
    WHERE ZTRASHED = 0 AND ZMODIFICATIONDATE >= ?
    ORDER BY ZMODIFICATIONDATE DESC
  `, [startTimestamp]);

  // Step 2. 合并当天内容用于后续embedding
  const todayContent = todayNotes.map(n => n.content).join("\n");

  // Step 3. 生成 embedding（如果无笔记则空）
  let related = [];
  if (todayContent.length > 0) {
    related = await semanticSearch(db, todayContent, limit);
  }

  // Step 4. 返回 GPT 可直接使用的上下文结构
  return {
    system_prompt: `
你是一位反思型 AI 助手（Lumi），负责帮助用户进行「每日洞察」。
你将基于以下两类数据输出深度反思报告：
1️⃣ 今日笔记（today）——代表今天的关注、情绪与思考主题。
2️⃣ 相关历史笔记（related）——代表这些主题在过去的延续与演变。

请分析：
- 今日笔记中最突出的主题、关键词、语气或情感。
- 与历史笔记相比，主题或态度有哪些演变。
- 这些变化可能代表用户的成长方向、困惑、或新的内在趋势。

输出格式：
【今日主题】
- …

【历史呼应】
- …

【🪞今日的洞察】
- 用 2～4 句话总结今日的核心洞察，风格温和、哲思。
  例如：「你今天反复提到‘系统’，这显示出你在思考自我秩序与自由的平衡。」`,
    context: {
      today: todayNotes,
      related
    }
  };
}