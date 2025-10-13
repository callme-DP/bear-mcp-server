// src/handler.js
import {
  getDbPath,
  createDb,
  searchNotes,
  retrieveNote,
  getAllTags,
  loadVectorIndex,
  initEmbedder,
  retrieveForRAG,
  insertNote,
  getTodayAndRelatedNotes,
} from "./utils.js";

let db = null;
let hasSemanticSearch = false;

export async function initContext() {
  const dbPath = getDbPath();
  db = createDb(dbPath);
  const embedderOK = await initEmbedder();
  const indexOK = await loadVectorIndex();
  hasSemanticSearch = embedderOK && indexOK;
  return { db, hasSemanticSearch };
}

export async function modifyNoteTag(db, noteId, removeTag, addTag) {
  await db.run(`DELETE FROM tags WHERE note_id = ? AND name = ?`, [
    noteId,
    removeTag,
  ]);
  await db.run(`INSERT INTO tags (note_id, name) VALUES (?, ?)`, [
    noteId,
    addTag,
  ]);
  return { success: true };
}

// ✅ 新增：统一标签大小写
async function normalizeTagCase(db, style = "lowercase") {
  const tags = await getAllTags(db);
  const updates = [];

  for (const tag of tags) {
    const normalized = style === "lowercase" ? tag.toLowerCase() : tag;
    if (normalized !== tag) {
      const notes = await db.all(`SELECT note_id FROM tags WHERE name = ?`, [
        tag,
      ]);
      for (const note of notes) {
        await modifyNoteTag(db, note.note_id, tag, normalized);
        updates.push({ note_id: note.note_id, from: tag, to: normalized });
      }
    }
  }

  return { updated: updates };
}

// ✅ 新增：统一顶层结构前缀（中英文混用）
async function unifyTagPrefix(db, mappings) {
  const updates = [];

  for (const [fromPrefix, toPrefix] of Object.entries(mappings)) {
    const likeClause = `${fromPrefix}/%`;
    const tags = await db.all(
      `SELECT DISTINCT name FROM tags WHERE name LIKE ?`,
      [likeClause]
    );

    for (const tag of tags.map((t) => t.name)) {
      const rest = tag.slice(fromPrefix.length + 1); // 移除旧前缀
      const newTag = `${toPrefix}/${rest}`;

      const notes = await db.all(`SELECT note_id FROM tags WHERE name = ?`, [
        tag,
      ]);
      for (const note of notes) {
        await modifyNoteTag(db, note.note_id, tag, newTag);
        updates.push({ note_id: note.note_id, from: tag, to: newTag });
      }
    }
  }

  return { updated: updates };
}

// ✅ 新增：标签合并
export async function mergeTags(db, args) {
  const { fromTags, toTag } = args;

  if (!Array.isArray(fromTags)) {
    throw new Error("fromTags must be an array.");
  }

  for (const tag of fromTags) {
    // 你之前应该有 update 或 delete 的语句
    const notes = await db.all(`SELECT note_id FROM tags WHERE name = ?`, [
      tag,
    ]);
    for (const { note_id } of notes) {
      // 替换旧 tag 为新 tag
      await db.run(`DELETE FROM tags WHERE note_id = ? AND name = ?`, [
        note_id,
        tag,
      ]);
      await db.run(`INSERT OR IGNORE INTO tags (note_id, name) VALUES (?, ?)`, [
        note_id,
        toTag,
      ]);
    }
  }

  return { success: true };
}

// ✅ 新增：根据标签精准查询
export async function findNotesByTag(db, tag, limit = 20) {
  const query = `
    SELECT n.id, n.title, n.content, n.creation_date, GROUP_CONCAT(t.name) as tags
    FROM notes n
    JOIN tags t ON n.id = t.note_id
    GROUP BY n.id
    HAVING SUM(CASE WHEN t.name = ? THEN 1 ELSE 0 END) > 0
    ORDER BY n.creation_date DESC
    LIMIT ?
  `;
  const rows = await db.all(query, [tag, limit]);
  return rows.map((row) => ({
    id: row.id,
    title: row.title,
    content: row.content,
    creation_date: row.creation_date,
    tags: row.tags?.split(",") || [],
  }));
}

// ✅ 主入口
export async function handleTool(tool, args) {
  if (!db) throw new Error("Database not initialized");

  switch (tool) {
    case "search_notes": {
      const { query, limit = 10, semantic = true } = args;
      const useSemantic = semantic && hasSemanticSearch;
      const notes = await searchNotes(db, query, limit, useSemantic);
      return {
        notes,
        searchMethod: useSemantic ? "semantic" : "keyword",
      };
    }
    case "get_note": {
      const { id } = args;
      const note = await retrieveNote(db, id);
      return { note };
    }
    case "get_tags": {
      const tags = await getAllTags(db);
      return { tags };
    }
    case "retrieve_for_rag": {
      if (!hasSemanticSearch) throw new Error("Semantic search not available");
      const { query, limit = 5 } = args;
      const context = await retrieveForRAG(db, query, limit);
      return { context };
    }
    case "modify_note_tag": {
      const { note_id, remove_tag, add_tag } = args;
      const updated = await modifyNoteTag(db, note_id, remove_tag, add_tag);
      console.log("✅ Modified note:", note.id, note.title);
      return { updated };
    }
    case "normalize_tag_case": {
      return await normalizeTagCase(db, args.style || "lowercase");
    }
    case "unify_tag_prefix": {
      return await unifyTagPrefix(db, args.mappings);
    }
    case "find_notes_by_tag": {
      const { tag, limit = 20 } = args;
      const notes = await findNotesByTag(db, tag, limit);
      return { notes };
    }
    case "create_note":
      return await insertNote(db, args.title, args.content);
    // 今日洞察
    case "daily_insight_context": {
      const hours = args && args.hours ? args.hours : 24;
      const limit = args && args.limit ? args.limit : 5;
      return await getTodayAndRelatedNotes(db, hours, limit);
    }

    /**
     * 例如：
     *  {
     * "tool": "merge_tags",
     * "args": {
     * "from_tags": ["Resource", "resource", "资源"],
     * "to_tag": "resource/阅读笔记"
     *   }
     * }
     */
    case "merge_tags": {
      return await mergeTags(db, args);
    }
    default:
      throw new Error(`Unknown tool: ${tool}`);
  }
}
