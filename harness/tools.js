import { initContext, handleTool } from "../src/handle.js";

const MOCK_NOTES = [
  {
    id: "mock-1",
    title: "AI Daily Notes",
    content: "AI tag note about model evaluation, prompt design, and workflow iteration.",
    tags: ["AI", "project/agent"],
  },
  {
    id: "mock-2",
    title: "Agent Harness Ideas",
    content: "Design a think-act-observe loop and add retries plus validation checks.",
    tags: ["AI", "engineering"],
  },
];

export class BearTools {
  constructor() {
    this.ready = false;
    this.mockMode = false;
    this._mockNotes = MOCK_NOTES;
  }

  async init() {
    try {
      await initContext();
      this.ready = true;
      this.mockMode = false;
      console.log("[tools] connected to Bear MCP handleTool context");
    } catch (err) {
      this.ready = false;
      this.mockMode = true;
      console.log(`[tools] init failed, fallback to mock: ${err.message}`);
    }
  }

  async searchNotesByTag(tag, limit = 5) {
    if (!this.ready) {
      return this._mockSearchByTag(tag, limit);
    }

    try {
      const broad = await handleTool("search_notes", {
        query: `#${tag}`,
        limit,
        semantic: true,
      });
      let notes = Array.isArray(broad?.notes) ? broad.notes : [];
      if (notes.length === 0) {
        const keyword = await handleTool("search_notes", {
          query: tag,
          limit,
          semantic: false,
        });
        notes = Array.isArray(keyword?.notes) ? keyword.notes : [];
      }
      return { notes, source: "mcp.search_notes" };
    } catch (err) {
      console.log(`[tools] search failed, fallback to mock: ${err.message}`);
      this.mockMode = true;
      return this._mockSearchByTag(tag, limit);
    }
  }

  async getNote(noteId) {
    if (!this.ready) {
      return this._mockGetNote(noteId);
    }

    try {
      const result = await handleTool("get_note", { id: noteId });
      if (result?.note) {
        return { note: result.note, source: "mcp.get_note" };
      }
      return this._mockGetNote(noteId);
    } catch (err) {
      console.log(`[tools] get_note failed, fallback to mock: ${err.message}`);
      this.mockMode = true;
      return this._mockGetNote(noteId);
    }
  }

  _mockSearchByTag(tag, limit) {
    const matched = this._mockNotes
      .filter((n) => (n.tags || []).some((t) => t.toLowerCase() === tag.toLowerCase()))
      .slice(0, limit);
    return { notes: matched, source: "mock.search" };
  }

  _mockGetNote(noteId) {
    const note = this._mockNotes.find((n) => n.id === noteId) || null;
    return { note, source: "mock.get_note" };
  }
}
