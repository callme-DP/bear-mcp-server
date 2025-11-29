// 面向 Bear 笔记的分块工具：
// - 先按 Markdown 标题（#、##...）切一级语义段
// - 再对每段做 token 窗口切片（带重叠）
// - 清洗 markdown/链接/HTML，丢弃过短噪声片段

const DEFAULT_WINDOW = 500; // 单个 chunk 目标 token 数
const DEFAULT_OVERLAP = 80; // chunk 之间的重叠 token 数
const MIN_TOKENS = 4;       // 丢弃长度低于该阈值的片段；粗分词规则下约等于 4 个中文词/短语，通常不少于 8-12 个汉字（随标点/空格有浮动）

// 粗粒度分词：按空白/标点切分，兼容中英文
function tokenize(text) {
  return (text || "")
    .split(/[^a-zA-Z0-9\u4e00-\u9fa5]+/) // keep CJK + alnum
    .filter(Boolean);
}

// 清洗 markdown/HTML/链接等噪声
function cleanText(raw) {
  if (!raw) return "";
  return raw
    .replace(/```[\s\S]*?```/g, " ")       // fenced code
    .replace(/`[^`]*`/g, " ")              // inline code
    .replace(/\!\[[^\]]*\]\([^\)]*\)/g, " ") // images
    .replace(/\[[^\]]*\]\([^\)]*\)/g, " ")   // links
    .replace(/https?:\/\/\S+/g, " ")
    .replace(/<[^>]+>/g, " ")              // html tags
    .replace(/[#>*_~\-]{2,}/g, " ")        // markdown bullets/emphasis
    .replace(/\s+/g, " ")
    .trim();
}

// 按标题切段，返回 { heading, body }
function splitByHeading(text) {
  const lines = (text || "").split("\n");
  const sections = [];
  let current = { heading: "", body: [] };

  for (const line of lines) {
    const m = line.match(/^(#{1,6})\s+(.*)$/);
    if (m) {
      // push previous section if it has content
      if (current.body.length > 0) {
        sections.push({ heading: current.heading, body: current.body.join("\n") });
      }
      current = { heading: m[2].trim(), body: [] };
    } else {
      current.body.push(line);
    }
  }
  if (current.body.length > 0 || current.heading) {
    sections.push({ heading: current.heading, body: current.body.join("\n") });
  }

  // If no headings were found, fall back to single section.
  if (sections.length === 0) {
    sections.push({ heading: "", body: text || "" });
  }
  return sections;
}

// 将单个段落按窗口 + 重叠切成 chunks
function windowSection(noteId, heading, body, opts = {}, startOrder = 0) {
  const windowSize = opts.windowTokens || DEFAULT_WINDOW;
  const overlap = opts.overlapTokens || DEFAULT_OVERLAP;
  const tokens = tokenize(cleanText(body));
  const chunks = [];

  let start = 0;
  let order = startOrder;
  while (start < tokens.length) {
    const end = Math.min(tokens.length, start + windowSize);
    const slice = tokens.slice(start, end);
    if (slice.length >= MIN_TOKENS) {
      chunks.push({
        chunk_id: `${noteId}::${order}`,
        note_id: noteId,
        heading,
        order,
        content: slice.join(" "),
      });
      order += 1;
    }
    if (end === tokens.length) break;
    start = Math.max(end - overlap, start + 1);
  }
  return { chunks, nextOrder: order };
}

// 主入口：对一篇 Bear 笔记 { id, title, content } 进行分块
export function chunkNote(note, opts = {}) {
  const { id, content } = note || {};
  const body = typeof content === "string" ? content : "";
  const sections = splitByHeading(body);
  const chunks = [];
  let order = 0;
  for (const sec of sections) {
    const heading = sec.heading || note.title || "";
    const res = windowSection(id, heading, sec.body, opts, order);
    chunks.push(...res.chunks);
    order = res.nextOrder;
  }
  return chunks;
}

// 便捷方法：分块后直接调用外部 embedFn
export async function chunkAndEmbed(note, embedFn, opts = {}) {
  const chunks = chunkNote(note, opts);
  const withEmbeddings = [];
  for (const c of chunks) {
    const embedding = await embedFn(c.content);
    withEmbeddings.push({ ...c, embedding });
  }
  return withEmbeddings;
}

// 调试辅助：返回 chunk 摘要，便于快速检查分段效果
export function chunkPreview(note, opts = {}) {
  const maxLen = opts.maxLen || 160;
  const chunks = chunkNote(note, opts);
  const summaries = chunks.map((c) => {
    const tokens = (c.content || "").split(/\s+/).filter(Boolean);
    const snippet = tokens.join(" ");
    const short = snippet.length > maxLen ? `${snippet.slice(0, maxLen)}...` : snippet;
    return {
      chunk_id: c.chunk_id,
      heading: c.heading,
      order: c.order,
      tokens: tokens.length,
      preview: short,
    };
  });
  if (opts.log) {
    for (const s of summaries) {
      console.log(s);
    }
  }
  return summaries;
}

export default {
  chunkNote,
  chunkAndEmbed,
  chunkPreview,
};
