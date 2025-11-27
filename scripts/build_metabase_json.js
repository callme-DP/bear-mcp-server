import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const metaPath = path.join(__dirname, '..', 'src', 'note_vectors', 'meta_v2.json');
const conceptsPath = path.join(__dirname, '..', 'out', 'llm_concepts.txt');
const outputPath = path.join(__dirname, '..', 'out', 'metabase_notes_concepts.json');

const readText = (filePath) => fs.readFileSync(filePath, 'utf8');

const parseConceptLines = (text) => {
  const map = new Map();
  text
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .forEach((line) => {
      const colonIndex = line.indexOf(':');
      if (colonIndex === -1) return;

      const rawTitle = line.slice(0, colonIndex).trim();
      const rawConcepts = line.slice(colonIndex + 1).trim();
      if (!rawTitle || !rawConcepts) return;

      const concepts = rawConcepts
        .split(',')
        .map((c) => c.trim())
        .filter(Boolean);

      map.set(rawTitle, concepts);
    });
  return map;
};

const pickAreaFromTags = (tags = []) => {
  if (!Array.isArray(tags) || tags.length === 0) return null;

  const sorted = [...tags].sort((a, b) => b.length - a.length);
  const sortedLower = sorted.map((tag) => tag.toLowerCase());

  for (let i = 0; i < sorted.length; i += 1) {
    const candidate = sorted[i];
    const candidateLower = sortedLower[i];
    const interactsWithOthers = sortedLower.some(
      (other, idx) =>
        idx !== i &&
        (candidateLower.includes(other) || other.includes(candidateLower)),
    );
    if (interactsWithOthers) return candidate;
  }

  return sorted[0];
};

const build = () => {
  const meta = JSON.parse(readText(metaPath));
  const conceptMap = parseConceptLines(readText(conceptsPath));

  const result = meta.map((note, index) => {
    const tags = Array.isArray(note.tags) ? note.tags : [];
    return {
      note_id: note.id,
      idx: typeof note.idx === 'number' ? note.idx : index,
      title: note.title,
      area: pickAreaFromTags(tags),
      concepts: conceptMap.get(note.title) || [],
      tags,
      creation_date: note.creation_date || null,
    };
  });

  fs.writeFileSync(outputPath, JSON.stringify(result, null, 2));
  console.log(`Generated ${result.length} records to ${outputPath}`);
};

build();
