import fs from 'fs';
import path from 'path';
import sqlite3 from 'sqlite3';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const jsonPath = path.join(__dirname, '..', 'out', 'metabase_notes_concepts.json');
const dbPath = path.join(__dirname, '..', 'out', 'metabase_notes_concepts.db');

const loadJson = () => JSON.parse(fs.readFileSync(jsonPath, 'utf8'));

const initDb = () => new sqlite3.Database(dbPath);

const main = () => {
  const data = loadJson();
  const db = initDb();

  db.serialize(() => {
    db.run('PRAGMA journal_mode = WAL');
    db.run('DROP TABLE IF EXISTS metabase_notes_concepts');
    db.run(
      `CREATE TABLE metabase_notes_concepts (
        note_id TEXT PRIMARY KEY,
        idx INTEGER,
        title TEXT,
        area TEXT,
        concepts TEXT,
        tags TEXT,
        creation_date TEXT
      )`
    );

    const stmt = db.prepare(
      'INSERT INTO metabase_notes_concepts (note_id, idx, title, area, concepts, tags, creation_date) VALUES (?, ?, ?, ?, ?, ?, ?)'
    );

    data.forEach((row) => {
      const concepts = Array.isArray(row.concepts) ? row.concepts : [];
      // Skip rows without concepts to avoid empty concept edges in downstream charts
      if (concepts.length === 0) return;

      stmt.run(
        row.note_id,
        row.idx,
        row.title,
        row.area,
        JSON.stringify(concepts),
        JSON.stringify(row.tags || []),
        row.creation_date || null
      );
    });

    stmt.finalize();
  });

  db.close((err) => {
    if (err) {
      console.error('Error closing database:', err.message);
    } else {
      console.log(`Wrote ${data.length} rows to ${dbPath}`);
    }
  });
};

main();
