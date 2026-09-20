"""Transactional single-host repository. Each operation owns its SQLite connection."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def now():
    return datetime.now(timezone.utc).isoformat()


def uid():
    return str(uuid4())


class Repository:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS entities (
                    kind TEXT NOT NULL, id TEXT PRIMARY KEY, book_id TEXT,
                    parent_id TEXT, position INTEGER DEFAULT 0, data TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS entity_book ON entities(kind, book_id, position);
                CREATE INDEX IF NOT EXISTS entity_parent ON entities(kind, parent_id, position);
                CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
                INSERT INTO schema_version SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM schema_version);
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(str(self.path), timeout=30)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def put(self, kind, value, db=None):
        if db is None:
            with self.connect() as conn:
                return self.put(kind, value, conn)
        db.execute('''INSERT INTO entities(kind,id,book_id,parent_id,position,data) VALUES(?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET data=excluded.data,book_id=excluded.book_id,
            parent_id=excluded.parent_id,position=excluded.position''',
            (kind, value['id'], value.get('book_id'), value.get('parent_id'), value.get('position', 0),
             json.dumps(value, ensure_ascii=False)))
        return value

    def get(self, kind, entity_id, db=None):
        if db is None:
            with self.connect() as conn:
                return self.get(kind, entity_id, conn)
        row = db.execute('SELECT data FROM entities WHERE kind=? AND id=?', (kind, entity_id)).fetchone()
        return json.loads(row['data']) if row else None

    def list(self, kind, book_id=None, parent_id=None, db=None):
        if db is None:
            with self.connect() as conn:
                return self.list(kind, book_id, parent_id, conn)
        query, args = 'SELECT data FROM entities WHERE kind=?', [kind]
        if book_id is not None:
            query += ' AND book_id=?'
            args.append(book_id)
        if parent_id is not None:
            query += ' AND parent_id=?'
            args.append(parent_id)
        query += ' ORDER BY position, rowid'
        return [json.loads(x['data']) for x in db.execute(query, args).fetchall()]

    def delete(self, kind, entity_id, db=None):
        if db is None:
            with self.connect() as conn:
                return self.delete(kind, entity_id, conn)
        db.execute('DELETE FROM entities WHERE kind=? AND id=?', (kind, entity_id))
