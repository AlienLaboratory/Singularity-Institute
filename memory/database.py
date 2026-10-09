import sqlite3
import json
import threading
from datetime import datetime, timezone
from pathlib import Path


class Database:

    def __init__(self, root: str):
        root = Path(root)

        self.path = root / "memory" / "institute.db"

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # ``check_same_thread=False`` is required because the single shared
        # connection is created on the main thread but is then used by the
        # worker thread(s) that drive ``agent.run(...)`` (see core/webui.py).
        # Without this, SQLite raises ``ProgrammingError: SQLite objects
        # created in a thread can only be used in that same thread``.
        # Concurrent use is still safe because WebRunner serialises tasks so
        # at most one thread touches the connection at a time; ``self._lock``
        # additionally guards each statement against accidental interleaving.
        self.connection = sqlite3.connect(
            self.path,
            check_same_thread=False
        )

        self.connection.row_factory = sqlite3.Row

        # Serialises all statements / commits so two threads can never issue
        # overlapping queries on the shared connection (prevents "database is
        # locked" / "recursive use" errors if callers ever overlap).
        self._lock = threading.RLock()

        self._initialize()

    def _initialize(self):

        cursor = self.connection.cursor()

        cursor.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            completed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(task_id) REFERENCES tasks(id)
        );

        CREATE TABLE IF NOT EXISTS tool_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER,
            tool TEXT NOT NULL,
            arguments TEXT NOT NULL,
            result TEXT,
            success INTEGER,
            created_at TEXT NOT NULL,
            FOREIGN KEY(task_id) REFERENCES tasks(id)
        );

        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER,
            event_type TEXT NOT NULL,
            data TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(task_id) REFERENCES tasks(id)
        );
        """)

        self.connection.commit()

    def _now(self):
        return datetime.now(timezone.utc).isoformat()

    def create_task(self, task: str):

        with self._lock:
            cursor = self.connection.execute(
                """
                INSERT INTO tasks
                (task, status, created_at)
                VALUES (?, ?, ?)
                """,
                (
                    task,
                    "running",
                    self._now()
                )
            )

            self.connection.commit()

            return cursor.lastrowid

    def complete_task(self, task_id: int, status="completed"):

        with self._lock:
            self.connection.execute(
                """
                UPDATE tasks
                SET status = ?, completed_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    self._now(),
                    task_id
                )
            )

            self.connection.commit()

    def add_message(
        self,
        task_id: int,
        role: str,
        content: str
    ):

        with self._lock:
            self.connection.execute(
                """
                INSERT INTO messages
                (task_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    task_id,
                    role,
                    content,
                    self._now()
                )
            )

            self.connection.commit()

    def add_tool_call(
        self,
        task_id: int,
        tool: str,
        arguments: dict,
        result: dict
    ):

        with self._lock:
            self.connection.execute(
                """
                INSERT INTO tool_calls
                (task_id, tool, arguments, result, success, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    tool,
                    json.dumps(arguments),
                    json.dumps(result),
                    int(result.get("success", False)),
                    self._now()
                )
            )

            self.connection.commit()

    def add_event(
        self,
        task_id: int,
        event_type: str,
        data=None
    ):

        with self._lock:
            self.connection.execute(
                """
                INSERT INTO events
                (task_id, event_type, data, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    task_id,
                    event_type,
                    json.dumps(data) if data else None,
                    self._now()
                )
            )

            self.connection.commit()

    def close(self):
        with self._lock:
            self.connection.close()