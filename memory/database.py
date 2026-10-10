import sqlite3
import json
import math
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

        # Core tables (existing schema, preserved verbatim).  ``IF NOT EXISTS``
        # keeps the migration idempotent and non-destructive.
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

        -- Per-task accounting (this slice). One row per task, enforced by
        -- the UNIQUE(task_id) constraint so repeated saves upsert the same
        -- row and never create duplicates. Nullable cost/token fields
        -- preserve "unavailable" values as SQL NULL (never a fabricated 0).
        CREATE TABLE IF NOT EXISTS accounting (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER NOT NULL,
            input_tokens INTEGER,
            output_tokens INTEGER,
            api_cost_usd REAL,
            electricity_cost_uah REAL,
            elapsed_seconds REAL,
            gpu_watts REAL,
            tariff_uah_per_kwh REAL,
            api_cost_reason TEXT,
            electricity_cost_reason TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT,
            UNIQUE(task_id),
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

    # ------------------------------------------------------------------
    # Accounting (persistent, per-task, one row per task)
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_optional_number(name, value):
        """Validate an optional numeric accounting field.

        ``None`` is allowed (means "unavailable").  Otherwise *value* must be
        an int/float (not bool), finite, and >= 0.  Rejects NaN, inf, and
        negative values so invalid input never reaches the table.
        """
        if value is None:
            return None
        if isinstance(value, bool):
            raise TypeError(f"{name} must be a finite number, got bool")
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"{name} must be an int or float, got {type(value).__name__}"
            )
        if math.isnan(value):
            raise ValueError(f"{name} must not be NaN")
        if math.isinf(value):
            raise ValueError(f"{name} must not be infinite")
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")
        return float(value)

    @staticmethod
    def _validate_optional_token(name, value):
        """Validate an optional token count (non-negative int) or ``None``."""
        if value is None:
            return None
        if isinstance(value, bool):
            raise TypeError(f"{name} must be a non-negative int, got bool")
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"{name} must be an int or float, got {type(value).__name__}"
            )
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"{name} must be finite")
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")
        return int(value)

    def save_accounting(
        self,
        task_id: int,
        input_tokens=None,
        output_tokens=None,
        api_cost_usd=None,
        electricity_cost_uah=None,
        elapsed_seconds=None,
        gpu_watts=None,
        tariff_uah_per_kwh=None,
        api_cost_reason=None,
        electricity_cost_reason=None,
    ) -> int:
        """Persist (or idempotently upsert) the accounting record for a task.

        Exactly one row exists per ``task_id`` (``UNIQUE(task_id)``).  Repeated
        saves for the same task replace the existing values and bump
        ``updated_at``; they never create a second row or inflate totals.

        Unavailable values are stored as SQL ``NULL`` together with a clear
        ``*_reason`` string.  Invalid numeric values (bool, non-numeric, NaN,
        inf, negative) are rejected before any write.

        Returns the ``task_id`` that was recorded.
        """
        input_tokens = self._validate_optional_token("input_tokens", input_tokens)
        output_tokens = self._validate_optional_token("output_tokens", output_tokens)
        api_cost_usd = self._validate_optional_number("api_cost_usd", api_cost_usd)
        electricity_cost_uah = self._validate_optional_number(
            "electricity_cost_uah", electricity_cost_uah
        )
        elapsed_seconds = self._validate_optional_number(
            "elapsed_seconds", elapsed_seconds
        )
        gpu_watts = self._validate_optional_number("gpu_watts", gpu_watts)
        tariff_uah_per_kwh = self._validate_optional_number(
            "tariff_uah_per_kwh", tariff_uah_per_kwh
        )

        now = self._now()

        with self._lock:
            self.connection.execute(
                """
                INSERT INTO accounting (
                    task_id, input_tokens, output_tokens,
                    api_cost_usd, electricity_cost_uah,
                    elapsed_seconds, gpu_watts, tariff_uah_per_kwh,
                    api_cost_reason, electricity_cost_reason,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    input_tokens = excluded.input_tokens,
                    output_tokens = excluded.output_tokens,
                    api_cost_usd = excluded.api_cost_usd,
                    electricity_cost_uah = excluded.electricity_cost_uah,
                    elapsed_seconds = excluded.elapsed_seconds,
                    gpu_watts = excluded.gpu_watts,
                    tariff_uah_per_kwh = excluded.tariff_uah_per_kwh,
                    api_cost_reason = excluded.api_cost_reason,
                    electricity_cost_reason = excluded.electricity_cost_reason,
                    updated_at = excluded.updated_at
                """,
                (
                    task_id,
                    input_tokens,
                    output_tokens,
                    api_cost_usd,
                    electricity_cost_uah,
                    elapsed_seconds,
                    gpu_watts,
                    tariff_uah_per_kwh,
                    api_cost_reason,
                    electricity_cost_reason,
                    now,
                    now,
                )
            )

            self.connection.commit()

            return task_id

    def get_accounting(self, task_id: int):
        """Return the single accounting row for ``task_id`` as a dict,
        or ``None`` if no record exists."""
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM accounting WHERE task_id = ?",
                (task_id,),
            ).fetchone()

        if row is None:
            return None
        return dict(row)

    def list_accounting(self):
        """Return all accounting rows as a list of dicts, oldest first."""
        with self._lock:
            rows = self.connection.execute(
                "SELECT * FROM accounting ORDER BY id"
            ).fetchall()
        return [dict(r) for r in rows]

    def aggregate_accounting(self) -> dict:
        """Compute lifetime accounting totals.

        Unavailable values (SQL ``NULL``) are **excluded** from sums and
        averages and are reported separately as ``unavailable_count`` -- they
        are never treated as zero.  USD and UAH figures are kept in separate
        sub-dicts and never mixed.
        """
        with self._lock:
            task_count = self.connection.execute(
                "SELECT COUNT(*) FROM accounting"
            ).fetchone()[0]

            usd = self.connection.execute(
                """
                SELECT
                    COUNT(api_cost_usd) AS available,
                    SUM(api_cost_usd)   AS total
                FROM accounting
                """
            ).fetchone()

            uah = self.connection.execute(
                """
                SELECT
                    COUNT(electricity_cost_uah) AS available,
                    SUM(electricity_cost_uah)   AS total
                FROM accounting
                """
            ).fetchone()

        def _summary(available: int, total, unavailable: int) -> dict:
            if available:
                average = total / available if total is not None else None
            else:
                average = None
            return {
                "available_count": available,
                "unavailable_count": unavailable,
                "total": total,
                "average": average,
            }

        return {
            "task_count": task_count,
            "usd": _summary(
                usd["available"],
                usd["total"],
                task_count - usd["available"],
            ),
            "uah": _summary(
                uah["available"],
                uah["total"],
                task_count - uah["available"],
            ),
        }

    def close(self):
        with self._lock:
            self.connection.close()
