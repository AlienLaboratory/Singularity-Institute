"""
Focused tests for persistent per-task accounting storage in SQLite
(memory/database.py), using a temporary database so the live institute.db is
never touched.

Covers:
    - migration on a fresh database (all tables created)
    - migration repeated twice without schema/data duplication
    - preservation of existing tables and records across re-initialisation
    - insert and retrieval of an accounting row
    - duplicate save for the same task_id without double-counting
    - NULL/unavailable cost handling and aggregate calculations (USD vs UAH)
    - rejection of invalid numeric values
    - lock/connection conventions (shared connection usable across threads)

Run from the institute root:
    python -m py_compile memory/database.py tests/test_database_accounting.py
    python -m unittest tests.test_database_accounting -v
"""

import math
import os
import sys
import tempfile
import threading
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from memory.database import Database

EXPECTED_TABLES = {"tasks", "messages", "tool_calls", "events", "accounting"}


class BaseTempDbTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="institute_acct_")
        # Database(root) opens <root>/memory/institute.db
        self.root = self._tmp.name
        self.db = Database(self.root)

    def tearDown(self):
        self.db.close()
        self._tmp.cleanup()


class TestMigration(BaseTempDbTest):
    def test_fresh_database_has_all_tables(self):
        names = {
            r["name"] for r in self.db.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        self.assertTrue(EXPECTED_TABLES.issubset(names))

    def test_accounting_has_unique_task_id(self):
        cols = self.db.connection.execute(
            "PRAGMA table_info(accounting)"
        ).fetchall()
        col_names = {c["name"] for c in cols}
        # Required fields exist.
        for required in (
            "task_id", "input_tokens", "output_tokens", "api_cost_usd",
            "electricity_cost_uah", "elapsed_seconds", "gpu_watts",
            "tariff_uah_per_kwh", "api_cost_reason", "electricity_cost_reason",
            "created_at",
        ):
            self.assertIn(required, col_names)
        # UNIQUE(task_id) constraint present.
        index_info = self.db.connection.execute(
            "PRAGMA index_list(accounting)"
        ).fetchall()
        unique_task_idx = None
        for idx in index_info:
            cols = self.db.connection.execute(
                f"PRAGMA index_info({idx['name']})"
            ).fetchall()
            if [c["name"] for c in cols] == ["task_id"]:
                unique_task_idx = idx
        self.assertIsNotNone(unique_task_idx, "expected UNIQUE index on task_id")
        self.assertTrue(unique_task_idx["unique"])


class TestMigrationIdempotent(BaseTempDbTest):
    def test_reopen_does_not_duplicate_schema(self):
        # Seed a task + accounting row.
        task_id = self.db.create_task("hello")
        self.db.save_accounting(
            task_id, input_tokens=100, output_tokens=50,
            api_cost_usd=0.05, electricity_cost_uah=1.0,
        )

        # Re-open the same file: __init__ runs _initialize() again.
        self.db.close()
        self.db = Database(self.root)

        # Tables are a set (no duplicates) and records are preserved.
        names = self.db.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [n["name"] for n in names]
        for t in ("tasks", "messages", "tool_calls", "events", "accounting"):
            self.assertEqual(table_names.count(t), 1, f"duplicate table {t}")

        self.assertEqual(
            self.db.connection.execute(
                "SELECT COUNT(*) FROM tasks").fetchone()[0], 1
        )
        self.assertEqual(
            self.db.connection.execute(
                "SELECT COUNT(*) FROM accounting").fetchone()[0], 1
        )
        row = self.db.get_accounting(task_id)
        self.assertEqual(row["api_cost_usd"], 0.05)

    def test_migration_run_twice_explicit(self):
        # Emulate running the migration script twice against a fresh DB.
        first = self.db.connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
        ).fetchone()[0]
        # Re-run the exact initialization once more.
        self.db._initialize()
        second = self.db.connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
        ).fetchone()[0]
        self.assertEqual(first, second)


class TestPreservation(BaseTempDbTest):
    def test_existing_records_preserved_after_migration(self):
        # Populate legacy tables (as a prior schema would have).
        task_id = self.db.create_task("legacy task")
        self.db.add_message(task_id, "user", "hi")
        self.db.add_event(task_id, "lifecycle", {"k": "v"})

        # Now run the (idempotent) migration again.
        self.db._initialize()

        self.assertEqual(
            self.db.connection.execute(
                "SELECT COUNT(*) FROM tasks").fetchone()[0], 1
        )
        self.assertEqual(
            self.db.connection.execute(
                "SELECT COUNT(*) FROM messages").fetchone()[0], 1
        )
        self.assertEqual(
            self.db.connection.execute(
                "SELECT COUNT(*) FROM events").fetchone()[0], 1
        )
        msg = self.db.connection.execute(
            "SELECT role, content FROM messages").fetchone()
        self.assertEqual((msg["role"], msg["content"]), ("user", "hi"))


class TestInsertAndRetrieve(BaseTempDbTest):
    def test_save_then_get(self):
        task_id = self.db.create_task("retrieve me")
        self.db.save_accounting(
            task_id,
            input_tokens=10_000,
            output_tokens=4_000,
            api_cost_usd=0.15,
            electricity_cost_uah=1.512,
            elapsed_seconds=3600.0,
            gpu_watts=350.0,
            tariff_uah_per_kwh=4.32,
        )
        row = self.db.get_accounting(task_id)
        self.assertIsNotNone(row)
        self.assertEqual(row["task_id"], task_id)
        self.assertEqual(row["input_tokens"], 10_000)
        self.assertEqual(row["output_tokens"], 4_000)
        self.assertAlmostEqual(row["api_cost_usd"], 0.15)
        self.assertAlmostEqual(row["electricity_cost_uah"], 1.512)
        self.assertEqual(row["elapsed_seconds"], 3600.0)
        self.assertIsNotNone(row["created_at"])

    def test_get_missing_returns_none(self):
        self.assertIsNone(self.db.get_accounting(999999))

    def test_list_accounting_returns_all(self):
        t1 = self.db.create_task("a")
        t2 = self.db.create_task("b")
        self.db.save_accounting(t1, api_cost_usd=0.1)
        self.db.save_accounting(t2, api_cost_usd=0.2)
        rows = self.db.list_accounting()
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["task_id"] for r in rows}, {t1, t2})


class TestDuplicateSave(BaseTempDbTest):
    def test_duplicate_save_same_task_no_double_count(self):
        task_id = self.db.create_task("dup")
        self.db.save_accounting(
            task_id, input_tokens=100, output_tokens=50,
            api_cost_usd=0.10, electricity_cost_uah=1.0,
        )
        # Repeated status polling / save must not add rows.
        self.db.save_accounting(
            task_id, input_tokens=100, output_tokens=50,
            api_cost_usd=0.10, electricity_cost_uah=1.0,
        )
        count = self.db.connection.execute(
            "SELECT COUNT(*) FROM accounting WHERE task_id=?",
            (task_id,),
        ).fetchone()[0]
        self.assertEqual(count, 1)
        agg = self.db.aggregate_accounting()
        self.assertEqual(agg["task_count"], 1)
        self.assertAlmostEqual(agg["usd"]["total"], 0.10)

    def test_duplicate_save_updates_values(self):
        task_id = self.db.create_task("update")
        self.db.save_accounting(task_id, api_cost_usd=0.10)
        self.db.save_accounting(task_id, api_cost_usd=0.20)
        row = self.db.get_accounting(task_id)
        self.assertAlmostEqual(row["api_cost_usd"], 0.20)
        count = self.db.connection.execute(
            "SELECT COUNT(*) FROM accounting WHERE task_id=?", (task_id,),
        ).fetchone()[0]
        self.assertEqual(count, 1)


class TestNullHandlingAndAggregates(BaseTempDbTest):
    def test_unavailable_stored_as_null_with_reason(self):
        task_id = self.db.create_task("partial")
        self.db.save_accounting(
            task_id,
            input_tokens=None,
            output_tokens=4_000,
            api_cost_usd=None,
            electricity_cost_uah=1.512,
            api_cost_reason="unavailable: missing input_tokens",
        )
        row = self.db.get_accounting(task_id)
        self.assertIsNone(row["input_tokens"])
        self.assertIsNone(row["api_cost_usd"])
        self.assertIsNotNone(row["api_cost_reason"])
        self.assertAlmostEqual(row["electricity_cost_uah"], 1.512)

    def test_aggregate_excludes_nulls_not_zero(self):
        t1 = self.db.create_task("full")
        self.db.save_accounting(
            t1, api_cost_usd=0.10, electricity_cost_uah=1.0
        )
        t2 = self.db.create_task("no_usd")
        self.db.save_accounting(t2, api_cost_usd=None, electricity_cost_uah=2.0)
        t3 = self.db.create_task("no_elec")
        self.db.save_accounting(t3, api_cost_usd=0.30, electricity_cost_uah=None)

        agg = self.db.aggregate_accounting()
        self.assertEqual(agg["task_count"], 3)

        usd = agg["usd"]
        self.assertEqual(usd["available_count"], 2)
        self.assertEqual(usd["unavailable_count"], 1)
        self.assertAlmostEqual(usd["total"], 0.40)
        self.assertAlmostEqual(usd["average"], 0.20)

        uah = agg["uah"]
        self.assertEqual(uah["available_count"], 2)
        self.assertEqual(uah["unavailable_count"], 1)
        self.assertAlmostEqual(uah["total"], 3.0)
        self.assertAlmostEqual(uah["average"], 1.5)

    def test_aggregate_empty(self):
        agg = self.db.aggregate_accounting()
        self.assertEqual(agg["task_count"], 0)
        self.assertIsNone(agg["usd"]["total"])
        self.assertIsNone(agg["usd"]["average"])
        self.assertIsNone(agg["uah"]["total"])
        self.assertEqual(agg["usd"]["available_count"], 0)

    def test_usd_and_uah_kept_separate(self):
        # A USD value must never leak into the UAH bucket and vice versa.
        t1 = self.db.create_task("separate")
        self.db.save_accounting(t1, api_cost_usd=0.15, electricity_cost_uah=1.5)
        agg = self.db.aggregate_accounting()
        self.assertAlmostEqual(agg["usd"]["total"], 0.15)
        self.assertAlmostEqual(agg["uah"]["total"], 1.5)
        self.assertNotEqual(agg["usd"], agg["uah"])


class TestInvalidValues(BaseTempDbTest):
    def test_negative_api_cost_rejected(self):
        t = self.db.create_task("x")
        with self.assertRaises(ValueError):
            self.db.save_accounting(t, api_cost_usd=-1.0)

    def test_nan_rejected(self):
        t = self.db.create_task("x")
        with self.assertRaises((ValueError, TypeError)):
            self.db.save_accounting(t, api_cost_usd=math.nan)

    def test_inf_rejected(self):
        t = self.db.create_task("x")
        with self.assertRaises((ValueError, TypeError)):
            self.db.save_accounting(t, electricity_cost_uah=math.inf)

    def test_non_numeric_rejected(self):
        t = self.db.create_task("x")
        with self.assertRaises(TypeError):
            self.db.save_accounting(t, api_cost_usd="0.15")

    def test_bool_rejected(self):
        t = self.db.create_task("x")
        with self.assertRaises(TypeError):
            self.db.save_accounting(t, api_cost_usd=True)

    def test_negative_tokens_rejected(self):
        t = self.db.create_task("x")
        with self.assertRaises(ValueError):
            self.db.save_accounting(t, input_tokens=-5)

    def test_invalid_write_does_not_create_row(self):
        t = self.db.create_task("x")
        with self.assertRaises((ValueError, TypeError)):
            self.db.save_accounting(t, api_cost_usd=-1.0)
        self.assertIsNone(self.db.get_accounting(t))


class TestLockingConventions(BaseTempDbTest):
    def test_shared_connection_usable_from_worker_thread(self):
        # Mirrors core/webui.py: connection created on main thread, then used by
        # a worker thread.  check_same_thread=False + self._lock must permit this.
        task_id = self.db.create_task("threaded")

        def worker():
            self.db.save_accounting(
                task_id, api_cost_usd=0.1, electricity_cost_uah=0.5
            )

        th = threading.Thread(target=worker)
        th.start()
        th.join()

        row = self.db.get_accounting(task_id)
        self.assertAlmostEqual(row["api_cost_usd"], 0.1)

    def test_concurrent_saves_same_task_no_lost_update(self):
        task_id = self.db.create_task("conc")

        def worker(val):
            self.db.save_accounting(task_id, api_cost_usd=val)

        threads = [threading.Thread(target=worker, args=(i / 10.0,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        count = self.db.connection.execute(
            "SELECT COUNT(*) FROM accounting WHERE task_id=?", (task_id,),
        ).fetchone()[0]
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
