"""
Integration tests wiring core/agent.py task lifecycle to the accounting
persistence layer (core/accounting.py helpers + Database.save_accounting).

Verifies, using a throwaway database so the live institute.db is never
written:
    - exactly ONE accounting row is persisted per finished task, for each of
      the three terminal states (COMPLETED, FAILED, MAX_STEPS);
    - accumulated token counts are preserved (including before a failure);
    - missing/unavailable token counts are stored as NULL with an explanatory
      reason -- never as a fabricated zero;
    - electricity cost is estimated from the measured elapsed time using the
      default GPU wattage and tariff;
    - a failure in accounting persistence never masks the original task
      outcome (completed still returns; failed still raises the original).

Run from the institute root:
    python -m unittest tests.test_agent_accounting_integration -v
"""

import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.agent import Agent, TaskState
from core.exceptions import AgentMaxStepsError
from memory.database import Database
from tools.base import ToolBase

import accounting as acct


# ---------------------------------------------------------------------------
# Test doubles (mirroring tests/test_agent_token_accounting.py)
# ---------------------------------------------------------------------------
class MetadataLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        if not self.responses:
            raise AssertionError("MetadataLLM exhausted")
        content, metadata = self.responses.pop(0)
        self.last_response = metadata
        return content


class NoLastResponseLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        if not self.responses:
            raise AssertionError("NoLastResponseLLM exhausted")
        return self.responses.pop(0)


class ExplodingAfterOneLLM:
    def __init__(self):
        self.calls = 0
        self.last_response = {"prompt_eval_count": 10, "eval_count": 5}

    def chat(self, messages):
        self.calls += 1
        if self.calls == 1:
            return "First response with metadata."
        raise RuntimeError("LLM exploded on second call")


class EchoTool(ToolBase):
    @property
    def name(self):
        return "echo"

    @property
    def description(self):
        return "Echo a message back."

    def schema(self):
        return {
            "name": self.name,
            "description": self.description,
            "operations": [
                {
                    "name": "run",
                    "description": "Echo",
                    "parameters": [
                        {"name": "msg", "type": "str", "description": "m", "required": True}
                    ],
                    "example_call": {"tool": "echo", "run": "x"},
                }
            ],
        }

    def execute(self, msg=""):
        return {"success": True, "echo": msg}


class FakeRegistry:
    def __init__(self, tools):
        self._tools = {t.name: t for t in tools}

    def get(self, name):
        return self._tools.get(name)

    def generate_tool_docs(self):
        return "(fake tool docs)"


class CountingDb:
    """A Database whose save_accounting() always raises, to prove that an
    accounting failure never masks the original task outcome/exception."""

    def __init__(self, real):
        self._real = real
        self.save_calls = 0

    def __getattr__(self, name):
        return getattr(self._real, name)

    def save_accounting(self, *args, **kwargs):
        self.save_calls += 1
        raise RuntimeError("accounting persistence exploded")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestAgentAccountingIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.db = Database(self.root)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def _one_row(self):
        return self.db.get_accounting(
            self.db.connection.execute(
                "SELECT id FROM tasks ORDER BY id DESC LIMIT 1"
            ).fetchone()["id"]
        )

    # COMPLETED: one row, tokens + electricity recorded.
    def test_completed_persists_one_accounting_row(self):
        llm = MetadataLLM([
            ("I'll inspect first.", {"prompt_eval_count": 10, "eval_count": 20}),
            ("Final answer.", {"prompt_eval_count": 15, "eval_count": 30}),
        ])
        agent = Agent(llm=llm, tools=FakeRegistry([EchoTool()]),
                      database=self.db, max_steps=5)

        result = agent.run("do a thing")
        self.assertEqual(result.state, TaskState.COMPLETED)

        row = self._one_row()
        self.assertIsNotNone(row)
        # Exactly one row for the task.
        count = self.db.connection.execute(
            "SELECT COUNT(*) FROM accounting").fetchone()[0]
        self.assertEqual(count, 1)

        # Accumulated tokens preserved (25 prompt / 50 eval).
        self.assertEqual(row["input_tokens"], 25)
        self.assertEqual(row["output_tokens"], 50)

        # API-equivalent cost is a real number (both token counts available).
        self.assertIsNotNone(row["api_cost_usd"])
        self.assertIsNone(row["api_cost_reason"])

        # Electricity is estimated from elapsed time with the defaults.
        self.assertIsNotNone(row["electricity_cost_uah"])
        self.assertIsNotNone(row["elapsed_seconds"])
        self.assertEqual(row["gpu_watts"], acct.DEFAULT_GPU_WATTS)
        self.assertEqual(row["tariff_uah_per_kwh"], acct.DEFAULT_TARIFF_UAH_PER_KWH)

        # Cross-check the electricity figure against the helper's own formula.
        expected_uah = (
            acct.DEFAULT_GPU_WATTS * row["elapsed_seconds"] / 3_600_000.0
            * acct.DEFAULT_TARIFF_UAH_PER_KWH
        )
        self.assertAlmostEqual(row["electricity_cost_uah"], expected_uah)

    # COMPLETED without token metadata: API cost NULL + reason, NOT zero.
    def test_completed_missing_tokens_stored_null_not_zero(self):
        llm = NoLastResponseLLM(["I'll summarize.", "Final."])
        agent = Agent(llm=llm, tools=FakeRegistry([EchoTool()]),
                      database=self.db, max_steps=5)

        agent.run("do a thing")

        row = self._one_row()
        self.assertIsNotNone(row)
        # Missing token counts -> NULL (None), never a fabricated 0.
        self.assertIsNone(row["input_tokens"])
        self.assertIsNone(row["output_tokens"])
        # API cost unavailable with a clear explanatory reason.
        self.assertIsNone(row["api_cost_usd"])
        self.assertIsNotNone(row["api_cost_reason"])
        # Electricity still estimated from measured elapsed time.
        self.assertIsNotNone(row["electricity_cost_uah"])

    # FAILED: one row, tokens from the one successful call preserved.
    def test_failed_preserves_accumulated_tokens(self):
        llm = ExplodingAfterOneLLM()
        agent = Agent(llm=llm, tools=FakeRegistry([EchoTool()]),
                      database=self.db, max_steps=5)

        with self.assertRaises(RuntimeError) as ctx:
            agent.run("cause a failure")
        self.assertIn("exploded", str(ctx.exception))

        row = self._one_row()
        self.assertIsNotNone(row)
        # The single successful call's tokens (10 prompt / 5 eval) preserved.
        self.assertEqual(row["input_tokens"], 10)
        self.assertEqual(row["output_tokens"], 5)
        # API cost is available (both counts present) -> real number.
        self.assertIsNotNone(row["api_cost_usd"])
        self.assertIsNone(row["api_cost_reason"])
        # Electricity still estimated.
        self.assertIsNotNone(row["electricity_cost_uah"])

    # MAX_STEPS: one row persisted before the exception propagates.
    def test_max_steps_persists_one_accounting_row(self):
        # Two LLM responses, both tools -> exhausts max_steps=2.
        llm = MetadataLLM([
            ("TOOL_CALL\n{\"tool\": \"echo\", \"msg\": \"a\"}\nEND_TOOL_CALL",
             {"prompt_eval_count": 4, "eval_count": 8}),
            ("TOOL_CALL\n{\"tool\": \"echo\", \"msg\": \"b\"}\nEND_TOOL_CALL",
             {"prompt_eval_count": 6, "eval_count": 12}),
        ])
        agent = Agent(llm=llm, tools=FakeRegistry([EchoTool()]),
                      database=self.db, max_steps=2)

        with self.assertRaises(AgentMaxStepsError):
            agent.run("never finishes")

        row = self._one_row()
        self.assertIsNotNone(row)
        count = self.db.connection.execute(
            "SELECT COUNT(*) FROM accounting").fetchone()[0]
        self.assertEqual(count, 1)
        # Tokens from both calls accumulated (4+6 / 8+12).
        self.assertEqual(row["input_tokens"], 10)
        self.assertEqual(row["output_tokens"], 20)
        self.assertIsNotNone(row["electricity_cost_uah"])

    # A save_accounting() failure must NOT mask a COMPLETED outcome.
    def test_accounting_failure_does_not_mask_completed(self):
        llm = MetadataLLM([
            ("I'll inspect first.", {"prompt_eval_count": 1, "eval_count": 2}),
            ("Final answer.", {"prompt_eval_count": 3, "eval_count": 4}),
        ])
        counting = CountingDb(self.db)
        agent = Agent(llm=llm, tools=FakeRegistry([EchoTool()]),
                      database=counting, max_steps=5)

        # Must return normally (COMPLETED), not raise the accounting error.
        result = agent.run("do a thing")
        self.assertEqual(result.state, TaskState.COMPLETED)
        self.assertEqual(result.answer, "Final answer.")
        # save_accounting was attempted but its failure was swallowed.
        self.assertGreaterEqual(counting.save_calls, 1)

    # A save_accounting() failure must NOT mask a FAILED exception.
    def test_accounting_failure_does_not_mask_failure(self):
        llm = ExplodingAfterOneLLM()
        counting = CountingDb(self.db)
        agent = Agent(llm=llm, tools=FakeRegistry([EchoTool()]),
                      database=counting, max_steps=5)

        with self.assertRaises(RuntimeError) as ctx:
            agent.run("cause a failure")
        # The ORIGINAL exception is what surfaces, not the accounting error.
        self.assertIn("exploded", str(ctx.exception))
        self.assertNotIn("accounting persistence exploded", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
