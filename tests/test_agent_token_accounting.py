"""
Focused tests for per-task token accounting in core/agent.py.

Run from the institute root:
    python -m unittest tests.test_agent_token_accounting -v
"""

import json
import os
import sys
import tempfile
import unittest

# Ensure the institute root is importable regardless of CWD.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.agent import Agent, TaskState, AgentResult
from core.exceptions import AgentMaxStepsError
from memory.database import Database
from tools.base import ToolBase


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class MetadataLLM:
    """
    A scripted LLM that exposes ``last_response`` after each call.

    ``responses`` is a list of (content, metadata_dict_or_None) tuples.
    metadata=None means last_response is set to None after that call.
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        if not self.responses:
            raise AssertionError("MetadataLLM exhausted - unexpected extra call")
        content, metadata = self.responses.pop(0)
        self.last_response = metadata
        return content


class NoLastResponseLLM:
    """
    A scripted LLM that does NOT define ``last_response`` at all.
    This simulates existing fake/test LLMs that predate the metadata feature.
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        if not self.responses:
            raise AssertionError("NoLastResponseLLM exhausted")
        return self.responses.pop(0)


class ExplodingAfterOneLLM:
    """
    Succeeds on the first chat() call (with metadata), then raises on the
    second call.
    """

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


def tool_call(tool, **kwargs):
    payload = {"tool": tool, **kwargs}
    return "TOOL_CALL\n" + json.dumps(payload) + "\nEND_TOOL_CALL"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAgentTokenAccounting(unittest.TestCase):
    """Focused tests for per-task token accounting in Agent.run()."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.db = Database(self.root)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    # ------------------------------------------------------------------
    # 1. Multiple LLM calls accumulating token counts
    # ------------------------------------------------------------------
    def test_multiple_calls_accumulate_tokens(self):
        """
        Three LLM calls, each with valid metadata. The AgentResult should
        carry the sum of prompt_eval_count and eval_count across all calls.
        """
        llm = MetadataLLM([
            ("I'll start by inspecting the repository.",
             {"prompt_eval_count": 10, "eval_count": 20}),
            ("Final answer text.",
             {"prompt_eval_count": 15, "eval_count": 30}),
        ])
        # Two calls: first narration triggers recovery, second is accepted.
        agent = Agent(llm=llm, tools=FakeRegistry([EchoTool()]),
                      database=self.db, max_steps=5)

        result = agent.run("do a thing")

        self.assertEqual(result.state, TaskState.COMPLETED)
        self.assertEqual(llm.calls, 2)
        # Accumulated: 10+15=25 prompt, 20+30=50 eval
        self.assertEqual(result.prompt_eval_count, 25)
        self.assertEqual(result.eval_count, 50)

    def test_multiple_calls_with_tool_accumulate_tokens(self):
        """
        Two LLM calls: one emits a tool call, the other gives the final
        answer. Token counts from both calls should accumulate.
        """
        llm = MetadataLLM([
            (tool_call("echo", msg="hello"),
             {"prompt_eval_count": 7, "eval_count": 13}),
            ("Done after tool.",
             {"prompt_eval_count": 3, "eval_count": 17}),
        ])
        agent = Agent(llm=llm, tools=FakeRegistry([EchoTool()]),
                      database=self.db, max_steps=5)

        result = agent.run("use the echo tool")

        self.assertEqual(result.state, TaskState.COMPLETED)
        self.assertEqual(result.prompt_eval_count, 10)  # 7+3
        self.assertEqual(result.eval_count, 30)        # 13+17

    # ------------------------------------------------------------------
    # 2. LLM without last_response
    # ------------------------------------------------------------------
    def test_llm_without_last_response(self):
        """
        An LLM that does not define the ``last_response`` attribute at all.
        The agent must still work correctly; token totals should be None
        (unavailable), not 0.
        """
        llm = NoLastResponseLLM([
            "I'll summarize the findings.",
            "Final answer text.",
        ])
        agent = Agent(llm=llm, tools=FakeRegistry([EchoTool()]),
                      database=self.db, max_steps=5)

        result = agent.run("do a thing")

        self.assertEqual(result.state, TaskState.COMPLETED)
        self.assertEqual(result.answer, "Final answer text.")
        # Totals are unavailable (None), not 0.
        self.assertIsNone(result.prompt_eval_count)
        self.assertIsNone(result.eval_count)

    # ------------------------------------------------------------------
    # 3. Missing or invalid metadata
    # ------------------------------------------------------------------
    def test_missing_metadata_marks_unavailable(self):
        """
        First call has valid metadata; second call has last_response=None.
        The totals should be marked unavailable (None) because one call's
        metadata is missing.
        """
        llm = MetadataLLM([
            ("I'll summarize.",
             {"prompt_eval_count": 10, "eval_count": 20}),
            ("Final.",
             None),  # last_response set to None
        ])
        agent = Agent(llm=llm, tools=FakeRegistry([EchoTool()]),
                      database=self.db, max_steps=5)

        result = agent.run("do a thing")

        self.assertEqual(result.state, TaskState.COMPLETED)
        # Because the second call had no metadata, both totals are unavailable.
        self.assertIsNone(result.prompt_eval_count)
        self.assertIsNone(result.eval_count)

    def test_invalid_metadata_field_marks_unavailable(self):
        """
        One call has prompt_eval_count as a string (invalid type).
        The prompt_eval_count total should be None, while eval_count
        (if valid) should still accumulate.
        """
        llm = MetadataLLM([
            ("I'll summarize.",
             {"prompt_eval_count": "not_a_number", "eval_count": 20}),
            ("Final.",
             {"prompt_eval_count": 15, "eval_count": 30}),
        ])
        agent = Agent(llm=llm, tools=FakeRegistry([EchoTool()]),
                      database=self.db, max_steps=5)

        result = agent.run("do a thing")

        self.assertEqual(result.state, TaskState.COMPLETED)
        # prompt_eval_count is unavailable due to the first call's bad value.
        self.assertIsNone(result.prompt_eval_count)
        # eval_count is valid in both calls, so it accumulates: 20+30=50.
        self.assertEqual(result.eval_count, 50)

    def test_last_response_not_a_dict(self):
        """
        last_response is set to a string instead of a dict.
        Both totals should be marked unavailable.
        """
        class BadMetadataLLM:
            def __init__(self):
                self.responses = ["I'll do it.", "Done."]
                self.calls = 0
            def chat(self, messages):
                self.calls += 1
                self.last_response = "not a dict"
                return self.responses.pop(0)

        llm = BadMetadataLLM()
        agent = Agent(llm=llm, tools=FakeRegistry([EchoTool()]),
                      database=self.db, max_steps=5)

        result = agent.run("do a thing")

        self.assertEqual(result.state, TaskState.COMPLETED)
        self.assertIsNone(result.prompt_eval_count)
        self.assertIsNone(result.eval_count)

    # ------------------------------------------------------------------
    # 4. Exception after at least one successful LLM call
    # ------------------------------------------------------------------
    def test_exception_after_successful_llm_call(self):
        """
        The first LLM call succeeds (with metadata), the second raises.
        The task should be marked FAILED, and the accumulated tokens from
        the first call should be persisted in the task_failed event.
        """
        llm = ExplodingAfterOneLLM()
        agent = Agent(llm=llm, tools=FakeRegistry([EchoTool()]),
                      database=self.db, max_steps=5)

        with self.assertRaises(RuntimeError) as ctx:
            agent.run("cause a failure")

        self.assertIn("exploded", str(ctx.exception))

        # Task should be marked failed.
        cur = self.db.connection.execute(
            "SELECT status FROM tasks ORDER BY id DESC LIMIT 1"
        )
        self.assertEqual(cur.fetchone()["status"], "failed")

        # The task_failed event should carry the token totals from the
        # one successful call.
        cur = self.db.connection.execute(
            "SELECT data FROM events WHERE event_type='task_failed'"
        )
        data = json.loads(cur.fetchone()["data"])
        self.assertEqual(data["prompt_eval_count"], 10)
        self.assertEqual(data["eval_count"], 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
