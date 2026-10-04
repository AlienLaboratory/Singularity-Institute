"""
Focused integration tests for the agent execution lifecycle.

Run from the institute root:
    python tests/test_agent_lifecycle.py
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

class FakeLLM:
    """Returns a scripted sequence of model responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        if not self.responses:
            raise AssertionError("FakeLLM exhausted - unexpected extra call")
        return self.responses.pop(0)


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


class FailingTool(ToolBase):
    @property
    def name(self):
        return "boom"

    @property
    def description(self):
        return "Always raises."

    def schema(self):
        return {
            "name": self.name,
            "description": self.description,
            "operations": [
                {
                    "name": "run",
                    "description": "run",
                    "parameters": [
                        {"name": "x", "type": "str", "description": "x", "required": True}
                    ],
                    "example_call": {"tool": "boom", "x": "1"},
                }
            ],
        }

    def execute(self, x=""):
        raise RuntimeError("intentional tool failure")


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


class AgentLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.db = Database(self.root)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def _events(self):
        cur = self.db.connection.execute(
            "SELECT event_type FROM events ORDER BY id"
        )
        return [row["event_type"] for row in cur.fetchall()]

    def _task_status(self):
        cur = self.db.connection.execute(
            "SELECT status FROM tasks ORDER BY id DESC LIMIT 1"
        )
        return cur.fetchone()["status"]

    # ------------------------------------------------------------------
    def test_normal_final_answer_completion(self):
        llm = FakeLLM(["Final answer text."])
        agent = Agent(llm=llm, tools=FakeRegistry([EchoTool()]),
                      database=self.db, max_steps=5)

        result = agent.run("do a thing")

        self.assertIsInstance(result, AgentResult)
        self.assertEqual(result.state, TaskState.COMPLETED)
        self.assertEqual(result.answer, "Final answer text.")
        self.assertEqual(self._task_status(), "completed")
        self.assertIn("task_started", self._events())
        self.assertIn("task_completed", self._events())
        # str() compatibility for CLI
        self.assertEqual(str(result), "Final answer text.")

    # ------------------------------------------------------------------
    def test_one_tool_call_then_final_answer(self):
        llm = FakeLLM([
            tool_call("echo", msg="hello"),
            "Done after tool."
        ])
        agent = Agent(llm=llm, tools=FakeRegistry([EchoTool()]),
                      database=self.db, max_steps=5)

        result = agent.run("use the echo tool")

        self.assertEqual(result.state, TaskState.COMPLETED)
        self.assertEqual(result.steps, 2)
        self.assertEqual(self._task_status(), "completed")

        events = self._events()
        self.assertIn("task_started", events)
        self.assertIn("tool_requested", events)
        self.assertIn("tool_completed", events)
        self.assertIn("task_completed", events)

        # tool_calls persistence retained
        cur = self.db.connection.execute(
            "SELECT tool, success FROM tool_calls"
        )
        rows = list(cur.fetchall())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["tool"], "echo")
        self.assertEqual(rows[0]["success"], 1)

    # ------------------------------------------------------------------
    def test_unknown_tool_is_returned_not_fatal(self):
        llm = FakeLLM([
            tool_call("does_not_exist", x="1"),
            "Recovered from unknown tool."
        ])
        agent = Agent(llm=llm, tools=FakeRegistry([EchoTool()]),
                      database=self.db, max_steps=5)

        # Should NOT raise; model recovers.
        result = agent.run("call a bad tool")

        self.assertEqual(result.state, TaskState.COMPLETED)

        cur = self.db.connection.execute(
            "SELECT success FROM tool_calls"
        )
        self.assertEqual(list(cur.fetchall())[0]["success"], 0)

    # ------------------------------------------------------------------
    def test_tool_execution_failure_is_recoverable(self):
        llm = FakeLLM([
            tool_call("boom", x="1"),
            "Recovered from tool failure."
        ])
        agent = Agent(llm=llm, tools=FakeRegistry([FailingTool()]),
                      database=self.db, max_steps=5)

        # A failing tool should be returned to the model, not crash the task.
        result = agent.run("call a failing tool")

        self.assertEqual(result.state, TaskState.COMPLETED)

        events = self._events()
        self.assertIn("tool_requested", events)
        self.assertIn("tool_completed", events)

        cur = self.db.connection.execute(
            "SELECT success FROM tool_calls"
        )
        self.assertEqual(list(cur.fetchall())[0]["success"], 0)

    # ------------------------------------------------------------------
    def test_max_step_exhaustion(self):
        # Always issues a tool call, never a final answer.
        llm = FakeLLM([tool_call("echo", msg=f"s{i}") for i in range(3)])
        agent = Agent(llm=llm, tools=FakeRegistry([EchoTool()]),
                      database=self.db, max_steps=3)

        with self.assertRaises(AgentMaxStepsError) as ctx:
            agent.run("endless tool loop")

        self.assertIsInstance(ctx.exception, AgentMaxStepsError)
        self.assertEqual(self._task_status(), "max_steps")

        events = self._events()
        self.assertIn("task_started", events)
        self.assertIn("tool_requested", events)
        self.assertIn("tool_completed", events)
        self.assertIn("max_steps_reached", events)

        # reason persisted in event data
        cur = self.db.connection.execute(
            "SELECT data FROM events WHERE event_type='max_steps_reached'"
        )
        data = json.loads(cur.fetchone()["data"])
        self.assertIn("reason", data)

        # exception is an AgentError and carries task id
        self.assertEqual(ctx.exception.task_id, 1)
        self.assertEqual(ctx.exception.max_steps, 3)

    # ------------------------------------------------------------------
    def test_agent_level_failure_persisted(self):
        class ExplodingLLM(FakeLLM):
            def chat(self, messages):
                raise RuntimeError("LLM blew up")

        llm = ExplodingLLM([])
        agent = Agent(llm=llm, tools=FakeRegistry([EchoTool()]),
                      database=self.db, max_steps=5)

        with self.assertRaises(RuntimeError):
            agent.run("cause a failure")

        self.assertEqual(self._task_status(), "failed")
        self.assertIn("task_failed", self._events())

    # ------------------------------------------------------------------
    def test_lifecycle_event_ordering(self):
        llm = FakeLLM([
            tool_call("echo", msg="x"),
            "Final."
        ])
        agent = Agent(llm=llm, tools=FakeRegistry([EchoTool()]),
                      database=self.db, max_steps=5)
        agent.run("ordered lifecycle")

        events = self._events()
        self.assertEqual(events, [
            "task_started",
            "tool_requested",
            "tool_completed",
            "task_completed",
        ])


if __name__ == "__main__":
    unittest.main(verbosity=2)
