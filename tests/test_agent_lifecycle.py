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
        # Records the `messages` passed to each chat() call so tests can
        # inspect what the agent sent to the model (e.g. recovery prompts).
        self.messages_log = []

    def chat(self, messages):
        self.calls += 1
        # Snapshot a copy: the agent mutates the list in-place each turn,
        # so storing a reference would alias every entry to the final list.
        self.messages_log.append([dict(m) for m in messages])
        if not self.responses:
            raise AssertionError("FakeLLM exhausted - unexpected extra call")
        return self.responses.pop(0)

    def last_prompt(self):
        """The content of the last message sent to the model."""
        return self.messages_log[-1][-1]["content"]


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
        """
        A legitimate final answer (no TOOL_CALL) still completes the task
        normally.

        Because a pre-tool non-tool response is indistinguishable from a
        narration of intent, the agent issues a bounded recovery prompt and
        accepts the model's confirming (still non-tool) response as the final
        answer. No TOOL_CALL is required for a final answer.
        """
        llm = FakeLLM([
            "I'll summarize the findings.",
            "Final answer text."
        ])
        agent = Agent(llm=llm, tools=FakeRegistry([EchoTool()]),
                      database=self.db, max_steps=5)

        result = agent.run("do a thing")

        self.assertIsInstance(result, AgentResult)
        self.assertEqual(result.state, TaskState.COMPLETED)
        self.assertEqual(result.answer, "Final answer text.")
        self.assertEqual(self._task_status(), "completed")
        self.assertIn("task_started", self._events())
        self.assertIn("recovery_prompt", self._events())
        self.assertIn("task_completed", self._events())
        # No TOOL_CALL was required for the final answer.
        cur = self.db.connection.execute("SELECT COUNT(*) AS n FROM tool_calls")
        self.assertEqual(cur.fetchone()["n"], 0)
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

        # A tool was executed first, so the final answer is accepted
        # immediately (no recovery prompt needed).
        self.assertNotIn("recovery_prompt", events)

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

    # ------------------------------------------------------------------
    # Narration-recovery tests (core Agent reliability bug)
    # ------------------------------------------------------------------
    def test_narration_then_tool_call_recovers(self):
        """
        Reproduces the exact failure: the model first narrates an intended
        action WITHOUT emitting a TOOL_CALL. The Agent must NOT terminate,
        must ask the model to execute the action, then accept the TOOL_CALL,
        run the tool, and finally accept the final answer.
        """
        llm = FakeLLM([
            "I'll start by inspecting the repository structure and the CLI file.",
            tool_call("echo", msg="done"),
            "All steps completed successfully."
        ])
        agent = Agent(llm=llm, tools=FakeRegistry([EchoTool()]),
                      database=self.db, max_steps=5)

        result = agent.run("do an actionable task")

        self.assertEqual(result.state, TaskState.COMPLETED)
        self.assertEqual(result.answer, "All steps completed successfully.")
        self.assertEqual(result.steps, 3)
        self.assertEqual(self._task_status(), "completed")

        # The agent did not stop on the narration: the tool executed.
        cur = self.db.connection.execute(
            "SELECT tool, success, result FROM tool_calls"
        )
        rows = list(cur.fetchall())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["tool"], "echo")
        self.assertEqual(rows[0]["success"], 1)
        self.assertIn("done", rows[0]["result"])

        events = self._events()
        self.assertIn("recovery_prompt", events)
        self.assertIn("tool_requested", events)
        self.assertIn("tool_completed", events)
        self.assertIn("task_completed", events)

        # The recovery prompt was actually sent to the model immediately after
        # the narration (i.e. before the second model turn produced the
        # TOOL_CALL).
        second_turn_messages = llm.messages_log[1]
        last_of_second_turn = second_turn_messages[-1]["content"]
        self.assertIn("did not include a TOOL_CALL", last_of_second_turn)

    def test_repeated_narration_is_bounded(self):
        """
        The model repeatedly produces narration without a TOOL_CALL. Recovery
        is bounded: the agent issues a recovery prompt a bounded number of
        times and then accepts a non-tool response as the final answer, so
        the task completes without an infinite loop and well before
        max_steps. With MAX_RECOVERY_ATTEMPTS == 1, the first narration
        triggers a single recovery prompt and the second non-tool response is
        accepted as the final answer.
        """
        # Provide more narrations than the budget; the agent must not consume
        # them all or loop forever.
        llm = FakeLLM([
            "I'll start by inspecting the repository.",
            "I'll now read the file contents.",
            "I'll wrap up my analysis."
        ])
        agent = Agent(llm=llm, tools=FakeRegistry([EchoTool()]),
                      database=self.db, max_steps=8)

        # Must not raise AgentMaxStepsError and must not loop forever.
        result = agent.run("a task the model keeps narrating")

        self.assertEqual(result.state, TaskState.COMPLETED)
        self.assertEqual(self._task_status(), "completed")
        self.assertEqual(result.steps, 2)

        events = self._events()
        self.assertIn("recovery_prompt", events)
        self.assertIn("task_completed", events)
        # Bounded: at most MAX_RECOVERY_ATTEMPTS recovery prompts are issued.
        self.assertEqual(
            events.count("recovery_prompt"),
            agent.MAX_RECOVERY_ATTEMPTS,
        )
        # No tool was ever executed in this scenario.
        cur = self.db.connection.execute("SELECT COUNT(*) AS n FROM tool_calls")
        self.assertEqual(cur.fetchone()["n"], 0)
        # Proof of boundedness: only two model turns were used (one recovery
        # plus the accepted final answer), not all of max_steps.
        self.assertEqual(llm.calls, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
