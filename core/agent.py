import json
from dataclasses import dataclass
from enum import Enum

from core.exceptions import AgentMaxStepsError
from memory.database import Database


class TaskState(Enum):
    """
    Explicit execution states for an agent task.

    A task is always in exactly one of these states; the state is persisted
    to the database so it is never left in an ambiguous condition.
    """

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    MAX_STEPS = "max_steps"


@dataclass
class AgentResult:
    """
    Structured outcome returned by :meth:`Agent.run`.

    ``str(result)`` is the model's final answer text so that existing callers
    that simply ``print(result)`` remain compatible.
    """

    task_id: int
    state: TaskState
    answer: str
    steps: int = 0
    error: str | None = None
    reason: str | None = None

    def __str__(self) -> str:  # keep CLI ``print(result)`` compatible
        return self.answer

    def __repr__(self) -> str:
        return (
            f"AgentResult(task_id={self.task_id}, "
            f"state={self.state.value!r}, steps={self.steps})"
        )


def build_system_prompt(registry) -> str:
    """
    Build the agent system prompt by combining static agent instructions
    with auto-generated tool documentation from the registry.

    No tool names, operations, or parameters are hard-coded here —
    they come entirely from the registry's structured schemas.
    """

    tool_docs = registry.generate_tool_docs()

    return f"""
You are the primary reasoning core of a local AI scientific research institute.

You are an agent, not merely a chatbot.

You have access to tools provided by the Python orchestration layer.

# Available Tools

{tool_docs}
# Usage Rules

- All filesystem paths are relative to the institute root.
- Do not create another institute directory.
- Never claim that an operation succeeded unless the tool result confirms it.
- After receiving a tool result, continue reasoning if additional work is required.

# Workflow for Code Changes

1. Inspect the existing code.
2. Identify the required change.
3. Modify the code using the filesystem tool.
4. Use the python tool to test or verify the change.
5. Inspect the result.
6. If the test fails, diagnose and modify the code.
7. Test again.
8. Only claim success after a successful tool result.

You are allowed to modify the institute codebase when explicitly asked to
implement or improve something.

Be concise when the task is simple.
"""


class Agent:

    def __init__(
        self,
        llm,
        tools,
        database,
        max_steps=20
    ):

        self.llm = llm
        self.tools = tools
        self.database = database
        self.max_steps = max_steps

        # Build the system prompt once, dynamically from the registry
        self._system_prompt = build_system_prompt(tools)

    def run(self, task: str) -> AgentResult:
        """
        Execute a task through a full agent lifecycle with explicit state
        transitions.

        Lifecycle:
          RUNNING -> (tool loop) -> COMPLETED | MAX_STEPS | FAILED

        Every significant transition is persisted via the database event
        mechanism:
          task_started, tool_requested, tool_completed,
          task_completed, task_failed, max_steps_reached

        Returns an :class:`AgentResult`. On max-step exhaustion an
        :class:`AgentMaxStepsError` is raised after the task has been
        persisted in the MAX_STEPS state.
        """

        task_id = self.database.create_task(task)
        state = TaskState.RUNNING

        self.database.add_event(
            task_id,
            "task_started",
            {"task": task, "state": state.value}
        )

        messages = [
            {
                "role": "system",
                "content": self._system_prompt
            },
            {
                "role": "user",
                "content": task
            }
        ]

        self.database.add_message(
            task_id,
            "user",
            task
        )

        steps = 0

        try:

            for step in range(self.max_steps):

                steps = step + 1
                print(f"\n--- STEP {steps} ---")

                response = self.llm.chat(messages)

                print(response)

                messages.append({
                    "role": "assistant",
                    "content": response
                })

                self.database.add_message(
                    task_id,
                    "assistant",
                    response
                )

                tool_call = self._extract_tool_call(response)

                # A response with no TOOL_CALL is the final answer.
                if not tool_call:

                    state = TaskState.COMPLETED

                    self.database.complete_task(
                        task_id,
                        TaskState.COMPLETED.value
                    )

                    self.database.add_event(
                        task_id,
                        "task_completed",
                        {"state": state.value, "steps": steps}
                    )

                    return AgentResult(
                        task_id=task_id,
                        state=state,
                        answer=response,
                        steps=steps,
                    )

                arguments = {
                    k: v for k, v in tool_call.items() if k != "tool"
                }

                # Record the tool request as a lifecycle event.
                self.database.add_event(
                    task_id,
                    "tool_requested",
                    {
                        "tool": tool_call["tool"],
                        "arguments": arguments,
                        "step": steps,
                    }
                )

                result = self._execute_tool(tool_call)

                print("\n[TOOL RESULT]")
                print(json.dumps(result, indent=2))

                # Existing tool_calls persistence (unchanged).
                self.database.add_tool_call(
                    task_id,
                    tool_call["tool"],
                    tool_call,
                    result
                )

                # Record the tool outcome as a lifecycle event.
                self.database.add_event(
                    task_id,
                    "tool_completed",
                    {
                        "tool": tool_call["tool"],
                        "success": bool(result.get("success", False)),
                        "step": steps,
                        "result": result,
                    }
                )

                messages.append({
                    "role": "user",
                    "content": (
                        "TOOL_RESULT\n"
                        + json.dumps(
                            result,
                            indent=2
                        )
                        + "\nEND_TOOL_RESULT"
                    )
                })

            # Exhausted all steps without a final answer.
            state = TaskState.MAX_STEPS
            reason = (
                f"Maximum agent steps ({self.max_steps}) exhausted "
                f"without a final answer."
            )

            self.database.complete_task(
                task_id,
                TaskState.MAX_STEPS.value
            )

            self.database.add_event(
                task_id,
                "max_steps_reached",
                {
                    "state": state.value,
                    "reason": reason,
                    "max_steps": self.max_steps,
                    "steps": steps,
                }
            )

            # Raised after persistence so the task state is unambiguous.
            raise AgentMaxStepsError(
                task_id=task_id,
                max_steps=self.max_steps,
                task=task,
            )

        except AgentMaxStepsError:
            # Already persisted in the MAX_STEPS state; re-raise as-is.
            raise

        except Exception as e:

            state = TaskState.FAILED

            self.database.complete_task(
                task_id,
                TaskState.FAILED.value
            )

            self.database.add_event(
                task_id,
                "task_failed",
                {"state": state.value, "error": str(e), "reason": str(e)}
            )

            raise

    def _extract_tool_call(self, text):
        """
        Extract the first valid tool call from the model response.

        TOOL_CALL and END_TOOL_CALL are recognized only when they appear
        as standalone lines. This prevents marker text contained inside
        JSON strings from being mistaken for protocol markers.
        """

        lines = text.splitlines()

        start_index = None

        for i, line in enumerate(lines):
            if line.strip() == "TOOL_CALL":
                start_index = i
                break

        if start_index is None:
            return None

        for i in range(start_index + 1, len(lines)):
            if lines[i].strip() != "END_TOOL_CALL":
                continue

            payload = "\n".join(
                lines[start_index + 1:i]
            ).strip()

            try:
                result = json.loads(payload)

            except json.JSONDecodeError as e:

                raise ValueError(
                    "Invalid tool call generated by model:\n"
                    + payload
                ) from e

            if not isinstance(result, dict):

                raise ValueError(
                    "Tool call must be a JSON object."
                )

            if "tool" not in result:

                raise ValueError(
                    "Tool call is missing the 'tool' field."
                )

            return result

        raise ValueError(
            "TOOL_CALL marker found without "
            "a matching END_TOOL_CALL marker."
        )

    def _execute_tool(self, call):
        """
        Execute a tool call and return a structured result dict.

        Tool-level failures (unknown tool or an exception raised by the tool)
        are NOT raised to the caller; they are returned as a result with
        ``success=False`` so the model can receive a TOOL_RESULT and recover.
        """

        tool_name = call["tool"]

        tool = self.tools.get(tool_name)

        if tool is None:

            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}"
            }

        arguments = dict(call)

        arguments.pop("tool")

        try:

            return tool.execute(**arguments)

        except Exception as e:

            return {
                "success": False,
                "error": str(e)
            }
