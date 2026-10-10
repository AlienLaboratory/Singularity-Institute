import json
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum

from core.exceptions import (
    AgentMalformedToolCallError,
    AgentMaxStepsError,
)
from memory.database import Database
from accounting import (
    DEFAULT_GPU_WATTS,
    DEFAULT_TARIFF_UAH_PER_KWH,
    TaskAccounting,
)


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

    Token-usage fields (``prompt_eval_count``, ``eval_count``) are
    accumulated across every LLM call within the task.  A value of ``None``
    means the metadata for at least one call was missing or malformed, so
    the total is *unavailable* rather than a silently-undercounted number.
    """

    task_id: int
    state: TaskState
    answer: str
    steps: int = 0
    error: str | None = None
    reason: str | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None

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

    # Bounded recovery budget for *narration before any tool* (see :meth:`run`)
    # and for *malformed TOOL_CALL blocks*. While the agent has not yet
    # executed any tool, a model response that contains no TOOL_CALL may be a
    # narration of an intended action rather than a final answer; likewise a
    # response that contains a TOOL_CALL marker but is syntactically malformed
    # is a recoverable model error rather than an immediately fatal one. To
    # stop either case from terminating an actionable task prematurely, the
    # agent issues a recovery prompt a bounded number of times (see the two
    # limits below) before either accepting a non-tool response as the final
    # answer or failing cleanly. Both limits are intentionally small so the
    # recovery can never loop indefinitely, and the overall loop remains
    # bounded by ``max_steps``.
    MAX_RECOVERY_ATTEMPTS = 1

    # Bounded recovery budget for *malformed TOOL_CALL blocks*. When the model
    # emits a TOOL_CALL marker that the strict parser rejects (bad JSON,
    # non-object payload, missing 'tool' field, or a missing END_TOOL_CALL
    # marker), NOTHING is executed. The agent then gives the model a bounded
    # number of chances to regenerate a valid call before failing cleanly.
    #
    # This limit is independent of the model's wording, is distinct from
    # ``MAX_RECOVERY_ATTEMPTS`` (which governs pre-tool narration), is shadable
    # per-instance for tests, and is always applied within the ``max_steps``
    # lifecycle bound so it can never exceed the normal step budget.
    MAX_MALFORMED_TOOL_CALL_RECOVERIES = 2

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

        # Per-task token-usage accumulators (reset at the start of each run()).
        # A value of None means the total is *unavailable* because at least
        # one LLM call had missing or malformed metadata.
        self._prompt_eval_total: int | None = None
        self._eval_total: int | None = None

        # Task start time (monotonic clock) used to estimate elapsed time for
        # the electricity-cost portion of the accounting row. Set at the start
        # of each run(); initialized to None so a task that never starts cannot
        # produce a fabricated elapsed value.
        self._task_start_time: float | None = None

    # ------------------------------------------------------------------
    # Token-usage accounting
    # ------------------------------------------------------------------

    def _accumulate_token_usage(self) -> None:
        """
        Read ``self.llm.last_response`` (if the LLM exposes it) and
        accumulate ``prompt_eval_count`` and ``eval_count`` into the
        per-task totals (``self._prompt_eval_total`` /
        ``self._eval_total``).

        Rules:
          * If ``last_response`` is absent (attribute not set, or None),
            both totals are marked **unavailable** (set to ``None``).
          * If ``last_response`` is not a dict, both totals are marked
            unavailable.
          * If a field (``prompt_eval_count`` / ``eval_count``) is missing
            from the dict, or is not a finite number (int/float, not bool),
            the *corresponding* total is marked unavailable.
          * Once a total is marked unavailable it stays unavailable for the
            remainder of the task (a missing value cannot be recovered by
            later calls).

        Existing fake/test LLMs that do not define ``last_response``
        continue to work: the totals simply become ``None``.
        """

        last_response = getattr(self.llm, "last_response", None)

        if last_response is None:
            # No metadata at all — both totals unavailable.
            self._prompt_eval_total = None
            self._eval_total = None
            return

        if not isinstance(last_response, dict):
            # Malformed: expected a dict but got something else.
            self._prompt_eval_total = None
            self._eval_total = None
            return

        # --- prompt_eval_count ---
        pec = last_response.get("prompt_eval_count")
        if (
            not isinstance(pec, (int, float))
            or isinstance(pec, bool)
        ):
            self._prompt_eval_total = None  # mark unavailable
        else:
            if self._prompt_eval_total is not None:
                self._prompt_eval_total += pec
            # If already None, it stays None.

        # --- eval_count ---
        ec = last_response.get("eval_count")
        if (
            not isinstance(ec, (int, float))
            or isinstance(ec, bool)
        ):
            self._eval_total = None  # mark unavailable
        else:
            if self._eval_total is not None:
                self._eval_total += ec
            # If already None, it stays None.

    def _reset_token_usage(self) -> None:
        """Reset per-task token totals to zero (available) for a new run."""
        self._prompt_eval_total = 0
        self._eval_total = 0

    # ------------------------------------------------------------------
    # Task accounting (API-equivalent cost + electricity) persistence
    # ------------------------------------------------------------------

    def _task_elapsed_seconds(self) -> float | None:
        """
        Return elapsed seconds since the task started, or ``None`` if the
        start time was never recorded. ``0`` is a valid value (a task that
        consumed no time legitimately costs no electricity).
        """
        if self._task_start_time is None:
            return None
        return time.monotonic() - self._task_start_time

    def _persist_task_accounting(self, task_id: int) -> None:
        """
        Persist exactly one accounting row for the just-finished task.

        Uses the :mod:`accounting` helpers to derive the Opus-5-equivalent API
        cost (USD) from the accumulated prompt/eval token counts and the
        estimated GPU electricity cost (UAH) from the measured elapsed time,
        then stores the row via :meth:`Database.save_accounting`.

        Rules honored:
          * Missing token counts (``None``) yield an **unavailable** API cost
            (``None`` + reason) -- never a fabricated ``0``.
          * Electricity is estimated with the module-level default GPU
            wattage and tariff.
          * Any failure here is logged and swallowed: accounting persistence
            must NEVER mask the original task outcome or exception.
        """
        try:
            accounting = TaskAccounting(
                input_tokens=self._prompt_eval_total,
                output_tokens=self._eval_total,
                elapsed_seconds=self._task_elapsed_seconds(),
            ).summarize(
                gpu_watts=DEFAULT_GPU_WATTS,
                tariff_uah_per_kwh=DEFAULT_TARIFF_UAH_PER_KWH,
            )

            self.database.save_accounting(
                task_id,
                input_tokens=accounting.input_tokens,
                output_tokens=accounting.output_tokens,
                api_cost_usd=accounting.api_cost_usd,
                electricity_cost_uah=accounting.electricity_cost_uah,
                elapsed_seconds=accounting.elapsed_seconds,
                gpu_watts=DEFAULT_GPU_WATTS,
                tariff_uah_per_kwh=DEFAULT_TARIFF_UAH_PER_KWH,
                api_cost_reason=accounting.api_cost_reason,
                electricity_cost_reason=accounting.electricity_cost_reason,
            )
        except Exception:
            # Accounting persistence must never mask the original task outcome
            # or exception; log and continue.
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Main execution loop
    # ------------------------------------------------------------------

    def run(self, task: str) -> AgentResult:
        """
        Execute a task through a full agent lifecycle with explicit state
        transitions.

        Lifecycle:
          RUNNING -> (tool loop) -> COMPLETED | MAX_STEPS | FAILED

        Every significant transition is persisted via the database event
        mechanism:
          task_started, tool_requested, tool_completed, tool_call_rejected,
          recovery_prompt, task_completed, task_failed, max_steps_reached

        Returns an :class:`AgentResult`. On max-step exhaustion an
        :class:`AgentMaxStepsError` is raised after the task has been
        persisted in the MAX_STEPS state.

        Model responses are dispatched by how ``_extract_tool_call`` treats
        them:

          * returns a dict       -> a valid TOOL_CALL; execute the tool.
          * returns None         -> no TOOL_CALL marker; treated as narration
                                    (bounded recovery) or a final answer.
          * raises ValueError    -> a TOOL_CALL marker was present but the
                                    strict parser rejected it. Nothing is
                                    executed; the agent issues a bounded
                                    recovery prompt (see
                                    ``MAX_MALFORMED_TOOL_CALL_RECOVERIES``)
                                    asking the model to regenerate a valid
                                    call before failing cleanly.

        A model response that merely narrates an intended action without
        emitting a TOOL_CALL does not immediately terminate an actionable
        task: a bounded number of recovery prompts are issued (see
        ``MAX_RECOVERY_ATTEMPTS``) before a non-tool response is accepted as
        the final answer. Once any tool has been executed, a non-tool
        response is a legitimate final answer and is accepted immediately.

        All recovery is bounded and always runs inside the single
        ``for range(max_steps)`` loop, so ``max_steps`` remains authoritative
        and no recovery path can loop indefinitely.

        Token usage: after every successful LLM call the agent reads
        ``self.llm.last_response`` (if available) and accumulates
        ``prompt_eval_count`` and ``eval_count`` across all calls in the
        task. If metadata is missing or malformed for any call the
        corresponding total is marked ``None`` (unavailable). The totals
        are included in the returned :class:`AgentResult` and in the
        persisted lifecycle events.
        """

        task_id = self.database.create_task(task)
        state = TaskState.RUNNING

        # Reset per-task token accumulators and record the task start time
        # (used later to estimate the electricity-cost portion of the
        # accounting row).
        self._reset_token_usage()
        self._task_start_time = time.monotonic()

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

        # Bounded recovery state.
        #
        # ``tool_executed`` flips to True as soon as the model emits a
        # TOOL_CALL (i.e. it has started acting). From that point on a
        # non-tool response is a legitimate final answer.
        #
        # ``recovery_attempts`` counts how many recovery prompts have been
        # issued *before the first tool* for narration (capped by
        # ``MAX_RECOVERY_ATTEMPTS``).
        #
        # ``malformed_recoveries`` counts how many recovery prompts have been
        # issued for malformed TOOL_CALL blocks (capped by
        # ``MAX_MALFORMED_TOOL_CALL_RECOVERIES``).
        #
        # Both counters are bounded independently of the model's wording, and
        # the outer ``for`` loop remains bounded by ``max_steps``.
        recovery_attempts = 0
        tool_executed = False
        malformed_recoveries = 0

        try:

            for step in range(self.max_steps):

                steps = step + 1
                print(f"\n--- STEP {steps} ---")

                response = self.llm.chat(messages)

                # Accumulate token usage from this successful LLM call.
                self._accumulate_token_usage()

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

                # Parse the tool call. ``_extract_tool_call`` stays strict:
                # it returns a dict for a valid call, None when there is no
                # TOOL_CALL marker, and raises ValueError when a marker is
                # present but malformed. Only the ValueError branch below is
                # new; it is the parser's own structured rejection signal and
                # does not depend on matching any specific output phrase.
                try:
                    tool_call = self._extract_tool_call(response)
                except ValueError as parse_error:
                    # A malformed TOOL_CALL was detected. Nothing is executed.
                    # The helper appends a recovery instruction (bounded by
                    # ``MAX_MALFORMED_TOOL_CALL_RECOVERIES``) and returns the
                    # updated recovery count so the loop can continue within
                    # the existing ``max_steps`` bound. If the budget is
                    # exhausted it raises AgentMalformedToolCallError, which
                    # is handled by the generic failure path below.
                    malformed_recoveries = self._recover_from_malformed_call(
                        task_id=task_id,
                        task=task,
                        steps=steps,
                        parse_error=parse_error,
                        messages=messages,
                        malformed_recoveries=malformed_recoveries,
                        max_recoveries=self.MAX_MALFORMED_TOOL_CALL_RECOVERIES,
                    )
                    continue

                # A response with no TOOL_CALL.
                if not tool_call:

                    # While no tool has been executed yet, a non-tool response
                    # may be a narration of an intended action (e.g.
                    # "I'll start by inspecting the repository") rather than a
                    # final answer. Issue a bounded recovery prompt and
                    # continue instead of terminating the task.
                    if (
                        not tool_executed
                        and recovery_attempts < self.MAX_RECOVERY_ATTEMPTS
                    ):

                        recovery_attempts += 1

                        self.database.add_event(
                            task_id,
                            "recovery_prompt",
                            {
                                "step": steps,
                                "recovery_attempts": recovery_attempts,
                                "reason": (
                                    "non-tool response issued before any "
                                    "tool execution"
                                ),
                            }
                        )

                        # Appended to the conversation for the LLM (consistent
                        # with how TOOL_RESULT messages are handled) and
                        # recorded as a lifecycle event above.
                        messages.append({
                            "role": "user",
                            "content": self._build_recovery_instruction()
                        })

                        continue

                    # Either a tool has already been executed (legitimate
                    # final answer) or the recovery budget before the first
                    # tool has been exhausted (bounded). Accept the response
                    # as the final answer.
                    state = TaskState.COMPLETED

                    self.database.complete_task(
                        task_id,
                        TaskState.COMPLETED.value
                    )

                    self.database.add_event(
                        task_id,
                        "task_completed",
                        {
                            "state": state.value,
                            "steps": steps,
                            "prompt_eval_count": self._prompt_eval_total,
                            "eval_count": self._eval_total,
                        }
                    )

                    # Persist one accounting row for this completed task.
                    self._persist_task_accounting(task_id)

                    return AgentResult(
                        task_id=task_id,
                        state=state,
                        answer=response,
                        steps=steps,
                        prompt_eval_count=self._prompt_eval_total,
                        eval_count=self._eval_total,
                    )

                # A TOOL_CALL was emitted: the model is acting. From this point
                # on, a later non-tool response is a final answer.
                tool_executed = True

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
                    "prompt_eval_count": self._prompt_eval_total,
                    "eval_count": self._eval_total,
                }
            )

            # Persist one accounting row for this max-steps task.
            self._persist_task_accounting(task_id)

            # Raised after persistence so the task state is unambiguous.
            raise AgentMaxStepsError(
                task_id=task_id,
                max_steps=self.max_steps,
                task=task,
            )

        except AgentMaxStepsError:
            # Already persisted in the MAX_STEPS state; re-raise as-is.
            raise

        except AgentMalformedToolCallError:
            # Bounded malformed-TOOL_CALL recovery budget exhausted. The
            # generic failure path below persists the FAILED state and re-
            # raises, so the task is never left in an ambiguous condition.
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
                {
                    "state": state.value,
                    "error": str(e),
                    "reason": str(e),
                    "prompt_eval_count": self._prompt_eval_total,
                    "eval_count": self._eval_total,
                }
            )

            # Persist one accounting row for this failed task, preserving the
            # token counts already accumulated before the failure.
            self._persist_task_accounting(task_id)

            raise

    # ------------------------------------------------------------------
    # Recovery helpers
    # ------------------------------------------------------------------

    def _recover_from_malformed_call(
        self,
        task_id,
        task,
        steps,
        parse_error,
        messages,
        malformed_recoveries,
        max_recoveries,
    ):
        """
        Respond to a malformed TOOL_CALL block (a TOOL_CALL marker that the
        strict parser rejected).

        NOTHING is executed for a malformed block: the tool is not invoked and
        no tool result is produced. This helper:

          * records the rejection as a ``tool_call_rejected`` lifecycle event
            (auditable as a NOT-executed call, with the parser's reason);
          * if the bounded recovery budget has not been exhausted, appends an
            explicit recovery instruction to the conversation for the model
            (asking it to regenerate a valid response) and RETURNS the updated
            recovery count so the caller can ``continue`` the loop within the
            existing ``max_steps`` bound;
          * if the budget is exhausted, raises
            :class:`AgentMalformedToolCallError` so the task fails cleanly
            rather than looping forever.

        The budget is a small, explicit, wording-independent count and is
        always applied inside the ``max_steps`` loop, so it can never exceed
        the normal lifecycle bound.
        """

        if malformed_recoveries < max_recoveries:

            malformed_recoveries += 1

            # Record that a malformed call was rejected and NOT executed.
            self.database.add_event(
                task_id,
                "tool_call_rejected",
                {
                    "step": steps,
                    "recovery_attempts": malformed_recoveries,
                    "reason": str(parse_error),
                    "executed": False,
                    "recovery_instruction_sent": True,
                }
            )

            # Emit the shared recovery event so the recovery path is
            # consistently auditable alongside narration recovery.
            self.database.add_event(
                task_id,
                "recovery_prompt",
                {
                    "step": steps,
                    "reason": "malformed TOOL_CALL rejected by the parser",
                    "recovery_attempts": malformed_recoveries,
                }
            )

            instruction = self._build_malformed_recovery_instruction(
                str(parse_error)
            )

            print("\n[TOOL CALL REJECTED] (not executed)")
            print(instruction)

            # Appended to the conversation for the LLM and recorded above.
            messages.append({
                "role": "user",
                "content": instruction
            })

            return malformed_recoveries

        # Budget exhausted: record the final rejection and fail cleanly.
        self.database.add_event(
            task_id,
            "tool_call_rejected",
            {
                "step": steps,
                "recovery_attempts": malformed_recoveries,
                "reason": str(parse_error),
                "executed": False,
                "recovery_instruction_sent": False,
            }
        )

        print(
            "\n[TOOL CALL REJECTED] (not executed) "
            "- recovery budget exhausted"
        )

        raise AgentMalformedToolCallError(
            task_id=task_id,
            task=task,
            attempts=malformed_recoveries,
        )

    def _build_recovery_instruction(self) -> str:
        """
        Follow-up instruction sent when the model narrated an intended action
        but did not execute it with a TOOL_CALL.

        It explicitly tells the model it announced an action without executing
        it and that it must now perform the action using the appropriate tool.
        It also leaves an escape hatch so a genuine final answer (work already
        complete, no tool needed) can still be given.
        """

        return (
            "You described an intended action, but your response did not "
            "include a TOOL_CALL to actually execute it. Do not restate your "
            "plan. Perform the action now using the appropriate tool, "
            "producing a valid TOOL_CALL ... END_TOOL_CALL block. If the task "
            "is in fact already complete and requires no further tool, state "
            "your final answer directly."
        )

    def _build_malformed_recovery_instruction(self, reason: str) -> str:
        """
        Follow-up instruction sent when the model emitted a malformed TOOL_CALL
        that the strict parser rejected (and therefore did not execute).

        It tells the model that its previous call was invalid and NOT executed,
        shows the parser's reason so the specific defect can be fixed, and asks
        it to regenerate a valid response. It also leaves an escape hatch so a
        genuine final answer (work already complete, no tool needed) can still
        be given, and forbids claiming a non-executed operation succeeded.
        """

        return (
            "Your previous response contained an invalid TOOL_CALL and was NOT "
            "executed. No tool was run and no result was produced. The parser "
            f"reported: {reason}\n\n"
            "Regenerate the response using the required TOOL_CALL format. If "
            "you need a tool, emit exactly one well-formed TOOL_CALL ... "
            "END_TOOL_CALL block containing a single valid JSON object with a "
            "'tool' field and the correct operation and parameters. Do not "
            "claim that the operation succeeded unless a tool result confirms "
            "it. If the task is in fact already complete and requires no "
            "further tool, state your final answer directly."
        )

    # ------------------------------------------------------------------
    # Tool-call extraction and execution
    # ------------------------------------------------------------------

    def _extract_tool_call(self, text):
        """
        Extract the first valid tool call from the model response.

        TOOL_CALL and END_TOOL_CALL are recognized only when they appear
        as standalone lines. This prevents marker text contained inside
        JSON strings from being mistaken for protocol markers.

        This parser stays strict: it returns a dict for a valid call, None
        when there is no TOOL_CALL marker, and raises ValueError when a
        marker is present but malformed. It never parses or invents a call
        from malformed text.
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
