class AgentError(Exception):
    """
    Base class for agent-level execution errors.

    Distinguishes genuine agent/orchestrator failures from ordinary tool
    execution failures (which are returned to the model as a TOOL_RESULT so
    it can recover).
    """


class AgentMaxStepsError(AgentError):
    """
    Raised when an agent task exhausts its maximum allowed steps without
    producing a final answer.

    By the time this exception is raised, the task has already been persisted
    in the MAX_STEPS state with a ``max_steps_reached`` event recorded, so the
    task is never left in an ambiguous state.
    """

    def __init__(self, task_id: int, max_steps: int, task: str):
        self.task_id = task_id
        self.max_steps = max_steps
        self.task = task
        super().__init__(
            f"Maximum agent steps ({max_steps}) exhausted for task #{task_id} "
            f"without a final answer."
        )


class AgentMalformedToolCallError(AgentError):
    """
    Raised when the agent exhausts its bounded recovery budget after the model
    repeatedly emitted malformed TOOL_CALL blocks that the strict parser
    rejected.

    This does NOT weaken the parser: ``_extract_tool_call`` remains strict and
    still refuses to parse a malformed block. The error is raised only by the
    agent lifecycle, and only after the model has been given a bounded number
    of chances to regenerate a valid TOOL_CALL and failed to do so.

    By the time it is raised, the task has already been persisted in the
    FAILED state via the standard task-failure path, so the task is never
    left in an ambiguous state.
    """

    def __init__(self, task_id: int, task: str, attempts: int):
        self.task_id = task_id
        self.attempts = attempts
        super().__init__(
            f"Task #{task_id} failed: the model repeatedly produced "
            f"malformed TOOL_CALL blocks that could not be regenerated into a "
            f"valid call ({attempts} recovery attempts exhausted)."
        )
