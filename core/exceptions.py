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
