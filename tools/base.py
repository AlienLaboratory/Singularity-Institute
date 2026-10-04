from abc import ABC, abstractmethod


class ToolBase(ABC):
    """
    Abstract base class for all institute tools.

    Every concrete tool must provide:
      - name: unique identifier used in TOOL_CALL dispatch
      - description: human-readable summary
      - schema(): structured metadata for auto-generated documentation
      - execute(**kwargs): the actual operation

    The schema() method returns a dictionary with the following shape:

    {
        "name": str,
        "description": str,
        "operations": [
            {
                "name": str,
                "description": str,
                "parameters": [
                    {
                        "name": str,
                        "type": str,
                        "description": str,
                        "required": bool
                    },
                    ...
                ],
                "example_call": { ... }   # a sample TOOL_CALL payload
            },
            ...
        ]
    }

    This allows ToolRegistry to generate model-facing documentation
    without any hard-coded strings in the agent prompt.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @abstractmethod
    def schema(self) -> dict:
        """Return structured self-description for documentation generation."""
        ...

    @abstractmethod
    def execute(self, **kwargs):
        """Execute a tool operation. Returns a dict result."""
        ...
