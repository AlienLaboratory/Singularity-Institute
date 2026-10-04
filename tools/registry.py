import json
from typing import Optional


class ToolRegistry:
    """
    Central registry for all tools available to the agent.

    Responsibilities:
      - Register / retrieve tools by name
      - Provide structured schemas for each tool
      - Generate model-facing tool documentation automatically
    """

    def __init__(self):
        self._tools: dict = {}

    def register(self, tool) -> None:
        """
        Register a tool instance.

        The tool must have a .name attribute (string).
        """
        if not hasattr(tool, "name") or not isinstance(tool.name, str):
            raise TypeError(
                f"Tool must have a 'name' attribute (str). Got: {type(tool)}"
            )
        self._tools[tool.name] = tool

    def get(self, name: str):
        """Retrieve a tool by name, or None if not found."""
        return self._tools.get(name)

    def list_tools(self) -> list[dict]:
        """
        Return a list of tool metadata dicts (name + description).
        """
        return [
            {"name": tool.name, "description": tool.description}
            for tool in self._tools.values()
        ]

    def get_tool_schema(self, name: str) -> Optional[dict]:
        """
        Return the full structured schema for a single tool, or None.
        """
        tool = self._tools.get(name)
        if tool is None:
            return None
        return tool.schema()

    def all_schemas(self) -> list[dict]:
        """
        Return the structured schemas for all registered tools.
        """
        return [tool.schema() for tool in self._tools.values()]

    def generate_tool_docs(self) -> str:
        """
        Generate a documentation block describing all registered tools
        and their operations, suitable for inclusion in the system prompt.

        This method is the single source of truth for model-facing
        tool documentation.  No tool details should be hard-coded
        elsewhere.
        """
        lines: list[str] = []

        for index, tool in enumerate(self._tools.values(), start=1):
            schema = tool.schema()

            lines.append(f"{index}. **{schema['name']}**")
            lines.append("")

            # Description
            lines.append(schema["description"].strip())
            lines.append("")

            # Operations
            for op in schema.get("operations", []):
                lines.append(f"### {op['name']}")
                lines.append("")
                if op.get("description"):
                    lines.append(op["description"].strip())
                    lines.append("")

                # Parameters
                params = op.get("parameters", [])
                if params:
                    lines.append("Parameters:")
                    for p in params:
                        req = "(required)" if p["required"] else "(optional)"
                        lines.append(
                            f"  - `{p['name']}` ({p['type']}) {req} — {p['description']}"
                        )
                    lines.append("")

                # Example call
                example = op.get("example_call")
                if example:
                    lines.append("Example:")
                    lines.append("")
                    lines.append("TOOL_CALL")
                    lines.append(json.dumps(example, indent=2))
                    lines.append("END_TOOL_CALL")
                    lines.append("")

        return "\n".join(lines)
