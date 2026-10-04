from pathlib import Path

from tools.base import ToolBase


class FilesystemTool(ToolBase):

    @property
    def name(self) -> str:
        return "filesystem"

    @property
    def description(self) -> str:
        return (
            "Provides safe filesystem operations inside the institute root. "
            "All paths are relative to the institute root directory."
        )

    def __init__(self, root: str):
        self.root = Path(root).resolve()

    def _safe_path(self, path: str) -> Path:
        candidate = (self.root / path).resolve()

        if candidate != self.root and self.root not in candidate.parents:
            raise PermissionError(
                f"Access outside institute root denied: {path}"
            )

        return candidate

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "operations": [
                {
                    "name": "create_directory",
                    "description": "Create a directory (and parents if needed).",
                    "parameters": [
                        {
                            "name": "operation",
                            "type": "str",
                            "description": "Must be 'create_directory'.",
                            "required": True,
                        },
                        {
                            "name": "path",
                            "type": "str",
                            "description": "Relative path from institute root.",
                            "required": True,
                        },
                    ],
                    "example_call": {
                        "tool": "filesystem",
                        "operation": "create_directory",
                        "path": "research/new_study",
                    },
                },
                {
                    "name": "write_file",
                    "description": "Write text content to a file. Parent directories are created automatically.",
                    "parameters": [
                        {
                            "name": "operation",
                            "type": "str",
                            "description": "Must be 'write_file'.",
                            "required": True,
                        },
                        {
                            "name": "path",
                            "type": "str",
                            "description": "Relative path from institute root.",
                            "required": True,
                        },
                        {
                            "name": "content",
                            "type": "str",
                            "description": "Text content to write to the file.",
                            "required": True,
                        },
                    ],
                    "example_call": {
                        "tool": "filesystem",
                        "operation": "write_file",
                        "path": "notes/todo.txt",
                        "content": "Hello world",
                    },
                },
                {
                    "name": "read_file",
                    "description": "Read and return the full text content of a file.",
                    "parameters": [
                        {
                            "name": "operation",
                            "type": "str",
                            "description": "Must be 'read_file'.",
                            "required": True,
                        },
                        {
                            "name": "path",
                            "type": "str",
                            "description": "Relative path from institute root.",
                            "required": True,
                        },
                    ],
                    "example_call": {
                        "tool": "filesystem",
                        "operation": "read_file",
                        "path": "notes/todo.txt",
                    },
                },
                {
                    "name": "list_directory",
                    "description": "List the names of items (files and directories) in a directory.",
                    "parameters": [
                        {
                            "name": "operation",
                            "type": "str",
                            "description": "Must be 'list_directory'.",
                            "required": True,
                        },
                        {
                            "name": "path",
                            "type": "str",
                            "description": "Relative path from institute root. Defaults to '.' (root).",
                            "required": False,
                        },
                    ],
                    "example_call": {
                        "tool": "filesystem",
                        "operation": "list_directory",
                        "path": ".",
                    },
                },
            ],
        }

    def execute(self, operation: str, **kwargs):

        if operation == "create_directory":
            path = self._safe_path(kwargs["path"])
            path.mkdir(parents=True, exist_ok=True)

            return {
                "success": True,
                "operation": operation,
                "path": str(path),
            }

        if operation == "write_file":
            path = self._safe_path(kwargs["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                kwargs["content"],
                encoding="utf-8"
            )

            return {
                "success": True,
                "operation": operation,
                "path": str(path),
            }

        if operation == "read_file":
            path = self._safe_path(kwargs["path"])

            return {
                "success": True,
                "operation": operation,
                "path": str(path),
                "content": path.read_text(encoding="utf-8"),
            }

        if operation == "list_directory":
            path = self._safe_path(kwargs.get("path", "."))

            return {
                "success": True,
                "operation": operation,
                "path": str(path),
                "items": [
                    item.name
                    for item in path.iterdir()
                ],
            }

        raise ValueError(f"Unknown filesystem operation: {operation}")
