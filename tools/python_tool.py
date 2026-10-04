import subprocess
import sys
import tempfile
from pathlib import Path

from tools.base import ToolBase


class PythonTool(ToolBase):

    @property
    def name(self) -> str:
        return "python"

    @property
    def description(self) -> str:
        return (
            "Execute Python code in a separate subprocess inside the institute "
            "workspace. Use for calculations, data analysis, numerical "
            "experiments, statistics, scientific computation, file generation, "
            "and running Python-based tests. The code is provided as a single "
            "string in the 'code' field."
        )

    def __init__(self, root: str, timeout: int = 60):

        self.root = Path(root).resolve()
        self.workspace = self.root / "workspace"
        self.workspace.mkdir(
            parents=True,
            exist_ok=True
        )

        self.timeout = timeout

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "operations": [
                {
                    "name": "execute",
                    "description": (
                        "Run a block of Python source code in an isolated "
                        "subprocess. Returns stdout, stderr, return_code, "
                        "and a success flag."
                    ),
                    "parameters": [
                        {
                            "name": "code",
                            "type": "str",
                            "description": (
                                "Complete Python source code to execute. "
                                "May span multiple lines."
                            ),
                            "required": True,
                        },
                    ],
                    "example_call": {
                        "tool": "python",
                        "code": "print(2 + 2)",
                    },
                },
            ],
        }

    def execute(self, code: str):

        if not isinstance(code, str):
            raise TypeError(
                "Python code must be provided as a string."
            )

        script_path = None

        try:

            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                encoding="utf-8",
                delete=False,
                dir=self.workspace
            ) as f:

                f.write(code)
                script_path = Path(f.name)

            process = subprocess.run(
                [
                    sys.executable,
                    str(script_path)
                ],
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            return {
                "success": process.returncode == 0,
                "return_code": process.returncode,
                "stdout": process.stdout,
                "stderr": process.stderr,
            }

        except subprocess.TimeoutExpired as e:

            partial = {}

            if getattr(e, "stdout", None) is not None:
                partial["stdout"] = e.stdout

            if getattr(e, "stderr", None) is not None:
                partial["stderr"] = e.stderr

            return {
                "success": False,
                "return_code": None,
                "error": (
                    f"Python execution exceeded the "
                    f"{self.timeout} second timeout and was terminated."
                ),
                **partial
            }

        except Exception as e:

            return {
                "success": False,
                "return_code": None,
                "error": str(e)
            }

        finally:

            if script_path is not None:

                try:
                    script_path.unlink()
                except Exception:
                    pass
