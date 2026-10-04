import os
import shutil
import subprocess
from pathlib import Path

from tools.base import ToolBase


def _resolve_git_executable() -> str:
    """Locate a usable git executable.

    Git is often installed but not present on the orchestrator PATH, so we
    first look on PATH and then fall back to the well-known Windows install
    locations. Returns the absolute path to git (or the bare name 'git' when
    nothing more specific is found, which still works if git is on PATH).
    """
    found = shutil.which("git")
    if found:
        return found

    candidates = [
        r"C:\Program Files\Git\cmd\git.exe",
        r"C:\Program Files (x86)\Git\cmd\git.exe",
        r"C:\Program Files\Git\bin\git.exe",
    ]
    if os.name == "nt" and os.environ.get("ProgramFiles"):
        pf = os.environ["ProgramFiles"]
        candidates.append(os.path.join(pf, "Git", "cmd", "git.exe"))
        candidates.append(os.path.join(pf, "Git", "bin", "git.exe"))
    pf86 = os.environ.get("ProgramFiles(x86)") if os.name == "nt" else None
    if pf86:
        candidates.append(os.path.join(pf86, "Git", "cmd", "git.exe"))

    for cand in candidates:
        if cand and os.path.isfile(cand):
            return cand

    return "git"


class GitTool(ToolBase):
    """Git version-control tool for the institute.

    Wraps the git executable (located on PATH or at its well-known Windows
    install path) so the agent can inspect and manage the institute
    repository. Commands run inside the institute root with a hard timeout.
    Working-directory changes are constrained to stay inside the institute
    root. Failures are returned as ``success=False`` results (never raised)
    so the model can inspect the output and recover.

    This is a controlled tool, not a true sandbox: it exists to support
    controlled autonomous development on this local machine.
    """

    @property
    def name(self) -> str:
        return "git"

    @property
    def description(self) -> str:
        return (
            "Run Git commands against the institute repository. Git is "
            "resolved from PATH or the well-known Windows install location, "
            "commands run inside the institute root with a hard timeout, and "
            "working directories are constrained to stay inside the institute "
            "root. Failures are returned (not raised) as success=false so the "
            "result can be inspected."
        )

    def __init__(self, root: str, timeout: int = 60):
        self.root = Path(root).resolve()
        self.timeout = timeout
        self.git_exe = _resolve_git_executable()

    # --- schema ----------------------------------------------------------
    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "operations": [
                {
                    "name": "execute",
                    "description": (
                        "Run a git subcommand (e.g. 'status', 'add', 'commit', "
                        "'log', 'diff', 'init', 'branch'). Returns success, "
                        "return_code, stdout, stderr, and the git command that "
                        "was run. Failures are returned (not raised) as "
                        "success=false so the result can be inspected."
                    ),
                    "parameters": [
                        {
                            "name": "command",
                            "type": "str",
                            "description": (
                                "The git subcommand to run, e.g. 'status', "
                                "'add', 'commit', 'log', 'diff', 'init'."
                            ),
                            "required": True,
                        },
                        {
                            "name": "args",
                            "type": "list[str]",
                            "description": (
                                "Optional list of additional arguments for the "
                                "git subcommand (e.g. ['-m', 'Initial commit'] "
                                "for commit, ['.'] for add)."
                            ),
                            "required": False,
                        },
                        {
                            "name": "working_directory",
                            "type": "str",
                            "description": (
                                "Optional working directory, relative to the "
                                "institute root. Defaults to the root itself. "
                                "Must stay inside the institute root."
                            ),
                            "required": False,
                        },
                    ],
                    "example_call": {
                        "tool": "git",
                        "command": "commit",
                        "args": ["-m", "Initial commit"],
                    },
                },
            ],
        }

    # --- helpers ---------------------------------------------------------
    def _safe_path(self, path: str) -> Path:
        candidate = (self.root / path).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise PermissionError(
                f"Working directory outside institute root denied: {path}"
            )
        return candidate

    def _clean_args(self, args) -> list:
        if args is None:
            return []
        if isinstance(args, str):
            # Tolerate a single string by splitting on whitespace.
            return args.split()
        if not isinstance(args, (list, tuple)):
            raise TypeError("'args' must be a list of strings when provided.")
        return [str(a) for a in args]

    # --- execution -------------------------------------------------------
    def execute(self, command: str, args=None, working_directory: str = None, **_):
        if not isinstance(command, str) or not command.strip():
            return {
                "success": False,
                "command": command,
                "return_code": None,
                "error": "A non-empty 'command' string is required.",
            }

        command = command.strip()

        try:
            extra_args = self._clean_args(args)
        except TypeError as e:
            return {
                "success": False,
                "command": command,
                "return_code": None,
                "error": str(e),
            }

        try:
            if working_directory:
                cwd = self._safe_path(working_directory)
            else:
                cwd = self.root
        except PermissionError as e:
            return {
                "success": False,
                "command": command,
                "return_code": None,
                "error": str(e),
            }

        return self._run([command, *extra_args], cwd)

    def _run(self, argv: list, cwd: Path) -> dict:
        result = {
            "success": False,
            "command": "git " + " ".join(argv),
            "return_code": None,
        }

        cmd = [self.git_exe, *argv]

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            result["return_code"] = proc.returncode
            if proc.stdout is not None:
                result["stdout"] = proc.stdout
            if proc.stderr is not None:
                result["stderr"] = proc.stderr
            result["success"] = proc.returncode == 0
            return result

        except subprocess.TimeoutExpired:
            result["error"] = (
                f"Git command exceeded the {self.timeout} second timeout "
                f"and was terminated."
            )
            return result

        except FileNotFoundError:
            result["error"] = (
                f"Git executable not found: {self.git_exe}. "
                f"Install Git or add it to PATH."
            )
            return result

        except Exception as e:
            result["error"] = str(e)
            return result
