import os
import re
import subprocess
from pathlib import Path

from tools.base import ToolBase


# --- Security policy (kept simple and transparent) -------------------------

# Unconditionally rejected patterns. Patterns are written to avoid false
# positives: e.g. ``format`` is matched only as the DOS disk formatter
# (``format <arg>`` / ``format.exe``), NOT as the harmless PowerShell cmdlets
# ``Format-Table`` / ``Format-List`` / ``Format-Hex``.
_DANGEROUS_PATTERNS = (
    r"\bformat\s+\S",      # format C:  /  format /v c:   (disk formatter)
    r"\bformat\.exe\b",
    r"\bdiskpart\b",        # disk-partition manager
    r"\bshutdown\b",        # power-off
    r"\brestart\s*-\s*computer\b",  # Restart-Computer
    r"\bstop\s*-\s*computer\b",     # Stop-Computer
    r"\bcmd\.exe\b",        # escape to native cmd shell
    r"\bcmd\s+/",           # cmd /c ...
)

# Removal verbs (Remove-Item and its common PowerShell aliases).
_REMOVAL_VERBS = {"remove-item", "rm", "del", "erase"}
_REMOVAL_PRESENT = re.compile(
    r"\b(remove\s*-?item|rm|del|erase)\b", re.IGNORECASE
)

# Switches that consume the following token as a value (so that token is the
# actual path argument, not a switch).
_RI_VALUE_FLAGS = {"-literalpath", "-lp", "-path", "-p"}
# ---------------------------------------------------------------------------


class ShellTool(ToolBase):
    """Controlled Windows shell execution for the agent.

    Commands run under ``powershell.exe -NoProfile -NonInteractive`` with the
    working directory pinned inside the institute root, a hard timeout, and a
    transparent set of dangerous-command rejections. Failures are returned as
    ``success=False`` results (never raised) so the model can inspect the
    output and recover.

    On timeout the spawned process and its entire child-process tree are
    force-killed (via ``TerminateProcess`` plus ``taskkill /F /T`` on
    Windows), so an interactive subprocess such as a hung ``winget`` prompt
    cannot leave a zombie that blocks the agent indefinitely.

    NOTE: This is a basic command-level policy, not a true sandbox. The
    purpose is controlled autonomous development on this local machine.
    """

    @property
    def name(self) -> str:
        return "shell"

    @property
    def description(self) -> str:
        return (
            "Execute a controlled Windows PowerShell command. The working "
            "directory is pinned to the institute root, execution has a hard "
            "timeout, and a small set of dangerous commands (format, diskpart, "
            "shutdown, restart-computer, cmd, and remove-item targeting paths "
            "outside the institute root or the institute root itself) are "
            "rejected. This is a basic command policy, not a true sandbox: "
            "the purpose is controlled autonomous development on this local "
            "machine."
        )

    def __init__(self, root: str, timeout: int = 60):
        self.root = Path(root).resolve()
        self.timeout = timeout

    # --- schema ----------------------------------------------------------
    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "operations": [
                {
                    "name": "execute",
                    "description": (
                        "Run a single Windows PowerShell command. Returns "
                        "success, return_code, stdout, stderr, and the "
                        "command. Command failures are returned (not raised) "
                        "as success=false so the result can be inspected."
                    ),
                    "parameters": [
                        {
                            "name": "command",
                            "type": "str",
                            "description": (
                                "A PowerShell command to execute (e.g. "
                                "'Get-ChildItem'). Executed with "
                                "-NoProfile -NonInteractive from the "
                                "institute root."
                            ),
                            "required": True,
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
                        "tool": "shell",
                        "command": "Get-Location",
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

    def _is_unsafe_path(self, arg: str) -> bool:
        """True if a removal target is outside the institute root, or equals
        the institute root itself. Failures to resolve are treated as unsafe
        (fail safe).

        Absolute paths (e.g. ``C:\\Windows``) are resolved directly rather
        than being joined onto the root, so that a foreign absolute path is
        correctly identified as outside the root.
        """
        s = arg.strip()
        if len(s) >= 2 and s[0] in '\u0022\u0027' and s[-1] in '\u0022\u0027':
            s = s[1:-1]
        if not s:
            return True
        try:
            if Path(s).is_absolute():
                candidate = Path(s).resolve()
            else:
                candidate = (self.root / s).resolve()
        except (OSError, ValueError, RuntimeError):
            return True  # could not resolve -> refuse to allow removal
        if candidate == self.root:
            return True
        if self.root not in candidate.parents:
            return True
        return False

    def _check_dangerous(self, command: str):
        """Return a human-readable rejection reason, or None if allowed."""
        for pattern in _DANGEROUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return (
                    f"Rejected: command matches the dangerous pattern "
                    f"'{pattern}'. The shell tool blocks this class of command."
                )

        if _REMOVAL_PRESENT.search(command):
            for arg in self._removal_args(command):
                if self._is_unsafe_path(arg):
                    return (
                        f"Rejected: remove-item targets a path outside the "
                        f"institute root, or the institute root itself: '{arg}'"
                    )
        return None

    def _removal_args(self, command: str) -> list:
        """Collect the path-bearing arguments of every removal invocation in
        the command. Scanning the whole command (reset on ';') means a
        dangerous removal is caught even if it is not the first commandlet."""
        tokens = command.split()
        result = []
        in_removal = False
        i = 0
        n = len(tokens)
        while i < n:
            t = tokens[i]
            lt = t.lower()
            if lt in {";", ";"}:
                in_removal = False
                i += 1
                continue
            verb = lt.rstrip(":").replace(" ", "-")
            if verb in _REMOVAL_VERBS or verb == "remove-item":
                in_removal = True
                i += 1
                continue
            if in_removal:
                if lt in _RI_VALUE_FLAGS:
                    if i + 1 < n:
                        result.append(tokens[i + 1])
                        i += 2
                        continue
                if not t.startswith("-") and verb not in _REMOVAL_VERBS:
                    result.append(t)
            i += 1
        return result

    # --- process-tree termination ----------------------------------------
    @staticmethod
    def _kill_process_tree(proc: subprocess.Popen) -> None:
        """Best-effort kill of ``proc`` and every process it spawned.

        On Windows we use ``taskkill /F /T /PID`` which walks the kernel
        parent/child tree and terminates all descendants. On POSIX we fall
        back to ``os.killpg`` over the new process group (created via
        ``CREATE_NEW_PROCESS_GROUP`` / ``start_new_session`` semantics). In
        either case ``proc.kill()`` is called first as a backstop.
        """
        # Kill the direct child first (fast path, works everywhere).
        try:
            proc.kill()
        except Exception:
            pass

        pid = getattr(proc, "pid", None)
        if pid is None:
            return

        if os.name == "nt":
            # taskkill /F  -> force-terminate (TerminateProcess-equivalent)
            # taskkill /T  -> terminate the entire process *tree*
            # taskkill /PID-> target by PID
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True,
                    timeout=5,
                )
            except Exception:
                pass
            return

        # POSIX fallback: kill the whole process group (the child was
        # started in its own group via start_new_session-style flag below).
        try:
            os.killpg(os.getpgid(pid), 15)  # SIGTERM to the group
        except Exception:
            pass
        # Brief grace, then SIGKILL the group.
        try:
            proc.wait(timeout=1)
        except Exception:
            pass
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(pid), 9)  # SIGKILL to the group
            except Exception:
                pass

    @staticmethod
    def _collect_partial(proc: subprocess.Popen) -> dict:
        """Best-effort retrieval of whatever output was captured before the
        process was killed. Returns a dict with optional ``stdout`` / ``stderr``
        keys and the eventual ``return_code`` (negative if signalled/killed).
        """
        out: dict = {"return_code": None}
        try:
            stdout, stderr = proc.communicate(timeout=3)
            if stdout is not None:
                out["stdout"] = stdout
            if stderr is not None:
                out["stderr"] = stderr
        except Exception:
            # The process is already (or being) killed; fall back to whatever
            # is recorded via poll(). No exception may escape.
            pass
        rc = proc.poll()
        if rc is None:
            # Rare: process did not register as exited yet; force it.
            try:
                proc.kill()
            except Exception:
                pass
            try:
                rc = proc.wait(timeout=2)
            except Exception:
                rc = None
        if rc is not None:
            out["return_code"] = rc
        return out

    # --- execution -------------------------------------------------------
    def execute(self, command: str, working_directory: str = None, **_):
        if not isinstance(command, str) or not command.strip():
            return {
                "success": False,
                "command": command,
                "return_code": None,
                "error": "A non-empty 'command' string is required.",
            }

        command = command.strip()

        reason = self._check_dangerous(command)
        if reason:
            return {
                "success": False,
                "command": command,
                "return_code": None,
                "error": reason,
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

        return self._run(command, cwd)

    def _run(self, command: str, cwd: Path) -> dict:
        result = {
            "success": False,
            "command": command,
            "return_code": None,
        }

        cmd = [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy", "Bypass",
            "-Command", command,
        ]

        creation_flags = 0
        if os.name == "nt":
            # CREATE_NEW_PROCESS_GROUP: isolates the child so a Ctrl+C sent
            # to it cannot propagate back to the agent's console. It also
            # lets us kill the tree as a unit on POSIX-like fallbacks.
            creation_flags = 0x00000200  # CREATE_NEW_PROCESS_GROUP
        start_new_session = (os.name != "nt")

        proc = None
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=creation_flags,
                start_new_session=start_new_session,
            )
            stdout, stderr = proc.communicate(timeout=self.timeout)
            result["return_code"] = proc.returncode
            if stdout is not None:
                result["stdout"] = stdout
            if stderr is not None:
                result["stderr"] = stderr
            result["success"] = proc.returncode == 0
            return result

        except subprocess.TimeoutExpired:
            # The child (and any descendant it spawned) may still be alive.
            # Hard-kill the entire tree before returning so we do not block
            # the agent. This is the key fix for the winget-store-prompt
            # incident: a grandchild that ignores parent death must also die.
            partial = self._kill_and_reap(proc)
            result["return_code"] = partial.get("return_code")
            if "stdout" in partial:
                result["stdout"] = partial["stdout"]
            if "stderr" in partial:
                result["stderr"] = partial["stderr"]
            result["error"] = (
                f"Command exceeded the {self.timeout} second timeout; the "
                f"process and its child process tree were force-terminated."
            )
            return result

        except Exception as e:
            result["error"] = str(e)
            return result

    def _kill_and_reap(self, proc: subprocess.Popen) -> dict:
        """Force-terminate ``proc`` and its descendants, then collect any
        buffered output. Never raises."""
        if proc is None:
            return {"return_code": None}
        self._kill_process_tree(proc)
        return self._collect_partial(proc)
