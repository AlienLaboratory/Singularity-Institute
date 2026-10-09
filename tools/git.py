import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from tools.base import ToolBase

# --- Security policy -------------------------------------------------------
# Only these subcommands may ever be invoked. Everything else is rejected
# before a git process is spawned.
ALLOWED_SUBCOMMANDS = frozenset({
    "status", "diff", "log", "show", "rev-parse",
    "add", "commit", "branch", "checkout",
})

# Destructive / interactive flags that must never be passed to these
# subcommands, even though the subcommand itself is allowed.
_BRANCH_REJECT_FLAGS   = frozenset({"-D", "-f", "--force", "--delete", "-M"})
_ADD_REJECT_FLAGS      = frozenset({"-i", "--interactive", "-p", "--patch", "-e", "--edit"})
_COMMIT_REJECT_FLAGS   = frozenset({"-i", "--interactive", "-t", "--template", "--no-verify"})
_CHECKOUT_REJECT_FLAGS = frozenset({"-f", "--force"})


# --- Git executable discovery (unchanged from original) ---------------------
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


# --- Path containment --------------------------------------------------------
class _PathEscapeError(PermissionError):
    """Raised when a path argument would escape the institute root."""


# --- Tool --------------------------------------------------------------------
class GitTool(ToolBase):
    """Hardened Git tool for the institute.

    Security guarantees (enforced before any git process is spawned):

    1.  Subcommand allow-list: only the commands in :data:`ALLOWED_SUBCOMMANDS`
        are ever executed.  Everything else is rejected immediately.
    2.  Destructive flag rejection: ``branch -D/-f/--force/--delete/-M``,
        ``checkout -f/--force``, ``add -i/-p/-e``, ``commit -i/-t/--no-verify``
        are all rejected even though their parent subcommand is allowed.
    3.  ``add`` path containment: every non-flag argument to ``add`` is
        resolved relative to the working directory and must remain inside
        the institute root.  ``../`` escapes and absolute paths outside the
        root are rejected.
    4.  Global / system Git configuration is fully isolated: the subprocess
        environment points ``GIT_CONFIG_GLOBAL``, ``GIT_CONFIG_SYSTEM`` and
        ``GIT_CONFIG_PROFILE`` at a private config file and sets
        ``GIT_CONFIG_NOSYSTEM=1``.  The user's real ``~/.gitconfig`` is
        never read or written.
    5.  All inherited ``GIT_*`` environment variables are stripped to
        prevent configuration injection via ``GIT_CONFIG_COUNT``,
        ``GIT_EXTERNAL_DIFF``, ``GIT_DIR``, ``GIT_SSH``, etc.
    6.  Git hooks are disabled at command-line priority: ``core.hooksPath``
        is forced via ``GIT_CONFIG_COUNT``/``GIT_CONFIG_KEY_0``/
        ``GIT_CONFIG_VALUE_0`` to an empty directory, overriding any
        repository-local ``.git/config`` setting.
    7.  Working-directory containment: ``working_directory`` must stay
        inside the institute root.
    8.  Hard timeout: every subprocess is killed after ``self.timeout``
        seconds.
    9.  Structured results: every call returns a dict with ``success``,
        ``return_code``, ``stdout``, ``stderr`` and (on failure) ``error``.
        Failures are never raised.
    """

    @property
    def name(self) -> str:
        return "git"

    @property
    def description(self) -> str:
        return (
            "Run a safe, allow-listed subset of Git commands against the "
            "institute repository (status, diff, log, show, rev-parse, add, "
            "commit, branch, checkout).  Destructive flags, config changes, "
            "pushes, and all other subcommands are rejected before a git "
            "process is spawned.  Global/system Git configuration is fully "
            "isolated and Git hooks are disabled.  Working directories are "
            "constrained to the institute root.  Hard timeout enforced.  "
            "Failures are returned (not raised) as success=false."
        )

    def __init__(self, root: str, timeout: int = 60):
        self.root = Path(root).resolve()
        self.timeout = timeout
        self.git_exe = _resolve_git_executable()
        # Lazily-created private config + hooks directory.
        self._isolated_dir: Path | None = None

    # --- schema --------------------------------------------------------------
    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "operations": [
                {
                    "name": "execute",
                    "description": (
                        "Run an allow-listed git subcommand (status, diff, log, "
                        "show, rev-parse, add, commit, branch, checkout). "
                        "Destructive flags, config, push, and all other "
                        "subcommands are rejected before spawning. Returns "
                        "success, return_code, stdout, stderr, and the git "
                        "command that was run. Failures are returned (not "
                        "raised) as success=false."
                    ),
                    "parameters": [
                        {
                            "name": "command",
                            "type": "str",
                            "description": (
                                "One of: status, diff, log, show, rev-parse, "
                                "add, commit, branch, checkout.  Any other "
                                "value is rejected."
                            ),
                            "required": True,
                        },
                        {
                            "name": "args",
                            "type": "list[str]",
                            "description": (
                                "Optional list of additional arguments for the "
                                "git subcommand (e.g. ['-m', 'Initial commit'] "
                                "for commit, ['.'] for add).  Destructive "
                                "flags such as -f, --force, -D, --delete, "
                                "-i, --interactive, -p, --patch, -e, --edit, "
                                "-t, --template, -M, --no-verify are rejected."
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

    # --- helpers -------------------------------------------------------------
    def _safe_path(self, path: str) -> Path:
        """Resolve *path* relative to the institute root and confirm the
        result stays inside it.  Raises :class:`PermissionError` otherwise."""
        candidate = (self.root / path).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise PermissionError(
                f"Working directory outside institute root denied: {path}"
            )
        return candidate

    def _clean_args(self, args) -> list[str]:
        if args is None:
            return []
        if isinstance(args, str):
            return args.split()
        if not isinstance(args, (list, tuple)):
            raise TypeError("'args' must be a list of strings when provided.")
        return [str(a) for a in args]

    # --- security: subcommand / flag gate ------------------------------------
    _FLAG_POLICY = {
        "branch":   _BRANCH_REJECT_FLAGS,
        "add":      _ADD_REJECT_FLAGS,
        "commit":   _COMMIT_REJECT_FLAGS,
        "checkout": _CHECKOUT_REJECT_FLAGS,
    }

    def _validate(self, command: str, args: list[str]) -> None:
        """Raise :class:`PermissionError` if the command or any argument is
        not permitted.  Called *before* any subprocess is spawned."""
        # 1. subcommand allow-list
        if command not in ALLOWED_SUBCOMMANDS:
            raise PermissionError(
                f"Git subcommand '{command}' is not allowed. "
                f"Allowed: {sorted(ALLOWED_SUBCOMMANDS)}"
            )

        # 2. destructive / interactive flag rejection
        reject = self._FLAG_POLICY.get(command, frozenset())
        for a in args:
            if a in reject:
                raise PermissionError(
                    f"Flag '{a}' is rejected for git {command}."
                )

        # 3. add path containment
        if command == "add":
            for a in args:
                if a.startswith("-"):
                    continue  # flag, not a path
                self._assert_path_inside_root(a)

    def _assert_path_inside_root(self, path_str: str) -> None:
        """Ensure a path argument to ``git add`` stays inside the root."""
        p = Path(path_str)
        if p.is_absolute():
            candidate = p.resolve()
        else:
            candidate = (self.root / p).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise _PathEscapeError(
                f"Path '{path_str}' escapes the institute root; rejected."
            )

    # --- security: isolated environment --------------------------------------
    def _ensure_isolated_dir(self) -> Path:
        """Lazily create a private directory containing:

        * an empty ``hooks/`` sub-directory (no scripts = no hooks run)
        * a ``gitconfig`` file that sets ``core.hooksPath`` to that empty
          directory and provides a neutral committer identity.

        This directory is used for ``GIT_CONFIG_GLOBAL`` and
        ``GIT_CONFIG_SYSTEM`` so the user's real global and system config
        files are never consulted.

        NOTE: paths in the gitconfig value are written with forward slashes
        (POSIX form) because gitconfig treats backslash as an escape
        character; a raw Windows backslash path would be an invalid line.
        """
        if self._isolated_dir is not None:
            return self._isolated_dir

        d = Path(tempfile.mkdtemp(prefix="institute_git_isolated_"))
        hooks = d / "hooks"
        hooks.mkdir(parents=True)

        hooks_posix = hooks.as_posix()
        gc = d / "gitconfig"
        gc.write_text(
            "[core]\n"
            f"\thooksPath = {hooks_posix}\n"
            "[user]\n"
            "\tname = Institute Autonomous\n"
            "\temail = autonomous@institute.local\n",
            encoding="utf-8",
        )
        self._isolated_dir = d
        return d

    def _isolated_env(self) -> dict:
        """Build the subprocess environment that fully isolates Git from
        user/global/system configuration, repository-local hooks, and
        external command injection.

        Security measures:
        - All inherited ``GIT_*`` environment variables are stripped to
          prevent configuration injection via ``GIT_CONFIG_COUNT``,
          ``GIT_CONFIG_KEY_*``, ``GIT_CONFIG_VALUE_*``,
          ``GIT_CONFIG_PARAMETERS``, ``GIT_EXTERNAL_DIFF``, ``GIT_DIR``,
          ``GIT_WORK_TREE``, ``GIT_SSH``, ``GIT_ASKPASS``, ``GIT_EDITOR``,
          ``GIT_SEQUENCE_EDITOR``, ``GIT_PAGER``, ``GIT_TRACE``, and
          other mechanism that could defeat the security policy.
        - Private config is used for global, system, and profile sources.
        - ``core.hooksPath`` is forced at command-line priority (highest)
          via ``GIT_CONFIG_COUNT``/``GIT_CONFIG_KEY_0``/``GIT_CONFIG_VALUE_0``
          to an empty hooks directory, overriding any repository-local
          ``.git/config`` setting.
        - ``GIT_EXTERNAL_DIFF`` and other external-command mechanisms are
          stripped so no arbitrary program can be launched through git.
        """
        d = self._ensure_isolated_dir()
        gc = str(d / "gitconfig")
        hooks = d / "hooks"

        # Strip ALL Git-related environment variables to prevent inherited
        # configuration injection. This removes GIT_CONFIG_COUNT,
        # GIT_CONFIG_KEY_*, GIT_CONFIG_VALUE_*, GIT_CONFIG_PARAMETERS,
        # GIT_EXTERNAL_DIFF, GIT_DIR, GIT_WORK_TREE, GIT_INDEX_FILE,
        # GIT_SSH, GIT_SSH_COMMAND, GIT_ASKPASS, GIT_EDITOR,
        # GIT_SEQUENCE_EDITOR, GIT_PAGER, GIT_TRACE, GIT_TRACE2,
        # GIT_ALTERNATE_OBJECT_DIRECTORIES, and any other GIT_* variable
        # that could be used to inject configuration or external commands.
        env = {}
        for k, v in os.environ.items():
            if k.startswith("GIT_"):
                continue
            env[k] = v

        # Point all config sources at the private config file (never the
        # user's real one).
        env["GIT_CONFIG_GLOBAL"]   = gc
        env["GIT_CONFIG_SYSTEM"]   = gc
        env["GIT_CONFIG_PROFILE"]  = gc
        env["GIT_CONFIG_NOSYSTEM"] = "1"

        # Force core.hooksPath at command-line level (highest priority in
        # Git's config hierarchy: command-line > environment > local >
        # global > system). This overrides any repository-local .git/config
        # setting of core.hooksPath, ensuring no repository-controlled hook
        # can execute.
        env["GIT_CONFIG_COUNT"]   = "1"
        env["GIT_CONFIG_KEY_0"]   = "core.hooksPath"
        env["GIT_CONFIG_VALUE_0"] = hooks.as_posix()

        return env

    # --- execution -----------------------------------------------------------
    def execute(self, command: str, args=None, working_directory: str = None, **_):
        # 0. type guard
        if not isinstance(command, str) or not command.strip():
            return {
                "success": False,
                "command": repr(command),
                "return_code": None,
                "error": "A non-empty 'command' string is required.",
            }
        command = command.strip().lower()

        # 1. argument normalisation
        try:
            extra_args = self._clean_args(args)
        except TypeError as e:
            return {
                "success": False,
                "command": command,
                "return_code": None,
                "error": str(e),
            }

        # 2. security gate (allow-list + flag + path) — BEFORE spawning
        try:
            self._validate(command, extra_args)
        except PermissionError as e:
            return {
                "success": False,
                "command": f"git {command} {' '.join(extra_args)}".strip(),
                "return_code": None,
                "error": f"Rejected: {e}",
            }

        # 3. working-directory containment
        try:
            if working_directory:
                cwd = self._safe_path(working_directory)
            else:
                cwd = self.root
        except PermissionError as e:
            return {
                "success": False,
                "command": f"git {command} {' '.join(extra_args)}".strip(),
                "return_code": None,
                "error": str(e),
            }

        return self._run([command, *extra_args], cwd)

    def _run(self, argv: list[str], cwd: Path) -> dict:
        result: dict = {
            "success": False,
            "command": "git " + " ".join(argv),
            "return_code": None,
        }

        cmd = [self.git_exe, *argv]
        env = self._isolated_env()

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
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
