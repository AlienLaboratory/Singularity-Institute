"""
Hardened Git tool tests.

Run from the institute root:
    python tests/test_git.py

Covers:
    - tool identity / ToolBase conformance
    - git executable resolution (PATH or well-known Windows path)
    - positive: init, status, add, commit, log, diff, branch, checkout,
      rev-parse, working-directory behavior
    - negative: subcommand allow-list (reset/clean/push/remote/config/
      rebase/gc/stash), destructive flags (branch -D, checkout --force),
      add path-escape rejection
    - global-config isolation: a `config --global` style attempt cannot
      modify the user's real global Git configuration
    - adversarial: repository-local hooks override attempt
    - adversarial: environment-injected Git configuration
    - adversarial: GIT_EXTERNAL_DIFF external program execution
    - adversarial: commit --no-verify rejection
    - failure handling (returned, not raised)
    - registry / schema integration (auto documentation)
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.git import GitTool, _resolve_git_executable, ALLOWED_SUBCOMMANDS
from tools.base import ToolBase
from tools.registry import ToolRegistry
from core.agent import build_system_prompt


def _norm(p: str) -> str:
    """Normalize a path for comparison (case-insensitive on Windows, and
    unified slashes so git's forward-slash output matches os.path form)."""
    return os.path.normcase(p.replace("/", os.sep))


def _git_available() -> bool:
    """True if a git executable can be resolved on this machine."""
    exe = _resolve_git_executable()
    try:
        if os.path.isfile(exe):
            return True
    except Exception:
        pass
    return shutil.which("git") is not None


GIT_AVAILABLE = _git_available()


def _init_repo(root: str) -> bool:
    """Initialize a git repo directly (bypassing the hardened tool, which
    intentionally excludes `init`). Used only by the test harness."""
    try:
        git = _resolve_git_executable()
        env = dict(os.environ)
        env["GIT_CONFIG_GLOBAL"] = ""
        env["GIT_CONFIG_NOSYSTEM"] = "1"
        p = subprocess.run(
            [git, "init", "-q", root],
            capture_output=True, text=True, timeout=15, env=env
        )
        return p.returncode == 0
    except Exception:
        return False


class GitToolTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.tool = GitTool(self.root, timeout=10)

    def tearDown(self):
        self.tmp.cleanup()

    def _init_repo(self):
        r = _init_repo(self.root)
        self.assertTrue(r)

    # ------------------------------------------------------------------
    def test_is_a_tool_base(self):
        self.assertIsInstance(self.tool, ToolBase)
        self.assertEqual(self.tool.name, "git")

    def test_allowed_subcommand_set(self):
        for c in ["status", "diff", "log", "show", "rev-parse",
                  "add", "commit", "branch", "checkout"]:
            self.assertIn(c, ALLOWED_SUBCOMMANDS)

    # ------------------------------------------------------------------
    def test_git_executable_resolves(self):
        self.assertTrue(GIT_AVAILABLE)
        self.assertTrue(self.tool.git_exe)

    # ==================================================================
    # Positive tests
    # ==================================================================
    @unittest.skipUnless(GIT_AVAILABLE, "git not available on this machine")
    def test_status(self):
        self._init_repo()
        status = self.tool.execute("status")
        self.assertTrue(status["success"], status)

    @unittest.skipUnless(GIT_AVAILABLE, "git not available on this machine")
    def test_add_commit_log(self):
        self._init_repo()
        with open(os.path.join(self.root, "hello.txt"), "w") as f:
            f.write("hello\n")
        add = self.tool.execute("add", args=["."])
        self.assertTrue(add["success"], add)
        commit = self.tool.execute("commit", args=["-m", "Initial commit"])
        self.assertTrue(commit["success"], commit)
        log = self.tool.execute("log", args=["--oneline"])
        self.assertTrue(log["success"], log)
        self.assertIn("Initial commit", log["stdout"])

    @unittest.skipUnless(GIT_AVAILABLE, "git not available on this machine")
    def test_diff_detects_change(self):
        self._init_repo()
        p = os.path.join(self.root, "f.txt")
        with open(p, "w") as f:
            f.write("v1\n")
        self.tool.execute("add", args=["."])
        self.tool.execute("commit", args=["-m", "c1"])
        with open(p, "w") as f:
            f.write("v2\n")
        diff = self.tool.execute("diff")
        self.assertTrue(diff["success"], diff)
        self.assertIn("v1", diff["stdout"])
        self.assertIn("v2", diff["stdout"])

    @unittest.skipUnless(GIT_AVAILABLE, "git not available on this machine")
    def test_branch_creation(self):
        self._init_repo()
        with open(os.path.join(self.root, "b.txt"), "w") as f:
            f.write("x\n")
        self.tool.execute("add", args=["."])
        self.tool.execute("commit", args=["-m", "c1"])
        b = self.tool.execute("branch", args=["feature"],
                              working_directory=None)
        self.assertTrue(b["success"], b)
        ls = self.tool.execute("branch", args=["-a"])
        self.assertIn("feature", ls["stdout"]) if ls["success"] else self.fail(ls)

    @unittest.skipUnless(GIT_AVAILABLE, "git not available on this machine")
    def test_checkout_branch(self):
        self._init_repo()
        with open(os.path.join(self.root, "b.txt"), "w") as f:
            f.write("x\n")
        self.tool.execute("add", args=["."])
        self.tool.execute("commit", args=["-m", "c1"])
        self.tool.execute("branch", args=["work"])
        co = self.tool.execute("checkout", args=["-b", "work2"])
        self.assertTrue(co["success"], co)

    @unittest.skipUnless(GIT_AVAILABLE, "git not available on this machine")
    def test_default_working_directory_is_institute_root(self):
        self._init_repo()
        result = self.tool.execute("rev-parse", args=["--show-toplevel"])
        self.assertTrue(result["success"], result)
        self.assertEqual(_norm(result["stdout"].strip()), _norm(self.root))

    @unittest.skipUnless(GIT_AVAILABLE, "git not available on this machine")
    def test_explicit_working_directory_inside_root(self):
        self._init_repo()
        sub = os.path.join(self.root, "sub")
        os.makedirs(sub)
        result = self.tool.execute(
            "rev-parse", args=["--show-toplevel"], working_directory="sub"
        )
        self.assertTrue(result["success"], result)
        self.assertEqual(_norm(result["stdout"].strip()), _norm(self.root))

    @unittest.skipUnless(GIT_AVAILABLE, "git not available on this machine")
    def test_working_directory_outside_root_is_rejected(self):
        result = self.tool.execute("status", working_directory="..")
        self.assertFalse(result["success"])
        self.assertIn("denied", result["error"].lower())
        self.assertIsNone(result["return_code"])

    # ==================================================================
    # Negative tests: subcommand allow-list
    # ==================================================================
    def _assert_rejected(self, command, args=None):
        result = self.tool.execute(command, args=args)
        self.assertIsInstance(result, dict)
        self.assertFalse(result["success"], f"expected rejection for {command}")
        self.assertIn("Rejected", result["error"]) if "error" in result \
            else self.assertIn("not allowed", result.get("error", ""))
        self.assertIsNone(result["return_code"])
        self.assertNotIn("stdout", result)
        return result

    def test_rejected_reset(self):
        self._assert_rejected("reset")

    def test_rejected_clean(self):
        self._assert_rejected("clean")

    def test_rejected_push(self):
        self._assert_rejected("push")

    def test_rejected_remote(self):
        self._assert_rejected("remote")

    def test_rejected_config(self):
        self._assert_rejected("config")

    def test_rejected_rebase(self):
        self._assert_rejected("rebase")

    def test_rejected_gc(self):
        self._assert_rejected("gc")

    def test_rejected_stash(self):
        self._assert_rejected("stash")

    def test_rejected_init(self):
        # init is not in the allow-list for the hardened tool
        self._assert_rejected("init")

    # ==================================================================
    # Negative tests: destructive flags
    # ==================================================================
    def test_rejected_branch_D(self):
        self._assert_rejected("branch", args=["-D", "main"])

    def test_rejected_branch_force(self):
        self._assert_rejected("branch", args=["--force", "main"])

    def test_rejected_branch_delete(self):
        self._assert_rejected("branch", args=["--delete", "-f", "main"])

    def test_rejected_checkout_force(self):
        self._assert_rejected("checkout", args=["--force", "main"])

    def test_rejected_checkout_f(self):
        self._assert_rejected("checkout", args=["-f", "main"])

    def test_rejected_add_interactive(self):
        self._assert_rejected("add", args=["-i", "."])

    def test_rejected_commit_interactive(self):
        self._assert_rejected("commit", args=["-i"])

    def test_rejected_commit_no_verify(self):
        """commit --no-verify must be rejected as it would bypass hook checks."""
        self._assert_rejected("commit", args=["--no-verify", "-m", "msg"])

    # ==================================================================
    # Negative tests: add path escape
    # ==================================================================
    def test_add_path_escape_rejected(self):
        self._assert_rejected("add", args=["../secret.txt"])

    def test_add_path_escape_double_rejected(self):
        self._assert_rejected("add", args=["../../etc/passwd"])

    @unittest.skipUnless(GIT_AVAILABLE, "git not available on this machine")
    def test_add_path_escape_outside_root_absolute(self):
        outside = os.path.abspath(os.path.join(self.root, ".."))
        result = self.tool.execute("add", args=[outside + "\\x.txt"])
        self.assertFalse(result["success"])

    # ==================================================================
    # Global config isolation
    # ==================================================================
    @unittest.skipUnless(GIT_AVAILABLE, "git not available on this machine")
    def test_global_config_attempt_cannot_modify_real_global_config(self):
        """A `config` subcommand attempt is rejected (not spawned), so the
        user's real global Git configuration can never be read or written by
        the tool.  We also prove that even if a git process WERE to run, it
        is pointed at a private config file, not the user's real one."""
        # 1) `config` is rejected outright -> no subprocess, no global change.
        result = self.tool.execute("config", args=["--global", "user.email", "evil@x"])
        self.assertFalse(result["success"])
        self.assertIn("Rejected", result["error"])
        self.assertIsNone(result["return_code"])

        # 2) The isolated environment never points at the real user config.
        env = self.tool._isolated_env()
        real_global = os.path.expanduser("~/.gitconfig")
        self.assertNotEqual(os.path.normcase(env["GIT_CONFIG_GLOBAL"]),
                            os.path.normcase(real_global))
        self.assertNotEqual(os.path.normcase(env["GIT_CONFIG_SYSTEM"]),
                            os.path.normcase(real_global))
        self.assertEqual(env["GIT_CONFIG_NOSYSTEM"], "1")

    # ------------------------------------------------------------------
    def test_empty_command_rejected(self):
        result = self.tool.execute("")
        self.assertFalse(result["success"])
        self.assertIn("required", result["error"].lower())

    def test_non_string_command_rejected(self):
        result = self.tool.execute(None)
        self.assertFalse(result["success"])
        self.assertIn("required", result["error"].lower())

    # ------------------------------------------------------------------
    def test_string_args_split(self):
        self.assertEqual(self.tool._clean_args("a b c"), ["a", "b", "c"])
        self.assertEqual(self.tool._clean_args(["a", "b"]), ["a", "b"])
        self.assertEqual(self.tool._clean_args(None), [])

    # ------------------------------------------------------------------
    def test_schema_and_registry_integration(self):
        registry = ToolRegistry()
        registry.register(self.tool)

        self.assertIsNotNone(registry.get("git"))
        self.assertEqual(registry.get_tool_schema("git")["name"], "git")

        docs = registry.generate_tool_docs()
        self.assertIn("git", docs)

        prompt = build_system_prompt(registry)
        self.assertIn("git", prompt)
        self.assertIn("allow-listed subset of Git commands", prompt)


# ============================================================================
# Adversarial security tests
# ============================================================================

class AdversarialHooksTest(unittest.TestCase):
    """Test that repository-local hooks configuration cannot execute
    malicious hook scripts during commit operations.

    Attack scenario: A repository's .git/config sets core.hooksPath to a
    directory containing a malicious pre-commit hook. The hardened tool must
    override this via command-line priority (GIT_CONFIG_COUNT mechanism)
    so the hook never executes.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.tool = GitTool(self.root, timeout=10)

    def tearDown(self):
        self.tmp.cleanup()

    @unittest.skipUnless(GIT_AVAILABLE, "git not available on this machine")
    def test_repo_local_hooks_cannot_execute(self):
        """A repository-local core.hooksPath pointing to a malicious hook
        directory must not cause that hook to run during commit."""
        _init_repo(self.root)

        # Create a malicious hooks directory outside the repo root.
        malicious_hooks_dir = os.path.join(self.tmp.name, "evil_hooks")
        os.makedirs(malicious_hooks_dir)

        # Create a marker file that the hook would create if executed.
        marker = os.path.join(self.tmp.name, "hook_executed.marker")

        # Write a malicious pre-commit hook that creates the marker file.
        hook_path = os.path.join(malicious_hooks_dir, "pre-commit")
        hook_content = f'#!/bin/sh\ntouch "{marker}"\n'
        with open(hook_path, "w", newline="\n") as f:
            f.write(hook_content)
        os.chmod(hook_path, 0o755)

        # Set the repository-local core.hooksPath to the malicious directory.
        # This simulates a compromised/malicious repository.
        git = _resolve_git_executable()
        env = dict(os.environ)
        env["GIT_CONFIG_GLOBAL"] = ""
        env["GIT_CONFIG_NOSYSTEM"] = "1"
        subprocess.run(
            [git, "config", "core.hooksPath", malicious_hooks_dir],
            cwd=self.root, capture_output=True, text=True, timeout=10, env=env
        )

        # Verify the local config was set (sanity check).
        cfg_check = subprocess.run(
            [git, "config", "--local", "--get", "core.hooksPath"],
            cwd=self.root, capture_output=True, text=True, timeout=10, env=env
        )
        self.assertEqual(cfg_check.stdout.strip(), malicious_hooks_dir)

        # Now perform a commit through the hardened tool.
        with open(os.path.join(self.root, "test.txt"), "w") as f:
            f.write("test content\n")
        self.tool.execute("add", args=["."])
        commit_result = self.tool.execute("commit", args=["-m", "test commit"])

        # The commit should succeed (hooks are disabled, so no failure).
        self.assertTrue(commit_result["success"], commit_result)

        # CRITICAL ASSERTION: The malicious hook must NOT have executed.
        self.assertFalse(
            os.path.exists(marker),
            f"SECURITY FAILURE: Repository-local hook executed! "
            f"Marker file was created: {marker}"
        )

    @unittest.skipUnless(GIT_AVAILABLE, "git not available on this machine")
    def test_repo_local_hooks_path_in_gitconfig(self):
        """Alternative attack: set hooksPath via .git/config file directly.
        Even with direct file manipulation, the command-line priority
        override must prevent hook execution."""
        _init_repo(self.root)

        # Create malicious hooks dir.
        malicious_dir = os.path.join(self.tmp.name, "malicious_hooks")
        os.makedirs(malicious_dir)

        marker = os.path.join(self.tmp.name, "direct_config_hook.marker")
        hook_path = os.path.join(malicious_dir, "pre-commit")
        with open(hook_path, "w", newline="\n") as f:
            f.write(f'#!/bin/sh\ntouch "{marker}"\n')
        os.chmod(hook_path, 0o755)

        # Directly modify .git/config to set hooksPath under [core].
        # NOTE: gitconfig interprets backslash as an escape character, so
        # we must use forward-slash (POSIX) form for the path value.
        gitconfig_path = os.path.join(self.root, ".git", "config")
        with open(gitconfig_path, "a") as f:
            f.write(f"\n\thooksPath = {malicious_dir.replace(chr(92), '/')}\n")

        # Perform commit through the hardened tool.
        with open(os.path.join(self.root, "data.txt"), "w") as f:
            f.write("data\n")
        self.tool.execute("add", args=["."])
        commit_result = self.tool.execute("commit", args=["-m", "safe commit"])

        self.assertTrue(commit_result["success"], commit_result)

        # The hook must NOT have executed.
        self.assertFalse(
            os.path.exists(marker),
            f"SECURITY FAILURE: Direct .git/config hooksPath override! "
            f"Marker: {marker}"
        )


class AdversarialEnvInjectionTest(unittest.TestCase):
    """Test that environment-injected Git configuration variables cannot
    override the isolated configuration or cause external command execution.

    Attack scenarios:
    - GIT_CONFIG_COUNT/KEY_0/VALUE_0 injection to override hooksPath
    - GIT_EXTERNAL_DIFF to launch arbitrary programs
    - GIT_CONFIG_PARAMETERS to inject configuration
    - GIT_DIR redirection
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.tool = GitTool(self.root, timeout=10)

    def tearDown(self):
        self.tmp.cleanup()

    @unittest.skipUnless(GIT_AVAILABLE, "git not available on this machine")
    def test_git_config_count_injection_blocked(self):
        """GIT_CONFIG_COUNT/KEY_0/VALUE_0 in the inherited environment
        must not be able to override the tool's hooksPath setting.
        The tool strips all GIT_* variables before passing to subprocess."""
        _init_repo(self.root)

        # Create a malicious hooks dir and hook.
        malicious_hooks = os.path.join(self.tmp.name, "injected_hooks")
        os.makedirs(malicious_hooks)
        marker = os.path.join(self.tmp.name, "env_inject_hook.marker")

        hook_path = os.path.join(malicious_hooks, "pre-commit")
        with open(hook_path, "w", newline="\n") as f:
            f.write(f'#!/bin/sh\ntouch "{marker}"\n')
        os.chmod(hook_path, 0o755)

        # Inject malicious GIT_CONFIG_* variables into os.environ
        # (simulating a hostile parent process environment).
        saved = {}
        injected_vars = {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "core.hooksPath",
            "GIT_CONFIG_VALUE_0": malicious_hooks,
        }
        for k, v in injected_vars.items():
            saved[k] = os.environ.get(k)
            os.environ[k] = v

        try:
            # Reset the tool's cached isolated dir so it rebuilds fresh.
            self.tool._isolated_dir = None

            with open(os.path.join(self.root, "injection.txt"), "w") as f:
                f.write("injection test\n")
            self.tool.execute("add", args=["."])
            commit_result = self.tool.execute("commit", args=["-m", "injection test"])

            self.assertTrue(commit_result["success"], commit_result)

            # The malicious hook must NOT have executed.
            self.assertFalse(
                os.path.exists(marker),
                f"SECURITY FAILURE: GIT_CONFIG_COUNT env injection succeeded! "
                f"Marker: {marker}"
            )
        finally:
            # Restore environment.
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    @unittest.skipUnless(GIT_AVAILABLE, "git not available on this machine")
    def test_git_external_diff_blocked(self):
        """GIT_EXTERNAL_DIFF in the inherited environment must not cause
        an external program to execute during a git diff operation."""
        _init_repo(self.root)

        # Create an 'external diff' script that creates a marker file.
        marker = os.path.join(self.tmp.name, "external_diff_executed.marker")
        diff_script = os.path.join(self.tmp.name, "evil_diff.sh")
        with open(diff_script, "w", newline="\n") as f:
            f.write(f'#!/bin/sh\ntouch "{marker}"\nexit 0\n')
        os.chmod(diff_script, 0o755)

        # Inject GIT_EXTERNAL_DIFF into the environment.
        saved = os.environ.get("GIT_EXTERNAL_DIFF")
        os.environ["GIT_EXTERNAL_DIFF"] = diff_script

        try:
            self.tool._isolated_dir = None

            # Create a file, commit, modify, then diff.
            with open(os.path.join(self.root, "diff_test.txt"), "w") as f:
                f.write("v1\n")
            self.tool.execute("add", args=["."])
            self.tool.execute("commit", args=["-m", "c1"])

            with open(os.path.join(self.root, "diff_test.txt"), "w") as f:
                f.write("v2\n")

            diff_result = self.tool.execute("diff")

            # The diff should either succeed with normal output or fail
            # gracefully. Either way, the external script must NOT run.
            self.assertFalse(
                os.path.exists(marker),
                f"SECURITY FAILURE: GIT_EXTERNAL_DIFF executed! "
                f"Marker: {marker}"
            )
        finally:
            if saved is None:
                os.environ.pop("GIT_EXTERNAL_DIFF", None)
            else:
                os.environ["GIT_EXTERNAL_DIFF"] = saved

    @unittest.skipUnless(GIT_AVAILABLE, "git not available on this machine")
    def test_git_config_parameters_injection_blocked(self):
        """GIT_CONFIG_PARAMETERS in the inherited environment must not
        inject configuration that overrides hooks isolation."""
        _init_repo(self.root)

        malicious_hooks = os.path.join(self.tmp.name, "param_hooks")
        os.makedirs(malicious_hooks)
        marker = os.path.join(self.tmp.name, "param_inject_hook.marker")

        hook_path = os.path.join(malicious_hooks, "pre-commit")
        with open(hook_path, "w", newline="\n") as f:
            f.write(f'#!/bin/sh\ntouch "{marker}"\n')
        os.chmod(hook_path, 0o755)

        # Inject GIT_CONFIG_PARAMETERS with malicious hooksPath.
        saved = os.environ.get("GIT_CONFIG_PARAMETERS")
        os.environ["GIT_CONFIG_PARAMETERS"] = f"--global core.hooksPath={malicious_hooks}"

        try:
            self.tool._isolated_dir = None

            with open(os.path.join(self.root, "param_test.txt"), "w") as f:
                f.write("param test\n")
            self.tool.execute("add", args=["."])
            commit_result = self.tool.execute("commit", args=["-m", "param test"])

            self.assertTrue(commit_result["success"], commit_result)

            self.assertFalse(
                os.path.exists(marker),
                f"SECURITY FAILURE: GIT_CONFIG_PARAMETERS injection succeeded! "
                f"Marker: {marker}"
            )
        finally:
            if saved is None:
                os.environ.pop("GIT_CONFIG_PARAMETERS", None)
            else:
                os.environ["GIT_CONFIG_PARAMETERS"] = saved

    @unittest.skipUnless(GIT_AVAILABLE, "git not available on this machine")
    def test_git_dir_redirection_blocked(self):
        """GIT_DIR in the inherited environment must not redirect git to
        a different repository. The tool strips GIT_DIR before spawning."""
        _init_repo(self.root)

        # Create a decoy repo.
        decoy_root = os.path.join(self.tmp.name, "decoy_repo")
        os.makedirs(decoy_root)
        git = _resolve_git_executable()
        env_init = dict(os.environ)
        env_init["GIT_CONFIG_GLOBAL"] = ""
        env_init["GIT_CONFIG_NOSYSTEM"] = "1"
        subprocess.run([git, "init", "-q", decoy_root],
                       capture_output=True, text=True, timeout=10, env=env_init)

        # Inject GIT_DIR pointing to the decoy.
        saved = os.environ.get("GIT_DIR")
        os.environ["GIT_DIR"] = os.path.join(decoy_root, ".git")

        try:
            self.tool._isolated_dir = None

            # status should still operate on the real repo, not the decoy.
            result = self.tool.execute("status")

            # If GIT_DIR leaked through, the repo root would be the decoy.
            # The tool should either reject or operate on the correct repo.
            if result["success"]:
                # Check via rev-parse that we're in the right repo.
                toplevel = self.tool.execute("rev-parse", args=["--show-toplevel"])
                if toplevel["success"]:
                    self.assertEqual(
                        _norm(toplevel["stdout"].strip()),
                        _norm(self.root),
                        f"SECURITY FAILURE: GIT_DIR redirected to decoy repo. "
                        f"Got: {toplevel['stdout'].strip()}, expected: {self.root}"
                    )
        finally:
            if saved is None:
                os.environ.pop("GIT_DIR", None)
            else:
                os.environ["GIT_DIR"] = saved


class AdversarialCommitSafetyTest(unittest.TestCase):
    """Test commit safety: multiple independent controls must prevent
    hook execution and configuration bypass."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.tool = GitTool(self.root, timeout=10)

    def tearDown(self):
        self.tmp.cleanup()

    @unittest.skipUnless(GIT_AVAILABLE, "git not available on this machine")
    def test_no_verify_flag_rejected(self):
        """commit --no-verify must be rejected as it would bypass all
        hook verification. This is a defense-in-depth control on top of
        the hooksPath override."""
        result = self.tool.execute("commit", args=["--no-verify", "-m", "msg"])
        self.assertFalse(result["success"])
        self.assertIn("Rejected", result.get("error", ""))
        self.assertIsNone(result["return_code"])

    def test_isolated_env_strips_all_git_variables(self):
        """The isolated environment must not contain any inherited GIT_*
        variables that could carry malicious configuration."""
        # Inject a GIT_* variable to verify it gets stripped.
        saved = os.environ.get("GIT_TEST_MARKER_VAR")
        os.environ["GIT_TEST_MARKER_VAR"] = "should_not_leak"

        try:
            self.tool._isolated_dir = None
            env = self.tool._isolated_env()

            # The injected variable must NOT appear in the child env.
            self.assertNotIn("GIT_TEST_MARKER_VAR", env)

            # But our own security variables must be present.
            self.assertIn("GIT_CONFIG_GLOBAL", env)
            self.assertIn("GIT_CONFIG_NOSYSTEM", env)
            self.assertIn("GIT_CONFIG_COUNT", env)
            self.assertIn("GIT_CONFIG_KEY_0", env)
            self.assertIn("GIT_CONFIG_VALUE_0", env)
        finally:
            if saved is None:
                os.environ.pop("GIT_TEST_MARKER_VAR", None)
            else:
                os.environ["GIT_TEST_MARKER_VAR"] = saved

    def test_isolated_env_has_no_external_diff(self):
        """The isolated environment must not contain GIT_EXTERNAL_DIFF."""
        saved = os.environ.get("GIT_EXTERNAL_DIFF")
        os.environ["GIT_EXTERNAL_DIFF"] = "/usr/bin/evil"

        try:
            self.tool._isolated_dir = None
            env = self.tool._isolated_env()
            self.assertNotIn("GIT_EXTERNAL_DIFF", env)
        finally:
            if saved is None:
                os.environ.pop("GIT_EXTERNAL_DIFF", None)
            else:
                os.environ["GIT_EXTERNAL_DIFF"] = saved


if __name__ == "__main__":
    unittest.main(verbosity=2)
