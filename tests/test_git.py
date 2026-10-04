"""
Focused tests for the Git tool.

Run from the institute root:
    python tests/test_git.py

Covers:
    - tool identity / ToolBase conformance
    - git executable resolution (PATH or well-known Windows path)
    - version, status, init, add, commit, log, diff
    - working-directory behavior (inside vs outside the institute root)
    - failure handling (returned, not raised)
    - registry / schema integration (auto documentation)
"""

import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.git import GitTool, _resolve_git_executable
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


class GitToolTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.tool = GitTool(self.root, timeout=10)

    def tearDown(self):
        self.tmp.cleanup()

    def _init_repo(self):
        self.tool.execute("init")
        self.tool.execute("config", args=["user.email", "institute@local"])
        self.tool.execute("config", args=["user.name", "Institute"])

    # ------------------------------------------------------------------
    def test_is_a_tool_base(self):
        self.assertIsInstance(self.tool, ToolBase)
        self.assertEqual(self.tool.name, "git")

    # ------------------------------------------------------------------
    def test_git_executable_resolves(self):
        self.assertTrue(GIT_AVAILABLE)
        self.assertTrue(self.tool.git_exe)

    @unittest.skipUnless(GIT_AVAILABLE, "git not available on this machine")
    def test_version(self):
        result = self.tool.execute("--version")
        self.assertTrue(result["success"], result)
        self.assertEqual(result["return_code"], 0)
        self.assertIn("git version", result["stdout"].lower())

    @unittest.skipUnless(GIT_AVAILABLE, "git not available on this machine")
    def test_init_and_status(self):
        init = self.tool.execute("init")
        self.assertTrue(init["success"], init)

        status = self.tool.execute("status")
        self.assertTrue(status["success"], status)
        # Empty repository: nothing is tracked yet.
        self.assertIn("nothing to commit", status["stdout"].lower())

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

        status = self.tool.execute("status")
        self.assertIn("nothing to commit", status["stdout"].lower())

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

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    @unittest.skipUnless(GIT_AVAILABLE, "git not available on this machine")
    def test_invalid_subcommand_is_returned_not_raised(self):
        result = self.tool.execute("definitely-not-a-real-command")
        self.assertIsInstance(result, dict)
        self.assertFalse(result["success"])
        self.assertNotEqual(result["return_code"], 0)

    # ------------------------------------------------------------------
    def test_empty_command_rejected(self):
        result = self.tool.execute("")
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
        self.assertIn("Run Git commands against the institute repository", prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
