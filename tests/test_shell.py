"""
Focused tests for the controlled shell execution tool.

Run from the institute root:
    python tests/test_shell.py

Covers:
    - successful command
    - failing command (returned, not raised)
    - timeout
    - working-directory behavior (inside vs outside the institute root)
    - dangerous-command rejection
    - registry / schema integration (auto documentation)
    - timeout recovery: a timed-out command returns instead of hanging, its
      result indicates timeout/failure, and elapsed time is bounded.
"""

import os
import sys
import tempfile
import time
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.shell import ShellTool
from tools.base import ToolBase
from tools.registry import ToolRegistry
from core.agent import build_system_prompt


class ShellToolTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.tool = ShellTool(self.root, timeout=3)

    def tearDown(self):
        self.tmp.cleanup()

    # ------------------------------------------------------------------
    def test_is_a_tool_base(self):
        self.assertIsInstance(self.tool, ToolBase)
        self.assertEqual(self.tool.name, "shell")

    # ------------------------------------------------------------------
    def test_successful_command(self):
        result = self.tool.execute("Write-Output 'hello-institute'")
        self.assertTrue(result["success"])
        self.assertEqual(result["return_code"], 0)
        self.assertIn("hello-institute", result["stdout"])
        self.assertIn("command", result)
        self.assertIn("stdout", result)
        self.assertIn("stderr", result)
        self.assertIn("return_code", result)

    # ------------------------------------------------------------------
    def test_failing_command_is_returned_not_raised(self):
        result = self.tool.execute("Throw 'boom'")
        self.assertIsInstance(result, dict)
        self.assertFalse(result["success"])
        self.assertNotEqual(result["return_code"], 0)
        self.assertIn("stderr", result)

    # ------------------------------------------------------------------
    def test_nonzero_exit_code(self):
        result = self.tool.execute("exit 7")
        self.assertFalse(result["success"])
        self.assertEqual(result["return_code"], 7)

    # ------------------------------------------------------------------
    def test_timeout(self):
        start = time.time()
        result = self.tool.execute("Start-Sleep -Seconds 10")
        elapsed = time.time() - start
        self.assertFalse(result["success"])
        self.assertIn("timeout", result["error"].lower())
        self.assertLess(elapsed, 9)

    # ------------------------------------------------------------------
    def test_default_working_directory_is_institute_root(self):
        result = self.tool.execute("(Get-Location).Path")
        self.assertTrue(result["success"])
        self.assertEqual(
            result["stdout"].strip().lower(),
            self.root.lower(),
        )

    # ------------------------------------------------------------------
    def test_explicit_working_directory_inside_root(self):
        os.makedirs(os.path.join(self.root, "sub"))
        result = self.tool.execute(
            "(Get-Location).Path",
            working_directory="sub",
        )
        self.assertTrue(result["success"])
        self.assertEqual(
            result["stdout"].strip().lower(),
            os.path.join(self.root, "sub").lower(),
        )

    # ------------------------------------------------------------------
    def test_working_directory_outside_root_is_rejected(self):
        result = self.tool.execute(
            "(Get-Location).Path",
            working_directory="..",
        )
        self.assertFalse(result["success"])
        self.assertIn("denied", result["error"].lower())
        self.assertIsNone(result["return_code"])

    # ------------------------------------------------------------------
    def test_dangerous_commands_rejected(self):
        cases = [
            "format C:",
            "diskpart",
            "shutdown /s /t 0",
            "Restart-Computer",
            "cmd /c dir",
        ]
        for cmd in cases:
            result = self.tool.execute(cmd)
            self.assertFalse(result["success"], f"expected rejection for: {cmd}")
            self.assertIn("Rejected", result["error"])
            self.assertIsNone(result["return_code"])

    # ------------------------------------------------------------------
    def test_format_cmdlets_not_false_positive(self):
        result = self.tool.execute("(1..3) | Format-Table -AutoSize")
        self.assertTrue(result["success"], result)

    # ------------------------------------------------------------------
    def test_remove_item_outside_root_rejected(self):
        result = self.tool.execute("Remove-Item -Path C:\\Windows")
        self.assertFalse(result["success"])
        self.assertIn("Rejected", result["error"])

        result = self.tool.execute("Remove-Item -Path C:\\Users")
        self.assertFalse(result["success"])
        self.assertIn("Rejected", result["error"])

    # ------------------------------------------------------------------
    def test_remove_item_institute_root_rejected(self):
        result = self.tool.execute(
            f"Remove-Item -Recurse -Force -LiteralPath '{self.root}'"
        )
        self.assertFalse(result["success"])
        self.assertIn("Rejected", result["error"])

    # ------------------------------------------------------------------
    def test_remove_item_inside_root_is_allowed(self):
        target = os.path.join(self.root, "scratch.txt")
        with open(target, "w", encoding="utf-8") as f:
            f.write("x")
        result = self.tool.execute(f"Remove-Item -LiteralPath '{target}'")
        self.assertTrue(result["success"], result)
        self.assertFalse(os.path.exists(target))

    # ------------------------------------------------------------------
    def test_remove_item_inside_root_absolute_path_allowed(self):
        target = os.path.join(self.root, "sub", "inner.txt")
        os.makedirs(os.path.dirname(target))
        with open(target, "w", encoding="utf-8") as f:
            f.write("x")
        result = self.tool.execute(f"Remove-Item -LiteralPath '{target}'")
        self.assertTrue(result["success"], result)
        self.assertFalse(os.path.exists(target))

    # ------------------------------------------------------------------
    def test_remove_item_after_semicolon_rejected(self):
        result = self.tool.execute("Write-Output hi; Remove-Item C:\\Windows")
        self.assertFalse(result["success"])
        self.assertIn("Rejected", result["error"])

    # ------------------------------------------------------------------
    def test_schema_and_registry_integration(self):
        registry = ToolRegistry()
        registry.register(self.tool)

        self.assertIsNotNone(registry.get("shell"))
        self.assertEqual(registry.get_tool_schema("shell")["name"], "shell")

        docs = registry.generate_tool_docs()
        self.assertIn("shell", docs)

        prompt = build_system_prompt(registry)
        self.assertIn("shell", prompt)
        self.assertIn("Execute a controlled Windows PowerShell command", prompt)

    # ------------------------------------------------------------------
    def test_empty_command_rejected(self):
        result = self.tool.execute("")
        self.assertFalse(result["success"])
        self.assertIn("required", result["error"].lower())


# ---------------------------------------------------------------------------
# Timeout-recovery regression test
# ---------------------------------------------------------------------------
class ShellTimeoutRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.tool = ShellTool(self.root, timeout=2)

    def tearDown(self):
        self.tmp.cleanup()

    def test_timeout_returns_and_bounded(self):
        """A timed-out command must return (not hang) within a bounded time,
        with a structured failure result indicating timeout."""
        start = time.time()
        result = self.tool.execute("Start-Sleep -Seconds 60")
        elapsed = time.time() - start

        # The tool RETURNED (asserting here proves no indefinite hang).
        self.assertIsInstance(result, dict)
        self.assertFalse(result["success"])
        self.assertIn("timeout", result["error"].lower())
        # Bounded: above the 2s timeout, well below the 60s sleep.
        self.assertGreaterEqual(elapsed, 1.5)
        self.assertLess(elapsed, 20)

    def test_successful_command_after_timeout_logic(self):
        """Normal commands still work after the timeout path is in place."""
        result = self.tool.execute("Write-Output 'ok'")
        self.assertTrue(result["success"])
        self.assertEqual(result["return_code"], 0)
        self.assertIn("ok", result["stdout"])

    def test_failing_command_after_timeout_logic(self):
        """A non-timeout failure is still returned, not raised."""
        result = self.tool.execute("Throw 'boom'")
        self.assertIsInstance(result, dict)
        self.assertFalse(result["success"])
        self.assertNotEqual(result["return_code"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
