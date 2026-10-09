"""
Unit tests for the testable CLI prompt-input helpers in ``core.cli``.

Run from the institute root:
    python tests/test_cli.py
"""

import builtins
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

# Ensure the institute root is importable regardless of CWD.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.cli import PromptInputError, parse_args, read_multiline, read_prompt_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fake_input(lines):
    """Return a callable mimicking ``input`` that yields ``lines`` in order,
    then raises ``EOFError`` (simulating the Ctrl+Z Enter / Ctrl+D terminator).

    Like the real ``input``, it first writes the prompt to ``sys.stdout``.
    """
    lines = list(lines)

    def _input(prompt=""):
        if prompt:
            sys.stdout.write(prompt)
            sys.stdout.flush()
        if not lines:
            raise EOFError
        return lines.pop(0)

    return _input


def read_multiline_with(lines):
    """Run ``read_multiline`` with a scripted ``input`` (then EOF) and capture
    stdout. Returns ``(result, stdout_text)``."""
    buf = io.StringIO()
    original_input = builtins.input
    builtins.input = fake_input(lines)
    try:
        with redirect_stdout(buf):
            result = read_multiline(">> ")
    finally:
        builtins.input = original_input
    return result, buf.getvalue()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class ReadPromptFileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, name, content):
        path = os.path.join(self.root, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_reads_contents_verbatim(self):
        path = self._write("task.txt", "line one\nline two\nline three")
        self.assertEqual(read_prompt_file(path), "line one\nline two\nline three")

    def test_preserves_leading_and_trailing_whitespace(self):
        # Contents become the task exactly as written - no stripping.
        path = self._write("task.txt", "\n\n  indented task  \n")
        self.assertEqual(read_prompt_file(path), "\n\n  indented task  \n")

    def test_preserves_internal_blank_lines(self):
        path = self._write("task.txt", "a\n\nb\n\n\nc")
        self.assertEqual(read_prompt_file(path), "a\n\nb\n\n\nc")

    def test_normalises_crlf_line_endings(self):
        path = os.path.join(self.root, "crlf.txt")
        with open(path, "wb") as f:
            f.write(b"first\r\nsecond\r\n")
        self.assertEqual(read_prompt_file(path), "first\nsecond\n")

    def test_empty_file_returns_empty_string(self):
        path = self._write("empty.txt", "")
        self.assertEqual(read_prompt_file(path), "")

    def test_missing_file_raises_prompt_input_error(self):
        with self.assertRaises(PromptInputError):
            read_prompt_file(os.path.join(self.root, "does_not_exist.txt"))

    def test_directory_raises_prompt_input_error(self):
        subdir = os.path.join(self.root, "adir")
        os.makedirs(subdir, exist_ok=True)
        with self.assertRaises(PromptInputError):
            read_prompt_file(subdir)

    def test_error_message_mentions_path(self):
        missing = os.path.join(self.root, "nope.txt")
        with self.assertRaises(PromptInputError) as ctx:
            read_prompt_file(missing)
        self.assertIn(missing, str(ctx.exception))


class ReadMultilineTest(unittest.TestCase):
    def test_joins_lines_with_newlines(self):
        result, _ = read_multiline_with(["alpha", "beta", "gamma"])
        self.assertEqual(result, "alpha\nbeta\ngamma")

    def test_preserves_blank_lines(self):
        result, _ = read_multiline_with(["one", "", "two", ""])
        self.assertEqual(result, "one\n\ntwo\n")

    def test_single_line(self):
        result, _ = read_multiline_with(["just one line"])
        self.assertEqual(result, "just one line")

    def test_eof_immediately_returns_empty_string(self):
        result, _ = read_multiline_with([])
        self.assertEqual(result, "")

    def test_prompts_are_displayed(self):
        _, out = read_multiline_with(["x", "y"])
        self.assertIn(">> ", out)
        self.assertEqual(out.count(">> ") >= 1, True)


class ParseArgsTest(unittest.TestCase):
    def test_defaults_prompt_file_to_none(self):
        args = parse_args([])
        self.assertIsNone(args.prompt_file)

    def test_prompt_file_flag(self):
        args = parse_args(["--prompt-file", "prompts/task.txt"])
        self.assertEqual(args.prompt_file, "prompts/task.txt")

    def test_prompt_file_equal_form(self):
        args = parse_args(["--prompt-file=prompts/task.txt"])
        self.assertEqual(args.prompt_file, "prompts/task.txt")

    def test_unknown_flag_is_rejected(self):
        with self.assertRaises(SystemExit):
            parse_args(["--not-a-real-flag"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
