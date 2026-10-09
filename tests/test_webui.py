"""
Unit tests for the Institute Web UI (core.webui) and CLI flags.

Run from the institute root:
    python tests/test_webui.py

These tests do NOT require a browser, Ollama, or a real Agent. They use
lightweight stubs to verify routing, HTTP handler behaviour, and CLI parsing.
"""

import json
import os
import sys
import threading
import time
import unittest

# Ensure the institute root is importable regardless of CWD.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.webui import Handler, WebRunner, run_web, OutputCapture, PAGE
from core.cli import parse_args


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class FakeResult:
    """Mimics the AgentResult dataclass (duck-typed)."""
    def __init__(self, task_id=1, state="completed", answer="done"):
        self.task_id = task_id
        self.state = type("S", (), {"value": state})
        self.answer = answer


class FakeAgent:
    """Minimal Agent stub that returns a canned result."""
    def __init__(self, result=None, exc=None):
        self._result = result
        self._exc = exc

    def run(self, task):
        if self._exc:
            raise self._exc
        return self._result or FakeResult()


class FakeDatabase:
    """Minimal Database stub."""
    def close(self):
        pass


# ---------------------------------------------------------------------------
# Handler.route tests (pure routing logic, no HTTP)
# ---------------------------------------------------------------------------

class TestHandlerRoute(unittest.TestCase):
    def setUp(self):
        self.agent = FakeAgent(FakeResult(task_id=99, answer="42"))
        self.db = FakeDatabase()
        self.runner = WebRunner(self.agent, self.db)
        self.handler = Handler(self.runner)

    def test_get_root_returns_html(self):
        code, ctype, body = self.handler.route("GET", "/", b"", {})
        self.assertEqual(code, 200)
        self.assertIn("text/html", ctype)
        self.assertIn(b"Institute", body)

    def test_get_api_status_returns_json(self):
        code, ctype, body = self.handler.route("GET", "/api/status", b"", {})
        self.assertEqual(code, 200)
        self.assertIn("application/json", ctype)
        data = json.loads(body)
        self.assertEqual(data["status"], "idle")

    def test_post_run_accepts_valid_task(self):
        body = json.dumps({"task": "test task"}).encode()
        code, ctype, resp = self.handler.route("POST", "/api/run", body, {})
        self.assertEqual(code, 202)
        data = json.loads(resp)
        self.assertEqual(data["status"], "accepted")

    def test_post_run_malformed_json(self):
        code, ctype, resp = self.handler.route("POST", "/api/run", b"not-json{", {})
        self.assertEqual(code, 400)
        data = json.loads(resp)
        self.assertIn("error", data)

    def test_post_run_empty_task(self):
        body = json.dumps({"task": "   "}).encode()
        code, ctype, resp = self.handler.route("POST", "/api/run", body, {})
        self.assertEqual(code, 400)

    def test_post_run_missing_task_field(self):
        body = json.dumps({"other": "x"}).encode()
        code, ctype, resp = self.handler.route("POST", "/api/run", body, {})
        self.assertEqual(code, 400)

    def test_post_run_empty_body(self):
        code, ctype, resp = self.handler.route("POST", "/api/run", b"", {})
        self.assertEqual(code, 400)

    def test_post_run_rejects_when_busy(self):
        # Manually acquire the runner lock to simulate a running task.
        self.runner._lock.acquire()
        try:
            body = json.dumps({"task": "second task"}).encode()
            code, ctype, resp = self.handler.route("POST", "/api/run", body, {})
            self.assertEqual(code, 409)
            data = json.loads(resp)
            self.assertIn("already running", data["error"].lower())
        finally:
            self.runner._lock.release()

    def test_404_unknown_path(self):
        code, ctype, resp = self.handler.route("GET", "/nonexistent", b"", {})
        self.assertEqual(code, 404)

    def test_query_string_stripped(self):
        code, ctype, body = self.handler.route("GET", "/?verbose=1", b"", {})
        self.assertEqual(code, 200)
        self.assertIn(b"Institute", body)


# ---------------------------------------------------------------------------
# WebRunner behaviour tests
# ---------------------------------------------------------------------------

class TestWebRunner(unittest.TestCase):
    def test_submit_runs_agent(self):
        agent = FakeAgent(FakeResult(task_id=7, answer="hello"))
        runner = WebRunner(agent, FakeDatabase())
        runner.submit("do something")
        status = runner.status()
        self.assertEqual(status["task_id"], 7)
        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["answer"], "hello")

    def test_submit_agent_exception(self):
        agent = FakeAgent(exc=RuntimeError("boom"))
        runner = WebRunner(agent, FakeDatabase())
        runner.submit("do something")
        status = runner.status()
        self.assertEqual(status["status"], "failed")
        self.assertIn("boom", status["error"])

    def test_start_rejects_when_busy(self):
        import threading as _t

        # Agent that blocks for ~0.3s so the worker still holds the lock
        barrier = _t.Event()

        class SlowAgent:
            def run(self, task):
                barrier.wait(timeout=1.0)
                return FakeResult(task_id=1, answer="done")

        runner = WebRunner(SlowAgent(), FakeDatabase())
        started1, _ = runner.start("task1")
        self.assertTrue(started1)
        # Worker thread is still running (waiting on barrier) → lock held
        started2, _ = runner.start("task2")
        self.assertFalse(started2)
        # Release the first worker
        barrier.set()
        time.sleep(0.2)

    def test_output_capture(self):
        cap = OutputCapture()
        cap.write("hello ")
        cap.write("world\n")
        cap.write("line3")
        self.assertIn("hello world", str(cap))
        self.assertEqual(len(cap.lines()), 2)


# ---------------------------------------------------------------------------
# Truthful completion status
# ---------------------------------------------------------------------------

class TestCompletionStatus(unittest.TestCase):
    """Feature B: a ``completed`` result with an empty/whitespace answer is
    surfaced as ``failed``; a non-empty answer stays ``completed``."""

    def test_completed_empty_answer_is_failed(self):
        agent = FakeAgent(FakeResult(task_id=3, state="completed", answer=""))
        runner = WebRunner(agent, FakeDatabase())
        runner.submit("task")
        status = runner.status()
        self.assertEqual(status["status"], "failed")
        self.assertTrue(status["error"])
        self.assertIn("final answer", status["error"].lower())

    def test_completed_whitespace_answer_is_failed(self):
        agent = FakeAgent(FakeResult(task_id=4, state="completed", answer="   \n\t  "))
        runner = WebRunner(agent, FakeDatabase())
        runner.submit("task")
        status = runner.status()
        self.assertEqual(status["status"], "failed")
        self.assertTrue(status["error"])

    def test_completed_valid_answer_remains_completed(self):
        agent = FakeAgent(FakeResult(task_id=5, state="completed", answer="42"))
        runner = WebRunner(agent, FakeDatabase())
        runner.submit("task")
        status = runner.status()
        self.assertEqual(status["status"], "completed")
        self.assertIsNone(status["error"])
        self.assertEqual(status["answer"], "42")

    def test_exception_still_failed_with_error(self):
        agent = FakeAgent(exc=ValueError("something broke"))
        runner = WebRunner(agent, FakeDatabase())
        runner.submit("task")
        status = runner.status()
        self.assertEqual(status["status"], "failed")
        self.assertIn("something broke", status["error"])


# ---------------------------------------------------------------------------
# Copy Output UI control
# ---------------------------------------------------------------------------

class TestCopyOutputUI(unittest.TestCase):
    """Feature A: the served HTML exposes a Copy Output control and the
    clipboard-handling logic (primary API + a non-silent fallback)."""

    def test_html_contains_copy_control_and_logic(self):
        html = PAGE
        # The button is present and labelled.
        self.assertIn('id="copy-output"', html)
        self.assertIn("Copy Output", html)
        # Primary clipboard path uses the browser Clipboard API.
        self.assertIn("navigator.clipboard", html)
        self.assertIn("writeText", html)
        # A fallback exists and reports failure rather than claiming success.
        self.assertIn("execCommand('copy')", html)
        self.assertIn("Copy failed", html)
        self.assertIn("Copied", html)
        # The click handler is wired up.
        self.assertIn("copyEl.addEventListener('click', copyOutput)", html)

    def test_copy_control_served_over_route(self):
        handler = Handler(WebRunner(FakeAgent(FakeResult()), FakeDatabase()))
        code, ctype, body = handler.route("GET", "/", b"", {})
        self.assertEqual(code, 200)
        text = body.decode("utf-8")
        self.assertIn("Copy Output", text)
        self.assertIn("navigator.clipboard", text)


# ---------------------------------------------------------------------------
# run_web smoke test (start, connect, shut down)
# ---------------------------------------------------------------------------

class TestRunWebServer(unittest.TestCase):
    """Verify the stdlib HTTP server actually starts and responds."""

    def test_server_serves_root(self):
        import http.server
        import urllib.request

        agent = FakeAgent(FakeResult(task_id=1, answer="ok"))
        runner = WebRunner(agent, FakeDatabase())

        # Find a free port
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        result = []

        def _serve():
            try:
                srv = run_web(host="127.0.0.1", port=port, runner=runner)
            except Exception as e:
                result.append(e)

        t = threading.Thread(target=_serve, daemon=True)
        t.start()

        # Wait for server to be ready
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1)
                break
            except Exception:
                time.sleep(0.05)
        else:
            self.fail("Server did not start within 5s")

        # GET / should return 200
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as r:
            self.assertEqual(r.status, 200)
            self.assertIn(b"Institute", r.read())

        # GET /api/status
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=2) as r:
            data = json.loads(r.read())
            self.assertIn("status", data)

        # POST /api/run with valid task
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/run",
            data=json.dumps({"task": "hi"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2) as r:
            self.assertEqual(r.status, 202)

        # Let the background thread clean up (it will block in serve_forever;
        # since it's a daemon thread, the process will exit fine in tests).
        time.sleep(0.1)


# ---------------------------------------------------------------------------
# CLI parsing tests
# ---------------------------------------------------------------------------

class TestCliWebFlags(unittest.TestCase):
    def test_web_defaults_false(self):
        args = parse_args([])
        self.assertFalse(args.web)

    def test_web_flag(self):
        args = parse_args(["--web"])
        self.assertTrue(args.web)

    def test_host_default(self):
        args = parse_args([])
        self.assertEqual(args.host, "127.0.0.1")

    def test_port_default(self):
        args = parse_args([])
        self.assertEqual(args.port, 8000)

    def test_custom_host_port(self):
        args = parse_args(["--web", "--host", "0.0.0.0", "--port", "9999"])
        self.assertTrue(args.web)
        self.assertEqual(args.host, "0.0.0.0")
        self.assertEqual(args.port, 9999)

    def test_port_is_int(self):
        args = parse_args(["--port", "1234"])
        self.assertIsInstance(args.port, int)
        self.assertEqual(args.port, 1234)

    def test_prompt_file_still_works(self):
        args = parse_args(["--prompt-file", "task.txt"])
        self.assertEqual(args.prompt_file, "task.txt")
        self.assertFalse(args.web)

    def test_web_and_prompt_file_coexist(self):
        args = parse_args(["--web", "--prompt-file", "task.txt"])
        self.assertTrue(args.web)
        self.assertEqual(args.prompt_file, "task.txt")


if __name__ == "__main__":
    unittest.main(verbosity=2)
