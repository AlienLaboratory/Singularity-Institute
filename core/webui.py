"""
Minimal local web UI for the Institute Agent.

Design goals (kept intentionally small):

* Reuses the EXISTING Agent loop (``Agent.run``). The agent, tools, and
  database wiring all live elsewhere (``core.agent`` / ``orchestrator.py``);
  this module only *drives* them over a local HTTP server.
* Uses only the Python standard library (``http.server``, ``json``,
  ``threading``, ``collections``). No external dependencies.
* Binds to ``127.0.0.1`` (localhost) only. No authentication is added
  because the service is deliberately unreachable from the network.
* Serialises agent tasks: at most one Agent task runs at a time. A second
  submission while one is in flight is rejected (HTTP 409) rather than
  starting a parallel Agent run.

The UI simply: shows a large multiline task box, a Run button, a live output
pane, a Copy Output button, and a status badge (idle / running / completed /
failed / max_steps). The user workflow is: start the Institute with ``--web``,
open the URL in a browser, paste the task, click Run. No :multi / :end /
Ctrl+Z / prompt files.
"""

import json
import http.server
import sys
import threading
from collections import deque

from core.exceptions import AgentMaxStepsError
from memory.database import Database

# Status values surfaced to the UI. ``running`` is a transient state derived
# from whether a task is in flight; the rest mirror the Agent's own outcome.
IDLE = "idle"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
MAX_STEPS = "max_steps"

# Hard cap on how many output lines the UI keeps in memory (bounded so a
# long-running task cannot grow memory unboundedly). Each kept line is
# additionally truncated so a single huge line cannot dominate memory.
MAX_OUTPUT_LINES = 2000
MAX_LINE_CHARS = 8000


class OutputCapture:
    """
    A thread-local ``sys.stdout`` sink that records the agent's live output
    into a bounded, thread-safe line buffer.

    Only ``write`` and ``flush`` are needed for ``print`` to work, plus
    context-manager support for ``with cap as stream:``. The buffer is a
    ``deque`` (thread-safe for single operations) with a hard ``maxlen``, and
    per-line writes are additionally serialised by a lock so ``write``/append
    are not interleaved.
    """

    def __init__(self, maxlen=MAX_OUTPUT_LINES, line_limit=MAX_LINE_CHARS):
        self._buffer = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._line_limit = line_limit
        self._pending = ""

    def write(self, text):
        if not text:
            return 0
        if self._line_limit:
            text = text[: self._line_limit]
        with self._lock:
            self._pending += text
            while "\n" in self._pending:
                line, self._pending = self._pending.split("\n", 1)
                self._buffer.append(line)
        return len(text)

    def flush(self):
        with self._lock:
            if self._pending:
                self._buffer.append(self._pending)
                self._pending = ""

    def __str__(self):
        """The captured output as a single string (oldest first)."""
        self.flush()
        with self._lock:
            return "\n".join(self._buffer)

    def lines(self):
        self.flush()
        with self._lock:
            return list(self._buffer)

    def clear(self):
        with self._lock:
            self._buffer.clear()
            self._pending = ""

    def __enter__(self):
        self._original = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc, tb):
        # Flush any buffered partial line, then restore the real stdout so the
        # main thread (and anything running after the task) is unaffected.
        self.flush()
        sys.stdout = self._original
        return False


class WebRunner:
    """
    Runs the existing Agent's tasks one at a time in a worker thread.

    * ``start(task)``    -> (started: bool, task_id). Non-blocking. Used by the
      HTTP handler to spawn a task (or report busy).
    * ``submit(task)``   -> blocking, for tests / programmatic one-shot runs.
    * ``running()``      -> True while a task is in flight.
    * ``status()``       -> a plain dict describing the current state.

    The runner holds a ``threading.Lock`` so a task is never started while
    another is in flight (requirement: prevent multiple simultaneous Agent
    tasks). Live output is captured per-run into ``OutputCapture``.
    """

    def __init__(self, agent, database, max_lines=MAX_OUTPUT_LINES):
        self.agent = agent
        self.database = database
        self.max_lines = max_lines
        self._lock = threading.Lock()
        self.capture = OutputCapture(maxlen=max_lines)
        self.task_id = None
        self.state = IDLE
        self.final_answer = ""
        self.error = None

    # -- state ---------------------------------------------------------------
    def running(self):
        """True if a task is in flight (lock held)."""
        return self._lock.locked()

    def status(self):
        """A plain dict describing the current task state (for the UI)."""
        state = RUNNING if self.running() else self.state
        return {
            "task_id": self.task_id,
            "status": state,
            "answer": self.final_answer,
            "error": self.error,
            "output": str(self.capture),
        }

    # -- execution -----------------------------------------------------------
    def _run_worker(self, task):
        """
        Execute one task on the *calling* thread (a worker thread when
        launched by :meth:`start`, the caller's thread for :meth:`submit`).

        Captures the agent's live stdout, drives ``agent.run`` (the existing
        Agent loop), and records the outcome / final answer.
        """
        self.capture.clear()
        self.error = None
        with self.capture:
            try:
                result = self.agent.run(task)
                self.task_id = result.task_id
                self.state = result.state.value
                self.final_answer = result.answer or ""
                # A task that reports ``completed`` but carries an empty or
                # whitespace-only final answer is not actually successful: do
                # not infer success from a finished worker thread alone. Report
                # it as failed with an informative error. Non-empty answers
                # keep the ``completed`` state untouched.
                if self.state == COMPLETED and not self.final_answer.strip():
                    self.state = FAILED
                    self.error = "Agent finished without producing a final answer."
            except AgentMaxStepsError as e:
                self.task_id = getattr(e, "task_id", None)
                self.state = MAX_STEPS
                self.final_answer = ""
                self.error = str(e)
            except Exception as e:
                self.state = FAILED
                self.final_answer = ""
                self.error = str(e)

    def start(self, task):
        """
        Start ``task`` on a background worker thread.

        Returns ``(started, task_id)``. If a task is already in flight the
        caller is told ``started=False`` (HTTP 409) and nothing is spawned.
        """
        if not self._lock.acquire(blocking=False):
            return False, None

        def _worker():
            try:
                self._run_worker(task)
            finally:
                self._lock.release()

        threading.Thread(target=_worker, daemon=True).start()
        return True, None

    def submit(self, task):
        """
        Run ``task`` on the calling thread, blocking until it finishes.

        If a task is already in flight this blocks until that task releases
        the lock, then runs (it never overlaps with another Agent task).
        """
        self._lock.acquire()
        try:
            self._run_worker(task)
        finally:
            self._lock.release()


class Handler:
    """
    Serves the static UI page and the JSON API on top of a :class:`WebRunner`.

    Endpoints:
      * ``GET  /``           -> the HTML page
      * ``GET  /api/status`` -> current state + captured output
      * ``POST /api/run``    -> body ``{"task": "..."}``; 202 to start, 409 if
                                a task is in flight, 400 for bad input
    """

    MAX_BODY = 5_000_000  # 5 MB hard cap on the POST body size

    def __init__(self, runner):
        self.runner = runner

    # -- routing -------------------------------------------------------------
    def route(self, method, path, body, headers):
        """
        Handle one request. Returns ``(status_code, content_type, body)``.

        ``body`` is a ``bytes`` payload (or ``b""``). ``headers`` is a
        case-insensitive mapping of the request headers (only the ``Content-Length``
        value is read from it for the body-size cap).
        """
        path = path.split("?", 1)[0]
        if method == "GET" and path == "/":
            return 200, "text/html; charset=utf-8", PAGE.encode("utf-8")
        if method == "GET" and path == "/api/status":
            return self._json(200, self.runner.status())
        if method == "POST" and path == "/api/run":
            return self._handle_run(body, headers)
        return self._json(404, {"error": "Not found"})

    # -- helpers -------------------------------------------------------------
    def _json(self, code, payload):
        return (
            code,
            "application/json; charset=utf-8",
            json.dumps(payload).encode("utf-8"),
        )

    def _handle_run(self, body, headers=None):
        # ``body`` is the raw request body as ``bytes`` (or ``b""``).
        # ``headers`` is an optional case-insensitive mapping used only to
        # read Content-Length; when absent we fall back to the byte length.
        headers = headers or {}
        content_length = 0
        try:
            content_length = int(headers.get("Content-Length", 0) or 0)
        except (TypeError, ValueError):
            content_length = 0
        if body:
            content_length = max(content_length, len(body))

        if content_length and content_length > self.MAX_BODY:
            return self._json(
                413, {"error": "Task too large (exceeds body size limit)"}
            )

        raw = body if isinstance(body, (bytes, bytearray)) else (body or "")
        try:
            payload = json.loads(raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw) if raw else None
        except (UnicodeDecodeError, ValueError):
            payload = None

        if not isinstance(payload, dict):
            return self._json(400, {"error": "Request body must be JSON"})

        task = payload.get("task")
        if not isinstance(task, str) or not task.strip():
            return self._json(400, {"error": "Task must be a non-empty string"})

        if self.runner.running():
            return self._json(
                409, {"error": "A task is already running; please wait for it to finish."}
            )

        started, task_id = self.runner.start(task)
        if not started:
            return self._json(
                409, {"error": "A task is already running; please wait for it to finish."}
            )
        return self._json(202, {"status": "accepted", "task_id": task_id})


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Institute</title>
  <style>
    :root { color-scheme: light dark; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; } 
    .wrap { max-width: 1000px; margin: 0 auto; padding: 24px; } 
    h1 { font-size: 20px; margin: 0 0 4px; }
    .muted { color: #888; font-size: 13px; }
    .row { display: flex; gap: 16px; margin-top: 16px; }
    .task { flex: 3; min-width: 0; }
    .side { flex: 1; min-width: 200px; }
    label { display: block; font-size: 13px; color: #555; margin-bottom: 6px; }
    textarea {
      width: 100%; min-height: 340px; resize: vertical; padding: 12px;
      font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
      font-size: 14px; line-height: 1.5; border: 1px solid #ccc; border-radius: 8px;
    }
    .actions { display: flex; gap: 8px; margin-top: 10px; }
    button {
      padding: 10px 18px; font-size: 14px; font-weight: 600; cursor: pointer;
      border: none; border-radius: 8px;
    }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    .run { background: #2563eb; color: #fff; }
    .stop { background: #dc2626; color: #fff; }
    .copy { background: #374151; color: #fff; font-size: 13px; padding: 6px 12px; }
    .copy.flash { background: #16a34a; }
    .output-head { display: flex; align-items: center; justify-content: space-between; }
    .output-head label { margin-bottom: 6px; }
    .status-badge {
      display: inline-block; padding: 4px 10px; border-radius: 999px;
      font-size: 13px; font-weight: 600; background: #e5e7eb; color: #374151;
    }
    .status-badge.running { background: #fef9c3; color: #854d0e; }
    .status-badge.completed { background: #dcfce7; color: #166534; }
    .status-badge.failed { background: #fee2e2; color: #991b1b; }
    .status-badge.max_steps { background: #ffedd5; color: #9a3412; }
    .status-detail { font-size: 13px; color: #555; margin-top: 6px; }
    .answer, .error {
      margin-top: 12px; padding: 10px 12px; border-radius: 8px;
      background: #f3f4f6; font-size: 13px; white-space: pre-wrap;
    }
    .error { background: #fee2e2; color: #991b1b; }
    .terminal {
      background: #111827; color: #e5e7eb; padding: 12px; border-radius: 8px;
      font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
      font-size: 12.5px; line-height: 1.5; white-space: pre-wrap;
      height: 460px; overflow-y: auto; margin-top: 8px;
    }
    .empty { color: #6b7280; font-style: italic; }
    @media (max-width: 720px) { .row { flex-direction: column; } }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Institute</h1>
    <div class="muted">Local task runner &middot; one task at a time &middot; localhost only</div>

    <div class="row">
      <div class="task">
        <label for="task">Task</label>
        <textarea id="task" placeholder="Paste your task here (blank lines are preserved). Click Run."></textarea>
        <div class="actions">
          <button id="run" class="run">Run</button>
          <button id="stop" class="stop" disabled title="Stopping a running task is not yet supported">Stop</button>
        </div>
      </div>
      <div class="side">
        <label>Status</label>
        <div>
          <span id="status" class="status-badge">idle</span>
          <div id="status-detail" class="status-detail">No task submitted yet.</div>
        </div>
        <div id="answer" class="answer" style="display:none"></div>
        <div id="error" class="error" style="display:none"></div>
      </div>
    </div>

    <div class="output-head">
      <label>Live output</label>
      <button id="copy-output" class="copy" type="button" title="Copy the full output to the clipboard">Copy Output</button>
    </div>
    <div id="output" class="terminal"><span class="empty">Output will appear here as the task runs.</span></div>
  </div>

  <script>
    (function () {
      var taskEl = document.getElementById('task');
      var runEl = document.getElementById('run');
      var stopEl = document.getElementById('stop');
      var statusEl = document.getElementById('status');
      var statusDetail = document.getElementById('status-detail');
      var answerEl = document.getElementById('answer');
      var errorEl = document.getElementById('error');
      var outputEl = document.getElementById('output');
      var copyEl = document.getElementById('copy-output');
      var lastOutput = '';
      var running = false;

      function render() {
        fetch('/api/status')
          .then(function (r) { return r.json(); })
          .then(function (s) {
            var st = s.status || 'idle';
            statusEl.textContent = st;
            statusEl.className = 'status-badge ' + st;
            statusDetail.textContent = (s.task_id != null) ? ('task #' + s.task_id) : 'No task submitted yet.';
            if (s.output) {
              outputEl.textContent = s.output;
              outputEl.scrollTop = outputEl.scrollHeight;
              lastOutput = s.output;
            }
            if (st === 'completed' && s.answer) {
              answerEl.textContent = s.answer;
              answerEl.style.display = 'block';
            } else if (st === 'idle') {
              answerEl.style.display = 'none';
            }
            if (st === 'failed' || st === 'max_steps') {
              errorEl.textContent = (s.error || 'Task did not complete successfully.');
              errorEl.style.display = 'block';
            } else if (st === 'running') {
              errorEl.style.display = 'none';
            }
            running = (st === 'running');
            runEl.disabled = running;
            stopEl.disabled = !running;
            stopEl.title = running
              ? 'Stopping a running task is not yet supported (wait for it to finish).'
              : 'Stopping a running task is not yet supported';
          })
          .catch(function (e) {
            statusEl.textContent = 'error';
            statusEl.className = 'status-badge failed';
            statusDetail.textContent = 'Lost connection to the server: ' + e;
          });
      }

      function run() {
        var task = taskEl.value;
        if (!task.trim()) { taskEl.focus(); return; }
        runEl.disabled = true;
        fetch('/api/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ task: task })
        })
          .then(function (r) { return r.json().then(function (b) { return { ok: r.ok, status: r.status, body: b }; }); })
          .then(function (res) {
            if (res.ok) {
              render();
            } else {
              statusDetail.textContent = (res.body && res.body.error) || ('Error (HTTP ' + res.status + ')');
              runEl.disabled = false;
            }
          })
          .catch(function (e) {
            statusDetail.textContent = 'Request failed: ' + e;
            runEl.disabled = false;
          });
      }

      function flashCopy(ok) {
        copyEl.textContent = ok ? 'Copied' : 'Copy failed';
        copyEl.classList.toggle('flash', ok);
        setTimeout(function () {
          copyEl.textContent = 'Copy Output';
          copyEl.classList.remove('flash');
        }, 2000);
      }

      function fallbackCopy(text, done) {
        // Legacy path for browsers without a working Clipboard API. This must
        // not silently claim success: it reports based on the real result of
        // document.execCommand('copy').
        try {
          var ta = document.createElement('textarea');
          ta.value = text;
          ta.setAttribute('readonly', '');
          ta.style.position = 'absolute';
          ta.style.left = '-9999px';
          document.body.appendChild(ta);
          ta.select();
          try { ta.setSelectionRange(0, ta.value.length); } catch (err) { /* ignore */ }
          var ok = document.execCommand('copy');
          document.body.removeChild(ta);
          done(!!ok);
        } catch (e) {
          done(false);
        }
      }

      function copyOutput() {
        // Copy the full output currently shown in the output pane (not just
        // any selected text). This does not stop, clear, or modify the task
        // or the output -- it is a read-only copy action.
        var text = outputEl.textContent || '';
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text)
            .then(function () { flashCopy(true); })
            .catch(function () { fallbackCopy(text, flashCopy); });
        } else {
          fallbackCopy(text, flashCopy);
        }
      }

      runEl.addEventListener('click', run);
      copyEl.addEventListener('click', copyOutput);
      taskEl.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); run(); }
      });
      render();
      setInterval(render, 700);
    })();
  </script>
</body>
</html>
"""


class InstituteRequestHandler(http.server.BaseHTTPRequestHandler):
    """
    A stdlib ``http.server`` request handler that delegates all routing to the
    shared :class:`Handler` (which wraps a :class:`WebRunner`).

    Keeping the routing logic in :class:`Handler` means the same code path is
    exercised both by the live HTTP server and by the tests (which call
    ``Handler.route`` directly), so the server itself is thin and trivial.
    """

    server_version = "Institute/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep the console quiet during runs
        pass

    @property
    def _handler(self):
        # The server instance (injected below) carries the shared Handler.
        return self.server.app_handler

    def _send(self, status, content_type, body):
        body = body if isinstance(body, (bytes, bytearray)) else str(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _respond(self, method):
        path = self.path
        headers = {}
        for k, v in self.headers.items():
            headers[k] = v
        body = b""
        if method in ("POST", "PUT", "PATCH"):
            length = int(self.headers.get("Content-Length", 0) or 0)
            if length > 0:
                body = self.rfile.read(length)
        status, content_type, payload = self._handler.route(method, path, body, headers)
        self._send(status, content_type, payload)

    def do_GET(self):
        self._respond("GET")

    def do_HEAD(self):
        self._respond("HEAD")

    def do_POST(self):
        self._respond("POST")


class InstituteHTTPServer(http.server.ThreadingHTTPServer):
    """A :class:`ThreadingHTTPServer` that carries the shared :class:`Handler`."""

    daemon_threads = True

    def __init__(self, address, app_handler):
        self.app_handler = app_handler
        super().__init__(address, InstituteRequestHandler)


def run_web(host="127.0.0.1", port=8000, runner=None, agent=None, database=None,
            max_lines=MAX_OUTPUT_LINES):
    """
    Build a :class:`WebRunner` (if not supplied) and start a blocking, local,
    standard-library HTTP server for the Institute Web UI.

    This is the entry point used by the ``--web`` CLI flag. It binds to
    ``host`` (``127.0.0.1`` by default, i.e. localhost only) on ``port`` and
    blocks until the process is interrupted (Ctrl+C / SIGINT).

    Args:
        host: Bind address. ``127.0.0.1`` by default so the UI is unreachable
            from the network.
        port: TCP port. ``8000`` by default.
        runner: An existing :class:`WebRunner` to reuse. When omitted one is
            built from ``agent`` (required in that case) and ``database``.
        agent: The existing :class:`core.agent.Agent` to drive tasks.
        database: The existing :class:`memory.database.Database` (passed to the
            runner for parity; the agent already holds the real reference).
        max_lines: Bounded output-buffer size in lines.

    Returns:
        The running :class:`InstituteHTTPServer` (useful for tests that want to
        stop it programmatically). In normal blocking use the function does not
        return until interrupted.
    """
    if runner is None:
        if agent is None:
            raise ValueError("run_web requires either a runner or an agent")
        runner = WebRunner(agent, database, max_lines=max_lines)

    app_handler = Handler(runner)
    server = InstituteHTTPServer((host, port), app_handler)

    print(f"Institute Web UI: http://{host}:{port}  (localhost only, one task at a time)")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down the Web UI.")
    finally:
        server.server_close()

    return server
