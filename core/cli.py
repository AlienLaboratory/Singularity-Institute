"""
Testable CLI helpers for the institute orchestrator.

These functions encapsulate the prompt-input logic (reading a task from a
file, reading a multiline interactive task, and command-line argument
parsing) so it can be unit-tested without launching the full agent, the LLM,
or any tool.

The agent, tools, and database wiring all stay in ``orchestrator.py``; this
module is intentionally free of those dependencies so its behaviour is easy
to reason about and verify.
"""

import argparse
from pathlib import Path


class PromptInputError(Exception):
    """
    Raised when prompt input cannot be produced in a valid form, e.g. the
    requested prompt file does not exist or is not a file.
    """


def read_prompt_file(path):
    """
    Read a task prompt from a text file and return its contents.

    The text is returned verbatim (no leading/trailing stripping) so the file
    contents become the user task exactly as written, including any leading
    blank lines, indentation, and internal newlines. Universal newline mode is
    used so CRLF line endings from Windows editors are normalised to ``\n``.

    Args:
        path: A filesystem path to a text file.

    Returns:
        The file's text content as a ``str``.

    Raises:
        PromptInputError: if ``path`` does not exist or is not a regular file.
    """
    p = Path(path)

    if not p.exists():
        raise PromptInputError(f"Prompt file not found: {path}")

    if not p.is_file():
        raise PromptInputError(f"Prompt path is not a file: {path}")

    with open(p, "r", encoding="utf-8") as f:
        return f.read()


def read_multiline(prompt_label="Institute> "):
    """
    Read a multiline task from the interactive console until end-of-input.

    Unlike the legacy ``:multi`` / ``:end`` protocol, this has no magic
    terminator token to type. The user simply types as many lines as they
    like (blank lines included) and finishes by sending an end-of-input
    marker:

      * Windows / PowerShell: ``Ctrl+Z`` then ``Enter``
      * Unix / Linux:         ``Ctrl+D``

    This is the most robust approach in a line-based console (such as
    PowerShell) because it never has to guess whether a particular line is a
    sentinel, and it preserves blank lines faithfully.

    Args:
        prompt_label: The prompt string shown before each line is read.

    Returns:
        The joined lines (``"\n".join(...)``). An empty string is returned
        if no input was received, or if the input was interrupted with
        ``Ctrl+C`` (the lines collected before the interruption are kept).
    """
    lines = []

    try:
        while True:
            lines.append(input(prompt_label))
    except EOFError:
        # Normal terminator (Ctrl+Z Enter on Windows, Ctrl+D on Unix).
        return "\n".join(lines)
    except KeyboardInterrupt:
        # Ctrl+C: keep whatever was already typed, abort gracefully.
        print()
        return "\n".join(lines)


def parse_args(argv=None):
    """
    Parse command-line arguments for the orchestrator.

    Args:
        argv: Argument list to parse. ``None`` means ``sys.argv[1:]``.

    Returns:
        An :class:`argparse.Namespace` with the following attributes:

          * ``prompt_file`` (``str | None``): path to a text file whose
            contents are used verbatim as the task; the orchestrator runs that
            task once and exits instead of entering the interactive loop.
          * ``web`` (``bool``): start the local Web UI instead of the
            interactive / single-run modes. ``False`` by default.
          * ``host`` (``str``): Web UI bind address. ``127.0.0.1`` by
            default so the UI is reachable from localhost only.
          * ``port`` (``int``): Web UI TCP port. ``8000`` by default.
    """
    parser = argparse.ArgumentParser(
        prog="orchestrator.py",
        description=(
            "Local AI scientific research institute orchestrator. "
            "Run interactively, or execute a single task from a file."
        ),
    )
    parser.add_argument(
        "--prompt-file",
        dest="prompt_file",
        metavar="PATH",
        default=None,
        help=(
            "Read the task prompt from a text file (contents used verbatim), "
            "run it once, then exit without entering the interactive loop."
        ),
    )
    parser.add_argument(
        "--web",
        dest="web",
        action="store_true",
        default=False,
        help="Start the local Web UI instead of the interactive loop. Binds to --host:--port.",
    )
    parser.add_argument(
        "--host",
        dest="host",
        metavar="HOST",
        default="127.0.0.1",
        help="Bind address for the Web UI. Defaults to 127.0.0.1 (localhost only).",
    )
    parser.add_argument(
        "--port",
        dest="port",
        type=int,
        metavar="PORT",
        default=8000,
        help="TCP port for the Web UI. Defaults to 8000.",
    )
    return parser.parse_args(argv)
