from pathlib import Path

import yaml

from core.agent import Agent
from core.cli import PromptInputError, parse_args, read_multiline, read_prompt_file
from core.exceptions import AgentMaxStepsError
from core.webui import WebRunner, run_web
from core.llm import OllamaClient
from memory.database import Database
from tools.registry import ToolRegistry
from tools.filesystem import FilesystemTool
from tools.python_tool import PythonTool
from tools.shell import ShellTool
from tools.git import GitTool


ROOT = Path(__file__).resolve().parent


def build_agent(config):
    """Construct the Database, LLM, tool registry, and Agent from config."""
    database = Database(str(ROOT))

    llm_config = config["llm"]

    llm = OllamaClient(
        base_url=llm_config["base_url"],
        model=llm_config["model"],
    )

    registry = ToolRegistry()

    filesystem_tool = FilesystemTool(str(ROOT))
    registry.register(filesystem_tool)

    python_tool = PythonTool(str(ROOT))
    registry.register(python_tool)

    shell_tool = ShellTool(str(ROOT))
    registry.register(shell_tool)

    git_tool = GitTool(str(ROOT))
    registry.register(git_tool)

    agent = Agent(
        llm=llm,
        tools=registry,
        database=database,
        max_steps=config["agent"]["max_steps"],
    )

    return database, llm_config, agent


def print_banner(llm_config):
    print()
    print("=" * 60)
    print("INSTITUTE")
    print("=" * 60)
    print(f"Root: {ROOT}")
    print(f"Model: {llm_config['model']}")
    print()


def print_commands():
    print("Commands:")
    print("  :task   - enter multiline task (finish with Ctrl+Z Enter)")
    print("  :multi  - enter multiline task (finish with :end)")
    print("  :help   - show commands")
    print("  :exit   - shut down")
    print()


def run_task(agent, task):
    """
    Run a single task and print its outcome, mirroring the interactive loop
    behaviour. Returns True on completion, False if a non-fatal agent error
    was printed (so the interactive loop can continue).
    """
    print()
    print("=" * 60)
    print("TASK")
    print("=" * 60)
    print(task)

    try:
        result = agent.run(task)

        print()
        print("=" * 60)
        print("FINAL ANSWER")
        print("=" * 60)
        print(result)
        return True

    except AgentMaxStepsError as e:
        print()
        print("=" * 60)
        print("MAX STEPS REACHED")
        print("=" * 60)
        print(str(e))
        return True

    except Exception as e:
        print()
        print("[AGENT ERROR]")
        print(str(e))
        return False


def main():
    # Load configuration
    config_path = ROOT / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    args = parse_args()

    database, llm_config, agent = build_agent(config)

    try:
        # -- Non-interactive single-run mode ------------------------------
        if args.prompt_file is not None:
            try:
                task = read_prompt_file(args.prompt_file)
            except PromptInputError as e:
                print(f"[PROMPT INPUT ERROR] {e}")
                raise SystemExit(1)

            print_banner(llm_config)
            run_task(agent, task)
            return

        # -- Web UI mode ---------------------------------------------------
        if args.web:
            runner = WebRunner(agent, database)
            run_web(host=args.host, port=args.port, runner=runner)
            return

        # -- Interactive mode ---------------------------------------------
        print_banner(llm_config)
        print_commands()

        while True:
            try:
                first_line = input("Institute> ")
            except (KeyboardInterrupt, EOFError):
                print("\nShutting down.")
                break

            if not first_line.strip():
                continue

            command = first_line.strip().lower()

            if command in {":exit", ":quit"}:
                print("Shutting down.")
                break

            if command == ":help":
                print()
                print(":task   - enter multiline task (finish with Ctrl+Z Enter)")
                print(":multi  - enter multiline task (finish with :end)")
                print(":help   - show commands")
                print(":exit   - shut down")
                print()
                continue

            if command == ":task":
                print("Enter your task, then finish with Ctrl+Z Enter (Ctrl+D on Unix).")
                print()

                task = read_multiline(">> ")

                if not task.strip():
                    print("(empty task; nothing to do)")
                    continue

            elif command == ":multi":
                print("Enter your task.")
                print("Type :end on a new line when finished.")
                print()

                lines = []

                while True:
                    try:
                        line = input(">> ")
                    except (KeyboardInterrupt, EOFError):
                        print("\nMultiline input cancelled.")
                        lines = []
                        break

                    if line.strip() == ":end":
                        break

                    lines.append(line)

                if not lines:
                    continue

                task = "\n".join(lines)

            else:
                task = first_line

            run_task(agent, task)

    finally:
        database.close()


if __name__ == "__main__":
    main()
