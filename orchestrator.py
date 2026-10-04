from pathlib import Path

import yaml

from core.agent import Agent
from core.exceptions import AgentMaxStepsError
from core.llm import OllamaClient
from memory.database import Database
from tools.registry import ToolRegistry
from tools.filesystem import FilesystemTool
from tools.python_tool import PythonTool
from tools.shell import ShellTool
from tools.git import GitTool


ROOT = Path(__file__).resolve().parent


def main():
    # Load configuration
    config_path = ROOT / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Database
    database = Database(str(ROOT))

    # LLM
    llm_config = config["llm"]

    llm = OllamaClient(
        base_url=llm_config["base_url"],
        model=llm_config["model"],
    )

    # Tools
    registry = ToolRegistry()

    filesystem_tool = FilesystemTool(str(ROOT))
    registry.register(filesystem_tool)

    python_tool = PythonTool(str(ROOT))
    registry.register(python_tool)

    shell_tool = ShellTool(str(ROOT))
    registry.register(shell_tool)

    git_tool = GitTool(str(ROOT))
    registry.register(git_tool)

    # Agent
    agent = Agent(
    llm=llm,
    tools=registry,
    database=database,
    max_steps=config["agent"]["max_steps"],
)

    print()
    print("=" * 60)
    print("INSTITUTE")
    print("=" * 60)
    print(f"Root: {ROOT}")
    print(f"Model: {llm_config['model']}")
    print()

    print("Commands:")
    print("  :multi  - enter multiline task")
    print("  :help   - show commands")
    print("  :exit   - shut down")
    print()

    while True:
        try:
            first_line = input("Institute> ")
        except (KeyboardInterrupt, EOFError):
            print("\nShutting down.")
            database.close()
            break

        if not first_line.strip():
            continue

        command = first_line.strip().lower()

        if command in {":exit", ":quit"}:
            print("Shutting down.")
            database.close()
            break

        if command == ":help":
            print()
            print(":multi  - enter multiline task")
            print(":help   - show commands")
            print(":exit   - shut down")
            print()
            continue

        if command == ":multi":
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

        except AgentMaxStepsError as e:
            print()
            print("=" * 60)
            print("MAX STEPS REACHED")
            print("=" * 60)
            print(str(e))

        except Exception as e:
            print()
            print("[AGENT ERROR]")
            print(str(e))


if __name__ == "__main__":
    main()
