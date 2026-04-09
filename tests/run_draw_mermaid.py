"""
Manual script: print the RAG workflow as Mermaid diagram.

Uses LangGraph get_graph().draw_mermaid() (see use-graph-api#mermaid).
Usage:
  uv run python tests/run_draw_mermaid.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.rag_agent import create_workflow


def main() -> None:
    app = create_workflow()
    print(app.get_graph().draw_mermaid())


if __name__ == "__main__":
    main()
