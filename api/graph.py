"""RAG workflow graph instance for the API. Created once and shared by the chat router."""

from src.rag_agent import create_workflow

agent_graph = create_workflow()
