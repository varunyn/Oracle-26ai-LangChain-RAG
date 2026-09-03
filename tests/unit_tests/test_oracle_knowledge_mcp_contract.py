import asyncio
from dataclasses import dataclass

from fastmcp import Client

from mcp_servers.oracle_knowledge import create_oracle_knowledge_server
from rag_agent.application.oracle_knowledge import OracleKnowledgeService


@dataclass
class Candidate:
    content: str
    metadata: dict[str, object]
    retrieval_score: float | None = 0.4


class Embedder:
    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]


class Retriever:
    def retrieve(self, collection, query_embedding, limit, metadata_filters=None):
        return [Candidate("content", {"source": "guide"})]

    def list_documents(self, collection):
        return [{"source": "guide", "title": "Guide"}]


def test_real_mcp_client_discovers_only_typed_knowledge_tools() -> None:
    asyncio.run(_assert_contract())


async def _assert_contract() -> None:
    service = OracleKnowledgeService(
        knowledge_bases={"docs": "RAW"},
        embedder=Embedder(),
        retriever=Retriever(),
        enable_reranker=False,
        default_knowledge_base="docs",
    )
    async with Client(create_oracle_knowledge_server(service)) as client:
        tools = await client.list_tools()
        assert [tool.name for tool in tools] == [
            "search_knowledge",
            "list_knowledge_bases",
            "list_documents",
        ]
        for tool in tools:
            assert tool.input_schema.get("type") == "object"
            assert tool.output_schema.get("type") == "object"
        search_schema = next(tool for tool in tools if tool.name == "search_knowledge")
        assert {"query", "knowledge_base", "limit", "metadata_filters"} <= set(
            search_schema.input_schema["properties"]
        )
        assert "collection_name" not in search_schema.input_schema["properties"]
        assert "table_name" not in search_schema.input_schema["properties"]
        assert search_schema.input_schema["properties"]["query"]["maxLength"] == 100000
        assert search_schema.input_schema["properties"]["limit"]["maximum"] == 100
        assert {"contract_version", "outcome", "evidence", "reranking_status"} <= set(
            search_schema.output_schema["properties"]
        )
        result = await client.call_tool(
            "search_knowledge", {"query": "hello", "knowledge_base": "docs"}
        )
        assert result.structured_content["outcome"] == "success"
        assert result.structured_content["evidence"][0]["rank"] == 1
        bases = await client.call_tool("list_knowledge_bases", {})
        assert bases.structured_content["knowledge_bases"] == [{"key": "docs"}]
        documents = await client.call_tool("list_documents", {"knowledge_base": "docs"})
        assert documents.structured_content["documents"][0]["source"] == "guide"
        raw_key = "RAW_ORACLE_COLLECTION_SECRET"
        forbidden = await client.call_tool(
            "search_knowledge", {"query": "hello", "knowledge_base": raw_key}
        )
        assert forbidden.structured_content["outcome"] == "forbidden"
        assert forbidden.structured_content["knowledge_base"] is None
        assert raw_key not in str(forbidden.structured_content)
        invalid = await client.call_tool(
            "search_knowledge", {"query": " ", "knowledge_base": raw_key}
        )
        assert invalid.structured_content["outcome"] == "invalid_request"
        assert invalid.structured_content["knowledge_base"] is None
        assert raw_key not in str(invalid.structured_content)
        forbidden_documents = await client.call_tool("list_documents", {"knowledge_base": raw_key})
        assert forbidden_documents.structured_content["outcome"] == "forbidden"
        assert forbidden_documents.structured_content["knowledge_base"] is None
        assert raw_key not in str(forbidden_documents.structured_content)
