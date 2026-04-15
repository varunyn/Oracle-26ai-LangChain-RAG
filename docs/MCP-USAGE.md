# MCP (Model Context Protocol) Usage

This project supports MCP in two ways:

| Role                       | Description                                                                                                                                                                                                                     |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Exposing an MCP server** | This repo runs MCP servers (e.g. `mcp_servers/mcp_semantic_search.py`) that expose tools (semantic search, list collections). Other clients—Next.js UI or external apps—call these servers.                                     |
| **Consuming MCP**          | The FastAPI backend (called by the Next.js UI) acts as an **MCP client**: it connects to one or more MCP server URLs (from `config.MCP_SERVERS_CONFIG`), loads tools from those servers, and lets the LLM use them during chat. |

The sections below are split by **exposing** vs **consuming**.

---

## Part 1: Exposing an MCP server (this project)

You can run MCP servers from this repo so that the RAG app (or any MCP client) can call their tools.

### Available MCP servers

| File                                 | Transport  | Description                                               |
| ------------------------------------ | ---------- | --------------------------------------------------------- |
| `mcp_servers/mcp_semantic_search.py` | HTTP/Stdio | Semantic search + collections (set `TRANSPORT` in config) |
| `mcp_servers/mcp_rag_server.py`      | HTTP/Stdio | Full RAG pipeline as `rag_ask` tool                       |

### Tools exposed by the semantic search server

1. **`semantic_search`** – Search for relevant documents
   - Parameters: `query` (required), `top_k` (default: 5), `collection_name` (optional), `search_mode` (optional: `vector`/`hybrid`/`text`)
2. **`get_collections`** – List all available collections
3. **`list_documents_in_collection`** – List documents in a collection
   - Parameters: `collection_name` (optional)

### Quick start: run the MCP server, then the UI

1. **Start the MCP server** (exposing tools):

   ```bash
    uv run python mcp_servers/mcp_semantic_search.py
   ```

   Server listens on `http://localhost:9000` by default (or `PORT` from config). This is the standalone MCP server runtime. The backend's MCP client configuration is separate and comes from `MCP_SERVERS_CONFIG`, which may point to other preset MCP endpoints unless you override it.

2. **Start the FastAPI backend** (which consumes MCP servers on behalf of the UI) via `./run_api.sh`, then use the Next.js frontend to chat with MCP tools (see below).

### Testing the MCP server (call it directly)

**Python:**

```python
import asyncio
from fastmcp import Client

async def test():
    client = Client("http://localhost:9000/mcp")
    async with client:
        result = await client.call_tool(
            "semantic_search",
            {"query": "Oracle 23AI", "top_k": 5, "search_mode": "hybrid"}
        )
        print(result)

asyncio.run(test())
```

Or use the manual scripts (no pytest):  
`uv run python tests/run_mcp_semantic_search.py`,  
`uv run python tests/run_mcp_list_collection.py`,  
`uv run python tests/run_mcp_rag.py` (for the standalone RAG MCP server),

**cURL:**

```bash
curl -X POST http://localhost:9000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | python -m json.tool
```

---

## Part 2: Consuming MCP (RAG backend and UIs)

The RAG backend and UIs **consume** MCP: they connect to MCP server(s) and attach their tools to the LLM. Configuration is in `.env` (or environment) via `MCP_SERVERS_CONFIG` and related options; see `.env.example`.

### 1. Use one MCP in RAG chat (Next.js app)

- In `.env` (or environment), set `MCP_SERVERS_CONFIG` (JSON) with a `"default"` server URL. Example:

  ```python
  MCP_SERVERS_CONFIG = {
      "default": {
          "transport": "streamable-http",
          "url": "http://localhost:9000/mcp",   # MCP server URL to consume
      },
  }
  ```

- Set `ENABLE_MCP_TOOLS = True`.
- Restart the backend. The Next.js app does not send MCP settings; the backend uses this config.

### 2. Add another MCP as a preset (Next.js / API)

- In `MCP_SERVERS_CONFIG`, add another entry, e.g.:

  ```python
  MCP_SERVERS_CONFIG = {
      "default": { "transport": "streamable-http", "url": "http://localhost:9000/mcp" },
      "context7": { "transport": "streamable-http", "url": "http://localhost:9000/mcp" },
  }
  ```

- **Next.js UI**: In the sidebar, paste the preset URL you want (for a local server, typically `http://localhost:9000/mcp`) and click **Connect / Reload tools**.
- **API note**: MCP-enabled chat is supported through `POST /api/langgraph/threads/{thread_id}/runs` with `mode="mcp"` or `mode="mixed"`.

### 3. Use multiple MCPs in RAG chat at once

#### Which servers and tools load

- **Servers**: `MCP_SERVER_KEYS` (optional) limits which keys from `MCP_SERVERS_CONFIG` are connected when loading tools. The request may also pass `mcp_server_keys` (same idea). This does not choose `mode`; it only filters which MCP endpoints are used.
- **Tools**: Tools come from `langchain_mcp_adapters.MultiServerMCPClient.get_tools()` (see `src/rag_agent/infrastructure/mcp_adapter_runtime.py`). Server names are prefixed on tool names when `tool_name_prefix=True` (e.g. `default.semantic_search`).
- **Large tool lists**: Optional cap via `MCP_TOOL_SELECTION_MAX_TOOLS` and `MCP_TOOL_SELECTION_ALWAYS_INCLUDE`; applied in `get_mcp_answer` / `_apply_mcp_tool_budget` in `mcp_agent.py` after tools load.

- Set which configured servers to load via `MCP_SERVER_KEYS` (optional; if unset, defaults follow `mcp_adapter_runtime._select_server_keys`, typically `"default"` when present).

  ```python
  MCP_SERVER_KEYS = ["default", "context7"]
  ```

- Ensure each key exists in `MCP_SERVERS_CONFIG`. Restart the backend.

### 4. Use an external MCP server (outside this project)

You can point this app at any HTTP MCP server (different repo or machine).

- **Next.js UI**: Use the sidebar MCP settings (no Streamlit app required) to point at the external server, then click **Connect / Reload tools**.
- **Preset in config**: Add an entry to `MCP_SERVERS_CONFIG` (e.g. `"external": { "transport": "streamable-http", "url": "http://YOUR_HOST:PORT/mcp/" }`) and use that URL in the UI or API.
- **Next.js**: The backend uses `MCP_SERVERS_CONFIG["default"]`; set it to your external URL and keep `ENABLE_MCP_TOOLS = True`. The frontend does not send MCP URL.
- **Cursor IDE**: To use an external MCP from Cursor, add it in Cursor Settings → MCP (HTTP URL or stdio command). That is independent of this app’s config.

---

## Configuration (.env / environment)

| Variable             | Used when     | Meaning                                                                                    |
| -------------------- | ------------- | ------------------------------------------------------------------------------------------ |
| `MCP_SERVERS_CONFIG` | **Consuming** | Dict of MCP server names → `{ "transport", "url" }`. Backend and UI connect to these URLs. |
| `MCP_SERVER_KEYS`    | **Consuming** | Optional list of keys from `MCP_SERVERS_CONFIG` to load (default: only `"default"`).       |
| `ENABLE_MCP_TOOLS`   | **Consuming** | If True, RAG chat attaches MCP tools from config; if False, MCP is disabled for chat.      |
| `MCP_SEARCH_MODE`    | **Consuming** | Default semantic-search mode for MCP servers in this repo: `vector`, `hybrid`, or `text`.  |
| `PORT`               | **Exposing**  | Port for this project’s MCP server (e.g. `mcp_semantic_search.py`).                        |
| `HOST`               | **Exposing**  | Listen address for this project’s MCP server.                                              |
| `TRANSPORT`          | **Exposing**  | `"streamable-http"` or `"stdio"` for the server.                                           |

---

## RAG vs MCP flow (mode)

Chat is handled by `ChatRuntimeService` in `api/services/graph_service.py` (no LangGraph graph). Request **`mode`** selects the path:

| API `mode` | Behavior                                                                                                                                                  |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `direct`   | LLM on chat history only; no vector search, no MCP tools.                                                                                                 |
| `rag`      | Vector similarity search + single answer prompt; MCP tools are not loaded.                                                                                |
| `mcp`      | MCP tools only (`get_mcp_answer_async`); tools from `langchain_mcp_adapters`.                                                                             |
| `mixed`    | **`oracle_retrieval`** (same retrieval as RAG) **and** MCP tools together in one tool loop; the model may call retrieval or an MCP tool in the same turn. |

**Follow-up transform:** Before mode dispatch, the service may detect a follow-up that should **reformat** the previous assistant answer (LLM JSON `kind: transform`) and return that answer without running RAG or MCP.

- **Default `mode`** (when not sent): `build_chat_config` in `api/dependencies.py` sets `mixed` when `ENABLE_MCP_TOOLS` is true and `MCP_SERVERS_CONFIG` is non-empty; otherwise `rag`.
- **API**: Send `mode` and optional `mcp_server_keys` to limit which MCP servers load (must match keys in `MCP_SERVERS_CONFIG`).
- **RAG path**: Uses Oracle vector similarity search and a single answer prompt in `ChatRuntimeService`.
- **MCP rounds**: `MCP_MAX_ROUNDS` (default 2) is passed in config; the tool loop in `mcp_agent.py` respects the configured max rounds.

### Testing mixed mode

**From the UI:** In the sidebar, set **Flow mode** to **Mixed (RAG + MCP)**. Send a question; the backend loads both `oracle_retrieval` and configured MCP tools in one agent tool loop, and the model decides which tools to call per turn.

**With curl:** `curl -s -X POST http://localhost:3002/api/langgraph/threads/demo-thread/runs -H "Content-Type: application/json" -d '{"assistant_id":"mcp_agent_executor","input":{"messages":[{"type":"human","content":"What is OCI CLI? Then compute 2+2."}],"mode":"mixed"}}'`

Use `"mode": "mcp"` for MCP tools only, `"mode": "rag"` for retrieval-only, `"mode": "direct"` for no retrieval and no MCP tools. Optionally send `"mcp_server_keys": ["default", "calculator"]` (keys must exist in `MCP_SERVERS_CONFIG`).

---

## Implementation (consuming side)

MCP and mixed chat modes load tools through **`langchain_mcp_adapters.MultiServerMCPClient`** (`src/rag_agent/infrastructure/mcp_adapter_runtime.py`; clients and tool lists are cached per connection set). The tool loop runs in **`src/rag_agent/infrastructure/mcp_agent.py`**, invoked from **`api/services/graph_service.py`**. RAG-only and direct modes do not load MCP tools.

### Flow diagram (high level)

```mermaid
flowchart TD
    A[POST /api/langgraph/threads/{thread_id}/runs] --> C{mode}
    C -->|direct| D[LLM on message history]
    C -->|rag| E[Vector search + answer prompt]
    C -->|mcp| F[MultiServerMCPClient.get_tools + get_mcp_answer_async]
    C -->|mixed| G[oracle_retrieval + MCP tools + get_mcp_answer_async]
    D --> H[Thread state + response]
    E --> H
    F --> H
    G --> H
```

| Path         | When         | Main modules                                                 |
| ------------ | ------------ | ------------------------------------------------------------ |
| **`rag`**    | `mode=rag`   | Oracle VS + `RAG_ANSWER_PROMPT_TEMPLATE` in `graph_service` |
| **`mcp`**    | `mode=mcp`   | `mcp_adapter_runtime` → `get_mcp_answer_async`              |
| **`mixed`**  | `mode=mixed` | `oracle_retrieval` tool + MCP tools → `get_mcp_answer_async`|
| **`direct`** | `mode=direct`| `get_llm().invoke` on history                               |

---

## Common issues

- **404 in browser**: MCP servers are APIs, not web pages. Use the UI or test scripts.
- **Connection refused**: Ensure the MCP server you are **consuming** is running and the URL in `MCP_SERVERS_CONFIG` (or the UI) is correct. If you are **exposing**, check `PORT` and `HOST` in config.
- **Tools not appearing**: Restart the backend after changing `.env`, and confirm `ENABLE_MCP_TOOLS=true`.
- **Wrong URL path**: MCP HTTP servers use the `/mcp` path (example: `http://localhost:9000/mcp`).
- **Import errors**: Activate the virtual environment and install dependencies (e.g. `uv sync`).
- **Database errors**: Check database connection settings in `.env` (used by the semantic search MCP server).

---

## Resources

- [FastMCP Documentation](https://gofastmcp.com/getting-started/welcome)
- [MCP Protocol Specification](https://modelcontextprotocol.io/)
