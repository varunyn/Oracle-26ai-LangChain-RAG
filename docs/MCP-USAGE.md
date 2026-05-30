# MCP (Model Context Protocol) Usage

This project supports MCP in two ways:

| Role                       | Description                                                                                                                                                                                                                     |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Exposing an MCP server** | This repo runs MCP servers (e.g. `mcp_servers/mcp_semantic_search.py`) that expose tools (semantic search, list collections). Other clients—Next.js UI or external apps—call these servers.                                     |
| **Consuming MCP**          | The FastAPI backend (called by the Next.js UI) acts as an **MCP client**: it connects to one or more configured MCP server URLs, loads tools from those servers, and lets the LLM use them during chat. |

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
   - Parameters: `query` (required), `top_k` (default: 5), `collection_name` (optional), `search_mode` (optional; only `vector` is currently supported)
2. **`get_collections`** – List all available collections
3. **`list_documents_in_collection`** – List documents in a collection
   - Parameters: `collection_name` (optional)

### Quick start: run the MCP server, then the UI

1. **Start the MCP server** (exposing tools):

   ```bash
    uv run python mcp_servers/mcp_semantic_search.py
   ```

   Server listens on `http://localhost:9000` by default (or `PORT` from config). This is the standalone MCP server runtime. The backend's MCP client configuration is separate and is normally managed from the frontend Settings page.

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
            {"query": "Oracle 23AI", "top_k": 5, "search_mode": "vector"}
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

The RAG backend and UIs **consume** MCP: they connect to MCP server(s) and attach their tools to the LLM. The Next.js Settings page is the primary configuration surface. It writes a server-side config file at `MCP_UI_CONFIG_FILE` so MCPs can be added, edited, disabled, tested, or deleted without editing `.env`.

### 1. Use one MCP in RAG chat (Next.js app)

- Set `ENABLE_MCP_TOOLS = True`.
- Open **Settings** from the chat header.
- Add a server with a stable key such as `default`.
- Set the transport and URL, for example `http://localhost:9000/mcp`.
- Use **Test connection** before saving when possible.

### 2. Optional seed config for headless or first-run setups

`MCP_SERVERS_CONFIG` is still supported as an optional seed or headless deployment config. If the UI config file does not exist yet, servers from this JSON object appear in Settings. After the first UI edit, the backend reads the UI-managed server list from `MCP_UI_CONFIG_FILE`.

```env
MCP_SERVERS_CONFIG={"default":{"transport":"streamable-http","url":"http://localhost:9000/mcp"}}
```

- **API note**: MCP-enabled chat is supported through `POST /api/langgraph/threads/{thread_id}/runs/stream` with `mode="mcp"` or `mode="mixed"`.

### 3. Use multiple MCPs in RAG chat at once

#### Which servers and tools load

- **Servers**: `MCP_SERVER_KEYS` (optional) limits which configured server keys are connected when loading tools. The request may also pass `mcp_server_keys` (same idea). This does not choose `mode`; it only filters which MCP endpoints are used.
- **Tools**: Tools come from `langchain_mcp_adapters.MultiServerMCPClient.get_tools()` (see `src/rag_agent/infrastructure/mcp_adapter_runtime.py`). Server names are prefixed on tool names when `tool_name_prefix=True` (e.g. `default.semantic_search`).

- Set which configured servers to load via `MCP_SERVER_KEYS` (optional; if unset, defaults follow `mcp_adapter_runtime._select_server_keys`, typically `"default"` when present).

  ```python
  MCP_SERVER_KEYS = ["default", "context7"]
  ```

- Ensure each key exists in Settings or in optional seed config. Restart the backend only after changing `.env`.

### 4. Use an external MCP server (outside this project)

You can point this app at any HTTP MCP server (different repo or machine).

- **Next.js UI**: Open **Settings**, add an MCP server with a stable key such as `external`, set the URL, and keep it enabled.
- **Preset in config**: Add an entry to `MCP_SERVERS_CONFIG` only if you want it to appear before any UI-managed override exists or you run without the frontend.
- **Next.js**: Keep `ENABLE_MCP_TOOLS = True`. The frontend manages MCP entries through backend config endpoints; chat requests still choose `mode="mcp"` or `mode="mixed"`.
- **Cursor IDE**: To use an external MCP from Cursor, add it in Cursor Settings → MCP (HTTP URL or stdio command). That is independent of this app’s config.

### OAuth-protected MCP servers

If your MCP server uses OAuth client credentials, configure it in **Settings** by selecting the server's **Auth mechanism** and entering the token URL, client ID, client secret, scope, audience, and grant type as needed. The client secret is stored server-side and is not returned to the browser after save.

You can also seed the same auth configuration from `.env` for headless or first-run setups:

```env
ENABLE_MCP_TOOLS=true
MCP_SERVERS_CONFIG={"default":{"transport":"streamable-http","url":"https://your-mcp-host/mcp","auth":{"type":"oauth_client_credentials","token_url":"https://auth.example.com/oauth/token","client_id":"your-client-id","client_secret":"your-client-secret","scope":"read:mcp"}}}
```

The backend fetches and refreshes the bearer token automatically and injects it into MCP requests.

---

## Configuration (.env / environment)

| Variable             | Used when     | Meaning                                                                                    |
| -------------------- | ------------- | ------------------------------------------------------------------------------------------ |
| `MCP_SERVERS_CONFIG` | **Consuming** | Optional seed/headless dict of MCP server names → `{ "transport", "url", "auth" }`. Settings is the primary UI-managed source. |
| `MCP_SERVER_KEYS`    | **Consuming** | Optional list of configured server keys to load.       |
| `ENABLE_MCP_TOOLS`   | **Consuming** | If True, RAG chat attaches MCP tools from config; if False, MCP is disabled for chat.      |
| `MCP_SEARCH_MODE`    | **Consuming** | Default semantic-search mode for MCP servers in this repo. Only `vector` is currently supported. |
| `PORT`               | **Exposing**  | Port for this project’s MCP server (e.g. `mcp_semantic_search.py`).                        |
| `HOST`               | **Exposing**  | Listen address for this project’s MCP server.                                              |
| `TRANSPORT`          | **Exposing**  | `"streamable-http"` or `"stdio"` for the server.                                           |

---

## RAG vs MCP flow (mode)

Chat is handled by `ChatRuntimeService` in `src/rag_agent/runtime/chat_service.py` (no LangGraph graph). Request **`mode`** selects the path:

| API `mode` | Behavior                                                                                                                                                  |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `direct`   | LLM on chat history only; no vector search, no MCP tools.                                                                                                 |
| `rag`      | Vector similarity search + single answer prompt; MCP tools are not loaded.                                                                                |
| `mcp`      | MCP tools only (`get_mcp_answer_async`); tools from `langchain_mcp_adapters`.                                                                             |
| `mixed`    | MCP tools plus the local **`oracle_retrieval`** tool. If a turn only needs retrieval, the runtime stops the agent turn after retrieval and streams the final answer through the RAG answer path. If the user explicitly requests another MCP tool/action, the agent may continue after retrieval before the RAG answer is synthesized. Non-retrieval MCP turns still use the MCP agent answer. |

**Follow-up transform:** Before mode dispatch, the service may detect a follow-up that should **reformat** the previous assistant answer (LLM JSON `kind: transform`) and return that answer without running RAG or MCP.

- **Default `mode`** (when not sent): `build_chat_config` in `api/dependencies.py` sets `mixed` when `ENABLE_MCP_TOOLS` is true and at least one MCP server is configured; otherwise `rag`.
- **API**: Send `mode` and optional `mcp_server_keys` to limit which MCP servers load.
- **RAG path**: Uses Oracle vector similarity search and a single answer prompt in `ChatRuntimeService`.
- **Mixed RAG handoff**: When `oracle_retrieval` returns docs, the final answer is synthesized by the RAG runtime so answer text can stream immediately after retrieval. If the request also explicitly references a non-RAG MCP tool, the runtime lets the MCP agent finish that tool work before the RAG synthesis step. Retrieval infrastructure errors are surfaced as retrieval failures, not as "no documents found."
- **MCP tool-call limit**: `MCP_MAX_ROUNDS` (default 4) caps total tool calls in one agent run. Mixed mode may need more than two calls when a turn combines Oracle retrieval with MCP tool actions.

### Testing mixed mode

**From the UI:** In the sidebar, set **Flow mode** to **Mixed (RAG + MCP)**. Send a question; the backend loads `oracle_retrieval` and configured MCP tools. If the turn uses retrieval and finds docs, the answer streams from the RAG answer path. If the user also asks for a non-RAG tool action, that tool result is included as supplemental context for the RAG answer. If the turn uses only MCP tools, the MCP agent answer is returned.

**With curl:** `curl -N -X POST http://localhost:3002/api/langgraph/threads/demo-thread/runs/stream -H "Content-Type: application/json" -d '{"assistant_id":"mcp_agent_executor","input":{"messages":[{"type":"human","content":"What is OCI CLI? Then compute 2+2."}],"mode":"mixed"}}'`

Use `"mode": "mcp"` for MCP tools only, `"mode": "rag"` for retrieval-only, `"mode": "direct"` for no retrieval and no MCP tools. Optionally send `"mcp_server_keys": ["default", "calculator"]` for configured server keys.

---

## Implementation (consuming side)

MCP and mixed chat modes load tools through **`langchain_mcp_adapters.MultiServerMCPClient`** (`src/rag_agent/infrastructure/mcp_adapter_runtime.py`; clients and tool lists are cached per connection set). The MCP tool loop runs through **`src/rag_agent/infrastructure/mcp_agent.py`**, invoked from **`src/rag_agent/runtime/chat_service.py`**. In mixed mode, `ChatRuntimeService` may stop the loop after `oracle_retrieval` and hand retrieved docs to the RAG runtime for answer synthesis. RAG-only and direct modes do not load MCP tools.

### Flow diagram (high level)

```mermaid
flowchart TD
    A[POST /api/langgraph/threads/{thread_id}/runs/stream] --> C{mode}
    C -->|direct| D[LLM on message history]
    C -->|rag| E[Vector search + answer prompt]
    C -->|mcp| F[MultiServerMCPClient.get_tools + get_mcp_answer_async]
    C -->|mixed| G[oracle_retrieval + MCP tools]
    G --> I{retrieval docs?}
    I -->|yes| J[RAG answer synthesis]
    I -->|no| K[MCP agent answer or retrieval error]
    D --> H[Thread state + response]
    E --> H
    F --> H
    J --> H
    K --> H
```

| Path         | When         | Main modules                                                 |
| ------------ | ------------ | ------------------------------------------------------------ |
| **`rag`**    | `mode=rag`   | Oracle VS + `RAG_ANSWER_PROMPT_TEMPLATE` in `chat_service` |
| **`mcp`**    | `mode=mcp`   | `mcp_adapter_runtime` → `get_mcp_answer_async`              |
| **`mixed`**  | `mode=mixed` | `oracle_retrieval` tool + MCP tools → RAG answer synthesis when docs are found; otherwise MCP agent answer or retrieval error |
| **`direct`** | `mode=direct`| `get_llm().invoke` on history                               |

---

## Common issues

- **404 in browser**: MCP servers are APIs, not web pages. Use the UI or test scripts.
- **Connection refused**: Ensure the MCP server you are **consuming** is running and the URL in Settings or optional seed config is correct. If you are **exposing**, check `PORT` and `HOST` in config.
- **Tools not appearing**: Confirm `ENABLE_MCP_TOOLS=true`, add or enable the server in Settings, and use **Test connection**. Restart the backend only after changing `.env`.
- **Wrong URL path**: MCP HTTP servers use the `/mcp` path (example: `http://localhost:9000/mcp`).
- **Import errors**: Activate the virtual environment and install dependencies (e.g. `uv sync`).
- **Database errors**: Check database connection settings in `.env` (used by the semantic search MCP server).

---

## Resources

- [FastMCP Documentation](https://gofastmcp.com/getting-started/welcome)
- [MCP Protocol Specification](https://modelcontextprotocol.io/)
