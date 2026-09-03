# Oracle Knowledge MCP compatibility

This page records deterministic protocol compatibility for the stable
production line: FastMCP 4.0.2, MCP Python SDK 2.1.1, and protocol
`2026-07-28`. It does not claim live Oracle/OCI or outer-client verification;
those checks belong to Ticket 08.

## Matrix

| Outer client/version | Transport | FastMCP / MCP SDK | Requested → negotiated | Status |
| --- | --- | --- | --- | --- |
| FastMCP Client 4.0.2 | STDIO | 4.0.2 / 2.1.1 | 2026-07-28 → 2026-07-28 | Supported |
| FastMCP Client 4.0.2 | Streamable HTTP | 4.0.2 / 2.1.1 | 2026-07-28 → 2026-07-28 | Supported |
| Codex CLI 0.143.0 | STDIO | 4.0.2 / 2.1.1 | unverified | Unverified |
| Codex CLI 0.143.0 | Streamable HTTP | 4.0.2 / 2.1.1 | unverified | Unverified |
| OpenWebUI 0.11.0 | Streamable HTTP | 4.0.2 / 2.1.1 | unverified | Unverified |
| OpenWebUI 0.11.0 | STDIO | 4.0.2 / 2.1.1 | unsupported | Unsupported |
| MCP Python SDK v2 client | Streamable HTTP | 4.0.2 / 2.1.1 | unverified | Unverified |

The supported rows passed initialization/negotiation, `tools/list`, successful
search, no-hit search, and sanitized backend failure on real STDIO and
Streamable HTTP profiles. Failed or unavailable rows are intentionally not
listed as deployment-supported.

## Verification

```bash
.venv/bin/python -m pytest tests/workflow_tests/test_oracle_knowledge_protocol_compatibility.py -q
.venv/bin/python -m pytest tests/workflow_tests/test_oracle_knowledge_deployment_profiles.py -q
.venv/bin/python tests/run_oracle_knowledge_client_compatibility.py --help
.venv/bin/python tests/run_oracle_knowledge_client_compatibility.py --transport stdio --report /tmp/oracle-knowledge-stdio.json
.venv/bin/python tests/run_oracle_knowledge_client_compatibility.py --transport streamable-http --report /tmp/oracle-knowledge-http.json
```

The manual harness uses deterministic providers and writes its report outside
STDIO stdout. It is suitable as a starting point for Codex/OpenWebUI outer
client checks. Codex CLI 0.143.0 is installed, but this matrix does not claim a
Codex handshake or tool invocation. No OpenWebUI runtime was installed.

## Stateless contract

Retrieval does not depend on protocol session identifiers, initialization-time
mutable state, sampling, roots, elicitation, or load-balancer affinity. STDIO
and network transports are deployment choices, not client-brand assumptions.

References: [FastMCP documentation](https://gofastmcp.com/getting-started/welcome),
[MCP specification](https://modelcontextprotocol.io/), and the
[MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk).
