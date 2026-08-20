# 04 — Preserve mixed evidence and configuration fidelity

**What to build:** Keep mixed-mode retrieval evidence, synthesis configuration, and MCP definition compatibility correct after durable reconstruction.

**Blocked by:** 03 — Reconstruct tool-agent turns through the tool loop.

**Status:** implemented

- [x] Reconstruct only current-turn retrieval evidence and preserve established failure/no-context behavior.
- [x] Reject incompatible recorded MCP configuration instead of silently adopting current settings.
