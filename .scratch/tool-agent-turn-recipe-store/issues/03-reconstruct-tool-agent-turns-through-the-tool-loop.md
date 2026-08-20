# 03 — Reconstruct tool-agent turns through the tool loop

**What to build:** Let MCP and mixed-mode tool execution resume with their original immutable selections while keeping recipes and live turns out of graph state and streams.

**Blocked by:** 01 — Prove durable turn identity and finalization feasibility; 02 — Add saver-owned recipe-store foundation.

**Status:** implemented, verification complete

- [x] Reconstruct a fresh typed turn from a claimed recipe for setup, tool-loop work, and composition.
- [x] Demonstrate interruption/resume with changed request defaults still uses original selections.

Evidence: `tests/workflow_tests/test_tool_agent_turn_reconstruction.py` covers setup-boundary
resume, tool-loop interruption/resume, raw-checkpoint exclusion, native message identity/order,
and incompatible MCP digest rejection. Legacy `runtime.context["tool_agent_turn"]` consumers
were removed from graph code and migrated tests.

Mixed-mode evidence correction: composition reconstructs the latest Oracle retrieval
from the persisted `ToolMessage.artifact`; live evidence remains execution-local and
is never placed in recipe/state/streams. Composition leases release in `finally`.

Final review corrections are covered by the same workflow suite: checkpoint-reloaded
Document artifacts are normalized and bounded to the latest HumanMessage; setup
replays `create_or_load` while tool-loop/composition reconstruction is load-only;
recipe mode, round limit, and synthesis model override changed runtime defaults;
and each LLM/tool side effect renews its fenced lease first.
