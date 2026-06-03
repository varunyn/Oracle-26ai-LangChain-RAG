# Oracle RAG Agent Blog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a repo-local Markdown blog draft for Oracle/OCI developers explaining this app's FastAPI, Oracle AI Vector Search, OCI Generative AI, MCP, and Next.js architecture.

**Architecture:** The blog will be a documentation artifact under `docs/blog/`. It will use the generated hand-drawn architecture image as the lead visual, existing app screenshots as supporting UI evidence, and links to existing setup docs instead of duplicating every setup instruction.

**Tech Stack:** Markdown, local PNG assets, existing repo documentation, FastAPI, Next.js, Oracle AI Vector Search, OCI Generative AI, MCP, Langfuse/OpenTelemetry.

---

### Task 1: Update The Design Spec With Review Feedback

**Files:**
- Modify: `docs/superpowers/specs/2026-06-01-oracle-rag-agent-blog-design.md`

- [ ] **Step 1: Add the generated hand-drawn architecture image as the primary visual**

Add this content under `## Screenshot Plan`:

```markdown
Primary visual:

- `images/oracle-rag-agent-hand-drawn-architecture.png`

Use the hand-drawn architecture image as the main system explainer because it is more distinctive and draws more attention than a plain flow diagram. Use Mermaid only as supporting material when a precise sequence or runtime branch is easier to read as a simple flow.
```

- [ ] **Step 2: Tighten the diagram boundary**

Change the `Include` list so it says:

```markdown
- The hand-drawn architecture image as the lead diagram
- Plain Markdown/Mermaid flow diagrams only where they clarify a specific runtime branch
```

- [ ] **Step 3: Verify the spec contains no placeholders**

Run:

```bash
rg -n "TBD|TODO|placeholder|COPY|FIXME|\\?\\?" docs/superpowers/specs/2026-06-01-oracle-rag-agent-blog-design.md
```

Expected: no matches and exit code 1.

### Task 2: Create The Blog Draft

**Files:**
- Create: `docs/blog/oracle-rag-agent-fastapi-oci-genai.md`

- [ ] **Step 1: Create the article with the approved title and lead image**

Start the file with:

```markdown
# Building an Oracle RAG Agent with FastAPI, Oracle AI Vector Search, OCI Generative AI, and MCP

![Hand-drawn architecture diagram for the Custom Oracle RAG Agent](../../images/oracle-rag-agent-hand-drawn-architecture.png)
```

- [ ] **Step 2: Write the introduction for Oracle/OCI developers**

The opening should state the problem directly: enterprise RAG needs retrieval, citations, runtime mode control, tool use, and observability. Avoid a generic "in this post" opening.

- [ ] **Step 3: Add the architecture section**

Include the real app components:

```markdown
- Next.js chat UI
- FastAPI backend
- `ChatRuntimeService`
- Oracle AI Vector Search through OracleVS-compatible tables
- OCI Generative AI chat, embeddings, and reranking
- MCP tools
- Langfuse and OpenTelemetry
```

- [ ] **Step 4: Link to setup docs without duplicating runnable checkpoints**

Point readers to the deeper operational docs instead of adding inline setup commands:

```markdown
For setup and implementation details, use the project docs:

- [Getting started](../GETTING-STARTED.md)
- [Configuration](../CONFIGURATION.md)
- [Database setup](../DATABASE-SETUP.md)
- [Document population](../DOCUMENT-POPULATION.md)
- [MCP usage](../MCP-USAGE.md)
- [Observability](../OBSERVABILITY.md)
```

- [ ] **Step 5: Add runtime mode explanations**

Document `rag`, `mcp`, `mixed`, and `direct`, and explain that mixed mode exposes Oracle retrieval and MCP tools together before synthesizing a grounded answer when retrieval returns documents.

- [ ] **Step 6: Add UI screenshots**

Reference these local files:

```markdown
![Chat workspace with document upload](../../images/oci-custom-rag-agent-chat-upload-panel.png)
![Processed sources table](../../images/oci-custom-rag-agent-processed-sources-table.png)
![Flow mode selector](../../images/oci-custom-rag-agent-flow-mode-selector.png)
![Model selector](../../images/oci-custom-rag-agent-model-selector.png)
```

- [ ] **Step 7: Add trade-offs and limitations**

Cover:

```markdown
- OCI and Oracle Database setup is real infrastructure work.
- Streaming is a backend/frontend contract.
- MCP configuration should be UI-managed for interactive use, with `.env` as seed/headless support.
- Observability is needed because one answer can include retrieval, model calls, tool calls, reranking, and streaming.
```

### Task 3: Verify The Blog Artifact

**Files:**
- Verify: `docs/blog/oracle-rag-agent-fastapi-oci-genai.md`
- Verify: `images/oracle-rag-agent-hand-drawn-architecture.png`

- [ ] **Step 1: Check all local image references exist**

Run:

```bash
rg -o "\\.\\./\\.\\./images/[^) ]+" docs/blog/oracle-rag-agent-fastapi-oci-genai.md
```

For each output path, confirm the corresponding file exists under `images/`.

- [ ] **Step 2: Check for placeholder language**

Run:

```bash
rg -n "TBD|TODO|placeholder|FIXME|insert screenshot|coming soon" docs/blog/oracle-rag-agent-fastapi-oci-genai.md docs/superpowers/specs/2026-06-01-oracle-rag-agent-blog-design.md
```

Expected: no matches and exit code 1.

- [ ] **Step 3: Review git diff for scope**

Run:

```bash
git diff -- docs/blog/oracle-rag-agent-fastapi-oci-genai.md docs/superpowers/specs/2026-06-01-oracle-rag-agent-blog-design.md images/oracle-rag-agent-hand-drawn-architecture.png
```

Expected: only the blog draft, spec review update, and generated image are included.
