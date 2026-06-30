# Docs site AGENTS.md

This guide applies only to `docs-site/`. It is the Astro/ReallySimpleDocs render layer; `docs/` remains the source-of-truth Markdown tree.

## Workflow

- Sync source content before inspecting or building: `npm run sync`.
- Local site: `npm run dev` (uses `DOCS_BASE_PATH=/`).
- Production-style build: `npm run build` (defaults to `/custom-rag-agent-app`, override with `DOCS_BASE_PATH`).
- Preview a built site: `npm run preview`.

## Boundaries

- Edit documentation content in `docs/`, not generated `docs-site/docs/` output.
- Keep Astro layouts, components, styles, and base-path handling in `docs-site/src/`.
- Preserve GitHub Pages project-site routing; local development uses `/`, while builds use `/custom-rag-agent-app` unless overridden.
- Do not commit `docs-site/dist/`, `.astro/`, `node_modules/`, or secrets.
- Use npm only for this Astro site; the application frontend under `frontend/` uses pnpm.

See the root `AGENTS.md` for repo-wide verification and changelog rules.
