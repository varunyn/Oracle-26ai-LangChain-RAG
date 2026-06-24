import { cp, mkdir, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const docsSiteRoot = dirname(here);
const repoRoot = dirname(docsSiteRoot);
const sourceRoot = join(repoRoot, "docs");
const targetRoot = join(docsSiteRoot, "docs");

const skippedNames = new Set(["_navbar.md", "_sidebar.md"]);

const navigation = [
  {
    type: "group",
    label: "Getting started",
    items: [
      { slug: "index", icon: "rocket" },
      { slug: "overview", icon: "book-open" },
      { slug: "configuration", icon: "settings" },
      { slug: "docker-setup", icon: "box" },
    ],
  },
  {
    type: "group",
    label: "Runtime",
    items: [
      { slug: "chat-streaming-protocol", icon: "radio" },
      { slug: "chat-memory-and-sessions", icon: "messages-square" },
      { slug: "server-owned-memory", icon: "database" },
      { slug: "mcp-usage", icon: "plug" },
    ],
  },
  {
    type: "group",
    label: "Data and operations",
    items: [
      { slug: "database-setup", icon: "table-2" },
      { slug: "document-population", icon: "file-plus-2" },
      { slug: "oci-session-token", icon: "key-round" },
      { slug: "tracing", icon: "activity" },
      { slug: "observability", icon: "line-chart" },
      { slug: "observability-routing", icon: "route" },
      { slug: "logging-analytics", icon: "scroll-text" },
    ],
  },
  {
    type: "group",
    label: "API",
    items: [
      { slug: "api/index", label: "API overview", icon: "braces" },
      {
        type: "submenu",
        label: "Endpoints",
        icon: "list-tree",
        items: [
          { slug: "api/00-overview/index", label: "Generated overview" },
          { slug: "api/10-health/index", label: "Health" },
          { slug: "api/20-chat/index", label: "Chat" },
          { slug: "api/30-documents/index", label: "Documents" },
          { slug: "api/60-config-suggestions-feedback/index", label: "Config and feedback" },
          { slug: "api/generated/endpoints", label: "Endpoint reference" },
          { slug: "api/generated/schemas", label: "Schema reference" },
        ],
      },
    ],
  },
];

async function listMarkdownFiles(dir) {
  const entries = await readdir(dir, { withFileTypes: true });
  const files = [];

  for (const entry of entries) {
    const absolutePath = join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await listMarkdownFiles(absolutePath)));
    } else if (entry.isFile() && entry.name.endsWith(".md") && !skippedNames.has(entry.name)) {
      files.push(absolutePath);
    }
  }

  return files;
}

function slugFor(relativePath) {
  if (relativePath === "GETTING-STARTED.md") return "index.md";
  if (relativePath === "README.md") return "overview.md";
  if (relativePath.endsWith("/README.md")) {
    return relativePath.replace(/README\.md$/, "index.md").toLowerCase().replace(/_/g, "-");
  }
  return relativePath.toLowerCase().replace(/_/g, "-");
}

function rewriteMarkdownLinks(markdown) {
  return markdown
    .replace(/\]\(docs\/GETTING-STARTED\.md\)/g, "](/)")
    .replace(/\]\(docs\/README\.md\)/g, "](/overview/)")
    .replace(/\]\(docs\/([A-Z0-9_-]+)\.md\)/g, (_, name) => `](/${name.toLowerCase().replace(/_/g, "-")}/)`)
    .replace(/\]\((GETTING-STARTED)\.md\)/g, "](/)")
    .replace(/\]\((README)\.md\)/g, "](/overview/)")
    .replace(/\]\(([A-Z0-9_-]+)\.md\)/g, (_, name) => `](/${name.toLowerCase().replace(/_/g, "-")}/)`);
}

await rm(targetRoot, { force: true, recursive: true });
await mkdir(targetRoot, { recursive: true });

const files = await listMarkdownFiles(sourceRoot);

for (const source of files) {
  const relativePath = relative(sourceRoot, source);
  const destination = join(targetRoot, slugFor(relativePath));
  await mkdir(dirname(destination), { recursive: true });
  const markdown = await readFile(source, "utf8");
  await writeFile(destination, rewriteMarkdownLinks(markdown), "utf8");
}

await writeFile(join(targetRoot, "docs.json"), `${JSON.stringify({ menu: navigation }, null, 2)}\n`, "utf8");

console.log(`Synced ${files.length} markdown files from docs/ to docs-site/docs/.`);
