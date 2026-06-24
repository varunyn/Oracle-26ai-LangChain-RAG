import { readdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const docsSiteRoot = dirname(here);
const distRoot = join(docsSiteRoot, "dist");
const basePath = normalizeBasePath(process.env.DOCS_BASE_PATH || "/custom-rag-agent-app");

function normalizeBasePath(value) {
  const clean = String(value || "").trim().replace(/^\/+|\/+$/g, "");
  return clean ? `/${clean}` : "";
}

async function listFiles(dir) {
  const entries = await readdir(dir, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await listFiles(path)));
    } else if (entry.isFile()) {
      files.push(path);
    }
  }
  return files;
}

function prefixRootPath(path) {
  if (!basePath) return path;
  if (!path.startsWith("/")) return path;
  if (path.startsWith(`${basePath}/`) || path === basePath) return path;
  if (path.startsWith("/_astro/")) return path;
  return `${basePath}${path}`;
}

function rewriteText(text) {
  if (!basePath) return text;
  return text
    .replace(/(href|src|content|data-index-url|data-md-url)="\/(?!\/|custom-rag-agent-app\/|_astro\/)([^"#?]*)([?#][^"]*)?"/g, (_, attr, path, suffix = "") => {
      return `${attr}="${prefixRootPath(`/${path}`)}${suffix}"`;
    })
    .replace(/"path":"\/(?!\/|custom-rag-agent-app\/|_astro\/)([^"]*)"/g, (_, path) => {
      return `"path":"${prefixRootPath(`/${path}`)}"`;
    })
    .replace(/"markdownPath":"\/(?!\/|custom-rag-agent-app\/|_astro\/)([^"]*)"/g, (_, path) => {
      return `"markdownPath":"${prefixRootPath(`/${path}`)}"`;
    });
}

const files = await listFiles(distRoot);
let changed = 0;

for (const file of files) {
  if (!/\.(html|json|js|txt)$/.test(file)) continue;
  const before = await readFile(file, "utf8");
  const after = rewriteText(before);
  if (after !== before) {
    await writeFile(file, after, "utf8");
    changed += 1;
  }
}

console.log(`Adjusted GitHub Pages base paths in ${changed} built file(s).`);
