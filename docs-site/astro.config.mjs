import mdx from "@astrojs/mdx";
import sitemap from "@astrojs/sitemap";
import reallySimpleDocs from "reallysimpledocs/astro";
import { defineConfig } from "astro/config";

const site = process.env.DOCS_SITE_URL || "https://varuyada.github.io";
const base = process.env.DOCS_BASE_PATH || "/";

export default defineConfig({
  site,
  base,
  output: "static",
  integrations: [
    reallySimpleDocs({
      docsDir: "./docs",
      routeBase: "/",
      style: "vega",
      customCss: ["./src/docs.css"],
      site: {
        title: "Custom RAG Agent",
        subtitle: "Documentation",
        description:
          "Setup, configuration, MCP, observability, and API documentation for the custom RAG agent app.",
        logo: {
          svg: '<svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M5 4h9.5a4.5 4.5 0 0 1 3.18 7.68l-1.34 1.34L20 20h-4.1l-2.8-5.6H8.2V20H5V4Zm3.2 3v4.5h6.05a2.25 2.25 0 0 0 0-4.5H8.2Z"/></svg>',
        },
      },
      components: {
        SidebarHeader: "./src/components/SidebarHeader.astro",
        SidebarFooter: "./src/components/SidebarFooter.astro",
        ContentHeader: "./src/components/ContentHeader.astro",
      },
    }),
    mdx(),
    sitemap(),
  ],
  vite: {
    server: {
      fs: {
        allow: [".."],
      },
    },
  },
});
