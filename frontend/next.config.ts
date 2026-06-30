import type { NextConfig } from "next";

const langgraphUrl =
  process.env.LANGGRAPH_BACKEND_URL ||
  process.env.NEXT_PUBLIC_LANGGRAPH_API_BASE ||
  "http://localhost:2024";
const backendUrl = process.env.FASTAPI_BACKEND_URL || langgraphUrl;

const nextConfig: NextConfig = {
  output: "standalone",
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${backendUrl}/api/:path*`,
      },
      {
        source: "/langgraph/:path*",
        destination: `${langgraphUrl}/:path*`,
      },
    ];
  },
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "models.dev",
      },
    ],
  },
  turbopack: {
    root: import.meta.dirname,
  },
};

export default nextConfig;
