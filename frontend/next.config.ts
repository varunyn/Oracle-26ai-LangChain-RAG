import type { NextConfig } from "next";

const backendUrl = process.env.FASTAPI_BACKEND_URL || "http://localhost:3002";
const langgraphUrl =
  process.env.LANGGRAPH_BACKEND_URL ||
  process.env.NEXT_PUBLIC_LANGGRAPH_API_BASE ||
  "http://localhost:2024";

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
    root: __dirname,
  },
};

export default nextConfig;
