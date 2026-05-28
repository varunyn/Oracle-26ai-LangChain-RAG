import type { NextConfig } from "next";

const backendUrl = process.env.FASTAPI_BACKEND_URL || "http://localhost:3002";

const nextConfig: NextConfig = {
  output: "standalone",
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${backendUrl}/api/:path*`,
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
