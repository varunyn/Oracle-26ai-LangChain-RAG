import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
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
