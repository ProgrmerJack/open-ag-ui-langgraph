import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  env: {
    NEXT_PUBLIC_COPILOT_PROVIDER:
      process.env.NEXT_PUBLIC_COPILOT_PROVIDER ??
      (process.env.OPENAI_API_KEY
        ? "openai"
        : process.env.GOOGLE_APPLICATION_CREDENTIALS
        ? "google"
        : "none"),
  },
  // Pin the workspace root for Turbopack to avoid mis-detection when multiple lockfiles exist
  // and ensure this app folder is treated as the root.
  // The `turbopack.root` field is supported by Next.js 15+.
  // @ts-ignore - `turbopack` may be missing from older Next.js type definitions
  turbopack: {
    root: __dirname,
  },
};

export default nextConfig;
