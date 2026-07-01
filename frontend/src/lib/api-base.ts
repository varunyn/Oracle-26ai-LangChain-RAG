function normalizeBase(base: string): string {
  return base.endsWith("/") ? base.slice(0, -1) : base;
}

export function getClientApiBase(): string {
  if (typeof window === "undefined") {
    return normalizeBase(
      process.env.FASTAPI_BACKEND_URL || "http://localhost:2024"
    );
  }

  const configuredBase = process.env.NEXT_PUBLIC_API_BASE?.trim();
  if (configuredBase) {
    return normalizeBase(configuredBase);
  }

  return "";
}

export function toApiUrl(path: string): string {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  const base = getClientApiBase();
  if (!base) {
    return normalizedPath;
  }
  const withSlash = base.endsWith("/") ? base : `${base}/`;
  return new URL(normalizedPath, withSlash).toString();
}
