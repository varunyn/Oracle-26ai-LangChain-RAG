function normalizeBase(base: string): string {
  return base.endsWith("/") ? base.slice(0, -1) : base;
}

export function getClientApiBase(): string {
  const configuredBase = process.env.NEXT_PUBLIC_API_BASE?.trim();
  if (configuredBase) {
    return normalizeBase(configuredBase);
  }

  if (typeof window !== "undefined") {
    return `${window.location.protocol}//${window.location.hostname}:3002`;
  }

  return "http://localhost:3002";
}

export function toApiUrl(path: string): string {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  const base = getClientApiBase();
  const withSlash = base.endsWith("/") ? base : `${base}/`;
  return new URL(normalizedPath, withSlash).toString();
}
