import { toApiUrl } from "@/lib/api-base";

export type McpServerConfig = {
  key: string;
  transport: "streamable-http" | "sse" | "stdio";
  url: string;
  enabled: boolean;
  auth: McpServerAuthConfig;
};

export type McpAuthType = "none" | "bearer" | "oauth_client_credentials";

export type McpServerAuthConfig = {
  type: McpAuthType;
  bearer_token?: string | null;
  bearer_token_set?: boolean;
  token_url?: string | null;
  client_id?: string | null;
  client_secret?: string | null;
  client_secret_set?: boolean;
  scope?: string | null;
  audience?: string | null;
  grant_type?: string | null;
  refresh_skew_seconds?: number;
};

export type McpServersResponse = {
  enable_mcp_tools: boolean;
  servers: McpServerConfig[];
};

export type McpConnectionTestResponse = {
  key: string;
  ok: boolean;
  tool_count: number;
  tools: Array<{
    name: string;
    description: string;
  }>;
  error: string | null;
};

export async function fetchMcpServers(): Promise<McpServersResponse> {
  const response = await fetch(toApiUrl("/api/config/mcp-servers"), {
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error("Failed to load MCP server settings");
  }
  return (await response.json()) as McpServersResponse;
}

export async function saveMcpServer(
  server: McpServerConfig
): Promise<McpServerConfig> {
  const response = await fetch(
    toApiUrl(`/api/config/mcp-servers/${encodeURIComponent(server.key)}`),
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        transport: server.transport,
        url: server.url,
        enabled: server.enabled,
        auth: server.auth,
      }),
    }
  );
  if (!response.ok) {
    throw new Error("Failed to save MCP server");
  }
  return (await response.json()) as McpServerConfig;
}

export async function setMcpServerEnabled(
  key: string,
  enabled: boolean
): Promise<McpServerConfig> {
  const response = await fetch(
    toApiUrl(`/api/config/mcp-servers/${encodeURIComponent(key)}/enabled`),
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled }),
    }
  );
  if (!response.ok) {
    throw new Error("Failed to update MCP server");
  }
  return (await response.json()) as McpServerConfig;
}

export async function deleteMcpServer(key: string): Promise<void> {
  const response = await fetch(
    toApiUrl(`/api/config/mcp-servers/${encodeURIComponent(key)}`),
    {
      method: "DELETE",
    }
  );
  if (!response.ok) {
    throw new Error("Failed to delete MCP server");
  }
}

export async function testMcpServerConnection(
  server: McpServerConfig
): Promise<McpConnectionTestResponse> {
  const response = await fetch(
    toApiUrl(`/api/config/mcp-servers/${encodeURIComponent(server.key)}/test`),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        transport: server.transport,
        url: server.url,
        enabled: server.enabled,
        auth: server.auth,
      }),
    }
  );
  if (!response.ok) {
    throw new Error("Failed to test MCP server");
  }
  return (await response.json()) as McpConnectionTestResponse;
}
