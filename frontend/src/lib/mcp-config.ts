import { toApiUrl } from "@/lib/api-base";

export type McpServerConfig = {
  key: string;
  transport: "streamable-http" | "sse" | "stdio";
  url: string;
  enabled: boolean;
};

export type McpServersResponse = {
  enable_mcp_tools: boolean;
  servers: McpServerConfig[];
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

export async function saveMcpServer(server: McpServerConfig): Promise<McpServerConfig> {
  const response = await fetch(
    toApiUrl(`/api/config/mcp-servers/${encodeURIComponent(server.key)}`),
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        transport: server.transport,
        url: server.url,
        enabled: server.enabled,
      }),
    },
  );
  if (!response.ok) {
    throw new Error("Failed to save MCP server");
  }
  return (await response.json()) as McpServerConfig;
}

export async function setMcpServerEnabled(
  key: string,
  enabled: boolean,
): Promise<McpServerConfig> {
  const response = await fetch(
    toApiUrl(`/api/config/mcp-servers/${encodeURIComponent(key)}/enabled`),
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled }),
    },
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
    },
  );
  if (!response.ok) {
    throw new Error("Failed to delete MCP server");
  }
}
