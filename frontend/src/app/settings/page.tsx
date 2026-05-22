"use client";

import Link from "next/link";
import {
  ArrowLeft,
  BarChart3,
  CircleCheck,
  CircleX,
  ExternalLink,
  Plus,
  RefreshCw,
  Save,
  Server,
  SquareActivity,
  Trash2,
} from "lucide-react";
import { memo, useCallback, useEffect, useMemo, useState } from "react";

import { useToast } from "@/components/toaster";
import {
  fetchAppConfig,
  type AppConfig,
  type ObservabilityLink,
} from "@/lib/config";
import {
  deleteMcpServer,
  fetchMcpServers,
  saveMcpServer,
  setMcpServerEnabled,
  testMcpServerConnection,
  type McpAuthType,
  type McpConnectionTestResponse,
  type McpServerAuthConfig,
  type McpServerConfig,
} from "@/lib/mcp-config";

type DraftServer = McpServerConfig & {
  isNew?: boolean;
};

const emptyDraft: DraftServer = {
  key: "",
  transport: "streamable-http",
  url: "",
  enabled: true,
  auth: {
    type: "none",
    grant_type: "client_credentials",
    refresh_skew_seconds: 30,
  },
  isNew: true,
};

const observabilityIconClass =
  "inline-flex size-9 shrink-0 items-center justify-center rounded-md border border-border bg-background text-muted-foreground";

const iconByObservabilityKey: Record<string, typeof BarChart3> = {
  grafana: BarChart3,
  langfuse: SquareActivity,
  oracle_logging_analytics: Server,
};

function cloneServer(server: McpServerConfig): DraftServer {
  return {
    ...server,
    auth: {
      ...server.auth,
      type: server.auth?.type ?? "none",
      grant_type: server.auth?.grant_type ?? "client_credentials",
      refresh_skew_seconds: server.auth?.refresh_skew_seconds ?? 30,
    },
  };
}

function normalizeDraftAuth(auth: Partial<McpServerAuthConfig>): McpServerAuthConfig {
  return {
    type: auth.type ?? "none",
    grant_type: auth.grant_type ?? "client_credentials",
    refresh_skew_seconds: auth.refresh_skew_seconds ?? 30,
    bearer_token: auth.bearer_token ?? null,
    token_url: auth.token_url ?? null,
    client_id: auth.client_id ?? null,
    client_secret: auth.client_secret ?? null,
    scope: auth.scope ?? null,
    audience: auth.audience ?? null,
  };
}

function observabilityStatusClass(link: ObservabilityLink): string {
  if (link.configured && link.url) {
    return "border-primary/25 bg-primary/[0.08] text-primary";
  }
  if (link.configured) {
    return "border-foreground/15 bg-muted text-foreground";
  }
  if (link.enabled) {
    return "border-destructive/30 bg-destructive/10 text-destructive";
  }
  return "border-border bg-muted text-muted-foreground";
}

type ServerListItemProps = {
  server: McpServerConfig;
  active: boolean;
  pending: boolean;
  onSelect: (server: McpServerConfig) => void;
  onToggle: (server: McpServerConfig) => void;
};

const ServerListItem = memo(function ServerListItem({
  server,
  active,
  pending,
  onSelect,
  onToggle,
}: ServerListItemProps): React.ReactElement {
  return (
    <div
      className={`grid grid-cols-[minmax(0,1fr)_3.25rem] items-center gap-3 px-3 py-3 ${
        active
          ? "bg-primary/[0.08] ring-1 ring-inset ring-primary/25"
          : "bg-card"
      }`}
    >
      <button
        type="button"
        onClick={() => onSelect(server)}
        aria-current={active ? "true" : undefined}
        className={`flex min-w-0 flex-1 items-center gap-3 rounded-md px-2 py-2 text-left transition-colors focus:outline-none focus:ring-2 focus:ring-ring ${
          active ? "hover:bg-primary/[0.08]" : "hover:bg-muted"
        }`}
      >
        <span
          className={`inline-flex size-8 shrink-0 items-center justify-center rounded-md border ${
            server.enabled
              ? "border-foreground/20 bg-foreground text-background"
              : "border-border bg-background text-muted-foreground"
          }`}
        >
          <Server className="size-4" aria-hidden />
        </span>
        <span className="min-w-0">
          <span className="block truncate text-sm font-medium text-foreground">
            {server.key}
          </span>
          <span className="block truncate text-xs text-muted-foreground">
            {server.url}
          </span>
        </span>
      </button>
      <button
        type="button"
        onClick={() => void onToggle(server)}
        disabled={pending}
        className={`relative inline-flex h-8 w-14 shrink-0 items-center rounded-full border transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60 ${
          server.enabled ? "border-foreground bg-foreground" : "border-border bg-muted"
        }`}
        aria-label={server.enabled ? "Disable MCP server" : "Enable MCP server"}
        title={server.enabled ? "Disable" : "Enable"}
      >
        <span
          className={`absolute left-0.5 top-1/2 size-5 -translate-y-1/2 rounded-full bg-background shadow-sm transition-transform ${
            server.enabled ? "translate-x-6" : "translate-x-0"
          }`}
        />
      </button>
    </div>
  );
});

type ObservabilityLinkItemProps = {
  link: ObservabilityLink;
};

const ObservabilityLinkItem = memo(function ObservabilityLinkItem({
  link,
}: ObservabilityLinkItemProps): React.ReactElement {
  const Icon = iconByObservabilityKey[link.key] ?? BarChart3;
  return (
    <div className="grid min-h-[11rem] content-between gap-4 px-5 py-5">
      <div className="flex min-w-0 items-start gap-3">
        <span className={observabilityIconClass}>
          <Icon className="size-4" aria-hidden />
        </span>
        <div className="min-w-0">
          <h3 className="truncate text-sm font-semibold text-foreground">
            {link.label}
          </h3>
          <p className="mt-1 text-xs leading-5 text-muted-foreground">{link.details}</p>
        </div>
      </div>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <span
          className={`inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-medium ${observabilityStatusClass(
            link,
          )}`}
        >
          {link.status}
        </span>
        {link.url ? (
          <a
            href={link.url}
            target="_blank"
            rel="noreferrer"
            className="inline-flex min-h-10 items-center gap-2 rounded-md border border-border bg-background px-3 py-2 text-sm font-medium text-foreground transition-colors hover:bg-muted focus:outline-none focus:ring-2 focus:ring-ring"
          >
            Open
            <ExternalLink className="size-4" aria-hidden />
          </a>
        ) : (
          <span className="text-xs text-muted-foreground">No link</span>
        )}
      </div>
    </div>
  );
});

export default function SettingsPage(): React.ReactElement {
  const { toast } = useToast();
  const [servers, setServers] = useState<McpServerConfig[]>([]);
  const [appConfig, setAppConfig] = useState<AppConfig | null>(null);
  const [draft, setDraft] = useState<DraftServer>(emptyDraft);
  const [selectedKey, setSelectedKey] = useState<string>("");
  const [mcpGloballyEnabled, setMcpGloballyEnabled] = useState(true);
  const [loading, setLoading] = useState(true);
  const [configLoading, setConfigLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<McpConnectionTestResponse | null>(null);
  const [pendingToggleKeys, setPendingToggleKeys] = useState<Set<string>>(
    () => new Set(),
  );

  const selectedServer = useMemo(
    () => servers.find((server) => server.key === selectedKey),
    [selectedKey, servers],
  );

  const loadServers = useCallback(async () => {
    setLoading(true);
    setConfigLoading(true);
    const [mcpResult, configResult] = await Promise.allSettled([
      fetchMcpServers(),
      fetchAppConfig(),
    ]);

    if (mcpResult.status === "fulfilled") {
      setServers(mcpResult.value.servers);
      setMcpGloballyEnabled(mcpResult.value.enable_mcp_tools);
      const first = mcpResult.value.servers[0];
      if (first) {
        setSelectedKey(first.key);
        setDraft(cloneServer(first));
      } else {
        setSelectedKey("");
        setDraft(emptyDraft);
      }
    } else {
      console.error(mcpResult.reason);
      toast.error("Could not load MCP settings");
    }

    if (configResult.status === "fulfilled") {
      setAppConfig(configResult.value);
    } else {
      console.error(configResult.reason);
      toast.error("Could not load observability settings");
    }

    setLoading(false);
    setConfigLoading(false);
  }, [toast]);

  useEffect(() => {
    void loadServers();
  }, [loadServers]);

  const selectServer = useCallback((server: McpServerConfig) => {
    setSelectedKey(server.key);
    setDraft(cloneServer(server));
    setTestResult(null);
  }, []);

  const startNewServer = useCallback(() => {
    setSelectedKey("");
    setDraft(emptyDraft);
    setTestResult(null);
  }, []);

  const updateDraft = useCallback((patch: Partial<DraftServer>) => {
    setDraft((current) => ({ ...current, ...patch }));
    setTestResult(null);
  }, []);

  const updateDraftAuth = useCallback((patch: Partial<McpServerAuthConfig>) => {
    setDraft((current) => ({
      ...current,
      auth: normalizeDraftAuth({ ...current.auth, ...patch }),
    }));
    setTestResult(null);
  }, []);

  const updateAuthType = useCallback(
    (type: McpAuthType) => {
      updateDraftAuth({ type });
    },
    [updateDraftAuth],
  );

  const handleSave = useCallback(async () => {
    if (pendingToggleKeys.size > 0) {
      toast.error("Wait for the enable change to finish before saving");
      return;
    }
    const key = draft.key.trim();
    const url = draft.url.trim();
    if (!key || !url) {
      toast.error("Server key and URL are required");
      return;
    }
    setSaving(true);
    try {
      const saved = await saveMcpServer({ ...draft, key, url });
      setServers((current) => {
        const exists = current.some((server) => server.key === saved.key);
        if (!exists) return [...current, saved];
        return current.map((server) => (server.key === saved.key ? saved : server));
      });
      setSelectedKey(saved.key);
      setDraft(cloneServer(saved));
      setTestResult(null);
      toast.success("MCP server saved");
    } catch (error) {
      console.error(error);
      toast.error("Could not save MCP server");
    } finally {
      setSaving(false);
    }
  }, [draft, pendingToggleKeys.size, toast]);

  const handleToggle = useCallback(
    async (server: McpServerConfig) => {
      setPendingToggleKeys((current) => new Set(current).add(server.key));
      try {
        const updated = await setMcpServerEnabled(server.key, !server.enabled);
        setServers((current) =>
          current.map((item) => (item.key === updated.key ? updated : item)),
        );
        setDraft((current) => (current.key === updated.key ? cloneServer(updated) : current));
        setTestResult(null);
      } catch (error) {
        console.error(error);
        toast.error("Could not update MCP server");
      } finally {
        setPendingToggleKeys((current) => {
          const next = new Set(current);
          next.delete(server.key);
          return next;
        });
      }
    },
    [toast],
  );

  const handleDelete = useCallback(
    async (key: string) => {
      try {
        await deleteMcpServer(key);
        const remaining = servers.filter((server) => server.key !== key);
        setServers(remaining);
        const next = remaining[0];
        if (next) {
          setSelectedKey(next.key);
          setDraft(cloneServer(next));
        } else {
          setSelectedKey("");
          setDraft(emptyDraft);
        }
        setTestResult(null);
        toast.success("MCP server deleted");
      } catch (error) {
        console.error(error);
        toast.error("Could not delete MCP server");
      }
    },
    [servers, toast],
  );

  const handleTestConnection = useCallback(async () => {
    const key = draft.key.trim();
    const url = draft.url.trim();
    if (!key || !url) {
      toast.error("Server key and URL are required before testing");
      return;
    }
    setTesting(true);
    setTestResult(null);
    try {
      const result = await testMcpServerConnection({ ...draft, key, url });
      setTestResult(result);
      if (result.ok) {
        toast.success(`Connection succeeded: ${result.tool_count} tools found`);
      } else {
        toast.error("Connection failed");
      }
    } catch (error) {
      console.error(error);
      toast.error("Could not test MCP server");
    } finally {
      setTesting(false);
    }
  }, [draft, toast]);

  const observabilityLinks = appConfig?.observability?.links ?? [];

  return (
    <main className="flex h-screen min-h-0 flex-col overflow-hidden bg-muted/20">
      <header className="shrink-0 border-b border-border bg-card px-5 py-4 shadow-sm sm:px-8">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4">
          <div className="flex min-w-0 items-center gap-3">
            <Link
              href="/"
              className="inline-flex size-10 shrink-0 items-center justify-center rounded-md border border-border bg-background text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
              aria-label="Back to chat"
              title="Back to chat"
            >
              <ArrowLeft className="size-4" aria-hidden />
            </Link>
            <div className="min-w-0">
              <h1 className="truncate text-xl font-semibold tracking-tight text-foreground">
                Settings
              </h1>
              <p className="truncate text-sm text-muted-foreground">
                MCP servers and observability
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={() => void loadServers()}
            className="inline-flex min-h-10 items-center gap-2 rounded-md border border-border bg-background px-3 py-2 text-sm font-medium text-foreground transition-colors hover:bg-muted focus:outline-none focus:ring-2 focus:ring-ring"
          >
            <RefreshCw className="size-4" aria-hidden />
            Refresh
          </button>
        </div>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto px-5 py-6 sm:px-8">
        <div className="mx-auto grid max-w-6xl gap-5 lg:grid-cols-[minmax(18rem,24rem)_1fr]">
          <section className="min-h-[28rem] rounded-lg border border-border bg-card">
            <div className="flex items-center justify-between gap-3 border-b border-border px-4 py-3">
              <div>
                <h2 className="text-sm font-semibold text-foreground">MCP servers</h2>
                <p className="text-xs text-muted-foreground">
                  {mcpGloballyEnabled ? "MCP tools are enabled" : "MCP tools are disabled"}
                </p>
              </div>
              <button
                type="button"
                onClick={startNewServer}
                className="inline-flex size-10 items-center justify-center rounded-md border border-border bg-background text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                aria-label="Add MCP server"
                title="Add MCP server"
              >
                <Plus className="size-4" aria-hidden />
              </button>
            </div>
            <div className="divide-y divide-border">
              {loading ? (
                <div className="grid gap-3 px-4 py-5" aria-busy="true">
                  {[0, 1, 2].map((item) => (
                    <div key={item} className="flex items-center gap-3">
                      <span className="size-8 rounded-md bg-muted" />
                      <span className="min-w-0 flex-1">
                        <span className="mb-2 block h-3 w-24 rounded bg-muted" />
                        <span className="block h-3 w-full max-w-52 rounded bg-muted" />
                      </span>
                    </div>
                  ))}
                </div>
              ) : servers.length === 0 ? (
                <div className="px-4 py-8 text-sm text-muted-foreground">
                  No MCP servers configured.
                </div>
              ) : (
                servers.map((server) => {
                  const active = server.key === selectedKey;
                  return (
                    <ServerListItem
                      key={server.key}
                      server={server}
                      active={active}
                      pending={pendingToggleKeys.has(server.key)}
                      onSelect={selectServer}
                      onToggle={handleToggle}
                    />
                  );
                })
              )}
            </div>
          </section>

          <section className="rounded-lg border border-border bg-card">
            <div className="border-b border-border px-5 py-4">
              <h2 className="text-sm font-semibold text-foreground">
                {selectedServer ? "Edit server" : "Add server"}
              </h2>
            </div>
            <div className="grid gap-5 px-5 py-5">
              <label className="grid gap-1.5">
                <span className="text-sm font-medium text-foreground">Key</span>
                <input
                  type="text"
                  value={draft.key}
                  onChange={(event) => updateDraft({ key: event.target.value })}
                  className="rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                  placeholder="default"
                />
              </label>
              <label className="grid gap-1.5">
                <span className="text-sm font-medium text-foreground">Transport</span>
                <select
                  value={draft.transport}
                  onChange={(event) =>
                    updateDraft({
                      transport: event.target.value as McpServerConfig["transport"],
                    })
                  }
                  className="rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                >
                  <option value="streamable-http">streamable-http</option>
                  <option value="sse">sse</option>
                  <option value="stdio">stdio</option>
                </select>
              </label>
              <label className="grid gap-1.5">
                <span className="text-sm font-medium text-foreground">URL</span>
                <input
                  type="text"
                  value={draft.url}
                  onChange={(event) => updateDraft({ url: event.target.value })}
                  className="rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                  placeholder="http://localhost:9000/mcp"
                />
              </label>
              <div className="grid gap-4 rounded-md border border-border bg-muted/20 px-3 py-3">
                <label className="grid gap-1.5">
                  <span className="text-sm font-medium text-foreground">Auth mechanism</span>
                  <select
                    value={draft.auth.type}
                    onChange={(event) => updateAuthType(event.target.value as McpAuthType)}
                    className="rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                  >
                    <option value="none">None</option>
                    <option value="bearer">Bearer token</option>
                    <option value="oauth_client_credentials">
                      OAuth client credentials
                    </option>
                  </select>
                </label>

                {draft.auth.type === "bearer" ? (
                  <label className="grid gap-1.5">
                    <span className="text-sm font-medium text-foreground">
                      Bearer token
                    </span>
                    <input
                      type="password"
                      value={draft.auth.bearer_token ?? ""}
                      onChange={(event) =>
                        updateDraftAuth({ bearer_token: event.target.value })
                      }
                      className="rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                      placeholder={
                        draft.auth.bearer_token_set
                          ? "Stored token unchanged"
                          : "Paste bearer token"
                      }
                      autoComplete="off"
                    />
                  </label>
                ) : null}

                {draft.auth.type === "oauth_client_credentials" ? (
                  <div className="grid gap-4">
                    <label className="grid gap-1.5">
                      <span className="text-sm font-medium text-foreground">
                        Token URL
                      </span>
                      <input
                        type="text"
                        value={draft.auth.token_url ?? ""}
                        onChange={(event) =>
                          updateDraftAuth({ token_url: event.target.value })
                        }
                        className="rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                        placeholder="https://auth.example.com/oauth/token"
                      />
                    </label>
                    <div className="grid gap-4 sm:grid-cols-2">
                      <label className="grid gap-1.5">
                        <span className="text-sm font-medium text-foreground">
                          Client ID
                        </span>
                        <input
                          type="text"
                          value={draft.auth.client_id ?? ""}
                          onChange={(event) =>
                            updateDraftAuth({ client_id: event.target.value })
                          }
                          className="rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                          autoComplete="off"
                        />
                      </label>
                      <label className="grid gap-1.5">
                        <span className="text-sm font-medium text-foreground">
                          Client secret
                        </span>
                        <input
                          type="password"
                          value={draft.auth.client_secret ?? ""}
                          onChange={(event) =>
                            updateDraftAuth({ client_secret: event.target.value })
                          }
                          className="rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                          placeholder={
                            draft.auth.client_secret_set
                              ? "Stored secret unchanged"
                              : "Client secret"
                          }
                          autoComplete="off"
                        />
                      </label>
                    </div>
                    <div className="grid gap-4 sm:grid-cols-2">
                      <label className="grid gap-1.5">
                        <span className="text-sm font-medium text-foreground">
                          Scope
                        </span>
                        <input
                          type="text"
                          value={draft.auth.scope ?? ""}
                          onChange={(event) =>
                            updateDraftAuth({ scope: event.target.value })
                          }
                          className="rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                          placeholder="read:mcp"
                        />
                      </label>
                      <label className="grid gap-1.5">
                        <span className="text-sm font-medium text-foreground">
                          Audience
                        </span>
                        <input
                          type="text"
                          value={draft.auth.audience ?? ""}
                          onChange={(event) =>
                            updateDraftAuth({ audience: event.target.value })
                          }
                          className="rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                          placeholder="Optional"
                        />
                      </label>
                    </div>
                    <div className="grid gap-4 sm:grid-cols-2">
                      <label className="grid gap-1.5">
                        <span className="text-sm font-medium text-foreground">
                          Grant type
                        </span>
                        <input
                          type="text"
                          value={draft.auth.grant_type ?? "client_credentials"}
                          onChange={(event) =>
                            updateDraftAuth({ grant_type: event.target.value })
                          }
                          className="rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                        />
                      </label>
                      <label className="grid gap-1.5">
                        <span className="text-sm font-medium text-foreground">
                          Refresh skew seconds
                        </span>
                        <input
                          type="number"
                          min={1}
                          value={draft.auth.refresh_skew_seconds ?? 30}
                          onChange={(event) =>
                            updateDraftAuth({
                              refresh_skew_seconds: Number(event.target.value) || 30,
                            })
                          }
                          className="rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                        />
                      </label>
                    </div>
                  </div>
                ) : null}
              </div>
              <label className="flex items-center justify-between gap-4 rounded-md border border-border bg-muted/30 px-3 py-3">
                <span>
                  <span className="block text-sm font-medium text-foreground">Enabled</span>
                  <span className="block text-xs text-muted-foreground">
                    Disabled servers remain saved but are excluded from chat runtime.
                  </span>
                </span>
                <input
                  type="checkbox"
                  checked={draft.enabled}
                  onChange={(event) => updateDraft({ enabled: event.target.checked })}
                  className="size-4 rounded border-input text-primary focus:ring-ring"
                />
              </label>
            </div>
            {testResult ? (
              <div
                className={`mx-5 rounded-md border px-3 py-3 text-sm ${
                  testResult.ok
                    ? "border-emerald-200 bg-emerald-50 text-emerald-950"
                    : "border-destructive/30 bg-destructive/10 text-destructive"
                }`}
              >
                <div className="flex items-start gap-2">
                  {testResult.ok ? (
                    <CircleCheck className="mt-0.5 size-4 shrink-0" aria-hidden />
                  ) : (
                    <CircleX className="mt-0.5 size-4 shrink-0" aria-hidden />
                  )}
                  <div className="min-w-0">
                    <p className="font-medium">
                      {testResult.ok
                        ? `${testResult.tool_count} tools found`
                        : "Connection failed"}
                    </p>
                    {testResult.error ? (
                      <p className="mt-1 break-words text-xs">{testResult.error}</p>
                    ) : null}
                    {testResult.tools.length > 0 ? (
                      <ul className="mt-2 grid gap-1 text-xs">
                        {testResult.tools.slice(0, 6).map((toolItem) => (
                          <li key={toolItem.name} className="truncate">
                            <span className="font-medium">{toolItem.name}</span>
                            {toolItem.description ? ` - ${toolItem.description}` : ""}
                          </li>
                        ))}
                      </ul>
                    ) : null}
                  </div>
                </div>
              </div>
            ) : null}
            <div className="flex flex-wrap items-center justify-between gap-3 border-t border-border px-5 py-4">
              <button
                type="button"
                onClick={() => void handleDelete(draft.key)}
                disabled={!selectedServer}
                className="inline-flex min-h-10 items-center gap-2 rounded-md border border-border bg-background px-3 py-2 text-sm font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:cursor-not-allowed disabled:opacity-45"
              >
                <Trash2 className="size-4" aria-hidden />
                Delete
              </button>
              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={() => void handleTestConnection()}
                  disabled={testing || saving}
                  className="inline-flex min-h-10 items-center gap-2 rounded-md border border-border bg-background px-3 py-2 text-sm font-medium text-foreground transition-colors hover:bg-muted disabled:cursor-not-allowed disabled:opacity-60"
                >
                  <SquareActivity className="size-4" aria-hidden />
                  {testing ? "Testing..." : "Test connection"}
                </button>
                <button
                  type="button"
                  onClick={() => void handleSave()}
                  disabled={saving || pendingToggleKeys.size > 0}
                  className="inline-flex min-h-10 items-center gap-2 rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  <Save className="size-4" aria-hidden />
                  {saving ? "Saving..." : "Save server"}
                </button>
              </div>
            </div>
          </section>

          <section className="rounded-lg border border-border bg-card lg:col-span-2">
            <div className="flex items-center justify-between gap-3 border-b border-border px-5 py-4">
              <div>
                <h2 className="text-sm font-semibold text-foreground">Observability</h2>
                <p className="text-xs text-muted-foreground">
                  Links are shown only from non-secret runtime configuration.
                </p>
              </div>
              <BarChart3 className="size-5 text-muted-foreground" aria-hidden />
            </div>
            {configLoading ? (
              <div className="grid gap-0 divide-y divide-border lg:grid-cols-3 lg:divide-x lg:divide-y-0" aria-busy="true">
                {[0, 1, 2].map((item) => (
                  <div key={item} className="grid min-h-[11rem] content-between gap-4 px-5 py-5">
                    <div className="flex items-start gap-3">
                      <span className="size-9 rounded-md bg-muted" />
                      <span className="min-w-0 flex-1">
                        <span className="mb-3 block h-3 w-28 rounded bg-muted" />
                        <span className="mb-2 block h-3 w-full rounded bg-muted" />
                        <span className="block h-3 w-3/4 rounded bg-muted" />
                      </span>
                    </div>
                    <div className="flex items-center justify-between gap-3">
                      <span className="h-6 w-20 rounded-full bg-muted" />
                      <span className="h-10 w-20 rounded-md bg-muted" />
                    </div>
                  </div>
                ))}
              </div>
            ) : observabilityLinks.length === 0 ? (
              <div className="px-5 py-8 text-sm text-muted-foreground">
                No observability destinations configured.
              </div>
            ) : (
              <div className="grid divide-y divide-border lg:grid-cols-3 lg:divide-x lg:divide-y-0">
                {observabilityLinks.map((link) => (
                  <ObservabilityLinkItem key={link.key} link={link} />
                ))}
              </div>
            )}
          </section>
        </div>
      </div>
    </main>
  );
}
