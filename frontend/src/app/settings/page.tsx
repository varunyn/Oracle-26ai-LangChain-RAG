"use client";

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
import Link from "next/link";
import { memo, useCallback, useEffect, useMemo, useState } from "react";

import { useToast } from "@/components/toaster";
import {
  type AppConfig,
  fetchAppConfig,
  type ObservabilityLink,
} from "@/lib/config";
import {
  deleteMcpServer,
  fetchMcpServers,
  type McpAuthType,
  type McpConnectionTestResponse,
  type McpServerAuthConfig,
  type McpServerConfig,
  saveMcpServer,
  setMcpServerEnabled,
  testMcpServerConnection,
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

function normalizeDraftAuth(
  auth: Partial<McpServerAuthConfig>
): McpServerAuthConfig {
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
          ? "bg-primary/[0.08] ring-1 ring-primary/25 ring-inset"
          : "bg-card"
      }`}
    >
      <button
        aria-current={active ? "true" : undefined}
        className={`flex min-w-0 flex-1 items-center gap-3 rounded-md px-2 py-2 text-left transition-colors focus:outline-none focus:ring-2 focus:ring-ring ${
          active ? "hover:bg-primary/[0.08]" : "hover:bg-muted"
        }`}
        onClick={() => onSelect(server)}
        type="button"
      >
        <span
          className={`inline-flex size-8 shrink-0 items-center justify-center rounded-md border ${
            server.enabled
              ? "border-foreground/20 bg-foreground text-background"
              : "border-border bg-background text-muted-foreground"
          }`}
        >
          <Server aria-hidden className="size-4" />
        </span>
        <span className="min-w-0">
          <span className="block truncate font-medium text-foreground text-sm">
            {server.key}
          </span>
          <span className="block truncate text-muted-foreground text-xs">
            {server.url}
          </span>
        </span>
      </button>
      <button
        aria-label={server.enabled ? "Disable MCP server" : "Enable MCP server"}
        className={`relative inline-flex h-8 w-14 shrink-0 items-center rounded-full border transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60 ${
          server.enabled
            ? "border-foreground bg-foreground"
            : "border-border bg-muted"
        }`}
        disabled={pending}
        onClick={() => void onToggle(server)}
        title={server.enabled ? "Disable" : "Enable"}
        type="button"
      >
        <span
          className={`absolute top-1/2 left-0.5 size-5 -translate-y-1/2 rounded-full bg-background shadow-sm transition-transform ${
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
          <Icon aria-hidden className="size-4" />
        </span>
        <div className="min-w-0">
          <h3 className="truncate font-semibold text-foreground text-sm">
            {link.label}
          </h3>
          <p className="mt-1 text-muted-foreground text-xs leading-5">
            {link.details}
          </p>
        </div>
      </div>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <span
          className={`inline-flex items-center rounded-full border px-2.5 py-1 font-medium text-xs ${observabilityStatusClass(
            link
          )}`}
        >
          {link.status}
        </span>
        {link.url ? (
          <a
            className="inline-flex min-h-10 items-center gap-2 rounded-md border border-border bg-background px-3 py-2 font-medium text-foreground text-sm transition-colors hover:bg-muted focus:outline-none focus:ring-2 focus:ring-ring"
            href={link.url}
            rel="noreferrer"
            target="_blank"
          >
            Open
            <ExternalLink aria-hidden className="size-4" />
          </a>
        ) : (
          <span className="text-muted-foreground text-xs">No link</span>
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
  const [testResult, setTestResult] =
    useState<McpConnectionTestResponse | null>(null);
  const [pendingToggleKeys, setPendingToggleKeys] = useState<Set<string>>(
    () => new Set()
  );

  const selectedServer = useMemo(
    () => servers.find((server) => server.key === selectedKey),
    [selectedKey, servers]
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
    void Promise.resolve().then(loadServers);
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
    [updateDraftAuth]
  );

  const handleSave = useCallback(async () => {
    if (pendingToggleKeys.size > 0) {
      toast.error("Wait for the enable change to finish before saving");
      return;
    }
    const key = draft.key.trim();
    const url = draft.url.trim();
    if (!(key && url)) {
      toast.error("Server key and URL are required");
      return;
    }
    setSaving(true);
    try {
      const saved = await saveMcpServer({ ...draft, key, url });
      setServers((current) => {
        const exists = current.some((server) => server.key === saved.key);
        if (!exists) {
          return [...current, saved];
        }
        return current.map((server) =>
          server.key === saved.key ? saved : server
        );
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
          current.map((item) => (item.key === updated.key ? updated : item))
        );
        setDraft((current) =>
          current.key === updated.key ? cloneServer(updated) : current
        );
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
    [toast]
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
    [servers, toast]
  );

  const handleTestConnection = useCallback(async () => {
    const key = draft.key.trim();
    const url = draft.url.trim();
    if (!(key && url)) {
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
      <header className="shrink-0 border-border border-b bg-card px-5 py-4 shadow-sm sm:px-8">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4">
          <div className="flex min-w-0 items-center gap-3">
            <Link
              aria-label="Back to chat"
              className="inline-flex size-10 shrink-0 items-center justify-center rounded-md border border-border bg-background text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
              href="/"
              title="Back to chat"
            >
              <ArrowLeft aria-hidden className="size-4" />
            </Link>
            <div className="min-w-0">
              <h1 className="truncate font-semibold text-foreground text-xl tracking-tight">
                Settings
              </h1>
              <p className="truncate text-muted-foreground text-sm">
                MCP servers and observability
              </p>
            </div>
          </div>
          <button
            className="inline-flex min-h-10 items-center gap-2 rounded-md border border-border bg-background px-3 py-2 font-medium text-foreground text-sm transition-colors hover:bg-muted focus:outline-none focus:ring-2 focus:ring-ring"
            onClick={() => void loadServers()}
            type="button"
          >
            <RefreshCw aria-hidden className="size-4" />
            Refresh
          </button>
        </div>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto px-5 py-6 sm:px-8">
        <div className="mx-auto grid max-w-6xl gap-5 lg:grid-cols-[minmax(18rem,24rem)_1fr]">
          <section className="min-h-[28rem] rounded-lg border border-border bg-card">
            <div className="flex items-center justify-between gap-3 border-border border-b px-4 py-3">
              <div>
                <h2 className="font-semibold text-foreground text-sm">
                  MCP servers
                </h2>
                <p className="text-muted-foreground text-xs">
                  {mcpGloballyEnabled
                    ? "MCP tools are enabled"
                    : "MCP tools are disabled"}
                </p>
              </div>
              <button
                aria-label="Add MCP server"
                className="inline-flex size-10 items-center justify-center rounded-md border border-border bg-background text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                onClick={startNewServer}
                title="Add MCP server"
                type="button"
              >
                <Plus aria-hidden className="size-4" />
              </button>
            </div>
            <div className="divide-y divide-border">
              {loading ? (
                <div aria-busy="true" className="grid gap-3 px-4 py-5">
                  {[0, 1, 2].map((item) => (
                    <div className="flex items-center gap-3" key={item}>
                      <span className="size-8 rounded-md bg-muted" />
                      <span className="min-w-0 flex-1">
                        <span className="mb-2 block h-3 w-24 rounded bg-muted" />
                        <span className="block h-3 w-full max-w-52 rounded bg-muted" />
                      </span>
                    </div>
                  ))}
                </div>
              ) : servers.length === 0 ? (
                <div className="px-4 py-8 text-muted-foreground text-sm">
                  No MCP servers configured.
                </div>
              ) : (
                servers.map((server) => {
                  const active = server.key === selectedKey;
                  return (
                    <ServerListItem
                      active={active}
                      key={server.key}
                      onSelect={selectServer}
                      onToggle={handleToggle}
                      pending={pendingToggleKeys.has(server.key)}
                      server={server}
                    />
                  );
                })
              )}
            </div>
          </section>

          <section className="rounded-lg border border-border bg-card">
            <div className="border-border border-b px-5 py-4">
              <h2 className="font-semibold text-foreground text-sm">
                {selectedServer ? "Edit server" : "Add server"}
              </h2>
            </div>
            <div className="grid gap-5 px-5 py-5">
              <label className="grid gap-1.5">
                <span className="font-medium text-foreground text-sm">Key</span>
                <input
                  className="rounded-md border border-input bg-background px-3 py-2 text-foreground text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                  onChange={(event) => updateDraft({ key: event.target.value })}
                  placeholder="default"
                  type="text"
                  value={draft.key}
                />
              </label>
              <label className="grid gap-1.5">
                <span className="font-medium text-foreground text-sm">
                  Transport
                </span>
                <select
                  className="rounded-md border border-input bg-background px-3 py-2 text-foreground text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                  onChange={(event) =>
                    updateDraft({
                      transport: event.target
                        .value as McpServerConfig["transport"],
                    })
                  }
                  value={draft.transport}
                >
                  <option value="streamable-http">streamable-http</option>
                  <option value="sse">sse</option>
                  <option value="stdio">stdio</option>
                </select>
              </label>
              <label className="grid gap-1.5">
                <span className="font-medium text-foreground text-sm">URL</span>
                <input
                  className="rounded-md border border-input bg-background px-3 py-2 text-foreground text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                  onChange={(event) => updateDraft({ url: event.target.value })}
                  placeholder="http://localhost:9000/mcp"
                  type="text"
                  value={draft.url}
                />
              </label>
              <div className="grid gap-4 rounded-md border border-border bg-muted/20 px-3 py-3">
                <label className="grid gap-1.5">
                  <span className="font-medium text-foreground text-sm">
                    Auth mechanism
                  </span>
                  <select
                    className="rounded-md border border-input bg-background px-3 py-2 text-foreground text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                    onChange={(event) =>
                      updateAuthType(event.target.value as McpAuthType)
                    }
                    value={draft.auth.type}
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
                    <span className="font-medium text-foreground text-sm">
                      Bearer token
                    </span>
                    <input
                      autoComplete="off"
                      className="rounded-md border border-input bg-background px-3 py-2 text-foreground text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                      onChange={(event) =>
                        updateDraftAuth({ bearer_token: event.target.value })
                      }
                      placeholder={
                        draft.auth.bearer_token_set
                          ? "Stored token unchanged"
                          : "Paste bearer token"
                      }
                      type="password"
                      value={draft.auth.bearer_token ?? ""}
                    />
                  </label>
                ) : null}

                {draft.auth.type === "oauth_client_credentials" ? (
                  <div className="grid gap-4">
                    <label className="grid gap-1.5">
                      <span className="font-medium text-foreground text-sm">
                        Token URL
                      </span>
                      <input
                        className="rounded-md border border-input bg-background px-3 py-2 text-foreground text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                        onChange={(event) =>
                          updateDraftAuth({ token_url: event.target.value })
                        }
                        placeholder="https://auth.example.com/oauth/token"
                        type="text"
                        value={draft.auth.token_url ?? ""}
                      />
                    </label>
                    <div className="grid gap-4 sm:grid-cols-2">
                      <label className="grid gap-1.5">
                        <span className="font-medium text-foreground text-sm">
                          Client ID
                        </span>
                        <input
                          autoComplete="off"
                          className="rounded-md border border-input bg-background px-3 py-2 text-foreground text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                          onChange={(event) =>
                            updateDraftAuth({ client_id: event.target.value })
                          }
                          type="text"
                          value={draft.auth.client_id ?? ""}
                        />
                      </label>
                      <label className="grid gap-1.5">
                        <span className="font-medium text-foreground text-sm">
                          Client secret
                        </span>
                        <input
                          autoComplete="off"
                          className="rounded-md border border-input bg-background px-3 py-2 text-foreground text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                          onChange={(event) =>
                            updateDraftAuth({
                              client_secret: event.target.value,
                            })
                          }
                          placeholder={
                            draft.auth.client_secret_set
                              ? "Stored secret unchanged"
                              : "Client secret"
                          }
                          type="password"
                          value={draft.auth.client_secret ?? ""}
                        />
                      </label>
                    </div>
                    <div className="grid gap-4 sm:grid-cols-2">
                      <label className="grid gap-1.5">
                        <span className="font-medium text-foreground text-sm">
                          Scope
                        </span>
                        <input
                          className="rounded-md border border-input bg-background px-3 py-2 text-foreground text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                          onChange={(event) =>
                            updateDraftAuth({ scope: event.target.value })
                          }
                          placeholder="read:mcp"
                          type="text"
                          value={draft.auth.scope ?? ""}
                        />
                      </label>
                      <label className="grid gap-1.5">
                        <span className="font-medium text-foreground text-sm">
                          Audience
                        </span>
                        <input
                          className="rounded-md border border-input bg-background px-3 py-2 text-foreground text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                          onChange={(event) =>
                            updateDraftAuth({ audience: event.target.value })
                          }
                          placeholder="Optional"
                          type="text"
                          value={draft.auth.audience ?? ""}
                        />
                      </label>
                    </div>
                    <div className="grid gap-4 sm:grid-cols-2">
                      <label className="grid gap-1.5">
                        <span className="font-medium text-foreground text-sm">
                          Grant type
                        </span>
                        <input
                          className="rounded-md border border-input bg-background px-3 py-2 text-foreground text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                          onChange={(event) =>
                            updateDraftAuth({ grant_type: event.target.value })
                          }
                          type="text"
                          value={draft.auth.grant_type ?? "client_credentials"}
                        />
                      </label>
                      <label className="grid gap-1.5">
                        <span className="font-medium text-foreground text-sm">
                          Refresh skew seconds
                        </span>
                        <input
                          className="rounded-md border border-input bg-background px-3 py-2 text-foreground text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                          min={1}
                          onChange={(event) =>
                            updateDraftAuth({
                              refresh_skew_seconds:
                                Number(event.target.value) || 30,
                            })
                          }
                          type="number"
                          value={draft.auth.refresh_skew_seconds ?? 30}
                        />
                      </label>
                    </div>
                  </div>
                ) : null}
              </div>
              <label className="flex items-center justify-between gap-4 rounded-md border border-border bg-muted/30 px-3 py-3">
                <span>
                  <span className="block font-medium text-foreground text-sm">
                    Enabled
                  </span>
                  <span className="block text-muted-foreground text-xs">
                    Disabled servers remain saved but are excluded from chat
                    runtime.
                  </span>
                </span>
                <input
                  checked={draft.enabled}
                  className="size-4 rounded border-input text-primary focus:ring-ring"
                  onChange={(event) =>
                    updateDraft({ enabled: event.target.checked })
                  }
                  type="checkbox"
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
                    <CircleCheck
                      aria-hidden
                      className="mt-0.5 size-4 shrink-0"
                    />
                  ) : (
                    <CircleX aria-hidden className="mt-0.5 size-4 shrink-0" />
                  )}
                  <div className="min-w-0">
                    <p className="font-medium">
                      {testResult.ok
                        ? `${testResult.tool_count} tools found`
                        : "Connection failed"}
                    </p>
                    {testResult.error ? (
                      <p className="mt-1 break-words text-xs">
                        {testResult.error}
                      </p>
                    ) : null}
                    {testResult.tools.length > 0 ? (
                      <ul className="mt-2 grid gap-1 text-xs">
                        {testResult.tools.slice(0, 6).map((toolItem) => (
                          <li className="truncate" key={toolItem.name}>
                            <span className="font-medium">{toolItem.name}</span>
                            {toolItem.description
                              ? ` - ${toolItem.description}`
                              : ""}
                          </li>
                        ))}
                      </ul>
                    ) : null}
                  </div>
                </div>
              </div>
            ) : null}
            <div className="flex flex-wrap items-center justify-between gap-3 border-border border-t px-5 py-4">
              <button
                className="inline-flex min-h-10 items-center gap-2 rounded-md border border-border bg-background px-3 py-2 font-medium text-muted-foreground text-sm transition-colors hover:bg-muted hover:text-foreground disabled:cursor-not-allowed disabled:opacity-45"
                disabled={!selectedServer}
                onClick={() => void handleDelete(draft.key)}
                type="button"
              >
                <Trash2 aria-hidden className="size-4" />
                Delete
              </button>
              <div className="flex flex-wrap items-center gap-2">
                <button
                  className="inline-flex min-h-10 items-center gap-2 rounded-md border border-border bg-background px-3 py-2 font-medium text-foreground text-sm transition-colors hover:bg-muted disabled:cursor-not-allowed disabled:opacity-60"
                  disabled={testing || saving}
                  onClick={() => void handleTestConnection()}
                  type="button"
                >
                  <SquareActivity aria-hidden className="size-4" />
                  {testing ? "Testing..." : "Test connection"}
                </button>
                <button
                  className="inline-flex min-h-10 items-center gap-2 rounded-md bg-primary px-3 py-2 font-medium text-primary-foreground text-sm transition-colors hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-60"
                  disabled={saving || pendingToggleKeys.size > 0}
                  onClick={() => void handleSave()}
                  type="button"
                >
                  <Save aria-hidden className="size-4" />
                  {saving ? "Saving..." : "Save server"}
                </button>
              </div>
            </div>
          </section>

          <section className="rounded-lg border border-border bg-card lg:col-span-2">
            <div className="flex items-center justify-between gap-3 border-border border-b px-5 py-4">
              <div>
                <h2 className="font-semibold text-foreground text-sm">
                  Observability
                </h2>
                <p className="text-muted-foreground text-xs">
                  Links are shown only from non-secret runtime configuration.
                </p>
              </div>
              <BarChart3 aria-hidden className="size-5 text-muted-foreground" />
            </div>
            {configLoading ? (
              <div
                aria-busy="true"
                className="grid gap-0 divide-y divide-border lg:grid-cols-3 lg:divide-x lg:divide-y-0"
              >
                {[0, 1, 2].map((item) => (
                  <div
                    className="grid min-h-[11rem] content-between gap-4 px-5 py-5"
                    key={item}
                  >
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
              <div className="px-5 py-8 text-muted-foreground text-sm">
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
