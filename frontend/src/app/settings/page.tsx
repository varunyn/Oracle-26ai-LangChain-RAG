"use client";

import Link from "next/link";
import {
  ArrowLeft,
  CircleCheck,
  CircleX,
  Plus,
  RefreshCw,
  Save,
  Server,
  SquareActivity,
  Trash2,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

import { useToast } from "@/components/toaster";
import {
  deleteMcpServer,
  fetchMcpServers,
  saveMcpServer,
  setMcpServerEnabled,
  testMcpServerConnection,
  type McpConnectionTestResponse,
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
  isNew: true,
};

function cloneServer(server: McpServerConfig): DraftServer {
  return { ...server };
}

export default function SettingsPage(): React.ReactElement {
  const { toast } = useToast();
  const [servers, setServers] = useState<McpServerConfig[]>([]);
  const [draft, setDraft] = useState<DraftServer>(emptyDraft);
  const [selectedKey, setSelectedKey] = useState<string>("");
  const [mcpGloballyEnabled, setMcpGloballyEnabled] = useState(true);
  const [loading, setLoading] = useState(true);
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
    try {
      const data = await fetchMcpServers();
      setServers(data.servers);
      setMcpGloballyEnabled(data.enable_mcp_tools);
      const first = data.servers[0];
      if (first) {
        setSelectedKey(first.key);
        setDraft(cloneServer(first));
      } else {
        setSelectedKey("");
        setDraft(emptyDraft);
      }
    } catch (error) {
      console.error(error);
      toast.error("Could not load MCP settings");
    } finally {
      setLoading(false);
    }
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

  return (
    <main className="flex h-screen min-h-0 flex-col overflow-hidden bg-muted/20">
      <header className="shrink-0 border-b border-border bg-card px-5 py-4 shadow-sm sm:px-8">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4">
          <div className="flex min-w-0 items-center gap-3">
            <Link
              href="/"
              className="inline-flex size-9 shrink-0 items-center justify-center rounded-md border border-border bg-background text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
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
                MCP server configuration
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={() => void loadServers()}
            className="inline-flex items-center gap-2 rounded-md border border-border bg-background px-3 py-2 text-sm font-medium text-foreground transition-colors hover:bg-muted focus:outline-none focus:ring-2 focus:ring-ring"
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
                className="inline-flex size-8 items-center justify-center rounded-md border border-border bg-background text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                aria-label="Add MCP server"
                title="Add MCP server"
              >
                <Plus className="size-4" aria-hidden />
              </button>
            </div>
            <div className="divide-y divide-border">
              {loading ? (
                <div className="px-4 py-8 text-sm text-muted-foreground">Loading servers...</div>
              ) : servers.length === 0 ? (
                <div className="px-4 py-8 text-sm text-muted-foreground">
                  No MCP servers configured.
                </div>
              ) : (
                servers.map((server) => {
                  const active = server.key === selectedKey;
                  return (
                    <div
                      key={server.key}
                      className={`grid grid-cols-[minmax(0,1fr)_3rem] items-center gap-3 px-3 py-3 ${
                        active ? "bg-muted/70" : "bg-card"
                      }`}
                    >
                      <button
                        type="button"
                        onClick={() => selectServer(server)}
                        className="flex min-w-0 flex-1 items-center gap-3 rounded-md px-2 py-2 text-left transition-colors hover:bg-muted"
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
                        onClick={() => void handleToggle(server)}
                        disabled={pendingToggleKeys.has(server.key)}
                        className={`relative inline-flex h-7 w-12 shrink-0 items-center rounded-full border transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60 ${
                          server.enabled
                            ? "border-foreground bg-foreground"
                            : "border-border bg-muted"
                        }`}
                        aria-label={server.enabled ? "Disable MCP server" : "Enable MCP server"}
                        title={server.enabled ? "Disable" : "Enable"}
                      >
                        <span
                          className={`absolute left-0.5 top-1/2 size-5 -translate-y-1/2 rounded-full bg-background shadow-sm transition-transform ${
                            server.enabled ? "translate-x-5" : "translate-x-0"
                          }`}
                        />
                      </button>
                    </div>
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
                className="inline-flex items-center gap-2 rounded-md border border-border bg-background px-3 py-2 text-sm font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:cursor-not-allowed disabled:opacity-45"
              >
                <Trash2 className="size-4" aria-hidden />
                Delete
              </button>
              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={() => void handleTestConnection()}
                  disabled={testing || saving}
                  className="inline-flex items-center gap-2 rounded-md border border-border bg-background px-3 py-2 text-sm font-medium text-foreground transition-colors hover:bg-muted disabled:cursor-not-allowed disabled:opacity-60"
                >
                  <SquareActivity className="size-4" aria-hidden />
                  {testing ? "Testing..." : "Test connection"}
                </button>
                <button
                  type="button"
                  onClick={() => void handleSave()}
                  disabled={saving || pendingToggleKeys.size > 0}
                  className="inline-flex items-center gap-2 rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  <Save className="size-4" aria-hidden />
                  {saving ? "Saving..." : "Save server"}
                </button>
              </div>
            </div>
          </section>
        </div>
      </div>
    </main>
  );
}
