"use client";

import Link from "next/link";
import {
  ArrowLeft,
  Plus,
  RefreshCw,
  Save,
  Server,
  Trash2,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

import { useToast } from "@/components/toaster";
import {
  deleteMcpServer,
  fetchMcpServers,
  saveMcpServer,
  setMcpServerEnabled,
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
  }, []);

  const startNewServer = useCallback(() => {
    setSelectedKey("");
    setDraft(emptyDraft);
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
        toast.success("MCP server deleted");
      } catch (error) {
        console.error(error);
        toast.error("Could not delete MCP server");
      }
    },
    [servers, toast],
  );

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
                      className={`flex items-center gap-3 px-3 py-3 ${
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
                        className={`relative h-6 w-11 rounded-full border transition-colors ${
                          server.enabled
                            ? "border-foreground bg-foreground"
                            : "border-border bg-muted"
                        }`}
                        aria-label={server.enabled ? "Disable MCP server" : "Enable MCP server"}
                        title={server.enabled ? "Disable" : "Enable"}
                      >
                        <span
                          className={`absolute top-0.5 size-4 rounded-full bg-background transition-transform ${
                            server.enabled ? "translate-x-5" : "translate-x-0.5"
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
                  onChange={(event) =>
                    setDraft((current) => ({ ...current, key: event.target.value }))
                  }
                  className="rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                  placeholder="default"
                />
              </label>
              <label className="grid gap-1.5">
                <span className="text-sm font-medium text-foreground">Transport</span>
                <select
                  value={draft.transport}
                  onChange={(event) =>
                    setDraft((current) => ({
                      ...current,
                      transport: event.target.value as McpServerConfig["transport"],
                    }))
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
                  onChange={(event) =>
                    setDraft((current) => ({ ...current, url: event.target.value }))
                  }
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
                  onChange={(event) =>
                    setDraft((current) => ({ ...current, enabled: event.target.checked }))
                  }
                  className="size-4 rounded border-input text-primary focus:ring-ring"
                />
              </label>
            </div>
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
          </section>
        </div>
      </div>
    </main>
  );
}
