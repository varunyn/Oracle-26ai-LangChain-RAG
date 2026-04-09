import { NextRequest } from "next/server";

const FASTAPI_BACKEND_URL =
  process.env.FASTAPI_BACKEND_URL || "http://localhost:3002";

/**
 * DELETE /api/threads/[thread_id] – proxy to backend to clear checkpointer memory for the thread.
 * Returns 204 on success, 404 if thread not found, 500 on proxy error.
 */
export async function DELETE(
  _request: NextRequest,
  { params }: { params: Promise<{ thread_id: string }> }
) {
  const { thread_id } = await params;
  if (!thread_id?.trim()) {
    return new Response(JSON.stringify({ error: "thread_id required" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }
  try {
    const res = await fetch(
      `${FASTAPI_BACKEND_URL}/api/threads/${encodeURIComponent(thread_id)}`,
      { method: "DELETE" }
    );
    if (res.status === 204) return new Response(null, { status: 204 });
    if (res.status === 404) {
      const data = await res.json().catch(() => ({}));
      return new Response(JSON.stringify(data), {
        status: 404,
        headers: { "Content-Type": "application/json" },
      });
    }
    const data = await res.json().catch(() => ({}));
    return new Response(
      JSON.stringify({
        error: (data as { error?: string }).error ?? "Delete thread failed",
      }),
      { status: res.status, headers: { "Content-Type": "application/json" } }
    );
  } catch (error) {
    console.error("Delete thread API error:", error);
    return new Response(
      JSON.stringify({ error: "Failed to delete thread" }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }
}
