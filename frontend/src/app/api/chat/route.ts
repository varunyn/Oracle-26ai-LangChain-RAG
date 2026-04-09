import { NextRequest, NextResponse } from 'next/server';

const FASTAPI_BACKEND_URL = process.env.FASTAPI_BACKEND_URL || 'http://localhost:3002';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const {
      messages,
      model,
      stream: streamRequested,
      thread_id,
      session_id,
      collection_name,
      enable_reranker,
      enable_tracing,
      mode,
      mcp_server_keys,
    } = body;

    const stream = streamRequested !== false;

    if (!messages || !Array.isArray(messages)) {
      return NextResponse.json(
        { error: 'Messages array is required' },
        { status: 400 }
      );
    }

    const hasUserMessage = messages.some((message: unknown) => {
      if (!message || typeof message !== 'object') return false;
      return 'role' in message && (message as { role?: unknown }).role === 'user';
    });

    if (!hasUserMessage) {
      return NextResponse.json(
        { error: 'At least one user message is required' },
        { status: 400 }
      );
    }

    const backendUrl = `${FASTAPI_BACKEND_URL}/api/chat`;
    const backendBody: Record<string, unknown> = {
      messages,
      model,
      stream,
      thread_id: thread_id ?? undefined,
      session_id: session_id ?? undefined,
      collection_name: collection_name ?? undefined,
      enable_reranker: enable_reranker ?? undefined,
      enable_tracing: enable_tracing ?? undefined,
      mode: mode ?? undefined,
      mcp_server_keys: Array.isArray(mcp_server_keys) ? mcp_server_keys : undefined,
    };

    const upstreamResponse = await fetch(backendUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(backendBody),
    });

    if (!upstreamResponse.ok || !upstreamResponse.body) {
      let message = 'Backend service error';
      try {
        const data = await upstreamResponse.clone().json();
        if (data && typeof data.error === 'string') {
          message = data.error;
        }
      } catch {
        try {
          const text = await upstreamResponse.clone().text();
          if (text) message = text;
        } catch {
          // ignore
        }
      }
      return NextResponse.json(
        { error: message || 'Backend service error' },
        { status: upstreamResponse.status || 502 }
      );
    }

    const banned = new Set([
      'connection',
      'keep-alive',
      'transfer-encoding',
      'content-length',
      'content-encoding',
    ]);

    const sanitized = new Headers();
    upstreamResponse.headers.forEach((value, key) => {
      const lower = key.toLowerCase();
      if (banned.has(lower)) return;
      sanitized.set(key, value);
    });

    if (stream) {
      sanitized.set('content-type', 'text/event-stream');
      sanitized.set('x-vercel-ai-ui-message-stream', 'v1');
      if (!sanitized.has('cache-control')) sanitized.set('cache-control', 'no-cache');
      sanitized.set('x-accel-buffering', 'no');
    }

    return new Response(upstreamResponse.body, {
      status: upstreamResponse.status,
      headers: sanitized,
    });
  } catch (error) {
    console.error('API route error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
