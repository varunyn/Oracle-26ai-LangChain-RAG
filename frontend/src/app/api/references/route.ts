import { NextRequest, NextResponse } from 'next/server';

const FASTAPI_BACKEND_URL = process.env.FASTAPI_BACKEND_URL || 'http://localhost:3002';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { messages, model, thread_id, session_id, collection_name, enable_reranker, enable_tracing } = body;

    if (!messages || !Array.isArray(messages)) {
      return NextResponse.json({ error: 'Messages array is required' }, { status: 400 });
    }

    const lastUserMessage = (() => {
      for (let i = messages.length - 1; i >= 0; i -= 1) {
        const msg = messages[i];
        if (!msg || typeof msg !== 'object') continue;
        if ('role' in msg && (msg as { role?: unknown }).role === 'user') return msg;
      }
      return null;
    })();

    if (!lastUserMessage) {
      return NextResponse.json({ error: 'Last user message is required' }, { status: 400 });
    }

    const res = await fetch(`${FASTAPI_BACKEND_URL}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: [lastUserMessage],
        model: model ?? undefined,
        stream: false,
        thread_id: thread_id ?? undefined,
        session_id: session_id ?? undefined,
        collection_name: collection_name ?? undefined,
        enable_reranker: enable_reranker ?? undefined,
        enable_tracing: enable_tracing ?? undefined,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      return NextResponse.json({ error: err.error ?? 'Backend error' }, { status: res.status });
    }
    const data = await res.json();
    return NextResponse.json({
      standalone_question: data.standalone_question,
      citations: data.citations ?? [],
      reranker_docs: data.reranker_docs ?? [],
    });
  } catch (error) {
    console.error('References API error:', error);
    return NextResponse.json({ error: 'Failed to load references' }, { status: 500 });
  }
}
