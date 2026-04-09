import { NextRequest, NextResponse } from 'next/server';

const FASTAPI_BACKEND_URL = process.env.FASTAPI_BACKEND_URL || 'http://localhost:3002';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { question, answer, feedback } = body;
    if (typeof question !== 'string' || typeof answer !== 'string' || typeof feedback !== 'number') {
      return NextResponse.json(
        { error: 'question (string), answer (string), and feedback (number 1-5) are required' },
        { status: 400 }
      );
    }
    const res = await fetch(`${FASTAPI_BACKEND_URL}/api/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, answer, feedback }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      return NextResponse.json(
        { error: data.detail ?? data.error ?? 'Feedback failed' },
        { status: res.status }
      );
    }
    return NextResponse.json(data);
  } catch (error) {
    console.error('Feedback API error:', error);
    return NextResponse.json({ error: 'Failed to submit feedback' }, { status: 500 });
  }
}
