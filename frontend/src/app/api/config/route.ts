import { NextResponse } from 'next/server';

const FASTAPI_BACKEND_URL = process.env.FASTAPI_BACKEND_URL || 'http://localhost:3002';

export async function GET() {
  try {
    const res = await fetch(`${FASTAPI_BACKEND_URL}/api/config`);
    if (!res.ok) {
      return NextResponse.json(
        { error: 'Backend config unavailable' },
        { status: res.status }
      );
    }
    const data = await res.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Config fetch error:', error);
    return NextResponse.json(
      { error: 'Failed to fetch config' },
      { status: 500 }
    );
  }
}
