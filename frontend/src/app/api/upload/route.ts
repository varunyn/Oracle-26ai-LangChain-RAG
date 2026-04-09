import { NextRequest, NextResponse } from 'next/server';

const FASTAPI_BACKEND_URL = process.env.FASTAPI_BACKEND_URL || 'http://localhost:3002';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const files = formData.getAll('files') as File[];
    const collectionName = formData.get('collection_name') as string | null;
    if (!files?.length) {
      return NextResponse.json({ error: 'No files provided', chunks_added: 0 }, { status: 400 });
    }
    const backendForm = new FormData();
    files.forEach((f) => backendForm.append('files', f));
    if (collectionName) backendForm.append('collection_name', collectionName);
    const res = await fetch(`${FASTAPI_BACKEND_URL}/api/documents/upload`, {
      method: 'POST',
      body: backendForm,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      return NextResponse.json({ error: data.error ?? 'Upload failed', chunks_added: 0 }, { status: res.status });
    }
    return NextResponse.json(data);
  } catch (error) {
    console.error('Upload API error:', error);
    return NextResponse.json({ error: 'Upload failed', chunks_added: 0 }, { status: 500 });
  }
}
