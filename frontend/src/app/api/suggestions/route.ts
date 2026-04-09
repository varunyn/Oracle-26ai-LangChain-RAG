import { NextRequest, NextResponse } from "next/server";

const FASTAPI_BACKEND_URL =
  process.env.FASTAPI_BACKEND_URL || "http://localhost:3002";

export async function POST(request: NextRequest) {
  try {
    const body = (await request.json().catch(() => null)) as
      | { last_message?: string; lastMessage?: string; model?: string }
      | null;
    const lastMessage =
      typeof body?.last_message === "string"
        ? body.last_message.trim()
        : typeof body?.lastMessage === "string"
          ? body.lastMessage.trim()
          : "";
    if (!lastMessage) {
      return NextResponse.json({ suggestions: [] }, { status: 200 });
    }

    const backendUrl = `${FASTAPI_BACKEND_URL}/api/suggestions`;
    const res = await fetch(backendUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        last_message: lastMessage,
        model: body?.model ?? undefined,
      }),
    });

    if (!res.ok) {
      return NextResponse.json(
        { suggestions: [] as string[] },
        { status: 200 }
      );
    }

    const data = (await res.json()) as { suggestions?: string[] };
    const suggestions = Array.isArray(data?.suggestions) ? data.suggestions : [];
    return NextResponse.json({ suggestions });
  } catch (error) {
    console.error("[suggestions]", error);
    return NextResponse.json(
      { suggestions: [] as string[] },
      { status: 200 }
    );
  }
}
