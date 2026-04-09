import { NextRequest, NextResponse } from "next/server";

const FASTAPI_BACKEND_URL =
  process.env.FASTAPI_BACKEND_URL || "http://localhost:3002";

export async function GET(request: NextRequest) {
  try {
    const collectionName = request.nextUrl.searchParams.get("collection_name");
    const backendUrl = new URL(`${FASTAPI_BACKEND_URL}/api/documents/sources`);
    if (collectionName) {
      backendUrl.searchParams.set("collection_name", collectionName);
    }

    const response = await fetch(backendUrl, { method: "GET" });
    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      return NextResponse.json(
        { error: (data as { detail?: string }).detail ?? "Failed to load sources", sources: [] },
        { status: response.status },
      );
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error("Processed sources API error:", error);
    return NextResponse.json(
      { error: "Failed to load sources", sources: [] },
      { status: 500 },
    );
  }
}
