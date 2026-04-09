import { NextRequest, NextResponse } from "next/server";

const FASTAPI_BACKEND_URL =
  process.env.FASTAPI_BACKEND_URL || "http://localhost:3002";

export async function DELETE(request: NextRequest) {
  try {
    const collectionName = request.nextUrl.searchParams.get("collection_name");
    const source = request.nextUrl.searchParams.get("source");

    if (!source?.trim()) {
      return NextResponse.json({ error: "source is required", deleted_chunks: 0 }, { status: 400 });
    }

    const backendUrl = new URL(`${FASTAPI_BACKEND_URL}/api/documents/source`);
    backendUrl.searchParams.set("source", source);
    if (collectionName) {
      backendUrl.searchParams.set("collection_name", collectionName);
    }

    const response = await fetch(backendUrl, { method: "DELETE" });
    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      return NextResponse.json(
        {
          error: (data as { detail?: string }).detail ?? "Failed to delete source",
          deleted_chunks: 0,
        },
        { status: response.status },
      );
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error("Delete source API error:", error);
    return NextResponse.json(
      { error: "Failed to delete source", deleted_chunks: 0 },
      { status: 500 },
    );
  }
}
