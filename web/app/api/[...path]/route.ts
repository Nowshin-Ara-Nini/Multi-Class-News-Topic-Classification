import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 90;
const paths: Record<string, string> = {
  "predict": "POST", "batch_predict": "POST", "model/info": "GET",
  "model/version": "GET", "models": "GET", "ready": "GET", "health": "GET",
};

async function proxy(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  const path = (await context.params).path.join("/");
  if (paths[path] !== request.method) return NextResponse.json({ error: "Not found" }, { status: 404 });
  const base = process.env.INFERENCE_API_URL;
  const key = process.env.INFERENCE_API_KEY;
  if (!base || !key) return NextResponse.json({ error: "Inference service is not configured yet." }, { status: 503 });
  try {
    let body: string | undefined;
    if (request.method === "POST") {
      const origin = request.headers.get("origin");
      if (origin && origin !== request.nextUrl.origin) return NextResponse.json({ error: "Invalid origin" }, { status: 403 });
      if (Number(request.headers.get("content-length")) > 2_100_000) return NextResponse.json({ error: "Request too large" }, { status: 413 });
      body = await request.text();
      if (body.length > 600_000) return NextResponse.json({ error: "Request too large" }, { status: 413 });
      let data;
      try { data = JSON.parse(body); } catch { return NextResponse.json({ error: "Invalid JSON" }, { status: 400 }); }
      const texts = path === "predict" ? [data.text] : data.texts;
      if (!Array.isArray(texts) || texts.length < 1 || texts.length > 100 ||
          texts.some((text) => typeof text !== "string" || !text.trim() || text.length > 5000)) {
        return NextResponse.json({ error: "Provide 1–100 nonempty headlines, each at most 5,000 characters." }, { status: 422 });
      }
    }
    const headers: Record<string, string> = { "Content-Type": "application/json", "X-API-Key": key };
    if (process.env.INFERENCE_VERCEL_BYPASS_SECRET) headers["x-vercel-protection-bypass"] = process.env.INFERENCE_VERCEL_BYPASS_SECRET;
    const response = await fetch(`${base.replace(/\/$/, "")}/${path}`, {
      method: request.method, headers, body, cache: "no-store", signal: AbortSignal.timeout(75000),
    });
    if (!response.ok) {
      const messages: Record<number, string> = {
        401: "Inference authentication is not configured correctly.",
        403: "Inference deployment access is not configured correctly.",
        429: "The model is busy. Please retry shortly.",
        503: "The model is unavailable or starting. Please retry shortly.",
        504: "The model took too long to respond. Please retry.",
      };
      return NextResponse.json({ error: messages[response.status] || "Inference request failed." }, { status: response.status });
    }
    return NextResponse.json(await response.json(), { headers: { "Cache-Control": "no-store" } });
  } catch {
    return NextResponse.json({ error: "The model could not be reached in time. Please retry." }, { status: 504 });
  }
}

export const GET = proxy;
export const POST = proxy;
