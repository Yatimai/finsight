import type { QueryResponse, VerificationResponse, ChatMessage } from "./types";

const API_BASE = "http://localhost:8000/api/v1";

export async function postQuery(
  question: string,
  conversationHistory: Pick<ChatMessage, "role" | "content">[],
  asyncVerification: boolean = false,
): Promise<QueryResponse> {
  const res = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      conversation_history: conversationHistory,
      async_verification: asyncVerification,
    }),
  });
  if (!res.ok) {
    throw new Error(`API error: ${res.status}`);
  }
  return res.json();
}

export async function getVerification(queryId: string): Promise<VerificationResponse> {
  const res = await fetch(`${API_BASE}/query/${encodeURIComponent(queryId)}/verification`);
  if (!res.ok) {
    throw new Error(`Verification API error: ${res.status}`);
  }
  return res.json();
}

export type StreamEvent =
  | { type: "meta"; query_id: string; sources: Array<Record<string, unknown>> }
  | { type: "token"; text: string }
  | {
      type: "done";
      citations: Array<Record<string, unknown>>;
      verification_status: string;
      confidence: number | null;
      latency_ms: number;
    }
  | { type: "error"; message: string };

/**
 * Stream a query as Server-Sent Events.
 * Yields parsed event objects from the server until the stream ends or an
 * "error" event is received. The caller is responsible for handling each
 * event type (meta / token / done / error).
 */
export async function* queryStream(
  question: string,
  conversationHistory: Pick<ChatMessage, "role" | "content">[],
): AsyncGenerator<StreamEvent> {
  const res = await fetch(`${API_BASE}/query/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      conversation_history: conversationHistory,
    }),
  });
  if (!res.ok || !res.body) {
    throw new Error(`Stream API error: ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE events are separated by double newlines. Keep the last (possibly
      // incomplete) fragment in the buffer for the next iteration.
      const parts = buffer.split("\n\n");
      buffer = parts.pop() ?? "";

      for (const part of parts) {
        const trimmed = part.trim();
        if (!trimmed || !trimmed.startsWith("data:")) continue;
        const payload = trimmed.slice("data:".length).trim();
        if (!payload) continue;
        try {
          yield JSON.parse(payload) as StreamEvent;
        } catch (err) {
          console.warn("Failed to parse SSE event", err, payload);
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

export function getPageImageUrl(documentId: string, page: number): string {
  return `${API_BASE}/pages/${encodeURIComponent(documentId)}/${page}`;
}
