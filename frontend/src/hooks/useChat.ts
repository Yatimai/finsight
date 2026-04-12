import { useCallback, useState } from "react";
import type { ChatMessage, Source, Citation } from "@/lib/types";
import { queryStream, getVerification } from "@/lib/api";
import { getMockResponse } from "@/lib/mockData";

// Polling constants for async verification
const VERIFICATION_POLL_INTERVAL_MS = 2000;
const VERIFICATION_POLL_MAX_ATTEMPTS = 60; // 60 * 2s = 120s max
const TERMINAL_VERIFICATION_STATUSES = new Set([
  "verified",
  "flagged",
  "low_confidence",
  "error",
  "parse_error",
  "disabled",
  "not_found",
]);

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  // Poll the verification endpoint until a terminal status is reached,
  // then update the message in place. Non-blocking: runs as a detached
  // promise after the initial response is received.
  const pollVerification = useCallback(async (queryId: string) => {
    for (let i = 0; i < VERIFICATION_POLL_MAX_ATTEMPTS; i++) {
      await new Promise((resolve) => setTimeout(resolve, VERIFICATION_POLL_INTERVAL_MS));
      try {
        const v = await getVerification(queryId);
        if (TERMINAL_VERIFICATION_STATUSES.has(v.status)) {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === queryId
                ? { ...m, verification_status: v.status, confidence: v.confidence }
                : m,
            ),
          );
          return;
        }
      } catch (err) {
        // Transient poll error: log and keep retrying
        console.warn("verification poll error", err);
      }
    }
    // Timed out without a terminal status
    console.warn(`verification polling timed out for query ${queryId}`);
  }, []);

  const sendMessage = useCallback(
    async (question: string) => {
      const userMsg: ChatMessage = {
        id: `user-${Date.now()}`,
        role: "user",
        content: question,
      };

      const loadingMsg: ChatMessage = {
        id: `loading-${Date.now()}`,
        role: "assistant",
        content: "",
        loading: true,
      };

      setMessages((prev) => [...prev, userMsg, loadingMsg]);
      setIsLoading(true);

      // Build conversation history for the API (exclude loading placeholder)
      const history = [...messages, userMsg].map((m) => ({
        role: m.role,
        content: m.content,
      }));

      try {
        // Consume the SSE stream: replace the loading placeholder with a
        // message that grows as tokens arrive, finalize on the "done" event,
        // and kick off verification polling if pending.
        let currentId = loadingMsg.id;
        let accumulated = "";

        for await (const event of queryStream(question, history)) {
          if (event.type === "meta") {
            currentId = event.query_id;
            const sources = event.sources as unknown as Source[];
            setMessages((prev) =>
              prev.map((m) =>
                m.id === loadingMsg.id
                  ? { ...m, id: event.query_id, sources, loading: false }
                  : m,
              ),
            );
          } else if (event.type === "token") {
            accumulated += event.text;
            const snapshot = accumulated;
            setMessages((prev) =>
              prev.map((m) => (m.id === currentId ? { ...m, content: snapshot } : m)),
            );
          } else if (event.type === "done") {
            const citations = event.citations as unknown as Citation[];
            setMessages((prev) =>
              prev.map((m) =>
                m.id === currentId
                  ? {
                      ...m,
                      citations,
                      confidence: event.confidence,
                      verification_status: event.verification_status,
                      latency_ms: event.latency_ms,
                    }
                  : m,
              ),
            );
            if (event.verification_status === "pending") {
              void pollVerification(currentId);
            }
          } else if (event.type === "error") {
            throw new Error(event.message);
          }
        }
      } catch {
        // Fallback to mock data when API is unreachable
        const mock = getMockResponse(question);
        const assistantMsg: ChatMessage = {
          id: mock.query_id,
          role: "assistant",
          content: mock.answer,
          sources: mock.sources,
          citations: mock.citations,
          confidence: mock.confidence,
          verification_status: mock.verification_status,
          latency_ms: mock.latency_ms,
        };
        setMessages((prev) =>
          prev.map((m) => (m.id === loadingMsg.id ? assistantMsg : m)),
        );
      } finally {
        setIsLoading(false);
      }
    },
    [messages, pollVerification],
  );

  const clearChat = useCallback(() => {
    setMessages([]);
  }, []);

  return { messages, isLoading, sendMessage, clearChat };
}
