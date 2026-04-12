import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ChatMessage } from "./ChatMessage";
import { useChat } from "@/hooks/useChat";

const SUGGESTED_QUESTIONS = [
  "Quel est le chiffre d'affaires de LVMH en 2024 ?",
  "Quelle est la marge opérationnelle de TotalEnergies au T4 ?",
  "Quelles sont les prévisions 2025 de Danone ?",
];

export function ChatInterface() {
  const { messages, isLoading, sendMessage, clearChat } = useChat();
  const [input, setInput] = useState("");
  // Ref attached directly to the scrollable div — no intermediary
  // component/wrapper. scrollTop applies to the actual scrollable element.
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to the bottom whenever messages change (including during
  // token streaming: each token replaces the messages array). The write is
  // scheduled for the next animation frame so the new content height is
  // committed before we set scrollTop.
  useEffect(() => {
    const frame = requestAnimationFrame(() => {
      if (scrollRef.current) {
        scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
      }
    });
    return () => cancelAnimationFrame(frame);
  }, [messages]);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;
    setInput("");
    sendMessage(trimmed);
  };

  const handleSuggestion = (question: string) => {
    if (isLoading) return;
    sendMessage(question);
  };

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border px-8 py-6">
        <div>
          <h1 className="text-5xl font-bold tracking-tight">FinSight</h1>
          <p className="text-base text-muted-foreground mt-1.5">
            Visual RAG pour documents financiers
          </p>
        </div>
        {messages.length > 0 && (
          <Button variant="ghost" size="sm" onClick={clearChat}>
            Nouvelle conversation
          </Button>
        )}
      </div>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-6">
        <div className="mx-auto max-w-3xl py-6">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-20">
              <h2 className="text-xl font-semibold mb-2">
                Posez une question sur vos documents financiers
              </h2>
              <p className="text-sm text-muted-foreground mb-8 text-center max-w-md">
                FinSight analyse les DEU 2024 (documents d'enregistrement
                universel) de 10 sociétés du CAC 40 avec citations vérifiables.
              </p>
              <div className="flex flex-wrap justify-center gap-2">
                {SUGGESTED_QUESTIONS.map((q) => (
                  <Button
                    key={q}
                    variant="outline"
                    size="sm"
                    className="text-xs"
                    onClick={() => handleSuggestion(q)}
                  >
                    {q}
                  </Button>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              {messages.map((msg) => (
                <ChatMessage key={msg.id} message={msg} />
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Input */}
      <div className="border-t border-border px-6 py-4">
        <form
          onSubmit={handleSubmit}
          className="mx-auto flex max-w-3xl items-center gap-3"
        >
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ex : Quel est le chiffre d'affaires de LVMH en 2024 ?"
            disabled={isLoading}
            className="flex-1"
          />
          <Button type="submit" disabled={isLoading || !input.trim()}>
            Envoyer
          </Button>
        </form>
        <p className="mx-auto max-w-3xl mt-2 text-center text-xs text-muted-foreground">
          ColQwen2.5 + Claude Sonnet/Opus — les réponses sont sourcées et
          vérifiées
        </p>
      </div>
    </div>
  );
}
