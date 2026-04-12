import { useState } from "react";
import Markdown from "react-markdown";
import type { ChatMessage as ChatMessageType, Source } from "@/lib/types";
import { ConfidenceBadge } from "./ConfidenceBadge";
import { PagePreview } from "./PagePreview";

interface ChatMessageProps {
  message: ChatMessageType;
}

const MAX_SOURCES_VISIBLE = 3;

/** Split text on [Page X] patterns, render markdown segments and citation buttons. */
function renderContent(
  text: string,
  sources: Source[],
  onClickCitation: (source: Source) => void,
) {
  const parts = text.split(/(\[Page \d+\])/g);
  return parts.map((part, i) => {
    const match = part.match(/^\[Page (\d+)\]$/);
    if (match) {
      const page = parseInt(match[1], 10);
      const source = sources.find((s) => s.page === page);
      return (
        <button
          key={i}
          onClick={() => source && onClickCitation(source)}
          className="inline-flex items-center rounded bg-primary/10 px-1.5 py-0.5 text-xs font-medium text-primary hover:bg-primary/20 transition-colors cursor-pointer"
          title={source ? `${source.document} — Page ${page}` : `Page ${page}`}
        >
          Page {page}
        </button>
      );
    }
    if (!part) return null;
    return (
      <Markdown
        key={i}
        components={{
          p: ({ children }) => <span>{children}</span>,
          strong: ({ children }) => (
            <strong className="font-semibold">{children}</strong>
          ),
          em: ({ children }) => <em>{children}</em>,
          ul: ({ children }) => (
            <ul className="list-disc pl-4 my-1">{children}</ul>
          ),
          ol: ({ children }) => (
            <ol className="list-decimal pl-4 my-1">{children}</ol>
          ),
          li: ({ children }) => <li className="my-0.5">{children}</li>,
        }}
      >
        {part}
      </Markdown>
    );
  });
}

export function ChatMessage({ message }: ChatMessageProps) {
  const [previewSource, setPreviewSource] = useState<Source | null>(null);
  const [showAllSources, setShowAllSources] = useState(false);

  if (message.loading) {
    return (
      <div className="flex justify-start">
        <div className="max-w-[80%] rounded-2xl rounded-bl-sm bg-muted px-4 py-3">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <div className="flex gap-1">
              <span className="animate-bounce [animation-delay:0ms]">.</span>
              <span className="animate-bounce [animation-delay:150ms]">.</span>
              <span className="animate-bounce [animation-delay:300ms]">.</span>
            </div>
            Analyse en cours
          </div>
        </div>
      </div>
    );
  }

  const isUser = message.role === "user";
  const sources = message.sources ?? [];
  const visibleSources = showAllSources
    ? sources
    : sources.slice(0, MAX_SOURCES_VISIBLE);
  const hiddenCount = sources.length - MAX_SOURCES_VISIBLE;

  return (
    <>
      <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
        <div
          className={`max-w-[80%] rounded-2xl px-4 py-3 ${
            isUser
              ? "rounded-br-sm bg-primary text-primary-foreground"
              : "rounded-bl-sm bg-muted"
          }`}
        >
          <div className="text-sm leading-relaxed whitespace-pre-wrap">
            {isUser
              ? message.content
              : renderContent(
                  message.content,
                  sources,
                  setPreviewSource,
                )}
          </div>

          {!isUser && (
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <ConfidenceBadge confidence={message.confidence} />
              {message.latency_ms != null && message.latency_ms > 0 && (
                <span className="text-xs text-muted-foreground">
                  {`${(message.latency_ms / 1000).toFixed(1)}s`}
                </span>
              )}
            </div>
          )}

          {!isUser && sources.length > 0 && (
            <div className="mt-2 flex flex-wrap items-center gap-1">
              {visibleSources.map((s, i) => (
                <button
                  key={i}
                  onClick={() => setPreviewSource(s)}
                  className="text-xs text-muted-foreground hover:text-foreground underline underline-offset-2 cursor-pointer transition-colors"
                >
                  {s.document} p.{s.page}
                </button>
              ))}
              {hiddenCount > 0 && !showAllSources && (
                <button
                  onClick={() => setShowAllSources(true)}
                  className="text-xs text-muted-foreground/70 hover:text-muted-foreground cursor-pointer transition-colors"
                >
                  Voir {hiddenCount} autre{hiddenCount > 1 ? "s" : ""} source
                  {hiddenCount > 1 ? "s" : ""}
                </button>
              )}
            </div>
          )}
        </div>
      </div>

      <PagePreview
        source={previewSource}
        open={previewSource !== null}
        onOpenChange={(open) => !open && setPreviewSource(null)}
      />
    </>
  );
}
