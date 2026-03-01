// Renders assistant reply text as markdown (tables, bold, lists, etc.)
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface MarkdownMessageProps {
  content: string;
  className?: string;
}

export function MarkdownMessage({ content, className = "" }: MarkdownMessageProps) {
  return (
    <div className={`markdown-message prose prose-sm dark:prose-invert max-w-none min-w-0 break-words ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          // Keep styling minimal inside chat bubbles
          p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
          ul: ({ children }) => <ul className="my-2 list-disc pl-4 space-y-0.5">{children}</ul>,
          ol: ({ children }) => <ol className="my-2 list-decimal pl-4 space-y-0.5">{children}</ol>,
          strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
          table: ({ children }) => (
            <div className="my-2 w-full max-w-full overflow-hidden rounded-lg border border-border bg-muted/30">
              <table className="w-full table-fixed border-collapse text-sm">{children}</table>
            </div>
          ),
          thead: ({ children }) => <thead className="bg-muted/50">{children}</thead>,
          th: ({ children }) => (
            <th className="break-words border border-border px-2 py-1.5 text-left font-medium">{children}</th>
          ),
          td: ({ children }) => (
            <td className="break-all border border-border px-2 py-1.5 align-top">{children}</td>
          ),
          tr: ({ children }) => <tr>{children}</tr>,
          tbody: ({ children }) => <tbody>{children}</tbody>,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
