import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { getPageImageUrl } from "@/lib/api";
import type { Source } from "@/lib/types";

interface PagePreviewProps {
  source: Source | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function PagePreview({ source, open, onOpenChange }: PagePreviewProps) {
  if (!source) return null;

  const imageUrl = getPageImageUrl(source.document, source.page);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-auto">
        <DialogHeader>
          <DialogTitle className="text-sm font-medium">
            {source.document} — Page {source.page}
          </DialogTitle>
        </DialogHeader>
        <div className="mt-2">
          <img
            src={imageUrl}
            alt={`${source.document} page ${source.page}`}
            className="w-full rounded border border-border"
            onError={(e) => {
              const el = e.currentTarget;
              el.style.display = "none";
              el.parentElement!.innerHTML =
                '<div class="flex items-center justify-center h-96 bg-muted rounded text-muted-foreground text-sm">Aperçu non disponible (backend hors ligne)</div>';
            }}
          />
        </div>
      </DialogContent>
    </Dialog>
  );
}
