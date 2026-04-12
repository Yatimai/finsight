import { Badge } from "@/components/ui/badge";

interface ConfidenceBadgeProps {
  confidence: number | null | undefined;
}

export function ConfidenceBadge({ confidence }: ConfidenceBadgeProps) {
  if (confidence == null) {
    return null;
  }

  const pct = Math.round(confidence * 100);

  let variant: "default" | "secondary" | "destructive";
  let label: string;

  if (confidence >= 0.85) {
    variant = "default";
    label = `Confiance ${pct}%`;
  } else if (confidence >= 0.7) {
    variant = "secondary";
    label = `Confiance ${pct}%`;
  } else {
    variant = "destructive";
    label = `Confiance ${pct}%`;
  }

  const colorClass =
    confidence >= 0.85
      ? "bg-emerald-600 hover:bg-emerald-700 text-white"
      : confidence >= 0.7
        ? "bg-amber-500 hover:bg-amber-600 text-white"
        : "bg-red-600 hover:bg-red-700 text-white";

  return (
    <Badge variant={variant} className={`text-xs ${colorClass}`}>
      {label}
    </Badge>
  );
}
