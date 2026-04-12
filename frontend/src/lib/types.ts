export interface Source {
  document: string;
  page: number;
  score: number;
  image_path: string;
}

export interface Citation {
  page: number;
}

export interface QueryResponse {
  query_id: string;
  answer: string;
  sources: Source[];
  citations: Citation[];
  confidence: number | null;
  verification_status: string;
  latency_ms: number;
}

export interface VerificationResponse {
  status: string;
  confidence: number | null;
  claims: Array<Record<string, unknown>>;
  summary: string;
  claims_verified: number;
  claims_contradicted: number;
  claims_not_found: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  citations?: Citation[];
  confidence?: number | null;
  verification_status?: string;
  latency_ms?: number;
  loading?: boolean;
}
