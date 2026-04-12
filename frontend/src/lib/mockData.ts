import type { QueryResponse } from "./types";

const MOCK_RESPONSES: Record<string, QueryResponse> = {
  default: {
    query_id: "mock-001",
    answer:
      "Le chiffre d'affaires de LVMH en 2024 s'élève à 84 683 millions d'euros, en baisse de 2% par rapport à 2023. La division Mode & Maroquinerie reste le premier contributeur avec 41 164 M€ [Page 12]. Les vins et spiritueux ont reculé de 11% à 5 907 M€ [Page 15].",
    sources: [
      { document: "LVMH_DEU_2024.pdf", page: 12, score: 0.87, image_path: "" },
      { document: "LVMH_DEU_2024.pdf", page: 15, score: 0.82, image_path: "" },
      { document: "LVMH_DEU_2024.pdf", page: 14, score: 0.76, image_path: "" },
    ],
    citations: [{ page: 12 }, { page: 15 }],
    confidence: 0.92,
    verification_status: "verified",
    latency_ms: 4230,
  },
  marge: {
    query_id: "mock-002",
    answer:
      "La marge opérationnelle de TotalEnergies au T4 2024 est de 12,3%, en hausse de 1,8 point par rapport au T3 [Page 8]. Cette amélioration est principalement due à la remontée des prix du Brent au-dessus de 80$/baril [Page 23].",
    sources: [
      { document: "TotalEnergies_DEU_2024.pdf", page: 8, score: 0.91, image_path: "" },
      { document: "TotalEnergies_DEU_2024.pdf", page: 23, score: 0.84, image_path: "" },
    ],
    citations: [{ page: 8 }, { page: 23 }],
    confidence: 0.88,
    verification_status: "verified",
    latency_ms: 3870,
  },
  abstention: {
    query_id: "mock-003",
    answer:
      "Cette information n'apparaît pas dans les documents disponibles. Les documents indexés couvrent les DEU 2024 de 10 sociétés du CAC 40, mais ne contiennent pas de données sur les prévisions 2025 de Danone.",
    sources: [],
    citations: [],
    confidence: null,
    verification_status: "skipped",
    latency_ms: 1200,
  },
  faible: {
    query_id: "mock-004",
    answer:
      "D'après les sources disponibles, le résultat net part du groupe Société Générale s'établit à 4 200 M€ en 2024 [Page 45]. Toutefois, la confiance sur ce chiffre est faible car la page source est un graphique peu lisible.",
    sources: [
      { document: "SocieteGenerale_DEU_2024.pdf", page: 45, score: 0.68, image_path: "" },
    ],
    citations: [{ page: 45 }],
    confidence: 0.62,
    verification_status: "low_confidence",
    latency_ms: 5100,
  },
};

const KEYS = Object.keys(MOCK_RESPONSES);
let index = 0;

export function getMockResponse(question: string): QueryResponse {
  // Simple keyword matching for demo variety
  const q = question.toLowerCase();
  if (q.includes("marge") || q.includes("opérationnel")) {
    return { ...MOCK_RESPONSES.marge, query_id: `mock-${Date.now()}` };
  }
  if (q.includes("danone") || q.includes("prévision") || q.includes("2025")) {
    return { ...MOCK_RESPONSES.abstention, query_id: `mock-${Date.now()}` };
  }
  if (q.includes("société générale") || q.includes("socgen")) {
    return { ...MOCK_RESPONSES.faible, query_id: `mock-${Date.now()}` };
  }
  // Cycle through responses
  const key = KEYS[index % KEYS.length];
  index++;
  return { ...MOCK_RESPONSES[key], query_id: `mock-${Date.now()}` };
}
