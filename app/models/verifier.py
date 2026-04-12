"""
Verifier: Claude Opus adversarial verification.
Verifies Sonnet's response against source documents fact-by-fact.
"""

import json

import anthropic

from app.config import AppConfig
from app.errors import call_anthropic_with_retry, extract_text_from_response
from app.logging import get_logger
from app.models.retriever import RetrievedPage
from indexing.utils import encode_image_base64

logger = get_logger("verifier")


VERIFICATION_SYSTEM_PROMPT = """Tu es un auditeur financier. Ton travail est de vérifier la fidélité d'une réponse par rapport aux documents sources.

Procède en 3 étapes :

ÉTAPE 1 — EXTRACTION
Liste chaque fait vérifiable dans la réponse (chiffres, dates, noms, relations causales, comparaisons). Numérote-les.

ÉTAPE 2 — VÉRIFICATION
Pour chaque fait, cherche la preuve dans les documents sources.
Cite la page et l'emplacement exact (tableau, paragraphe, graphique).
Si le fait implique un calcul, vérifie le calcul.

ÉTAPE 3 — VERDICT
Pour chaque fait, donne un verdict :
- CONFIRMÉ : preuve trouvée et cohérente
- CONTREDIT : preuve trouvée mais contredit la réponse (précise la valeur correcte)
- NON TROUVÉ : aucune preuve dans les documents

Puis donne un score de confiance global de 0.0 à 1.0.

FORMATS NUMÉRIQUES :
- Les documents peuvent utiliser différentes conventions (US, FR, UK)
- US : virgule = séparateur milliers, point = décimale (ex: $648,125.00 = six-cent-quarante-huit-mille-cent-vingt-cinq dollars)
- FR : espace/point = séparateur milliers, virgule = décimale (ex: 648 125,00 €)
- Quand tu compares des chiffres, normalise-les AVANT de conclure CONFIRMÉ ou CONTREDIT
- Ne marque CONTREDIT que si les valeurs sont réellement différentes après normalisation

Réponds UNIQUEMENT avec un JSON structuré comme suit :
{
  "claims": [
    {
      "id": 1,
      "claim": "Le CA 2023 est de 86,2 milliards",
      "verdict": "CONFIRMÉ",
      "evidence": "Page 12, tableau des résultats consolidés, ligne Chiffre d'affaires",
      "correction": null
    }
  ],
  "confidence": 0.95,
  "summary": "Tous les faits vérifiables sont confirmés par les documents sources."
}"""


class Verifier:
    """Verifies generation responses using Claude Opus."""

    def __init__(self, config: AppConfig, client: anthropic.AsyncAnthropic):
        self.config = config
        self.client = client
        self.enabled = config.verification.enabled

    async def verify(
        self,
        query: str,
        answer: str,
        pages: list[RetrievedPage],
    ) -> dict:
        """
        Verify an answer against source pages.
        Tries fallback models if the primary model fails.

        Returns:
            Dict with "status", "confidence", "claims", "summary",
            "input_tokens", "output_tokens"
        """
        if not self.enabled:
            return self._disabled_result()

        content = self._build_verification_content(query, answer, pages)
        models = [self.config.verification.model, *self.config.verification.fallback_models]

        for model in models:

            async def _api_call(m=model):
                response = await self.client.messages.create(
                    model=m,
                    max_tokens=2048,
                    temperature=0.0,
                    system=[
                        {
                            "type": "text",
                            "text": VERIFICATION_SYSTEM_PROMPT,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[{"role": "user", "content": content}],
                )
                return response

            try:
                response = await call_anthropic_with_retry(
                    _api_call,
                    max_retries=self.config.error_handling.verification_max_retries,
                    backoff_base=self.config.error_handling.backoff_base,
                    component="verifier",
                )
                text = extract_text_from_response(response)
                result = self._parse_verification(text)
                result["input_tokens"] = response.usage.input_tokens
                result["output_tokens"] = response.usage.output_tokens
                result["cache_read_tokens"] = getattr(response.usage, "cache_read_input_tokens", 0)
                result["model_used"] = model
                if model != self.config.verification.model:
                    logger.warning("verification_fallback_used", model=model)
                return result
            except Exception as e:
                logger.warning("verification_model_failed", model=model, error=str(e))
                continue

        return self._error_result("Tous les modèles de vérification indisponibles")

    def should_abstain(self, verification_result: dict) -> bool:
        """Check if the system should abstain based on verification."""
        if verification_result.get("status") == "error":
            return False  # Don't discard a good answer due to verification failure
        confidence = verification_result.get("confidence")
        if confidence is None:
            return False
        return confidence < self.config.verification.confidence_threshold

    def _build_verification_content(
        self,
        query: str,
        answer: str,
        pages: list[RetrievedPage],
    ) -> list[dict]:
        """Build multimodal verification content."""
        content = []

        # Source documents
        content.append({"type": "text", "text": "DOCUMENTS SOURCES :"})

        for page in pages:
            encoded = self._encode_image(page)
            if encoded:
                image_data, media_type = encoded
                content.append({"type": "text", "text": f"[Page {page.page_number} — {page.source_filename}]"})
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    }
                )

        # Question and answer to verify
        content.append({"type": "text", "text": f"QUESTION POSÉE :\n{query}\n\nRÉPONSE À VÉRIFIER :\n{answer}"})

        return content

    def _parse_verification(self, text: str) -> dict:
        """Parse Opus's JSON verification response."""
        try:
            # Handle markdown code fences
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            data = json.loads(cleaned)

            claims = data.get("claims", [])
            confidence = float(data.get("confidence", 0.0))
            summary = data.get("summary", "")

            # Determine status
            contradicted = any(c.get("verdict") == "CONTREDIT" for c in claims)
            not_found = sum(1 for c in claims if c.get("verdict") == "NON TROUVÉ")
            total_claims = len(claims)

            if contradicted or (total_claims > 0 and not_found / total_claims > 0.5):
                status = "flagged"
            elif confidence < self.config.verification.confidence_threshold:
                status = "low_confidence"
            else:
                status = "verified"

            return {
                "status": status,
                "confidence": confidence,
                "claims": claims,
                "summary": summary,
                "claims_verified": sum(1 for c in claims if c.get("verdict") == "CONFIRMÉ"),
                "claims_contradicted": sum(1 for c in claims if c.get("verdict") == "CONTREDIT"),
                "claims_not_found": not_found,
            }

        except (json.JSONDecodeError, KeyError, ValueError):
            # If parsing fails, return raw text as summary
            return {
                "status": "parse_error",
                "confidence": 0.0,
                "claims": [],
                "summary": text[:500],
                "claims_verified": 0,
                "claims_contradicted": 0,
                "claims_not_found": 0,
            }

    def _encode_image(self, page: RetrievedPage) -> tuple[str, str] | None:
        """Load and base64-encode a page image. Returns (data, media_type)."""
        return encode_image_base64(page.image_path)

    def _disabled_result(self) -> dict:
        return {
            "status": "disabled",
            "confidence": None,
            "claims": [],
            "summary": "Verification disabled",
            "claims_verified": 0,
            "claims_contradicted": 0,
            "claims_not_found": 0,
        }

    def _error_result(self, error: str) -> dict:
        return {
            "status": "error",
            "confidence": None,
            "claims": [],
            "summary": f"Verification failed: {error}",
            "claims_verified": 0,
            "claims_contradicted": 0,
            "claims_not_found": 0,
        }
