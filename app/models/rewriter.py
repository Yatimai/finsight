"""
RAG Fusion query rewriter.
Uses Sonnet to generate multiple semantically distinct interpretations
of a user query for parallel retrieval.
"""

import json

import anthropic

from app.config import AppConfig
from app.errors import (
    RewritingFallbackError,
    ServiceUnavailableError,
    call_anthropic_with_retry,
)

REWRITE_SYSTEM_PROMPT = """Tu es un spécialiste en reformulation de requêtes pour un système de recherche documentaire financier.

Ton rôle : transformer la question de l'utilisateur en {max_rewrites} reformulations SÉMANTIQUEMENT DISTINCTES qui couvrent les différentes interprétations possibles de la question.

RÈGLES :
- Chaque reformulation doit être une question autonome (compréhensible sans contexte)
- Les reformulations doivent être DIVERSIFIÉES (pas des synonymes, mais des angles différents)
- Si la question est déjà précise, varier l'angle d'analyse (montant, évolution, comparaison)
- Intégrer le contexte de conversation si fourni
- Répondre UNIQUEMENT avec un JSON array de strings, rien d'autre

EXEMPLE :
Question : "les marges"
Contexte : l'utilisateur a précédemment posé des questions sur LVMH en 2023
Réponse : ["Quelle est la marge brute de LVMH en 2023 ?", "Quelle est la marge opérationnelle de LVMH en 2023 ?", "Comment la marge nette de LVMH a-t-elle évolué en 2023 ?"]

EXEMPLE :
Question : "et en 2022 ?"
Contexte : la question précédente portait sur le chiffre d'affaires 2023
Réponse : ["Quel est le chiffre d'affaires en 2022 ?", "Comment le chiffre d'affaires a-t-il évolué entre 2022 et 2023 ?", "Quel est le détail du chiffre d'affaires par segment en 2022 ?"]"""


class QueryRewriter:
    """
    Rewrites user queries into multiple interpretations using RAG Fusion.

    When max_rewrites=1, acts as a simple context-aware rewriter.
    When max_rewrites=3, generates diverse interpretations for RRF fusion.
    """

    def __init__(self, config: AppConfig, client: anthropic.AsyncAnthropic):
        self.config = config
        self.client = client
        self.max_rewrites = config.rewriting.max_rewrites
        self.enabled = config.rewriting.enabled

    async def rewrite(
        self,
        query: str,
        conversation_history: list[dict] | None = None,
    ) -> list[str]:
        """
        Rewrite a query into multiple interpretations.

        Args:
            query: The user's raw query
            conversation_history: Previous Q&A pairs for context

        Returns:
            List of rewritten queries (1 if disabled, max_rewrites otherwise)
        """
        if not self.enabled:
            return [query]

        # Build context from conversation history
        context = self._build_context(conversation_history)

        # Build the user message
        user_message = f'Question : "{query}"'
        if context:
            user_message = f"{context}\n{user_message}"

        try:
            queries = await self._call_sonnet(user_message)

            # Validate: must be a list of non-empty strings
            if not isinstance(queries, list) or len(queries) == 0:
                return [query]

            queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]

            if not queries:
                return [query]

            return queries[: self.max_rewrites]

        except ServiceUnavailableError as e:
            # Fallback: use raw query
            raise RewritingFallbackError(f"Rewriting failed, falling back to raw query: {query}") from e
        except Exception:
            # Any other error: fall back silently
            return [query]

    async def _call_sonnet(self, user_message: str) -> list[str]:
        """Call Sonnet to generate rewritten queries."""

        async def _api_call():
            response = await self.client.messages.create(
                model=self.config.generation.model,  # Use same Sonnet as generation
                max_tokens=512,
                temperature=0.3,  # Slight creativity for diverse rewrites
                system=[
                    {
                        "type": "text",
                        "text": REWRITE_SYSTEM_PROMPT.format(max_rewrites=self.max_rewrites),
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_message}],
            )
            return response

        response = await call_anthropic_with_retry(
            _api_call,
            max_retries=self.config.error_handling.rewriting_max_retries,
            backoff_base=self.config.error_handling.backoff_base,
            component="rewriter",
        )

        # Parse JSON response
        text = response.content[0].text.strip()

        # Handle potential markdown code fences
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        return json.loads(text)

    def _build_context(self, conversation_history: list[dict] | None) -> str:
        """Build conversation context string from history."""
        if not conversation_history:
            return ""

        # Take last 3 exchanges max
        recent = conversation_history[-3:]
        lines = []
        for exchange in recent:
            q = exchange.get("question", "")
            a = exchange.get("answer", "")
            if q:
                lines.append(f"Q: {q}")
            if a:
                # Truncate long answers
                a_short = a[:200] + "..." if len(a) > 200 else a
                lines.append(f"R: {a_short}")

        if lines:
            return "Contexte de conversation :\n" + "\n".join(lines) + "\n"
        return ""
