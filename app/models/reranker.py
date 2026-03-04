"""
Visual reranker using MonoQwen2-VL.
Scores each (query, page image) pair and reorders by relevance.
"""

from __future__ import annotations

import time

import torch
from PIL import Image

from app.config import RerankingConfig
from app.logging import get_logger
from app.models.retriever import RetrievedPage

logger = get_logger("reranker")


class VisualReranker:
    """
    Reranks retrieved pages using MonoQwen2-VL-v0.1.

    Scores each (query, page image) pair with a forward pass,
    then softmax on "True"/"False" logits (MonoT5-style).
    Model is loaded lazily on first rerank() call.
    """

    def __init__(self, config: RerankingConfig):
        self.config = config
        self._model = None
        self._processor = None
        self._true_token_id: int | None = None
        self._false_token_id: int | None = None

    def _load_model(self) -> None:
        """Load model and processor lazily on first use."""
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoProcessor

        logger.info("loading_reranker", model=self.config.model)
        t0 = time.time()

        self._processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self._model.eval()

        # Cache token IDs for True/False
        self._true_token_id = self._processor.tokenizer.convert_tokens_to_ids("True")
        self._false_token_id = self._processor.tokenizer.convert_tokens_to_ids("False")

        elapsed = (time.time() - t0) * 1000
        logger.info("reranker_loaded", latency_ms=round(elapsed))

    def _score_single(self, query: str, image: Image.Image) -> float:
        """Score a single (query, image) pair. Returns P(True)."""
        self._load_model()

        prompt = f"Is the following image relevant to the query: '{query}'? Answer True or False."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Get logits for the last token position
        last_logits = outputs.logits[0, -1, :]
        true_logit = last_logits[self._true_token_id]
        false_logit = last_logits[self._false_token_id]

        # Softmax over True/False
        probs = torch.softmax(torch.stack([true_logit, false_logit]), dim=0)
        return probs[0].item()

    def rerank(self, query: str, pages: list[RetrievedPage]) -> list[RetrievedPage]:
        """
        Rerank pages by visual relevance to the query.

        Loads images, scores each page, returns pages sorted by reranking score (desc).
        """
        if not pages:
            return []

        scored: list[tuple[float, RetrievedPage]] = []
        for page in pages:
            page.load_image()
            score = self._score_single(query, page.image)
            scored.append((score, page))
            logger.debug(
                "reranker_score",
                page=page.page_number,
                document=page.source_filename,
                score=round(score, 4),
            )

        # Sort by reranking score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        return [page for _, page in scored]
