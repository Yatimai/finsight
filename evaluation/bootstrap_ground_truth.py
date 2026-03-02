"""Bootstrap ground truth by running retrieval on candidate questions.

Connects to Qdrant in embedded mode, encodes each question with ColQwen2,
retrieves top-5 pages, and fills in source_document + source_pages.
Optionally generates candidate answers with Sonnet (--with-generation).

Requirements:
    - Qdrant snapshot in data/qdrant/ (~3.2 GB)
    - ColQwen2 model (~4 GB RAM, CPU OK, ~30s load time)
    - Optional: ANTHROPIC_API_KEY for --with-generation
"""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_DRAFT_PATH = Path("evaluation/questions_draft.json")
DEFAULT_OUTPUT_PATH = Path("evaluation/ground_truth.json")


def bootstrap(
    draft_path: Path,
    output_path: Path,
    with_generation: bool = False,
    top_k: int = 5,
    remote_url: str | None = None,
) -> None:
    """Run retrieval on each question and fill ground truth fields.

    Args:
        draft_path: Path to questions_draft.json.
        output_path: Where to write ground_truth.json.
        with_generation: Also generate candidate answers with Sonnet.
        top_k: Number of pages to retrieve per question.
        remote_url: If set, connect to a remote Qdrant server instead of embedded mode.
    """
    from qdrant_client import QdrantClient

    from app.config import AppConfig
    from app.models.retriever import Retriever

    # Load draft questions
    with open(draft_path) as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions from {draft_path}")

    # Initialize retriever (loads ColQwen2 + Qdrant)
    config = AppConfig()
    qdrant_client = None
    if remote_url:
        print(f"Connecting to remote Qdrant at {remote_url}...")
        qdrant_client = QdrantClient(url=remote_url)
    print("Initializing retriever (loading ColQwen2 model)...")
    retriever = Retriever(config, qdrant_client=qdrant_client)

    # Optional: initialize generator for candidate answers
    generator = None
    if with_generation:
        import anthropic
        import httpx

        from app.models.generator import Generator

        client = anthropic.AsyncAnthropic(
            api_key=config.anthropic.api_key,
            timeout=httpx.Timeout(config.anthropic.timeout_seconds),
        )
        generator = Generator(config, client)
        print("Generator initialized (Sonnet).")

    # Process each question
    for i, q in enumerate(questions, 1):
        qid = q["id"]
        question = q["question"]
        category = q["category"]

        print(f"[{i}/{len(questions)}] {qid}: {question[:60]}...")

        # Skip abstention questions — they have no source by design
        if category == "abstention":
            q["source_document"] = ""
            q["source_pages"] = []
            q["expected_answer"] = "Cette question est hors du scope des documents disponibles."
            print("  → abstention (skipped retrieval)")
            continue

        # Retrieve top-k pages
        pages, _ = retriever.retrieve([question], top_k=top_k)

        if not pages:
            print("  → no pages retrieved")
            continue

        # Fill source document and pages
        q["source_document"] = pages[0].source_filename
        q["source_pages"] = sorted(set(p.page_number for p in pages))

        print(f"  → {q['source_document']}, pages: {q['source_pages']}")

        # Optional: generate candidate answer
        if generator is not None:
            import asyncio

            try:
                gen_result = asyncio.run(generator.generate(question, pages))
                q["expected_answer"] = gen_result["answer"]
                print(f"  → answer: {q['expected_answer'][:80]}...")
            except Exception as e:
                print(f"  → generation failed: {e}")
                q["expected_answer"] = ""

    # Save ground truth
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    print(f"\nGround truth saved to {output_path}")

    # Summary
    filled = sum(1 for q in questions if q.get("source_pages"))
    answered = sum(1 for q in questions if q.get("expected_answer"))
    print(f"Pages filled: {filled}/{len(questions)}")
    print(f"Answers filled: {answered}/{len(questions)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap ground truth by running retrieval on candidate questions.",
    )
    parser.add_argument(
        "--draft",
        type=Path,
        default=DEFAULT_DRAFT_PATH,
        help=f"Path to draft questions (default: {DEFAULT_DRAFT_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output ground truth path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--with-generation",
        action="store_true",
        help="Generate candidate answers with Sonnet (requires ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of pages to retrieve per question (default: 5)",
    )
    parser.add_argument(
        "--remote",
        type=str,
        default=None,
        help="Remote Qdrant URL (default: use embedded mode from config)",
    )
    args = parser.parse_args(argv)

    if not args.draft.exists():
        print(f"Error: Draft file not found: {args.draft}", file=sys.stderr)
        return 1

    bootstrap(
        draft_path=args.draft,
        output_path=args.output,
        with_generation=args.with_generation,
        top_k=args.top_k,
        remote_url=args.remote,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
