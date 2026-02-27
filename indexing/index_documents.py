"""
Indexing pipeline: PDF → Page Images → ColQwen2 Vision Encoder → Qdrant

This script is designed to run on Google Colab (free GPU) or any machine
with a CUDA GPU. It processes PDFs into page images, encodes them with
ColQwen2's vision encoder, and stores multi-vector embeddings in Qdrant.

Usage:
    python -m indexing.index_documents --config config.yaml
    python -m indexing.index_documents --pdf data/documents/rapport_2023.pdf
    python -m indexing.index_documents --dir data/documents/
"""

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.logging import get_logger, setup_logging

logger = get_logger("indexing")

import torch
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    MultiVectorComparator,
    MultiVectorConfig,
    PointStruct,
    VectorParams,
)

from app.config import AppConfig, load_config
from indexing.utils import (
    compute_document_hash,
    iter_pdf_files,
    pdf_to_images,
    save_page_images,
)

# ---------------------------------------------------------------------------
# ColQwen2 model loader
# ---------------------------------------------------------------------------


class ColQwen2Encoder:
    """Wraps ColQwen2 for encoding document page images and text queries."""

    def __init__(self, model_name: str = "vidore/colqwen2-v1.0"):
        self.model_name = model_name
        self.device = self._select_device()
        self.dtype = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        self.model = None
        self.processor = None

    def _select_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def load(self):
        """Load model and processor. Call once before encoding."""
        from colpali_engine.models import ColQwen2, ColQwen2Processor

        logger.info("loading_model", model=self.model_name, device=str(self.device), dtype=str(self.dtype))

        attn_impl = None
        if self.device.type == "cuda":
            try:
                from transformers.utils.import_utils import is_flash_attn_2_available

                if is_flash_attn_2_available():
                    attn_impl = "flash_attention_2"
            except ImportError:
                pass

        self.model = ColQwen2.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map=str(self.device),
            attn_implementation=attn_impl,
        ).eval()

        self.processor = ColQwen2Processor.from_pretrained(self.model_name)
        logger.info("model_loaded", flash_attention=attn_impl or "disabled")

    def encode_images(self, images: list[Image.Image], batch_size: int = 8) -> list[torch.Tensor]:
        """
        Encode page images into multi-vector embeddings.

        Returns a list of tensors, each of shape (num_patches, embedding_dim).
        For ColQwen2, this is typically (768, 128) per page.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            batch_inputs = self.processor.process_images(batch).to(self.device)

            with torch.no_grad():
                batch_embeddings = self.model(**batch_inputs)

            # Move to CPU and store individually
            for j in range(len(batch)):
                embedding = batch_embeddings[j].cpu().float()
                all_embeddings.append(embedding)

            # Free GPU memory
            del batch_inputs, batch_embeddings
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return all_embeddings

    def encode_query(self, query: str) -> torch.Tensor:
        """
        Encode a text query into multi-vector embedding.
        Returns tensor of shape (num_tokens, embedding_dim).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        batch_queries = self.processor.process_queries([query]).to(self.device)

        with torch.no_grad():
            query_embedding = self.model(**batch_queries)

        return query_embedding[0].cpu().float()

    @property
    def embedding_dim(self) -> int:
        """Dimension of each vector in the multi-vector representation."""
        return 128


# ---------------------------------------------------------------------------
# Qdrant storage
# ---------------------------------------------------------------------------


class QdrantStorage:
    """Manages Qdrant collection for multi-vector page embeddings."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.collection_name = config.qdrant.collection_name

        if config.qdrant.mode == "embedded":
            self.client = QdrantClient(path=config.qdrant.path)
        else:
            self.client = QdrantClient(url=config.qdrant.remote_url)

    def ensure_collection(self, embedding_dim: int = 128):
        """Create collection if it doesn't exist."""
        collections = [c.name for c in self.client.get_collections().collections]

        if self.collection_name not in collections:
            # For multi-vector with MaxSim, we use named vectors
            # Each page gets a variable number of patch vectors
            # Qdrant handles this via multivector mode
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "colqwen2": VectorParams(
                        size=embedding_dim,
                        distance=Distance.COSINE,
                        multivector_config=MultiVectorConfig(
                            comparator=MultiVectorComparator.MAX_SIM,
                        ),
                    )
                },
            )
            logger.info("collection_created", name=self.collection_name)
        else:
            logger.info("collection_exists", name=self.collection_name)

    def page_exists(self, document_hash: str, page_number: int) -> bool:
        """Check if a page is already indexed (for idempotency)."""
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter={
                "must": [
                    {"key": "document_hash", "match": {"value": document_hash}},
                    {"key": "page_number", "match": {"value": page_number}},
                ]
            },
            limit=1,
        )
        return len(results[0]) > 0

    def document_exists(self, document_hash: str) -> bool:
        """Check if any page of this document is already indexed."""
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter={
                "must": [
                    {"key": "document_hash", "match": {"value": document_hash}},
                ]
            },
            limit=1,
        )
        return len(results[0]) > 0

    def store_page(
        self,
        point_id: int,
        embedding: torch.Tensor,
        metadata: dict,
    ):
        """Store a single page embedding with metadata."""
        # Convert embedding tensor to list of lists for Qdrant
        vectors = embedding.tolist()  # shape: (num_patches, dim) → list[list[float]]

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector={"colqwen2": vectors},
                    payload=metadata,
                )
            ],
        )

    def get_next_id(self) -> int:
        """Get the next available point ID."""
        collection_info = self.client.get_collection(self.collection_name)
        return collection_info.points_count

    def count_pages(self) -> int:
        """Total number of indexed pages."""
        collection_info = self.client.get_collection(self.collection_name)
        return collection_info.points_count


# ---------------------------------------------------------------------------
# Index tracker (for idempotency)
# ---------------------------------------------------------------------------


class IndexTracker:
    """
    Tracks which documents have been indexed.
    Stores document hashes and metadata in a JSON file.
    """

    def __init__(self, tracker_path: str = "./data/index_tracker.json"):
        self.path = Path(tracker_path)
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path) as f:
                return json.load(f)
        return {"documents": {}}

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def is_indexed(self, document_hash: str) -> bool:
        return document_hash in self.data["documents"]

    def mark_indexed(
        self,
        document_hash: str,
        filename: str,
        num_pages: int,
        first_point_id: int,
    ):
        self.data["documents"][document_hash] = {
            "filename": filename,
            "num_pages": num_pages,
            "first_point_id": first_point_id,
            "indexed_at": datetime.now(UTC).isoformat(),
        }
        self._save()


# ---------------------------------------------------------------------------
# Main indexing function
# ---------------------------------------------------------------------------


def index_document(
    pdf_path: Path,
    encoder: ColQwen2Encoder,
    storage: QdrantStorage,
    tracker: IndexTracker,
    config: AppConfig,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    """
    Index a single PDF document.

    Returns a summary dict with indexing results.
    """
    filename = pdf_path.name
    logger.info("processing_document", filename=filename)

    # Step 1: Idempotency check
    doc_hash = compute_document_hash(pdf_path)
    if tracker.is_indexed(doc_hash) and not force:
        logger.info("already_indexed", filename=filename, hash=doc_hash[:12])
        return {"filename": filename, "status": "skipped", "reason": "already_indexed"}

    # Step 2: PDF → images
    logger.info("converting_pdf", filename=filename, dpi=config.data.dpi)
    t0 = time.time()
    images = pdf_to_images(pdf_path, dpi=config.data.dpi)
    num_pages = len(images)
    logger.info("pages_extracted", filename=filename, num_pages=num_pages, duration_s=round(time.time() - t0, 1))

    # Step 3: Save page images (needed for Claude API at query time)
    logger.info("saving_images", filename=filename)
    doc_id = doc_hash[:12]
    saved_paths = save_page_images(images, config.data.pages_dir, doc_id)

    # Step 4: Encode with ColQwen2
    logger.info("encoding", filename=filename, batch_size=config.data.batch_size)
    t0 = time.time()
    embeddings = encoder.encode_images(images, batch_size=config.data.batch_size)
    encode_time = time.time() - t0
    logger.info(
        "encoded",
        filename=filename,
        pages=len(embeddings),
        duration_s=round(encode_time, 1),
        per_page_s=round(encode_time / num_pages, 2),
    )

    # Step 5: Store in Qdrant (skip in dry-run mode)
    if dry_run:
        logger.info("dry_run_skip_qdrant", filename=filename, num_pages=num_pages)
        return {
            "filename": filename,
            "status": "dry_run",
            "document_id": doc_id,
            "num_pages": num_pages,
            "encode_time_s": round(encode_time, 1),
        }

    logger.info("storing_qdrant", filename=filename)
    first_point_id = storage.get_next_id()

    for i, (embedding, image_path) in enumerate(zip(embeddings, saved_paths, strict=False)):
        page_number = i + 1
        point_id = first_point_id + i

        metadata = {
            "document_id": doc_id,
            "document_hash": doc_hash,
            "source_filename": filename,
            "page_number": page_number,
            "total_pages": num_pages,
            "image_path": str(image_path),
            "indexed_at": datetime.now(UTC).isoformat(),
            "num_patches": embedding.shape[0],
        }

        storage.store_page(point_id, embedding, metadata)

    # Step 6: Mark as indexed
    tracker.mark_indexed(doc_hash, filename, num_pages, first_point_id)

    logger.info(
        "document_indexed",
        filename=filename,
        num_pages=num_pages,
        first_id=first_point_id,
        last_id=first_point_id + num_pages - 1,
    )

    return {
        "filename": filename,
        "status": "indexed",
        "document_id": doc_id,
        "num_pages": num_pages,
        "encode_time_s": round(encode_time, 1),
        "first_point_id": first_point_id,
    }


def index_directory(
    directory: Path,
    encoder: ColQwen2Encoder,
    storage: QdrantStorage,
    tracker: IndexTracker,
    config: AppConfig,
    force: bool = False,
    dry_run: bool = False,
) -> list[dict]:
    """Index all PDFs in a directory."""
    pdf_files = list(iter_pdf_files(directory))

    if not pdf_files:
        logger.warning("no_pdfs_found", directory=str(directory))
        return []

    logger.info("indexing_directory", directory=str(directory), file_count=len(pdf_files))

    results = []
    for pdf_path in pdf_files:
        result = index_document(pdf_path, encoder, storage, tracker, config, force=force, dry_run=dry_run)
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Index financial PDFs into the multimodal RAG system.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file (default: config.yaml)")
    parser.add_argument("--pdf", type=str, default=None, help="Path to a single PDF to index")
    parser.add_argument("--dir", type=str, default=None, help="Path to directory of PDFs to index")
    parser.add_argument("--force", action="store_true", help="Force re-indexing even if document already exists")
    parser.add_argument("--dry-run", action="store_true", help="Run pipeline without writing to Qdrant or tracker")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    setup_logging(config.observability.log_level)

    # Initialize components
    logger.info("initializing_encoder")
    encoder = ColQwen2Encoder(model_name=config.retrieval.model)
    encoder.load()

    logger.info("initializing_qdrant")
    storage = QdrantStorage(config)
    storage.ensure_collection(embedding_dim=encoder.embedding_dim)

    tracker_path = str(Path(config.data.documents_dir).parent / "index_tracker.json")
    tracker = IndexTracker(tracker_path=tracker_path)

    # Index
    t_total = time.time()

    if args.pdf:
        results = [
            index_document(Path(args.pdf), encoder, storage, tracker, config, force=args.force, dry_run=args.dry_run)
        ]
    elif args.dir:
        results = index_directory(
            Path(args.dir), encoder, storage, tracker, config, force=args.force, dry_run=args.dry_run
        )
    else:
        # Default: index from config documents_dir
        results = index_directory(
            Path(config.data.documents_dir), encoder, storage, tracker, config, force=args.force, dry_run=args.dry_run
        )

    # Summary
    total_time = time.time() - t_total
    indexed = [r for r in results if r["status"] == "indexed"]
    skipped = [r for r in results if r["status"] == "skipped"]
    dry_runs = [r for r in results if r["status"] == "dry_run"]
    total_pages = sum(r.get("num_pages", 0) for r in indexed + dry_runs)

    logger.info(
        "indexing_complete",
        documents_indexed=len(indexed),
        documents_skipped=len(skipped),
        documents_dry_run=len(dry_runs),
        total_pages=total_pages,
        total_time_s=round(total_time, 1),
        qdrant_total_pages=storage.count_pages() if not args.dry_run else 0,
    )


if __name__ == "__main__":
    main()
