"""Tests for indexing.index_documents — regression tests for bugs fixed.

All tests mock ColQwen2Encoder and QdrantStorage so they run without GPU in CI.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from PIL import Image

from indexing.index_documents import IndexTracker, index_document

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_pdf(tmp_path: Path, name: str = "test.pdf") -> Path:
    """Create a minimal fake PDF file for hashing."""
    pdf = tmp_path / name
    pdf.write_bytes(b"%PDF-1.4 fake content " + name.encode())
    return pdf


def _make_mock_encoder() -> MagicMock:
    """Return a mock ColQwen2Encoder that produces fake embeddings."""
    encoder = MagicMock()
    encoder.embedding_dim = 128
    # encode_images returns one tensor per image: shape (4, 128) = 4 patches
    encoder.encode_images.side_effect = lambda images, **kw: [torch.randn(4, 128) for _ in images]
    return encoder


def _make_mock_storage() -> MagicMock:
    """Return a mock QdrantStorage."""
    storage = MagicMock()
    storage.get_next_id.return_value = 0
    storage.count_pages.return_value = 0
    return storage


def _make_config(tmp_path: Path):
    """Return a minimal AppConfig with tmp_path-based directories."""
    from app.config import AppConfig

    return AppConfig(
        data={
            "documents_dir": str(tmp_path / "documents"),
            "pages_dir": str(tmp_path / "pages"),
            "dpi": 72,
            "batch_size": 2,
        }
    )


# ---------------------------------------------------------------------------
# TestIndexTracker
# ---------------------------------------------------------------------------


class TestIndexTracker:
    def test_persistence_and_reload(self, tmp_path):
        tracker_file = tmp_path / "tracker.json"
        tracker = IndexTracker(tracker_path=str(tracker_file))
        assert not tracker.is_indexed("abc123")

        tracker.mark_indexed("abc123", "test.pdf", 5, 0)
        assert tracker.is_indexed("abc123")

        # Reload from disk
        tracker2 = IndexTracker(tracker_path=str(tracker_file))
        assert tracker2.is_indexed("abc123")
        assert tracker2.data["documents"]["abc123"]["filename"] == "test.pdf"
        assert tracker2.data["documents"]["abc123"]["num_pages"] == 5

    def test_creates_parent_directory(self, tmp_path):
        nested = tmp_path / "deep" / "nested" / "tracker.json"
        tracker = IndexTracker(tracker_path=str(nested))
        tracker.mark_indexed("hash1", "doc.pdf", 1, 0)
        assert nested.exists()

    def test_empty_tracker_has_no_documents(self, tmp_path):
        tracker = IndexTracker(tracker_path=str(tmp_path / "empty.json"))
        assert not tracker.is_indexed("anything")


# ---------------------------------------------------------------------------
# TestIndexDocumentForce
# ---------------------------------------------------------------------------


class TestIndexDocumentForce:
    @patch("indexing.index_documents.pdf_to_images")
    @patch("indexing.index_documents.save_page_images")
    def test_skips_already_indexed_without_force(self, mock_save, mock_pdf2img, tmp_path):
        pdf = _make_fake_pdf(tmp_path)
        config = _make_config(tmp_path)
        encoder = _make_mock_encoder()
        storage = _make_mock_storage()

        tracker = IndexTracker(tracker_path=str(tmp_path / "tracker.json"))

        # First indexing
        mock_pdf2img.return_value = [Image.new("RGB", (10, 10))]
        mock_save.return_value = [tmp_path / "page_0001.png"]
        result1 = index_document(pdf, encoder, storage, tracker, config)
        assert result1["status"] == "indexed"

        # Second call without force → skip
        result2 = index_document(pdf, encoder, storage, tracker, config)
        assert result2["status"] == "skipped"
        assert result2["reason"] == "already_indexed"

    @patch("indexing.index_documents.pdf_to_images")
    @patch("indexing.index_documents.save_page_images")
    def test_reindexes_with_force(self, mock_save, mock_pdf2img, tmp_path):
        pdf = _make_fake_pdf(tmp_path)
        config = _make_config(tmp_path)
        encoder = _make_mock_encoder()
        storage = _make_mock_storage()

        tracker = IndexTracker(tracker_path=str(tmp_path / "tracker.json"))

        # First indexing
        mock_pdf2img.return_value = [Image.new("RGB", (10, 10))]
        mock_save.return_value = [tmp_path / "page_0001.png"]
        result1 = index_document(pdf, encoder, storage, tracker, config)
        assert result1["status"] == "indexed"

        # Second call WITH force → re-index
        result2 = index_document(pdf, encoder, storage, tracker, config, force=True)
        assert result2["status"] == "indexed"


# ---------------------------------------------------------------------------
# TestDryRun
# ---------------------------------------------------------------------------


class TestDryRun:
    @patch("indexing.index_documents.pdf_to_images")
    @patch("indexing.index_documents.save_page_images")
    def test_dry_run_skips_qdrant_and_tracker(self, mock_save, mock_pdf2img, tmp_path):
        pdf = _make_fake_pdf(tmp_path)
        config = _make_config(tmp_path)
        encoder = _make_mock_encoder()
        storage = _make_mock_storage()

        tracker = IndexTracker(tracker_path=str(tmp_path / "tracker.json"))

        mock_pdf2img.return_value = [Image.new("RGB", (10, 10)), Image.new("RGB", (10, 10))]
        mock_save.return_value = [tmp_path / "p1.png", tmp_path / "p2.png"]

        result = index_document(pdf, encoder, storage, tracker, config, dry_run=True)

        assert result["status"] == "dry_run"
        assert result["num_pages"] == 2

        # Qdrant was NOT called
        storage.store_page.assert_not_called()
        storage.get_next_id.assert_not_called()

        # Tracker was NOT written
        assert not tracker.is_indexed(result["document_id"])

    @patch("indexing.index_documents.pdf_to_images")
    @patch("indexing.index_documents.save_page_images")
    def test_dry_run_still_calls_encoder(self, mock_save, mock_pdf2img, tmp_path):
        pdf = _make_fake_pdf(tmp_path)
        config = _make_config(tmp_path)
        encoder = _make_mock_encoder()
        storage = _make_mock_storage()

        tracker = IndexTracker(tracker_path=str(tmp_path / "tracker.json"))

        images = [Image.new("RGB", (10, 10))]
        mock_pdf2img.return_value = images
        mock_save.return_value = [tmp_path / "p1.png"]

        index_document(pdf, encoder, storage, tracker, config, dry_run=True)

        # PDF conversion and encoding DID happen
        mock_pdf2img.assert_called_once()
        encoder.encode_images.assert_called_once()

    @patch("indexing.index_documents.pdf_to_images")
    @patch("indexing.index_documents.save_page_images")
    def test_dry_run_with_force_still_processes(self, mock_save, mock_pdf2img, tmp_path):
        """dry_run + force: even if already indexed, process but don't write."""
        pdf = _make_fake_pdf(tmp_path)
        config = _make_config(tmp_path)
        encoder = _make_mock_encoder()
        storage = _make_mock_storage()

        tracker = IndexTracker(tracker_path=str(tmp_path / "tracker.json"))

        mock_pdf2img.return_value = [Image.new("RGB", (10, 10))]
        mock_save.return_value = [tmp_path / "p1.png"]

        # First: index normally
        result1 = index_document(pdf, encoder, storage, tracker, config)
        assert result1["status"] == "indexed"

        # Now dry_run + force: should process (not skip) but not write
        result2 = index_document(pdf, encoder, storage, tracker, config, force=True, dry_run=True)
        assert result2["status"] == "dry_run"
