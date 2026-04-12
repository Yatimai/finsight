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


def _setup_chunked_mocks(mock_save, mock_chunked, mock_count, tmp_path, num_pages=1):
    """Configure mocks for chunked indexing pipeline."""
    mock_count.return_value = num_pages
    images = [Image.new("RGB", (10, 10)) for _ in range(num_pages)]
    mock_chunked.return_value = iter([(0, images)])
    mock_save.return_value = [tmp_path / f"page_{i + 1:04d}.png" for i in range(num_pages)]


class TestIndexDocumentForce:
    @patch("indexing.index_documents.pdf_page_count")
    @patch("indexing.index_documents.pdf_to_images_chunked")
    @patch("indexing.index_documents.save_page_images")
    def test_skips_already_indexed_without_force(self, mock_save, mock_chunked, mock_count, tmp_path):
        pdf = _make_fake_pdf(tmp_path)
        config = _make_config(tmp_path)
        encoder = _make_mock_encoder()
        storage = _make_mock_storage()

        tracker = IndexTracker(tracker_path=str(tmp_path / "tracker.json"))

        # First indexing
        _setup_chunked_mocks(mock_save, mock_chunked, mock_count, tmp_path)
        result1 = index_document(pdf, encoder, storage, tracker, config)
        assert result1["status"] == "indexed"

        # Second call without force → skip
        result2 = index_document(pdf, encoder, storage, tracker, config)
        assert result2["status"] == "skipped"
        assert result2["reason"] == "already_indexed"

    @patch("indexing.index_documents.pdf_page_count")
    @patch("indexing.index_documents.pdf_to_images_chunked")
    @patch("indexing.index_documents.save_page_images")
    def test_reindexes_with_force(self, mock_save, mock_chunked, mock_count, tmp_path):
        pdf = _make_fake_pdf(tmp_path)
        config = _make_config(tmp_path)
        encoder = _make_mock_encoder()
        storage = _make_mock_storage()

        tracker = IndexTracker(tracker_path=str(tmp_path / "tracker.json"))

        # First indexing
        _setup_chunked_mocks(mock_save, mock_chunked, mock_count, tmp_path)
        result1 = index_document(pdf, encoder, storage, tracker, config)
        assert result1["status"] == "indexed"

        # Second call WITH force → re-index
        _setup_chunked_mocks(mock_save, mock_chunked, mock_count, tmp_path)
        result2 = index_document(pdf, encoder, storage, tracker, config, force=True)
        assert result2["status"] == "indexed"


# ---------------------------------------------------------------------------
# TestDryRun
# ---------------------------------------------------------------------------


class TestDryRun:
    @patch("indexing.index_documents.pdf_page_count")
    @patch("indexing.index_documents.pdf_to_images_chunked")
    @patch("indexing.index_documents.save_page_images")
    def test_dry_run_skips_qdrant_and_tracker(self, mock_save, mock_chunked, mock_count, tmp_path):
        pdf = _make_fake_pdf(tmp_path)
        config = _make_config(tmp_path)
        encoder = _make_mock_encoder()
        storage = _make_mock_storage()

        tracker = IndexTracker(tracker_path=str(tmp_path / "tracker.json"))

        _setup_chunked_mocks(mock_save, mock_chunked, mock_count, tmp_path, num_pages=2)

        result = index_document(pdf, encoder, storage, tracker, config, dry_run=True)

        assert result["status"] == "dry_run"
        assert result["num_pages"] == 2

        # Qdrant was NOT called
        storage.store_page.assert_not_called()
        storage.get_next_id.assert_not_called()

        # Tracker was NOT written
        assert not tracker.is_indexed(result["document_id"])

    @patch("indexing.index_documents.pdf_page_count")
    @patch("indexing.index_documents.pdf_to_images_chunked")
    @patch("indexing.index_documents.save_page_images")
    def test_dry_run_still_calls_encoder(self, mock_save, mock_chunked, mock_count, tmp_path):
        pdf = _make_fake_pdf(tmp_path)
        config = _make_config(tmp_path)
        encoder = _make_mock_encoder()
        storage = _make_mock_storage()

        tracker = IndexTracker(tracker_path=str(tmp_path / "tracker.json"))

        _setup_chunked_mocks(mock_save, mock_chunked, mock_count, tmp_path)

        index_document(pdf, encoder, storage, tracker, config, dry_run=True)

        # Chunked conversion and encoding DID happen
        mock_chunked.assert_called_once()
        encoder.encode_images.assert_called_once()

    @patch("indexing.index_documents.pdf_page_count")
    @patch("indexing.index_documents.pdf_to_images_chunked")
    @patch("indexing.index_documents.save_page_images")
    def test_dry_run_with_force_still_processes(self, mock_save, mock_chunked, mock_count, tmp_path):
        """dry_run + force: even if already indexed, process but don't write."""
        pdf = _make_fake_pdf(tmp_path)
        config = _make_config(tmp_path)
        encoder = _make_mock_encoder()
        storage = _make_mock_storage()

        tracker = IndexTracker(tracker_path=str(tmp_path / "tracker.json"))

        # First: index normally
        _setup_chunked_mocks(mock_save, mock_chunked, mock_count, tmp_path)
        result1 = index_document(pdf, encoder, storage, tracker, config)
        assert result1["status"] == "indexed"

        # Now dry_run + force: should process (not skip) but not write
        _setup_chunked_mocks(mock_save, mock_chunked, mock_count, tmp_path)
        result2 = index_document(pdf, encoder, storage, tracker, config, force=True, dry_run=True)
        assert result2["status"] == "dry_run"


# ---------------------------------------------------------------------------
# TestChunkedIndexing
# ---------------------------------------------------------------------------


class TestChunkedIndexing:
    @patch("indexing.index_documents.pdf_page_count")
    @patch("indexing.index_documents.pdf_to_images_chunked")
    @patch("indexing.index_documents.save_page_images")
    def test_multi_chunk_stores_correct_page_numbers(self, mock_save, mock_chunked, mock_count, tmp_path):
        """5 pages, chunk_size=2 → 3 chunks, page numbers 1-5 stored correctly."""
        pdf = _make_fake_pdf(tmp_path)
        config = _make_config(tmp_path)
        config.data.chunk_size = 2
        encoder = _make_mock_encoder()
        storage = _make_mock_storage()
        tracker = IndexTracker(tracker_path=str(tmp_path / "tracker.json"))

        mock_count.return_value = 5

        # Simulate 3 chunks: (0, 2 images), (2, 2 images), (4, 1 image)
        chunk_data = [
            (0, [Image.new("RGB", (10, 10)), Image.new("RGB", (10, 10))]),
            (2, [Image.new("RGB", (10, 10)), Image.new("RGB", (10, 10))]),
            (4, [Image.new("RGB", (10, 10))]),
        ]
        mock_chunked.return_value = iter(chunk_data)

        # save_page_images is called per chunk
        mock_save.side_effect = [
            [tmp_path / "page_0001.png", tmp_path / "page_0002.png"],
            [tmp_path / "page_0003.png", tmp_path / "page_0004.png"],
            [tmp_path / "page_0005.png"],
        ]

        result = index_document(pdf, encoder, storage, tracker, config)

        assert result["status"] == "indexed"
        assert result["num_pages"] == 5

        # 5 pages stored in Qdrant
        assert storage.store_page.call_count == 5

        # Verify page numbers are 1..5
        stored_page_numbers = [
            call.kwargs["metadata"]["page_number"] if "metadata" in call.kwargs else call.args[4]["page_number"]
            for call in storage.store_page.call_args_list
        ]
        assert stored_page_numbers == [1, 2, 3, 4, 5]

        # Verify point IDs are sequential
        stored_point_ids = [call.args[0] for call in storage.store_page.call_args_list]
        assert stored_point_ids == [0, 1, 2, 3, 4]

    @patch("indexing.index_documents.pdf_page_count")
    @patch("indexing.index_documents.pdf_to_images_chunked")
    @patch("indexing.index_documents.save_page_images")
    def test_save_called_with_correct_offsets(self, mock_save, mock_chunked, mock_count, tmp_path):
        """Verify save_page_images receives correct page_offset per chunk."""
        pdf = _make_fake_pdf(tmp_path)
        config = _make_config(tmp_path)
        config.data.chunk_size = 2
        encoder = _make_mock_encoder()
        storage = _make_mock_storage()
        tracker = IndexTracker(tracker_path=str(tmp_path / "tracker.json"))

        mock_count.return_value = 3

        chunk_data = [
            (0, [Image.new("RGB", (10, 10)), Image.new("RGB", (10, 10))]),
            (2, [Image.new("RGB", (10, 10))]),
        ]
        mock_chunked.return_value = iter(chunk_data)
        mock_save.side_effect = [
            [tmp_path / "page_0001.png", tmp_path / "page_0002.png"],
            [tmp_path / "page_0003.png"],
        ]

        index_document(pdf, encoder, storage, tracker, config)

        # save_page_images called twice with correct offsets
        assert mock_save.call_count == 2
        assert mock_save.call_args_list[0].kwargs["page_offset"] == 0
        assert mock_save.call_args_list[1].kwargs["page_offset"] == 2
