"""Tests for PDF/image utilities."""

from unittest.mock import MagicMock, patch

from PIL import Image

from indexing.utils import (
    compute_document_hash,
    encode_image_base64,
    pdf_page_count,
    pdf_to_images_chunked,
    save_page_images,
)


class TestComputeDocumentHash:
    def test_consistent_hash(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"fake pdf content")
        h1 = compute_document_hash(pdf)
        h2 = compute_document_hash(pdf)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_different_content_different_hash(self, tmp_path):
        pdf1 = tmp_path / "test1.pdf"
        pdf2 = tmp_path / "test2.pdf"
        pdf1.write_bytes(b"content A")
        pdf2.write_bytes(b"content B")
        assert compute_document_hash(pdf1) != compute_document_hash(pdf2)


class TestSavePageImages:
    def test_saves_images(self, tmp_path):
        images = [Image.new("RGB", (100, 100), color="red") for _ in range(3)]
        paths = save_page_images(images, tmp_path / "pages", "doc123")
        assert len(paths) == 3
        assert all(p.exists() for p in paths)
        assert paths[0].name == "page_0001.png"
        assert paths[2].name == "page_0003.png"

    def test_creates_doc_subdirectory(self, tmp_path):
        images = [Image.new("RGB", (50, 50))]
        save_page_images(images, tmp_path / "pages", "abc")
        assert (tmp_path / "pages" / "abc").is_dir()

    def test_saves_with_page_offset(self, tmp_path):
        images = [Image.new("RGB", (50, 50)) for _ in range(2)]
        paths = save_page_images(images, tmp_path / "pages", "doc1", page_offset=10)
        assert paths[0].name == "page_0011.png"
        assert paths[1].name == "page_0012.png"
        assert all(p.exists() for p in paths)


class TestEncodeImageBase64:
    def test_encodes_existing_image(self, tmp_path):
        img = Image.new("RGB", (10, 10), color="blue")
        path = tmp_path / "test.png"
        img.save(path, "PNG")
        result = encode_image_base64(path)
        assert result is not None
        data, media_type = result
        assert len(data) > 0
        assert media_type == "image/png"

    def test_returns_none_for_missing(self):
        result = encode_image_base64("/nonexistent/path.png")
        assert result is None

    def test_large_image_compressed_to_jpeg(self, tmp_path):
        """Images > 4 MB are re-encoded as JPEG to stay under the 5 MB API limit."""
        import base64

        # Create a large image (~5+ MB as PNG via random noise)
        import numpy as np

        arr = np.random.randint(0, 255, (3000, 3000, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        path = tmp_path / "large.png"
        img.save(path, "PNG")
        assert path.stat().st_size > 4 * 1024 * 1024, "Test image must be > 4 MB"

        result = encode_image_base64(path)
        assert result is not None
        data, media_type = result
        assert media_type == "image/jpeg"

        raw = base64.b64decode(data)
        assert len(raw) < 5 * 1024 * 1024, f"Compressed image too large: {len(raw)}"
        assert raw[:2] == b"\xff\xd8", "Not a valid JPEG"


class TestPdfPageCount:
    @patch("indexing.utils.fitz", create=True)
    def test_returns_page_count(self, mock_fitz_module, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=7)

        with patch("indexing.utils._pdf_to_images_pymupdf"), patch.dict("sys.modules", {"fitz": mock_fitz_module}):
            mock_fitz_module.open.return_value = mock_doc
            result = pdf_page_count(pdf)

        assert result == 7
        mock_doc.close.assert_called_once()

    def test_raises_for_missing_file(self):
        import pytest

        with pytest.raises(FileNotFoundError):
            pdf_page_count("/nonexistent/file.pdf")


class TestPdfToImagesChunked:
    @patch("indexing.utils.pdf_page_count")
    @patch("indexing.utils._pdf_to_images_pymupdf")
    def test_yields_chunks(self, mock_pymupdf, mock_count, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")

        mock_count.return_value = 5

        def fake_pymupdf(path, dpi, start_page=0, end_page=None):
            count = (end_page or 5) - start_page
            return [Image.new("RGB", (10, 10)) for _ in range(count)]

        mock_pymupdf.side_effect = fake_pymupdf

        chunks = list(pdf_to_images_chunked(pdf, dpi=72, chunk_size=2))

        assert len(chunks) == 3  # 2 + 2 + 1
        assert chunks[0][0] == 0
        assert len(chunks[0][1]) == 2
        assert chunks[1][0] == 2
        assert len(chunks[1][1]) == 2
        assert chunks[2][0] == 4
        assert len(chunks[2][1]) == 1

    @patch("indexing.utils.pdf_page_count")
    @patch("indexing.utils._pdf_to_images_pymupdf")
    def test_single_chunk_when_small(self, mock_pymupdf, mock_count, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")

        mock_count.return_value = 3
        mock_pymupdf.return_value = [Image.new("RGB", (10, 10)) for _ in range(3)]

        chunks = list(pdf_to_images_chunked(pdf, dpi=72, chunk_size=50))

        assert len(chunks) == 1
        assert chunks[0][0] == 0
        assert len(chunks[0][1]) == 3
