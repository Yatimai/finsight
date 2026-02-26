"""Tests for PDF/image utilities."""

from PIL import Image

from indexing.utils import (
    compute_document_hash,
    encode_image_base64,
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


class TestEncodeImageBase64:
    def test_encodes_existing_image(self, tmp_path):
        img = Image.new("RGB", (10, 10), color="blue")
        path = tmp_path / "test.png"
        img.save(path, "PNG")
        result = encode_image_base64(path)
        assert result is not None
        assert len(result) > 0

    def test_returns_none_for_missing(self):
        result = encode_image_base64("/nonexistent/path.png")
        assert result is None
