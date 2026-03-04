"""
PDF processing utilities.
Converts PDF pages to images and computes document hashes for idempotency.
"""

import hashlib
from collections.abc import Generator
from pathlib import Path

from PIL import Image


def compute_document_hash(pdf_path: str | Path) -> str:
    """Compute SHA-256 hash of a PDF file for idempotency checks."""
    sha256 = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def pdf_to_images(
    pdf_path: str | Path,
    dpi: int = 300,
) -> list[Image.Image]:
    """
    Convert a PDF file to a list of PIL Images, one per page.

    Uses pymupdf (fitz) for conversion — faster and lighter than pdf2image/poppler.
    Falls back to pdf2image if pymupdf is not available.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        return _pdf_to_images_pymupdf(pdf_path, dpi)
    except ImportError:
        return _pdf_to_images_pdf2image(pdf_path, dpi)


def _pdf_to_images_pymupdf(
    pdf_path: Path, dpi: int, start_page: int = 0, end_page: int | None = None
) -> list[Image.Image]:
    """Convert PDF to images using pymupdf (fitz).

    Args:
        pdf_path: Path to the PDF file.
        dpi: Resolution for rendering.
        start_page: First page index (0-based, inclusive).
        end_page: Last page index (0-based, exclusive). None = all pages.
    """
    import fitz  # pymupdf

    images = []
    zoom = dpi / 72  # fitz uses 72 DPI by default
    matrix = fitz.Matrix(zoom, zoom)

    doc = fitz.open(str(pdf_path))
    try:
        if end_page is None:
            end_page = len(doc)
        for page_idx in range(start_page, min(end_page, len(doc))):
            page = doc[page_idx]
            pix = page.get_pixmap(matrix=matrix)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(img)
    finally:
        doc.close()

    return images


def _pdf_to_images_pdf2image(pdf_path: Path, dpi: int) -> list[Image.Image]:
    """Fallback: convert PDF to images using pdf2image (requires poppler)."""
    from pdf2image import convert_from_path

    return convert_from_path(str(pdf_path), dpi=dpi)


def pdf_page_count(pdf_path: str | Path) -> int:
    """Return the number of pages in a PDF without loading images into memory."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        import fitz  # pymupdf

        doc = fitz.open(str(pdf_path))
        try:
            return len(doc)
        finally:
            doc.close()
    except ImportError:
        from pdf2image import pdfinfo_from_path

        info = pdfinfo_from_path(str(pdf_path))
        return info["Pages"]


def pdf_to_images_chunked(
    pdf_path: str | Path,
    dpi: int = 300,
    chunk_size: int = 50,
) -> Generator[tuple[int, list[Image.Image]], None, None]:
    """
    Yield chunks of page images from a PDF to limit peak memory usage.

    Yields:
        Tuples of (chunk_start_page_0indexed, list_of_images).
        chunk_start is 0-based so page_number = chunk_start + i + 1.
    """
    pdf_path = Path(pdf_path)
    total = pdf_page_count(pdf_path)

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        try:
            images = _pdf_to_images_pymupdf(pdf_path, dpi, start_page=start, end_page=end)
        except ImportError:
            from pdf2image import convert_from_path

            images = convert_from_path(str(pdf_path), dpi=dpi, first_page=start + 1, last_page=end)
        yield start, images


def save_page_images(
    images: list[Image.Image],
    output_dir: str | Path,
    document_id: str,
    page_offset: int = 0,
) -> list[Path]:
    """
    Save page images to disk as PNGs.
    Returns list of saved file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create document subdirectory
    doc_dir = output_dir / document_id
    doc_dir.mkdir(exist_ok=True)

    saved_paths = []
    for page_num, img in enumerate(images, start=page_offset + 1):
        filename = f"page_{page_num:04d}.png"
        filepath = doc_dir / filename
        img.save(filepath, "PNG")
        saved_paths.append(filepath)

    return saved_paths


def load_page_image(page_path: str | Path) -> Image.Image:
    """Load a single page image from disk."""
    return Image.open(page_path).convert("RGB")


def encode_image_base64(image_path: str | Path) -> str | None:
    """
    Load and base64-encode a page image.
    Returns None if the file doesn't exist.
    """
    import base64

    image_path = Path(image_path)
    if not image_path.exists():
        return None

    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def iter_pdf_files(directory: str | Path) -> Generator[Path, None, None]:
    """Iterate over all PDF files in a directory."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    for path in sorted(directory.glob("*.pdf")):
        yield path

    # Also check subdirectories one level deep
    for path in sorted(directory.glob("*/*.pdf")):
        yield path
