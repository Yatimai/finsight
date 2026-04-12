"""
Border cropping for document page images.

Removes empty white margins before encoding with ColQwen2.5,
so the vision model focuses on actual content (tables, charts, text)
rather than wasting tokens on blank borders.

Adapted from Visual RAG Toolkit's crop_empty implementation.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CropConfig:
    """Configuration for border cropping."""

    color_threshold: int = 240
    min_white_fraction: float = 0.99
    content_density_sides: float = 0.001
    content_density_bottom: float = 1e-6
    preserve_border_px: int = 1


def crop_empty(image: Image.Image, config: CropConfig | None = None) -> tuple[Image.Image, dict]:
    """
    Crop empty white borders from a document page image.

    Scans rows/columns from each edge inward to find where content starts,
    then crops to the content bounding box (plus a small padding).

    Args:
        image: PIL Image to crop
        config: Cropping parameters (uses defaults if None)

    Returns:
        Tuple of (cropped_image, metadata_dict) where metadata includes
        whether cropping was applied and the crop box coordinates.
    """
    if config is None:
        config = CropConfig()

    img = image.convert("RGB")
    arr = np.array(img)
    intensity = arr.mean(axis=2)

    def _find_border_start(axis: int) -> int:
        """Find first row/col with enough non-white content."""
        size = intensity.shape[axis]
        for i in range(size):
            pixels = intensity[i, :] if axis == 0 else intensity[:, i]
            white_frac = float(np.mean(pixels > config.color_threshold))
            non_white = 1.0 - white_frac
            if (white_frac < config.min_white_fraction) and (non_white > config.content_density_sides):
                return i
        return size

    def _find_border_end(axis: int, min_density: float) -> int:
        """Find last row/col with enough non-white content."""
        size = intensity.shape[axis]
        for i in range(size - 1, -1, -1):
            pixels = intensity[i, :] if axis == 0 else intensity[:, i]
            white_frac = float(np.mean(pixels > config.color_threshold))
            non_white = 1.0 - white_frac
            if (white_frac < config.min_white_fraction) and (non_white > min_density):
                return i + 1
        return 0

    top = _find_border_start(0)
    left = _find_border_start(1)
    right = _find_border_end(1, config.content_density_sides)
    bottom = _find_border_end(0, config.content_density_bottom)

    width, height = img.size
    pad = max(config.preserve_border_px, 0)
    if pad > 0:
        left = max(left - pad, 0)
        top = max(top - pad, 0)
        right = min(right + pad, width)
        bottom = min(bottom + pad, height)

    crop_box = (left, top, right, bottom)
    valid = 0 <= crop_box[0] < crop_box[2] <= width and 0 <= crop_box[1] < crop_box[3] <= height

    if not valid:
        return image, {"applied": False, "crop_box": None, "original_size": (width, height)}

    cropped = img.crop(crop_box)
    return cropped, {
        "applied": True,
        "crop_box": list(crop_box),
        "original_size": (width, height),
        "cropped_size": (cropped.width, cropped.height),
    }
