"""Tests for indexing.preprocessing — border cropping for document pages."""

import numpy as np
from PIL import Image

from indexing.preprocessing import CropConfig, crop_empty


class TestCropEmpty:
    def test_all_white_image_returns_original(self):
        """A fully white image has no content to crop to → returns original."""
        white = Image.new("RGB", (200, 300), (255, 255, 255))
        cropped, meta = crop_empty(white)
        assert not meta["applied"]
        assert meta["crop_box"] is None
        assert cropped.size == white.size

    def test_image_with_content_gets_cropped(self):
        """An image with a black rectangle on white background gets cropped."""
        img = Image.new("RGB", (400, 600), (255, 255, 255))
        arr = np.array(img)
        # Draw a black block in the center (rows 100-400, cols 50-350)
        arr[100:400, 50:350, :] = 0
        img = Image.fromarray(arr)

        cropped, meta = crop_empty(img)
        assert meta["applied"]
        assert meta["crop_box"] is not None

        # Cropped should be smaller than original
        assert cropped.width < img.width or cropped.height < img.height
        # But still contain the content
        assert cropped.width >= 300  # 350 - 50 = 300 content cols
        assert cropped.height >= 300  # 400 - 100 = 300 content rows

    def test_preserves_content_pixels(self):
        """Content pixels must survive cropping unchanged."""
        img = Image.new("RGB", (200, 200), (255, 255, 255))
        arr = np.array(img)
        # Red square at (50, 50) to (100, 100)
        arr[50:100, 50:100] = [255, 0, 0]
        img = Image.fromarray(arr)

        cropped, meta = crop_empty(img)
        assert meta["applied"]

        cropped_arr = np.array(cropped)
        # The red pixels should still exist in the cropped image
        red_mask = (cropped_arr[:, :, 0] == 255) & (cropped_arr[:, :, 1] == 0) & (cropped_arr[:, :, 2] == 0)
        assert red_mask.sum() == 50 * 50  # 2500 red pixels preserved

    def test_no_crop_on_full_content_image(self):
        """An image that's entirely dark has no borders to crop → crop box is nearly full size."""
        dark = Image.new("RGB", (100, 100), (50, 50, 50))
        cropped, meta = crop_empty(dark)
        assert meta["applied"]
        # Should be almost the full image (minus preserve_border_px adjustments)
        assert cropped.width >= 98
        assert cropped.height >= 98

    def test_custom_config(self):
        """Custom config parameters are respected."""
        img = Image.new("RGB", (200, 200), (255, 255, 255))
        arr = np.array(img)
        # Light gray border area (pixel value 245 > default threshold 240)
        arr[10:190, 10:190] = [230, 230, 230]  # Below threshold → content
        img = Image.fromarray(arr)

        config = CropConfig(color_threshold=240, preserve_border_px=5)
        _, meta = crop_empty(img, config=config)
        assert meta["applied"]

    def test_preserve_border_px(self):
        """preserve_border_px adds padding around the detected content box."""
        img = Image.new("RGB", (300, 300), (255, 255, 255))
        arr = np.array(img)
        arr[100:200, 100:200, :] = 0
        img = Image.fromarray(arr)

        config_no_pad = CropConfig(preserve_border_px=0)
        cropped_no_pad, _ = crop_empty(img, config=config_no_pad)

        config_with_pad = CropConfig(preserve_border_px=10)
        cropped_with_pad, _ = crop_empty(img, config=config_with_pad)

        # With padding, the result should be larger
        assert cropped_with_pad.width >= cropped_no_pad.width
        assert cropped_with_pad.height >= cropped_no_pad.height
