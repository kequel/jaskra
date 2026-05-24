import pytest
import io
import numpy as np
import os
import tempfile
import cv2
import torch
from PIL import Image

"""Pytest configuration and shared fixtures for test suite.

This module provides common test fixtures used across the test suite for API,
pipeline, and other components, including sample images, masks, and mock outputs.
"""

@pytest.fixture
def sample_image() -> io.BytesIO:
    """Dummy JPEG used by both API and pipeline tests."""
    img = Image.new("RGB", (100, 100), color=(180, 100, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


@pytest.fixture
def sample_image_bytes(sample_image) -> bytes:
    """Raw bytes from sample_image - useful when bytes are needed instead of BytesIO."""
    return sample_image.read()


@pytest.fixture
def mock_disc_mask() -> np.ndarray:
    """Binary disc mask (512x512) with a filled circle in the center."""
    mask = np.zeros((512, 512), dtype=np.float32)
    for y in range(512):
        for x in range(512):
            if (x - 256) ** 2 + (y - 256) ** 2 <= 150 ** 2:
                mask[y, x] = 1.0
    return mask


@pytest.fixture
def mock_cup_mask() -> np.ndarray:
    """Binary cup mask - smaller circle inside the disc (CDR ~0.5)."""
    mask = np.zeros((512, 512), dtype=np.float32)
    for y in range(512):
        for x in range(512):
            if (x - 256) ** 2 + (y - 256) ** 2 <= 75 ** 2:
                mask[y, x] = 1.0
    return mask


@pytest.fixture
def mock_segmentation_mask() -> np.ndarray:
    """Empty segmentation mask (100x100, uint8)."""
    return np.zeros((100, 100), dtype=np.uint8)


@pytest.fixture
def mock_pipeline_output_positive():
    """
    Simulates GlaucomaPipeline.run() output for a glaucoma-positive case.
    CDR = 0.70, which exceeds the 0.65 threshold → has_glaucoma = True.
    """
    full_img = np.zeros((100, 100, 3), dtype=np.uint8)
    crops = [(10, 10, 90, 90)]
    masks = [np.zeros((2, 80, 80), dtype=np.float32)]
    cdr_val = 0.70
    gt_masks = None
    cdr_gt = 0.0
    return (full_img, crops, masks, cdr_val, gt_masks, cdr_gt)


@pytest.fixture
def mock_pipeline_output_negative():
    """
    Simulates GlaucomaPipeline.run() output for a glaucoma-negative case.
    CDR = 0.40, which is below the 0.65 threshold → has_glaucoma = False.
    """
    full_img = np.zeros((100, 100, 3), dtype=np.uint8)
    crops = [(10, 10, 90, 90)]
    masks = [np.zeros((2, 80, 80), dtype=np.float32)]
    cdr_val = 0.40
    gt_masks = None
    cdr_gt = 0.0
    return (full_img, crops, masks, cdr_val, gt_masks, cdr_gt)


@pytest.fixture
def dummy_arrays():
    """Returns a dictionary with numpy and torch arrays for unit testing."""
    pred_disc = np.zeros((20, 20), dtype=np.uint8)
    pred_disc[5:15, 5:15] = 1
    pred_disc[1, 1] = 1

    pred_cup = np.zeros((20, 20), dtype=np.uint8)
    pred_cup[8:12, 8:12] = 1
    pred_cup[18, 18] = 1

    return {
        "mask_b": np.array([[0, 100], [255, 255]], dtype=np.uint8),
        "mask_norm": np.array([[0, 1], [2, 0]], dtype=np.uint8),
        "logits": torch.tensor([[[[1.0, 1.0], [-1.0, -1.0]], [[1.0, -1.0], [-1.0, -1.0]]]]),
        "targets": torch.tensor([[[[1.0, 1.0], [0.0, 0.0]], [[1.0, 1.0], [0.0, 0.0]]]]),
        "pred_disc": pred_disc,
        "pred_cup": pred_cup,
        "dummy_out": torch.randn(2, 2, 10, 10),
        "dummy_mask": torch.randint(0, 2, (2, 2, 10, 10)).float()
    }


@pytest.fixture
def dummy_dataset():
    """Generates a temporary filesystem with images and masks for dataset tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = os.path.join(tmpdir, "images")
        mask_dir = os.path.join(tmpdir, "masks")
        save_dir = os.path.join(tmpdir, "models")
        os.makedirs(img_dir)
        os.makedirs(mask_dir)
        os.makedirs(save_dir)

        cv2.imwrite(os.path.join(img_dir, "test_eye.png"), np.zeros((10, 10, 3), dtype=np.uint8))
        cv2.imwrite(os.path.join(mask_dir, "test_eye.png"), np.zeros((10, 10), dtype=np.uint8))
        cv2.imwrite(os.path.join(mask_dir, "test_eye_cup.png"), np.zeros((10, 10), dtype=np.uint8))
        cv2.imwrite(os.path.join(mask_dir, "test_eye_disc.png"), np.zeros((10, 10), dtype=np.uint8))

        cv2.imwrite(os.path.join(img_dir, "skipped_eye.png"), np.zeros((10, 10, 3), dtype=np.uint8))
        cv2.imwrite(os.path.join(img_dir, "broken_sample.png"), np.zeros((10, 10, 3), dtype=np.uint8))
        cv2.imwrite(os.path.join(mask_dir, "broken_sample.png"), np.zeros((10, 10), dtype=np.uint8))

        for i in range(3):
            cv2.imwrite(os.path.join(img_dir, f"test_{i}.png"), np.zeros((16, 16, 3), dtype=np.uint8))
            cv2.imwrite(os.path.join(mask_dir, f"test_{i}.png"), np.zeros((16, 16), dtype=np.uint8))

        yield img_dir, mask_dir, save_dir