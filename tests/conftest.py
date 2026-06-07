import pytest
import io
import cv2
import numpy as np
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

@pytest.fixture(scope="session")
def yolo_glaucoma_model():
    """
    Session-scoped fixture to load the YOLO model once for all tests.
    """
    from ultralytics import YOLO
    return YOLO('../../ai/yolo/weights/last.pt')


@pytest.fixture
def synthetic_fundus_image() -> np.ndarray:
    """Synthetic 512x512 BGR image mimicking a fundus photo with a bright disc region.

    A dark reddish background with a bright yellowish ellipse in the center
    gives the YOLO model a realistic-enough target to exercise its detection path.
    """
    img = np.full((512, 512, 3), (30, 20, 60), dtype=np.uint8)  # dark reddish BGR
    cv2.ellipse(img, (256, 256), (80, 80), 0, 0, 360, (180, 220, 240), -1)
    return img


@pytest.fixture
def sample_full_image_512() -> np.ndarray:
    """512x512 BGR image with a distinctive pattern for template-matching tests."""
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.circle(img, (256, 256), 100, (200, 200, 200), -1)
    cv2.rectangle(img, (200, 200), (312, 312), (255, 255, 255), -1)
    return img


@pytest.fixture
def sample_crop_from_center(sample_full_image_512) -> np.ndarray:
    """A crop taken from the center of sample_full_image_512 (200x200 region)."""
    return sample_full_image_512[156:356, 156:356].copy()