import pytest
import io
import numpy as np
from PIL import Image

"""Pytest configuration and shared fixtures for test suite.

This module provides common test fixtures used across the test suite for API,
pipeline, and other components, including sample images, masks, and mock outputs.
"""

@pytest.fixture
def sample_image() -> io.BytesIO:
    """Dummy JPEG image used by both API and pipeline tests."""
    img = Image.new("RGB", (100, 100), color=(180, 100, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf

@pytest.fixture
def sample_image_bytes(sample_image) -> bytes:
    """Raw bytes from sample_image - useful when bytes need to be passed instead of BytesIO."""
    return sample_image.read()

@pytest.fixture
def mock_segmentation_mask() -> np.ndarray:
    """Dummy segmentation mask (100×100, uint8)."""
    return np.zeros((100, 100), dtype=np.uint8)

@pytest.fixture
def mock_segmentation_mask_with_disc() -> np.ndarray:
    """Mask with a circle representing the optic disc."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    # tarcza w centrum: 20px radius
    center = (50, 50)
    for y in range(100):
        for x in range(100):
            if (x - center[0]) ** 2 + (y - center[1]) ** 2 <= 20 ** 2:
                mask[y, x] = 255
    return mask

@pytest.fixture
def mock_pipeline_output_positive():
    """
    Simulates GlaucomaPipeline.run() output for positive case (glaucoma present).
    Format: (full_img, crops, masks, cdr_val, labels, confs)
    """
    return (None, [], [], 0.70, [1], [0.98])

@pytest.fixture
def mock_pipeline_output_negative():
    """
    Simulates GlaucomaPipeline.run() output for negative case (no glaucoma).
    Format: (full_img, crops, masks, cdr_val, labels, confs)
    """
    return (None, [], [], 0.30, [0], [0.85])