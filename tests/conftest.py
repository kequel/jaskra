import pytest
import io
from PIL import Image

@pytest.fixture
def sample_image():
    """
    Generates a dummy eye image for testing pipeline and API.
    """
    img = Image.new("RGB", (100, 100), color=(180, 100, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf

@pytest.fixture
def mock_segmentation_mask():
    """
    Returns a dummy numpy-like array representing a mask.
    """
    import numpy as np
    return np.zeros((100, 100), dtype=np.uint8)