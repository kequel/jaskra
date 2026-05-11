import base64
import io
import pytest
import httpx
from PIL import Image


pytestmark = pytest.mark.integration  # this file runs only with -m integration


BASE_URL = "http://localhost:8000"


@pytest.fixture(scope="module")
def http_client():
    """httpx client with timeout - assumes a running server on localhost:8000."""
    with httpx.Client(base_url=BASE_URL, timeout=30) as client:
        yield client


@pytest.fixture(scope="module")
def real_image_bytes() -> bytes:
    """Real JPEG image as bytes."""
    img = Image.new("RGB", (100, 100), color=(180, 100, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestGlaucomaIntegration:
    """
    Integration tests for the POST /analyze-glaucoma endpoint against a live FastAPI server.
    No mocks - verifies the full stack: server, pipeline, and model inference.
    Requires a running server on localhost:8000.
    """

    def test_endpoint_responds_200(self, http_client, real_image_bytes):
        """
        TEST NAME: test_endpoint_responds_200
        COMPONENT: POST /analyze-glaucoma, full stack (server + pipeline + models)
        1. Send a real JPEG image to the live server.
        2. Assert HTTP 200 status code.
        """
        response = http_client.post(
            "/analyze-glaucoma",
            files={"file": ("test.jpg", real_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200

    def test_response_structure(self, http_client, real_image_bytes):
        """
        TEST NAME: test_response_structure
        COMPONENT: POST /analyze-glaucoma, JSONResponse in main.py
        1. Send a real JPEG image to the live server.
        2. Parse JSON response.
        3. Assert all four required keys are present: has_glaucoma, confidence, cup_to_disc_ratio, image_base64.
        """
        response = http_client.post(
            "/analyze-glaucoma",
            files={"file": ("test.jpg", real_image_bytes, "image/jpeg")},
        )
        data = response.json()
        expected = {"has_glaucoma", "confidence", "cup_to_disc_ratio", "image_base64"}
        assert expected <= data.keys(), f"Missing keys: {expected - data.keys()}"

    def test_response_value_types(self, http_client, real_image_bytes):
        """
        TEST NAME: test_response_value_types
        COMPONENT: POST /analyze-glaucoma, value construction in main.py
        1. Send a real JPEG image to the live server.
        2. Assert has_glaucoma is a bool, confidence and cup_to_disc_ratio are in [0.0, 1.0].
        3. Assert image_base64 decodes without error.
        """
        response = http_client.post(
            "/analyze-glaucoma",
            files={"file": ("test.jpg", real_image_bytes, "image/jpeg")},
        )
        data = response.json()

        assert isinstance(data["has_glaucoma"], bool)
        assert 0.0 <= data["confidence"] <= 1.0
        assert 0.0 <= data["cup_to_disc_ratio"] <= 1.0
        base64.b64decode(data["image_base64"])  # raises if invalid

    def test_missing_file_returns_422(self, http_client):
        """
        TEST NAME: test_missing_file_returns_422
        COMPONENT: POST /analyze-glaucoma, FastAPI request validation (live server)
        1. Send POST with no body to the live server.
        2. Assert HTTP 422 Unprocessable Entity.
        """
        response = http_client.post("/analyze-glaucoma")
        assert response.status_code == 422