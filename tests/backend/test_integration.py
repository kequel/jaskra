import base64
import io
import pytest
import httpx
from PIL import Image


pytestmark = pytest.mark.integration  # this file runs only with -m integration


BASE_URL = "http://localhost:8000"


@pytest.fixture(scope="module")
def http_client():
    """httpx client with timeout - assumes running server on localhost:8000."""
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
    Integration tests - require a running FastAPI server.

    Run in CI 'integration-tests' job with real server in background.
    Locally: first run `uvicorn main:app`, then `pytest -m integration`.
    """

    def test_endpoint_responds_200(self, http_client, real_image_bytes):
        """Server responds with 200 to valid request."""
        response = http_client.post(
            "/analyze-glaucoma",
            files={"file": ("test.jpg", real_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200

    def test_response_structure(self, http_client, real_image_bytes):
        """Response contains all required fields."""
        response = http_client.post(
            "/analyze-glaucoma",
            files={"file": ("test.jpg", real_image_bytes, "image/jpeg")},
        )
        data = response.json()
        expected = {"has_glaucoma", "confidence", "cup_to_disc_ratio", "image_base64"}
        assert expected <= data.keys(), f"Missing keys: {expected - data.keys()}"

    def test_response_value_types(self, http_client, real_image_bytes):
        """Fields have correct types and ranges."""
        response = http_client.post(
            "/analyze-glaucoma",
            files={"file": ("test.jpg", real_image_bytes, "image/jpeg")},
        )
        data = response.json()

        assert isinstance(data["has_glaucoma"], bool)
        assert 0.0 <= data["confidence"] <= 1.0
        assert 0.0 <= data["cup_to_disc_ratio"] <= 1.0
        base64.b64decode(data["image_base64"])  # raises exception if invalid

    def test_missing_file_returns_422(self, http_client):
        """Server rejects request without file."""
        response = http_client.post("/analyze-glaucoma")
        assert response.status_code == 422