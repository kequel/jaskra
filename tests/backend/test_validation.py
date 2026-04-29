import pytest
from fastapi.testclient import TestClient

from main import app
from tests.base_test import BaseTest

client = TestClient(app)


class TestInputValidation(BaseTest):
    """
    Input validation tests - verify API behavior with malformed data.
    Do not require pipeline mocks - server should reject requests early.
    """

    def test_missing_file_returns_422(self):
        """Missing file returns 422 Unprocessable Entity (FastAPI validation)."""
        response = client.post("/analyze-glaucoma")
        assert response.status_code == 422

    def test_invalid_content_type_not_200(self):
        """Text file instead of image should not return 200."""
        files = {"file": ("test.txt", b"not an image", "text/plain")}
        response = client.post("/analyze-glaucoma", files=files)
        assert response.status_code != 200

    def test_empty_file_not_200(self):
        """Empty file should not return 200."""
        files = {"file": ("empty.jpg", b"", "image/jpeg")}
        response = client.post("/analyze-glaucoma", files=files)
        assert response.status_code != 200

    def test_corrupted_jpeg_not_200(self):
        """File with .jpg extension but corrupted data should not return 200."""
        files = {"file": ("corrupt.jpg", b"\xff\xd8\xff" + b"\x00" * 50, "image/jpeg")}
        response = client.post("/analyze-glaucoma", files=files)
        assert response.status_code != 200