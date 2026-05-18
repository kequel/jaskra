import pytest
from fastapi.testclient import TestClient

from main import app
from tests.base_test import BaseTest

client = TestClient(app)


class TestInputValidation(BaseTest):
    """
    Input validation tests for the POST /analyze-glaucoma endpoint.
    No mocks - verifies that malformed requests are rejected before reaching the pipeline.
    """

    def test_missing_file_returns_422(self):
        """
        TEST NAME: test_missing_file_returns_422
        COMPONENT: POST /analyze-glaucoma, FastAPI request validation
        1. Send POST request to /analyze-glaucoma with no body.
        2. Assert HTTP 422 Unprocessable Entity (FastAPI rejects missing required field).
        """
        response = client.post("/analyze-glaucoma")
        assert response.status_code == 422

    def test_invalid_content_type_not_200(self):
        """
        TEST NAME: test_invalid_content_type_not_200
        COMPONENT: POST /analyze-glaucoma, PIL.Image.open() in main.py
        1. Send a plain text file with content-type text/plain.
        2. Assert response is not HTTP 200 (PIL.Image.open() should raise on non-image bytes).
        """
        files = {"file": ("test.txt", b"not an image", "text/plain")}
        response = client.post("/analyze-glaucoma", files=files)
        assert response.status_code != 200

    def test_empty_file_not_200(self):
        """
        TEST NAME: test_empty_file_not_200
        COMPONENT: POST /analyze-glaucoma, PIL.Image.open() in main.py
        1. Send a file with image/jpeg content-type but zero bytes of content.
        2. Assert response is not HTTP 200.
        """
        files = {"file": ("empty.jpg", b"", "image/jpeg")}
        response = client.post("/analyze-glaucoma", files=files)
        assert response.status_code != 200

    def test_corrupted_jpeg_not_200(self):
        """
        TEST NAME: test_corrupted_jpeg_not_200
        COMPONENT: POST /analyze-glaucoma, PIL.Image.open() in main.py
        1. Send a file with valid JPEG magic bytes (0xFF 0xD8 0xFF) followed by garbage data.
        2. Assert response is not HTTP 200 (PIL should fail to decode truncated JPEG).
        """
        files = {"file": ("corrupt.jpg", b"\xff\xd8\xff" + b"\x00" * 50, "image/jpeg")}
        response = client.post("/analyze-glaucoma", files=files)
        assert response.status_code != 200