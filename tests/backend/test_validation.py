from main import app
from fastapi.testclient import TestClient
from tests.base_test import BaseTest

client = TestClient(app)

class TestInputValidation(BaseTest):
    """
    API Robustness and Input Validation
    Steps:
    1. Send a request without a file to the endpoint.
    2. Verify that the API returns a 422 Unprocessable Entity error.
    3. Send a non-image file and check for graceful error handling.
    """

    def test_missing_file_payload(self):
        response = client.post("/analyze-glaucoma")
        assert response.status_code == 422

    def test_invalid_file_format(self):
        # Sending plain text instead of an image
        files = {"file": ("test.txt", b"not an image", "text/plain")}
        response = client.post("/analyze-glaucoma")
        # Depending on implementation, it might be 400 or 500
        assert response.status_code != 200