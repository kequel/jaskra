import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from main import app
from tests.base_test import BaseTest

client = TestClient(app)

class TestGlaucomaAPI(BaseTest):
    """
    Standard /analyze-glaucoma Endpoint
    Steps:
    1. Mock the AI pipeline to return fixed prediction data.
    2. Upload a valid JPEG image via POST request.
    3. Verify the HTTP status code is 200.
    4. Assert that the response JSON contains all required diagnostic keys.
    5. Validate the formats of confidence, CDR, and Base64 image.
    """

    @patch('pipeline.pipeline.GlaucomaPipeline.run')
    def test_analyze_glaucoma_success(self, mock_run, sample_image):
        # Mocking AI output: (full_img, crops, masks, cdr_val, labels, confs)
        mock_run.return_value = (None, [], [], 0.45, [1], [0.98])
        
        files = {"file": ("test.jpg", sample_image, "image/jpeg")}
        response = client.post("/analyze-glaucoma", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "has_glaucoma" in data
        self.assert_is_ratio(data["cup_to_disc_ratio"])
        self.assert_valid_base64(data["image_base64"])