import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from main import app
from tests.base_test import BaseTest

client = TestClient(app)

class TestGlaucomaAPI(BaseTest):
    """
    Unit tests for the /analyze-glaucoma endpoint.

    Uses TestClient (without real server) and mocks the pipeline.
    Verifies HTTP logic: status codes, response structure, data formats.
    Marked as unit test (no 'integration' marker by default).
    """

    @patch("pipeline.pipeline.GlaucomaPipeline.run")
    def test_success_positive_diagnosis(self, mock_run, sample_image, mock_pipeline_output_positive):
        """Endpoint returns 200 with correct structure for detected glaucoma."""
        mock_run.return_value = mock_pipeline_output_positive

        response = client.post(
            "/analyze-glaucoma",
            files={"file": ("test.jpg", sample_image, "image/jpeg")},
        )

        assert response.status_code == 200
        data = response.json()
        self.assert_glaucoma_response(data)
        assert data["has_glaucoma"] is True

    @patch("pipeline.pipeline.GlaucomaPipeline.run")
    def test_success_negative_diagnosis(self, mock_run, sample_image, mock_pipeline_output_negative):
        """Endpoint returns 200 for healthy eye."""
        mock_run.return_value = mock_pipeline_output_negative

        response = client.post(
            "/analyze-glaucoma",
            files={"file": ("test.jpg", sample_image, "image/jpeg")},
        )

        assert response.status_code == 200
        data = response.json()
        self.assert_glaucoma_response(data)
        assert data["has_glaucoma"] is False

    @patch("pipeline.pipeline.GlaucomaPipeline.run")
    def test_cdr_boundary_values(self, mock_run, sample_image):
        """CDR = 0.0 and CDR = 1.0 are boundary values - should pass validation."""
        for cdr in [0.0, 1.0]:
            mock_run.return_value = (None, [], [], cdr, [0], [0.5])
            response = client.post(
                "/analyze-glaucoma",
                files={"file": ("test.jpg", sample_image, "image/jpeg")},
            )
            assert response.status_code == 200
            self.assert_is_ratio(response.json()["cup_to_disc_ratio"])