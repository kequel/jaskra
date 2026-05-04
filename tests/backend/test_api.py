import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from main import app
from tests.base_test import BaseTest

client = TestClient(app)

PIPELINE_PATCH = "pipeline.pipeline.GlaucomaPipeline.run"


class TestGlaucomaAPI(BaseTest):
    """
    This class contains unit tests for the /analyze-glaucoma endpoint, 
    focusing on the API's response structure and logic based on mocked pipeline outputs.
    """

    @patch(PIPELINE_PATCH)
    def test_success_positive_diagnosis(self, mock_run, sample_image, mock_pipeline_output_positive):
        """
        TEST NAME: test_success_positive_diagnosis
        COMPONENT: POST /analyze-glaucoma
        1. Mock GlaucomaPipeline.run() to return CDR=0.70 (above 0.65 threshold).
        2. POST a valid JPEG image to /analyze-glaucoma.
        3. Assert HTTP 200 status code.
        4. Assert response contains all required keys and valid formats.
        5. Assert has_glaucoma is True (CDR > 0.65).
        """
        mock_run.return_value = mock_pipeline_output_positive

        response = client.post(
            "/analyze-glaucoma",
            files={"file": ("test.jpg", sample_image, "image/jpeg")},
        )

        assert response.status_code == 200
        data = response.json()
        self.assert_glaucoma_response(data)
        assert data["has_glaucoma"] is True

    @patch(PIPELINE_PATCH)
    def test_success_negative_diagnosis(self, mock_run, sample_image, mock_pipeline_output_negative):
        """
        TEST NAME: test_success_negative_diagnosis
        COMPONENT: POST /analyze-glaucoma
        1. Mock GlaucomaPipeline.run() to return CDR=0.40 (below 0.65 threshold).
        2. POST a valid JPEG image to /analyze-glaucoma.
        3. Assert HTTP 200 status code.
        4. Assert response contains all required keys and valid formats.
        5. Assert has_glaucoma is False (CDR < 0.65).
        """
        mock_run.return_value = mock_pipeline_output_negative

        response = client.post(
            "/analyze-glaucoma",
            files={"file": ("test.jpg", sample_image, "image/jpeg")},
        )

        assert response.status_code == 200
        data = response.json()
        self.assert_glaucoma_response(data)
        assert data["has_glaucoma"] is False

    @patch(PIPELINE_PATCH)
    def test_pipeline_returns_none_yields_zero_values(self, mock_run, sample_image):
        """
        TEST NAME: test_pipeline_returns_none_yields_zero_values
        COMPONENT: POST /analyze-glaucoma, fallback branch in main.py
        1. Mock GlaucomaPipeline.run() to return None (YOLO found nothing, image unreadable).
        2. POST a valid JPEG image to /analyze-glaucoma.
        3. Assert HTTP 200 status code.
        4. Assert has_glaucoma is False, confidence is 0.0, cup_to_disc_ratio is 0.0.
        """
        mock_run.return_value = None

        response = client.post(
            "/analyze-glaucoma",
            files={"file": ("test.jpg", sample_image, "image/jpeg")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["has_glaucoma"] is False
        assert data["confidence"] == 0.0
        assert data["cup_to_disc_ratio"] == 0.0

    @patch(PIPELINE_PATCH)
    def test_cdr_boundary_at_threshold(self, mock_run, sample_image, mock_pipeline_output_positive):
        """
        TEST NAME: test_cdr_boundary_at_threshold
        COMPONENT: POST /analyze-glaucoma, CDR threshold logic in main.py
        1. Mock pipeline to return CDR=0.65 (exactly at threshold).
        2. POST a valid JPEG image.
        3. Assert has_glaucoma is False (threshold is strict: > 0.65, not >=).
        """
        # CDR exactly at 0.65 - threshold in main.py is strictly > 0.65
        boundary_output = list(mock_pipeline_output_positive)
        boundary_output[3] = 0.65
        mock_run.return_value = tuple(boundary_output)

        response = client.post(
            "/analyze-glaucoma",
            files={"file": ("test.jpg", sample_image, "image/jpeg")},
        )

        assert response.status_code == 200
        assert response.json()["has_glaucoma"] is False

    @patch(PIPELINE_PATCH)
    def test_cdr_boundary_values_pass_ratio_validation(self, mock_run, sample_image, mock_pipeline_output_positive):
        """
        TEST NAME: test_cdr_boundary_values_pass_ratio_validation
        COMPONENT: POST /analyze-glaucoma, assert_is_ratio helper
        1. Mock pipeline to return CDR=0.0, then CDR=1.0.
        2. POST a valid JPEG image for each value.
        3. Assert HTTP 200 for both.
        4. Assert cup_to_disc_ratio passes [0.0, 1.0] range validation.
        """
        for cdr in [0.0, 1.0]:
            output = list(mock_pipeline_output_positive)
            output[3] = cdr
            mock_run.return_value = tuple(output)

            response = client.post(
                "/analyze-glaucoma",
                files={"file": ("test.jpg", sample_image, "image/jpeg")},
            )

            assert response.status_code == 200
            self.assert_is_ratio(response.json()["cup_to_disc_ratio"])