import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from main import app
from tests.base_test import BaseTest

client = TestClient(app)


class TestGlaucomaStreaming(BaseTest):
    """
    Tests for the /analyze-glaucoma-stream endpoint (NDJSON streaming).

    Verifies stream structure, not diagnostic logic - covered in test_api.py.
    """

    @patch("pipeline.pipeline.GlaucomaPipeline.run")
    def test_stream_returns_200(self, mock_run, sample_image, mock_pipeline_output_positive):
        """Streaming endpoint returns 200."""
        mock_run.return_value = mock_pipeline_output_positive
        response = client.post(
            "/analyze-glaucoma-stream",
            files={"file": ("test.jpg", sample_image, "image/jpeg")},
        )
        assert response.status_code == 200

    @patch("pipeline.pipeline.GlaucomaPipeline.run")
    def test_stream_emits_progress_steps(self, mock_run, sample_image, mock_pipeline_output_positive):
        """Stream must contain at least one event with a 'step' field."""
        mock_run.return_value = mock_pipeline_output_positive
        response = client.post(
            "/analyze-glaucoma-stream",
            files={"file": ("test.jpg", sample_image, "image/jpeg")},
        )

        events = [json.loads(line) for line in response.iter_lines() if line]
        steps = [e.get("step") for e in events if "step" in e]
        assert len(steps) > 0, "Stream contains no progress events (no 'step' field)"

    @patch("pipeline.pipeline.GlaucomaPipeline.run")
    def test_stream_final_event_has_success_status(self, mock_run, sample_image, mock_pipeline_output_positive):
        """Final stream event must have status='success' and complete data."""
        mock_run.return_value = mock_pipeline_output_positive
        response = client.post(
            "/analyze-glaucoma-stream",
            files={"file": ("test.jpg", sample_image, "image/jpeg")},
        )

        events = [json.loads(line) for line in response.iter_lines() if line]
        assert events, "Strumień jest pusty"

        final = events[-1]
        assert final.get("status") == "success", f"Ostatni event: {final}"
        assert "data" in final, "Brak klucza 'data' w ostatnim evencie"
        self.assert_glaucoma_response(final["data"])

    @patch("pipeline.pipeline.GlaucomaPipeline.run")
    def test_stream_steps_are_ordered(self, mock_run, sample_image, mock_pipeline_output_positive):
        """Step numbering in stream should be in ascending order."""
        mock_run.return_value = mock_pipeline_output_positive
        response = client.post(
            "/analyze-glaucoma-stream",
            files={"file": ("test.jpg", sample_image, "image/jpeg")},
        )

        events = [json.loads(line) for line in response.iter_lines() if line]
        steps = [e["step"] for e in events if "step" in e]

        if len(steps) > 1:
            assert steps == sorted(steps), f"Steps are not sorted: {steps}"