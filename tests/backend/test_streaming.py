import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from main import app
from tests.base_test import BaseTest

client = TestClient(app)

PIPELINE_PATCH = "pipeline.GlaucomaPipeline.run"


class TestGlaucomaStreaming(BaseTest):
    """
    Unit tests for the POST /analyze-glaucoma-stream endpoint (NDJSON streaming).
    Pipeline is mocked — verifies stream structure, step ordering, and final event format.
    Diagnostic logic is covered in test_api.py.
    """

    @patch(PIPELINE_PATCH)
    def test_stream_returns_200(self, mock_run, sample_image, mock_pipeline_output_positive):
        """
        TEST NAME: test_stream_returns_200
        COMPONENT: POST /analyze-glaucoma-stream, StreamingResponse in main.py
        1. Mock GlaucomaPipeline.run() to prevent real model execution.
        2. POST a valid JPEG image to /analyze-glaucoma-stream.
        3. Assert HTTP 200 status code.
        """
        mock_run.return_value = mock_pipeline_output_positive
        response = client.post(
            "/analyze-glaucoma-stream",
            files={"file": ("test.jpg", sample_image, "image/jpeg")},
        )
        assert response.status_code == 200

    @patch(PIPELINE_PATCH)
    def test_stream_emits_progress_steps(self, mock_run, sample_image, mock_pipeline_output_positive):
        """
        TEST NAME: test_stream_emits_progress_steps
        COMPONENT: POST /analyze-glaucoma-stream, event_generator() in main.py
        1. Mock GlaucomaPipeline.run() with positive output.
        2. POST an image and collect the NDJSON stream line by line.
        3. Parse each line as JSON.
        4. Assert at least one event contains a "step" field (progress was emitted).
        """
        mock_run.return_value = mock_pipeline_output_positive
        response = client.post(
            "/analyze-glaucoma-stream",
            files={"file": ("test.jpg", sample_image, "image/jpeg")},
        )

        events = [json.loads(line) for line in response.iter_lines() if line]
        steps = [e.get("step") for e in events if "step" in e]
        assert len(steps) > 0, "Stream contains no progress events (no 'step' field)"

    @patch(PIPELINE_PATCH)
    def test_stream_final_event_has_success_status(self, mock_run, sample_image, mock_pipeline_output_positive):
        """
        TEST NAME: test_stream_final_event_has_success_status
        COMPONENT: POST /analyze-glaucoma-stream, final yield in event_generator()
        1. Mock GlaucomaPipeline.run() with positive output.
        2. POST an image and collect the full NDJSON stream.
        3. Assert the last event has status="success".
        4. Assert last event contains a "data" key with the full diagnosis payload.
        5. Validate the diagnosis payload structure via assert_glaucoma_response().
        """
        mock_run.return_value = mock_pipeline_output_positive
        response = client.post(
            "/analyze-glaucoma-stream",
            files={"file": ("test.jpg", sample_image, "image/jpeg")},
        )

        events = [json.loads(line) for line in response.iter_lines() if line]
        assert events, "Stream is empty"

        final = events[-1]
        assert final.get("status") == "success", f"Last event: {final}"
        assert "data" in final, "Missing 'data' key in final event"
        self.assert_glaucoma_response(final["data"])

    @patch(PIPELINE_PATCH)
    def test_stream_steps_are_ordered(self, mock_run, sample_image, mock_pipeline_output_positive):
        """
        TEST NAME: test_stream_steps_are_ordered
        COMPONENT: POST /analyze-glaucoma-stream, step numbering in event_generator()
        1. Mock GlaucomaPipeline.run() with positive output.
        2. POST an image and collect the full NDJSON stream.
        3. Extract all "step" values from events that contain the field.
        4. Assert the list of steps is in ascending order.
        """
        mock_run.return_value = mock_pipeline_output_positive
        response = client.post(
            "/analyze-glaucoma-stream",
            files={"file": ("test.jpg", sample_image, "image/jpeg")},
        )

        events = [json.loads(line) for line in response.iter_lines() if line]
        steps = [e["step"] for e in events if "step" in e]

        if len(steps) > 1:
            assert steps == sorted(steps), f"Steps are not in ascending order: {steps}"

    @patch(PIPELINE_PATCH)
    def test_stream_contains_five_steps(self, mock_run, sample_image, mock_pipeline_output_positive):
        """
        TEST NAME: test_stream_contains_five_steps
        COMPONENT: POST /analyze-glaucoma-stream, event_generator() step count
        1. Mock GlaucomaPipeline.run() assuming models are already initialized
        (skips the step-2 "Loading models" branch).
        2. POST an image and collect events.
        3. Assert the last step value equals 5 (matches the hardcoded final yield).
        """
        mock_run.return_value = mock_pipeline_output_positive
        response = client.post(
            "/analyze-glaucoma-stream",
            files={"file": ("test.jpg", sample_image, "image/jpeg")},
        )

        events = [json.loads(line) for line in response.iter_lines() if line]
        steps = [e["step"] for e in events if "step" in e]
        assert steps[-1] == 5, f"Expected final step to be 5, got: {steps[-1]}"