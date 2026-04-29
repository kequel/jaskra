import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch
from main import app
from tests.base_test import BaseTest

client = TestClient(app)

class TestGlaucomaStreaming(BaseTest):
    """
    Streaming /analyze-glaucoma-stream Endpoint
    Steps:
    1. Mock the AI pipeline to prevent real model execution.
    2. Post an image to the streaming endpoint.
    3. Iterate through the response stream line by line.
    4. Verify that the stream provides progress steps (e.g., step 1, 2, 3).
    5. Ensure the final chunk is a 'success' status with full result data.
    """

    @patch('pipeline.pipeline.GlaucomaPipeline.run')
    def test_analyze_glaucoma_stream_flow(self, mock_run, sample_image):
        mock_run.return_value = (None, [], [], 0.70, [0], [0.99])
        
        files = {"file": ("test.jpg", sample_image, "image/jpeg")}
        response = client.post("/analyze-glaucoma-stream", files=files)
        
        assert response.status_code == 200
        
        # Read and parse the NDJSON stream
        events = [json.loads(line) for line in response.iter_lines() if line]
        
        # Validation: Check if progress steps were emitted
        steps = [e.get("step") for e in events if "step" in e]
        assert len(steps) > 0
        
        # Final event check
        final_event = events[-1]
        assert final_event["status"] == "success"
        assert "image_base64" in final_event["data"]
        self.assert_valid_base64(final_event["data"]["image_base64"])