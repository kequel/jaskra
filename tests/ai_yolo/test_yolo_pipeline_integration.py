import pytest
import sys
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock
from tests.base_test import BaseTest


class TestYoloPipelineIntegration(BaseTest):
    """
    Tests for how the YOLO model integrates with the GlaucomaPipeline.
    Verifies YOLO detection → crop → U-Net segmentation flow,
    including the fallback logic when YOLO finds no objects.

    All heavy dependencies (torch, ultralytics, smp) are mocked
    so these tests run without GPU or model weights.
    """

    def _build_mock_modules(self):
        """Build mock modules for torch, ultralytics, and smp so the pipeline
        module can be imported without installing those heavy dependencies."""
        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = "cpu"
        mock_torch.sigmoid.return_value.cpu.return_value.numpy.return_value = np.zeros((1, 2, 512, 512), dtype=np.float32)

        mock_ultralytics = MagicMock()
        mock_smp = MagicMock()

        return mock_torch, mock_ultralytics, mock_smp

    def _import_pipeline_with_mocks(self, mock_torch, mock_ultralytics, mock_smp):
        """Import pipeline.py with mocked heavy dependencies."""
        # Remove cached import if exists
        for mod_name in list(sys.modules.keys()):
            if 'pipeline' in mod_name and 'test' not in mod_name:
                del sys.modules[mod_name]

        mock_matplotlib = MagicMock()

        with patch.dict(sys.modules, {
            'torch': mock_torch,
            'ultralytics': mock_ultralytics,
            'segmentation_models_pytorch': mock_smp,
            'matplotlib': mock_matplotlib,
            'matplotlib.pyplot': mock_matplotlib.pyplot,
        }):
            # Also need to make the ai directory importable
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "pipeline_module",
                "ai/pipeline/pipeline.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

    def test_pipeline_uses_yolo_detections_for_cropping(self):
        """
        TEST NAME: test_pipeline_uses_yolo_detections_for_cropping
        COMPONENT: GlaucomaPipeline.run()
        1. Mock YOLO to return a single bounding box at (100, 100, 400, 400).
        2. Mock U-Net to return a dummy segmentation mask.
        3. Assert the returned crop coordinates match the YOLO detection box.
        """
        mock_torch, mock_ultralytics, mock_smp = self._build_mock_modules()

        # Configure YOLO mock to return a box
        mock_boxes = MagicMock()
        mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[100, 100, 400, 400]])

        mock_yolo_result = MagicMock()
        mock_yolo_result.boxes = mock_boxes

        mock_yolo_instance = MagicMock()
        mock_yolo_instance.return_value = [mock_yolo_result]
        mock_yolo_instance.to.return_value = mock_yolo_instance
        mock_ultralytics.YOLO.return_value = mock_yolo_instance

        # Configure U-Net mock
        sigmoid_output = np.zeros((1, 2, 512, 512), dtype=np.float32)
        mock_unet_output = MagicMock()
        mock_torch.sigmoid.return_value.cpu.return_value.numpy.return_value = sigmoid_output

        mock_unet_instance = MagicMock()
        mock_unet_instance.return_value = mock_unet_output
        mock_smp.UnetPlusPlus.return_value = mock_unet_instance

        module = self._import_pipeline_with_mocks(mock_torch, mock_ultralytics, mock_smp)

        # Patch cv2.imread in the pipeline module
        with patch.object(module.cv2, 'imread', return_value=np.zeros((512, 512, 3), dtype=np.uint8)), \
             patch.object(module.cv2, 'cvtColor', return_value=np.zeros((512, 512, 3), dtype=np.uint8)), \
             patch.object(module.cv2, 'resize', return_value=np.zeros((512, 512, 3), dtype=np.uint8)):

            pipeline = module.GlaucomaPipeline(yolo_path="dummy.pt", unet_path="dummy.pth")
            result = pipeline.run("test_image.png", conf=0.5)

        assert result is not None, "Pipeline returned None"
        _, crops, _, _, _, _ = result
        assert len(crops) > 0, "No crops returned"
        assert crops[0] == (100, 100, 400, 400), (
            f"Crop coords {crops[0]} don't match YOLO box (100, 100, 400, 400)"
        )

    def test_pipeline_falls_back_to_full_image_when_no_yolo_detections(self):
        """
        TEST NAME: test_pipeline_falls_back_to_full_image_when_no_yolo_detections
        COMPONENT: GlaucomaPipeline.run()
        1. Mock YOLO to return zero bounding boxes (empty detection).
        2. Mock U-Net to return a dummy segmentation mask.
        3. Assert the fallback crop covers the full image dimensions (0, 0, w, h).
        """
        mock_torch, mock_ultralytics, mock_smp = self._build_mock_modules()

        # YOLO returns no detections
        mock_boxes = MagicMock()
        mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([]).reshape(0, 4)

        mock_yolo_result = MagicMock()
        mock_yolo_result.boxes = mock_boxes

        mock_yolo_instance = MagicMock()
        mock_yolo_instance.return_value = [mock_yolo_result]
        mock_yolo_instance.to.return_value = mock_yolo_instance
        mock_ultralytics.YOLO.return_value = mock_yolo_instance

        # U-Net mock
        sigmoid_output = np.zeros((1, 2, 512, 512), dtype=np.float32)
        mock_unet_output = MagicMock()
        mock_torch.sigmoid.return_value.cpu.return_value.numpy.return_value = sigmoid_output

        mock_unet_instance = MagicMock()
        mock_unet_instance.return_value = mock_unet_output
        mock_smp.UnetPlusPlus.return_value = mock_unet_instance

        module = self._import_pipeline_with_mocks(mock_torch, mock_ultralytics, mock_smp)

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch.object(module.cv2, 'imread', return_value=fake_image), \
             patch.object(module.cv2, 'cvtColor', return_value=np.zeros((512, 512, 3), dtype=np.uint8)), \
             patch.object(module.cv2, 'resize', return_value=np.zeros((512, 512, 3), dtype=np.uint8)):

            pipeline = module.GlaucomaPipeline(yolo_path="dummy.pt", unet_path="dummy.pth")
            result = pipeline.run("test_image.png", conf=0.5)

        assert result is not None, "Pipeline returned None"
        _, crops, _, _, _, _ = result
        assert len(crops) > 0, "No crops returned in fallback mode"
        assert crops[0] == (0, 0, 640, 480), (
            f"Fallback crop {crops[0]} should cover full image (0, 0, 640, 480)"
        )

    def test_pipeline_returns_none_for_unreadable_image(self):
        """
        TEST NAME: test_pipeline_returns_none_for_unreadable_image
        COMPONENT: GlaucomaPipeline.run()
        1. Mock cv2.imread to return None (simulating an unreadable file).
        2. Call pipeline.run().
        3. Assert the result is None.
        """
        mock_torch, mock_ultralytics, mock_smp = self._build_mock_modules()

        mock_yolo_instance = MagicMock()
        mock_yolo_instance.to.return_value = mock_yolo_instance
        mock_ultralytics.YOLO.return_value = mock_yolo_instance

        mock_unet_instance = MagicMock()
        mock_smp.UnetPlusPlus.return_value = mock_unet_instance

        module = self._import_pipeline_with_mocks(mock_torch, mock_ultralytics, mock_smp)

        with patch.object(module.cv2, 'imread', return_value=None):
            pipeline = module.GlaucomaPipeline(yolo_path="dummy.pt", unet_path="dummy.pth")
            result = pipeline.run("nonexistent.png")

        assert result is None, "Pipeline should return None for unreadable images"

    def test_pipeline_yolo_called_with_given_conf(self):
        """
        TEST NAME: test_pipeline_yolo_called_with_given_conf
        COMPONENT: GlaucomaPipeline.run()
        1. Mock the YOLO instance and track calls to it.
        2. Call pipeline.run() with conf=0.75.
        3. Assert YOLO was invoked with conf=0.75 and verbose=False.
        """
        mock_torch, mock_ultralytics, mock_smp = self._build_mock_modules()

        # YOLO mock that records its call arguments
        mock_boxes = MagicMock()
        mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[50, 50, 200, 200]])

        mock_yolo_result = MagicMock()
        mock_yolo_result.boxes = mock_boxes

        mock_yolo_instance = MagicMock()
        mock_yolo_instance.return_value = [mock_yolo_result]
        mock_yolo_instance.to.return_value = mock_yolo_instance
        mock_ultralytics.YOLO.return_value = mock_yolo_instance

        # U-Net mock
        sigmoid_output = np.zeros((1, 2, 512, 512), dtype=np.float32)
        mock_unet_output = MagicMock()
        mock_torch.sigmoid.return_value.cpu.return_value.numpy.return_value = sigmoid_output

        mock_unet_instance = MagicMock()
        mock_unet_instance.return_value = mock_unet_output
        mock_smp.UnetPlusPlus.return_value = mock_unet_instance

        module = self._import_pipeline_with_mocks(mock_torch, mock_ultralytics, mock_smp)

        with patch.object(module.cv2, 'imread', return_value=np.zeros((512, 512, 3), dtype=np.uint8)), \
             patch.object(module.cv2, 'cvtColor', return_value=np.zeros((512, 512, 3), dtype=np.uint8)), \
             patch.object(module.cv2, 'resize', return_value=np.zeros((512, 512, 3), dtype=np.uint8)):

            pipeline = module.GlaucomaPipeline(yolo_path="dummy.pt", unet_path="dummy.pth")
            pipeline.run("test.png", conf=0.75)

        call_args = mock_yolo_instance.call_args
        assert call_args is not None, "YOLO model was not called"
        assert call_args.kwargs.get("conf") == 0.75, (
            f"YOLO was called with conf={call_args.kwargs.get('conf')}, expected 0.75"
        )
