import numpy as np
import pytest
from tests.base_test import BaseTest
from tests.ai_pipeline.conftest import make_yolo_result, make_unet_output


class TestRunFallback(BaseTest):
    """
    Tests for GlaucomaPipeline.run() when YOLO returns no detections.
    Verifies that the pipeline falls back gracefully to the full image ROI
    and handles a missing image file without raising an exception.
    """

    def test_fallback_when_yolo_returns_empty_bboxes(
        self, pipeline_instance, saved_test_image
    ):
        """
        TEST NAME: test_fallback_when_yolo_returns_empty_bboxes
        COMPONENT: GlaucomaPipeline.run()
        1. Configure mock YOLO to return zero bounding boxes.
        2. Call run() with a valid image path.
        3. Assert the return value is not None (pipeline did not abort).
        """
        pipeline_instance._mock_yolo.return_value = [
            make_yolo_result(np.empty((0, 4), dtype=np.float32))
        ]
        pipeline_instance._mock_unet.return_value = make_unet_output()

        result = pipeline_instance.run(str(saved_test_image))

        assert result is not None

    def test_fallback_bbox_covers_full_image(
        self, pipeline_instance, saved_test_image, dummy_bgr_image
    ):
        """
        TEST NAME: test_fallback_bbox_covers_full_image
        COMPONENT: GlaucomaPipeline.run()
        1. Configure mock YOLO to return zero bounding boxes.
        2. Call run() and unpack the crops list from the result.
        3. Assert there is exactly one crop and it spans the full image dimensions.
        """
        h, w = dummy_bgr_image.shape[:2]
        pipeline_instance._mock_yolo.return_value = [
            make_yolo_result(np.empty((0, 4), dtype=np.float32))
        ]
        pipeline_instance._mock_unet.return_value = make_unet_output()

        _, crops, _, _, _, _ = pipeline_instance.run(str(saved_test_image))

        assert len(crops) == 1
        x1, y1, x2, y2 = crops[0]
        assert (x1, y1) == (0, 0)
        assert (x2, y2) == (w, h)

    def test_run_returns_none_for_missing_image(self, pipeline_instance):
        """
        TEST NAME: test_run_returns_none_for_missing_image
        COMPONENT: GlaucomaPipeline.run()
        1. Call run() with a path that does not exist on disk.
        2. Assert the return value is None.
        """
        result = pipeline_instance.run("nonexistent/path/image.png")

        assert result is None


class TestRunWithDetection(BaseTest):
    """
    Tests for GlaucomaPipeline.run() when YOLO returns valid detections.
    Verifies output structure, CDR value range, and mask dimensions.
    """

    def test_run_with_valid_yolo_detection_returns_six_tuple(
        self, pipeline_instance, saved_test_image, dummy_bgr_image
    ):
        """
        TEST NAME: test_run_with_valid_yolo_detection_returns_six_tuple
        COMPONENT: GlaucomaPipeline.run()
        1. Configure mock YOLO to return one bounding box slightly inset from image edges.
        2. Call run() with a valid image path.
        3. Assert the result is not None and is a tuple of exactly 6 elements.
        """
        h, w = dummy_bgr_image.shape[:2]
        bbox = np.array([[5, 5, w - 5, h - 5]], dtype=np.float32)
        pipeline_instance._mock_yolo.return_value = [make_yolo_result(bbox)]
        pipeline_instance._mock_unet.return_value = make_unet_output()

        output = pipeline_instance.run(str(saved_test_image))

        assert output is not None
        assert len(output) == 6

    def test_run_cdr_is_float_in_valid_range(
        self, pipeline_instance, saved_test_image, dummy_bgr_image
    ):
        """
        TEST NAME: test_run_cdr_is_float_in_valid_range
        COMPONENT: GlaucomaPipeline.run()
        1. Configure YOLO to detect the full image and UNet to return high-confidence output.
        2. Call run() and extract the CDR value from the result.
        3. Assert CDR is a float within [0.0, 1.0].
        """
        h, w = dummy_bgr_image.shape[:2]
        pipeline_instance._mock_yolo.return_value = [
            make_yolo_result(np.array([[0, 0, w, h]], dtype=np.float32))
        ]
        pipeline_instance._mock_unet.return_value = make_unet_output(disc_val=2.0, cup_val=1.0)

        _, _, _, cdr, _, _ = pipeline_instance.run(str(saved_test_image))

        assert isinstance(cdr, float)
        assert 0.0 <= cdr <= 1.0

    def test_run_returns_correct_masks_shape(
        self, pipeline_instance, saved_test_image, dummy_bgr_image
    ):
        """
        TEST NAME: test_run_returns_correct_masks_shape
        COMPONENT: GlaucomaPipeline.run()
        1. Configure YOLO to detect the full image and UNet to return default output.
        2. Call run() and extract the masks list from the result.
        3. Assert at least one mask exists and its shape is (2, 512, 512).
        """
        h, w = dummy_bgr_image.shape[:2]
        pipeline_instance._mock_yolo.return_value = [
            make_yolo_result(np.array([[0, 0, w, h]], dtype=np.float32))
        ]
        pipeline_instance._mock_unet.return_value = make_unet_output()

        _, _, masks, _, _, _ = pipeline_instance.run(str(saved_test_image))

        assert len(masks) >= 1
        assert masks[0].shape == (2, 512, 512)


class TestRunIntegration(BaseTest):
    """
    End-to-end integration tests for GlaucomaPipeline.run().
    Verifies correct output types, multi-bbox handling, exception safety,
    and ground-truth behaviour when masks_dir is None.
    """

    def test_full_pipeline_returns_correct_types(
        self, pipeline_instance, saved_test_image, dummy_bgr_image
    ):
        """
        TEST NAME: test_full_pipeline_returns_correct_types
        COMPONENT: GlaucomaPipeline.run()
        1. Configure YOLO and UNet mocks for a full-image detection.
        2. Call run() and unpack all six return values.
        3. Assert each value has the expected Python/NumPy type.
        """
        h, w = dummy_bgr_image.shape[:2]
        pipeline_instance._mock_yolo.return_value = [
            make_yolo_result(np.array([[0, 0, w, h]], dtype=np.float32))
        ]
        pipeline_instance._mock_unet.return_value = make_unet_output()

        result = pipeline_instance.run(str(saved_test_image))
        assert result is not None

        full_img, crops, masks, cdr, gt_masks, cdr_gt = result
        assert isinstance(full_img, np.ndarray)
        assert isinstance(crops, list)
        assert isinstance(masks, list)
        assert isinstance(cdr, float)
        assert gt_masks is None or isinstance(gt_masks, np.ndarray)
        assert isinstance(cdr_gt, float)

    def test_full_pipeline_with_multiple_bboxes_returns_multiple_masks(
        self, pipeline_instance, saved_test_image, dummy_bgr_image
    ):
        """
        TEST NAME: test_full_pipeline_with_multiple_bboxes_returns_multiple_masks
        COMPONENT: GlaucomaPipeline.run()
        1. Configure YOLO to return two non-overlapping bounding boxes.
        2. Call run() and extract the masks list.
        3. Assert exactly two masks are returned.
        """
        h, w = dummy_bgr_image.shape[:2]
        bboxes = np.array([
            [0, 0, w // 2, h // 2],
            [w // 2, 0, w, h],
        ], dtype=np.float32)
        pipeline_instance._mock_yolo.return_value = [make_yolo_result(bboxes)]
        pipeline_instance._mock_unet.return_value = make_unet_output()

        _, _, masks, _, _, _ = pipeline_instance.run(str(saved_test_image))

        assert len(masks) == 2

    def test_no_exception_raised_during_full_flow(
        self, pipeline_instance, saved_test_image, dummy_bgr_image
    ):
        """
        TEST NAME: test_no_exception_raised_during_full_flow
        COMPONENT: GlaucomaPipeline.run()
        1. Configure YOLO and UNet mocks for a standard full-image run.
        2. Call run() inside a try/except block.
        3. Assert no exception is raised; fail the test explicitly if one occurs.
        """
        h, w = dummy_bgr_image.shape[:2]
        pipeline_instance._mock_yolo.return_value = [
            make_yolo_result(np.array([[0, 0, w, h]], dtype=np.float32))
        ]
        pipeline_instance._mock_unet.return_value = make_unet_output()

        try:
            pipeline_instance.run(str(saved_test_image))
        except Exception as exc:
            pytest.fail(f"run() raised an unexpected exception: {exc}")

    def test_gt_masks_none_when_no_masks_dir(
        self, pipeline_instance, saved_test_image, dummy_bgr_image
    ):
        """
        TEST NAME: test_gt_masks_none_when_no_masks_dir
        COMPONENT: GlaucomaPipeline.run()
        1. Confirm pipeline_instance has masks_dir=None.
        2. Call run() with a valid image.
        3. Assert gt_masks is None and cdr_gt equals 0.0.
        """
        assert pipeline_instance.masks_dir is None
        h, w = dummy_bgr_image.shape[:2]
        pipeline_instance._mock_yolo.return_value = [
            make_yolo_result(np.array([[0, 0, w, h]], dtype=np.float32))
        ]
        pipeline_instance._mock_unet.return_value = make_unet_output()

        _, _, _, _, gt_masks, cdr_gt = pipeline_instance.run(str(saved_test_image))

        assert gt_masks is None
        assert cdr_gt == 0.0
