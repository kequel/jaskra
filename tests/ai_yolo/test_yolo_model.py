import pytest
import numpy as np
from tests.base_test import BaseTest


class TestYoloInference(BaseTest):
    """
    Tests for YOLO model inference stability and output format.
    Verifies that the model produces correctly structured results
    on both synthetic and real inputs without crashing.
    """

    def test_inference_on_dummy_image_returns_boxes_attribute(self, yolo_glaucoma_model):
        """
        TEST NAME: test_inference_on_dummy_image_returns_boxes_attribute
        COMPONENT: ultralytics.YOLO.__call__()
        1. Create a synthetic black 512x512 image (BGR format).
        2. Execute model inference on the dummy image.
        3. Assert that the result object contains the 'boxes' attribute for detection.
        """
        dummy_img = np.zeros((512, 512, 3), dtype=np.uint8)
        results = yolo_glaucoma_model(dummy_img)

        assert len(results) > 0, "Model failed to return any result objects"
        assert hasattr(results[0], 'boxes'), "Output format error: 'boxes' attribute missing"

    def test_inference_returns_xyxy_format_tensors(self, yolo_glaucoma_model, synthetic_fundus_image):
        """
        TEST NAME: test_inference_returns_xyxy_format_tensors
        COMPONENT: ultralytics.YOLO.__call__()
        1. Run inference on a synthetic fundus image.
        2. Access the bounding boxes in xyxy format.
        3. Assert that the xyxy tensor has exactly 4 columns (x1, y1, x2, y2).
        """
        results = yolo_glaucoma_model(synthetic_fundus_image)
        boxes = results[0].boxes

        xyxy = boxes.xyxy.cpu().numpy()
        if len(xyxy) > 0:
            assert xyxy.shape[1] == 4, (
                f"Expected 4 columns in xyxy tensor, got {xyxy.shape[1]}"
            )

    def test_inference_confidence_scores_within_valid_range(self, yolo_glaucoma_model, synthetic_fundus_image):
        """
        TEST NAME: test_inference_confidence_scores_within_valid_range
        COMPONENT: ultralytics.YOLO.__call__()
        1. Run inference on a synthetic fundus image.
        2. Extract confidence scores from the result.
        3. Assert all confidence values are between 0.0 and 1.0.
        """
        results = yolo_glaucoma_model(synthetic_fundus_image)
        confs = results[0].boxes.conf.cpu().numpy()

        for i, c in enumerate(confs):
            assert 0.0 <= c <= 1.0, (
                f"Detection {i} has out-of-range confidence {c}"
            )

    def test_high_conf_threshold_reduces_detection_count(self, yolo_glaucoma_model, synthetic_fundus_image):
        """
        TEST NAME: test_high_conf_threshold_reduces_detection_count
        COMPONENT: ultralytics.YOLO.predict()
        1. Run inference with a low confidence threshold (0.01).
        2. Run inference again with a high confidence threshold (0.99).
        3. Assert that the high-threshold run returns fewer or equal detections.
        """
        results_low = yolo_glaucoma_model.predict(synthetic_fundus_image, conf=0.01, verbose=False)
        results_high = yolo_glaucoma_model.predict(synthetic_fundus_image, conf=0.99, verbose=False)

        count_low = len(results_low[0].boxes)
        count_high = len(results_high[0].boxes)

        assert count_high <= count_low, (
            f"High threshold ({count_high}) should yield ≤ detections than low ({count_low})"
        )

    def test_bounding_box_coordinates_within_image_bounds(self, yolo_glaucoma_model, synthetic_fundus_image):
        """
        TEST NAME: test_bounding_box_coordinates_within_image_bounds
        COMPONENT: ultralytics.YOLO.__call__()
        1. Run inference on a 512x512 synthetic fundus image.
        2. Extract all bounding box coordinates (xyxy).
        3. Assert that all coordinates fall within [0, 512].
        """
        h, w = synthetic_fundus_image.shape[:2]
        results = yolo_glaucoma_model(synthetic_fundus_image)
        xyxy = results[0].boxes.xyxy.cpu().numpy()

        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = box
            assert 0 <= x1 <= w and 0 <= x2 <= w, (
                f"Detection {i}: x coords ({x1:.1f}, {x2:.1f}) outside [0, {w}]"
            )
            assert 0 <= y1 <= h and 0 <= y2 <= h, (
                f"Detection {i}: y coords ({y1:.1f}, {y2:.1f}) outside [0, {h}]"
            )
            assert x1 < x2 and y1 < y2, (
                f"Detection {i}: invalid box (x1={x1:.1f} >= x2={x2:.1f} or y1={y1:.1f} >= y2={y2:.1f})"
            )

    def test_inference_on_non_square_image_runs_without_error(self, yolo_glaucoma_model):
        """
        TEST NAME: test_inference_on_non_square_image_runs_without_error
        COMPONENT: ultralytics.YOLO.__call__()
        1. Create a rectangular 640x480 dummy image.
        2. Run model inference on it.
        3. Assert that the result list is non-empty and has 'boxes' attribute.
        """
        rect_img = np.zeros((480, 640, 3), dtype=np.uint8)
        results = yolo_glaucoma_model(rect_img)

        assert len(results) > 0, "Model returned empty results on rectangular image"
        assert hasattr(results[0], 'boxes'), "'boxes' attribute missing for rectangular input"


class TestYoloClassMapping(BaseTest):
    """
    Tests for YOLO model class definition and naming conventions.
    The ROI detection model is expected to have a single class (ID 0).
    """

    def test_model_output_classes_match_specification(self, yolo_glaucoma_model):
        """
        TEST NAME: test_model_output_classes_match_specification
        COMPONENT: YOLO.model.names
        1. Access the model's class dictionary.
        2. Verify that class ID 0 exists.
        3. Assert that the class name at ID 0 is '0' (as per training spec).
        """
        class_map = yolo_glaucoma_model.names
        assert 0 in class_map, "Class ID 0 is missing from the model definition"
        assert class_map[0] == '0', f"Unexpected class name: expected '0', got '{class_map[0]}'"

    def test_model_has_single_class(self, yolo_glaucoma_model):
        """
        TEST NAME: test_model_has_single_class
        COMPONENT: YOLO.model.names
        1. Access the model's class dictionary.
        2. Assert that there is exactly one class defined (ROI detection is single-class).
        """
        class_map = yolo_glaucoma_model.names
        assert len(class_map) == 1, (
            f"Expected 1 class for ROI detection, found {len(class_map)}: {class_map}"
        )

    def test_detected_class_ids_are_valid(self, yolo_glaucoma_model, synthetic_fundus_image):
        """
        TEST NAME: test_detected_class_ids_are_valid
        COMPONENT: ultralytics.YOLO.__call__()
        1. Run inference on a synthetic fundus image.
        2. Extract predicted class IDs from all detections.
        3. Assert all class IDs exist in the model's class dictionary.
        """
        results = yolo_glaucoma_model(synthetic_fundus_image)
        cls_ids = results[0].boxes.cls.cpu().numpy()
        valid_ids = set(yolo_glaucoma_model.names.keys())

        for i, cid in enumerate(cls_ids):
            assert int(cid) in valid_ids, (
                f"Detection {i}: class ID {int(cid)} not in model classes {valid_ids}"
            )


class TestYoloClinicalSample(BaseTest):
    """
    Tests for YOLO detection on a known clinical fundus image (1.png).
    Verifies the model's functional ability to detect optic nerve structures
    on a verified positive sample.
    """

    def test_detection_on_known_positive_sample_finds_objects(self, yolo_glaucoma_model):
        """
        TEST NAME: test_detection_on_known_positive_sample_finds_objects
        COMPONENT: ultralytics.YOLO.predict()
        1. Load the verified clinical image '1.png' known to contain the optic nerve head.
        2. Perform prediction with a confidence threshold of 0.25.
        3. Assert that at least one bounding box is detected.
        """
        results = yolo_glaucoma_model.predict(
            '../../tests/ai_yolo/1.png', save=False, conf=0.25, verbose=False
        )
        detected_count = len(results[0].boxes)
        assert detected_count > 0, (
            "Functional Failure: Model failed to detect objects on verified sample 1.png"
        )

    def test_clinical_sample_confidence_above_minimum(self, yolo_glaucoma_model):
        """
        TEST NAME: test_clinical_sample_confidence_above_minimum
        COMPONENT: ultralytics.YOLO.predict()
        1. Run prediction on the clinical sample '1.png' with conf=0.25.
        2. Extract the highest confidence score from the detections.
        3. Assert the best confidence is at least 0.5 (model should be confident on real data).
        """
        results = yolo_glaucoma_model.predict(
            '../../tests/ai_yolo/1.png', save=False, conf=0.25, verbose=False
        )
        confs = results[0].boxes.conf.cpu().numpy()
        assert len(confs) > 0, "No detections found on clinical sample"

        best_conf = float(confs.max())
        assert best_conf >= 0.5, (
            f"Best detection confidence ({best_conf:.3f}) is below 0.5 on a known positive"
        )

    def test_clinical_sample_detection_box_has_positive_area(self, yolo_glaucoma_model):
        """
        TEST NAME: test_clinical_sample_detection_box_has_positive_area
        COMPONENT: ultralytics.YOLO.predict()
        1. Run prediction on the clinical sample '1.png'.
        2. Take the first (highest-confidence) detection's bounding box.
        3. Assert the box has positive width and height (non-degenerate).
        """
        results = yolo_glaucoma_model.predict(
            '../../tests/ai_yolo/1.png', save=False, conf=0.25, verbose=False
        )
        xyxy = results[0].boxes.xyxy.cpu().numpy()
        assert len(xyxy) > 0, "No detections to validate"

        x1, y1, x2, y2 = xyxy[0]
        width = x2 - x1
        height = y2 - y1
        assert width > 0 and height > 0, (
            f"Detection box has non-positive dimensions: w={width:.1f}, h={height:.1f}"
        )