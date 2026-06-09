import pytest
import os
import numpy as np
import cv2
from tests.base_test import BaseTest
from ai.yolo.auto_labeling_script.main import (
    preprocess,
    edge_map,
    yolo_from_bbox,
    save_yolo_label,
    draw_preview,
    find_best_square_roi,
    process_one_pair,
)


class TestPreprocess(BaseTest):
    """
    Tests for the preprocess() function that converts BGR images
    to grayscale with CLAHE histogram equalization.
    """

    def test_preprocess_returns_single_channel(self):
        """
        TEST NAME: test_preprocess_returns_single_channel
        COMPONENT: preprocess()
        1. Create a 100x100 BGR dummy image.
        2. Call preprocess().
        3. Assert the output is a 2D (single-channel) grayscale array.
        """
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = preprocess(img)
        assert result.ndim == 2, f"Expected 2D grayscale, got ndim={result.ndim}"

    def test_preprocess_preserves_dimensions(self):
        """
        TEST NAME: test_preprocess_preserves_dimensions
        COMPONENT: preprocess()
        1. Create a 200x300 BGR image.
        2. Call preprocess().
        3. Assert the output spatial dimensions match (200, 300).
        """
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        result = preprocess(img)
        assert result.shape == (200, 300), (
            f"Expected shape (200, 300), got {result.shape}"
        )

    def test_preprocess_output_dtype_is_uint8(self):
        """
        TEST NAME: test_preprocess_output_dtype_is_uint8
        COMPONENT: preprocess()
        1. Create a BGR image with varied pixel values.
        2. Call preprocess().
        3. Assert the result dtype is uint8 (suitable for template matching).
        """
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = preprocess(img)
        assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"


class TestEdgeMap(BaseTest):
    """
    Tests for the edge_map() function that applies Canny edge detection.
    """

    def test_edge_map_returns_binary_image(self):
        """
        TEST NAME: test_edge_map_returns_binary_image
        COMPONENT: edge_map()
        1. Create a grayscale image with a sharp rectangle.
        2. Call edge_map().
        3. Assert all pixel values are either 0 or 255 (Canny binary output).
        """
        gray = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(gray, (20, 20), (80, 80), 255, -1)
        edges = edge_map(gray)
        unique_vals = set(np.unique(edges))
        assert unique_vals.issubset({0, 255}), (
            f"Edge map contains non-binary values: {unique_vals}"
        )

    def test_edge_map_preserves_dimensions(self):
        """
        TEST NAME: test_edge_map_preserves_dimensions
        COMPONENT: edge_map()
        1. Create a 150x200 grayscale image.
        2. Call edge_map().
        3. Assert the output dimensions match (150, 200).
        """
        gray = np.zeros((150, 200), dtype=np.uint8)
        edges = edge_map(gray)
        assert edges.shape == (150, 200), (
            f"Expected shape (150, 200), got {edges.shape}"
        )

    def test_edge_map_on_uniform_image_returns_no_edges(self):
        """
        TEST NAME: test_edge_map_on_uniform_image_returns_no_edges
        COMPONENT: edge_map()
        1. Create a uniform grayscale image (all pixels = 128).
        2. Call edge_map().
        3. Assert the edge map is all zeros (no gradients to detect).
        """
        gray = np.full((100, 100), 128, dtype=np.uint8)
        edges = edge_map(gray)
        assert np.sum(edges) == 0, "Uniform image should produce no edges"


class TestYoloFromBbox(BaseTest):
    """
    Tests for yolo_from_bbox() which converts pixel-space bounding
    boxes to normalized YOLO format (x_center, y_center, w, h).
    """

    def test_centered_box_returns_center_coordinates(self):
        """
        TEST NAME: test_centered_box_returns_center_coordinates
        COMPONENT: yolo_from_bbox()
        1. Define a box exactly centered in a 100x100 image (25,25,50,50).
        2. Call yolo_from_bbox().
        3. Assert center coordinates are (0.5, 0.5).
        """
        xc, yc, w, h = yolo_from_bbox(25, 25, 50, 50, 100, 100)
        assert abs(xc - 0.5) < 1e-6, f"x_center={xc}, expected 0.5"
        assert abs(yc - 0.5) < 1e-6, f"y_center={yc}, expected 0.5"

    def test_full_image_box_returns_unit_dimensions(self):
        """
        TEST NAME: test_full_image_box_returns_unit_dimensions
        COMPONENT: yolo_from_bbox()
        1. Define a box covering the entire 200x200 image (0,0,200,200).
        2. Call yolo_from_bbox().
        3. Assert width and height are both 1.0 (full coverage).
        """
        xc, yc, w, h = yolo_from_bbox(0, 0, 200, 200, 200, 200)
        assert abs(w - 1.0) < 1e-6, f"width={w}, expected 1.0"
        assert abs(h - 1.0) < 1e-6, f"height={h}, expected 1.0"

    def test_all_values_within_zero_one(self):
        """
        TEST NAME: test_all_values_within_zero_one
        COMPONENT: yolo_from_bbox()
        1. Define a box at (10, 20, 30, 40) in a 512x512 image.
        2. Call yolo_from_bbox().
        3. Assert all four returned values are within [0.0, 1.0].
        """
        xc, yc, w, h = yolo_from_bbox(10, 20, 30, 40, 512, 512)
        for name, val in [("xc", xc), ("yc", yc), ("w", w), ("h", h)]:
            assert 0.0 <= val <= 1.0, f"{name}={val} is outside [0, 1]"

    def test_known_values_quarter_box(self):
        """
        TEST NAME: test_known_values_quarter_box
        COMPONENT: yolo_from_bbox()
        1. Define a box at top-left quarter (0, 0, 50, 50) of a 100x100 image.
        2. Call yolo_from_bbox().
        3. Assert the normalized values are (0.25, 0.25, 0.5, 0.5).
        """
        xc, yc, w, h = yolo_from_bbox(0, 0, 50, 50, 100, 100)
        assert abs(xc - 0.25) < 1e-6, f"x_center={xc}, expected 0.25"
        assert abs(yc - 0.25) < 1e-6, f"y_center={yc}, expected 0.25"
        assert abs(w - 0.5) < 1e-6, f"width={w}, expected 0.5"
        assert abs(h - 0.5) < 1e-6, f"height={h}, expected 0.5"


class TestSaveYoloLabel(BaseTest):
    """
    Tests for save_yolo_label() which writes YOLO-format label files.
    """

    def test_label_file_created_with_correct_format(self, tmp_path):
        """
        TEST NAME: test_label_file_created_with_correct_format
        COMPONENT: save_yolo_label()
        1. Call save_yolo_label() with known box parameters.
        2. Read the created label file.
        3. Assert the file contains one line with 5 space-separated values (class + 4 coords).
        """
        txt_path = str(tmp_path / "test_label.txt")
        save_yolo_label(txt_path, class_id=0, x=100, y=100, w=200, h=200, img_w=512, img_h=512)

        assert os.path.exists(txt_path), "Label file was not created"
        with open(txt_path, "r") as f:
            content = f.read().strip()
        parts = content.split()
        assert len(parts) == 5, f"Expected 5 values, got {len(parts)}: {content}"

    def test_label_class_id_matches_input(self, tmp_path):
        """
        TEST NAME: test_label_class_id_matches_input
        COMPONENT: save_yolo_label()
        1. Call save_yolo_label() with class_id=0.
        2. Read the output file and parse the first token.
        3. Assert the class ID in the file matches 0.
        """
        txt_path = str(tmp_path / "test_label.txt")
        save_yolo_label(txt_path, class_id=0, x=50, y=50, w=100, h=100, img_w=512, img_h=512)

        with open(txt_path, "r") as f:
            first_token = f.read().strip().split()[0]
        assert first_token == "0", f"Expected class ID '0', got '{first_token}'"

    def test_label_coordinates_are_normalized(self, tmp_path):
        """
        TEST NAME: test_label_coordinates_are_normalized
        COMPONENT: save_yolo_label()
        1. Call save_yolo_label() with a box inside a 512x512 image.
        2. Parse the four coordinate values from the file.
        3. Assert all coordinates are within [0.0, 1.0].
        """
        txt_path = str(tmp_path / "test_label.txt")
        save_yolo_label(txt_path, class_id=0, x=100, y=150, w=200, h=180, img_w=512, img_h=512)

        with open(txt_path, "r") as f:
            parts = f.read().strip().split()
        coords = [float(v) for v in parts[1:]]
        for i, c in enumerate(coords):
            assert 0.0 <= c <= 1.0, f"Coordinate {i} = {c} is outside [0, 1]"


class TestDrawPreview(BaseTest):
    """
    Tests for draw_preview() which saves annotated preview images
    with bounding box overlays.
    """

    def test_preview_image_is_saved(self, tmp_path):
        """
        TEST NAME: test_preview_image_is_saved
        COMPONENT: draw_preview()
        1. Create a dummy 512x512 BGR image.
        2. Call draw_preview() with arbitrary box coordinates.
        3. Assert the output file exists at the specified path.
        """
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        out_path = str(tmp_path / "preview.jpg")
        draw_preview(img, x=100, y=100, w=200, h=200, score=0.85, out_path=out_path)

        assert os.path.exists(out_path), "Preview image was not saved"

    def test_preview_image_has_same_dimensions(self, tmp_path):
        """
        TEST NAME: test_preview_image_has_same_dimensions
        COMPONENT: draw_preview()
        1. Create a 300x400 BGR image and draw a preview.
        2. Reload the saved preview image.
        3. Assert the saved image has the same spatial dimensions as the input.
        """
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        out_path = str(tmp_path / "preview.jpg")
        draw_preview(img, x=50, y=50, w=100, h=100, score=0.9, out_path=out_path)

        saved = cv2.imread(out_path)
        assert saved is not None, "Failed to reload saved preview"
        assert saved.shape[:2] == (300, 400), (
            f"Expected dimensions (300, 400), got {saved.shape[:2]}"
        )


class TestFindBestSquareRoi(BaseTest):
    """
    Tests for find_best_square_roi() which uses template matching
    to locate a cropped image region within the full source image.
    """

    def test_returns_tuple_with_five_elements(self, sample_full_image_512, sample_crop_from_center):
        """
        TEST NAME: test_returns_tuple_with_five_elements
        COMPONENT: find_best_square_roi()
        1. Provide a full image and a known crop from its center.
        2. Call find_best_square_roi().
        3. Assert the return value is a tuple of 5 elements (x, y, w, h, score).
        """
        result = find_best_square_roi(
            sample_full_image_512, sample_crop_from_center,
            min_size=40, max_size=300, step=8
        )
        assert result is not None, "find_best_square_roi returned None"
        assert len(result) == 5, f"Expected 5-element tuple, got {len(result)}"

    def test_match_score_is_positive_for_known_crop(self, sample_full_image_512, sample_crop_from_center):
        """
        TEST NAME: test_match_score_is_positive_for_known_crop
        COMPONENT: find_best_square_roi()
        1. Provide a full image and a crop taken directly from it.
        2. Call find_best_square_roi().
        3. Assert the matching score is positive (indicating a valid match).
        """
        result = find_best_square_roi(
            sample_full_image_512, sample_crop_from_center,
            min_size=40, max_size=300, step=8
        )
        assert result is not None, "find_best_square_roi returned None"
        _, _, _, _, score = result
        assert score > 0.0, f"Expected positive match score, got {score}"

    def test_roi_coordinates_within_image_bounds(self, sample_full_image_512, sample_crop_from_center):
        """
        TEST NAME: test_roi_coordinates_within_image_bounds
        COMPONENT: find_best_square_roi()
        1. Provide a 512x512 full image and a known crop.
        2. Call find_best_square_roi().
        3. Assert the returned ROI box (x, y, w, h) stays within image boundaries.
        """
        h, w = sample_full_image_512.shape[:2]
        result = find_best_square_roi(
            sample_full_image_512, sample_crop_from_center,
            min_size=40, max_size=300, step=8
        )
        assert result is not None, "find_best_square_roi returned None"
        x, y, rw, rh, _ = result

        assert x >= 0 and y >= 0, f"ROI origin ({x}, {y}) is negative"
        assert x + rw <= w, f"ROI exceeds image width: x={x}, w={rw}, img_w={w}"
        assert y + rh <= h, f"ROI exceeds image height: y={y}, h={rh}, img_h={h}"


class TestProcessOnePair(BaseTest):
    """
    Tests for process_one_pair() which runs the full auto-labeling pipeline
    for a single image pair (full + cropped) and writes a YOLO label file.
    """

    def test_label_file_created_for_matching_pair(self, tmp_path, sample_full_image_512, sample_crop_from_center):
        """
        TEST NAME: test_label_file_created_for_matching_pair
        COMPONENT: process_one_pair()
        1. Save a full image and its center crop to temp files.
        2. Call process_one_pair() to generate a YOLO label.
        3. Assert the output label file exists and is non-empty.
        """
        full_path = str(tmp_path / "full.png")
        crop_path = str(tmp_path / "crop.png")
        txt_path = str(tmp_path / "label.txt")

        cv2.imwrite(full_path, sample_full_image_512)
        cv2.imwrite(crop_path, sample_crop_from_center)

        process_one_pair(
            full_image_path=full_path,
            cropped_image_path=crop_path,
            output_txt_path=txt_path,
            class_id=0,
            min_size=40,
            max_size=300,
            step=8,
            min_score=0.0,
        )

        assert os.path.exists(txt_path), "Label file was not created"
        assert os.path.getsize(txt_path) > 0, "Label file is empty"

    def test_result_dict_contains_required_keys(self, tmp_path, sample_full_image_512, sample_crop_from_center):
        """
        TEST NAME: test_result_dict_contains_required_keys
        COMPONENT: process_one_pair()
        1. Save image pair to temp files and call process_one_pair().
        2. Capture the returned result dictionary.
        3. Assert it contains the keys: x, y, w, h, score, txt_path.
        """
        full_path = str(tmp_path / "full.png")
        crop_path = str(tmp_path / "crop.png")
        txt_path = str(tmp_path / "label.txt")

        cv2.imwrite(full_path, sample_full_image_512)
        cv2.imwrite(crop_path, sample_crop_from_center)

        result = process_one_pair(
            full_image_path=full_path,
            cropped_image_path=crop_path,
            output_txt_path=txt_path,
            class_id=0,
            min_size=40,
            max_size=300,
            step=8,
            min_score=0.0,
        )

        expected_keys = {"x", "y", "w", "h", "score", "txt_path"}
        self.assert_response_keys(result, expected_keys)

    def test_raises_on_missing_full_image(self, tmp_path):
        """
        TEST NAME: test_raises_on_missing_full_image
        COMPONENT: process_one_pair()
        1. Provide a nonexistent full image path and a valid crop path.
        2. Call process_one_pair().
        3. Assert that a ValueError is raised for the missing image.
        """
        crop_img = np.zeros((100, 100, 3), dtype=np.uint8)
        crop_path = str(tmp_path / "crop.png")
        cv2.imwrite(crop_path, crop_img)

        with pytest.raises(ValueError, match="Failed to load image"):
            process_one_pair(
                full_image_path=str(tmp_path / "nonexistent.png"),
                cropped_image_path=crop_path,
                output_txt_path=str(tmp_path / "label.txt"),
            )

    def test_raises_on_missing_crop_image(self, tmp_path):
        """
        TEST NAME: test_raises_on_missing_crop_image
        COMPONENT: process_one_pair()
        1. Provide a valid full image path and a nonexistent crop path.
        2. Call process_one_pair().
        3. Assert that a ValueError is raised for the missing crop.
        """
        full_img = np.zeros((512, 512, 3), dtype=np.uint8)
        full_path = str(tmp_path / "full.png")
        cv2.imwrite(full_path, full_img)

        with pytest.raises(ValueError, match="Failed to load crop image"):
            process_one_pair(
                full_image_path=full_path,
                cropped_image_path=str(tmp_path / "nonexistent.png"),
                output_txt_path=str(tmp_path / "label.txt"),
            )
