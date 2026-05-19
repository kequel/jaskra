import numpy as np
import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_yolo_result(bboxes_xyxy: np.ndarray) -> MagicMock:
    result = MagicMock()
    boxes = MagicMock()
    boxes.xyxy = MagicMock()
    boxes.xyxy.cpu.return_value.numpy.return_value = bboxes_xyxy
    result.boxes = boxes
    return result


def _make_unet_output(disc_val: float = 0.9, cup_val: float = 0.7) -> torch.Tensor:
    output = torch.zeros(1, 2, 512, 512)
    output[0, 0] = disc_val
    output[0, 1] = cup_val
    return output


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def dummy_bgr_image() -> np.ndarray:
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)


@pytest.fixture()
def saved_test_image(tmp_path: Path, dummy_bgr_image: np.ndarray) -> Path:
    import cv2
    img_path = tmp_path / "test_fundus.png"
    cv2.imwrite(str(img_path), dummy_bgr_image)
    return img_path


@pytest.fixture()
def pipeline_instance():
    with (
        patch("pipeline.YOLO") as mock_yolo_cls,
        patch("pipeline.smp.UnetPlusPlus") as mock_unet_cls,
        patch("pipeline.torch.load") as mock_torch_load,
    ):
        mock_yolo_instance = MagicMock()
        mock_yolo_instance.to.return_value = mock_yolo_instance
        mock_yolo_cls.return_value = mock_yolo_instance

        mock_unet_instance = MagicMock()
        mock_unet_instance.to.return_value = mock_unet_instance
        mock_unet_instance.load_state_dict.return_value = None
        mock_unet_cls.return_value = mock_unet_instance

        mock_torch_load.return_value = {"model_state_dict": {}}

        from pipeline import GlaucomaPipeline
        instance = GlaucomaPipeline(
            yolo_path="fake/yolo.pt",
            unet_path="fake/unet.pth",
            masks_dir=None,
            device=torch.device("cpu"),
        )
        instance._mock_yolo = mock_yolo_instance
        instance._mock_unet = mock_unet_instance
        return instance


# ---------------------------------------------------------------------------
# Unit Tests — calculate_cdr
# ---------------------------------------------------------------------------

class TestCalculateCdr:

    def test_correct_ratio_with_known_masks(self, pipeline_instance):
        # disc: 400px, cup: 100px -> CDR = sqrt(100/400) = 0.5
        disc = np.ones((20, 20), dtype=bool)
        cup = np.zeros((20, 20), dtype=bool)
        cup[:10, :10] = True

        cdr = pipeline_instance.calculate_cdr(disc, cup)
        assert pytest.approx(cdr, abs=1e-6) == 0.5

    def test_zero_disc_area_returns_zero(self, pipeline_instance):
        # empty disc mask must not cause division by zero
        disc = np.zeros((10, 10), dtype=bool)
        cup = np.ones((10, 10), dtype=bool)

        cdr = pipeline_instance.calculate_cdr(disc, cup)
        assert cdr == 0.0

    def test_full_cup_equal_to_disc(self, pipeline_instance):
        disc = np.ones((10, 10), dtype=bool)
        cup = np.ones((10, 10), dtype=bool)

        cdr = pipeline_instance.calculate_cdr(disc, cup)
        assert pytest.approx(cdr, abs=1e-6) == 1.0

    def test_zero_cup_area(self, pipeline_instance):
        disc = np.ones((10, 10), dtype=bool)
        cup = np.zeros((10, 10), dtype=bool)

        cdr = pipeline_instance.calculate_cdr(disc, cup)
        assert cdr == 0.0


# ---------------------------------------------------------------------------
# Unit Tests — _preprocess
# ---------------------------------------------------------------------------

class TestPreprocess:

    def test_output_tensor_shape(self, pipeline_instance, dummy_bgr_image):
        tensor = pipeline_instance._preprocess(dummy_bgr_image)
        assert tensor.shape == (1, 3, 512, 512)

    def test_output_is_float_tensor(self, pipeline_instance, dummy_bgr_image):
        tensor = pipeline_instance._preprocess(dummy_bgr_image)
        assert tensor.dtype == torch.float32

    def test_output_is_on_cpu(self, pipeline_instance, dummy_bgr_image):
        tensor = pipeline_instance._preprocess(dummy_bgr_image)
        assert tensor.device.type == "cpu"

    def test_normalization_shifts_mean(self, pipeline_instance):
        # white image channel 0 (R) after ImageNet normalization: (1.0 - 0.485) / 0.229 ≈ 2.249
        white_bgr = np.full((64, 64, 3), 255, dtype=np.uint8)
        tensor = pipeline_instance._preprocess(white_bgr)

        expected = pytest.approx((1.0 - 0.485) / 0.229, abs=0.01)
        assert tensor[0, 0].mean().item() == expected

    def test_bgr_to_rgb_conversion(self, pipeline_instance):
        # pure blue in BGR -> after conversion, channel 2 (B in RGB) should dominate
        blue_bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        blue_bgr[:, :, 0] = 255

        tensor = pipeline_instance._preprocess(blue_bgr)
        assert tensor[0, 2].mean().item() > tensor[0, 0].mean().item()


# ---------------------------------------------------------------------------
# Unit Tests — run (fallback & detection)
# ---------------------------------------------------------------------------

class TestRunFallback:

    def test_fallback_when_yolo_returns_empty_bboxes(self, pipeline_instance, saved_test_image):
        # no detections -> pipeline should fall back to full image ROI
        pipeline_instance._mock_yolo.return_value = [
            _make_yolo_result(np.empty((0, 4), dtype=np.float32))
        ]
        pipeline_instance._mock_unet.return_value = _make_unet_output()

        assert pipeline_instance.run(str(saved_test_image)) is not None

    def test_fallback_bbox_covers_full_image(self, pipeline_instance, saved_test_image, dummy_bgr_image):
        h, w = dummy_bgr_image.shape[:2]
        pipeline_instance._mock_yolo.return_value = [
            _make_yolo_result(np.empty((0, 4), dtype=np.float32))
        ]
        pipeline_instance._mock_unet.return_value = _make_unet_output()

        _, crops, _, _, _, _ = pipeline_instance.run(str(saved_test_image))

        assert len(crops) == 1
        x1, y1, x2, y2 = crops[0]
        assert (x1, y1) == (0, 0)
        assert (x2, y2) == (w, h)

    def test_run_returns_none_for_missing_image(self, pipeline_instance):
        assert pipeline_instance.run("nonexistent/path/image.png") is None


class TestRunWithDetection:

    def test_run_with_valid_yolo_detection(self, pipeline_instance, saved_test_image, dummy_bgr_image):
        h, w = dummy_bgr_image.shape[:2]
        bbox = np.array([[5, 5, w - 5, h - 5]], dtype=np.float32)

        pipeline_instance._mock_yolo.return_value = [_make_yolo_result(bbox)]
        pipeline_instance._mock_unet.return_value = _make_unet_output()

        output = pipeline_instance.run(str(saved_test_image))
        assert output is not None
        assert len(output) == 6

    def test_run_cdr_is_float_in_valid_range(self, pipeline_instance, saved_test_image, dummy_bgr_image):
        h, w = dummy_bgr_image.shape[:2]
        pipeline_instance._mock_yolo.return_value = [
            _make_yolo_result(np.array([[0, 0, w, h]], dtype=np.float32))
        ]
        pipeline_instance._mock_unet.return_value = _make_unet_output(disc_val=2.0, cup_val=1.0)

        _, _, _, cdr, _, _ = pipeline_instance.run(str(saved_test_image))
        assert isinstance(cdr, float)
        assert 0.0 <= cdr <= 1.0

    def test_run_returns_correct_masks_shape(self, pipeline_instance, saved_test_image, dummy_bgr_image):
        h, w = dummy_bgr_image.shape[:2]
        pipeline_instance._mock_yolo.return_value = [
            _make_yolo_result(np.array([[0, 0, w, h]], dtype=np.float32))
        ]
        pipeline_instance._mock_unet.return_value = _make_unet_output()

        _, _, masks, _, _, _ = pipeline_instance.run(str(saved_test_image))
        assert len(masks) >= 1
        assert masks[0].shape == (2, 512, 512)


# ---------------------------------------------------------------------------
# Integration Tests — full run flow
# ---------------------------------------------------------------------------

class TestRunIntegration:

    def test_full_pipeline_returns_correct_types(self, pipeline_instance, saved_test_image, dummy_bgr_image):
        h, w = dummy_bgr_image.shape[:2]
        pipeline_instance._mock_yolo.return_value = [
            _make_yolo_result(np.array([[0, 0, w, h]], dtype=np.float32))
        ]
        pipeline_instance._mock_unet.return_value = _make_unet_output()

        result = pipeline_instance.run(str(saved_test_image))
        assert result is not None

        full_img, crops, masks, cdr, gt_masks, cdr_gt = result
        assert isinstance(full_img, np.ndarray)
        assert isinstance(crops, list)
        assert isinstance(masks, list)
        assert isinstance(cdr, float)
        assert gt_masks is None or isinstance(gt_masks, np.ndarray)
        assert isinstance(cdr_gt, float)

    def test_full_pipeline_with_multiple_bboxes(self, pipeline_instance, saved_test_image, dummy_bgr_image):
        h, w = dummy_bgr_image.shape[:2]
        bboxes = np.array([
            [0, 0, w // 2, h // 2],
            [w // 2, 0, w, h],
        ], dtype=np.float32)

        pipeline_instance._mock_yolo.return_value = [_make_yolo_result(bboxes)]
        pipeline_instance._mock_unet.return_value = _make_unet_output()

        _, _, masks, _, _, _ = pipeline_instance.run(str(saved_test_image))
        assert len(masks) == 2

    def test_no_exception_raised_during_full_flow(self, pipeline_instance, saved_test_image, dummy_bgr_image):
        h, w = dummy_bgr_image.shape[:2]
        pipeline_instance._mock_yolo.return_value = [
            _make_yolo_result(np.array([[0, 0, w, h]], dtype=np.float32))
        ]
        pipeline_instance._mock_unet.return_value = _make_unet_output()

        try:
            pipeline_instance.run(str(saved_test_image))
        except Exception as exc:
            pytest.fail(f"run() raised an unexpected exception: {exc}")

    def test_gt_masks_none_when_no_masks_dir(self, pipeline_instance, saved_test_image, dummy_bgr_image):
        assert pipeline_instance.masks_dir is None
        h, w = dummy_bgr_image.shape[:2]
        pipeline_instance._mock_yolo.return_value = [
            _make_yolo_result(np.array([[0, 0, w, h]], dtype=np.float32))
        ]
        pipeline_instance._mock_unet.return_value = _make_unet_output()

        _, _, _, _, gt_masks, cdr_gt = pipeline_instance.run(str(saved_test_image))
        assert gt_masks is None
        assert cdr_gt == 0.0


# ---------------------------------------------------------------------------
# Integration Tests — save_diagnostic_plot
# ---------------------------------------------------------------------------

class TestSaveDiagnosticPlot:

    @pytest.fixture()
    def plot_inputs(self, dummy_bgr_image):
        h, w = dummy_bgr_image.shape[:2]
        crops = [(0, 0, w, h)]
        mask = np.zeros((2, 512, 512), dtype=np.float32)
        mask[0, :200, :200] = 0.9
        mask[1, :100, :100] = 0.8
        return dummy_bgr_image, crops, [mask], 0.707, None, 0.0

    def _make_mock_plt(self):
        mock_fig = MagicMock()
        mock_axes = np.empty((2, 2), dtype=object)
        ax_mocks = []
        for i in range(2):
            for j in range(2):
                ax = MagicMock()
                mock_axes[i, j] = ax
                ax_mocks.append(ax)
        return mock_fig, mock_axes, ax_mocks

    def test_savefig_called_with_correct_path(self, tmp_path, plot_inputs):
        from pipeline import save_diagnostic_plot

        save_path = tmp_path / "test_report.png"
        img, crops, masks, cdr, gt_m, cdr_gt = plot_inputs

        with patch("pipeline.plt") as mock_plt:
            mock_fig, mock_axes, _ = self._make_mock_plt()
            mock_plt.subplots.return_value = (mock_fig, mock_axes)
            save_diagnostic_plot(img, crops, masks, cdr, gt_m, cdr_gt, save_path)
            mock_plt.savefig.assert_called_once_with(save_path)

    def test_plt_close_called_after_save(self, tmp_path, plot_inputs):
        from pipeline import save_diagnostic_plot

        save_path = tmp_path / "report.png"
        img, crops, masks, cdr, gt_m, cdr_gt = plot_inputs

        with patch("pipeline.plt") as mock_plt:
            mock_fig, mock_axes, _ = self._make_mock_plt()
            mock_plt.subplots.return_value = (mock_fig, mock_axes)
            save_diagnostic_plot(img, crops, masks, cdr, gt_m, cdr_gt, save_path)
            mock_plt.close.assert_called_once()

    def test_all_four_subplots_receive_data(self, tmp_path, plot_inputs):
        from pipeline import save_diagnostic_plot

        save_path = tmp_path / "report.png"
        img, crops, masks, cdr, gt_m, cdr_gt = plot_inputs

        with patch("pipeline.plt") as mock_plt:
            mock_fig, mock_axes, ax_mocks = self._make_mock_plt()
            mock_plt.subplots.return_value = (mock_fig, mock_axes)
            save_diagnostic_plot(img, crops, masks, cdr, gt_m, cdr_gt, save_path)

            for ax in ax_mocks[:3]:
                ax.imshow.assert_called()
            # no GT -> axes[1,1] uses text() instead of imshow()
            ax_mocks[3].text.assert_called()

    def test_with_ground_truth_masks(self, tmp_path, dummy_bgr_image):
        from pipeline import save_diagnostic_plot

        h, w = dummy_bgr_image.shape[:2]
        mask = np.zeros((2, 512, 512), dtype=np.float32)
        mask[0, :200, :200] = 0.9
        mask[1, :100, :100] = 0.8

        gt_masks = np.zeros((2, 512, 512), dtype=np.float32)
        gt_masks[0, :180, :180] = 1.0
        gt_masks[1, :90, :90] = 1.0

        with patch("pipeline.plt") as mock_plt:
            mock_fig, mock_axes, ax_mocks = self._make_mock_plt()
            mock_plt.subplots.return_value = (mock_fig, mock_axes)
            save_diagnostic_plot(
                dummy_bgr_image, [(0, 0, w, h)], [mask],
                cdr=0.5, gt_masks=gt_masks, cdr_gt=0.45,
                save_path=tmp_path / "report_gt.png",
            )
            # GT present -> axes[1,1] should call imshow(), not text()
            ax_mocks[3].imshow.assert_called()
            ax_mocks[3].text.assert_not_called()
