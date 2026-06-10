import numpy as np
import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# YOLO / UNet mock helpers
# ---------------------------------------------------------------------------

def make_yolo_result(bboxes_xyxy: np.ndarray) -> MagicMock:
    result = MagicMock()
    boxes = MagicMock()
    boxes.xyxy = MagicMock()
    boxes.xyxy.cpu.return_value.numpy.return_value = bboxes_xyxy
    result.boxes = boxes
    return result


def make_unet_output(disc_val: float = 0.9, cup_val: float = 0.7) -> torch.Tensor:
    output = torch.zeros(1, 2, 512, 512)
    output[0, 0] = disc_val
    output[0, 1] = cup_val
    return output


# ---------------------------------------------------------------------------
# Image fixtures
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


# ---------------------------------------------------------------------------
# Pipeline fixture
# ---------------------------------------------------------------------------

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
# Diagnostic plot fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def plot_inputs(dummy_bgr_image):
    h, w = dummy_bgr_image.shape[:2]
    crops = [(0, 0, w, h)]
    mask = np.zeros((2, 512, 512), dtype=np.float32)
    mask[0, :200, :200] = 0.9
    mask[1, :100, :100] = 0.8
    return dummy_bgr_image, crops, [mask], 0.707, None, 0.0


@pytest.fixture()
def mock_plt_subplots():
    """Returns (mock_fig, mock_axes_2x2, flat_list_of_ax_mocks)."""
    mock_fig = MagicMock()
    mock_axes = np.empty((2, 2), dtype=object)
    ax_mocks = []
    for i in range(2):
        for j in range(2):
            ax = MagicMock()
            mock_axes[i, j] = ax
            ax_mocks.append(ax)
    return mock_fig, mock_axes, ax_mocks
