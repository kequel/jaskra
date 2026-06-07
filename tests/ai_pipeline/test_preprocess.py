import numpy as np
import pytest
import torch
from tests.base_test import BaseTest


class TestPreprocess(BaseTest):
    """
    Tests for GlaucomaPipeline._preprocess().
    Verifies output tensor shape, dtype, device placement,
    ImageNet normalisation correctness, and BGR→RGB channel conversion.
    """

    def test_output_tensor_shape(self, pipeline_instance, dummy_bgr_image):
        """
        TEST NAME: test_output_tensor_shape
        COMPONENT: GlaucomaPipeline._preprocess()
        1. Pass a 64x64 BGR image to _preprocess().
        2. Capture the returned tensor.
        3. Assert its shape is (1, 3, 512, 512).
        """
        tensor = pipeline_instance._preprocess(dummy_bgr_image)

        assert tensor.shape == (1, 3, 512, 512)

    def test_output_is_float32_tensor(self, pipeline_instance, dummy_bgr_image):
        """
        TEST NAME: test_output_is_float32_tensor
        COMPONENT: GlaucomaPipeline._preprocess()
        1. Pass a uint8 BGR image to _preprocess().
        2. Capture the returned tensor.
        3. Assert dtype is torch.float32.
        """
        tensor = pipeline_instance._preprocess(dummy_bgr_image)

        assert tensor.dtype == torch.float32

    def test_output_is_on_cpu(self, pipeline_instance, dummy_bgr_image):
        """
        TEST NAME: test_output_is_on_cpu
        COMPONENT: GlaucomaPipeline._preprocess()
        1. Initialise a pipeline with device=cpu.
        2. Pass a dummy BGR image to _preprocess().
        3. Assert the tensor's device type is "cpu".
        """
        tensor = pipeline_instance._preprocess(dummy_bgr_image)

        assert tensor.device.type == "cpu"

    def test_normalization_shifts_mean_for_white_image(self, pipeline_instance):
        """
        TEST NAME: test_normalization_shifts_mean_for_white_image
        COMPONENT: GlaucomaPipeline._preprocess()
        1. Create a 64x64 all-white BGR image (pixel value 255).
        2. Pass it to _preprocess().
        3. Assert channel 0 mean ≈ (1.0 - 0.485) / 0.229 ≈ 2.249 (ImageNet normalisation).
        """
        white_bgr = np.full((64, 64, 3), 255, dtype=np.uint8)

        tensor = pipeline_instance._preprocess(white_bgr)

        expected = pytest.approx((1.0 - 0.485) / 0.229, abs=0.01)
        assert tensor[0, 0].mean().item() == expected

    def test_bgr_to_rgb_conversion(self, pipeline_instance):
        """
        TEST NAME: test_bgr_to_rgb_conversion
        COMPONENT: GlaucomaPipeline._preprocess()
        1. Create a 64x64 pure-blue BGR image (channel 0 = 255, others = 0).
        2. Pass it to _preprocess().
        3. Assert tensor channel 2 mean is greater than channel 0 mean,
           confirming BGR was converted to RGB before normalisation.
        """
        blue_bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        blue_bgr[:, :, 0] = 255

        tensor = pipeline_instance._preprocess(blue_bgr)

        assert tensor[0, 2].mean().item() > tensor[0, 0].mean().item()
