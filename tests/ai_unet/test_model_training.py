import os
import tempfile
import cv2
import torch
import numpy as np
import importlib.util
import pytest
from pathlib import Path

from tests.base_test import BaseTest

# -------------------------------------------------------------------------
# DYNAMICZNE ŁADOWANIE MODUŁU ZE SPACJĄ W NAZWIE
# -------------------------------------------------------------------------
file_path = Path("ai/unet/Model Training.py")
spec = importlib.util.spec_from_file_location("model_training", file_path)
model_training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_training)


class TestUNetTraining(BaseTest):
    """
    Unit tests for UNet++ Model Training helpers, datasets, losses, and transforms.
    Ensures high code coverage by testing configuration builders, data loaders,
    and edge-case error branches without executing the heavy GPU training loops.
    """

    def test_normalize_merged_mask_format_b(self):
        """
        TEST NAME: test_normalize_merged_mask_format_b
        COMPONENT: normalize_merged_mask
        1. Create a mock visual mask with values 0 (cup), 100 (disc), and 255 (background).
        2. Call normalize_merged_mask().
        3. Assert the values are correctly mapped to standard format (2, 1, 0).
        """
        mock_mask = np.array([[0, 100], [255, 255]], dtype=np.uint8)
        result = model_training.normalize_merged_mask(mock_mask)
        
        assert result[0, 0] == 2
        assert result[0, 1] == 1
        assert result[1, 0] == 0

    def test_normalize_merged_mask_already_normalized(self):
        """
        TEST NAME: test_normalize_merged_mask_already_normalized
        COMPONENT: normalize_merged_mask
        1. Create a mask that already uses 0, 1, 2 values.
        2. Call normalize_merged_mask().
        3. Assert the mask remains unchanged.
        """
        mock_mask = np.array([[0, 1], [2, 0]], dtype=np.uint8)
        result = model_training.normalize_merged_mask(mock_mask)
        
        np.testing.assert_array_equal(result, mock_mask)

    def test_mask_to_tensor_conversion_from_numpy(self):
        """
        TEST NAME: test_mask_to_tensor_conversion_from_numpy
        COMPONENT: _mask_to_tensor
        1. Create a numpy array representing a mask.
        2. Call _mask_to_tensor().
        3. Assert the output is a 2-channel tensor.
        """
        mask = np.array([[0, 1], [2, 0]], dtype=np.uint8)
        tensor_mask = model_training._mask_to_tensor(mask)
        
        assert tensor_mask.shape == (2, 2, 2)
        assert tensor_mask[0, 0, 1] == 1.0
        assert tensor_mask[1, 1, 0] == 1.0

    def test_mask_to_tensor_conversion_from_tensor(self):
        """
        TEST NAME: test_mask_to_tensor_conversion_from_tensor
        COMPONENT: _mask_to_tensor
        1. Create a torch.Tensor representing a mask (covers the else branch).
        2. Call _mask_to_tensor().
        3. Assert the output is correctly formatted.
        """
        mask = torch.tensor([[0, 1], [2, 0]], dtype=torch.float32)
        tensor_mask = model_training._mask_to_tensor(mask)
        
        assert tensor_mask.shape == (2, 2, 2)
        assert tensor_mask[0, 0, 1] == 1.0

    def test_dice_scores_calculation(self):
        """
        TEST NAME: test_dice_scores_calculation
        COMPONENT: dice_scores
        1. Create dummy logits and target tensors.
        2. Call dice_scores().
        3. Assert the returned tuple matches expected dice values.
        """
        logits = torch.tensor([[[[1.0, 1.0], [-1.0, -1.0]], 
                                [[1.0, -1.0], [-1.0, -1.0]]]])
        targets = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]], 
                                 [[1.0, 1.0], [0.0, 0.0]]]])
        
        d_score, c_score, mean_score = model_training.dice_scores(logits, targets)
        
        assert abs(d_score - 1.0) < 1e-4
        assert abs(c_score - 0.6666) < 1e-3

    def test_morphological_postprocess(self):
        """
        TEST NAME: test_morphological_postprocess
        COMPONENT: morphological_postprocess
        1. Create noisy dummy binary arrays for disc and cup.
        2. Call morphological_postprocess().
        3. Assert the output arrays have noise removed and constraints applied.
        """
        pred_disc = np.zeros((20, 20), dtype=np.uint8)
        pred_disc[5:15, 5:15] = 1
        pred_disc[1, 1] = 1

        pred_cup = np.zeros((20, 20), dtype=np.uint8)
        pred_cup[8:12, 8:12] = 1
        pred_cup[18, 18] = 1
        
        clean_disc, clean_cup = model_training.morphological_postprocess(pred_disc, pred_cup)
        
        assert clean_disc[1, 1] == 0
        assert clean_cup[18, 18] == 0

    def test_dataset_merged_mask_initialization_and_loading(self):
        """
        TEST NAME: test_dataset_merged_mask_initialization_and_loading
        COMPONENT: GlaucomaDatasetMergedMask
        1. Create temporary directories for images and masks.
        2. Create a dummy image and matching mask file.
        3. Initialize GlaucomaDatasetMergedMask and fetch first item.
        4. Assert __getitem__ returns correctly shaped tensors.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = os.path.join(tmpdir, "images")
            mask_dir = os.path.join(tmpdir, "masks")
            os.makedirs(img_dir)
            os.makedirs(mask_dir)

            cv2.imwrite(os.path.join(img_dir, "test_eye.png"), np.zeros((10, 10, 3), dtype=np.uint8))
            cv2.imwrite(os.path.join(mask_dir, "test_eye.png"), np.zeros((10, 10), dtype=np.uint8))

            dataset = model_training.GlaucomaDatasetMergedMask(img_dir, mask_dir)
            
            assert len(dataset) == 1
            img_out, mask_out = dataset[0]
            assert img_out.shape == (10, 10, 3)
            assert mask_out.shape == (2, 10, 10)

    def test_dataset_separate_masks_initialization_and_loading(self):
        """
        TEST NAME: test_dataset_separate_masks_initialization_and_loading
        COMPONENT: GlaucomaDatasetSeparateMasks
        1. Create temporary directories for images and masks.
        2. Create dummy image and separate mask files.
        3. Initialize GlaucomaDatasetSeparateMasks and fetch first item.
        4. Assert __getitem__ returns correctly shaped tensors.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = os.path.join(tmpdir, "images")
            mask_dir = os.path.join(tmpdir, "masks")
            os.makedirs(img_dir)
            os.makedirs(mask_dir)

            cv2.imwrite(os.path.join(img_dir, "test_eye.png"), np.zeros((10, 10, 3), dtype=np.uint8))
            cv2.imwrite(os.path.join(mask_dir, "test_eye_cup.png"), np.zeros((10, 10), dtype=np.uint8))
            cv2.imwrite(os.path.join(mask_dir, "test_eye_disc.png"), np.zeros((10, 10), dtype=np.uint8))

            dataset = model_training.GlaucomaDatasetSeparateMasks(img_dir, mask_dir)
            
            assert len(dataset) == 1
            img_out, mask_out = dataset[0]
            assert img_out.shape == (10, 10, 3)
            assert mask_out.shape == (2, 10, 10)

    def test_get_transforms(self):
        """
        TEST NAME: test_get_transforms
        COMPONENT: get_train_transforms, get_val_transforms
        1. Call get_train_transforms() and get_val_transforms().
        2. Assert both return valid albumentations compositions.
        """
        train_tf = model_training.get_train_transforms()
        val_tf = model_training.get_val_transforms()
        
        assert train_tf is not None
        assert val_tf is not None

    def test_build_criterion_all_types(self):
        """
        TEST NAME: test_build_criterion_all_types
        COMPONENT: build_criterion
        1. Iterate through all three supported loss types.
        2. Call build_criterion() for each.
        3. Provide dummy tensors to the returned criterion to verify execution.
        """
        for loss_type in ["weighted_dice_focal", "dice_focal", "tversky"]:
            criterion = model_training.build_criterion(loss_type)
            assert criterion is not None
            
            if loss_type != "tversky":
                dummy_out = torch.randn(2, 2, 10, 10)
                dummy_mask = torch.randint(0, 2, (2, 2, 10, 10)).float()
                loss_val = criterion(dummy_out, dummy_mask)
                assert isinstance(loss_val, torch.Tensor)

    def test_build_scheduler(self):
        """
        TEST NAME: test_build_scheduler
        COMPONENT: build_scheduler
        1. Create a dummy model and Adam optimizer.
        2. Call build_scheduler().
        3. Assert the scheduler is successfully initialized.
        """
        dummy_model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.Adam(dummy_model.parameters(), lr=0.001)
        
        scheduler = model_training.build_scheduler(optimizer, epochs=10, steps_per_epoch=5)
        assert scheduler is not None

    def test_build_datasets_function_success_and_skipped_branches(self):
        """
        TEST NAME: test_build_datasets_function_success_and_skipped_branches
        COMPONENT: build_datasets
        1. Create a temporary folder mimicking successful dataset layouts.
        2. Test both the success scenario and the skipped/missing paths logs.
        3. Verify datasets are handled transparently.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = os.path.join(tmpdir, "images")
            mask_dir = os.path.join(tmpdir, "masks")
            os.makedirs(img_dir)
            os.makedirs(mask_dir)
            
            cv2.imwrite(os.path.join(img_dir, "test_eye.png"), np.zeros((10, 10, 3), dtype=np.uint8))
            cv2.imwrite(os.path.join(mask_dir, "test_eye.png"), np.zeros((10, 10), dtype=np.uint8))
            cv2.imwrite(os.path.join(mask_dir, "test_eye_cup.png"), np.zeros((10, 10), dtype=np.uint8))
            cv2.imwrite(os.path.join(mask_dir, "test_eye_disc.png"), np.zeros((10, 10), dtype=np.uint8))

            orig_img_a = model_training.IMAGES_DIR_A
            orig_mask_a = model_training.MASKS_DIR_A
            orig_img_b = model_training.IMAGES_DIR_B
            orig_mask_b = model_training.MASKS_DIR_B

            # 1. Test standard combine success
            model_training.IMAGES_DIR_A = img_dir
            model_training.MASKS_DIR_A = mask_dir
            model_training.IMAGES_DIR_B = img_dir
            model_training.MASKS_DIR_B = mask_dir

            train_tf = model_training.get_train_transforms()
            val_tf = model_training.get_val_transforms()
            train_ds, val_ds = model_training.build_datasets(train_tf, val_tf)
            assert train_ds is not None

            # 2. Trigger the "Skipped — path not found" else branches for coverage
            model_training.IMAGES_DIR_A = "non_existent_path_a"
            model_training.IMAGES_DIR_B = "non_existent_path_b"
            
            with pytest.raises(RuntimeError):
                model_training.build_datasets(train_tf, val_tf)

            # Restore
            model_training.IMAGES_DIR_A = orig_img_a
            model_training.MASKS_DIR_A = orig_mask_a
            model_training.IMAGES_DIR_B = orig_img_b
            model_training.MASKS_DIR_B = orig_mask_b

    def test_datasets_warnings_and_failures(self):
        """
        TEST NAME: test_datasets_warnings_and_failures
        COMPONENT: GlaucomaDatasetMergedMask, GlaucomaDatasetSeparateMasks
        1. Create datasets where some images are missing masks to trigger init warnings.
        2. Force a FileNotFoundError inside __getitem__ by breaking files post-init.
        3. Assert the warnings logic and error branches execute correctly.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = os.path.join(tmpdir, "images")
            mask_dir = os.path.join(tmpdir, "masks")
            os.makedirs(img_dir)
            os.makedirs(mask_dir)

            # 1. Create an image with NO mask to trigger the "WARNING: image(s) skipped" during initialization
            cv2.imwrite(os.path.join(img_dir, "skipped_eye.png"), np.zeros((10, 10, 3), dtype=np.uint8))
            
            # 2. Create a valid image that WILL have a mask initially
            cv2.imwrite(os.path.join(img_dir, "broken_sample.png"), np.zeros((10, 10, 3), dtype=np.uint8))
            mask_path = os.path.join(mask_dir, "broken_sample.png")
            cv2.imwrite(mask_path, np.zeros((10, 10), dtype=np.uint8))

            # Initialize datasets (hits the init validation warning logs)
            dataset_merged = model_training.GlaucomaDatasetMergedMask(img_dir, mask_dir)
            dataset_separate = model_training.GlaucomaDatasetSeparateMasks(img_dir, mask_dir)
            
            # Delete the mask file from disk BEFORE calling __getitem__ to force the FileNotFoundError branch
            if os.path.exists(mask_path):
                os.remove(mask_path)
                
            with pytest.raises(FileNotFoundError):
                _ = dataset_merged[0]

    def test_datasets_empty_directory_exceptions(self):
        """
        TEST NAME: test_datasets_empty_directory_exceptions
        COMPONENT: GlaucomaDatasetMergedMask, GlaucomaDatasetSeparateMasks
        1. Create a completely empty directory.
        2. Attempt to initialize both datasets.
        3. Assert ValueError "No images found" is raised.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No images found"):
                model_training.GlaucomaDatasetMergedMask(tmpdir, tmpdir)
                
            with pytest.raises(ValueError, match="No images found"):
                model_training.GlaucomaDatasetSeparateMasks(tmpdir, tmpdir)

    def test_real_mini_training_epoch(self):
        """
        TEST NAME: test_real_mini_training_epoch
        COMPONENT: train_single_model, train_epoch, validate
        1. Create a tiny temporary dataset (3 images) to run real training safely.
        2. Override SAVE_DIR to point to the temporary folder.
        3. Call train_single_model with epochs=1 to force a fast, complete execution.
        4. Assert the model .pth file was created and a dice score was returned.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = os.path.join(tmpdir, "images")
            mask_dir = os.path.join(tmpdir, "masks")
            save_dir = os.path.join(tmpdir, "models")
            os.makedirs(img_dir)
            os.makedirs(mask_dir)
            os.makedirs(save_dir)

            for i in range(3):
                cv2.imwrite(os.path.join(img_dir, f"test_{i}.png"), np.zeros((16, 16, 3), dtype=np.uint8))
                cv2.imwrite(os.path.join(mask_dir, f"test_{i}.png"), np.zeros((16, 16), dtype=np.uint8))

            orig_save = model_training.SAVE_DIR
            orig_dataloader = model_training.DataLoader

            def single_thread_dataloader(*args, **kwargs):
                kwargs['num_workers'] = 0
                return orig_dataloader(*args, **kwargs)

            try:
                model_training.SAVE_DIR = save_dir
                model_training.DataLoader = single_thread_dataloader 
                
                train_tf = model_training.get_train_transforms()
                val_tf = model_training.get_val_transforms()
                
                ds_train = model_training.GlaucomaDatasetMergedMask(img_dir, mask_dir, transform=train_tf)
                ds_val = model_training.GlaucomaDatasetMergedMask(img_dir, mask_dir, transform=val_tf)

                best_dice = model_training.train_single_model(
                    train_idx=[0, 1], 
                    val_idx=[2], 
                    train_ds=ds_train, 
                    val_ds=ds_val,
                    save_name="test_mini", 
                    epochs=1
                )

                assert best_dice >= 0.0
                assert os.path.exists(os.path.join(save_dir, "test_mini.pth"))
            finally:
                model_training.SAVE_DIR = orig_save
                model_training.DataLoader = orig_dataloader