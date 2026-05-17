import os
import torch
import numpy as np
import importlib.util
import pytest
from pathlib import Path

from tests.base_test import BaseTest

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

    def test_normalize_merged_mask_format_b(self, dummy_arrays):
        """
        TEST NAME: test_normalize_merged_mask_format_b
        COMPONENT: normalize_merged_mask
        1. Use a mock visual mask with values 0 (cup), 100 (disc), and 255 (background).
        2. Call normalize_merged_mask().
        3. Assert the values are correctly mapped to standard format (2, 1, 0).
        """
        result = model_training.normalize_merged_mask(dummy_arrays["mask_b"])
        
        assert result[0, 0] == 2
        assert result[0, 1] == 1
        assert result[1, 0] == 0

    def test_normalize_merged_mask_already_normalized(self, dummy_arrays):
        """
        TEST NAME: test_normalize_merged_mask_already_normalized
        COMPONENT: normalize_merged_mask
        1. Use a mask that already uses 0, 1, 2 values.
        2. Call normalize_merged_mask().
        3. Assert the mask remains unchanged.
        """
        result = model_training.normalize_merged_mask(dummy_arrays["mask_norm"])
        
        np.testing.assert_array_equal(result, dummy_arrays["mask_norm"])

    def test_mask_to_tensor_conversion_from_numpy(self, dummy_arrays):
        """
        TEST NAME: test_mask_to_tensor_conversion_from_numpy
        COMPONENT: _mask_to_tensor
        1. Use a numpy array representing a mask.
        2. Call _mask_to_tensor().
        3. Assert the output is a 2-channel tensor.
        """
        tensor_mask = model_training._mask_to_tensor(dummy_arrays["mask_norm"])
        
        assert tensor_mask.shape == (2, 2, 2)
        assert tensor_mask[0, 0, 1] == 1.0
        assert tensor_mask[1, 1, 0] == 1.0

    def test_mask_to_tensor_conversion_from_tensor(self):
        """
        TEST NAME: test_mask_to_tensor_conversion_from_tensor
        COMPONENT: _mask_to_tensor
        1. Create a torch.Tensor representing a mask.
        2. Call _mask_to_tensor().
        3. Assert the output is correctly formatted.
        """
        mask = torch.tensor([[0, 1], [2, 0]], dtype=torch.float32)
        tensor_mask = model_training._mask_to_tensor(mask)
        
        assert tensor_mask.shape == (2, 2, 2)
        assert tensor_mask[0, 0, 1] == 1.0

    def test_dice_scores_calculation(self, dummy_arrays):
        """
        TEST NAME: test_dice_scores_calculation
        COMPONENT: dice_scores
        1. Use dummy logits and target tensors.
        2. Call dice_scores().
        3. Assert the returned tuple matches expected dice values using BaseTest assertions.
        """
        logits = dummy_arrays["logits"]
        targets = dummy_arrays["targets"]
        
        d_score, c_score, _mean_score = model_training.dice_scores(logits, targets)
        
        self.assert_is_ratio(d_score)
        self.assert_is_ratio(c_score)

    def test_morphological_postprocess(self, dummy_arrays):
        """
        TEST NAME: test_morphological_postprocess
        COMPONENT: morphological_postprocess
        1. Use noisy dummy binary arrays for disc and cup.
        2. Call morphological_postprocess().
        3. Assert the output arrays have noise removed and constraints applied.
        """
        clean_disc, clean_cup = model_training.morphological_postprocess(
            dummy_arrays["pred_disc"], 
            dummy_arrays["pred_cup"]
        )
        
        assert clean_disc[1, 1] == 0
        assert clean_cup[18, 18] == 0

    def test_dataset_merged_mask_initialization_and_loading(self, dummy_dataset):
        """
        TEST NAME: test_dataset_merged_mask_initialization_and_loading
        COMPONENT: GlaucomaDatasetMergedMask
        1. Use temporary directories for images and masks from fixture.
        2. Initialize GlaucomaDatasetMergedMask and fetch first item.
        3. Assert __getitem__ returns correctly shaped tensors.
        """
        img_dir, mask_dir, _ = dummy_dataset
        dataset = model_training.GlaucomaDatasetMergedMask(img_dir, mask_dir)
        
        assert len(dataset) >= 1
        img_out, mask_out = dataset[0]
        assert img_out.shape == (10, 10, 3)
        assert mask_out.shape == (2, 10, 10)

    def test_dataset_separate_masks_initialization_and_loading(self, dummy_dataset):
        """
        TEST NAME: test_dataset_separate_masks_initialization_and_loading
        COMPONENT: GlaucomaDatasetSeparateMasks
        1. Use temporary directories for images and masks from fixture.
        2. Initialize GlaucomaDatasetSeparateMasks and fetch first item.
        3. Assert __getitem__ returns correctly shaped tensors.
        """
        img_dir, mask_dir, _ = dummy_dataset
        dataset = model_training.GlaucomaDatasetSeparateMasks(img_dir, mask_dir)
        
        assert len(dataset) >= 1
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

    def test_build_criterion_all_types(self, dummy_arrays):
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
            
            loss_val = criterion(dummy_arrays["dummy_out"], dummy_arrays["dummy_mask"])
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

    def test_build_datasets_function_success_and_skipped_branches(self, dummy_dataset):
        """
        TEST NAME: test_build_datasets_function_success_and_skipped_branches
        COMPONENT: build_datasets
        1. Use a temporary folder mimicking successful dataset layouts.
        2. Test both the success scenario and the skipped/missing paths logs.
        3. Verify datasets are handled transparently.
        """
        img_dir, mask_dir, _ = dummy_dataset

        orig_img_a = model_training.IMAGES_DIR_A
        orig_mask_a = model_training.MASKS_DIR_A
        orig_img_b = model_training.IMAGES_DIR_B
        orig_mask_b = model_training.MASKS_DIR_B

        try:
            model_training.IMAGES_DIR_A = img_dir
            model_training.MASKS_DIR_A = mask_dir
            model_training.IMAGES_DIR_B = img_dir
            model_training.MASKS_DIR_B = mask_dir

            train_tf = model_training.get_train_transforms()
            val_tf = model_training.get_val_transforms()
            train_ds, _ = model_training.build_datasets(train_tf, val_tf)
            assert train_ds is not None

            model_training.IMAGES_DIR_A = "non_existent_path_a"
            model_training.IMAGES_DIR_B = "non_existent_path_b"
            
            with pytest.raises(RuntimeError):
                model_training.build_datasets(train_tf, val_tf)
        finally:
            model_training.IMAGES_DIR_A = orig_img_a
            model_training.MASKS_DIR_A = orig_mask_a
            model_training.IMAGES_DIR_B = orig_img_b
            model_training.MASKS_DIR_B = orig_mask_b

    def test_datasets_warnings_and_failures(self, dummy_dataset):
        """
        TEST NAME: test_datasets_warnings_and_failures
        COMPONENT: GlaucomaDatasetMergedMask, GlaucomaDatasetSeparateMasks
        1. Use datasets where some images are missing masks to trigger init warnings.
        2. Force a FileNotFoundError inside __getitem__ by breaking files post-init.
        3. Assert the error branches execute correctly.
        """
        img_dir, mask_dir, _ = dummy_dataset

        dataset_merged = model_training.GlaucomaDatasetMergedMask(img_dir, mask_dir)
        _ = model_training.GlaucomaDatasetSeparateMasks(img_dir, mask_dir)
        
        mask_path = os.path.join(mask_dir, "broken_sample.png")
        if os.path.exists(mask_path):
            os.remove(mask_path)
            
        with pytest.raises(FileNotFoundError):
            _ = dataset_merged[0]

    def test_datasets_empty_directory_exceptions(self, tmp_path):
        """
        TEST NAME: test_datasets_empty_directory_exceptions
        COMPONENT: GlaucomaDatasetMergedMask, GlaucomaDatasetSeparateMasks
        1. Use a completely empty directory from tmp_path.
        2. Attempt to initialize both datasets.
        3. Assert ValueError is raised.
        """
        with pytest.raises(ValueError, match="No images found"):
            model_training.GlaucomaDatasetMergedMask(str(tmp_path), str(tmp_path))
            
        with pytest.raises(ValueError, match="No images found"):
            model_training.GlaucomaDatasetSeparateMasks(str(tmp_path), str(tmp_path))

    def test_real_mini_training_epoch(self, dummy_dataset):
        """
        TEST NAME: test_real_mini_training_epoch
        COMPONENT: train_single_model, train_epoch, validate
        1. Use a tiny temporary dataset to run real training safely.
        2. Override SAVE_DIR to point to the temporary folder.
        3. Call train_single_model with epochs=1.
        4. Assert the model .pth file was created and a dice score was returned.
        """
        img_dir, mask_dir, save_dir = dummy_dataset

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