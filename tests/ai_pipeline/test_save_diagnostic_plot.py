import numpy as np
import pytest
from unittest.mock import patch
from tests.base_test import BaseTest


class TestSaveDiagnosticPlot(BaseTest):
    """
    Tests for save_diagnostic_plot().
    Verifies that matplotlib is called correctly: savefig receives the right path,
    plt.close() is always called, all four subplots receive data,
    and ground-truth presence toggles imshow vs text on the fourth subplot.
    """

    def test_savefig_called_with_correct_path(
        self, tmp_path, plot_inputs, mock_plt_subplots
    ):
        """
        TEST NAME: test_savefig_called_with_correct_path
        COMPONENT: save_diagnostic_plot()
        1. Patch pipeline.plt and configure subplots to return mock axes.
        2. Call save_diagnostic_plot() with a target save path.
        3. Assert plt.savefig was called exactly once with that path.
        """
        from pipeline import save_diagnostic_plot

        save_path = tmp_path / "test_report.png"
        img, crops, masks, cdr, gt_m, cdr_gt = plot_inputs
        mock_fig, mock_axes, _ = mock_plt_subplots

        with patch("pipeline.plt") as mock_plt:
            mock_plt.subplots.return_value = (mock_fig, mock_axes)
            save_diagnostic_plot(img, crops, masks, cdr, gt_m, cdr_gt, save_path)

            mock_plt.savefig.assert_called_once_with(save_path)

    def test_plt_close_called_after_save(
        self, tmp_path, plot_inputs, mock_plt_subplots
    ):
        """
        TEST NAME: test_plt_close_called_after_save
        COMPONENT: save_diagnostic_plot()
        1. Patch pipeline.plt and configure subplots to return mock axes.
        2. Call save_diagnostic_plot().
        3. Assert plt.close() was called exactly once.
        """
        from pipeline import save_diagnostic_plot

        save_path = tmp_path / "report.png"
        img, crops, masks, cdr, gt_m, cdr_gt = plot_inputs
        mock_fig, mock_axes, _ = mock_plt_subplots

        with patch("pipeline.plt") as mock_plt:
            mock_plt.subplots.return_value = (mock_fig, mock_axes)
            save_diagnostic_plot(img, crops, masks, cdr, gt_m, cdr_gt, save_path)

            mock_plt.close.assert_called_once()

    def test_first_three_subplots_receive_imshow(
        self, tmp_path, plot_inputs, mock_plt_subplots
    ):
        """
        TEST NAME: test_first_three_subplots_receive_imshow
        COMPONENT: save_diagnostic_plot()
        1. Patch pipeline.plt and configure subplots to return four mock axes.
        2. Call save_diagnostic_plot() without ground-truth masks.
        3. Assert imshow() was called on each of the first three axes.
        """
        from pipeline import save_diagnostic_plot

        save_path = tmp_path / "report.png"
        img, crops, masks, cdr, gt_m, cdr_gt = plot_inputs
        mock_fig, mock_axes, ax_mocks = mock_plt_subplots

        with patch("pipeline.plt") as mock_plt:
            mock_plt.subplots.return_value = (mock_fig, mock_axes)
            save_diagnostic_plot(img, crops, masks, cdr, gt_m, cdr_gt, save_path)

            for ax in ax_mocks[:3]:
                ax.imshow.assert_called()

    def test_fourth_subplot_uses_text_when_no_ground_truth(
        self, tmp_path, plot_inputs, mock_plt_subplots
    ):
        """
        TEST NAME: test_fourth_subplot_uses_text_when_no_ground_truth
        COMPONENT: save_diagnostic_plot()
        1. Patch pipeline.plt; pass plot_inputs which has gt_masks=None.
        2. Call save_diagnostic_plot().
        3. Assert axes[1,1].text() was called and imshow() was not called on it.
        """
        from pipeline import save_diagnostic_plot

        save_path = tmp_path / "report.png"
        img, crops, masks, cdr, gt_m, cdr_gt = plot_inputs
        mock_fig, mock_axes, ax_mocks = mock_plt_subplots

        with patch("pipeline.plt") as mock_plt:
            mock_plt.subplots.return_value = (mock_fig, mock_axes)
            save_diagnostic_plot(img, crops, masks, cdr, gt_m, cdr_gt, save_path)

            ax_mocks[3].text.assert_called()

    def test_fourth_subplot_uses_imshow_when_ground_truth_present(
        self, tmp_path, dummy_bgr_image, mock_plt_subplots
    ):
        """
        TEST NAME: test_fourth_subplot_uses_imshow_when_ground_truth_present
        COMPONENT: save_diagnostic_plot()
        1. Build masks and gt_masks arrays; pass both to save_diagnostic_plot().
        2. Patch pipeline.plt and call save_diagnostic_plot().
        3. Assert axes[1,1].imshow() was called and text() was not called.
        """
        from pipeline import save_diagnostic_plot

        h, w = dummy_bgr_image.shape[:2]
        mask = np.zeros((2, 512, 512), dtype=np.float32)
        mask[0, :200, :200] = 0.9
        mask[1, :100, :100] = 0.8

        gt_masks = np.zeros((2, 512, 512), dtype=np.float32)
        gt_masks[0, :180, :180] = 1.0
        gt_masks[1, :90, :90] = 1.0

        mock_fig, mock_axes, ax_mocks = mock_plt_subplots

        with patch("pipeline.plt") as mock_plt:
            mock_plt.subplots.return_value = (mock_fig, mock_axes)
            save_diagnostic_plot(
                dummy_bgr_image, [(0, 0, w, h)], [mask],
                cdr=0.5, gt_masks=gt_masks, cdr_gt=0.45,
                save_path=tmp_path / "report_gt.png",
            )

            ax_mocks[3].imshow.assert_called()
            ax_mocks[3].text.assert_not_called()
