import numpy as np
import pytest
from tests.base_test import BaseTest


class TestCalculateCdr(BaseTest):
    """
    Tests for GlaucomaPipeline.calculate_cdr().
    Verifies correct CDR computation from boolean disc/cup masks,
    including edge cases: empty disc, empty cup, and fully overlapping masks.
    """

    def test_correct_ratio_with_known_masks(self, pipeline_instance):
        """
        TEST NAME: test_correct_ratio_with_known_masks
        COMPONENT: GlaucomaPipeline.calculate_cdr()
        1. Create a 20x20 disc mask (all True) and a cup mask with 10x10 True region (100 px).
        2. Call calculate_cdr() — expected CDR = sqrt(100/400) = 0.5.
        3. Assert the result equals 0.5 within floating-point tolerance.
        """
        disc = np.ones((20, 20), dtype=bool)
        cup = np.zeros((20, 20), dtype=bool)
        cup[:10, :10] = True

        cdr = pipeline_instance.calculate_cdr(disc, cup)

        assert pytest.approx(cdr, abs=1e-6) == 0.5

    def test_zero_disc_area_returns_zero(self, pipeline_instance):
        """
        TEST NAME: test_zero_disc_area_returns_zero
        COMPONENT: GlaucomaPipeline.calculate_cdr()
        1. Create an empty disc mask (all False) and a full cup mask.
        2. Call calculate_cdr().
        3. Assert the result is 0.0 — no division-by-zero exception raised.
        """
        disc = np.zeros((10, 10), dtype=bool)
        cup = np.ones((10, 10), dtype=bool)

        cdr = pipeline_instance.calculate_cdr(disc, cup)

        assert cdr == 0.0

    def test_full_cup_equal_to_disc_returns_one(self, pipeline_instance):
        """
        TEST NAME: test_full_cup_equal_to_disc_returns_one
        COMPONENT: GlaucomaPipeline.calculate_cdr()
        1. Create identical all-True disc and cup masks (10x10).
        2. Call calculate_cdr() — cup area equals disc area, so CDR = 1.0.
        3. Assert the result equals 1.0 within floating-point tolerance.
        """
        disc = np.ones((10, 10), dtype=bool)
        cup = np.ones((10, 10), dtype=bool)

        cdr = pipeline_instance.calculate_cdr(disc, cup)

        assert pytest.approx(cdr, abs=1e-6) == 1.0

    def test_zero_cup_area_returns_zero(self, pipeline_instance):
        """
        TEST NAME: test_zero_cup_area_returns_zero
        COMPONENT: GlaucomaPipeline.calculate_cdr()
        1. Create a full disc mask (all True) and an empty cup mask (all False).
        2. Call calculate_cdr().
        3. Assert the result is 0.0.
        """
        disc = np.ones((10, 10), dtype=bool)
        cup = np.zeros((10, 10), dtype=bool)

        cdr = pipeline_instance.calculate_cdr(disc, cup)

        assert cdr == 0.0
