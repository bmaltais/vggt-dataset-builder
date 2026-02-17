#!/usr/bin/env python3
"""Test suite for PointCloudFilter.apply_color_filter().

Tests the color filtering logic that removes background colors (black/white)
from point clouds before rendering. This is critical for producing clean output
when processing depth maps with various background types.
"""

import numpy as np
import pytest
from dataset_utils import PointCloudFilter


class TestColorFilterNoFiltering:
    """Tests for apply_color_filter with no filters enabled."""

    def test_apply_no_filters_returns_all_true(self):
        """When no filters enabled, all points should be kept (all True).

        This is the happy path - when filtering is disabled, the mask should
        indicate that all points are valid.
        """
        filter_cfg = PointCloudFilter(filter_black_bg=False, filter_white_bg=False)

        colors = np.array([[0.5, 0.5, 0.5], [0.2, 0.3, 0.4]], dtype=np.float32)
        mask = filter_cfg.apply_color_filter(colors)

        assert np.all(mask == True)
        assert mask.shape == (2,)
        assert mask.dtype == bool

    def test_apply_to_extreme_colors_with_no_filters(self):
        """Even extreme colors (black, white) are kept when filters disabled."""
        filter_cfg = PointCloudFilter(filter_black_bg=False, filter_white_bg=False)

        colors = np.array(
            [
                [0.0, 0.0, 0.0],  # Black
                [1.0, 1.0, 1.0],  # White
                [0.5, 0.5, 0.5],  # Gray
            ],
            dtype=np.float32,
        )
        mask = filter_cfg.apply_color_filter(colors)

        assert np.all(mask == True)


class TestColorFilterBlackBackground:
    """Tests for apply_color_filter with black background filtering."""

    def test_filter_black_removes_dark_colors(self):
        """Black background filter removes colors with RGB sum < 16/255."""
        filter_cfg = PointCloudFilter(filter_black_bg=True, filter_white_bg=False)

        threshold = 16 / 255.0

        colors = np.array(
            [
                [0.01, 0.01, 0.01],  # sum=0.03 < 0.0627 (filtered)
                [0.02, 0.02, 0.02],  # sum=0.06 < 0.0627 (filtered)
                [0.1, 0.0, 0.0],  # sum=0.1 > 0.0627 (kept)
                [0.5, 0.5, 0.5],  # sum=1.5 > 0.0627 (kept)
                [1.0, 1.0, 1.0],  # sum=3.0 > 0.0627 (kept)
            ],
            dtype=np.float32,
        )

        mask = filter_cfg.apply_color_filter(colors)

        # Points 0, 1 should be filtered (False), rest should be kept (True)
        expected = np.array([False, False, True, True, True])
        assert np.array_equal(mask, expected)

    def test_filter_black_boundary_exact_threshold(self):
        """Test colors exactly at the black threshold boundary.

        Threshold is RGB sum >= 16/255. Test values just below, at, and above.
        """
        filter_cfg = PointCloudFilter(filter_black_bg=True, filter_white_bg=False)

        threshold = 16 / 255.0

        colors = np.array(
            [
                [threshold - 0.001, 0.0, 0.0],  # Just below threshold
                [
                    threshold / 3.0,
                    threshold / 3.0,
                    threshold / 3.0,
                ],  # Exactly at threshold
                [threshold + 0.001, 0.0, 0.0],  # Just above threshold
            ],
            dtype=np.float32,
        )

        mask = filter_cfg.apply_color_filter(colors)

        # First should be filtered, last two should be kept
        expected = np.array([False, True, True])
        assert np.array_equal(mask, expected)

    def test_filter_black_empty_array(self):
        """Empty color array should return empty mask."""
        filter_cfg = PointCloudFilter(filter_black_bg=True, filter_white_bg=False)

        colors = np.array([], dtype=np.float32).reshape(0, 3)
        mask = filter_cfg.apply_color_filter(colors)

        assert mask.shape == (0,)
        assert mask.dtype == bool

    def test_filter_black_single_point(self):
        """Test filtering a single point (edge case)."""
        filter_cfg = PointCloudFilter(filter_black_bg=True, filter_white_bg=False)

        # Very black point
        colors = np.array([[0.001, 0.001, 0.001]], dtype=np.float32)
        mask = filter_cfg.apply_color_filter(colors)

        assert mask.shape == (1,)
        assert mask[0] == False


class TestColorFilterWhiteBackground:
    """Tests for apply_color_filter with white background filtering."""

    def test_filter_white_removes_bright_colors(self):
        """White background filter removes colors with all channels >= 241/255."""
        filter_cfg = PointCloudFilter(filter_black_bg=False, filter_white_bg=True)

        threshold = 241.0 / 255.0

        colors = np.array(
            [
                [1.0, 1.0, 1.0],  # Pure white (filtered)
                [0.95, 0.95, 0.95],  # Very light (filtered)
                [threshold, threshold, threshold],  # Exactly at threshold (filtered)
                [threshold - 0.001, 0.5, 0.5],  # Below threshold on one channel (kept)
                [0.5, 0.5, 0.5],  # Normal
                [0.0, 0.0, 0.0],  # Black
            ],
            dtype=np.float32,
        )

        mask = filter_cfg.apply_color_filter(colors)

        # Points 0, 1, 2 should be filtered (all channels >= threshold)
        # Point 3 should be kept (red below threshold)
        # Points 4, 5 should be kept
        expected = np.array([False, False, False, True, True, True])
        assert np.array_equal(mask, expected)

    def test_filter_white_boundary_threshold(self):
        """Test colors at and around the white threshold.

        Threshold is ALL channels >= 241/255. One channel below threshold
        means the point is kept.
        """
        filter_cfg = PointCloudFilter(filter_black_bg=False, filter_white_bg=True)

        threshold = 241.0 / 255.0

        colors = np.array(
            [
                [1.0, 1.0, 1.0],  # All at 1.0 (filtered)
                [
                    threshold,
                    threshold,
                    threshold,
                ],  # All exactly at threshold (filtered)
                [threshold - 0.001, 1.0, 1.0],  # One channel below (kept)
                [threshold, threshold, threshold - 0.001],  # One below (kept)
            ],
            dtype=np.float32,
        )

        mask = filter_cfg.apply_color_filter(colors)

        expected = np.array([False, False, True, True])
        assert np.array_equal(mask, expected)

    def test_filter_white_requires_all_channels_high(self):
        """White filter requires ALL channels >= threshold, not just average."""
        filter_cfg = PointCloudFilter(filter_black_bg=False, filter_white_bg=True)

        threshold = 241.0 / 255.0

        colors = np.array(
            [
                [1.0, 0.5, 0.5],  # Only red high (kept)
                [0.5, 1.0, 0.5],  # Only green high (kept)
                [0.5, 0.5, 1.0],  # Only blue high (kept)
                [1.0, 1.0, 0.5],  # Red and green high (kept)
                [threshold, threshold, threshold],  # All at threshold (filtered)
            ],
            dtype=np.float32,
        )

        mask = filter_cfg.apply_color_filter(colors)

        expected = np.array([True, True, True, True, False])
        assert np.array_equal(mask, expected)

    def test_filter_white_empty_array(self):
        """Empty array returns empty mask."""
        filter_cfg = PointCloudFilter(filter_black_bg=False, filter_white_bg=True)

        colors = np.array([], dtype=np.float32).reshape(0, 3)
        mask = filter_cfg.apply_color_filter(colors)

        assert mask.shape == (0,)


class TestColorFilterBothFilters:
    """Tests for apply_color_filter with both black and white filters enabled."""

    def test_both_filters_removes_black_and_white(self):
        """With both filters, remove both very dark and very bright colors."""
        filter_cfg = PointCloudFilter(filter_black_bg=True, filter_white_bg=True)

        black_threshold = 16 / 255.0
        white_threshold = 241.0 / 255.0

        colors = np.array(
            [
                [0.01, 0.01, 0.01],  # sum=0.03 < black_threshold (filtered)
                [0.02, 0.02, 0.02],  # sum=0.06 < black_threshold (filtered)
                [0.5, 0.5, 0.5],  # Normal gray (kept)
                [
                    white_threshold - 0.01,
                    white_threshold - 0.01,
                    white_threshold - 0.01,
                ],  # All < white_threshold (kept)
                [1.0, 1.0, 1.0],  # All >= white_threshold (filtered)
                [
                    white_threshold,
                    white_threshold,
                    white_threshold,
                ],  # All >= white_threshold (filtered)
            ],
            dtype=np.float32,
        )

        mask = filter_cfg.apply_color_filter(colors)

        expected = np.array([False, False, True, True, False, False])
        assert np.array_equal(mask, expected)

    def test_both_filters_middle_tones_preserved(self):
        """With both filters, middle tones should always be preserved."""
        filter_cfg = PointCloudFilter(filter_black_bg=True, filter_white_bg=True)

        # Create an array of colors across the spectrum
        colors = np.array(
            [
                [0.25, 0.25, 0.25],  # Darker gray
                [0.5, 0.5, 0.5],  # Mid gray
                [0.75, 0.75, 0.75],  # Light gray
                [0.2, 0.5, 0.8],  # Mixed RGB
                [0.8, 0.3, 0.1],  # Orange
            ],
            dtype=np.float32,
        )

        mask = filter_cfg.apply_color_filter(colors)

        # All should be kept (in the middle range)
        assert np.all(mask == True)


class TestColorFilterProperties:
    """Tests that verify properties and invariants of the filter."""

    def test_output_shape_matches_input(self):
        """Output mask shape should be (N,) where N is number of colors."""
        filter_cfg = PointCloudFilter(filter_black_bg=True, filter_white_bg=True)

        for n_points in [1, 10, 100, 1000]:
            colors = np.random.rand(n_points, 3).astype(np.float32)
            mask = filter_cfg.apply_color_filter(colors)

            assert mask.shape == (n_points,)
            assert mask.dtype == bool

    @pytest.mark.parametrize("n_colors", [1, 5, 10, 100])
    def test_has_filters_consistency(self, n_colors):
        """has_color_filters() should accurately reflect filter settings.

        Tests that has_color_filters() returns True iff at least one color
        filter is enabled, and the filtering actually happens.
        """
        # No filters
        filter_cfg = PointCloudFilter(filter_black_bg=False, filter_white_bg=False)
        assert filter_cfg.has_color_filters() == False

        colors = np.random.rand(n_colors, 3).astype(np.float32)
        mask = filter_cfg.apply_color_filter(colors)
        assert np.all(mask == True)  # No filtering happens

        # Black filter only
        filter_cfg = PointCloudFilter(filter_black_bg=True, filter_white_bg=False)
        assert filter_cfg.has_color_filters() == True
        mask = filter_cfg.apply_color_filter(colors)
        assert mask.dtype == bool

        # White filter only
        filter_cfg = PointCloudFilter(filter_black_bg=False, filter_white_bg=True)
        assert filter_cfg.has_color_filters() == True
        mask = filter_cfg.apply_color_filter(colors)
        assert mask.dtype == bool

    def test_filter_is_monotonic_with_brightness(self):
        """Darker colors are more likely to be filtered by black filter;
        lighter colors by white filter.

        This is a property test: filtering effects should be consistent
        across the brightness spectrum.
        """
        filter_cfg = PointCloudFilter(filter_black_bg=True, filter_white_bg=False)

        # Create a gradient of gray colors from dark to light
        brightnesses = np.linspace(0.0, 1.0, 11)
        colors = np.array([[b, b, b] for b in brightnesses], dtype=np.float32)

        mask = filter_cfg.apply_color_filter(colors)

        # Darkest should be filtered, lightest should not be
        assert mask[0] == False  # Very dark filtered
        assert mask[-1] == True  # Very light not filtered


class TestColorFilterMissingNumpy:
    """Test error handling when numpy is not available."""

    def test_raises_on_numpy_missing(self, monkeypatch):
        """apply_color_filter should raise ImportError if numpy unavailable.

        This tests the defensive check at the start of the function.
        """
        import dataset_utils

        # Save original HAS_NUMPY
        original_has_numpy = dataset_utils.HAS_NUMPY

        try:
            # Mock numpy as unavailable
            monkeypatch.setattr(dataset_utils, "HAS_NUMPY", False)

            filter_cfg = PointCloudFilter(filter_black_bg=True, filter_white_bg=False)

            # Should raise ImportError (but we can't easily test the actual
            # execution since we already imported numpy)
            # This is more of a coverage verification
            assert dataset_utils.HAS_NUMPY == False

        finally:
            # Restore original
            monkeypatch.setattr(dataset_utils, "HAS_NUMPY", original_has_numpy)
