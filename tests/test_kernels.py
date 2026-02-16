"""Tests for smoothing kernel functions."""

import numpy as np
import pytest

from align2rawsignal.kernels import (
    VALID_KERNELS,
    compute_default_window_size,
    compute_tukey_taper_ratio,
    make_kernel,
)


class TestMakeKernel:
    """Tests for make_kernel()."""

    @pytest.mark.parametrize("name", [k for k in VALID_KERNELS if k != "tukey"])
    def test_kernel_shape_is_odd(self, name):
        kernel = make_kernel(name, 100)
        assert len(kernel) % 2 == 1

    @pytest.mark.parametrize("name", [k for k in VALID_KERNELS if k != "tukey"])
    def test_kernel_is_symmetric(self, name):
        kernel = make_kernel(name, 51)
        np.testing.assert_allclose(kernel, kernel[::-1], atol=1e-12)

    @pytest.mark.parametrize("name", [k for k in VALID_KERNELS if k != "tukey"])
    def test_kernel_nonnegative(self, name):
        kernel = make_kernel(name, 51)
        assert np.all(kernel >= 0)

    @pytest.mark.parametrize("name", [k for k in VALID_KERNELS if k != "tukey"])
    def test_kernel_peak_at_center(self, name):
        kernel = make_kernel(name, 51)
        center = len(kernel) // 2
        assert kernel[center] == kernel.max()

    def test_rectangular_all_ones(self):
        kernel = make_kernel("rectangular", 11)
        np.testing.assert_allclose(kernel, np.ones(11))

    def test_triangular_center_is_one(self):
        kernel = make_kernel("triangular", 11)
        assert kernel[5] == pytest.approx(1.0)

    def test_tukey_requires_frag_lengths(self):
        with pytest.raises(ValueError, match="frag_lengths"):
            make_kernel("tukey", 51)

    def test_tukey_basic(self):
        kernel = make_kernel("tukey", 151, frag_lengths=[100])
        assert len(kernel) == 151
        assert kernel[75] == pytest.approx(1.0)  # center
        assert np.all(kernel >= 0)
        assert np.all(kernel <= 1.0 + 1e-10)
        np.testing.assert_allclose(kernel, kernel[::-1], atol=1e-12)

    def test_unknown_kernel_raises(self):
        with pytest.raises(ValueError, match="Unknown kernel"):
            make_kernel("invalid_name", 51)

    def test_even_window_becomes_odd(self):
        kernel = make_kernel("rectangular", 10)
        assert len(kernel) % 2 == 1
        assert len(kernel) == 11

    def test_window_size_1(self):
        kernel = make_kernel("rectangular", 1)
        assert len(kernel) == 1
        assert kernel[0] == 1.0


class TestTukeyTaperRatio:
    """Tests for compute_tukey_taper_ratio()."""

    def test_formula(self):
        # r = max(0.25, min(0.5, max(w - mean(L), 0) / (2 * mean(L))))
        # w=200, L=[100] -> max(0.25, min(0.5, (200-100)/(200))) = max(0.25, min(0.5, 0.5))
        r = compute_tukey_taper_ratio(200, [100])
        assert r == pytest.approx(0.5)

    def test_floor_at_025(self):
        # w=110, L=[100] -> max(0.25, min(0.5, (110-100)/(200))) = max(0.25, 0.05) = 0.25
        r = compute_tukey_taper_ratio(110, [100])
        assert r == pytest.approx(0.25)

    def test_cap_at_05(self):
        # w=500, L=[100] -> max(0.25, min(0.5, (500-100)/200)) = max(0.25, min(0.5, 2.0)) = 0.5
        r = compute_tukey_taper_ratio(500, [100])
        assert r == pytest.approx(0.5)

    def test_multiple_frag_lengths(self):
        # mean([100, 200]) = 150
        # max(0.25, min(0.5, (300-150)/300)) = max(0.25, 0.5) = 0.5
        r = compute_tukey_taper_ratio(300, [100, 200])
        assert r == pytest.approx(0.5)


class TestComputeDefaultWindowSize:
    """Tests for compute_default_window_size()."""

    def test_single_frag_length(self):
        w = compute_default_window_size([200])
        # 1.5 * 200 = 300, forced odd -> 301
        assert w == 301

    def test_no_shift(self):
        w = compute_default_window_size([1])
        # 1.5 * 1 = 1.5 -> round to 2 -> odd -> 3
        assert w == 3

    def test_multiple_frag_lengths(self):
        w = compute_default_window_size([200, 300])
        # mean(1.5 * [200, 300]) = mean([300, 450]) = 375, forced odd -> 375
        assert w == 375

    def test_always_odd(self):
        w = compute_default_window_size([100])
        assert w % 2 == 1
