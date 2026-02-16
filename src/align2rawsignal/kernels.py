"""Smoothing kernel functions for signal processing.

All kernels are symmetric, centered arrays whose values represent weights.
Kernels are NOT normalized to sum to 1 -- the raw weighted sums are used
so that local cumulative mappability represents an effective position count.
"""

from __future__ import annotations

from typing import List

import numpy as np


VALID_KERNELS = (
    "rectangular",
    "triangular",
    "epanechnikov",
    "biweight",
    "triweight",
    "cosine",
    "gaussian",
    "tukey",
)


def make_kernel(
    name: str,
    window_size: int,
    frag_lengths: List[int] | None = None,
) -> np.ndarray:
    """Create a 1-D smoothing kernel.

    Args:
        name: Kernel type (one of VALID_KERNELS).
        window_size: Total width of the kernel in base pairs.
        frag_lengths: Fragment lengths (required for tukey taper ratio computation).

    Returns:
        Float64 numpy array of length ``window_size`` (forced odd for symmetry).
    """
    if name not in VALID_KERNELS:
        raise ValueError(
            f"Unknown kernel '{name}'. Valid kernels: {', '.join(VALID_KERNELS)}"
        )

    w = max(1, int(window_size))
    if w % 2 == 0:
        w += 1
    half = w // 2

    # Normalized position array: u in [-1, 1]
    x = np.arange(-half, half + 1, dtype=np.float64)
    u = x / (half + 1)  # stays strictly within (-1, 1)

    if name == "rectangular":
        kernel = np.ones(w, dtype=np.float64)

    elif name == "triangular":
        kernel = 1.0 - np.abs(u)

    elif name == "epanechnikov":
        kernel = np.maximum(0.0, 1.0 - u**2)

    elif name == "biweight":
        kernel = np.maximum(0.0, (1.0 - u**2)) ** 2

    elif name == "triweight":
        kernel = np.maximum(0.0, (1.0 - u**2)) ** 3

    elif name == "cosine":
        kernel = np.where(np.abs(u) < 1.0, np.cos(np.pi * u / 2.0), 0.0)

    elif name == "gaussian":
        sigma = half / 3.0
        kernel = np.exp(-0.5 * (x / sigma) ** 2)

    elif name == "tukey":
        if frag_lengths is None:
            raise ValueError("frag_lengths is required for tukey kernel")
        r = compute_tukey_taper_ratio(w, frag_lengths)
        kernel = _tukey_window(w, r)

    return kernel


def compute_tukey_taper_ratio(window_size: int, frag_lengths: List[int]) -> float:
    """Compute the Tukey kernel taper ratio per the original WIGGLER formula.

    r = max(0.25, min(0.5, max(w - mean(L), 0) / (2 * mean(L))))
    """
    mean_l = float(np.mean(frag_lengths))
    if mean_l <= 0:
        return 0.25
    return max(0.25, min(0.5, max(window_size - mean_l, 0) / (2.0 * mean_l)))


def _tukey_window(size: int, taper_ratio: float) -> np.ndarray:
    """Generate a Tukey (cosine-tapered) window.

    The window has a flat central portion of width (1 - taper_ratio) * size
    and cosine-tapered edges.

    Args:
        size: Window length (should be odd).
        taper_ratio: Fraction of the window that is tapered (0 = rectangular, 1 = Hann).

    Returns:
        Float64 array of shape (size,) with values in [0, 1].
    """
    if size <= 1:
        return np.ones(size, dtype=np.float64)
    if taper_ratio <= 0:
        return np.ones(size, dtype=np.float64)
    if taper_ratio >= 1:
        return np.hanning(size).astype(np.float64)

    kernel = np.ones(size, dtype=np.float64)
    taper_len = int(taper_ratio * (size - 1) / 2.0)

    if taper_len > 0:
        # Left taper
        t = np.arange(taper_len, dtype=np.float64)
        taper = 0.5 * (1.0 + np.cos(np.pi * (t / taper_len - 1.0)))
        kernel[:taper_len] = taper
        # Right taper (mirror)
        kernel[-taper_len:] = taper[::-1]

    return kernel


def compute_default_window_size(frag_lengths: List[int]) -> int:
    """Compute default smoothing window size: mean(1.5 * fragLen), forced odd."""
    w = int(round(np.mean([1.5 * fl for fl in frag_lengths])))
    if w % 2 == 0:
        w += 1
    return max(1, w)
