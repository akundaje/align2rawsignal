"""Output writers for bedGraph, wiggle, and MAT formats."""

from __future__ import annotations

import logging
import sys
from typing import IO, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def open_output(output_path: Optional[str]) -> IO:
    """Open an output file handle. Returns stdout for None or 'stdout'."""
    if output_path is None or output_path == "stdout":
        return sys.stdout
    if output_path == "stderr":
        return sys.stderr
    return open(output_path, "w")


def write_bedgraph(
    fh: IO,
    chrom: str,
    signal: np.ndarray,
    mask: np.ndarray,
    precision: int = 6,
) -> None:
    """Write signal as bedGraph, merging consecutive positions with equal values.

    Args:
        fh: Output file handle.
        chrom: Chromosome name.
        signal: Signal values array (length = chrom size).
        mask: Boolean array; True = valid position to output.
        precision: Decimal places for signal values.
    """
    n = len(signal)
    if n == 0:
        return

    fmt = f"%.{precision}g"
    i = 0
    while i < n:
        # Skip masked positions
        if not mask[i]:
            i += 1
            continue
        # Start a new span
        start = i
        val = signal[i]
        i += 1
        # Extend while value is the same and position is valid
        while i < n and mask[i] and signal[i] == val:
            i += 1
        fh.write(f"{chrom}\t{start}\t{i}\t{fmt % val}\n")


def write_wiggle(
    fh: IO,
    chrom: str,
    signal: np.ndarray,
    mask: np.ndarray,
    precision: int = 6,
) -> None:
    """Write signal as wiggle (variableStep) format.

    Uses variableStep to efficiently skip unmappable positions.

    Args:
        fh: Output file handle.
        chrom: Chromosome name.
        signal: Signal values array.
        mask: Boolean array; True = valid position.
        precision: Decimal places for signal values.
    """
    n = len(signal)
    if n == 0:
        return

    fmt = f"%.{precision}g"
    fh.write(f"variableStep chrom={chrom}\n")

    valid_indices = np.nonzero(mask)[0]
    for idx in valid_indices:
        # wiggle format uses 1-based coordinates
        fh.write(f"{idx + 1}\t{fmt % signal[idx]}\n")


def write_mat(
    output_path: str,
    results: Dict[str, Dict],
) -> None:
    """Write results as MATLAB .mat file.

    Args:
        output_path: Path for the .mat file.
        results: Dict mapping chromosome name to dict with keys:
            'signal': signal array, 'mask': valid-position mask,
            and optionally 'local_cummap': local cumulative mappability.
    """
    from scipy.io import savemat

    mat_dict = {}
    for chrom, data in results.items():
        safe_name = chrom.replace("-", "_")
        signal = data["signal"].copy()
        signal[~data["mask"]] = np.nan
        mat_dict[safe_name] = signal
        if "local_cummap" in data:
            mat_dict[f"maxTags_{safe_name}"] = data["local_cummap"]

    savemat(output_path, mat_dict, do_compression=True)
    logger.info("Wrote MAT file: %s", output_path)


def write_local_cummap_bedgraph(
    fh: IO,
    chrom: str,
    local_cummap: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Write local cumulative mappability as bedGraph."""
    write_bedgraph(fh, chrom, local_cummap, mask, precision=4)
