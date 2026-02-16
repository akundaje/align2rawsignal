"""Core signal processing pipeline for align2rawsignal.

Implements the WIGGLER algorithm:
1. Shift read starts by fragment length / 2 (strand-specific)
2. Smooth shifted counts with a kernel
3. Compute local cumulative mappability with the same kernel
4. Normalize signal (6 normalization schemes)
5. Filter low-mappability positions
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import fftconvolve

from align2rawsignal.kernels import (
    VALID_KERNELS,
    compute_default_window_size,
    make_kernel,
)
from align2rawsignal.readers import (
    ReadsDict,
    compute_total_mappable_bases,
    get_chromosome_lengths,
    load_mappability,
    read_alignment_file,
    validate_inputs,
)
from align2rawsignal.writers import (
    open_output,
    write_bedgraph,
    write_local_cummap_bedgraph,
    write_mat,
    write_wiggle,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the signal processing pipeline."""

    input_files: List[str]
    seq_dir: str
    umap_dir: str
    output_file: Optional[str] = None
    output_format: str = "bg"
    frag_lengths: List[int] = field(default_factory=lambda: [1])
    window_size: Optional[int] = None
    kernel_name: str = "tukey"
    norm_flag: int = 5
    filter_threshold: float = 0.25
    local_cummap_file: Optional[str] = None
    log_file: Optional[str] = None


def build_shifted_counts(
    reads: List[Tuple[int, bool]],
    chrom_length: int,
    tag_shift: int,
) -> np.ndarray:
    """Create a shifted read-count array for one replicate on one chromosome.

    Each read is shifted by ``tag_shift`` in the 5'-to-3' direction:
    - Forward (+) strand reads: position += tag_shift
    - Reverse (-) strand reads: position -= tag_shift

    Args:
        reads: List of (position, is_forward_strand) tuples.
        chrom_length: Length of the chromosome.
        tag_shift: Number of bases to shift (floor(fragLen / 2)).

    Returns:
        int32 array of read counts at each position (0-based).
    """
    counts = np.zeros(chrom_length, dtype=np.int32)

    for pos, is_forward in reads:
        if is_forward:
            shifted = pos + tag_shift
        else:
            shifted = pos - tag_shift
        if 0 <= shifted < chrom_length:
            counts[shifted] += 1

    return counts


def build_shifted_mappability(
    mappability: np.ndarray,
    tag_shift: int,
) -> np.ndarray:
    """Create strand-shifted mappability for one replicate.

    Combines forward-shifted and reverse-shifted mappability to account
    for reads from both strands contributing to each position after shifting.

    shifted_map[j] = map[j - tag_shift] + map[j + tag_shift]

    Args:
        mappability: Binary uint8 mappability array.
        tag_shift: Shift amount (floor(fragLen / 2)).

    Returns:
        Float32 array of combined shifted mappability.
    """
    n = len(mappability)
    mf = mappability.astype(np.float32)
    shifted = np.zeros(n, dtype=np.float32)

    # Forward strand: reads at position (j - shift) on + strand land at j
    if tag_shift > 0:
        shifted[tag_shift:] += mf[:-tag_shift]
        # Reverse strand: reads at position (j + shift) on - strand land at j
        shifted[:-tag_shift] += mf[tag_shift:]
    else:
        # No shift: both strands contribute at the same position
        shifted = 2.0 * mf

    return shifted


def smooth(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply kernel smoothing via FFT convolution.

    Uses scipy.signal.fftconvolve for O(n log n) performance on
    chromosome-length arrays.

    Args:
        signal: Input signal array.
        kernel: Smoothing kernel (1-D, typically a few hundred elements).

    Returns:
        Smoothed signal array (same length as input, float64).
    """
    # fftconvolve with mode='same' keeps the output the same length as input
    result = fftconvolve(signal.astype(np.float64), kernel, mode="same")
    # Clamp any tiny negative values from floating-point artifacts
    np.maximum(result, 0.0, out=result)
    return result


def normalize_signal(
    smoothed_counts: np.ndarray,
    local_cummap: np.ndarray,
    total_reads: int,
    total_mappable: int,
    window_size: int,
    norm_flag: int,
) -> np.ndarray:
    """Apply normalization to the smoothed signal.

    Normalization schemes:
        0: raw counts (no normalization)
        1: signal(i) * (1e9 / total_reads)
        2: (signal(i) / winsize) * (1e9 / total_reads)
        3: (signal(i) / localCumMap(i)) * (1e9 / total_reads)
        4: (signal(i) / winsize) * (total_mappable / total_reads)
        5: (signal(i) / localCumMap(i)) * (total_mappable / total_reads)

    Args:
        smoothed_counts: Smoothed read counts (summed across replicates).
        local_cummap: Local cumulative mappability (summed across replicates).
        total_reads: Total mapped reads across all replicates.
        total_mappable: Total mappable bases (both strands, all chromosomes).
        window_size: Smoothing window size.
        norm_flag: Normalization method (0-5).

    Returns:
        Normalized signal array (float64).
    """
    signal = smoothed_counts.astype(np.float64)

    if norm_flag == 0:
        return signal

    if total_reads == 0:
        return np.zeros_like(signal)

    if norm_flag == 1:
        return signal * (1e9 / total_reads)

    if norm_flag == 2:
        return (signal / window_size) * (1e9 / total_reads)

    if norm_flag == 3:
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(
                local_cummap > 0,
                (signal / local_cummap) * (1e9 / total_reads),
                0.0,
            )
        return result

    if norm_flag == 4:
        return (signal / window_size) * (total_mappable / total_reads)

    if norm_flag == 5:
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(
                local_cummap > 0,
                (signal / local_cummap) * (total_mappable / total_reads),
                0.0,
            )
        return result

    raise ValueError(f"Invalid normalization flag: {norm_flag}. Must be 0-5.")


def apply_mappability_filter(
    local_cummap: np.ndarray,
    filter_threshold: float,
) -> np.ndarray:
    """Create a boolean mask filtering out low-mappability positions.

    Args:
        local_cummap: Local cumulative mappability array.
        filter_threshold: If <= 1, percentage of max local_cummap.
            If > 1, absolute threshold on local_cummap values.

    Returns:
        Boolean array: True = position passes filter (valid).
    """
    if filter_threshold <= 1.0:
        max_cummap = local_cummap.max() if len(local_cummap) > 0 else 0
        threshold = filter_threshold * max_cummap
    else:
        threshold = filter_threshold

    return local_cummap > threshold


def process_chromosome(
    chrom: str,
    chrom_length: int,
    reads_per_replicate: List[List[Tuple[int, bool]]],
    mappability: np.ndarray,
    kernel: np.ndarray,
    frag_lengths: List[int],
    norm_flag: int,
    filter_threshold: float,
    total_reads_per_replicate: List[int],
    total_mappable: int,
    window_size: int,
) -> Dict:
    """Process a single chromosome through the full pipeline.

    Returns:
        Dict with keys: 'signal', 'mask', 'local_cummap'.
    """
    n_replicates = len(reads_per_replicate)

    # Accumulate smoothed signal and local cumulative mappability across replicates
    total_smoothed = np.zeros(chrom_length, dtype=np.float64)
    total_local_cummap = np.zeros(chrom_length, dtype=np.float64)
    total_reads = 0

    for rep_idx in range(n_replicates):
        frag_len = frag_lengths[rep_idx] if rep_idx < len(frag_lengths) else frag_lengths[0]
        tag_shift = frag_len // 2
        reads = reads_per_replicate[rep_idx]

        # Build shifted read counts
        counts = build_shifted_counts(reads, chrom_length, tag_shift)

        # Build shifted mappability
        shifted_map = build_shifted_mappability(mappability, tag_shift)

        # Smooth both with the kernel
        smoothed_counts = smooth(counts, kernel)
        local_cummap = smooth(shifted_map, kernel)

        total_smoothed += smoothed_counts
        total_local_cummap += local_cummap
        total_reads += total_reads_per_replicate[rep_idx]

    # Normalize
    normalized = normalize_signal(
        total_smoothed,
        total_local_cummap,
        total_reads,
        total_mappable,
        window_size,
        norm_flag,
    )

    # Apply mappability filter
    mask = apply_mappability_filter(total_local_cummap, filter_threshold)

    return {
        "signal": normalized,
        "mask": mask,
        "local_cummap": total_local_cummap,
    }


def run_pipeline(config: PipelineConfig) -> None:
    """Execute the full align2rawsignal pipeline.

    Args:
        config: Pipeline configuration.
    """
    # --- Setup logging ---
    _setup_logging(config.log_file)

    logger.info("align2rawsignal v3.0.0 (Python)")
    logger.info("Input files: %s", config.input_files)
    logger.info("Normalization: %d", config.norm_flag)

    n_replicates = len(config.input_files)

    # --- Resolve fragment lengths ---
    frag_lengths = config.frag_lengths
    if len(frag_lengths) == 1 and n_replicates > 1:
        frag_lengths = frag_lengths * n_replicates
    if len(frag_lengths) != n_replicates:
        raise ValueError(
            f"Number of fragment lengths ({len(frag_lengths)}) must be 1 or "
            f"equal to number of input files ({n_replicates})"
        )
    logger.info("Fragment lengths: %s", frag_lengths)

    # --- Compute window size and build kernel ---
    window_size = config.window_size or compute_default_window_size(frag_lengths)
    logger.info("Smoothing window: %d", window_size)
    logger.info("Smoothing kernel: %s", config.kernel_name)

    kernel = make_kernel(config.kernel_name, window_size, frag_lengths=frag_lengths)

    # --- Read chromosome sequences ---
    logger.info("Reading chromosome lengths from %s", config.seq_dir)
    chrom_lengths = get_chromosome_lengths(config.seq_dir)

    # --- Validate chromosomes ---
    valid_chroms = validate_inputs(config.seq_dir, config.umap_dir, chrom_lengths)
    if not valid_chroms:
        raise RuntimeError("No valid chromosomes found. Check sequence and mappability directories.")

    # --- Read alignment files ---
    all_reads: List[ReadsDict] = []
    total_reads_per_replicate: List[int] = []

    for filepath in config.input_files:
        logger.info("Reading alignment file: %s", filepath)
        reads, total = read_alignment_file(filepath)
        all_reads.append(reads)
        total_reads_per_replicate.append(total)

    logger.info("Total reads per replicate: %s", total_reads_per_replicate)

    # --- Compute total mappable bases ---
    logger.info("Computing total mappable bases...")
    total_mappable = compute_total_mappable_bases(config.umap_dir, valid_chroms)
    logger.info("Total mappable bases (both strands): %d", total_mappable)

    # --- Process chromosomes ---
    output_format = config.output_format.lower()
    mat_results: Dict[str, Dict] = {}

    # Open output for streaming formats
    out_fh = None
    cummap_fh = None
    if output_format in ("bg", "bedgraph"):
        out_fh = open_output(config.output_file)
    elif output_format in ("wig", "wiggle"):
        out_fh = open_output(config.output_file)

    if config.local_cummap_file and output_format != "mat":
        cummap_fh = open_output(config.local_cummap_file)

    try:
        for chrom_idx, chrom in enumerate(valid_chroms):
            chrom_len = chrom_lengths[chrom]
            logger.info(
                "Processing %s (%d/%d, %d bp)",
                chrom,
                chrom_idx + 1,
                len(valid_chroms),
                chrom_len,
            )

            # Load mappability for this chromosome
            mappability = load_mappability(config.umap_dir, chrom)
            if len(mappability) != chrom_len:
                mappability = mappability[:chrom_len]

            # Gather reads per replicate for this chromosome
            reads_per_rep = []
            for rep_reads in all_reads:
                reads_per_rep.append(rep_reads.get(chrom, []))

            # Process
            result = process_chromosome(
                chrom=chrom,
                chrom_length=chrom_len,
                reads_per_replicate=reads_per_rep,
                mappability=mappability,
                kernel=kernel,
                frag_lengths=frag_lengths,
                norm_flag=config.norm_flag,
                filter_threshold=config.filter_threshold,
                total_reads_per_replicate=total_reads_per_replicate,
                total_mappable=total_mappable,
                window_size=window_size,
            )

            # Write output
            if output_format in ("bg", "bedgraph"):
                write_bedgraph(out_fh, chrom, result["signal"], result["mask"])
            elif output_format in ("wig", "wiggle"):
                write_wiggle(out_fh, chrom, result["signal"], result["mask"])
            elif output_format == "mat":
                mat_results[chrom] = result

            # Write local cummap if requested
            if cummap_fh is not None:
                write_local_cummap_bedgraph(
                    cummap_fh, chrom, result["local_cummap"], result["mask"]
                )

            logger.info(
                "  %s: %d valid positions (%.1f%%)",
                chrom,
                result["mask"].sum(),
                100.0 * result["mask"].sum() / max(1, chrom_len),
            )

        # Write MAT output (all chromosomes at once)
        if output_format == "mat":
            if not config.output_file or config.output_file in ("stdout", "stderr"):
                raise ValueError("MAT output requires an output file path (-o=<file>)")
            write_mat(config.output_file, mat_results)

    finally:
        if out_fh is not None and out_fh not in (sys.stdout, sys.stderr):
            out_fh.close()
        if cummap_fh is not None and cummap_fh not in (sys.stdout, sys.stderr):
            cummap_fh.close()

    logger.info("Done.")


def _setup_logging(log_file: Optional[str]) -> None:
    """Configure logging based on user preference."""
    root_logger = logging.getLogger("align2rawsignal")
    root_logger.setLevel(logging.INFO)

    if log_file is None:
        # No logging requested
        root_logger.addHandler(logging.NullHandler())
        return

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    if log_file == "stdout":
        handler = logging.StreamHandler(sys.stdout)
    elif log_file == "stderr":
        handler = logging.StreamHandler(sys.stderr)
    else:
        handler = logging.FileHandler(log_file)

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
