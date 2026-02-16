"""Tests for the core signal processing pipeline."""

import io
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from align2rawsignal.pipeline import (
    PipelineConfig,
    apply_mappability_filter,
    build_shifted_counts,
    build_shifted_mappability,
    normalize_signal,
    process_chromosome,
    smooth,
)
from align2rawsignal.kernels import make_kernel
from align2rawsignal.writers import write_bedgraph, write_wiggle


class TestBuildShiftedCounts:
    """Tests for build_shifted_counts()."""

    def test_no_shift(self):
        reads = [(10, True), (20, False), (30, True)]
        counts = build_shifted_counts(reads, 50, tag_shift=0)
        assert counts[10] == 1
        assert counts[20] == 1
        assert counts[30] == 1
        assert counts.sum() == 3

    def test_forward_shift(self):
        reads = [(10, True)]  # forward strand, shift right
        counts = build_shifted_counts(reads, 50, tag_shift=5)
        assert counts[15] == 1
        assert counts[10] == 0
        assert counts.sum() == 1

    def test_reverse_shift(self):
        reads = [(20, False)]  # reverse strand, shift left
        counts = build_shifted_counts(reads, 50, tag_shift=5)
        assert counts[15] == 1
        assert counts[20] == 0
        assert counts.sum() == 1

    def test_out_of_bounds_filtered(self):
        reads = [(2, False)]  # shift left by 5 -> -3, out of bounds
        counts = build_shifted_counts(reads, 50, tag_shift=5)
        assert counts.sum() == 0

    def test_stacking(self):
        reads = [(10, True), (10, True), (10, True)]
        counts = build_shifted_counts(reads, 50, tag_shift=0)
        assert counts[10] == 3

    def test_empty_reads(self):
        counts = build_shifted_counts([], 100, tag_shift=5)
        assert counts.sum() == 0
        assert len(counts) == 100


class TestBuildShiftedMappability:
    """Tests for build_shifted_mappability()."""

    def test_no_shift(self):
        mappability = np.array([1, 1, 0, 1, 1], dtype=np.uint8)
        result = build_shifted_mappability(mappability, tag_shift=0)
        # No shift -> each position counted twice (both strands)
        np.testing.assert_allclose(result, [2, 2, 0, 2, 2])

    def test_with_shift(self):
        mappability = np.array([0, 0, 1, 0, 0], dtype=np.uint8)
        result = build_shifted_mappability(mappability, tag_shift=1)
        # Forward shift: map[j-1] for j=1..4 -> [0,0,0,1,0]
        # Reverse shift: map[j+1] for j=0..3 -> [0,1,0,0,0]
        # Combined: [0, 1, 0, 1, 0]
        np.testing.assert_allclose(result, [0, 1, 0, 1, 0])

    def test_all_mappable(self):
        mappability = np.ones(10, dtype=np.uint8)
        result = build_shifted_mappability(mappability, tag_shift=2)
        # Edges lose coverage due to shift
        assert result[0] == 1.0  # only reverse strand contributes
        assert result[-1] == 1.0  # only forward strand contributes
        assert result[3] == 2.0  # both strands contribute


class TestSmooth:
    """Tests for smooth()."""

    def test_identity_with_delta_kernel(self):
        signal = np.array([0, 0, 0, 5, 0, 0, 0], dtype=np.float64)
        kernel = np.array([1.0])
        result = smooth(signal, kernel)
        np.testing.assert_allclose(result, signal, atol=1e-10)

    def test_rectangular_kernel_is_sum(self):
        signal = np.zeros(20, dtype=np.float64)
        signal[10] = 1.0
        kernel = np.ones(5, dtype=np.float64)
        result = smooth(signal, kernel)
        # The impulse response should be 5 positions of value 1.0
        assert result[10] == pytest.approx(1.0, abs=1e-10)
        assert result.sum() == pytest.approx(5.0, abs=1e-10)

    def test_nonnegative_output(self):
        rng = np.random.default_rng(42)
        signal = rng.integers(0, 10, size=100).astype(np.float64)
        kernel = make_kernel("triangular", 11)
        result = smooth(signal, kernel)
        assert np.all(result >= 0)


class TestNormalizeSignal:
    """Tests for normalize_signal()."""

    def test_norm_0_raw(self):
        signal = np.array([1.0, 2.0, 3.0])
        cummap = np.array([10.0, 10.0, 10.0])
        result = normalize_signal(signal, cummap, 1000, 1000000, 100, norm_flag=0)
        np.testing.assert_allclose(result, signal)

    def test_norm_1(self):
        signal = np.array([1.0, 2.0, 3.0])
        cummap = np.array([10.0, 10.0, 10.0])
        result = normalize_signal(signal, cummap, 1000, 1000000, 100, norm_flag=1)
        expected = signal * (1e9 / 1000)
        np.testing.assert_allclose(result, expected)

    def test_norm_5_fold_change(self):
        signal = np.array([10.0, 20.0, 30.0])
        cummap = np.array([100.0, 100.0, 100.0])
        total_reads = 1000
        total_mappable = 10000
        result = normalize_signal(signal, cummap, total_reads, total_mappable, 100, norm_flag=5)
        # norm5: (signal / cummap) * (total_mappable / total_reads)
        expected = (signal / cummap) * (total_mappable / total_reads)
        np.testing.assert_allclose(result, expected)

    def test_norm_5_zero_cummap(self):
        signal = np.array([10.0, 0.0])
        cummap = np.array([100.0, 0.0])
        result = normalize_signal(signal, cummap, 1000, 10000, 100, norm_flag=5)
        assert result[1] == 0.0  # division by zero -> 0

    def test_invalid_norm_flag(self):
        with pytest.raises(ValueError, match="Invalid normalization"):
            normalize_signal(np.array([1.0]), np.array([1.0]), 1, 1, 1, norm_flag=6)


class TestApplyMappabilityFilter:
    """Tests for apply_mappability_filter()."""

    def test_percentage_threshold(self):
        cummap = np.array([100.0, 50.0, 25.0, 10.0, 0.0])
        mask = apply_mappability_filter(cummap, 0.25)
        # threshold = 0.25 * 100 = 25; must be > 25
        np.testing.assert_array_equal(mask, [True, True, False, False, False])

    def test_absolute_threshold(self):
        cummap = np.array([100.0, 50.0, 25.0, 10.0, 0.0])
        mask = apply_mappability_filter(cummap, 30.0)
        np.testing.assert_array_equal(mask, [True, True, False, False, False])


class TestProcessChromosome:
    """Integration test for process_chromosome()."""

    def test_basic_pipeline(self):
        chrom_len = 100
        mappability = np.ones(chrom_len, dtype=np.uint8)
        reads = [(50, True)]  # single read at position 50, forward strand
        kernel = make_kernel("rectangular", 5)

        result = process_chromosome(
            chrom="chr1",
            chrom_length=chrom_len,
            reads_per_replicate=[reads],
            mappability=mappability,
            kernel=kernel,
            frag_lengths=[1],
            norm_flag=0,
            filter_threshold=0.25,
            total_reads_per_replicate=[1],
            total_mappable=200,
            window_size=5,
        )

        assert "signal" in result
        assert "mask" in result
        assert "local_cummap" in result
        assert len(result["signal"]) == chrom_len
        assert result["signal"].sum() > 0
        assert result["mask"].any()

    def test_unmappable_regions_filtered(self):
        chrom_len = 50
        mappability = np.zeros(chrom_len, dtype=np.uint8)
        mappability[20:30] = 1  # only positions 20-29 are mappable
        reads = [(25, True)]
        kernel = make_kernel("rectangular", 3)

        result = process_chromosome(
            chrom="chr1",
            chrom_length=chrom_len,
            reads_per_replicate=[reads],
            mappability=mappability,
            kernel=kernel,
            frag_lengths=[1],
            norm_flag=0,
            filter_threshold=0.25,
            total_reads_per_replicate=[1],
            total_mappable=20,
            window_size=3,
        )

        # Positions outside the mappable region should be masked
        assert not result["mask"][0]
        assert not result["mask"][49]


class TestWriters:
    """Tests for output writers."""

    def test_bedgraph_merges_equal_values(self):
        signal = np.array([1.0, 1.0, 1.0, 2.0, 2.0])
        mask = np.array([True, True, True, True, True])
        buf = io.StringIO()
        write_bedgraph(buf, "chr1", signal, mask)
        lines = buf.getvalue().strip().split("\n")
        assert len(lines) == 2
        assert lines[0].startswith("chr1\t0\t3\t")
        assert lines[1].startswith("chr1\t3\t5\t")

    def test_bedgraph_skips_masked(self):
        signal = np.array([1.0, 1.0, 1.0])
        mask = np.array([True, False, True])
        buf = io.StringIO()
        write_bedgraph(buf, "chr1", signal, mask)
        lines = buf.getvalue().strip().split("\n")
        assert len(lines) == 2  # two separate spans

    def test_wiggle_format(self):
        signal = np.array([1.5, 0.0, 2.5])
        mask = np.array([True, True, True])
        buf = io.StringIO()
        write_wiggle(buf, "chr1", signal, mask)
        output = buf.getvalue()
        assert "variableStep chrom=chr1" in output
        # wiggle uses 1-based coordinates
        assert "1\t" in output
        assert "3\t" in output

    def test_wiggle_skips_masked(self):
        signal = np.array([1.5, 0.0, 2.5])
        mask = np.array([True, False, True])
        buf = io.StringIO()
        write_wiggle(buf, "chr1", signal, mask)
        lines = buf.getvalue().strip().split("\n")
        # header + 2 data lines (position 2 is masked)
        assert len(lines) == 3


class TestEndToEnd:
    """End-to-end test with on-disk files."""

    def test_tagalign_to_bedgraph(self):
        """Create a minimal tagAlign + reference and run through the pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a tiny chromosome FASTA
            seq_dir = tmpdir / "seq"
            seq_dir.mkdir()
            chrom_seq = "N" * 200
            (seq_dir / "chr1.fa").write_text(f">chr1\n{chrom_seq}\n")

            # Create mappability file (all mappable)
            umap_dir = tmpdir / "umap"
            umap_dir.mkdir()
            mappability = np.ones(200, dtype=np.uint8)
            mappability.tofile(str(umap_dir / "chr1.uint8.unique"))

            # Create a simple tagAlign file
            tagalign = tmpdir / "reads.tagAlign"
            with open(tagalign, "w") as f:
                for pos in range(80, 120):
                    f.write(f"chr1\t{pos}\t{pos + 36}\tread_{pos}\t1000\t+\n")
                for pos in range(90, 130):
                    f.write(f"chr1\t{pos}\t{pos + 36}\tread_{pos}\t1000\t-\n")

            # Run pipeline
            output_file = tmpdir / "output.bedgraph"
            config = PipelineConfig(
                input_files=[str(tagalign)],
                seq_dir=str(seq_dir),
                umap_dir=str(umap_dir),
                output_file=str(output_file),
                output_format="bg",
                frag_lengths=[100],
                norm_flag=5,
                filter_threshold=0.25,
                log_file="stderr",
            )

            from align2rawsignal.pipeline import run_pipeline

            run_pipeline(config)

            # Check output exists and has content
            assert output_file.exists()
            content = output_file.read_text()
            lines = [l for l in content.strip().split("\n") if l]
            assert len(lines) > 0
            # All lines should be valid bedGraph
            for line in lines:
                parts = line.split("\t")
                assert len(parts) == 4
                assert parts[0] == "chr1"
                assert int(parts[1]) >= 0
                assert int(parts[2]) > int(parts[1])
                assert float(parts[3]) >= 0
