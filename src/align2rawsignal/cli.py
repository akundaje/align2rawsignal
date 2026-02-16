"""Command-line interface for align2rawsignal.

Provides argument parsing compatible with the original MATLAB tool's interface.
"""

from __future__ import annotations

import argparse
import sys

from align2rawsignal.kernels import VALID_KERNELS
from align2rawsignal.pipeline import PipelineConfig, run_pipeline


DESCRIPTION = """\
--------------------------------------------------------------
Program: align2rawsignal (Converts tagAlign/BAM files into normalized signal)
Version: 3.0.0 (Python)
Original author: Anshul Kundaje
--------------------------------------------------------------

Creates genome-wide raw or normalized signal tracks from aligned
sequencing reads (BAM/tagAlign).
"""

EPILOG = """\
Examples:
  # Basic usage with bedGraph output
  align2rawsignal -i reads.bam -s /seq/hg38 -u /umap/hg38/globalmap_k36tok54 -of bg

  # Multiple replicates with different fragment lengths
  align2rawsignal -i rep1.bam -i rep2.bam -s /seq/hg38 -u /umap/hg38/globalmap_k36tok54 \\
      -l 200 -l 180 -of bg -o signal.bedgraph

  # Raw counts (no normalization)
  align2rawsignal -i reads.tagAlign.gz -s /seq/hg19 -u /umap/hg19/globalmap_k20tok54 \\
      -n 0 -of bg
"""


def parse_args(argv: list[str] | None = None) -> PipelineConfig:
    """Parse command-line arguments into a PipelineConfig."""
    parser = argparse.ArgumentParser(
        prog="align2rawsignal",
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Input options ---
    parser.add_argument(
        "-i",
        action="append",
        required=True,
        metavar="ALIGNFILE",
        dest="input_files",
        help="Input tagAlign/BAM file (may be specified multiple times for replicates)",
    )
    parser.add_argument(
        "-s",
        required=True,
        metavar="SEQDIR",
        dest="seq_dir",
        help="Directory containing chromosome FASTA files (chr*.fa)",
    )
    parser.add_argument(
        "-u",
        required=True,
        metavar="UMAPDIR",
        dest="umap_dir",
        help="Directory containing binary mappability tracks (chr*.uint8.unique)",
    )

    # --- Output options ---
    parser.add_argument(
        "-o",
        metavar="OUTFILE",
        dest="output_file",
        default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "-of",
        metavar="FORMAT",
        dest="output_format",
        default="bg",
        choices=["wig", "bg", "mat", "wiggle", "bedgraph", "matfile"],
        help="Output format: wig, bg, or mat (default: bg)",
    )
    parser.add_argument(
        "-m",
        metavar="CUMMAPFILE",
        dest="local_cummap_file",
        default=None,
        help="Output local cumulative mappability to this file",
    )
    parser.add_argument(
        "-v",
        metavar="LOGFILE",
        dest="log_file",
        default=None,
        help="Log file (stdout, stderr, or file path; default: off)",
    )

    # --- Normalization ---
    parser.add_argument(
        "-n",
        type=int,
        metavar="NORMFLAG",
        dest="norm_flag",
        default=5,
        choices=range(6),
        help=(
            "Normalization method 0-5 (default: 5 = fold-change). "
            "0=none, 1-4=experimental, 5=fold-change"
        ),
    )

    # --- Parameters ---
    parser.add_argument(
        "-l",
        action="append",
        type=int,
        metavar="FRAGLEN",
        dest="frag_lengths",
        help=(
            "Fragment length (2 * tag-shift). May be specified multiple times "
            "to match each input file. Default: 1 (no extension)"
        ),
    )
    parser.add_argument(
        "-w",
        type=int,
        metavar="WINSIZE",
        dest="window_size",
        default=None,
        help="Smoothing window size (default: mean(1.5 * fragLen))",
    )
    parser.add_argument(
        "-k",
        metavar="KERNEL",
        dest="kernel_name",
        default="tukey",
        choices=list(VALID_KERNELS),
        help="Smoothing kernel (default: tukey)",
    )
    parser.add_argument(
        "-f",
        type=float,
        metavar="THRESHOLD",
        dest="filter_threshold",
        default=0.25,
        help=(
            "Mappability filter threshold. <=1: percentage of max localCumMap; "
            ">1: absolute count (default: 0.25)"
        ),
    )

    args = parser.parse_args(argv)

    # Default fragment length
    if args.frag_lengths is None:
        args.frag_lengths = [1]

    # Normalize output format aliases
    fmt_map = {"wiggle": "wig", "bedgraph": "bg", "matfile": "mat"}
    args.output_format = fmt_map.get(args.output_format, args.output_format)

    return PipelineConfig(
        input_files=args.input_files,
        seq_dir=args.seq_dir,
        umap_dir=args.umap_dir,
        output_file=args.output_file,
        output_format=args.output_format,
        frag_lengths=args.frag_lengths,
        window_size=args.window_size,
        kernel_name=args.kernel_name,
        norm_flag=args.norm_flag,
        filter_threshold=args.filter_threshold,
        local_cummap_file=args.local_cummap_file,
        log_file=args.log_file,
    )


def main(argv: list[str] | None = None) -> None:
    """Entry point for the align2rawsignal CLI."""
    config = parse_args(argv)
    try:
        run_pipeline(config)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
