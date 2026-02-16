"""Input readers for alignment files, reference sequences, and mappability maps."""

from __future__ import annotations

import gzip
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Type alias: chromosome -> list of (position, strand) tuples
# strand: True = forward (+), False = reverse (-)
ReadsDict = Dict[str, List[Tuple[int, bool]]]


def read_alignment_file(filepath: str) -> Tuple[ReadsDict, int]:
    """Read an alignment file (BAM or tagAlign) and return reads grouped by chromosome.

    Multi-mapping reads are filtered out.

    Returns:
        (reads_by_chrom, total_reads) where reads_by_chrom maps
        chromosome name to list of (position, is_forward_strand) tuples.
    """
    filepath = str(filepath)
    if filepath.endswith(".bam") or filepath.endswith(".bam.gz"):
        return _read_bam(filepath)
    return _read_tagalign(filepath)


def _read_bam(filepath: str) -> Tuple[ReadsDict, int]:
    """Read a BAM file, filtering multi-mapping and non-primary alignments."""
    import pysam

    reads: ReadsDict = defaultdict(list)
    total = 0

    with pysam.AlignmentFile(filepath, "rb") as bam:
        for read in bam.fetch(until_eof=True):
            # Skip unmapped, secondary, supplementary, and QC-failed reads
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if read.flag & 0x200:  # vendor QC fail
                continue
            # Filter multi-mapping: MAPQ == 0 typically means ambiguous
            if read.mapping_quality == 0:
                continue
            # Also check NH tag if present (number of hits)
            try:
                if read.get_tag("NH") > 1:
                    continue
            except KeyError:
                pass

            chrom = read.reference_name
            pos = read.reference_start  # 0-based
            is_forward = not read.is_reverse
            reads[chrom].append((pos, is_forward))
            total += 1

    logger.info("Read %d uniquely-mapped reads from %s", total, filepath)
    return dict(reads), total


def _read_tagalign(filepath: str) -> Tuple[ReadsDict, int]:
    """Read a tagAlign file (BED-like: chrom start end name score strand).

    tagAlign files are assumed to be pre-filtered for multi-mapping reads.
    """
    reads: ReadsDict = defaultdict(list)
    total = 0

    open_fn = gzip.open if filepath.endswith(".gz") else open
    with open_fn(filepath, "rt") as fh:
        for line in fh:
            line = line.rstrip("\n\r")
            if not line or line.startswith("#") or line.startswith("track"):
                continue
            fields = line.split("\t")
            if len(fields) < 6:
                continue
            chrom = fields[0]
            start = int(fields[1])  # 0-based
            strand = fields[5].strip()
            is_forward = strand == "+"
            reads[chrom].append((start, is_forward))
            total += 1

    logger.info("Read %d reads from %s", total, filepath)
    return dict(reads), total


def get_chromosome_lengths(seq_dir: str) -> Dict[str, int]:
    """Get chromosome lengths from FASTA files in the sequence directory.

    Expects one FASTA file per chromosome named chr*.fa.
    """
    chrom_lengths: Dict[str, int] = {}
    seq_path = Path(seq_dir)

    for fa_file in sorted(seq_path.glob("*.fa")):
        chrom = fa_file.stem  # e.g., "chr1" from "chr1.fa"
        length = 0
        with open(fa_file, "r") as fh:
            for line in fh:
                if not line.startswith(">"):
                    length += len(line.rstrip("\n\r"))
        chrom_lengths[chrom] = length
        logger.debug("Chromosome %s: %d bp", chrom, length)

    logger.info("Found %d chromosomes in %s", len(chrom_lengths), seq_dir)
    return chrom_lengths


def load_mappability(umap_dir: str, chrom: str) -> np.ndarray:
    """Load binary mappability track for a single chromosome.

    Mappability files are uint8 binary files named chr*.uint8.unique.
    Each byte is 0 (not uniquely mappable) or 1 (uniquely mappable).
    """
    umap_path = Path(umap_dir)
    map_file = umap_path / f"{chrom}.uint8.unique"

    if not map_file.exists():
        raise FileNotFoundError(f"Mappability file not found: {map_file}")

    mappability = np.fromfile(str(map_file), dtype=np.uint8)
    logger.debug(
        "Loaded mappability for %s: %d positions, %d mappable (%.1f%%)",
        chrom,
        len(mappability),
        mappability.sum(),
        100.0 * mappability.sum() / max(1, len(mappability)),
    )
    return mappability


def compute_total_mappable_bases(umap_dir: str, chromosomes: List[str]) -> int:
    """Compute total mappable bases across all chromosomes (both strands).

    Returns 2 * sum(mappability) to account for both strands.
    """
    total = 0
    for chrom in chromosomes:
        mmap = load_mappability(umap_dir, chrom)
        total += int(mmap.sum())
    # Both strands contribute, so double the count
    return 2 * total


def validate_inputs(
    seq_dir: str, umap_dir: str, chrom_lengths: Dict[str, int]
) -> List[str]:
    """Validate that sequence and mappability files match.

    Returns the list of valid chromosome names (present in both directories).
    """
    umap_path = Path(umap_dir)
    valid_chroms = []

    for chrom, seq_len in sorted(chrom_lengths.items()):
        map_file = umap_path / f"{chrom}.uint8.unique"
        if not map_file.exists():
            logger.warning(
                "Skipping %s: no mappability file found at %s", chrom, map_file
            )
            continue
        map_len = os.path.getsize(str(map_file))
        if map_len != seq_len:
            logger.warning(
                "Skipping %s: sequence length (%d) != mappability length (%d)",
                chrom,
                seq_len,
                map_len,
            )
            continue
        valid_chroms.append(chrom)

    logger.info(
        "Validated %d chromosomes (of %d)", len(valid_chroms), len(chrom_lengths)
    )
    return valid_chroms
