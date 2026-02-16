# CLAUDE.md - AI Assistant Guide for align2rawsignal

## Project Overview

**align2rawsignal** (a.k.a. **WIGGLER**) is a bioinformatics tool that creates genome-wide raw or normalized signal coverage tracks from aligned sequencing reads in BAM or tagAlign format. Originally a MATLAB compiled binary used in the September 2012 ENCODE papers, this is now a pure **Python** implementation.

- **Author**: Anshul Kundaje
- **License**: MIT
- **Primary language**: Python (>= 3.9)
- **Domain**: Computational genomics / epigenomics signal processing

### Supported assay types
- ChIP-seq (transcription factor and histone)
- DNase-seq and FAIRE-seq
- MNase-seq (nucleosome positioning)

### Key capabilities
- Filters multi-mapping reads
- Applies configurable tag-shift and smoothing (8 kernel types)
- Computes local cumulative mappability to distinguish unmappable positions from true zeros
- Implements 6 normalization schemes (0-5); default is fold-change (n=5)
- Outputs bedGraph, wiggle, or MATLAB .mat formats

## Repository Structure

```
align2rawsignal/
  pyproject.toml                  # Package metadata, dependencies, build config
  README.md                       # User-facing documentation (original)
  LICENSE                         # MIT License
  CLAUDE.md                       # This file
  src/
    align2rawsignal/
      __init__.py                 # Package init, version
      __main__.py                 # python -m align2rawsignal entry point
      cli.py                      # Argument parsing and CLI entry point
      pipeline.py                 # Main signal processing pipeline orchestration
      kernels.py                  # Smoothing kernel functions (8 types)
      readers.py                  # I/O: BAM, tagAlign, FASTA, mappability readers
      writers.py                  # I/O: bedGraph, wiggle, MAT output writers
  tests/
    __init__.py
    test_kernels.py               # Kernel function tests
    test_pipeline.py              # Pipeline, signal processing, and writer tests
```

### Reference data layout (not in repo, user-provided)
```
/umap/<version>/globalmap_k<min>tok<max>/   # Mappability maps (chr*.uint8.unique)
/seq/<version>/                             # Chromosome FASTA files (chr*.fa)
```

## Build & Install

```bash
# Install in editable mode (development)
pip install -e ".[dev]"

# Install for production
pip install .
```

After installation, the `align2rawsignal` command is available on the PATH.

## Dependencies

| Dependency | Purpose |
|---|---|
| numpy >= 1.21 | Array operations, vectorized computation |
| scipy >= 1.7 | FFT convolution (`fftconvolve`), `.mat` output (`savemat`) |
| pysam >= 0.19 | BAM file reading |
| pytest >= 7.0 | Testing (dev only) |

Reference data (not Python packages):
- Chromosome FASTA files (`chr*.fa`)
- Binary mappability maps (`chr*.uint8.unique`)

## Development Workflow

### Running tests
```bash
pytest                    # Run all tests
pytest tests/ -v          # Verbose output
pytest tests/ -x          # Stop on first failure
```

### Code organization
- **`readers.py`** -- All input I/O (BAM via pysam, tagAlign text parsing, FASTA length reading, binary mappability loading)
- **`kernels.py`** -- 8 kernel types (rectangular, triangular, epanechnikov, biweight, triweight, cosine, gaussian, tukey) plus default window size computation
- **`pipeline.py`** -- Core algorithm: `build_shifted_counts()`, `build_shifted_mappability()`, `smooth()` (FFT convolution), `normalize_signal()`, `apply_mappability_filter()`, `process_chromosome()`, `run_pipeline()`
- **`writers.py`** -- Output in bedGraph (with span merging), wiggle (variableStep), and MATLAB .mat formats
- **`cli.py`** -- Argument parsing matching original tool's CLI flags

### Architecture decisions
- **Chromosome-by-chromosome processing**: Only one chromosome's data is in memory at a time, keeping RAM usage bounded
- **FFT convolution**: `scipy.signal.fftconvolve` for O(n log n) smoothing on chromosome-length arrays
- **Streaming output**: bedGraph and wiggle formats are written chromosome-by-chromosome (no full-genome buffer)
- **MAT output**: Accumulated in memory (necessary for single-file output via `scipy.io.savemat`)

## Key Technical Details

### Algorithm summary
1. Estimate fragment length per replicate (2 * tagShift)
2. Remove multi-mapping reads (MAPQ=0 / NH>1 for BAM; tagAlign assumed pre-filtered)
3. Shift read starts in 5'-to-3' direction by floor(L/2) (strand-specific)
4. Count shifted read-start counts at each position
5. Build shifted mappability: `shifted_map[j] = map[j - shift] + map[j + shift]`
6. Apply kernel-weighted smoothing to both counts and shifted mappability (default: Tukey kernel, window = 1.5 * L)
7. Compute expected read counts: `expected(i) = localCumMap(i) * totalReads / totalMappable`
8. Normalize: `signal(i) = observed(i) / expected(i)` (for n=5, fold-change)
9. Filter positions with low local mappability (default threshold: 25% of max)

### CLI parameters (compatible with original tool)
- `-i` input alignment files (multiple replicates allowed)
- `-s` chromosome sequence directory (mandatory)
- `-u` mappability map directory (mandatory)
- `-o` output file (default: stdout)
- `-of` output format: `bg` (default), `wig`, or `mat`
- `-l` fragment length per replicate (default: 1)
- `-w` smoothing window size (default: mean(1.5 * fragLen))
- `-k` smoothing kernel type (default: tukey)
- `-n` normalization method 0-5 (default: 5)
- `-f` mappability filter threshold (default: 0.25)
- `-m` output local cumulative mappability file
- `-v` log file (stdout/stderr/path; default: off)

### Normalization schemes
- `0`: No normalization (raw smoothed counts)
- `1`: `signal(i) * (1e9 / totalReads)`
- `2`: `(signal(i) / winsize) * (1e9 / totalReads)`
- `3`: `(signal(i) / localCumMap(i)) * (1e9 / totalReads)`
- `4`: `(signal(i) / winsize) * (totalMappable / totalReads)`
- `5`: `(signal(i) / localCumMap(i)) * (totalMappable / totalReads)` (fold-change, default)

## Conventions for AI Assistants

1. **Run tests after changes**: Always run `pytest` to verify no regressions
2. **Respect the domain context**: terminology like "tags", "reads", "mappability", "fragment length", and "tag shift" have specific bioinformatics meanings
3. **Efficiency matters**: This processes chromosome-length arrays (100M-250M positions). Use numpy vectorized operations and avoid Python loops over positions
4. **Memory awareness**: Process one chromosome at a time. Avoid holding multiple chromosome-length arrays simultaneously
5. **CLI compatibility**: The CLI flags (`-i`, `-s`, `-u`, `-l`, `-w`, `-k`, `-n`, `-f`, `-of`, `-o`, `-v`, `-m`) match the original MATLAB tool for backward compatibility
6. **Output signal caveats**: Wiggler output should not be directly used for cross-experiment comparisons or differential analysis (see README)
