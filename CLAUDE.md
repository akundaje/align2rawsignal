# CLAUDE.md - AI Assistant Guide for align2rawsignal

## Project Overview

**align2rawsignal** (a.k.a. **WIGGLER**) is a bioinformatics tool that creates genome-wide raw or normalized signal coverage tracks from aligned sequencing reads in BAM or tagAlign format. It was used in the September 2012 ENCODE papers.

- **Author**: Anshul Kundaje
- **License**: MIT
- **Primary language**: MATLAB (compiled to standalone Linux 64-bit executable via MATLAB Compiler)
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

This repository is **documentation-only** in its current state. The actual MATLAB source, compiled binaries, and reference data are not checked in.

```
align2rawsignal/
  README.md        # Comprehensive usage docs, installation, method description
  LICENSE           # MIT License
  CLAUDE.md         # This file
```

### Expected (full) directory layout per README
```
align2rawsignal/
  /src/*.m                                      # MATLAB source files
  /bin/align2rawsignal                          # Compiled executable
  /umap/<version>/globalmap_k<min>tok<max>/     # Mappability maps (chr*.uint8.unique)
  /seq/<version>/                               # Chromosome FASTA files (chr*.fa)
```

## Build System

- **Compiler**: MATLAB Compiler (MCR V7.14 / MATLAB R2010b)
- **No Makefile, CMake, or package manager** is used
- The compiled binary requires the MATLAB Compiler Runtime (MCR) to execute (free download, ~500 MB)

There are no build commands to run within this repository.

## Dependencies

| Dependency | Purpose | Required? |
|---|---|---|
| MATLAB Compiler Runtime (MCR) V7.14 | Execute compiled MATLAB binary | Yes |
| samtools | BAM file processing | Only for BAM input |
| Chromosome FASTA files (`chr*.fa`) | Reference sequences | Yes |
| Mappability maps (`chr*.uint8.unique`) | Uniqueness/mappability tracks | Yes |

## Development Workflow

### No tests or CI
There is no test suite, CI/CD pipeline, linter, or formatter configured in this repository.

### Git conventions
- The repository has a single main branch with 9 commits (2015-2018)
- All historical commits are documentation updates to `README.md`
- Use descriptive commit messages (the existing style is brief: "Updated README", "Update README.md")

### Making changes
Since the repository contains only documentation files:
1. Edit `README.md` for user-facing documentation changes
2. The README is the authoritative reference for installation, usage, parameters, and method description
3. No build or test step is needed after changes

## Key Technical Details (for understanding the codebase)

### Algorithm summary
1. Estimate fragment length per replicate (2 * tagShift)
2. Remove multi-mapping reads
3. Shift read starts in 5'-to-3' direction by L/2 (strand-specific)
4. Count shifted read-start counts at each position
5. Apply kernel-weighted smoothing (default: Tukey kernel, window = 1.5 * L)
6. Compute local cumulative mappability using same kernel procedure
7. Compute expected read counts assuming uniform distribution
8. Normalize: signal(i) = observed(i) / expected(i) (for n=5)
9. Filter positions with low local mappability (default threshold: 25%)

### Important parameters
- `-i` input alignment files (multiple replicates allowed)
- `-s` chromosome sequence directory (mandatory)
- `-u` mappability map directory (mandatory)
- `-l` fragment length per replicate
- `-w` smoothing window size
- `-k` smoothing kernel type
- `-n` normalization method (0-5)
- `-f` mappability filter threshold
- `-of` output format (wig/bg/mat)

### Runtime characteristics
- ~20 min for typical TF ChIP-seq (2-3 replicates)
- ~30-40 min for large DNase-seq datasets
- Minimum 2 GB RAM; more memory improves speed
- Output files range from 500 MB to several GB uncompressed
- bedGraph output is faster and more compact than wiggle

## Conventions for AI Assistants

1. **Do not modify README.md structure** without explicit request -- it is the primary documentation and has a well-established format
2. **Respect the domain context**: this is a genomics signal processing tool; terminology like "tags", "reads", "mappability", "fragment length", and "tag shift" have specific bioinformatics meanings
3. **No source code exists in the repo** -- do not attempt to create MATLAB source files or binaries unless explicitly asked
4. **The tool is archival**: last updated in 2015/2018; changes should preserve backward compatibility with existing documentation references
5. **Output signal caveats**: Wiggler output should not be directly used for cross-experiment comparisons or differential analysis (see README tips)
