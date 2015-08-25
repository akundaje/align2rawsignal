# align2rawsignal
A.K.A. WIGGLER: Creates genome-wide raw or normalized signal tracks from aligned sequencing reads (BAM/tagAlign)

Wiggler can generate genome-wide signal coverage tracks for ChIP-seq, DNase-seq, FAIRE-seq and MNase-seq datasets. These tracks were used in all the Sept 2012 ENCODE papers (http://nature.com/encode)

See or for detailed description of method.

**Author**: Anshul Kundaje

**Email**: akundaje at kundaje dot net

**Latest Update**: August 2015

##Introduction
align2rawsignal (aka. WIGGLER .. because it generates wiggle files) reads in a set of tagAlign/BAM files, filters out multi-mapping tags and creates a consolidated genome-wide signal coverage file using various tag-shift and smoothing parameters as well as various normalization schemes

The method accounts for the following:

* depth of sequencing
* the mappabilty of the genome (based on read length and ambiguous bases)
* differentiates between positions that shown 0 signal simply because they are unmappable vs positions that are mappable by have no reads. The former are not represented in the output wiggle or bedgraph files while the latter are represented as 0s.
* different tag shifts for the different datasets being combined
* Several types of normalization are implemented. (See usage below)

Wiggler does NOT implement input-DNA/control corrections which can be important for some applications and genomes. I intend to add this feature at some point.

This tool is primarily used with the following kinds of functional sequencing data
* TF and histone ChIP-seq
* DNase and FAIRE-seq
* MNase-seq for nucleosome positioning

Signal values generated by Wiggler should NOT be used as-is for the foll. applications directly
* In general, if you are comparing signal values across multiple experiments, you should not use signal values from wiggler directly. This is because even though wiggler normalizes for sequencing depth, mappability etc. it cannot account for basic data quality (signal-to-noise ratio) differences between experiments. You will want to use some type of explicit cross-experiment normalization.
* For differential analysis across experiments (e.g. you have ChIP-seq data for a TF in 2 conditions), it is much better to use statistical methods explicitly designed to capture this. e.g. DE-seq, EdgeR or other differential analysis methods.
If you are using norm=5 (fold-change) as the normalization method withing Wiggler, it is recommended to use an asinh() or log() transform on the values for statistical correlative analyses.

## Compatibility
* Linux 64 bit system with an installation of the free Matlab Compiler Runtime (MCR)
* Supported input formats are tagAlign and BAM files (Single end reads ONLY)
* Supported output formats are bedgraph, wiggle and .mat (matlab)

## Directory structure
```
align2rawsignal/
        /src/                                               --- MATLAB .m source files                  
                *.m
        /bin/                                               --- binary files
                align2rawsignal

        /umap/<version>                                     --- Optional directory containing uniqueness maps
                globalmap_k<mink>tok<maxk>/chr*.unique

        /seq/<version>                                      --- Optional directory containing chromosome sequences
                chr*.fa
```

## Required Files
1. The align2rawsignal executable
2. A directory containing all chromosome sequences in fasta format e.g. align2rawsignal/seq/hg18Female. There should be one file per chromosome in this directory chr*.fa. You can download them from
  * hg18: http://hgdownload.cse.ucsc.edu/goldenPath/hg18/bigZips/chromFa.zip
  * encodeHg19Female: http://hgdownload-test.cse.ucsc.edu/goldenPath/hg19/encodeDCC/referenceSequences/femaleByChrom/ (*.gz files)
  * encodeHg19Male: http://hgdownload-test.cse.ucsc.edu/goldenPath/hg19/encodeDCC/referenceSequences/maleByChrom/ (*.gz files)
  * **NOTE:** Remove the _random*.fa and other contig files
  * **NOTE:** If you are working with a female genome, delete or move the chrY.fa file out of the directory pointed to by --seq-dir
3. A directory containing the global uniqueness/mappability map for a range of kmer lengths e.g. umap/encodeHg19Female/globalmap_k20tok54/
  * There should be one file per chromosome in this directory chr*.uint8.unique.
  * NOTE: The mappability file name prefix for each chromosome MUST correspond to the prefix of the sequence file for chromosome e.g. chr1.fa <=> chr1.uint8.unique
  * The mappability tracks can be downloaded from http://www.broadinstitute.org/~anshul/projects/umap/
  * You can unzip these using tar -xvzf globalmap_k`*`tok`*`.tgz
  * The mappability tracks currently support reads upto 54 bp for most organisms. I will be upgrading these to support upto 300 bp reads by Dec 2013.

## Installation Instructions
**NOTE:** These are installation/running instructions for 64-bit LINUX distributions. If you need executables for other platforms please contact Anshul (anshul_at_kundaje_dot_net)

### 1. MCR Installation
In order to run the align2rawsignal code and/or any MATLAB compiled code, you will need the MATLAB runtime library. Please only use the MCR version referenced in this README. This version of the executable was compiled using MCR V7.14 which is equivalent to R2010b release. You can download the MCR here http://www.broadinstitute.org/~anshul/softwareRepo/MCR2010b.bin

If you haven't installed the MCR, you MUST do that using this command

`./MCR2010b.bin -console`

If you need to specify a specific temp directory then also use the option `-is:tempdir <tempdirname>`

The installer will prompt you to select the directory (<MCR_ROOT>) you want to install the MCR into. e.g. `/home/akundaje/software/mcroot`

**NOTE:** Make sure your installation directory has write permissions and has atleast 500 MB of disk space.

The installation should go smoothly with the above command. However, if you are interested in other installation options you can consult http://www.mathworks.com/access/helpdesk/help/toolbox/compiler/bru23df-1.html

**NOTE:** You need to install the MCR ONLY once on the machine/cluster you plan to run MATLAB compiled code.

If you want to uninstall the MCR , follow this procedure:

1. Navigate to your MCR installation directory using the cd command.
2. `cd` into the `_uninst` directory
3. Run the `uninstaller.bin` program.
`./uninstaller.bin -console`

### 2. Setting paths
You need to set the following environment variables for the compiled MATLAB code to run correctly. These environment variables MUST be set before calling the align2rawsignal executable or any other MATLAB compiled code.

You can add the following lines to your `.bashrc` or `.cshrc` file if you want to avoid settings these variables everytime you want to run the code

If you are using the bash shell or modifying .bashrc then use

```
MCRROOT=<MCR_ROOT>/v714
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MCRROOT}/runtime/glnxa64
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MCRROOT}/bin/glnxa64
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MCRROOT}/sys/os/glnxa64
MCRJRE=${MCRROOT}/sys/java/jre/glnxa64/jre/lib/amd64
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MCRJRE}/native_threads
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MCRJRE}/server
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MCRJRE}
XAPPLRESDIR=${MCRROOT}/X11/app-defaults
export LD_LIBRARY_PATH
export XAPPLRESDIR
```

If you are using the csh shell or modifying .cshrc then use
```
setenv MCRROOT <MCR_ROOT>/v714
setenv LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:${MCRROOT}/runtime/glnxa64
setenv LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:${MCRROOT}/bin/glnxa64
setenv LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:${MCRROOT}/sys/os/glnxa64 
setenv LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:${MCRROOT}/sys/java/jre/glnxa64/jre/lib/amd64/native_threads
setenv LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:${MCRROOT}/sys/java/jre/glnxa64/jre/lib/amd64/server
setenv LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:${MCRROOT}/sys/java/jre/glnxa64/jre/lib/amd64
setenv XAPPLRESDIR ${MCRROOT}/X11/app-defaults
```

### 3. Samtools
If you are working with BAM input files then the samTools executable is required to be in your system PATH. You can get samtools from here http://samtools.sourceforge.net/

You will need to add samtools to your path so that align2rawsignal can call samtools.

`export PATH="<directory_containing_samtools_binary>:${PATH}"`

If you are running align2rawsignal on a cluster make sure the cluster nodes have this directory in their $PATH. You can set it in your submit script.

### 4. Now you can run the align2rawsignal executable
For options/help simply type

`./align2rawsignal`

OR

`./align2rawsignal --help`

OR
```
export $PATH=<directory_containing_align2rawsignal>/bin:$PATH # add the /bin directory to your path`
align2rawsignal # call align2rawsignal
```

### 5. Running instructions on a cluster

  * Make sure the shell you use in your submit script is bash
  `#!/bin/bash`
  * Make sure all the MCR environment variables have been explicity set in your submit script (See above.)
  * You can prevent bottlenecking by also defining the following environment variable
    * `export MCR_CACHE_ROOT=$TMPDIR   # If $TMPDIR is defined` OR
    * `export MCR_CACHE_ROOT=<ValidTmpDir> # where <ValidTmpDir> can be for example /tmp or some other temporary directory on the local cluster node`
  * Your submit script must have the path to samtools in your $PATH variable if you are reading in BAM files
  * I also recommend that you set the `$TMP` environment variable to a valid temporary directory with sufficient disk space (incase it isn't already set).
`export TMP=<ValidTmpDir>`

## SOME TIPS
1. You should have sufficient disk space. Each signal file can range from 500 MB to several gigabytes (uncompressed)
2. You will need atleast 2GB of memory. The more the memory the faster the code runs. You can set the memory usage as a parameter to the program.
3. Most of the running time is spent outputing the signal bedGraph or wiggle files.
4. **BedGraph output is typically faster and more condensed than wiggle files and easier and faster to convert to bigwigs.**
5. A run takes about 20 mins for 2 or 3 tagAlign/BAM replicate files corresponding to a typical transcription factor ChIP-seq experiment
6. A run takes about 30-40 mins for 3 very large tagAlign/BAM replicate files corresponding to DNAse-seq experiments.
7. If you run the code with the `--output-max-tags` option, it takes longer. This is because it has to output another massive bedgraph file.
8. Longer tag-extension lengths run faster than shorter ones. This is because longer <extLen> produce longer contiguous chunks of identical signal values and hence produce smaller bedGraph files.
9. The code filters out illegal reads (bad start/stops). It also checks that the chromosome sequence lengths and mappability track lengths are the same.
10. If you are outputing the signal file to stdout, then you will not see output immediately since the code has to read in the data and filter the reads and then start calculating the output signal. Use the `--v=<logfile>` option to see what the code is upto.
11. The left to right order of the tagAlign file names and the extension length parameters MUST match ie. the leftmost extension length parameter is assumed to match the leftmost tagAlign file name
12. If you are reading in BAM files, samtools is required.
13. **Chromosome sequence file name prefixes, chromosome mappability file name prefixes and chromosome names in the tagAlign/BAM files MUST match and contain the 'chr' prefix e.g. chr1.fa, chr1.uint8.unique and chr1 in the tagAlign/BAM files. Remove any additional .fa files from the sequence directory that do not have matching .unique files in the mappability directory**
14. Some valid and invalid usage scenarios for `--v` option.
  * If you dont want any logging then you should not use the `--v` option at all. By default there is no logging.
    
    `VALID: --v=stdout , --v=stderr , --v=<File>`
    
    `INVALID: --v , --v=`
15. Some valid and invalid usage scenarios for the -o option
  * By default, the wiggle/bedgraph output goes to stdout. For this you dont need to specify the `-o` option.
  * You can pipe the output to gzip or to some file.
    
    `VALID: -o=stdout , -o=stderr , -o=<file>`
     
    `INVALID: -o , -o=`
  * For .mat output you MUST specify an output file i.e. `-o=<file>`
  
## Usage/Parameters
```
Below are the usage instructions with various options and parameters

--------------------------------------------------------------
Program: align2rawsignal (Converts tagAlign/BAM files into normalized signal)
Version: 2.0
Contact: Anshul Kundaje (akundaje@stanford.edu)
--------------------------------------------------------------

USAGE: align2rawsignal -i=<alignFname> -s=<seqDir> <OPTIONAL ARGUMENTS>

--help/-h print usage information and quit

----------------
INPUT OPTIONS:
----------------

-i=<alignFname> (MANDATORY, MULTIPLE ALLOWED)
One or more tagAlign/BAM files (replicates) as input.
The tagAlign files can have extensions (gz,tagAlign). BAM files MUST have extension (bam,bam.gz)

-s=<seqDir> (MANDATORY)
Full path to directory containing chromosome fasta files (eg. chr1.fa ..)
The file names MUST match the chromosome names used in the tagAlign files
e.g: /seq/hg19

-u=<uMapDir> (MANDATORY)
Full path to directory containing binary mappability tracks.
The directory name must be of the form [PATH]/globalmap_k<min>tok<max>
e.g: /umap/hg19/globalmap_k20tok54

----------------
OUTPUT OPTIONS:
----------------

-o=<oFname> (OPTIONAL)
Full path and name of output signal file.
Set to stdout if you want to print to stdout which is also the default behavior
Default: stdout

-of=<outputFormat> (OPTIONAL)
Output signal file format
wiggle (wig) or bedGraph (bg) or matfile (mat)
Default: mat

-m=<localCumMapFile> (OPTIONAL)
Calculate, for each position 'i' in the genome, the maximum number of uniquely mappable
surrounding positions that contribute to the signal value at position 'i'.
This is a function of the mappability of the surrounding positions, 
tag extension/smoothing length and number of replicates.
The local cumulative mappability is output in <maxTagsFile>.
If -of=mat then, <localCumMapFile> can have the same name as <oFname>.
In this case, the local cummap output is stored as a separate set of variables with prefix maxTags
in the .mat file.
Default: local cumMap is not output to a file

-v=<logFile> (OPTIONAL)
verbose mode.
<logFile> Full path and name of file for logging.
Set to stdout/stderr if you want to output logging info to stdout/stderr 
Default: off

-n=<normFLag> (OPTIONAL)
a flag indicating whether the signal output should be normalized
<normalization_flag> = 0,1,2,3,4,5
Default: 5 (fold change wrt. expected value from a uniform distribution of reads)
If you want raw counts use 0. The other flags (1..4) are experimental.
0: no normalization
1: normSignal(i) = signal(i) * (1e9 / #reads)
2: normSignal(i) = (signal(i)/winsize) * (1e9 / #reads)
3: normSignal(i) = (signal(i)/localCumMap(i)) * (1e9 / #reads)
4: normSignal(i) = (signal(i)/winsize) * (#total_mappable_bases / #reads)
5: normSignal(i) = (signal(i)/localCumMap(i)) * (#total_mappable_bases / #reads)

----------------
PARAMETERS:
----------------

-l=<fragLen> (OPTIONAL, MULTIPLE ALLOWED)
Fragment-length == 2*Tag-shift
Default: 1 (no extension)
Tags are shifted by floor(fragLen/2) in a 3' direction, relative to the strand of the tag
NOTE: If a single fragLen is specified then it is applied to all tagAlign/BAM files.
NOTE: Multiple arguments of this type are allowed. In such a case, 
      number of fragLen arguments MUST BE == no. of Align files.
      The ORDER of these arguments is important. 
      e.g. The first <fragLen> is matched with the first tagAlign/BAM file and so on.

-w=<smoothingWindow> (OPTIONAL)
Smoothing window size for signal
Default: mean(1.5*fragLen)

-k=<smoothingKernel> (OPTIONAL)
Smoothing kernel to use 
Valid kernels (rectangular,triangular,epanechnikov,biweight,triweight,cosine,gaussian,tukey)
Default: tukey (with taper ratio of max( 0.25 , min (0.5,max(w-mean(l),0)/(2*mean(l))) )

-f=<localCumMapFilter> (OPTIONAL)
Will nullify positions that have localCumMap <= <mappability_threshold>
<mappability_threshold> <= 1 implies threshold is in terms of percentage localCumMap
          e.g. 0.1 means atleast 10 percent of the positions in the smoothing window MUST be mappable 
               > 1 implies threshold is on actual maxtags values
          e.g. 30 means atleast 30 positions (per replicate) in the extension/smoothing window 
               MUST be mappable 
Default: 0.25

-mm=<memory> (OPTIONAL)
Total memory to use in GB
Default: 2
--------------------------------------------------------------------------------------------------
```

## Brief Description of method
The method can combine and jointly normalize data from multiple datasets/replicates of the same experiment

1. You must estimate the predominant fragment length (2\*tagShift) for each replicate separately. We tend to use strand cross-correlation analysis to estimate the tag shift. You set -l=2\*tagShift or fragment-length
  * For ChIP-seq/MNase-seq l=fragment length
  * For DNase-seq, l=1 (no tag shift)
2. All multimapping reads are removed from each replicate BAM/tagAlign file

For each replicate

1. Read starts are shifted in the 5' to 3' direction by L/2 (strand specific)
2. We count shifted read-start counts at each position 'i' (adding both strands)
3. For each position 'i', we computed a weighted sum of read counts in a window of a particular length 'w' centered at 'i'. We use a kernel to compute a weighted sum. By default w is set to 1.5\*L. The default tukey kernel has a central window which has weights of 1 and then it tapers to 0 following a cosine curve.The length of the central window (constant '1' weights) is approximately set to the fragment length 'L' and the total width of the kernel 'w' should be generally set to the fragment size based on size selection or the estimated fragment length. Basically, this procedure is equivalent to a smooth tag extension of 'w'.
4. You need to provide a mappability track that marks each position 'i' as 1 (uniquely mappable) and 0 (not uniquely mappable). This track is dependent on read length. We compute a local cumulative mappability for each position 'i' using the same strand shift and tukey kernel window procedure as for read counts. This gives us the effective number of positions in the local kernel window that can potentially contribute read counts based on mappability. Lets call this m(i)
5. We compute the expected cumulative read counts at each position 'i' if reads were all mapped reads were uniformly distributed along the genome at mappable locations as
   
   ```
    = m(i) X t / g
    
           where t is the total number of mapped reads in that replicate
           
           where g is the total mappable genome size over both strands
   ```
6. We add cumulative read counts from all replicates at each position 'i' = R(i)
7. We add expected cumulative read counts from all replicates at each position 'i' = E(i)
8. Normalized signal (default n=5) at position 'i' = R(i) / E(i)
9. Positions that have E(i) < 25% max(E(i)) are filtered out as missing/unreliable data points due to very low local mappability and hence unreliable signal. These positions 'i' are not present in the bedgraphs. All other positions are listed with a signal value. Hence a position 'i' with signal value 0 refers to a reliable mappable location but with no reads mapping to it (compared to missing/unreliable data).

Effectively, the signal is normalized for

1. differences in total number of mapped reads
2. different fragment lengths/smoothing
3. number of replicates
4. differences in local mappability
5. Missing values due to unreliable/low mappability are differentiated from true 0s.
