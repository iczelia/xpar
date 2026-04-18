# xpar

[![Build](https://github.com/iczelia/xpar/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/iczelia/xpar/actions/workflows/build.yml)

xpar - an error/erasure code system guarding data integrity.
Licensed under the terms of GNU GPL version 3 or later - see COPYING.
Report issues to Kamila Szewczyk <k@iczelia.net>.
Project homepage: https://github.com/iczelia/xpar

xpar in joint mode generates a slightly inflated (by about 12%) parity-guarded
file from a given data file. Such a file can be recovered as long as no more
than about 6.2% of the data is corrupted. xpar internally uses a (255,223)-RS
code over an 8-bit Galois field.

can be consiered a sibling project of par2.

[![Packaging status](https://repology.org/badge/vertical-allrepos/xpar.svg)](https://repology.org/project/xpar/versions)

## Building

```
# If using a git clone (not needed for source packages), first...
$ ./bootstrap

# Optimised x86-64
$ ./configure --enable-x86_64 --enable-native --enable-lto && make && sudo make install

# Optimised aarch64
$ ./configure --enable-aarch64 --enable-native --enable-lto && make && sudo make install

# Generic x86-64
$ ./configure --enable-x86_64 && make && sudo make install

# Generic aarch64
$ ./configure --enable-aarch64 && make && sudo make install

# Generic unknown architecture
$ ./configure && make && sudo make install
```

NOTE TO LINUX DISTRIBUTION MAINTAINERS: Simple builds with no configure flags will likely max out at a couple hundred megabytes of encoding and decoding and miss out on most assembly-level optimisations to the tool. They are provided for compatibility with simple systems (Amiga, MS-DOS, etc). To get suitable performance as indicated with the benchmarks, build with OpenMP and follow the instructions above.

### MS-DOS (DJGPP)

xpar builds for MS-DOS via the DJGPP cross-toolchain, producing a 32-bit
protected-mode binary that runs on Windows 9x DOS box, FreeDOS, DOSEMU2,
DOSBox-X, or a real DOS machine with any DPMI host (CWSDPMI is bundled
in the stubified `xpar.exe`). Configure with `--host=i586-pc-msdosdjgpp`.
Limitations: single-threaded, no SIMD dispatch (i386 baseline), maximum
file size 2 GiB (DJGPP's 32-bit `off_t`; FAT32's 4 GiB ceiling makes
the remaining 2-4 GiB window narrow enough to punt on). Long-filename
paths work on hosts with an LFN driver (Win9x, DOSEMU2, DOSBox-X,
FreeDOS + DOSLFN); pure DOS is stuck with 8.3.

## Usage

Consult the man page.

## Development 

A rough outline of some development-related topics below.

## Repository management

As it stands:
- `contrib/` - holds scripts and other non-source files that are not present
  in the distribution tarball and not supported.
- `NEWS` - will contain the release notes for each release and needs to be
  modified before each release.
- `ChangeLog` - generated via `make update-ChangeLog`; intended to be
  re-generated before each release.

Code style:
- Two space indent, brace on the same line, middle pointers - `char * p;`.

## Benchmarks

`enwik10` = first 10 GB of the English Wikipedia XML dump, a common benchmark corpus for compression and erasure coding tools. Each tool runs 3 times under `hyperfine --runs 3 --warmup 0`.

- CPU: AMD Ryzen 9 5950X 16-core / 32-thread (Zen 3)
- Kernel: Linux 6.8.0-101-generic x86-64
- Binaries: all C/C++ tools built with `-march=x86-64-v3 -O3`, statically linked.

xpar tested via:
- `xpar -Jefq -i 1 enwik10` (joint mode, interlacing 1)
- `xpar -Jefq -i 2 enwik10` (joint mode, interlacing 2)
- `xpar -Jefq -i 3 enwik10` (joint mode, interlacing 3)
- `xpar -Jsefq enwik10` (joint systematic mode, parity-only sidecar)
- `xpar -Lefq --dshards=10 --pshards=1 --out-prefix=enwik10.xpa enwik10` (sharded mode, Leopard FFT)
- `xpar -Wefq --dshards=10 --pshards=1 --out-prefix=enwik10.xpa enwik10` (sharded mode, Vandermonde + Berlekamp-Welch)

Joint mode creates a single `.xpa` file, in the systematic mode both the `.xpa` parity file and the original input are needed for recovery. Sharded mode creates 11 files: 10 data shards (systematic, not re-encoded) and 1 parity shard.

par2 family tested via:
- `par2 create -q -r10 enwik10.par2 enwik10`
- `par2-turbo create -q -r10 enwik10.par2 enwik10`
- `parpar -s 1M -r 10% -o enwik10.par2 enwik10`

Writes one small index file (`enwik10.par2`) plus several recovery volumes (`enwik10.vol000+01.par2`, `vol001+02`, etc) totalling ~10% of the input. The original `enwik10` is preserved and referenced by its per-slice MD5.

zfec tested via:
- `zfec -k 10 -m 11 -f enwik10`, which writes 11 files: `enwik10.000` through `enwik10.010`; 10 data shards (systematic) plus 1 parity shard, 10 of which are needed for recovery.

| tool | mean wall (s) | throughput (MB/s) | on-disk output (bytes) | artefacts |
| :--- | ---: | ---: | ---: | :--- |
| `xpar-sharded-fft` | 11.48 | 830.5 | 11,000,000,352 | 11 shards (10 data + 1 parity) |
| `xpar-sharded-van` | 14.57 | 654.5 | 11,000,000,352 | 11 shards (10 data + 1 parity) |
| `zfec`             | 20.07 | 475.2 | 11,000,000,033 | 11 shards (10 data + 1 parity) |
| `par2-turbo`       | 25.63 | 372.2 |  1,001,465,372 | index + recovery volumes (sidecar) |
| `xpar-joint-sys`   | 28.91 | 329.9 |  1,973,094,259 | single parity-only sidecar |
| `xpar-joint-i3`    | 33.85 | 281.7 | 11,441,157,089 | single archive |
| `xpar-joint-i1`    | 37.49 | 254.4 | 11,973,094,409 | single archive |
| `xpar-joint-i2`    | 40.46 | 235.7 | 11,437,146,731 | single archive |
| `parpar`           | 48.56 | 196.4 |  1,009,386,576 | index + recovery volumes (sidecar) |
| `par2cmdline`      | 114.82 | 83.1 |  1,001,465,336 | index + recovery volumes (sidecar) |

## to-do

- preserve file times & permissions; needs a header change (v2.0?), low priority.
- shard manifests like par2/parpar
- variable redundancy in joint modes.
- spec for the binary file format(s).
- `--progress` option to periodically print progress to stderr.