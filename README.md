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
