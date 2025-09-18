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

## Usage

Consult the man page.

## Disclaimer

The file format will change until the stable version v1.0 is reached.
Do not use xpar for critical data, do not expect backwards or forwards
compatibility until then.

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

Quiet AMD Ryzen 9 5900X 12-Core Processor running Linux 6.12.38+deb13-amd64 #1 SMP PREEMPT_DYNAMIC Debian 6.12.38-1 (2025-07-16) x86_64 GNU/Linux.

```
% clang --version
Debian clang version 19.1.7 (3+b2)
Target: x86_64-pc-linux-gnu
Thread model: posix
InstalledDir: /usr/lib/llvm-19/bin
% ./configure --enable-x86_64 --enable-native --enable-lto CC=clang
% make -j8
% head -c 1000000000 /dev/urandom > data.bin   # 1 GB of random data to protect.
% hyperfine './xpar -Jef data.bin'
Benchmark 1: ./xpar -Jef data.bin
  Time (mean ± σ):      5.508 s ±  0.055 s    [User: 4.093 s, System: 0.903 s]
  Range (min … max):    5.360 s …  5.562 s    10 runs
% hyperfine './xpar -Jdf data.bin.xpa'
Benchmark 1: ./xpar -Jdf data.bin.xpa
  Time (mean ± σ):      8.772 s ±  0.110 s    [User: 7.554 s, System: 0.783 s]
  Range (min … max):    8.478 s …  8.893 s    10 runs
% hyperfine 'rm -f *.par2 && par2create data.bin'
Benchmark 1: rm -f *.par2 && par2create data.bin
  Time (mean ± σ):      6.529 s ±  0.043 s    [User: 154.445 s, System: 0.217 s]
  Range (min … max):    6.474 s …  6.624 s    10 runs
% hyperfine 'par2 verify data.bin.par2'
Benchmark 1: par2 verify data.bin.par2
  Time (mean ± σ):      3.659 s ±  0.011 s    [User: 3.605 s, System: 0.058 s]
  Range (min … max):    3.653 s …  3.688 s    10 runs
% hyperfine 'rm data.bin.xpa* && ./xpar -Sef --dshards=50 --pshards=4 --out-prefix=data.bin.xpa data.bin'
Benchmark 1: rm data.bin.xpa* && ./xpar -Sef --dshards=50 --pshards=4 --out-prefix=data.bin.xpa data.bin
  Time (mean ± σ):      2.154 s ±  0.016 s    [User: 0.812 s, System: 0.782 s]
  Range (min … max):    2.135 s …  2.188 s    10 runs
% hyperfine './xpar -Sdf data.org data.bin.xpa*'
Benchmark 1: ./xpar -Sdf data.org data.bin.xpa*
  Time (mean ± σ):      1.480 s ±  0.014 s    [User: 0.029 s, System: 0.999 s]
  Range (min … max):    1.464 s …  1.505 s    10 runs
```

Numbers:
- Encode + Decode speed on random data (most adversarial scenario; joint mode, no interlacing): 244MB/s, 132MB/s respectively.
- Encode + Decode speed on random data (sharded mode, 50 + 4): 1231MB/s, 34482MB/s respectively.
- PAR2 Encode + Decode speed on random data (sharded mode): 153MB/s, 277MB/s.

## TO-DO

- Hook up sse2neon inside lmode.c
- GPU-based jmode.c