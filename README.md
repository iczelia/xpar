# xpar
xpar - an error/erasure code system guarding data integrity.
Licensed under the terms of GNU GPL version 3 or later - see COPYING.
Report issues to Kamila Szewczyk <k@iczelia.net>.
Project homepage: https://github.com/iczelia/xpar

xpar in joint mode generates a slightly inflated (by about 12%) parity-guarded
file from a given data file. Such a file can be recovered as long as no more
than about 6.2% of the data is corrupted. xpar internally uses a (255,223)-RS
code over an 8-bit Galois field.

can be consiered a sibling project of par2.

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
```

Encode + Decode speed on random data (most adversarial scenario; joint mode, no interlacing): 244MB/s, 132MB/s respectively.
