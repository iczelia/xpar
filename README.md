# xpar

xpar protects files and directories with error and erasure correction. It can
restore data damaged by bad sectors, bit flips, truncation, media rot, or
transmission failures.

xpar is licensed under GNU GPL version 3 only. See [COPYING](COPYING).
Report issues to Kamila Szewczyk <k@iczelia.net>. The project is hosted at
<https://github.com/iczelia/xpar>.

[![Packaging status](https://repology.org/badge/vertical-allrepos/xpar.svg)](https://repology.org/project/xpar/versions)

## Quick start

```sh
xpar create -r 10% -o backup movie.mkv
xpar verify backup.xpa
xpar repair --in-place backup.xpa
```

`create` writes recovery volumes. `verify` checks data without changing it and
exits 0 when clean, 1 when repairable, or 2 when beyond repair. `repair` restores
damage covered by the available recovery data.

xpar combines an outer Reed-Solomon code with an optional inner code. The outer
code reconstructs missing data. The inner code corrects bit errors and supports
interleaving for burst damage. Repairs can update only the damaged ranges.

See the [format specification](doc/xpar-format.tex) and the `xpar(1)` manual for
the full format and command reference. `make docs` builds the PDF specification.

## Installation

Use your software vendor's package when available, or download a binary from the
GitHub Releases page. To build a release tarball from source, run

```sh
./configure
make
sudo make install
```

`configure` detects supported SIMD extensions. Runtime dispatch selects the
best available kernel.

| Configure option | Effect |
| --- | --- |
| `--disable-simd` | Build only scalar kernels |
| `--disable-threads` | Build without multithreading |
| `--enable-sanitizers` | Enable ASan and UBSan for tests |

## Basic usage

Protect a directory tree with 15 percent redundancy and file deduplication.

```sh
xpar create -r 15% --dedup=file -o photos -R 448CANON
```

The source tree is unchanged. `photos.xpa` stores the manifest, while
`photos.v*.xpa` stores recovery data. The manifest includes names, sizes,
checksums, and selected filesystem metadata.

`-r` accepts a slice count, percentage, byte size, or multiple such as `2x`.
`--dedup=file` stores identical files once. Chunk deduplication is also
available.

Inspect a set with `list` or `info`.

```sh
xpar list --dedup photos.xpa
xpar info photos.xpa
```

Use `--json` for JSON Lines output.

```sh
xpar verify --json photos.xpa
```

A fatal error still ends the stream with a summary record, so consumers can
identify why processing stopped.

## Repair behavior

`repair --in-place` preserves names, inodes, and hard links where possible. It
also recreates missing empty files, directories, and symbolic links. Damaged or
stale set volumes can be rebuilt from valid replicas.

An undo journal is written before protected data changes. A successful repair
removes it. After an interruption, run `xpar undo` to restore the previous
state. An unreadable or invalid journal is retained because xpar cannot safely
determine whether writes began. Use `--replace-journal` only when discarding it
is intentional.

`repair --dry-run --exit-on-change` exits 1 when a repair would write data and 0
when no change is needed.

## Why xpar

Most erasure-correction tools reconstruct whole missing blocks. xpar divides
each slice into checksum-protected cells and treats only damaged cells as
erasures. This can reduce both recovery cost and the amount written during a
repair.

Cells are at most 64 KiB by default. Slices below 4 KiB use the whole slice as
the erasure unit. `--cell` selects an explicit size.

The optional inner code protects against bit errors. In sidecar and split sets,
`--armour=metadata` protects critical metadata. `--armour=all` also protects
slice tables and recovery slices. An armoured archive protects its entire
packet stream.

At the default GF(2^8) RS(255,223) code, full armour adds about 14 percent to
the recovery volumes. At `-r 10%`, this is about 1.5 percent of the protected
data. Larger recovery slices may use GF(2^16) with less overhead.

Use `--armour-t=4` or higher. With `t=1`, some two-symbol damage can be mistaken
for a one-symbol error. The outer checksum catches the failed repair.

`scrub` reads all data and recovery volumes and reports corrected symbols. Add
`--deep` to recompute and compare recovery data.

```sh
xpar scrub --deep photos.xpa
```

## Layouts

xpar supports three layouts.

| Layout | Storage |
| --- | --- |
| `sidecar` | Original files remain in place. An index and recovery volumes sit beside them. |
| `split` | Raw `base.dNN` files hold consecutive stream ranges. Concatenating them reconstructs the stream. |
| `armoured` | One self-contained `base.xpa` holds the manifest, data, and recovery stream. Use `extract` to unpack it. |

Use `recover` to rebuild a missing set volume. The recovery count must cover all
slices in that volume.

```sh
xpar recover --volume=disc.d01 disc.xpa
```

`explain` prints a shell recipe for extracting an armoured region without xpar
or error correction. For an armoured layout this region contains the protected
data. For sidecar and split layouts it contains critical metadata.

## Generations

`add` appends a generation for new or changed files. Unchanged data is inherited
instead of stored again. `prune` removes unneeded generations, and `consolidate`
re-encodes a chain as one generation.

Verification and repair normally process the chain from oldest to newest.
`--generation=G` selects one generation. `info --deps` shows dependencies and
recovery consumed by superseded slices.

`prune` and `consolidate` refuse to rewrite a chain containing an unreadable
generation. They use a maintenance journal so `xpar repair` can finish or roll
back an interrupted update.

## Authentication

`--auth-key=FILE` authenticates metadata and content with a keyed MAC. It does
not encrypt the set. Every later operation, including `explain`, requires the
key and exits 6 if the key is missing or wrong.

Authentication is effective only when the verifier always supplies a key. An
attacker without the key cannot forge a MAC, but can remove authentication and
retag the set. A keyless verifier would then see an unauthenticated set. The
calling script must therefore require the key.

`--auth-only` omits public CRCs and whole-file hashes from an authenticated set.

## Comparison

| Capability | xpar | par2cmdline | par2cmdline-turbo | ParPar | zfec |
| --- | --- | --- | --- | --- | --- |
| Protect file sets or trees | yes | yes | yes | yes | one file |
| Store names and metadata | yes | yes | yes | yes | no |
| Correct bit errors | yes | no | no | no | no |
| Use erasure units smaller than a block | yes | no | no | no | no |
| Repair in place | yes | no | no | n/a | no |
| Interleave burst errors | yes | no | no | no | no |
| Authenticate with a key | yes | no | no | no | no |
| Check media health | yes | no | no | no | no |
| Resync after insertion or deletion | yes | yes | yes | n/a | no |
| Add recovery later | yes | yes | yes | yes | no |
| Incremental generations | yes | no | no | no | no |
| Deduplicate content | yes | no | no | no | no |
| Published format | yes | yes | yes | yes | yes |
| GPU backend | no | no | no | yes | no |

xpar is less mature than PAR2 and is not PAR2-compatible. ParPar can be faster
and provides an OpenCL backend, which xpar does not currently plan to add.

## Exit status

| Code | Meaning |
| --- | --- |
| 0 | Success, or clean data under `verify` |
| 1 | Repairable damage, or a change reported by `--exit-on-change` |
| 2 | Damage beyond the available recovery |
| 3 | Set not found, invalid set, or unsupported version |
| 4 | Usage error |
| 5 | I/O error |
| 6 | Authentication error |
| 7 | No plan fits within `-m`, or memory is exhausted |
| 8 | Internal error |

## Performance and portability

Single-core performance is comparable to par2cmdline-turbo and ParPar. Parallel
workloads are less optimized. xpar reads output back after writing by default.
Use `--no-verify-after` to skip that pass.

Optimized kernels cover x86 SSSE3, SSE4.2, AVX2, GFNI, GFNI-512, and VBMI;
ARM NEON, PMULL, and SVE; PowerPC VSX; and RISC-V vector extensions. The binary
format is platform-independent.

The default Windows target uses UTF-16 paths and requires Vista or later.

```sh
CC=x86_64-w64-mingw32-gcc ./configure \
  --host=x86_64-w64-mingw32 --with-windows-target=nt
make
```

The separate `win95` target uses ASCII paths, an i486 scalar baseline, and a PE
4.0 header.

```sh
CC=i686-w64-mingw32-gcc ./configure \
  --host=i686-w64-mingw32 --with-windows-target=win95
make
```

The Win95 executable imports only `KERNEL32.DLL`. DJGPP builds target MS-DOS
with `--host=i586-pc-msdosdjgpp` and use an i386 scalar baseline.

## Upgrading from xpar 1.x

xpar 2.0 cannot read version 1 archives. Decode them with xpar 1.x, then protect
the result with xpar 2.0. Version 2 also replaces mode flags with verbs, so
scripts written for version 1 need updating.
