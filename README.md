# xpar

xpar protects files and directory trees with error and erasure correction. It
can repair bad sectors, bit flips, truncation, media rot, and transfer damage.

xpar is licensed under GNU GPL version 3 only. See [COPYING](COPYING). Report
issues to Kamila Szewczyk <k@iczelia.net>. The project is hosted at
<https://github.com/iczelia/xpar>.

[![Packaging status](https://repology.org/badge/vertical-allrepos/xpar.svg)](https://repology.org/project/xpar/versions)

## Quick start

```sh
xpar create -r 10% -o backup movie.mkv
xpar verify backup.xpa
xpar repair --in-place backup.xpa
```

`create` writes recovery volumes. `verify` returns 0 for clean data, 1 for
repairable damage, and 2 for damage beyond the available recovery. `repair`
restores what the set covers.

See the [format specification](doc/xpar-format.tex) and [xpar(1)](xpar.1) for
the format and full command reference. `make docs` builds the specification.

## Installation

Use your package manager or download a binary from GitHub Releases. To build a
release tarball, run

```sh
./configure
make
sudo make install
```

The project uses C99. A Git checkout needs `./bootstrap` first. Useful options
follow.

| Configure option | Effect |
| --- | --- |
| `--disable-simd` | Build scalar kernels only |
| `--disable-threads` | Build without threads |
| `--enable-sanitizers` | Enable ASan and UBSan for tests |

## Sets

xpar has three layouts.

| Layout | Storage |
| --- | --- |
| `sidecar` | Keep the originals; store an index and recovery beside them |
| `split` | Store consecutive raw stream ranges in `base.dNN` files |
| `armoured` | Store manifest, data, and recovery in one `base.xpa` |

Protect a tree with 15 percent redundancy and file deduplication as follows.

```sh
xpar create -r 15% --dedup=file -o photos -R 448CANON
```

The manifest records names, sizes, checksums, and selected filesystem metadata.
`--dedup=chunk` also deduplicates repeated ranges. `list` prints the manifest;
`info` prints geometry and redundancy. `--json` selects JSON Lines output and
ends a fatal stream with a summary record.

`-r` accepts a slice count, percentage, byte size, or multiple such as `2x`.
`recover` rebuilds a missing set volume. `explain` prints a shell recipe for
copying the armoured region without xpar or correction.

## Damage and repair

The outer Reed-Solomon code reconstructs missing data. xpar divides slices into
checksum-protected cells and erases only damaged cells, which limits decoding
and writes after local damage. Cells are at most 64 KiB by default; `--cell`
sets the size.

The optional inner code corrects bit errors. `--armour=metadata` protects
critical metadata in sidecar and split sets. `--armour=all` also protects slice
tables and recovery slices; armoured archives protect the whole packet stream.
The default GF(2^8) RS(255,223) code adds about 14 percent to recovery volumes.
Use `--armour-t=4` or more: at `t=1`, some two-symbol damage can resemble one
bad symbol. The outer checksum rejects the wrong correction.

`scrub` checks the inner code and reports corrected symbols. `--deep` also
recomputes recovery data.

```sh
xpar scrub --deep photos.xpa
```

`repair --in-place` keeps names, inodes, and hard links where possible. It can
recreate missing empty files, directories, links, and set volumes. An undo
journal precedes protected writes and is removed after success. Run `xpar undo`
after an interruption. An unreadable journal is retained; replace it only with
`--replace-journal` when losing it is intentional.

`repair --dry-run --exit-on-change` returns 1 when a repair would write and 0
when nothing needs changing. Resynchronisation can find data shifted by inserted
or deleted bytes.

## Generations and authentication

`add` appends a generation and inherits unchanged data. `prune` removes old
generations; `consolidate` rewrites a chain as one. Verification normally walks
oldest to newest. `--generation=G` selects one, and `info --deps` shows its
dependencies. Maintenance rewrites use a journal and refuse unreadable chains.

`--auth-key=FILE` authenticates metadata and content with a keyed MAC; it does
not encrypt. Every later operation needs the key and returns 6 when it is absent
or wrong. `--auth-only` omits public CRCs and whole-file hashes.

Always require the key when checking an authenticated set. An attacker cannot
forge its MAC, but can remove authentication and retag the set; a keyless check
would then accept an unauthenticated set.

## Exit status

| Code | Meaning |
| --- | --- |
| 0 | Success, or clean data under `verify` |
| 1 | Repairable damage, or change under `--exit-on-change` |
| 2 | Damage beyond the available recovery |
| 3 | Set not found, invalid, or unsupported |
| 4 | Usage error |
| 5 | I/O error |
| 6 | Authentication error |
| 7 | No plan fits `-m`, or memory is exhausted |
| 8 | Internal error |

## Performance and ports

xpar reads output back after writing unless `--no-verify-after` is given.
Runtime dispatch covers x86 SSSE3, SSE4.2, AVX2, GFNI, GFNI-512, and VBMI; ARM
NEON, PMULL, and SVE; PowerPC VSX; and RISC-V vectors. The format is portable.

Windows NT builds use UTF-16 paths. Windows 95 and DJGPP/MS-DOS have separate
scalar targets; see [INSTALL](INSTALL) for commands. xpar 2.0 cannot read 1.x
archives. Decode those with xpar 1.x, then protect the result with xpar 2.0.

xpar adds bit-error correction, cell-sized erasures, in-place repair, burst
interleaving, authentication, resync, generations, and deduplication beyond a
basic PAR-style workflow. It is not PAR2-compatible and has no GPU backend.
