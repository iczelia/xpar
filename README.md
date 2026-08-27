# xpar

xpar is an error and erasure correction system for guarding data integrity.
Licensed under the GNU GPL version 3 only; see COPYING.

Report issues to Kamila Szewczyk <k@iczelia.net>.
Project homepage: https://github.com/iczelia/xpar

[![Packaging status](https://repology.org/badge/vertical-allrepos/xpar.svg)](https://repology.org/project/xpar/versions)

## Synopsis

xpar can be pointed at a file or a directory to create a recovery volume beside
it.  They can be later used to restore files that fall victim to bad sectors,
bit flips, truncated copies, media rot or transmission failure.

```
% xpar create -r 10% -o backup movie.mkv     # 10% redundancy
% xpar verify backup.xpa                     # writes nothing, exits 0/1/2
% xpar repair --in-place backup.xpa          # fix what broke
```

xpar and its file format are competitors to PAR2, PAR2-turbo, ParPar, QuickPar,
Par3, zfec, and numerous front-ends to ISA-L.  The tool's core operation presents
a handful of key improvements: xpar corrects bit errors as well as lost/invalid
blocks, i.e. it uses two layers of Reed-Solomon codes to operate, repairs are done
in-place, and small (optionally transposed, for better burst error correction)
blocks are used for the inner code.

Its file format has been extensively documented by the formal specification
in [doc/xpar-format.tex](doc/xpar-format.tex); `make docs` typesets it, and a
ready-made `xpar-format-<version>.pdf` is attached to every release.  A manual
page is also supplied.

## Upgrading from xpar 1.x

xpar 2.0 is a rewrite.  The container format is new and this version is unable
to read xpar 1.x archives; no 1.x decoder is shipped.  1.x files are refused
processing:

```
% xpar verify old.xpa
xpar: 'old.xpa' is an xpar 1.0 joint-mode archive.
xpar: decode it with xpar 1.x, then re-protect it.
```

To move, decode with an xpar 1.x binary and re-protect the result with 2.0.

The command line also changed: operations are verbs rather than mode flags,
so `-J`, `-W`, `-L`, `-s`, `-t`, `--interlacing`, `-H`/`--integrity` and
`--auth` are gone.  Scripts written against xpar 1.x will not run unchanged.

## Installation

If your software vendor packages xpar, it is the easiest to get it from there.
Otherwise, consider downloading the fitting for your CPU architecture and
operating system binary file from the Releases tab.

In order to build from source code, download a distribution tarball from the
releases tab and execute the following commands:

```
% ./configure && make && sudo make install
```

`configure` probes for all SIMD extensions that xpar can use.  Each becomes
a convenience library with its own compilation flags, while the kernel is
picked at the runtime based on signals from CPUID + OSXSAVE/XCR0 (x86),
HWCAP or `riscv_hwprobe`.

| option | effect |
| :--- | :--- |
| `--disable-simd` | scalar kernels only |
| `--disable-threads` | single-threaded build |
| `--enable-sanitizers` | ASan + UBSan, for the test suite |

## Basic usage

`create` protects files or whole directory trees (when `-R` is specified).  The
argument `-r` specifies the intended amount of redundancy as a percentage, byte
size, or a multiple (e.g. `-r 2x`).

```
% xpar create -r 15% --dedup=file -o photos -R 448CANON
xpar: photos: 8 entries, 245 slices of 4096 bytes, 37 recovery slices in 5 volumes
```

The `448CANON` directory will not be altered. `photos.xpa` will hold the
manifest of the data protected (i.e., names, sizes, modes, modification and
creation times, checksums, permissions, ...), while `photos.v*.xpa` will hold
the Reed-Solomon recovery data, split across volumes in a doubling ladder.

xpar provides built-in deduplication.  Identical files are thus stored only
once.  `list --dedup` shows shared contents:

```
% xpar list --dedup photos.xpa
generation 0  set 42f3a6c40f8ec6b826a1850458c005eb  8 entries
  t         size  gen  mode   mtime                 name
  f       300000    0  0664   2026-08-21T21:40:13Z  pics/IMG0001.jpg
      extent 0 + 300000  in generation 0  refs=2
  ...
  f       300000    0  0664   2026-08-21T21:40:13Z  pics/IMG0001 (Copy).jpg
      extent 0 + 300000  in generation 0  refs=2
```

`verify` checks the integrity of the data.  Nothing is ever written to disk.
Three exit codes/ERRORLEVELs are possible: 0 (clean), 1 (damaged but
repairable), 2 (hopelessly broken).

```
% xpar verify backup.xpa
xpar: movie.mkv: content differs
xpar: 3956 slices of 5056 bytes, 396 recovery slices, erasure unit cell of 5056 bytes (1 per slice)
xpar: coverage: tree (1 entry)
xpar: damaged: 1 entry (0 missing), 1 slice, 1 cell; deepest column 1
xpar: status: repairable
```

`--json` may be passed for a more machine friendly output:

```
{"type":"summary","t":4864,"status":"clean","exit":0,"slices_checked":1221,
 "slices_bad":0,"cells_bad":0,"column_depth":0,"column_groups":0,
 "recovery_available":366,"recovery_needed":0,"entries_damaged":0,
 "entries_alias_only":0,"entries_opaque":0,"entries_superseded":0,
 "syndromes":0,"bytes_read":10001216,"bytes_written":0}
```

In order to repair broken archives, one may use the `repair` command:

```
% xpar repair --in-place backup.xpa
xpar: 1 cell damaged, 0 copied, 1 decoded; 1 write, 5056 bytes; 1 entry repaired (0 further names share a repaired inode).
```

In this simulation, one randomly garbled byte in a 20 MB file cost a write
of only 5 KiB.  The file keeps its name, inode, and hard links.  An undo journal
is written first and removed on success.  This ensures data integrity if the
machine powers off mid repair, or the tool crashes, `xpar undo` replays stale
journal files.  `repair` issues a write only after the corrected result matches
the stored BLAKE3 checksum, and then re-verifies the finished file.

## Why xpar?

Mainly because erasure correction is not error correction.  Unlike PAR2,
PAR2Turbo, PAR3 (Draft), ParPar, or zfec, xpar is also an error-correcting
container.  It is thus the only tool that can use a partially correct
information.  It is radically more common for files to contain the (mildly)
wrong data rather than outright disappear.  Was it otherwise, why does every
respectable tool under the sun include a checksum in its binary format?
Firstly, erasures are correctible within a block.

```
% xpar verify b.xpa
xpar: big.bin: content differs
xpar: 48 slices of 2097152 bytes, 5 recovery slices, erasure unit cell of 65536 bytes (32 per slice)
xpar: coverage: tree (1 entry)
xpar: damaged: 1 entry (0 missing), 23 slices, 30 cells; deepest column 3
xpar: status: repairable

% xpar repair --in-place b.xpa
xpar: 30 cells damaged, 0 copied, 30 decoded; 28 writes, 1966080 bytes; 1 entry repaired (0 further names share a repaired inode).
% cmp big.bin big.keep && echo ok
ok
```

Secondly, errors can be optionally corrected within the erasure slices using
an inner error-correcting code.  xpar can augment the slice data with additional
redundancy to protect against spurious bit-flips.

```
% xpar verify arch.xpa
xpar: 74 slices of 4096 bytes, 7 recovery slices, erasure unit cell of 4096 bytes (1 per slice)
xpar: armoured metadata: 1 region corrected, 0 past the inner code
xpar: coverage: tree (1 entry)
xpar: status: clean

% xpar scrub arch.xpa
xpar: inner code: 1 regions, 169 codewords, 0 clean, 169 corrected, 0 past capacity
xpar: corrected symbols: 2000 total, worst codeword 20
xpar:   codewords corrected at 11 symbols: 25
xpar:   codewords corrected at 12 symbols: 16
xpar:   ...
xpar:   codewords corrected at 20 symbols: 2
```

Such granular information is typically unachievable with other tools.

Thirdly, errors are not necessarily corruption.  xpar handles insertions or
deletions as well:

```
% xpar repair --in-place p.xpa
xpar: ./data.bin: found 732 displaced slices with 732 strong confirmations.
xpar: 733 cells damaged, 733 copied, 0 decoded; 2 writes, 3000000 bytes; 1 entry repaired (0 further names share a repaired inode).
% cmp data.bin data.keep && echo identical
identical
```

Finally, the xpar format has been designed in order to be easily extractable
(without error correction, of course) via only standard and simple UNIX
commands.  This provides long-term data security, in case this tool ever
disappears off the face of the planet.

```
% xpar explain p.xpa
[...]
  symbol width W   1 byte (GF(2^8))
  code             RS(255, 223), t = 16
  interleave D     1
  frame            255 bytes on disk, 223 of plaintext
  frames           28
[...]
set -e
in=p.xpa
out=recovered.bin
W=1; n=255; k=223; D=1; hdr=384
Fd=$((D*k*W))          # plaintext bytes per frame = 223
Fx=$((D*n*W))          # disk bytes per frame      = 255
frames=28
off=648               # stream_offset from the prologue
len=1008               # stream_length from the prologue

# 1. drop the prologue in one read, so no later step needs to skip it
dd if="$in" of=region.bin bs=$hdr skip=1 status=none

# 2. take the first Fd bytes of every Fx-byte frame.
f=0
while [ $f -lt $frames ]; do
  dd if=region.bin bs=$Fx skip=$f count=1 status=none | head -c $Fd
  f=$((f+1))
done > plain.bin

# 3. the protected stream is len bytes at off inside that plaintext
if [ $off -gt 0 ]; then
  dd if=plain.bin bs=$off skip=1 status=none | head -c $len > "$out"
else
  head -c $len plain.bin > "$out"
fi
# end of recipe
```

## Comparison

| | xpar | par2cmdline | par2cmdline-turbo | ParPar | zfec |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Protect file sets / trees | yes | yes | yes | yes | one file |
| Filenames + metadata stored | yes | yes | yes | yes | no |
| Arbitrary redundancy | yes (`%`, count, size, `1x`) | yes | yes | yes | k, m free |
| Verify without repairing | yes | yes | yes | creation-focused | no |
| **Corrects bit errors** | **yes** | no | no | no | no |
| **Erasure unit finer than a block** | **yes (64 KiB cells)** | no | no | no | no |
| **Repairs in place** | **yes, + undo journal** | rename + recreate | rename + recreate | n/a | no |
| **Burst-error interleaving** | **yes** | no | no | no | no |
| **Keyed authentication** | **yes** | no | no | no | no |
| **Media-health check** | **yes (`scrub`)** | no | no | no | no |
| Resync after insert/delete | yes | yes | yes | n/a | no |
| Add recovery later | yes | yes | yes | yes | no |
| Memory cap / multi-pass | yes | yes | yes | yes | streams |
| Incremental generations | yes | no | no | no | no |
| Deduplication | yes (file / chunk) | no | no | no | no |
| Published format spec | yes | yes | yes | yes | yes |
| GFNI / AVX-512 | yes | no | yes | yes | no |
| SVE, RISC-V V | yes | no | yes | yes | no |
| GPU (OpenCL) | no | no | no | yes | no |
| Legacy targets | yes | no | no | no | no |

Some drawbacks of xpar:

- It's not as mature; PAR2 is interoperable and very solid.
- ParPar often has a speed edge over xpar and there are currently no plans
  for an OpenCL backend.

## Other features

- `add` / `prune` / `consolidate`: manages a chain of generations in the `xpa`
  file.  `add` appends a new generation detailing the changes against the
  current disk contents.  Unchanged files are inherited and not re-stored.
  Since redundancy is per-generation, `info --deps` shows old generations that
  are still required for operation. `prune` removes generations that are
  unreferenced.  `consolidate` collapses generational chains, merging all
  generations into one, and re-encodes the archive.
```
% xpar add -r 15% photos.xpa -R pics
xpar: generation 1, set 9c1d0e5b7a3f48d2b6e0c4a1f75d3e88: 9 entries
      (1 added, 1 changed, 7 inherited, 0 dropped), 200000 new stream
      bytes, 7 recovery slices in 3 volumes.
```
- `recover`: regenerate lost volume(s) from the surviving set.  For example,
  when with `--layout=split --volumes=4`, the data lives in header-free
  `.d00` - `.d03` files, one per disk, then:
```
% rm disc.d01
% xpar recover --volume=disc.d01 disc.xpa
xpar: recovered disc.d01 from survivor and parity slices (1253376 bare stream bytes).
% cat disc.d0* > joined && cmp joined vid.mp4 && echo identical
identical
```
- `--auth-key=FILE`: authenticate the set with a keyed MAC, as a precaution
  against doctored archives.  Every subsequent operation requires the key:
```
% xpar verify s.xpa
xpar: This set is authenticated; supply --auth-key=FILE.
% xpar verify --auth-key=wrong.bin s.xpa
xpar: The authentication key is wrong for this set.
```
- `--memory`: sets the maximum working set size for the planner.
```
% xpar create -m 1M -r 20% -o tiny data.bin
xpar: No plan fits: raise -m to 2.0 MiB, note that no -b fits this -m,
      or use --codec=matrix (which does not fit either at -m 1.0 MiB).
```
- `info`: displays information about the erasure and error correction
  data type.
```
  geometry   : Z = 2097152 (2.0 MiB), S = 48, L = 100000000 (95.4 MiB)
  cells      : Y = 65536 bytes, K = 32 per slice; the erasure unit is
               (slice, column), not a whole slice
  codec      : matrix over GF(2^8), recovery axis 2^8 = 208 slices
  redundancy : R = 5 (10.4% of S), 5 recovery slices present
  armour     : GF(2^8) RS(255, 223), t = 16, D = 1
  ...
  field      : S + R = 53 <= 256, so GF(2^8)
  memory     : work buffers 22.1 MiB;  read-ahead 0 B;  stage + hash 32.0 MiB
               total 54.1 MiB
  cells      : Y = 65,536 B, K = 32 per slice (last cell 65,536 B)
               erasure budget is 5 per column, not 5 per set
  passes     : 1 sequential read totalling 100,000,000 bytes
```
- `benchmark`: displays the performance metrics for low-level SIMD kernels used
  by xpar.
```
% xpar benchmark --tiers
xpar: benchmark: V-HASH, V-CRC and V-GEN KATs ok
xpar: benchmark: tier         operation         bytes     time       rate
xpar: benchmark: gfni256      gf8-mac         8388608 bytes      175 us  45714.29 MiB/s
xpar: benchmark: gf tier gfni256  ok
xpar: benchmark: gfni512      gf8-mac         8388608 bytes      218 us  36697.25 MiB/s
xpar: benchmark: gf tier gfni512  ok
xpar: benchmark: ...
xpar: benchmark: gfni256      armour-gf8       285440 bytes     1028 us    264.80 MiB/s
xpar: benchmark: gfni256      armour-gf16     8616960 bytes     7867 us   1044.59 MiB/s
xpar: benchmark: armour tier gfni256  ok
xpar: benchmark: ...
xpar: benchmark: crc32c sse4.2, blake3 avx2
xpar: benchmark: 9 tiers agree with scalar
```

The following exit codes/ERRORLEVELs of all sub-commands are possible:
- 0: No error.
- 1: File damaged but repairable.
- 2: File hopelessly damaged.
- 3: Not found / Not an xpar set / Wrong version.
- 4: Wrong usage.
- 5: I/O error.
- 6: Authentication error.
- 7: No feasible plan found.
- 8: Internal error.

## Binary layouts

Three options are offered:

- `--layout=sidecar` (default): files are never touched and stay where they
  were. `base.xpa` and `base.v*.xpa` sit beside them. Extraction is never
  necessary.
- `--layout=split`: the protected data is written into files `base.d00`,
  `base.d01`, etc., which are raw data volumes with no header, no trailer
  and no padding, so that `cat base.d* > file` reconstructs the original
  stream exactly.
- `--layout=armoured`: one self-contained `base.xpa` containing three
  replicated header copies, alongside an inner-coded packet stream with the
  manifest, the data and the recovery.  Use `extract` to unpack, rather than
  concatenation.

## Performance

Generally, xpar comes close in terms of performance to par2cmdline-turbo and
ParPar in the single core mode.  Parallel workloads are not optimised as well
yet.

Further, xpar issues a mandatory readback (that can nonetheless be disabled
with `--no-verify-after`) to ensure that the data has been written correctly.

## Portability

Specialised kernels exist for x86's extensions SSSE3 / SSE4.2 / AVX2 / GFNI
/ GFNI-512 / VBMI, ARM NEON / PMULL / SVE, PowerPC VSX, and
RISC-V vector (both shuffle and clmul paths).  The binary format is
platform-agnostic.

Windows specifically has two targets; the default `nt` target uses UTF-16,
extended-length paths and a Vista+ API floor.  It is used like so:

```
% CC=x86_64-w64-mingw32-gcc ./configure --host=x86_64-w64-mingw32 \
      --with-windows-target=nt && make
```

The `win95` target is a separate ASCII-only path with an i486 (1989) scalar
baseline, a freestanding startup/runtime and a PE 4.0 header:

```
% CC=i686-w64-mingw32-gcc ./configure --host=i686-w64-mingw32 \
      --with-windows-target=win95 && make
```

That executable imports only `KERNEL32.DLL`. MS-DOS builds through DJGPP
(`--host=i586-pc-msdosdjgpp`), likewise with an i386 (1985) scalar baseline
and no optional SIMD.  A 32-bit NT build may contain SSSE3 and SSE4.2 objects.
