# xpar

xpar is an error and erasure correction system for guarding data integrity.
Licensed under the GNU GPL version 3 only; see COPYING.

Report issues to Kamila Szewczyk <k@iczelia.net>.
Project homepage: https://github.com/iczelia/xpar

[![Packaging status](https://repology.org/badge/vertical-allrepos/xpar.svg)](https://repology.org/project/xpar/versions)

## Synopsis

xpar creates recovery volumes for files and directories. Use them to restore
data damaged by bad sectors, bit flips, truncated copies, media rot, or
transmission failures.

```
% xpar create -r 10% -o backup movie.mkv     # 10% redundancy
% xpar verify backup.xpa                     # writes nothing, exits 0/1/2
% xpar repair --in-place backup.xpa          # fix what broke
```

Alternatives include PAR2, PAR2-turbo, ParPar, QuickPar, Par3, zfec, and ISA-L
front-ends. xpar uses two Reed-Solomon layers to correct both bit errors and
lost or invalid blocks. Repairs are performed in place. The inner code uses
small blocks that can be transposed for better burst-error correction.

The format is specified in [doc/xpar-format.tex](doc/xpar-format.tex). Run
`make docs` to typeset it; releases include `xpar-format-<version>.pdf`. A
manual page is also included.

## Upgrading from xpar 1.x

xpar 2.0 uses a new container format and cannot read xpar 1.x archives. It
rejects 1.x files with migration instructions:

```
% xpar verify old.xpa
xpar: 'old.xpa' is an xpar 1.0 joint-mode archive.
xpar: decode it with xpar 1.x, then re-protect it.
```

Decode with xpar 1.x, then protect the result with xpar 2.0.

The command line also changed: operations are verbs rather than mode flags,
so `-J`, `-W`, `-L`, `-s`, `-t`, `--interlacing`, `-H`/`--integrity` and
`--auth` are gone.  Scripts written against xpar 1.x will not run unchanged.

## Installation

Install your software vendor's xpar package when available. Otherwise,
download the binary for your operating system and CPU from the Releases tab.

To build from source, download a release tarball and run:

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
xpar: photos: 9 entries, 513 slices of 4096 bytes, 77 recovery slices in 6 volumes
```

The `448CANON` directory will not be altered. `photos.xpa` will hold the
manifest of the data protected (i.e., names, sizes, modes, modification and
creation times, checksums, permissions, ...), while `photos.v*.xpa` will hold
the Reed-Solomon recovery data, split across volumes in a doubling ladder.

xpar provides built-in deduplication.  Identical files are thus stored only
once.  `list --dedup` shows shared contents:

```
% xpar list --dedup photos.xpa
generation 0  set e356e96ef0495c7c3360960a49d50a2b  9 entries
  t         size  gen  mode   mtime                 name
  d            0    0  0775   2026-08-28T18:30:49Z  448CANON
  f       300000    0  0664   2026-08-28T18:30:49Z  448CANON/IMG0001 (Copy).jpg
      extent 0 + 300000  in generation 0  refs=2
  f       300000    0  0664   2026-08-28T18:30:49Z  448CANON/IMG0001.jpg
      extent 0 + 300000  in generation 0  refs=2
  ...
```

`verify` checks the integrity of the data, and writes nothing beside the
set: an armoured archive is staged in memory when its plaintext fits `-m`,
otherwise under `$TMPDIR` (or `TMP`/`TEMP`), and the stage is removed on
exit.  Only when none of those variables is set and the plaintext exceeds
`-m` does the stage fall back to a temporary file beside the archive.
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
% xpar verify --json photos.xpa | tail -1
{"type":"summary","t":7019,"schema":1,"status":"clean","exit":0,
 "slices_checked":513,"slices_bad":0,"cells_bad":0,"cells_superseded":0,
 "column_depth":0,"column_groups":0,"recovery_available":77,
 "recovery_needed":0,"entries_damaged":0,"entries_alias_only":0,
 "entries_opaque":0,"entries_inherited_damaged":0,"entries_superseded":0,
 "volumes_substituted":0,"volumes_to_rewrite":0,"syndromes":0,
 "bytes_read":2400000,"bytes_written":0}
```

A fatal error ends the stream with a summary of its own, so a consumer
never has to guess why the output stopped:

```
% xpar verify --json nosuch.xpa
{"type":"summary","t":26,"schema":1,"status":"error","exit":3,
 "message":"No xpar set found for 'nosuch.xpa'; use 'xpar --help' to list verbs."}
```

Repair a damaged archive with the `repair` command:

```
% xpar repair --in-place backup.xpa
xpar: 1 cell damaged, 0 copied, 1 decoded; 1 write, 5056 bytes; 1 entry repaired (0 further names share a repaired inode).
```

In this simulation, one randomly garbled byte in a 20 MB file cost a write
of only 5 KiB.  The file keeps its name, inode, and hard links.  An undo journal
is written first and removed on success.  This ensures data integrity if the
machine powers off mid repair, or the tool crashes, `xpar undo` replays stale
journal files.  The journal is created with mode 0600 and never follows a
symlink planted under its name.  `repair` issues a write only after the
corrected result matches the stored BLAKE3 checksum, and then re-verifies the
finished file.

`repair --in-place` also puts back what is missing rather than only what is
damaged: empty files, directories and symbolic links are recreated from the
manifest with their recorded mode and times, a packet-bearing volume whose
packets fail their checksums is rewritten from intact replicas in the other
volumes, and a split data volume that was renamed or is the wrong length is
restored to its recorded name and length.  `repair --dry-run
--exit-on-change` exits 1 when writes would be made and 0 when they would
not, which is what a monitoring check wants.

## Why xpar?

Erasure correction reconstructs missing blocks; error correction can also use
the intact parts of damaged blocks. Unlike PAR2, PAR2Turbo, PAR3 (Draft),
ParPar, and zfec, xpar supports both. Within a block, xpar uses checksums to
identify intact cells and treats only damaged cells as erasures.

Cells are `min(64 KiB, Z)`, so a slice size large enough to reach the 64 KiB
ceiling has to be asked for; here `-s 2M` gives 32 cells per slice.  Below
the 4 KiB floor a slice has no cells at all and the erasure unit is the whole
slice.

```
% xpar create -s 2M -r 10% -o b big.bin        # big.bin is 100 MB
xpar: b: 1 entry, 48 slices of 2097152 bytes, 5 recovery slices in 3 volumes
% xpar verify b.xpa
xpar: big.bin: content differs
xpar: 48 slices of 2097152 bytes, 5 recovery slices, erasure unit cell of 65536 bytes (32 per slice)
xpar: coverage: tree (1 entry)
xpar: damaged: 1 entry (0 missing), 22 slices, 29 cells; deepest column 3
xpar: damage has 17 column patterns; repair needs that many decode plans
xpar: status: repairable

% xpar repair --in-place b.xpa
xpar: 29 cells damaged, 0 copied, 29 decoded; 29 writes, 1900544 bytes; 1 entry repaired (0 further names share a repaired inode).
% cmp big.bin big.keep && echo ok
ok
```

An optional inner code adds redundancy that corrects bit errors in place. In
sidecar and split sets `--armour=metadata`, the default, covers the critical
metadata group; `--armour=all` also wraps every slice-table packet and every
recovery slice in an inner code of its own, so a flipped bit in a recovery
volume costs a few parity symbols instead of a whole Z-byte slice. In the
armoured layout the inner code covers the whole archive, which is what
`--armour=all` means there.

`--armour=all` costs about 14% of the recovery volumes at the default GF(2^8)
RS(255,223) code, so about 1.5% of the protected data at `-r 10%`. Once a
recovery slice is large enough for the wider GF(2^16) code to pay for its
128 KiB frame, xpar picks that instead and the cost falls under 2%;
`--armour-field` forces either field. Bare split data volumes stay raw
whatever the level, and a clean `verify` of an `all` set runs within about a
tenth of the time a `metadata` one takes.

```
% xpar verify arch.xpa
xpar: 2 stored packets failed a checksum; replicas were used. Rewrite the set volumes to clear them.
xpar: 74 slices of 4096 bytes, 7 recovery slices, erasure unit cell of 4096 bytes (1 per slice)
xpar: armoured regions: 1 corrected, 0 past the inner code
xpar: coverage: tree (1 entry)
xpar: status: clean

% xpar scrub arch.xpa
xpar: 2 stored packets failed a checksum; replicas were used. Rewrite the set volumes to clear them.
xpar: 74 slices of 4096 bytes, 7 recovery slices, erasure unit cell of 4096 bytes (1 per slice)
xpar: armoured regions: 1 corrected, 0 past the inner code
xpar: coverage: tree (1 entry)
xpar: status: clean
xpar: recovery: 7 slices named, 7 present, 2 packets failed their checksum
xpar: recovery: 1 packets had invalid lengths
xpar: inner code: 7 regions, 35 codewords, 30 clean, 5 corrected, 0 past capacity
xpar: corrected symbols: 67 total, worst codeword 16
xpar:   codewords corrected at 11 symbols: 1
xpar:   codewords corrected at 12 symbols: 1
xpar:   codewords corrected at 13 symbols: 1
xpar:   codewords corrected at 15 symbols: 1
xpar:   codewords corrected at 16 symbols: 1
xpar: scrub: exit 1
```

`--armour-t` sets the correction power of that code directly.  Keep it at
four or more: at `--armour-t=1` a codeword carries two parity symbols, so
a two-symbol error is silently miscorrected rather than flagged, and only
the outer checksums notice.

xpar also handles inserted or deleted data:

```
% xpar repair --in-place p.xpa
xpar: ./data.bin: found 732 displaced slices (732 confirmations).
xpar: restored 1 overlong entry.
xpar: 733 cells damaged, 733 copied, 0 decoded; 2 writes, 3000000 bytes; 1 entry repaired (0 further names share a repaired inode).
% cmp data.bin data.keep && echo identical
identical
```

The format can also be extracted without error correction using standard UNIX
commands, so archives remain accessible without xpar.  What the recipe yields
depends on the layout: with `--layout=armoured` it is the protected data
itself, while in a sidecar or split set the data is the original files and
the recipe extracts the critical metadata group instead.

```
% xpar explain p.xpa
p.xpa is a packet-bearing xpar volume.

The protected data is not in here: in the sidecar and split layouts the
original files are the data, and they are never rewritten or armoured.
What is armoured is the critical metadata group, one ARMG packet holding
the set descriptor, the manifest and the slice checksums.
The recipe below recovers the first ARMG packet's plaintext, which begins
with "XPAR2PKT". That packet is at file offset 72 and its payload
begins at 168.

  code             RS(255, 223), t = 16 over GF(2^8)
  interleave D     1
  frame            255 bytes on disk, 223 of plaintext
  frames           17

# xpar hand-recovery recipe for p.xpa
# this extracts the armoured critical metadata group
[...]
set -e
in=p.xpa
out=recovered.bin
W=1; n=255; k=223; D=1; hdr=168
Fd=$((D*k*W))          # plaintext bytes per frame = 223
Fx=$((D*n*W))          # disk bytes per frame      = 255
frames=17
off=0               # stream_offset from the prologue
len=3728               # stream_length from the prologue

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
| **Erasure unit finer than a block** | **yes (cells of 4 KiB to 64 KiB; the erasure unit is a cell, not a slice)** | no | no | no | no |
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
  are still required for operation, with a `superseded/R` column counting the
  slices nothing still reads against the recovery that generation carries.
  `prune` removes generations that are unreferenced.  `consolidate` collapses
  generational chains, merging all generations into one, and re-encodes the
  archive.
  A generation cannot check or rebuild the bytes it inherits, so on a set of
  more than one generation `verify`, `repair` and `scrub` walk the whole
  ancestry oldest first unless `--generation=G` picks one; `--chain` asks for
  that walk explicitly and is now the default.  Damage that falls in
  inherited bytes is reported as such, and slices an ancestor no longer owns
  count as erasures for the decoder: they can exhaust that ancestor's
  recovery, and `verify` says so, but with nothing damaged there is nothing
  to decode and the verdict stays clean.  Omitting `-r` makes `add` inherit
  the generation it extends and `consolidate` the widest ratio in the chain,
  and `add` warns when a changed file spends an ancestor's recovery budget.
  Under `--rescan=hash` an entry whose content still matches inherits its
  bytes, so a metadata-only change adds no stream data.
  A generation whose volumes are on disk but whose descriptor survives
  nowhere, not even in the replicas its own recovery volumes carry, is
  reported as damaged rather than dropped; `consolidate` and `prune` refuse
  to rewrite a chain around it.
```
% xpar add -r 15% photos.xpa -R pics
xpar: generation 1, set 09817e407913ea82a05c5cf7ebf32144: 10 entries (1 added, 2 changed, 7 inherited, 0 dropped), 500000 new stream bytes, 18 recovery slices in 5 volumes.
xpar: warning: redundancy falls from 15.0% to 14.6%; pass -r to keep the old ratio.
xpar: warning: generation 0: 74 of its 513 slices are superseded and count as erasures; only 3 of 77 recovery slices remain for its 7 inherited entries.

% xpar verify photos.xpa
xpar: pics/IMG0002.jpg: superseded
xpar: 513 slices of 4096 bytes, 77 recovery slices, erasure unit cell of 4096 bytes (1 per slice)
xpar: coverage: tree (9 entries)
xpar: superseded: 1 entry excluded from this generation's verdict
xpar: 74 slices of generation 0 are superseded by a later generation and count as erasures; consolidate to restore full protection
xpar: status: clean
xpar: 123 slices of 4096 bytes, 18 recovery slices, erasure unit cell of 4096 bytes (1 per slice)
xpar: coverage: tree (10 entries)
xpar: status: clean
```
- `recover`: regenerate lost volume(s) from the surviving set.  For example,
  when with `--layout=split --volumes=4`, the data lives in header-free
  `.d00` - `.d03` files, one per disk, then:
```
% xpar create --layout=split --volumes=4 -r 30% -o disc vid.mp4
xpar: disc: 1 entry, 1221 slices of 4096 bytes, 366 recovery slices in 4 volumes
% rm disc.d01
% xpar recover --volume=disc.d01 disc.xpa
xpar: recovered disc.d01 from survivor and parity slices (1249280 bare stream bytes).
% cat disc.d0* > joined && cmp joined vid.mp4 && echo identical
identical
```
  R has to cover the slices of the lost volume: one of four volumes is a
  quarter of the set, so a lower `-r` cannot rebuild it, and the refusal
  names the shortfall.
```
% xpar create --layout=split --volumes=4 -r 15% -o disc vid.mp4
xpar: disc: 1 entry, 1221 slices of 4096 bytes, 183 recovery slices in 4 volumes
% rm disc.d01
% xpar recover --volume=disc.d01 disc.xpa
xpar: Volume 'disc.d01' cannot be reconstructed from the surviving data and recovery slices: too few recovery slices: 305 needed, 183 exist, 122 short.
```
- `--auth-key=FILE`: authenticate the set with a keyed MAC, as a precaution
  against doctored archives.  Every subsequent operation requires the key,
  `explain` included, and a missing or wrong key exits 6:
```
% xpar verify s.xpa
xpar: This set is authenticated; supply --auth-key=FILE.
% xpar verify --auth-key=wrong.bin s.xpa
xpar: The authentication key is wrong for this set.
```
  Authentication only protects a set when the verifier always supplies the
  key.  An attacker without the key cannot forge a MAC, but can strip the
  AUTH packets and re-tag the whole set, and a keyless `verify` will then
  report that unauthenticated set clean.  Nothing in an archive can force a
  verifier to ask for a key it never passes.  `--auth-only`, which drops the
  public CRCs and whole-file hashes, works at any set size.
- `--memory`: sets the maximum working set size for the planner.
```
% xpar create -m 1M -r 20% -o tiny data.bin
xpar: No plan fits: raise -m to 2.0 MiB; no -b fits this -m; --codec=matrix does not fit either at -m 1.0 MiB.
```
- `info`: displays information about the erasure and error correction
  data type, along with the creator string, any comment packets, and, for
  `--layout=armoured`, the armour overhead the archive actually carries on
  disk rather than the code's nominal rate.
```
% xpar info b.xpa
  set        : f660c2f96605d7a0337b69477b9a8b65
  format     : 2.0, layout sidecar
  generation : 0 of 1 (the newest)
  geometry   : Z = 2097152 (2.0 MiB), S = 48, L = 100000000 (95.4 MiB)
               stream base 0, 1 entries
  cells      : Y = 65536 bytes, K = 32 per slice; the erasure unit is
               (slice, column), not a whole slice
  codec      : matrix over GF(2^8), recovery axis 2^8 = 208 slices
  redundancy : R = 5 (10.4% of S), 5 recovery slices present
  tags       : CRC32C per slice, BLAKE3 strong tag of 8 bytes
  dedup      : level 0 (none)
  armour     : GF(2^8) RS(255, 223), t = 16, D = 1
               level metadata: the critical packet group is armoured
               frame 255 bytes on disk carrying 223 of plaintext
               correctable burst 15 bytes anywhere in a frame
               code overhead 14.350%
               on disk 20520 bytes for 17836 of plaintext, overhead 15.048%
  ...
  creator    : xpar 2.0
  chain      : 1 generation
    gen 0    set f660c2f9...  95.4 MiB    S=48 R=5 (10.4%)  volumes 4  <- selected
  plan       : to repair this generation
  geometry   : L = 100,000,000  Z = 2,097,152  S = 48  R = 5
  field      : S + R = 53 <= 256, so GF(2^8)
  codec      : matrix  (GF(2^8), C = 2,097,152 B)
  memory     : work buffers 22.1 MiB;  read-ahead 0 B;  stage + hash 32.0 MiB
               total 54.1 MiB
  cells      : Y = 65,536 B, K = 32 per slice (last cell 65,536 B)
               erasure budget is 5 per column, not 5 per set
               SLCL = 6,144 B
  passes     : 1 sequential read totalling 100,000,000 bytes
```
- `benchmark`: measures xpar's low-level SIMD kernels.
```
% xpar benchmark --tiers
xpar: benchmark: V-HASH, V-CRC and V-GEN KATs ok
xpar: benchmark: tier         operation         bytes     time       rate
xpar: benchmark: gfni256      gf8-mac         8388608 bytes      187 us  42780.75 MiB/s
xpar: benchmark: gf tier gfni256  ok
xpar: benchmark: gfni512      gf8-mac         8388608 bytes      194 us  41237.11 MiB/s
xpar: benchmark: gf tier gfni512  ok
xpar: benchmark: ...
xpar: benchmark: gfni256      armour-gf8       285440 bytes     1133 us    240.26 MiB/s
xpar: benchmark: gfni256      armour-gf16     8616960 bytes     8773 us    936.71 MiB/s
xpar: benchmark: armour tier gfni256  ok
xpar: benchmark: ...
xpar: benchmark: crc32c sse4.2, blake3 avx2
xpar: benchmark: 9 tiers agree with scalar
```

All commands use these exit codes/ERRORLEVELs:
- 0: No error.
- 1: File damaged but repairable.
- 2: File hopelessly damaged.
- 3: Not found / Not an xpar set / Wrong version.
- 4: Wrong usage.
- 5: I/O error.
- 6: Authentication error.
- 7: No feasible plan under `-m`, or out of memory.
- 8: Internal error.

## Binary layouts

xpar supports three layouts:

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
  concatenation.  Here `--armour=all` is the only level: the archive is one
  protected region.

## Performance

Single-core performance is comparable to par2cmdline-turbo and ParPar. Parallel
workloads are less optimized.

xpar reads output back by default to verify the write. Use `--no-verify-after`
to disable this check.

## Portability

Specialised kernels exist for x86's extensions SSSE3 / SSE4.2 / AVX2 / GFNI
/ GFNI-512 / VBMI, ARM NEON / PMULL / SVE, PowerPC VSX, and
RISC-V vector (both shuffle and clmul paths).  The binary format is
platform-agnostic.

Windows has two targets. The default `nt` target uses UTF-16, extended-length
paths, and Vista or later. Build it with:

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
