#!/bin/sh
#  Copyright (C) 2022-2026 Kamila Szewczyk
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; version 3 of the License only.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program. If not, see <http://www.gnu.org/licenses/>.

# Measure recovery limits, costs and matched-redundancy competitors.

set -e

here=`cd \`dirname "$0"\` && pwd`
top=`cd "$here/.." && pwd`
. "$here/lib.sh"

out=bench-results
size=1073741824
reps=3
seed=20260823
jobs=
cold=none
xpar=${XPAR:-}
competitors=
keep=
which=all

usage() {
  cat <<'EOF'
usage: experiments.sh [options] [experiment...]

Experiments:
  envelope      recovery outcome by maximum column depth
  scatter       concentrated versus scattered faults
  cellsize      cell size versus metadata and repair cost
  amplify       I/O per repaired byte
  scaling       throughput and memory by corpus size
  tree          mixed-file workload with each deduplication mode
  baseline      xpar and PAR tools at matched redundancy
  all           run all experiments (default)

Options:
  --out DIR          result directory (default: bench-results)
  --xpar PATH        xpar binary
  --competitors FILE path to competitors.json
  --size BYTES       corpus bytes for fixed-size experiments
  --reps N           repetitions per measurement (default: 3)
  --seed N           corpus seed
  --jobs N           xpar worker threads
  --cold WHICH       cache mode: none (default) or drop
  --keep             preserve generated corpora and sets
  -h, --help         show this help
EOF
  exit 0
}

while test $# -gt 0; do
  case $1 in
    --out) out=$2;  shift 2 ;;
    --xpar) xpar=$2;  shift 2 ;;
    --competitors) competitors=$2;  shift 2 ;;
    --size) size=$2;  shift 2 ;;
    --reps) reps=$2;  shift 2 ;;
    --seed) seed=$2;  shift 2 ;;
    --jobs) jobs=$2;  shift 2 ;;
    --cold) cold=$2;  shift 2 ;;
    --keep) keep=1;  shift ;;
    -h|--help) usage ;;
    -*) die "unknown option: $1 (try --help)" ;;
    *) which="$which $1";  shift ;;
  esac
done
case $which in "all "*) which=`echo "$which" | sed 's/^all //'` ;; esac
test -n "$which" || which=all

bench_find_tools
bench_open_output
bench_probe_cold
bench_environment
test -n "$keep" || trap 'rm -rf "$work"' EXIT HUP INT TERM

say "output: $out"

par2turbo=;  par2ref=;  par3bin=;  parparjs=;  nodebin=

tool_binary() {
  grep '"name":"'"$1"'"' "$competitors" 2>/dev/null |
    grep '"status":"ok"' |
    sed -n 's/.*"binary":"\([^"]*\)".*/\1/p' | head -1
}

if test -n "$competitors" && test -r "$competitors"; then
  cp "$competitors" "$out/competitors.json"
  par2turbo=`tool_binary par2cmdline-turbo`
  par2ref=`tool_binary par2cmdline`
  par3bin=`tool_binary par3cmdline`
  _pp=`tool_binary parpar`
  nodebin=`echo "$_pp" | awk '{print $1}'`
  parparjs=`echo "$_pp" | awk '{print $2}'`
  say "baselines: par2-turbo=${par2turbo:-none} par2=${par2ref:-none}"
  say "baselines: par3=${par3bin:-none} ParPar=${parparjs:-none}"
fi

runs() { case " $which " in *" $1 "*|*" all "*) return 0 ;; esac;  return 1; }

ops_column_depth() {   # <depth> <columns>: `depth` cells in each column
  damage_ops=
  _j=0
  while test "$_j" -lt "$2" && test "$_j" -lt "$g_k"; do
    _i=0
    while test "$_i" -lt "$1"; do
      damage_ops="$damage_ops cell=$_i,$_j"
      _i=$((_i + 1))
    done
    _j=$((_j + 1))
  done
}

ops_concentrated() {   # <cells>: all in column 0, so depth is the count
  damage_ops=
  _i=0
  while test "$_i" -lt "$1" && test "$_i" -lt "$g_s"; do
    damage_ops="$damage_ops cell=$_i,0"
    _i=$((_i + 1))
  done
}

ops_spread() {   # <cells>: one per slice, walking columns, so depth stays low
  damage_ops=
  _i=0
  while test "$_i" -lt "$1" && test "$_i" -lt "$g_s"; do
    damage_ops="$damage_ops cell=$_i,$((_i % g_k))"
    _i=$((_i + 1))
  done
}

ops_offsets() {
  offsets=
  par_ops=
  for _op in $damage_ops; do
    _sc=${_op#cell=}
    _s=${_sc%%,*}
    _c=${_sc#*,}
    _o=$((_s * g_z + _c * g_y))
    offsets="$offsets $_o"
    par_ops="$par_ops rand=$_o,96"
  done
}

setup_create() { rm -f "$sdir"/set*.xpa; }

check_create() {
  test -f "$sdir/set.xpa" || { sig=no-set;  return 1; }
  split_archive "$sdir/set" "$sdir/set.xpa"
  f_archive_bytes=$archive_total;  f_payload_bytes=$payload_total
  f_meta_bytes=$meta_total
  read_geometry "$sdir/set.xpa"
  f_slice_size=$g_z;  f_cell_bytes=$g_y;  f_slices=$g_s
  f_recovery_slices=$g_r
  sig="archive=$f_archive_bytes slices=$g_s recovery=$g_r"
}

check_verify() {
  f_scan_bytes=`jnum0 "$work/out.json" bytes_read summary`
  sig="read=$f_scan_bytes"
}

setup_repair() {
  cp "$corpus" "$sdir/data.bin"
  # shellcheck disable=SC2086
  "$damage" "$sdir/data.bin" -Z "$g_z" -Y "$g_y" -n 96 seed=$seed \
    $damage_ops || return 1
  "$xpar" verify --json "$sdir/set.xpa" > "$work/pre.json" 2>/dev/null || true
  f_damaged_cells=`jnum0 "$work/pre.json" cells_bad summary`
  f_damaged_slices=`jnum0 "$work/pre.json" slices_bad summary`
  f_column_depth=`jnum0 "$work/pre.json" column_depth summary`
  f_column_groups=`jnum0 "$work/pre.json" column_groups summary`
  _rd=`jnum0 "$work/pre.json" bytes_read summary`
  f_scan_bytes=$((_rd + f_archive_bytes))
}

# A refused repair must leave the damaged input unchanged.
check_repair() {
  f_repaired_bytes=`jnum0 "$work/out.json" bytes_written repair`
  if test "$f_expect" -eq 0; then
    cmp -s "$sdir/data.bin" "$corpus" || { sig=not-repaired;  return 1; }
    sig="cells=$f_damaged_cells depth=$f_column_depth wrote=$f_repaired_bytes"
  else
    if cmp -s "$sdir/data.bin" "$corpus"; then
      sig=unexpectedly-repaired;  return 1
    fi
    sig="refused cells=$f_damaged_cells depth=$f_column_depth"
  fi
}

make_set() {   # make_set <dir> <create args...>
  sdir=$1;  shift
  rm -rf "$sdir";  mkdir -p "$sdir"
  cp "$corpus" "$sdir/data.bin"
  "$xpar" create --reproducible --dedup=none --align=none --no-verify-after \
    --json "$@" -o "$sdir/set" "$sdir/data.bin" > /dev/null 2>&1 || return 1
  read_geometry "$sdir/set.xpa"
  f_archive_bytes=`archive_bytes "$sdir/set"`
}

corpus=$work/corpus.bin
make_corpus() {   # make_corpus <bytes>
  test -s "$corpus" && test "`file_bytes "$corpus"`" = "$1" && return 0
  say "generating corpus: $1 bytes, seed $seed"
  "$mkdata" "$seed" "$1" "$corpus" --pattern=random
}

exp_envelope() {
  say "experiment: envelope"
  make_corpus "$size"
  make_set "$work/env" -s 1M --cell=65536 --codec=matrix --field=16 -r 5% ||
    { warn "envelope: set creation failed";  return 0; }
  say "geometry: Z=$g_z Y=$g_y K=$g_k S=$g_s R=$g_r"

  # Sample powers of two and the recovery boundary.
  depths=
  d=1
  while test "$d" -lt "$g_r"; do
    depths="$depths $d"
    d=$((d * 2))
  done
  test "$g_r" -lt 2 || depths="$depths $((g_r - 1))"
  depths="$depths $g_r $((g_r + 1)) $((g_r + 2))"
  say "column depths:$depths"

  for width in 1 4 16; do
    test "$width" -le "$g_k" || continue
    for d in $depths; do
      test "$d" -le "$g_s" || continue
      ops_column_depth "$d" "$width"
      test -n "$damage_ops" || continue
      reset_row
      f_experiment=envelope;  f_op=repair;  f_codec=$g_codec
      f_field=$g_field;  f_layout=sidecar;  f_corpus=random
      f_corpus_bytes=$size;  f_slice_size=$g_z;  f_cell_bytes=$g_y
      f_slices=$g_s;  f_recovery_slices=$g_r;  f_recovery_spec=5%
      f_archive_bytes=`archive_bytes "$work/env/set"`
      f_damage="depth$d-width$width"
      f_note="width=$width"
      if test "$d" -le "$g_r"; then f_expect=0;  else f_expect=2; fi
      bench_measure setup_repair check_repair \
        "$xpar" repair --in-place --no-journal --json "$sdir/set.xpa"
    done
  done
}

par_repair_argv() {   # <kind> <bin> <dir>: prints the argv to time
  case $1 in
    par2) echo "$2 repair -q -- $3/set.par2" ;;
    par3) echo "$2 r -q $3/set.par3" ;;
  esac
}

# Sample powers of two up to scattered-fault capacity.
# Powers of two to see the shape, plus the three points either side of R
# and of R*K, because a claim about where a format stops needs the point
# where it stopped and not the octave that contains it.
scatter_points() {   # <slices> <recovery> <columns>
  _s=$1;  _r=$2;  _k=$3
  _cap=$(( _r * _k ))
  test "$_cap" -le "$_s" || _cap=$_s
  _pts=
  _n=1
  while test "$_n" -le "$_cap"; do
    _pts="$_pts $_n"
    _n=$((_n * 2))
  done
  for _b in $((_r - 1)) $_r $((_r + 1)) $((_cap - 1)) $_cap; do
    test "$_b" -ge 1 && test "$_b" -le "$_s" && _pts="$_pts $_b"
  done
  scatter_ns=`for _p in $_pts; do echo "$_p"; done | sort -n -u | tr '\n' ' '`
}

exp_scatter() {
  say "experiment: scatter"
  make_corpus "$size"

  sdir=$work/scat
  rm -rf "$sdir";  mkdir -p "$sdir"
  cp "$corpus" "$sdir/data.bin"
  reset_row
  f_experiment=scatter;  f_op=create;  f_tool=xpar;  f_codec=matrix
  f_field=16;  f_layout=sidecar;  f_corpus=random;  f_corpus_bytes=$size
  f_slice_size=1048576;  f_cell_bytes=65536;  f_recovery_spec=5%
  f_expect=0;  f_note="block=1048576 cell=65536"
  bench_measure setup_create check_create \
    "$xpar" create --reproducible --dedup=none --align=none \
      --no-verify-after --codec=matrix --field=16 -r 5% -s 1M \
      --cell=65536 --json -o "$sdir/set" "$sdir/data.bin" || true
  test -f "$sdir/set.xpa" ||
    { warn "scatter: set creation failed";  return 0; }
  read_geometry "$sdir/set.xpa"
  say "geometry: Z=$g_z Y=$g_y K=$g_k S=$g_s R=$g_r"
  scat_z=$g_z;  scat_y=$g_y;  scat_k=$g_k;  scat_s=$g_s;  scat_r=$g_r
  scat_arch=`archive_bytes "$work/scat/set"`
  scatter_points "$g_s" "$g_r" "$g_k"
  say "fault counts:$scatter_ns"

  for n in $scatter_ns; do
    for shape in concentrated spread; do
      g_z=$scat_z;  g_y=$scat_y;  g_k=$scat_k;  g_s=$scat_s
      if test "$shape" = concentrated; then ops_concentrated "$n"
      else ops_spread "$n"; fi
      test -n "$damage_ops" || continue
      reset_row
      f_experiment=scatter;  f_op=repair;  f_tool=xpar;  f_codec=matrix
      f_field=16;  f_layout=sidecar;  f_corpus=random
      f_corpus_bytes=$size;  f_slice_size=$scat_z;  f_cell_bytes=$scat_y
      f_slices=$scat_s;  f_recovery_slices=$scat_r;  f_recovery_spec=5%
      f_archive_bytes=$scat_arch;  f_damage=$shape;  f_note="n=$n"
      if test "$shape" = concentrated; then
        test "$n" -le "$scat_r" && f_expect=0 || f_expect=2
      else
        _d=$(( (n + scat_k - 1) / scat_k ))
        test "$_d" -le "$scat_r" && f_expect=0 || f_expect=2
      fi
      sdir=$work/scat
      bench_measure setup_repair check_repair \
        "$xpar" repair --in-place --no-journal --json "$sdir/set.xpa"
    done
  done

  # Compare both matched block size and matched erasure granularity.
  test -z "$par2turbo" ||
    exp_scatter_par par2 par2cmdline-turbo "$par2turbo" "$scat_z"
  test -z "$par2turbo" ||
    exp_scatter_par par2 par2cmdline-turbo-fine "$par2turbo" "$scat_y"
  test -z "$par3bin" ||
    exp_scatter_par par3 par3cmdline "$par3bin" "$scat_z"
}

exp_scatter_par() {   # <kind> <label> <binary> <blocksize>
  _kind=$1;  _lab=$2;  _bin=$3;  _bs=$4
  say "baseline: $_lab, block $_bs"
  pdir=$work/scat-$_lab
  rm -rf "$pdir";  mkdir -p "$pdir"
  cp "$corpus" "$pdir/data.bin"

  setup_par_create() {
    rm -f "$pdir"/set.par2 "$pdir"/set.vol*.par2 "$pdir"/set*.par3
  }
  check_par_create() {
    split_archive "$pdir/set" "$pdir/set.$_kind"
    par_archive=$archive_total
    f_archive_bytes=$archive_total;  f_payload_bytes=$payload_total
    f_meta_bytes=$meta_total
    test "$par_archive" -gt 0 || { sig=no-set;  return 1; }
    sig="archive=$par_archive payload=$payload_total"
  }
  reset_row
  f_experiment=scatter;  f_op=create;  f_tool=$_lab;  f_layout=$_kind
  f_corpus=random;  f_corpus_bytes=$size;  f_slice_size=$_bs
  f_recovery_spec=5%;  f_expect=-1;  f_note="block=$_bs"
  case $_kind in
    par2) bench_measure setup_par_create check_par_create \
            "$_bin" create -q -s"$_bs" -r5 -- "$pdir/set.par2" \
            "$pdir/data.bin" ;;
    par3) bench_measure setup_par_create check_par_create \
            "$_bin" c -q -s"$_bs" -r5 "$pdir/set.par3" "$pdir/data.bin" ;;
  esac
  par_archive=`archive_bytes "$pdir/set"`
  if test "$par_archive" -eq 0; then
    warn "$_lab: no archive created with block size $_bs"
    rm -rf "$pdir"
    return 0
  fi
  par_argv=`par_repair_argv "$_kind" "$_bin" "$pdir"`

  setup_par() {
    cp "$corpus" "$pdir/data.bin"
    # shellcheck disable=SC2086
    "$damage" "$pdir/data.bin" $par_ops || return 1
  }
  # Judge foreign repairs by output bytes, not exit conventions.
  check_par() {
    if cmp -s "$pdir/data.bin" "$corpus"; then
      f_note="n=$f_damaged_cells recovered";  sig="recovered"
    else
      f_note="n=$f_damaged_cells lost";  sig="lost"
    fi
  }

  for n in $scatter_ns; do
    for shape in concentrated spread; do
      g_z=$scat_z;  g_y=$scat_y;  g_k=$scat_k;  g_s=$scat_s
      if test "$shape" = concentrated; then ops_concentrated "$n"
      else ops_spread "$n"; fi
      test -n "$damage_ops" || continue
      ops_offsets
      reset_row
      f_experiment=scatter;  f_op=repair;  f_tool=$_lab;  f_layout=$_kind
      f_corpus=random;  f_corpus_bytes=$size;  f_slice_size=$_bs
      f_recovery_spec=5%;  f_archive_bytes=$par_archive;  f_damage=$shape
      f_note="n=$n";  f_damaged_cells=$n;  f_damaged_slices=$n
      f_column_depth=$n;  f_scan_bytes=$((size + par_archive))
      f_expect=-1
      # shellcheck disable=SC2086
      bench_measure setup_par check_par $par_argv
    done
  done
}

exp_cellsize() {
  say "experiment: cellsize"
  make_corpus "$size"
  for y in 4096 8192 16384 65536 262144 1048576; do
    cdir=$work/cell$y
    rm -rf "$cdir";  mkdir -p "$cdir"
    cp "$corpus" "$cdir/data.bin"
    reset_row
    f_experiment=cellsize;  f_op=create;  f_codec=matrix;  f_field=16
    f_layout=sidecar;  f_corpus=random;  f_corpus_bytes=$size
    f_slice_size=1048576;  f_cell_bytes=$y;  f_recovery_spec=5%
    f_expect=0;  f_note="Y=$y"
    sdir=$cdir
    bench_measure setup_create check_create \
      "$xpar" create --reproducible --dedup=none --align=none \
        --no-verify-after --codec=matrix --field=16 -r 5% -s 1M \
        --cell="$y" --json -o "$cdir/set" "$cdir/data.bin" || true
    test -f "$cdir/set.xpa" || { rm -rf "$cdir";  continue; }
    read_geometry "$cdir/set.xpa"
    f_archive_bytes=`archive_bytes "$cdir/set"`
    say "geometry: Y=$g_y K=$g_k; archive=$f_archive_bytes bytes"

    reset_row
    f_experiment=cellsize;  f_op=verify;  f_codec=$g_codec;  f_field=$g_field
    f_layout=sidecar;  f_corpus=random;  f_corpus_bytes=$size
    f_slice_size=$g_z;  f_cell_bytes=$g_y;  f_slices=$g_s
    f_recovery_slices=$g_r;  f_recovery_spec=5%
    f_archive_bytes=`archive_bytes "$cdir/set"`;  f_expect=0;  f_note="Y=$g_y"
    bench_measure setup_none check_verify \
      "$xpar" verify --json "$cdir/set.xpa"

    # Keep fault count fixed so repaired bytes scale with Y.
    ops_spread 64
    reset_row
    f_experiment=cellsize;  f_op=repair;  f_codec=$g_codec;  f_field=$g_field
    f_layout=sidecar;  f_corpus=random;  f_corpus_bytes=$size
    f_slice_size=$g_z;  f_cell_bytes=$g_y;  f_slices=$g_s
    f_recovery_slices=$g_r;  f_recovery_spec=5%
    f_archive_bytes=`archive_bytes "$cdir/set"`
    f_damage=spread;  f_expect=0;  f_note="Y=$g_y"
    sdir=$cdir
    bench_measure setup_repair check_repair \
      "$xpar" repair --in-place --no-journal --json "$cdir/set.xpa"
    rm -rf "$cdir"
  done
}

exp_amplify() {
  say "experiment: amplify"
  make_corpus "$size"
  make_set "$work/amp" -s 1M --cell=65536 --codec=matrix --field=16 -r 5% ||
    { warn "amplify: set creation failed";  return 0; }
  say "geometry: Z=$g_z Y=$g_y K=$g_k S=$g_s R=$g_r"
  for n in 1 2 8 32 128 512; do
    test "$n" -le "$g_s" || break
    ops_spread "$n"
    test -n "$damage_ops" || continue
    reset_row
    f_experiment=amplify;  f_op=repair;  f_codec=$g_codec;  f_field=$g_field
    f_layout=sidecar;  f_corpus=random;  f_corpus_bytes=$size
    f_slice_size=$g_z;  f_cell_bytes=$g_y;  f_slices=$g_s
    f_recovery_slices=$g_r;  f_recovery_spec=5%
    f_archive_bytes=`archive_bytes "$work/amp/set"`
    f_damage=spread;  f_expect=0;  f_note="cells=$n"
    bench_measure setup_repair check_repair \
      "$xpar" repair --in-place --no-journal --json "$sdir/set.xpa"
  done
}

exp_scaling() {
  say "experiment: scaling"
  # Compare fixed 1 MiB slices with planner-selected geometry.
  for geom in fixed auto; do
    if test "$geom" = fixed; then _sflag="-s 1M";  else _sflag=; fi
    exp_scaling_arm "$geom"
  done
  corpus=$work/corpus.bin
}

exp_scaling_arm() {
  _geom=$1
  for n in 268435456 1073741824 4294967296 17179869184; do
    _avail=`df -Pk "$work" 2>/dev/null | awk 'NR==2 {print $4 * 1024}'`
    if test -n "$_avail" && test "$_avail" -lt $((n * 3)); then
      warn "scaling: skipping $n-byte corpus (insufficient disk space)"
      continue
    fi
    corpus=$work/corpus-$n.bin
    make_corpus "$n"
    sdir=$work/scale-$_geom-$n
    rm -rf "$sdir";  mkdir -p "$sdir"
    cp "$corpus" "$sdir/data.bin"

    reset_row
    f_experiment=scaling;  f_op=create;  f_codec=matrix;  f_field=16
    f_layout=sidecar;  f_corpus=random;  f_corpus_bytes=$n
    f_recovery_spec=5%;  f_expect=0;  f_note="$_geom"
    # shellcheck disable=SC2086
    bench_measure setup_create check_create \
      "$xpar" create --reproducible --dedup=none --align=none \
        --no-verify-after --codec=matrix --field=16 -r 5% $_sflag \
        --cell=65536 --json -o "$sdir/set" "$sdir/data.bin" || true
    test -f "$sdir/set.xpa" || { rm -rf "$sdir" "$corpus";  continue; }
    read_geometry "$sdir/set.xpa"
    f_archive_bytes=`archive_bytes "$sdir/set"`

    reset_row
    f_experiment=scaling;  f_op=verify;  f_codec=$g_codec;  f_field=$g_field
    f_layout=sidecar;  f_corpus=random;  f_corpus_bytes=$n
    f_slice_size=$g_z;  f_cell_bytes=$g_y;  f_slices=$g_s
    f_recovery_slices=$g_r;  f_recovery_spec=5%;  f_expect=0
    f_archive_bytes=`archive_bytes "$sdir/set"`;  f_note="$_geom"
    bench_measure setup_none check_verify \
      "$xpar" verify --json "$sdir/set.xpa"

    ops_spread 64
    reset_row
    f_experiment=scaling;  f_op=repair;  f_codec=$g_codec;  f_field=$g_field
    f_layout=sidecar;  f_corpus=random;  f_corpus_bytes=$n
    f_slice_size=$g_z;  f_cell_bytes=$g_y;  f_slices=$g_s
    f_recovery_slices=$g_r;  f_recovery_spec=5%
    f_archive_bytes=`archive_bytes "$sdir/set"`
    f_damage=spread;  f_expect=0;  f_note="$_geom"
    bench_measure setup_repair check_repair \
      "$xpar" repair --in-place --no-journal --json "$sdir/set.xpa"

    rm -rf "$sdir" "$corpus"
  done
}

exp_tree() {
  say "experiment: tree"
  tdir=$work/tree
  test -d "$tdir" || sh "$here/mktree.sh" "$seed" "$size" "$tdir" ||
    { warn "tree: corpus generation failed";  return 0; }
  tfiles=`find "$tdir" -type f | wc -l | tr -d ' '`
  tbytes=`find "$tdir" -type f -printf '%s\n' 2>/dev/null |
          awk '{s+=$1} END {print s+0}'`
  say "tree corpus: $tfiles files, $tbytes bytes"
  rm -rf "$work/tree.orig"
  cp -a "$tdir" "$work/tree.orig"

  # Place the sidecar set beside the tree it names.
  setup_tree_create() {
    rm -f "$tbase"*.xpa
    rm -rf "$tdir"
    cp -a "$work/tree.orig" "$tdir"
  }
  check_tree_create() {
    test -f "$tbase.xpa" || { sig=no-set;  return 1; }
    split_archive "$tbase" "$tbase.xpa"
    f_archive_bytes=$archive_total;  f_payload_bytes=$payload_total
    f_meta_bytes=$meta_total
    read_geometry "$tbase.xpa"
    f_slice_size=$g_z;  f_cell_bytes=$g_y;  f_slices=$g_s
    f_recovery_slices=$g_r
    sig="archive=$archive_total slices=$g_s recovery=$g_r"
  }
  setup_tree_repair() {
    rm -rf "$tdir"
    cp -a "$work/tree.orig" "$tdir"
    _n=0
    for _f in `find "$tdir" -type f -size +200k | sort | head -24`; do
      "$damage" "$_f" "rand=1024,96" || return 1
      _n=$((_n + 1))
    done
    test "$_n" -gt 0 || return 1
    "$xpar" verify --json "$tbase.xpa" > "$work/pre.json" 2>/dev/null || true
    f_damaged_cells=`jnum0 "$work/pre.json" cells_bad summary`
    f_damaged_slices=`jnum0 "$work/pre.json" slices_bad summary`
    f_column_depth=`jnum0 "$work/pre.json" column_depth summary`
    f_column_groups=`jnum0 "$work/pre.json" column_groups summary`
    f_scan_bytes=`jnum0 "$work/pre.json" bytes_read summary`
    # Reject ineffective or unexpectedly broad damage.
    test "$f_damaged_cells" -gt 0 || return 1
    test "$f_damaged_cells" -lt 200 || return 1
  }
  check_tree_repair() {
    diff -r -q "$tdir" "$work/tree.orig" > /dev/null 2>&1 ||
      { sig=not-repaired;  return 1; }
    f_repaired_bytes=`jnum0 "$work/out.json" bytes_written repair`
    sig="cells=$f_damaged_cells depth=$f_column_depth"
  }

  for dd in none file chunk; do
    tbase=$work/set-$dd
    reset_row
    f_experiment=tree;  f_op=create;  f_codec=matrix;  f_field=16
    f_layout=sidecar;  f_corpus="tree-$tfiles-files";  f_corpus_bytes=$tbytes
    f_slice_size=1048576;  f_recovery_spec=5%;  f_expect=0;  f_note="dedup=$dd"
    bench_measure setup_tree_create check_tree_create \
      "$xpar" create -R --reproducible --dedup="$dd" --no-verify-after \
        --codec=matrix --field=16 -r 5% -s 1M --cell=65536 --json \
        -o "$tbase" "$tdir" || true
    test -f "$tbase.xpa" ||
      { warn "tree: set creation failed with dedup=$dd";  continue; }
    read_geometry "$tbase.xpa"
    _arch=`archive_bytes "$tbase"`

    reset_row
    f_experiment=tree;  f_op=verify;  f_codec=$g_codec;  f_field=$g_field
    f_layout=sidecar;  f_corpus="tree-$tfiles-files";  f_corpus_bytes=$tbytes
    f_slice_size=$g_z;  f_cell_bytes=$g_y;  f_slices=$g_s
    f_recovery_slices=$g_r;  f_recovery_spec=5%;  f_expect=0
    f_archive_bytes=$_arch;  f_note="dedup=$dd"
    bench_measure setup_none check_verify \
      "$xpar" verify --json "$tbase.xpa"

    reset_row
    f_experiment=tree;  f_op=repair;  f_codec=$g_codec;  f_field=$g_field
    f_layout=sidecar;  f_corpus="tree-$tfiles-files";  f_corpus_bytes=$tbytes
    f_slice_size=$g_z;  f_cell_bytes=$g_y;  f_slices=$g_s
    f_recovery_slices=$g_r;  f_recovery_spec=5%;  f_expect=0
    f_archive_bytes=$_arch;  f_damage=entries;  f_note="dedup=$dd"
    bench_measure setup_tree_repair check_tree_repair \
      "$xpar" repair --in-place --no-journal --json "$tbase.xpa"
    rm -f "$tbase"*.xpa
  done
  rm -rf "$tdir" "$work/tree.orig"
}

base_one_xpar() {   # <codec> <field>
  sdir=$work/base-$1-$2
  rm -rf "$sdir";  mkdir -p "$sdir"
  cp "$corpus" "$sdir/data.bin"
  reset_row
  f_experiment=baseline;  f_op=create;  f_tool=xpar-$1;  f_codec=$1
  f_field=$2;  f_layout=sidecar;  f_corpus=random;  f_corpus_bytes=$size
  f_slice_size=1048576;  f_recovery_spec=5%;  f_expect=0
  bench_measure setup_create check_create \
    "$xpar" create --reproducible --dedup=none --align=none \
      --no-verify-after --codec="$1" --field="$2" -r 5% -s 1M \
      --cell=65536 --json -o "$sdir/set" "$sdir/data.bin" || true
  test -f "$sdir/set.xpa" || { rm -rf "$sdir";  return 0; }
  read_geometry "$sdir/set.xpa"
  _arch=`archive_bytes "$sdir/set"`

  reset_row
  f_experiment=baseline;  f_op=verify;  f_tool=xpar-$1;  f_codec=$1
  f_field=$2;  f_layout=sidecar;  f_corpus=random;  f_corpus_bytes=$size
  f_slice_size=$g_z;  f_cell_bytes=$g_y;  f_slices=$g_s
  f_recovery_slices=$g_r;  f_recovery_spec=5%;  f_archive_bytes=$_arch
  f_expect=0
  bench_measure setup_none check_verify "$xpar" verify --json "$sdir/set.xpa"

  ops_spread 32
  reset_row
  f_experiment=baseline;  f_op=repair;  f_tool=xpar-$1;  f_codec=$1
  f_field=$2;  f_layout=sidecar;  f_corpus=random;  f_corpus_bytes=$size
  f_slice_size=$g_z;  f_cell_bytes=$g_y;  f_slices=$g_s
  f_recovery_slices=$g_r;  f_recovery_spec=5%;  f_archive_bytes=$_arch
  f_damage=spread;  f_expect=0
  bench_measure setup_repair check_repair \
    "$xpar" repair --in-place --no-journal --json "$sdir/set.xpa"
  rm -rf "$sdir"
}

base_one_par2() {   # <label> <binary>
  _lab=$1;  _bin=$2
  pdir=$work/base-$_lab
  rm -rf "$pdir";  mkdir -p "$pdir"
  cp "$corpus" "$pdir/data.bin"

  setup_par2_create() { rm -f "$pdir"/set*.par2; }
  check_par2_create() {
    test -f "$pdir/set.par2" || { sig=no-set;  return 1; }
    split_archive "$pdir/set" "$pdir/set.par2"
    f_archive_bytes=$archive_total;  f_payload_bytes=$payload_total
    f_meta_bytes=$meta_total
    sig="archive=$archive_total payload=$payload_total"
  }
  reset_row
  f_experiment=baseline;  f_op=create;  f_tool=$_lab;  f_layout=par2
  f_corpus=random;  f_corpus_bytes=$size;  f_slice_size=1048576
  f_recovery_spec=5%;  f_expect=0
  bench_measure setup_par2_create check_par2_create \
    "$_bin" create -q -s1048576 -r5 -- "$pdir/set.par2" "$pdir/data.bin" ||
    true
  test -f "$pdir/set.par2" || { rm -rf "$pdir";  return 0; }
  _arch=`archive_bytes "$pdir/set"`

  reset_row
  f_experiment=baseline;  f_op=verify;  f_tool=$_lab;  f_layout=par2
  f_corpus=random;  f_corpus_bytes=$size;  f_slice_size=1048576
  f_recovery_spec=5%;  f_archive_bytes=$_arch;  f_expect=-1
  f_scan_bytes=$((size + _arch))
  bench_measure setup_none check_none "$_bin" verify -q -- "$pdir/set.par2"

  g_z=1048576;  g_y=1048576;  g_k=1;  g_s=$((size / 1048576))
  ops_spread 32
  ops_offsets
  setup_par2_repair() {
    cp "$corpus" "$pdir/data.bin"
    # shellcheck disable=SC2086
    "$damage" "$pdir/data.bin" $par_ops || return 1
  }
  check_par2_repair() {
    cmp -s "$pdir/data.bin" "$corpus" || { sig=not-repaired;  return 1; }
    sig="par2 blocks=32"
  }
  reset_row
  f_experiment=baseline;  f_op=repair;  f_tool=$_lab;  f_layout=par2
  f_corpus=random;  f_corpus_bytes=$size;  f_slice_size=1048576
  f_recovery_spec=5%;  f_archive_bytes=$_arch;  f_damage=spread
  f_damaged_cells=32;  f_damaged_slices=32;  f_column_depth=32
  f_repaired_bytes=$((32 * 1048576));  f_scan_bytes=$((size + _arch))
  f_expect=-1
  bench_measure setup_par2_repair check_par2_repair \
    "$_bin" repair -q -- "$pdir/set.par2"
  rm -rf "$pdir"
}

base_one_parpar() {
  test -n "$parparjs" || return 0
  pdir=$work/base-parpar
  rm -rf "$pdir";  mkdir -p "$pdir"
  cp "$corpus" "$pdir/data.bin"
  setup_pp() { rm -f "$pdir"/set*.par2; }
  check_pp() {
    test -f "$pdir/set.par2" || { sig=no-set;  return 1; }
    split_archive "$pdir/set" "$pdir/set.par2"
    f_archive_bytes=$archive_total;  f_payload_bytes=$payload_total
    f_meta_bytes=$meta_total
    sig="archive=$archive_total payload=$payload_total"
  }
  reset_row
  f_experiment=baseline;  f_op=create;  f_tool=parpar;  f_layout=par2
  f_corpus=random;  f_corpus_bytes=$size;  f_slice_size=1048576
  f_recovery_spec=5%;  f_expect=0;  f_note="encoder only"
  bench_measure setup_pp check_pp \
    "$nodebin" "$parparjs" -s 1048576b -r 5% -o "$pdir/set.par2" \
      "$pdir/data.bin" || true
  rm -rf "$pdir"
}

base_one_par3() {
  pdir=$work/base-par3
  rm -rf "$pdir";  mkdir -p "$pdir"
  cp "$corpus" "$pdir/data.bin"

  setup_par3_create() { rm -f "$pdir"/set*.par3; }
  check_par3_create() {
    test -f "$pdir/set.par3" || { sig=no-set;  return 1; }
    split_archive "$pdir/set" "$pdir/set.par3"
    f_archive_bytes=$archive_total;  f_payload_bytes=$payload_total
    f_meta_bytes=$meta_total
    sig="archive=$archive_total payload=$payload_total"
  }
  reset_row
  f_experiment=baseline;  f_op=create;  f_tool=par3cmdline;  f_layout=par3
  f_corpus=random;  f_corpus_bytes=$size;  f_slice_size=1048576
  f_recovery_spec=5%;  f_expect=0
  bench_measure setup_par3_create check_par3_create \
    "$par3bin" c -q -s1048576 -r5 "$pdir/set.par3" "$pdir/data.bin" || true
  test -f "$pdir/set.par3" || { rm -rf "$pdir";  return 0; }
  _arch=`archive_bytes "$pdir/set"`

  reset_row
  f_experiment=baseline;  f_op=verify;  f_tool=par3cmdline;  f_layout=par3
  f_corpus=random;  f_corpus_bytes=$size;  f_slice_size=1048576
  f_recovery_spec=5%;  f_archive_bytes=$_arch;  f_expect=-1
  f_scan_bytes=$((size + _arch))
  bench_measure setup_none check_none "$par3bin" v -q "$pdir/set.par3"

  g_z=1048576;  g_y=1048576;  g_k=1;  g_s=$((size / 1048576))
  ops_spread 32
  ops_offsets
  setup_par3_repair() {
    cp "$corpus" "$pdir/data.bin"
    # shellcheck disable=SC2086
    "$damage" "$pdir/data.bin" $par_ops || return 1
  }
  check_par3_repair() {
    cmp -s "$pdir/data.bin" "$corpus" || { sig=not-repaired;  return 1; }
    sig="par3 blocks=32"
  }
  reset_row
  f_experiment=baseline;  f_op=repair;  f_tool=par3cmdline;  f_layout=par3
  f_corpus=random;  f_corpus_bytes=$size;  f_slice_size=1048576
  f_recovery_spec=5%;  f_archive_bytes=$_arch;  f_damage=spread
  f_damaged_cells=32;  f_damaged_slices=32;  f_column_depth=32
  f_repaired_bytes=$((32 * 1048576));  f_scan_bytes=$((size + _arch))
  f_expect=-1
  bench_measure setup_par3_repair check_par3_repair \
    "$par3bin" r -q "$pdir/set.par3"
  rm -rf "$pdir"
}

exp_baseline() {
  say "experiment: baseline"
  make_corpus "$size"
  base_one_xpar matrix 16
  base_one_xpar fft 16
  test -z "$par2turbo" || base_one_par2 par2cmdline-turbo "$par2turbo"
  test -z "$par2ref"   || base_one_par2 par2cmdline "$par2ref"
  test -z "$par3bin"   || base_one_par3
  base_one_parpar
}

runs envelope && exp_envelope
runs scatter  && exp_scatter
runs cellsize && exp_cellsize
runs amplify  && exp_amplify
runs scaling  && exp_scaling
runs tree     && exp_tree
runs baseline && exp_baseline

echo
bench_finish
