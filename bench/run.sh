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

# Benchmark create, verify and repair across codec parameters.

set -e

here=`cd \`dirname "$0"\` && pwd`
top=`cd "$here/.." && pwd`
. "$here/lib.sh"

out=bench-results
size=268435456
reps=3
seed=20260823
jobs=
cold=none
matrix=default
xpar=${XPAR:-}
keep=

usage() {
  cat <<'EOF'
usage: run.sh [options]

  --out DIR        result directory (default: bench-results)
  --xpar PATH      binary under test (default: $XPAR, or ../xpar)
  --size BYTES     corpus size (default: 268435456)
  --reps N         repetitions per measurement (default: 3)
  --seed N         corpus seed (default: 20260823)
  --jobs N         worker threads passed to xpar (default: xpar's own)
  --cold WHICH     none (default) or drop, which needs privileges
  --quick          32 MiB corpus, one repetition, a short matrix
  --full           the long matrix
  --keep           leave the corpus and the sets behind
  -h, --help       this message
EOF
  exit 0
}

while test $# -gt 0; do
  case $1 in
    --out)    out=$2;   shift 2 ;;
    --xpar)   xpar=$2;  shift 2 ;;
    --size)   size=$2;  shift 2 ;;
    --reps)   reps=$2;  shift 2 ;;
    --seed)   seed=$2;  shift 2 ;;
    --jobs)   jobs=$2;  shift 2 ;;
    --cold)   cold=$2;  shift 2 ;;
    --quick)  size=33554432;  reps=1;  matrix=quick;  shift ;;
    --full)   matrix=full;  shift ;;
    --keep)   keep=1;  shift ;;
    -h|--help) usage ;;
    *) die "unknown option $1; try --help" ;;
  esac
done

bench_find_tools
bench_open_output
bench_probe_cold
bench_environment
test -n "$keep" || trap 'rm -rf "$work"' EXIT HUP INT TERM

say "writing to $out"
sed 's/^/  /' "$out/environment.json"

say "generating a $size byte corpus from seed $seed"
echo "$mkdata $seed $size $work/corpus.bin --pattern=random" >> "$cmdlog"
"$mkdata" "$seed" "$size" "$work/corpus.bin" --pattern=random
pristine=$work/corpus.bin

say "kernel tiers"
echo "$xpar benchmark --tiers --json --quiet" >> "$cmdlog"
"$xpar" benchmark --tiers --json --quiet > "$out/kernels.json" 2>/dev/null ||
  warn "benchmark --tiers failed"

setup_create() { rm -f "$sdir"/set*.xpa; }

check_create() {
  if test ! -f "$sdir/set.xpa"; then sig=no-set;  return 1; fi
  read_geometry "$sdir/set.xpa"
  account_archive "$sdir/set" "$g_r" "$g_z"
  f_archive_bytes=$archive_total
  f_nominal_payload_bytes=$archive_nominal
  f_format_overhead_bytes=$archive_overhead
  f_slice_size=$g_z;  f_cell_bytes=$g_y;  f_slices=$g_s
  f_recovery_slices=$g_r
  sig="archive=$f_archive_bytes slices=$g_s recovery=$g_r"
}

check_verify() {
  _rd=`jnum0 "$work/out.json" bytes_read summary`
  f_scan_bytes=$((_rd + f_archive_bytes))
  f_damaged_cells=`jnum0 "$work/out.json" cells_bad summary`
  sig="read=$_rd cells=$f_damaged_cells"
}

# Reapply identical corruption before each repetition.
setup_repair() {
  # Test hook: skip restores after the first repetition.
  if test "${XPAR_BENCH_BREAK:-}" = repair && test "$1" -ne 1; then
    return 0
  fi
  cp "$pristine" "$sdir/data.bin"
  rm -f "$sdir"/set.g*.jrnl "$sdir"/*.journal 2>/dev/null || true
  # shellcheck disable=SC2086
  "$damage" "$sdir/data.bin" -Z "$g_z" -Y "$g_y" -n 96 seed=$seed \
    $damage_ops || return 1
  "$xpar" verify --json "$sdir/set.xpa" > "$work/pre.json" 2>/dev/null || true
  f_damaged_cells=`jnum0 "$work/pre.json" cells_bad summary`
  f_damaged_slices=`jnum0 "$work/pre.json" slices_bad summary`
  f_column_depth=`jnum0 "$work/pre.json" column_depth summary`
  f_column_groups=`jnum0 "$work/pre.json" column_groups summary`
  f_scan_bytes=`jnum0 "$work/pre.json" bytes_read summary`
  f_scan_bytes=$((f_scan_bytes + f_archive_bytes))
  test "$f_damaged_cells" -gt 0 || return 1
}

check_repair() {
  if ! cmp -s "$sdir/data.bin" "$pristine"; then
    sig=not-repaired;  return 1
  fi
  f_repaired_bytes=`jnum0 "$work/out.json" bytes_written repair`
  sig="cells=$f_damaged_cells depth=$f_column_depth wrote=$f_repaired_bytes"
}

ops_whole_slices() {   # <count>
  damage_ops=
  _i=0
  while test "$_i" -lt "$1"; do
    _j=0
    while test "$_j" -lt "$g_k"; do
      damage_ops="$damage_ops cell=$_i,$_j"
      _j=$((_j + 1))
    done
    _i=$((_i + 1))
  done
}

# Spread cells across columns to keep column depth low.
ops_scattered() {   # <cells>
  damage_ops=
  _n=0
  _i=0
  while test "$_n" -lt "$1" && test "$_i" -lt "$g_s"; do
    _j=$((_i % g_k))
    damage_ops="$damage_ops cell=$_i,$_j"
    _n=$((_n + 1))
    _i=$((_i + 1))
  done
}

# Record unreachable parameter combinations without aborting the matrix.

case $matrix in
  quick)   codecs="matrix fft";  fields="16";     recs="10%";
           slices="0" ;;
  full)    codecs="matrix fft";  fields="8 16";   recs="5% 10% 25% 50%";
           slices="0 262144 1048576 4194304" ;;
  *)       codecs="matrix fft";  fields="8 16";   recs="5% 10% 25%";
           slices="0 1048576 4194304" ;;
esac

jflag=
test -z "$jobs" || jflag="-j $jobs"

for codec in $codecs; do
 for field in $fields; do
  for rec in $recs; do
   for slice in $slices; do
    sdir=$work/$codec-gf$field-$rec-$slice
    rm -rf "$sdir";  mkdir -p "$sdir"
    cp "$pristine" "$sdir/data.bin"
    if test "$slice" = 0; then sflag=;  slabel=auto
    else sflag="-s $slice";  slabel=$slice; fi

    echo
    say "=== $codec, GF(2^$field), recovery $rec, slice $slabel ==="

    reset_row
    f_experiment=throughput;  f_op=create;  f_codec=$codec;  f_field=$field
    f_recovery_spec=$rec;  f_layout=sidecar;  f_corpus=random
    f_corpus_bytes=$size;  f_slice_size=$slice;  f_expect=0
    # Status 4 (usage) and 7 (no feasible plan) are valid sweep results.
    f_refusals="4 7"
    # shellcheck disable=SC2086
    bench_measure setup_create check_create \
      "$xpar" create --reproducible --dedup=none --align=none \
        --no-verify-after --codec="$codec" --field="$field" -r "$rec" \
        $sflag $jflag --json -o "$sdir/set" "$sdir/data.bin"

    if test ! -f "$sdir/set.xpa"; then
      warn "skipping the rest of $sdir: no set was written"
      rm -rf "$sdir"
      continue
    fi
    read_geometry "$sdir/set.xpa"
    archive=`archive_bytes "$sdir/set"`

    for vop in verify verify-strong; do
      test "$vop" = verify && vflag= || vflag=--strong
      reset_row
      f_experiment=throughput;  f_op=$vop;  f_codec=$g_codec
      f_field=$g_field;  f_recovery_spec=$rec;  f_recovery_slices=$g_r
      f_layout=sidecar;  f_corpus=random;  f_corpus_bytes=$size
      f_slice_size=$g_z;  f_cell_bytes=$g_y;  f_slices=$g_s
      f_archive_bytes=$archive;  f_expect=0
      # shellcheck disable=SC2086
      bench_measure setup_none check_verify \
        "$xpar" verify $vflag $jflag --json "$sdir/set.xpa"
    done

    # Compare whole-slice damage with equal-count scattered cells.
    lost=$g_r
    test "$lost" -le 4 || lost=4
    for dmg in slices scatter; do
      if test "$dmg" = slices; then
        ops_whole_slices "$lost"
        cells=$((lost * g_k))
      else
        cells=$((lost * g_k))
        test "$cells" -le "$g_s" || cells=$g_s
        ops_scattered "$cells"
      fi
      test -n "$damage_ops" || continue
      reset_row
      f_experiment=throughput;  f_op=repair-$dmg;  f_codec=$g_codec
      f_field=$g_field;  f_recovery_spec=$rec;  f_recovery_slices=$g_r
      f_layout=sidecar;  f_corpus=random;  f_corpus_bytes=$size
      f_slice_size=$g_z;  f_cell_bytes=$g_y;  f_slices=$g_s
      f_archive_bytes=$archive;  f_damage=$dmg;  f_expect=0
      # shellcheck disable=SC2086
      bench_measure setup_repair check_repair \
        "$xpar" repair --in-place --no-journal $jflag --json "$sdir/set.xpa"
    done

    rm -rf "$sdir"
   done
  done
 done
done

echo
bench_finish
