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

# Reproducible benchmark harness producing environment, command and raw data.

set -e

prog=`basename "$0"`
here=`cd \`dirname "$0"\` && pwd`
top=`cd "$here/.." && pwd`

die() { echo "$prog: $*" >&2;  exit 1; }

# Options.

out=bench-results
size=268435456          # 256 MiB
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

Sizes are bytes.  Every number the harness chooses is recorded in
environment.json, so a run reproduces from that file alone.
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

# Tools.

if test -z "$xpar"; then
  if   test -x "$top/xpar";     then xpar=$top/xpar
  elif test -x "$top/xpar.exe"; then xpar=$top/xpar.exe
  else die "no xpar binary; pass --xpar"
  fi
fi
case $xpar in
  /*) ;;
  *)  xpar=`cd \`dirname "$xpar"\` && pwd`/`basename "$xpar"` ;;
esac
test -x "$xpar" || die "$xpar is not executable"

mkdata=$top/tests/mkdata
damage=$top/tests/damage
timeit=$top/bench/timeit
for t in "$mkdata" "$damage" "$timeit"; do
  test -x "$t" || test -x "$t.exe" ||
    die "$t is not built; run 'make bench-tools'"
done
test -x "$mkdata" || mkdata=$mkdata.exe
test -x "$damage" || damage=$damage.exe
test -x "$timeit" || timeit=$timeit.exe

mkdir -p "$out"
out=`cd "$out" && pwd`
work=$out/work
rm -rf "$work"
mkdir -p "$work"
if test -z "$keep"; then trap 'rm -rf "$work"' EXIT HUP INT TERM; fi

csv=$out/results.csv
jsonl=$out/results.json
cmdlog=$out/commands.log
: > "$csv";  : > "$jsonl";  : > "$cmdlog"

# Environment facts; unavailable values are recorded as null.

jstr() {
  if test -z "$1"; then printf 'null'
  else
    printf '"%s"' \
      "`printf '%s' "$1" | sed 's/\\\\/\\\\\\\\/g; s/"/\\\\"/g; s/	/ /g'`"
  fi
}

cpu_model=
if test -r /proc/cpuinfo; then
  cpu_model=`sed -n 's/^model name[ 	]*: *//p' /proc/cpuinfo | head -1`
  test -n "$cpu_model" ||
    cpu_model=`sed -n 's/^Model[ 	]*: *//p' /proc/cpuinfo | head -1`
elif command -v sysctl > /dev/null 2>&1; then
  cpu_model=`sysctl -n machdep.cpu.brand_string 2>/dev/null || true`
fi

cores=
if command -v nproc > /dev/null 2>&1; then cores=`nproc`
elif command -v getconf > /dev/null 2>&1; then
  cores=`getconf _NPROCESSORS_ONLN 2>/dev/null || true`
fi

memkb=
test -r /proc/meminfo &&
  memkb=`sed -n 's/^MemTotal: *\([0-9]*\).*/\1/p' /proc/meminfo`

governor=
test -r /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor &&
  governor=`cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor`

turbo=
test -r /sys/devices/system/cpu/intel_pstate/no_turbo &&
  turbo=`cat /sys/devices/system/cpu/intel_pstate/no_turbo`

cc=
ccver=
if test -r "$top/config.log"; then
  cc=`sed -n 's/^ *CC='\''\(.*\)'\''$/\1/p' "$top/config.log" | head -1`
fi
test -n "$cc" || cc=${CC:-cc}
ccver=`$cc --version 2>/dev/null | head -1 || true`

configure_line=
test -r "$top/config.log" &&
  configure_line=`sed -n 's/^  \$ \(.*configure.*\)$/\1/p' \
                    "$top/config.log" | head -1`

commit=
if test -d "$top/.git" && command -v git > /dev/null 2>&1; then
  commit=`cd "$top" && git rev-parse HEAD 2>/dev/null || true`
  dirty=`cd "$top" && git status --porcelain 2>/dev/null | head -1`
  test -z "$dirty" || commit="$commit+dirty"
fi

version=`"$xpar" --version 2>&1 | head -1`
fstype=
command -v stat > /dev/null 2>&1 &&
  fstype=`stat -f -c %T "$work" 2>/dev/null || true`
started=`date -u '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || true`

{
  printf '{\n'
  printf '  "schema": 1,\n'
  printf '  "started_utc": %s,\n'     "`jstr "$started"`"
  printf '  "xpar_version": %s,\n'    "`jstr "$version"`"
  printf '  "xpar_path": %s,\n'       "`jstr "$xpar"`"
  printf '  "git_commit": %s,\n'      "`jstr "$commit"`"
  printf '  "configure": %s,\n'       "`jstr "$configure_line"`"
  printf '  "cc": %s,\n'              "`jstr "$cc"`"
  printf '  "cc_version": %s,\n'      "`jstr "$ccver"`"
  printf '  "uname": %s,\n'           "`jstr "\`uname -a\`"`"
  printf '  "cpu_model": %s,\n'       "`jstr "$cpu_model"`"
  printf '  "cores": %s,\n'           "${cores:-null}"
  printf '  "mem_total_kb": %s,\n'    "${memkb:-null}"
  printf '  "scaling_governor": %s,\n' "`jstr "$governor"`"
  printf '  "intel_pstate_no_turbo": %s,\n' "${turbo:-null}"
  printf '  "filesystem": %s,\n'      "`jstr "$fstype"`"
  printf '  "cache_mode": %s,\n'      "`jstr "$cold"`"
  printf '  "corpus_seed": %s,\n'     "$seed"
  printf '  "corpus_bytes": %s,\n'    "$size"
  printf '  "repetitions": %s,\n'     "$reps"
  printf '  "matrix": %s,\n'          "`jstr "$matrix"`"
  printf '  "jobs": %s\n'             "`jstr "$jobs"`"
  printf '}\n'
} > "$out/environment.json"

echo "$prog: writing to $out"
sed 's/^/  /' "$out/environment.json"

if test "$cold" = drop && test ! -w /proc/sys/vm/drop_caches; then
  echo "$prog: WARNING: /proc/sys/vm/drop_caches is not writable;" >&2
  echo "$prog:          the run will be warm and is recorded as such" >&2
  cold=none
  sed -i.bak 's/"cache_mode": "drop"/"cache_mode": "none"/' \
      "$out/environment.json" 2>/dev/null || true
  rm -f "$out/environment.json.bak"
fi

# Flush prior writes before every measurement.
settle() {
  sync 2>/dev/null || true
  test "$cold" = drop || return 0
  echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
}

# Run one measurement.

printf 'run_id,seed,op,codec,field,slice_size,recovery,layout,' >> "$csv"
printf 'jobs,rep,input_bytes,elapsed_us,maxrss_kb,cold,status\n' >> "$csv"

run_id=0

measure() {   # <op> <codec> <field> <slice> <recovery> <layout> <cmd...>
  op=$1;  m_codec=$2;  m_field=$3;  m_slice=$4;  m_rec=$5;  m_layout=$6
  shift 6
  rep=1
  while test "$rep" -le "$reps"; do
    run_id=`expr $run_id + 1`
    echo "# run $run_id  $op  rep $rep" >> "$cmdlog"
    echo "$*" >> "$cmdlog"
    settle
    st=0
    "$timeit" "$work/timing" "$@" > "$work/out.log" 2>&1 || st=$?
    us=`sed -n 's/^elapsed_us=//p' "$work/timing"`
    rss=`sed -n 's/^maxrss_kb=//p' "$work/timing"`
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$run_id" "$seed" "$op" "$m_codec" "$m_field" "$m_slice" "$m_rec" \
      "$m_layout" "${jobs:-auto}" "$rep" "$size" "$us" "$rss" "$cold" \
      "$st" >> "$csv"
    printf '{"run_id":%s,"seed":%s,"op":"%s","codec":"%s","field":%s,' \
      "$run_id" "$seed" "$op" "$m_codec" "$m_field" >> "$jsonl"
    printf '"slice_size":"%s","recovery":"%s","layout":"%s","jobs":"%s",' \
      "$m_slice" "$m_rec" "$m_layout" "${jobs:-auto}" >> "$jsonl"
    printf '"rep":%s,"input_bytes":%s,"elapsed_us":%s,"maxrss_kb":%s,' \
      "$rep" "$size" "${us:-0}" "${rss:-0}" >> "$jsonl"
    printf '"cold":"%s","status":%s}\n' "$cold" "$st" >> "$jsonl"
    if test "$st" -ne 0; then
      echo "$prog: WARNING: $op exited $st; see $cmdlog run $run_id" >&2
      sed 's/^/  | /' "$work/out.log" >&2
    else
      echo "  $op  ${m_codec}/gf${m_field}  r=${m_rec}  rep $rep  ${us} us"
    fi
    rep=`expr $rep + 1`
  done
}

# Seeded corpus shared across hosts.

echo "$prog: generating a $size byte corpus from seed $seed"
echo "$mkdata $seed $size $work/corpus.bin --pattern=random" >> "$cmdlog"
"$mkdata" "$seed" "$size" "$work/corpus.bin" --pattern=random

# Kernel microbenchmark.

echo "$prog: kernel tiers"
echo "$xpar benchmark --tiers --json --quiet" >> "$cmdlog"
"$xpar" benchmark --tiers --json --quiet > "$out/kernels.json" 2>/dev/null ||
  echo "$prog: WARNING: benchmark --tiers failed" >&2

# Benchmark matrix.

# Record unsupported combinations instead of aborting the matrix.
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
    tag="$codec-gf$field-$rec-$slice"
    sdir=$work/$tag
    rm -rf "$sdir";  mkdir -p "$sdir"
    cp "$work/corpus.bin" "$sdir/data.bin"
    if test "$slice" = 0; then sflag=;  slabel=auto
    else sflag="-s $slice";  slabel=$slice; fi

    echo
    echo "$prog: === $codec, GF(2^$field), recovery $rec, slice $slabel ==="

    # shellcheck disable=SC2086
    measure create "$codec" "$field" "$slabel" "$rec" sidecar \
      "$xpar" create --reproducible --dedup=none --align=none \
        --no-verify-after --codec="$codec" --field="$field" -r "$rec" \
        $sflag $jflag -o "$sdir/set" "$sdir/data.bin"

    if test ! -f "$sdir/set.xpa"; then
      echo "$prog: skipping the rest of $tag: no set was written" >&2
      continue
    fi

    # shellcheck disable=SC2086
    measure verify "$codec" "$field" "$slabel" "$rec" sidecar \
      "$xpar" verify $jflag "$sdir/set.xpa"

    # shellcheck disable=SC2086
    measure verify-strong "$codec" "$field" "$slabel" "$rec" sidecar \
      "$xpar" verify --strong $jflag "$sdir/set.xpa"

    # Damage the same leading slices in each comparable repair run.
    "$xpar" info --json "$sdir/set.xpa" > "$sdir/geom.json" 2>/dev/null || true
    Z=`sed -n 's/.*"slice_size":\([0-9]*\).*/\1/p' "$sdir/geom.json" | head -1`
    Y=`sed -n 's/.*"cell_bytes":\([0-9]*\).*/\1/p' "$sdir/geom.json" | head -1`
    R=`sed -n 's/.*"recovery":\([0-9]*\).*/\1/p' "$sdir/geom.json" | head -1`
    if test -n "$Z" && test -n "$R" && test "$R" -gt 0; then
      test -n "$Y" && test "$Y" -gt 0 || Y=$Z
      K=`expr \( $Z + $Y - 1 \) / $Y`
      lost=$R
      test "$lost" -le 4 || lost=4
      ops=
      i=0
      while test "$i" -lt "$lost"; do
        j=0
        while test "$j" -lt "$K"; do
          ops="$ops cell=$i,$j"
          j=`expr $j + 1`
        done
        i=`expr $i + 1`
      done
      echo "$damage $sdir/data.bin -Z $Z -Y $Y -n 96 $ops" >> "$cmdlog"
      # shellcheck disable=SC2086
      "$damage" "$sdir/data.bin" -Z "$Z" -Y "$Y" -n 96 $ops
      # shellcheck disable=SC2086
      measure repair "$codec" "$field" "$slabel" "$rec" sidecar \
        "$xpar" repair --in-place --no-journal $jflag "$sdir/set.xpa"
    fi

    #  An armoured archive is the layout with the most work per byte, so
    #  it is measured separately rather than folded into the average.
    if test "$matrix" = full; then
      # shellcheck disable=SC2086
      measure create-armoured "$codec" "$field" "$slabel" "$rec" armoured \
        "$xpar" create --reproducible --layout=armoured --dedup=none \
          --no-verify-after --codec="$codec" --field="$field" -r "$rec" \
          $sflag $jflag -o "$sdir/arm" "$work/corpus.bin"
    fi

    rm -rf "$sdir"
   done
  done
 done
done

echo
echo "$prog: done"
echo "$prog:   $csv"
echo "$prog:   $jsonl"
echo "$prog:   $out/environment.json"
echo "$prog:   $out/kernels.json"
echo "$prog:   $cmdlog"
echo "$prog: plot with: python3 $here/plot.py $out"
