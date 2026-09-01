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

# Verify benchmark setup and work-signature checks.

. "${srcdir:-.}/lib.sh" 2> /dev/null || . "`dirname "$0"`/lib.sh"

run_sh=$abs_top_srcdir/bench/run.sh
test -r "$run_sh" || skip_all "bench/run.sh is not in this tree"

#  Allow cross builds to provide a launcher.
if test -z "${TIMEIT:-}"; then
  TIMEIT=$abs_top_builddir/bench/timeit
  test -x "$TIMEIT" || TIMEIT=$TIMEIT.exe
fi
test -x "$TIMEIT" ||
  skip_all "bench/timeit is not built; run 'make bench-tools'"
export TIMEIT MKDATA DAMAGE

bench_extra=
test "$xpar_test_dos" != yes || bench_extra="--codec matrix"

#  A missing CSV has zero bad rows.
bad_rows() {
  test -f "$1" || { echo 0;  return 0; }
  awk -F, 'NR == 1 { for (i = 1; i <= NF; i++) if ($i == "work_ok") c = i
                     next }
           $c != 1 { n++ }
           END { print n + 0 }' "$1"
}

usable_rows() {
  awk -F, 'NR > 1 { n++ } END { print n + 0 }' "$1"
}

#  Report per-measurement benchmark time.
bench_timing() {
  note "run.sh took $2 s; per-measurement timings:"
  sed -n 's/^.*repetition \([0-9]*\): \([0-9]*\) us.*$/\1 \2/p' "$1" |
    awk '{ printf "  | rep %s: %d ms\n", $1, $2 / 1000 }' >&2
  grep -E "kernel tiers took|time spent in sync" "$1" | sed 's/^/  | /' >&2
}

step "a clean run reports every repetition usable"

bench_size=4194304
xpar_config_defined XPAR_DOS && bench_size=524288
case `xpar_host` in *djgpp* | *msdos*) bench_size=524288 ;; esac
rc=0
t0=`date +%s`
"$XPAR_SH" "$run_sh" --quick --xpar "$XPAR" --out ok \
  --size "$bench_size" --reps 2 --seed "$XPAR_TEST_SEED" $bench_extra \
  > clean.log 2>&1 || rc=$?
bench_timing clean.log $((`date +%s` - t0))
if test "$rc" -eq 0; then ok
else bad "the harness exited $rc on a clean run"
     sed 's/^/  | /' clean.log | tail -20 >&2; fi

if test -s ok/results.csv; then
  ok
  equal "rows that failed the work check" "`bad_rows ok/results.csv`" 0
  n=`usable_rows ok/results.csv`
  if test "$n" -ge 8; then ok
  else bad "only $n measurements were recorded"; fi
else
  bad "no results.csv was written"
fi

if test -s clean.log; then
  sigs=`sed -n 's/.*\[\(.*\)\]$/\1/p' clean.log | sort -u | nlines`
  reps=`sed -n 's/.*\[\(.*\)\]$/\1/p' clean.log | nlines`
  if test "$reps" -gt 0 && test "$sigs" -lt "$reps"; then ok
  else bad "no two repetitions shared a work signature"; fi
fi

step "a repetition that skipped its restore is caught"

rc=0
t0=`date +%s`
XPAR_BENCH_BREAK=repair "$XPAR_SH" "$run_sh" --quick --xpar "$XPAR" \
  --out broken --size "$bench_size" --reps 2 --seed "$XPAR_TEST_SEED" \
  $bench_extra > broken.log 2>&1 || rc=$?
bench_timing broken.log $((`date +%s` - t0))

if test "$rc" -ne 0; then ok
else bad "the harness exited 0 with a repetition that did no work"; fi

if grep -q 'setup for repetition 2 failed' broken.log ||
   grep -q 'did different work than rep 1' broken.log ||
   test "`bad_rows broken/results.csv`" -gt 0; then
  ok
else
  bad "the harness did not report the broken repetition"
  sed 's/^/  | /' broken.log | tail -20 >&2
fi

step "an unexpected status fails the measurement"

rc=0
t0=`date +%s`
XPAR_BENCH_BREAK=status "$XPAR_SH" "$run_sh" --quick --xpar "$XPAR" \
  --out status --size "$bench_size" --reps 2 --seed "$XPAR_TEST_SEED" \
  $bench_extra > status.log 2>&1 || rc=$?
bench_timing status.log $((`date +%s` - t0))

if test "$rc" -ne 0; then ok
else bad "the harness accepted unexpected status 9"; fi

if test "`bad_rows status/results.csv`" -gt 0; then
  ok
else bad "no failed measurement was recorded"; fi

# Undeclared statuses must not be recorded as unsupported.
posing=`awk -F, 'NR == 1 {
    for (i = 1; i <= NF; i++) {
      if ($i == "expected_unsupported") u = i
      if ($i == "status") s = i
    }
    next }
  u && $u != "" && $s == 9 { n++ }
  END { print n + 0 }' status/results.csv 2>/dev/null || echo 0`
equal "undeclared status recorded as unsupported" "$posing" 0

summary
