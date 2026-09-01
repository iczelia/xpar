#!/bin/sh
# Run the complete DOS suite across three persistent FreeDOS guests.

set -e

prog=${0##*/}
fail() { echo "$prog: $*" >&2; exit 1; }

dos_build=${1:?usage: run-parallel-suite.sh DOS-BUILD FREEDOS-ROOT [WORKDIR]}
freedos_root=${2:?FreeDOS root required}
suite_root=${3:-./dos-suite}

script_dir=`cd "$(dirname "$0")" && pwd`
mkdir -p "$suite_root"
suite_root=`cd "$suite_root" && pwd`

pids=
names=
launch() {
  shard=$1
  programs=$2
  tests=$3
  names="$names $shard"
  echo "$prog: starting $shard shard"
  (
    started=`date +%s`
    shard_status=0
    XPAR_DOS_PROGRAM_TESTS="$programs" XPAR_DOS_TESTS="$tests" \
      "$script_dir/run-suite.sh" "$dos_build" "$freedos_root" \
      "$suite_root/$shard" > "$suite_root/$shard.log" 2>&1 || \
      shard_status=$?
    finished=`date +%s`
    printf '%s\n' "$shard_status" > "$suite_root/$shard.status"
    echo "$prog: $shard shard finished in $((finished - started)) s"
    exit "$shard_status"
  ) &
  pids="$pids $!"
}

launch core 'TUNIT TCODEC TCENTRAL' \
  'CENTRAL.SH FAULTS.SH SAFETY.SH SANITY.SH BUILD.SH HOSTFLT.SH'
launch regression '' 'REGRESS.SH'
launch performance '' 'PERF.SH BENCH.SH'

status=0
for pid in $pids; do wait "$pid" || status=1; done

for shard in $names; do
  echo
  echo "===== FreeDOS $shard shard ====="
  cat "$suite_root/$shard.log"
done

test "$status" -eq 0 || fail "one or more FreeDOS shards failed"
echo "$prog: all FreeDOS shards passed"
