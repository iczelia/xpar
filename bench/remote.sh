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

# Run benchmarks remotely under --dir, then copy results back.

set -e

prog=`basename "$0"`
here=`cd \`dirname "$0"\` && pwd`
top=`cd "$here/.." && pwd`

host=
dir=
out=remote-results
what="run"
build_competitors=
args=

usage() {
  cat <<'EOF'
usage: remote.sh --host HOST --dir DIR [options] [-- experiment args...]

  --host HOST        ssh destination
  --dir DIR          remote workspace (all remote files stay here)
  --out DIR          local results directory (default: remote-results)
  --what WHICH       run, experiments, or both (default: run)
  --competitors      build benchmark baselines remotely
  -h, --help         show this help

Arguments after -- are passed to the remote benchmark script.
EOF
  exit 0
}

while test $# -gt 0; do
  case $1 in
    --host) host=$2;  shift 2 ;;
    --dir)  dir=$2;   shift 2 ;;
    --out)  out=$2;   shift 2 ;;
    --what) what=$2;  shift 2 ;;
    --competitors) build_competitors=1;  shift ;;
    --) shift;  args=$*;  break ;;
    -h|--help) usage ;;
    *) echo "$prog: unknown option: $1 (try --help)" >&2;  exit 1 ;;
  esac
done

test -n "$host" || { echo "$prog: missing required option: --host" >&2; exit 1; }
test -n "$dir" || { echo "$prog: missing required option: --dir" >&2; exit 1; }

say() { echo "$prog: $*"; }
warn() { echo "$prog: $*" >&2; }

say "creating source archive"
( cd "$top" && make dist-gzip > /dev/null )
tarball=`cd "$top" && ls -t xpar-*.tar.gz | head -1`
test -n "$tarball" || { echo "$prog: source archive missing" >&2; exit 1; }

say "uploading $tarball to $host:$dir"
ssh "$host" "mkdir -p '$dir'"
scp -q "$top/$tarball" "$host:$dir/"

say "building on: $host"
ssh "$host" "set -e
  cd '$dir'
  rm -rf xpar
  mkdir xpar
  tar -C xpar --strip-components=1 -xf '$tarball'
  cd xpar
  ./configure > configure.log 2>&1
  make -j\`nproc\` > build.log 2>&1
  make -j\`nproc\` bench-tools >> build.log 2>&1
  ./xpar --version | head -1"

if test -n "$build_competitors"; then
  say "building baselines on: $host"
  ssh "$host" "cd '$dir/xpar' && sh bench/competitors.sh \
    --root '$dir/competitors' --jobs \`nproc\` > '$dir/competitors.log' 2>&1 \
    || tail -20 '$dir/competitors.log'"
fi

comp=
ssh "$host" "test -r '$dir/competitors/competitors.json'" 2>/dev/null &&
  comp="--competitors $dir/competitors/competitors.json"

case $what in
  run|both)
    say "running throughput matrix"
    ssh "$host" "cd '$dir/xpar' && sh bench/run.sh --out '$dir/results-run' \
      $args" || warn "throughput matrix failed"
    ;;
esac
case $what in
  experiments|both)
    say "running experiments"
    ssh "$host" "cd '$dir/xpar' && sh bench/experiments.sh \
      --out '$dir/results-exp' $comp $args" ||
      warn "experiments failed"
    ;;
esac

mkdir -p "$out"
for d in results-run results-exp; do
  ssh "$host" "test -d '$dir/$d'" 2>/dev/null || continue
  say "downloading results: $d"
  ssh "$host" "cd '$dir' && tar czf - --exclude=work '$d'" |
    tar -C "$out" -xzf -
done
ssh "$host" "test -r '$dir/competitors/competitors.json'" 2>/dev/null &&
  scp -q "$host:$dir/competitors/competitors.json" "$out/" || true

say "results: $out"
ls -R "$out" | head -40
