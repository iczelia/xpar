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

#  contrib/ci-check.sh [path/to/xpar] [compat-dir]
#
#  The binary defaults to ./xpar[.exe]. A compat directory contains
#  corpus.bin and an armoured compat.xpa from another host.

set -e

prog=`basename "$0"`

fail() { echo "$prog: $*" >&2; exit 1; }
step() { echo; echo "$prog: --- $* ---"; }

xpar=${1:-}
if test -z "$xpar"; then
  if   test -f ./xpar;     then xpar=./xpar
  elif test -f ./xpar.exe; then xpar=./xpar.exe
  else fail "no ./xpar or ./xpar.exe here; pass the binary as \$1"
  fi
fi
test -f "$xpar" || fail "$xpar: not found"
case $xpar in
  /*) ;;
  *)  xpar=`pwd`/$xpar ;;
esac

compat=${2:-}
if test -n "$compat"; then
  test -d "$compat" || fail "$compat: not a directory"
  compat=`cd "$compat" && pwd`
fi
test -r /dev/urandom || fail "/dev/urandom is required"

work=`pwd`/ci-check.$$
rm -rf "$work"
mkdir "$work"
trap 'cd /; rm -rf "$work"' EXIT HUP INT TERM
log=$work/last.log
cd "$work"

#  Run a command and require exit status WANT.
run() {
  want=$1; shift
  got=0
  "$@" > "$log" 2>&1 || got=$?
  if test "$got" -ne "$want"; then
    echo "$prog: expected exit $want, got $got, from:" >&2
    echo "  $*" >&2
    sed 's/^/  | /' "$log" >&2
    exit 1
  fi
  sed 's/^/  | /' "$log"
}

#  Overwrite LENGTH bytes at OFFSET in FILE.
smash() {
  dd if=/dev/urandom of="$1" bs=1 seek="$2" count="$3" conv=notrunc 2>/dev/null
}

same() {
  cmp "$1" "$2" || fail "$1 and $2 differ"
}

step "kernel tiers agree with scalar"
run 0 "$xpar" --version
run 0 "$xpar" benchmark --tiers

#  Files around the default slice size, including duplicate content.
step "corpus"
mkdir -p tree/sub
dd if=/dev/urandom of=tree/big.bin   bs=1024 count=700 2>/dev/null
dd if=/dev/urandom of=tree/small.bin bs=1024 count=37  2>/dev/null
cp tree/big.bin tree/sub/twin.bin
printf 'the quick brown fox jumps over the lazy dog\n' > tree/sub/note.txt
cp -R tree tree.orig
ls -l tree tree/sub

#  Sidecar layout.
step "sidecar: create"
run 0 "$xpar" create -R -r 25% --dedup=file -o sidecar tree

step "sidecar: inspect"
run 0 "$xpar" verify sidecar.xpa
run 0 "$xpar" verify --strong sidecar.xpa
run 0 "$xpar" scrub --deep sidecar.xpa
run 0 "$xpar" list sidecar.xpa
run 0 "$xpar" list --json sidecar.xpa
run 0 "$xpar" info sidecar.xpa
run 0 "$xpar" explain sidecar.xpa

step "sidecar: scattered damage is repaired in place"
smash tree/big.bin 100000 24
smash tree/small.bin 4096 8
run 1 "$xpar" verify sidecar.xpa
run 0 "$xpar" repair --in-place --paranoid sidecar.xpa
same tree/big.bin tree.orig/big.bin
same tree/small.bin tree.orig/small.bin
run 0 "$xpar" verify --strong sidecar.xpa

step "sidecar: a burst across a slice boundary is repaired"
smash tree/big.bin 262100 600
run 1 "$xpar" verify sidecar.xpa
run 0 "$xpar" repair --in-place sidecar.xpa
same tree/big.bin tree.orig/big.bin

step "sidecar: a truncated entry is rebuilt"
dd if=tree/small.bin of=tree/small.cut bs=1024 count=20 2>/dev/null
mv tree/small.cut tree/small.bin
run 1 "$xpar" verify sidecar.xpa
run 0 "$xpar" repair --in-place sidecar.xpa
same tree/small.bin tree.orig/small.bin

step "sidecar: a lost volume is regenerated"
victim=`ls sidecar.v*.xpa | head -1`
rm -f "$victim"
run 0 "$xpar" verify sidecar.xpa
run 0 "$xpar" recover --volume="`basename "$victim"`" sidecar.xpa
test -f "$victim" || fail "recover did not write back $victim"
run 0 "$xpar" verify sidecar.xpa

step "sidecar: a second generation protects a new file"
dd if=/dev/urandom of=tree/late.bin bs=1024 count=64 2>/dev/null
run 0 "$xpar" add -R -r 25% sidecar.xpa tree
run 0 "$xpar" verify --chain sidecar.xpa

#  Cross-check dispatched and scalar kernels in both directions.
step "kernels: encode scalar, decode dispatched"
cp tree.orig/big.bin cross.bin
run 0 "$xpar" create --simd=scalar -r 25% -o crossa cross.bin
smash cross.bin 65536 40
run 1 "$xpar" verify crossa.xpa
run 0 "$xpar" repair --in-place crossa.xpa
same cross.bin tree.orig/big.bin

step "kernels: encode dispatched, decode scalar"
run 0 "$xpar" create -r 25% -o crossb cross.bin
smash cross.bin 131072 40
run 1 "$xpar" verify --simd=scalar crossb.xpa
run 0 "$xpar" repair --simd=scalar --in-place crossb.xpa
same cross.bin tree.orig/big.bin

#  Armoured layout.
step "armoured: create and extract"
run 0 "$xpar" create --layout=armoured -r 25% -o armour cross.bin
cp armour.xpa prologue.xpa
mkdir armour.out
run 0 "$xpar" extract --to armour.out armour.xpa
same armour.out/cross.bin tree.orig/big.bin

step "armoured: damage inside the volume is repaired"
smash armour.xpa 200000 900
run 1 "$xpar" verify armour.xpa
mkdir armour.fix
run 0 "$xpar" repair --to armour.fix armour.xpa
same armour.fix/cross.bin tree.orig/big.bin

#  Recover a lost prologue from an otherwise intact archive.
step "armoured: a wrecked prologue is brute-forced back"
smash prologue.xpa 0 12
run 0 "$xpar" recover-prologue prologue.xpa

#  Split layout.
step "split: create, extract and verify"
run 0 "$xpar" create --layout=split -R -r 25% -o split tree.orig
mkdir split.out
run 0 "$xpar" extract --to split.out split.xpa
same split.out/tree.orig/big.bin tree.orig/big.bin
run 0 "$xpar" verify split.xpa

#  Planner edge cases.
step "geometry: explicit slice size, matrix codec, GF(2^16)"
run 0 "$xpar" create -s 1K --codec=matrix --field=16 -r 30% -o geom1 cross.bin
run 0 "$xpar" verify geom1.xpa

step "geometry: explicit slice count, fft codec, interleaved armour"
run 0 "$xpar" create -b 64 --codec=fft --depth=4 -r 30% -o geom2 cross.bin
run 0 "$xpar" verify geom2.xpa

step "geometry: an empty file and a one-byte file"
: > empty.bin
printf 'x' > one.bin
run 0 "$xpar" create -r 50% -o geom3 empty.bin one.bin
run 0 "$xpar" verify geom3.xpa

step "geometry: single-threaded planning"
run 0 "$xpar" create -j 1 -r 25% -o geom4 one.bin
run 0 "$xpar" verify -j 1 geom4.xpa

#  Unrecoverable damage.
step "hopeless damage is reported as such"
run 0 "$xpar" create -r 5% -o doomed cross.bin
rm -f doomed.v*.xpa
smash cross.bin 0 8192
run 2 "$xpar" verify doomed.xpa

#  Cross-host compatibility corpus.
if test -n "$compat"; then
  step "compat: a set written on another host reads back here"
  cp "$compat/compat.xpa" .
  run 0 "$xpar" verify compat.xpa
  run 0 "$xpar" info compat.xpa
  mkdir compat.out
  run 0 "$xpar" extract --to compat.out compat.xpa
  same compat.out/corpus.bin "$compat/corpus.bin"

  step "compat: and repairs it"
  smash compat.xpa 40000 700
  run 1 "$xpar" verify compat.xpa
  mkdir compat.fix
  run 0 "$xpar" repair --to compat.fix compat.xpa
  same compat.fix/corpus.bin "$compat/corpus.bin"
fi

echo
echo "$prog: all phases passed"
