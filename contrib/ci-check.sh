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

set -e

prog=`basename "$0"`

fail() { echo "$prog: $*" >&2; exit 1; }
phase="(starting up)"
step() { phase="$*";  echo;  echo "$prog: --- $* ---"; }

xpar=${1:-}
if test -z "$xpar"; then
  if   test -f ./xpar;     then xpar=./xpar
  elif test -f ./xpar.exe; then xpar=./xpar.exe
  else fail "xpar not found. Pass its path as \$1"
  fi
fi
test -f "$xpar" || fail "$xpar: not found"
case $xpar in
  /*|?:[/\\]*) ;;
  *)  xpar=`pwd`/$xpar ;;
esac

compat=${2:-}
if test -n "$compat"; then
  test -d "$compat" || fail "$compat: not a directory"
  compat=`cd "$compat" && pwd`
fi
if test ! -r /dev/urandom && test ! -x "${DAMAGE:-}"; then
  fail "no damage source"
fi

_test_cfg=${XPAR_TEST_CONFIG_H:-}
if test -n "$_test_cfg" && test -f "$_test_cfg" &&
   grep -q '^#define XPAR_DOS ' "$_test_cfg"; then
  _work_id=`printf '%s' "ci-check.$$" | cksum |
    awk '{ printf "%07d", $1 % 10000000 }'`
  work=`pwd`/D$_work_id
else
  work=`pwd`/ci-check.$$
fi
rm -rf "$work"
mkdir "$work"
trap 'cd /; rm -rf "$work"' EXIT HUP INT TERM
log=$work/last.log
cd "$work"

explain_status() {
  st=$1
  if test "$st" -gt 128 && test "$st" -lt 192; then
    sig=`expr "$st" - 128`
    case $sig in
      2)  echo "killed by SIGINT" ;;
      4)  echo "CRASHED: SIGILL, illegal instruction" ;;
      6)  echo "CRASHED: SIGABRT, aborted" ;;
      7)  echo "CRASHED: SIGBUS, bad address" ;;
      8)  echo "CRASHED: SIGFPE, arithmetic fault" ;;
      9)  echo "killed by SIGKILL" ;;
      11) echo "CRASHED: SIGSEGV, invalid memory reference" ;;
      13) echo "killed by SIGPIPE" ;;
      15) echo "killed by SIGTERM" ;;
      *)  echo "killed by signal $sig" ;;
    esac
    return
  fi
  case $st in
    3221225477) echo "CRASHED: 0xC0000005, access violation" ;       return ;;
    3221225725) echo "CRASHED: 0xC00000FD, stack overflow" ;         return ;;
    3221225620) echo "CRASHED: 0xC0000094, integer divide by zero" ; return ;;
    3221225786) echo "interrupted (0xC000013A)" ;                    return ;;
    32212*)     echo "CRASHED: Windows exception $st" ;              return ;;
  esac
  case $st in
    0) echo "clean" ;;
    1) echo "damaged, repairable" ;;
    2) echo "damage beyond the recovery data" ;;
    3) echo "not found, or not an xpar set" ;;
    4) echo "usage error" ;;
    5) echo "I/O error" ;;
    6) echo "authentication failure" ;;
    7) echo "no plan fits the memory ceiling" ;;
    8) echo "internal error (a bug)" ;;
    *) echo "unrecognised status" ;;
  esac
}

describe_subject() {
  echo "$prog: the binary under test:" >&2
  ls -l "$xpar" 2>&1 | sed 's/^/  /' >&2
  if command -v file > /dev/null 2>&1; then
    file "$xpar" 2>&1 | sed 's/^/  /' >&2
  fi
}

run() {
  want=$1; shift
  got=0
  "$@" > "$log" 2>&1 || got=$?
  if test "$got" -ne "$want"; then
    echo >&2
    echo "$prog: FAILED in phase: $phase" >&2
    echo "$prog:   command : $*" >&2
    echo "$prog:   expected: $want (`explain_status "$want"`)" >&2
    echo "$prog:   got     : $got (`explain_status "$got"`)" >&2
    if test -s "$log"; then
      echo "$prog:   output  :" >&2
      sed 's/^/  | /' "$log" >&2
    else
      echo "$prog:   output  : none -- it produced nothing at all" >&2
    fi
    describe_subject
    exit 1
  fi
  if test -s "$log"; then sed 's/^/  | /' "$log"; fi
}

smash() {
  if test -r /dev/urandom; then
    dd if=/dev/urandom of="$1" bs=1 seek="$2" count="$3" \
      conv=notrunc 2>/dev/null
  else
    "$DAMAGE" "$1" "rand=$2,$3"
  fi
}

random_file() {
  if test -r /dev/urandom; then
    dd if=/dev/urandom of="$1" bs=1024 count="$2" 2>/dev/null
  else
    "$MKDATA" "$3" "`expr "$2" \* 1024`" "$1"
  fi
}

same() {
  cmp "$1" "$2" || fail "$1 and $2 differ"
}

step "the binary runs at all"
run 0 "$xpar" --version

step "kernel tiers agree with scalar"
run 0 "$xpar" benchmark --tiers

step "corpus"
mkdir -p tree/sub
big_kib=700
small_kib=37
cfg=${XPAR_TEST_CONFIG_H:-}
sidecar_volumes='side.v*.xpa'
doomed_volumes='doom.v*.xpa'
if test -f "$cfg" && grep -q '^#define XPAR_DOS ' "$cfg"; then
  # These sizes cover every damage offset below.
  big_kib=264
  small_kib=32
  sidecar_volumes='SIDE.V??'
  doomed_volumes='DOOM.V??'
fi
random_file tree/big.bin "$big_kib" 701
random_file tree/small.bin "$small_kib" 37
cp tree/big.bin tree/sub/twin.bin
printf 'the quick brown fox jumps over the lazy dog\n' > tree/sub/note.txt
cp -R tree orig
ls -l tree tree/sub

step "sidecar: create"
run 0 "$xpar" create -R -r 25% --dedup=file -o side tree

step "sidecar: inspect"
run 0 "$xpar" verify side.xpa
run 0 "$xpar" verify --strong side.xpa
run 0 "$xpar" scrub --deep side.xpa
run 0 "$xpar" list side.xpa
run 0 "$xpar" list --json side.xpa
run 0 "$xpar" info side.xpa
run 0 "$xpar" explain side.xpa

step "sidecar: scattered damage is repaired in place"
smash tree/big.bin 100000 24
smash tree/small.bin 4096 8
run 1 "$xpar" verify side.xpa
run 0 "$xpar" repair --in-place --paranoid side.xpa
same tree/big.bin orig/big.bin
same tree/small.bin orig/small.bin
run 0 "$xpar" verify --strong side.xpa

step "sidecar: a burst across a slice boundary is repaired"
smash tree/big.bin 262100 600
run 1 "$xpar" verify side.xpa
run 0 "$xpar" repair --in-place side.xpa
same tree/big.bin orig/big.bin

step "sidecar: a truncated entry is rebuilt"
dd if=tree/small.bin of=tree/small.cut bs=1024 count=20 2>/dev/null
mv tree/small.cut tree/small.bin
run 1 "$xpar" verify side.xpa
run 0 "$xpar" repair --in-place side.xpa
same tree/small.bin orig/small.bin

step "sidecar: a lost volume is regenerated"
victim=`ls $sidecar_volumes | head -1`
rm -f "$victim"
run 0 "$xpar" verify side.xpa
recover_volume=`basename "$victim"`
run 0 "$xpar" recover --volume="$recover_volume" side.xpa
test -f "$victim" || fail "recover did not write back $victim"
run 0 "$xpar" verify side.xpa

step "sidecar: a second generation protects a new file"
random_file tree/late.bin 64 64
run 0 "$xpar" add -R -r 25% side.xpa tree
run 0 "$xpar" verify --chain side.xpa

step "kernels: encode scalar, decode dispatched"
cp orig/big.bin cross.bin
run 0 "$xpar" create --simd=scalar -r 25% -o xsca cross.bin
smash cross.bin 65536 40
run 1 "$xpar" verify xsca.xpa
run 0 "$xpar" repair --in-place xsca.xpa
same cross.bin orig/big.bin

step "kernels: encode dispatched, decode scalar"
run 0 "$xpar" create -r 25% -o xscb cross.bin
smash cross.bin 131072 40
run 1 "$xpar" verify --simd=scalar xscb.xpa
run 0 "$xpar" repair --simd=scalar --in-place xscb.xpa
same cross.bin orig/big.bin

step "armoured: create and extract"
run 0 "$xpar" create --layout=armoured -r 25% -o armo cross.bin
cp armo.xpa prlg.xpa
mkdir aout
run 0 "$xpar" extract --to aout armo.xpa
same aout/cross.bin orig/big.bin

step "armoured: damage inside the volume is repaired"
smash armo.xpa 200000 900
run 1 "$xpar" verify armo.xpa
mkdir afix
run 0 "$xpar" repair --to afix armo.xpa
same afix/cross.bin orig/big.bin

step "armoured: a wrecked prologue is brute-forced back"
smash prlg.xpa 0 12
run 0 "$xpar" recover-prologue prlg.xpa

step "split: create, extract and verify"
run 0 "$xpar" create --layout=split -R -r 25% -o splt orig
mkdir sout
run 0 "$xpar" extract --to sout splt.xpa
same sout/orig/big.bin orig/big.bin
run 0 "$xpar" verify splt.xpa

step "geometry: explicit slice size, matrix codec, GF(2^16)"
geom_input=cross.bin
if test "$big_kib" -ne 700; then
  dd if=cross.bin of=geometry.bin bs=1024 count=64 2>/dev/null
  geom_input=geometry.bin
fi
run 0 "$xpar" create -s 1K --codec=matrix --field=16 -r 30% -o gma1 "$geom_input"
run 0 "$xpar" verify gma1.xpa

step "geometry: explicit slice count, fft codec, interleaved armour"
run 0 "$xpar" create -b 64 --codec=fft --depth=4 -r 30% -o gma2 "$geom_input"
run 0 "$xpar" verify gma2.xpa

step "geometry: an empty file and a one-byte file"
: > empty.bin
printf 'x' > one.bin
run 0 "$xpar" create -r 50% -o gma3 empty.bin one.bin
run 0 "$xpar" verify gma3.xpa

step "geometry: single-threaded planning"
run 0 "$xpar" create -j 1 -r 25% -o gma4 one.bin
run 0 "$xpar" verify -j 1 gma4.xpa

step "hopeless damage is reported as such"
run 0 "$xpar" create -r 5% -o doom cross.bin
rm -f $doomed_volumes
smash cross.bin 0 8192
run 2 "$xpar" verify doom.xpa

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
