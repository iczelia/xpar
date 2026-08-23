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

#  Run the DJGPP compatibility check under DOSBox-X.
#
#    contrib/dos/run-check.sh DOS-XPAR NATIVE-XPAR CWSDPMI.EXE [WORKDIR]
#
#  Requires dosbox-x and the DJGPP cross-compiler on PATH.

set -e

prog=`basename "$0"`
fail() { echo "$prog: $*" >&2; exit 1; }

dos_xpar=${1:?usage: run-check.sh DOS-XPAR NATIVE-XPAR CWSDPMI [WORKDIR]}
native_xpar=${2:?native xpar required}
cwsdpmi=${3:?CWSDPMI.EXE required}
root=${4:-./dos-check}

for f in "$dos_xpar" "$native_xpar" "$cwsdpmi"; do
  test -f "$f" || fail "$f: not found"
done
command -v dosbox-x > /dev/null || fail "dosbox-x is not on PATH"

srcdir=`cd \`dirname "$0"\` && pwd`
here=`pwd`
case $native_xpar in /*) ;; *) native_xpar=$here/$native_xpar ;; esac

rm -rf "$root"
mkdir -p "$root/work" "$root/out"
root=`cd "$root" && pwd`
work=$root/work
out=$root/out

#  Prepare the DOS drive.
cp "$dos_xpar" "$work/XPAR.EXE"
cp "$cwsdpmi"  "$work/CWSDPMI.EXE"
cp "$srcdir/ci-check.bat" "$work/ci-check.bat"

: "${CC_FOR_DOS:=i586-pc-msdosdjgpp-gcc}"
command -v "$CC_FOR_DOS" > /dev/null \
  || fail "$CC_FOR_DOS is not on PATH"
"$CC_FOR_DOS" -O2 -o "$work/RUN2.EXE" "$srcdir/run2.c"

#  Build the cross-host corpus.
test -r /dev/urandom || fail "/dev/urandom is needed to build the corpus"
( cd "$work"
  dd if=/dev/urandom of=BIG.BIN bs=1024 count=40 2>/dev/null
  cp BIG.BIN "$root/BIG.orig"
  "$native_xpar" create -r 25% -o SIDE BIG.BIN > /dev/null

  #  Create an unrepairable set.
  dd if=/dev/urandom of=DOOM.BIN bs=1024 count=40 2>/dev/null
  "$native_xpar" create -r 2% -o DOOMED DOOM.BIN > /dev/null
  rm -f DOOMED.v*.xpa
  dd if=/dev/urandom of=DOOM.BIN bs=1024 count=20 conv=notrunc 2>/dev/null

  #  Prepare damage for the repair test.
  cp BIG.BIN BAD.BIN
  dd if=/dev/urandom of=BAD.BIN bs=1 seek=8000 count=200 conv=notrunc 2>/dev/null

  #  Create input for a DOS-generated set.
  dd if=/dev/urandom of=MADE.BIN bs=1024 count=24 2>/dev/null )

#  Run the DOS check.
sed -e "s|@WORK@|$work|" -e "s|@OUT@|$out|" \
    "$srcdir/dosbox-x.conf" > "$root/dosbox-x.conf"

echo "$prog: starting DOSBox-X"
SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy} \
SDL_AUDIODRIVER=${SDL_AUDIODRIVER:-dummy} \
  dosbox-x -conf "$root/dosbox-x.conf" -exit > "$root/dosbox.log" 2>&1 \
  || fail "dosbox-x exited non-zero; see $root/dosbox.log"

#  Check the results.
result=$out/RESULT.TXT
test -f "$result" || fail "missing RESULT.TXT; see $root/dosbox.log"
tr -d '\r' < "$result" > "$result.clean"

for log in "$out"/*.TXT; do
  case $log in *RESULT*) continue ;; esac
  echo "--- `basename "$log"` ---"
  tr -d '\r' < "$log" | sed 's/^/  | /'
done

want="version verify-clean verify-strong info list explain
      verify-damaged repair verify-repaired create verify-own
      verify-unrepairable ALLDONE"
missing=
for step in $want; do
  grep -qx "$step" "$result.clean" || missing="$missing $step"
done
if test -n "$missing"; then
  echo "$prog: missing steps:$missing" >&2
  echo "$prog: completed:" >&2
  sed 's/^/  /' "$result.clean" >&2
  exit 1
fi

cmp "$work/BIG.BIN" "$root/BIG.orig" \
  || fail "DOS repair output differs from the original"

#  Verify the DOS-generated set on the host.
( cd "$work" && "$native_xpar" verify --strong DOSMADE.xpa ) \
  || fail "host cannot verify the DOS-generated set"
echo "$prog: the set MS-DOS wrote verifies on this host"

echo
echo "$prog: MS-DOS check passed (`wc -l < "$result.clean"` steps)"
