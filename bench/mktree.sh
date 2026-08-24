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

# Deterministic corpus with many small files and a few large ones.

set -e

prog=`basename "$0"`
here=`cd \`dirname "$0"\` && pwd`
top=`cd "$here/.." && pwd`

seed=${1:-20260823}
target=${2:-268435456}
dest=${3:-tree}
mkdata=${MKDATA:-$top/tests/mkdata}
test -x "$mkdata" || mkdata=$mkdata.exe
test -x "$mkdata" ||
  { echo "$prog: required tool not built: $mkdata" >&2;  exit 1; }

# Park-Miller RNG; return via $rnd to preserve state.
rng=$(( seed % 2147483647 ))
test "$rng" -gt 0 || rng=1
rnd() {
  _hi=$(( rng / 127773 ))
  _lo=$(( rng % 127773 ))
  rng=$(( 16807 * _lo - 2836 * _hi ))
  test "$rng" -gt 0 || rng=$(( rng + 2147483647 ))
  rnd=$(( (rng / 32768) % $1 ))
}

rm -rf "$dest"
mkdir -p "$dest"

total=0
n=0
dirs=12
d=0
while test "$d" -lt "$dirs"; do
  mkdir -p "$dest/d`printf %02d $d`/sub"
  d=$(( d + 1 ))
done

# Backup-like distribution: median near 8 KiB, tail up to a few MiB.
while test "$total" -lt "$target"; do
  rnd 1000;  bucket=$rnd
  if test "$bucket" -lt 880; then
    lo=512;       span=32
  elif test "$bucket" -lt 980; then
    lo=16384;     span=32
  else
    lo=1048576;   span=8
  fi
  rnd $span;  mult=$rnd
  bytes=$(( lo * (mult + 1) ))
  left=$(( target - total ))
  test "$bytes" -le "$left" || bytes=$left
  test "$bytes" -gt 0 || break

  rnd $dirs;  d=$rnd
  rnd 2;  sub=$rnd
  if test "$sub" -eq 0; then
    path=$dest/d`printf %02d $d`/f`printf %05d $n`.bin
  else
    path=$dest/d`printf %02d $d`/sub/f`printf %05d $n`.bin
  fi
  pat=random
  test "$bucket" -lt 40 && pat=text
  "$mkdata" $(( seed + n )) "$bytes" "$path" --pattern=$pat
  total=$(( total + bytes ))
  n=$(( n + 1 ))
done

# Add bounded duplicates for deduplication tests.
i=0
cap=$(( target / 32 ))
test "$cap" -gt 65536 || cap=65536
find "$dest" -type f -size +64k -size -$(( cap / 1024 + 1 ))k |
  sort | head -8 > "$dest/.big" 2>/dev/null
while test "$i" -lt 8; do
  rnd $dirs;  d=$rnd
  src=`sed -n "$(( i + 1 ))p" "$dest/.big"`
  test -n "$src" || break
  cp "$src" "$dest/d`printf %02d $d`/dup`printf %02d $i`.bin" || true
  i=$(( i + 1 ))
done
rm -f "$dest/.big"

files=`find "$dest" -type f | wc -l | tr -d ' '`
bytes=`find "$dest" -type f -printf '%s\n' 2>/dev/null |
        awk '{s+=$1} END {print s+0}'`
test -n "$bytes" && test "$bytes" -gt 0 || bytes=$total
echo "$prog: created $files files ($bytes bytes) in $dest"
