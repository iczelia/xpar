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

#  Print reproducible-set digests for each layout.
#
#    repro-digest.sh <xpar> <mkdata> [work-dir]
#
#  Compare the output across architectures.

set -e
xpar=$1;  mkdata=$2;  work=${3:-repro-work}

test -x "$xpar" || {
  echo "repro-digest: xpar is not executable: $xpar" >&2;  exit 2;
}
test -x "$mkdata" || {
  echo "repro-digest: mkdata is not executable: $mkdata" >&2;  exit 2;
}

#  Resolve tools before entering the work directory.
case $xpar   in /*) ;;  *) xpar=`pwd`/$xpar ;;  esac
case $mkdata in /*) ;;  *) mkdata=`pwd`/$mkdata ;;  esac

sum() {
  if command -v sha256sum > /dev/null 2>&1; then sha256sum
  else shasum -a 256
  fi
}

rm -rf "$work";  mkdir -p "$work"
#  Fixed fixture corpus.
"$mkdata" 4242 3000000 "$work/corpus.bin" --pattern=random

for lay in sidecar split armoured; do
  rm -f "$work"/set.*
  ( cd "$work" && "$xpar" create --reproducible --layout=$lay \
                          -r 20% -s 64K --field=8 --codec=matrix \
                          -o set corpus.bin ) > /dev/null 2>&1
  d=`cat "$work"/set.xpa "$work"/set.v*.xpa "$work"/set.d* 2> /dev/null |
       sum | cut -d' ' -f1`
  echo "$lay $d"
done
