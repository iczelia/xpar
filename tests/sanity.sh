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

# Run the release sanity check under make check.

prog=`basename "$0"`

: "${abs_top_builddir:=..}"
: "${abs_top_srcdir:=..}"

if test -z "${XPAR:-}"; then
  if   test -x "$abs_top_builddir/xpar";     then XPAR=$abs_top_builddir/xpar
  elif test -x "$abs_top_builddir/xpar.exe"; then
    XPAR=$abs_top_builddir/xpar.exe
  else echo "$prog: no xpar binary" >&2;  exit 99
  fi
fi

script=$abs_top_srcdir/contrib/ci-check.sh
test -r "$script" || { echo "$prog: $script is missing" >&2;  exit 99; }

# The sanity check requires random damage input.
test -r /dev/urandom || { echo "$prog: SKIP: no /dev/urandom" >&2;  exit 77; }

# Resolve the fixture before changing directory.
if test -n "${XPAR_COMPAT:-}"; then
  case $XPAR_COMPAT in
    /*|?:[/\\]*) ;;
    *) XPAR_COMPAT=`pwd`/$XPAR_COMPAT ;;
  esac
fi

work=`pwd`/tw-sanity.$$
rm -rf "$work"
mkdir "$work" || { echo "$prog: cannot create $work" >&2;  exit 99; }
trap 'cd /; rm -rf "$work"' EXIT HUP INT TERM
cd "$work" || exit 99

# Use the cross-host fixture when available.
if test -n "${XPAR_COMPAT:-}" && test -d "$XPAR_COMPAT"; then
  exec "${SHELL:-/bin/sh}" "$script" "$XPAR" "$XPAR_COMPAT"
fi
exec "${SHELL:-/bin/sh}" "$script" "$XPAR"
