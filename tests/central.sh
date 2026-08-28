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

# End-to-end tests for per-column recovery capacity.
# One unaligned, undeduplicated file keeps stream and file offsets equal.

. "${srcdir:-.}/lib.sh" 2> /dev/null || . "`dirname "$0"`/lib.sh"

# damage_profile <file> <depth> <columns...>
#   Erase `depth` cells in each named column, at slices 0 .. depth - 1.
damage_profile() {
  _file=$1;  _depth=$2;  shift 2
  _ops=
  for _col in "$@"; do
    _i=0
    while test "$_i" -lt "$_depth"; do
      _ops="$_ops cell=$_i,$_col"
      _i=`expr $_i + 1`
    done
  done
  test -n "$_ops" || return 0
  # shellcheck disable=SC2086
  damage "$_file" -Z "$Z" -Y "$Y" -n 96 seed=$XPAR_TEST_SEED $_ops
}

all_columns() {
  _j=0;  _out=
  while test "$_j" -lt "$K"; do _out="$_out $_j";  _j=`expr $_j + 1`; done
  echo "$_out"
}

# one_config <label> <bytes> <create options...>
one_config() {
  label=$1;  bytes=$2;  shift 2
  step "$label"

  rm -rf c;  mkdir c;  cdto c
  mkfile data.bin "$bytes"
  cp data.bin pristine.bin

  if ! run 0 "$XPAR" create --dedup=none --align=none -o set "$@" data.bin
  then cd ..;  return 0; fi

  read_geometry set.xpa
  note "Z=$Z S=$S Y=$Y K=$K R=$R L=$L"
  if test "$K" -lt 2; then
    #  With no columns, test whole-slice recovery.
    note "K is 1 here, so the erasure unit is the whole slice"
    run 0 "$XPAR" verify set.xpa
    damage_profile data.bin 1 0
    run 1 "$XPAR" verify set.xpa
    run 0 "$XPAR" repair --in-place set.xpa
    same data.bin pristine.bin
    cd ..;  return 0
  fi
  if test "$R" -ge "$S"; then
    note "R >= S, so a column cannot be pushed past its budget"
  fi
  columns=`all_columns`

  # -- The capacity law: R erasures in every column at once. ------------
  cp pristine.bin data.bin
  # shellcheck disable=SC2086
  damage_profile data.bin "$R" $columns
  "$XPAR" verify --json set.xpa > v.json 2> "$log"
  equal "verify status" "`json_str v.json status summary`" repairable
  equal "cells reported bad" "`json_num v.json cells_bad summary`" \
        "`expr $K \* $R`"
  equal "deepest column" "`json_num v.json column_depth summary`" "$R"
  equal "recovery needed" "`json_num v.json recovery_needed summary`" "$R"
  run 1 "$XPAR" verify set.xpa
  run 0 "$XPAR" repair --in-place set.xpa
  same data.bin pristine.bin
  run 0 "$XPAR" verify --strong set.xpa
  note "K*R = `expr $K \* $R` cells lost and recovered"

  # One column beyond its budget.
  if test "$R" -lt "$S"; then
    cp pristine.bin data.bin
    rnd "$K";  victim=$rnd
    others=
    for j in $columns; do
      test "$j" -eq "$victim" || others="$others $j"
    done
    # shellcheck disable=SC2086
    damage_profile data.bin "$R" $others
    damage_profile data.bin `expr $R + 1` "$victim"
    "$XPAR" verify --json set.xpa > v.json 2> "$log"
    equal "verify status" "`json_str v.json status summary`" unrepairable
    equal "deepest column" "`json_num v.json column_depth summary`" \
          "`expr $R + 1`"
    equal "recovery needed" "`json_num v.json recovery_needed summary`" \
          "`expr $R + 1`"
    equal "recovery available" \
          "`json_num v.json recovery_available summary`" "$R"
    run 2 "$XPAR" verify set.xpa
    run 2 "$XPAR" repair --in-place set.xpa
    #  Refusing is required; quietly writing wrong bytes is not.
    if cmp -s data.bin pristine.bin; then
      bad "repair refused yet the file was restored, which cannot be"
    else
      ok
    fi
    note "one column at R+1 = `expr $R + 1` was refused, the rest untouched"
  fi

  # -- Whole slices: K cells each, one erasure of budget each. ----------
  cp pristine.bin data.bin
  ops=
  i=0
  while test "$i" -lt "$R"; do
    j=0
    while test "$j" -lt "$K"; do
      ops="$ops cell=$i,$j"
      j=`expr $j + 1`
    done
    i=`expr $i + 1`
  done
  # shellcheck disable=SC2086
  damage data.bin -Z "$Z" -Y "$Y" -n 96 seed=$XPAR_TEST_SEED $ops
  "$XPAR" verify --json set.xpa > v.json 2> "$log"
  equal "deepest column with R whole slices" \
        "`json_num v.json column_depth summary`" "$R"
  run 0 "$XPAR" repair --in-place set.xpa
  same data.bin pristine.bin

  # -- A burst across a cell boundary marks both cells. -----------------
  cp pristine.bin data.bin
  damage data.bin "rand=`expr $Y - 8`,16"
  "$XPAR" verify --json set.xpa > v.json 2> "$log"
  equal "a burst across a cell boundary" \
        "`json_num v.json cells_bad summary`" 2
  equal "at depth one" "`json_num v.json column_depth summary`" 1
  run 0 "$XPAR" repair --in-place set.xpa
  same data.bin pristine.bin

  # -- A burst across a slice boundary is one erasure in two columns. ---
  if test "$S" -gt 1; then
    cp pristine.bin data.bin
    damage data.bin "rand=`expr $Z - 8`,16"
    "$XPAR" verify --json set.xpa > v.json 2> "$log"
    equal "a burst across a slice boundary" \
          "`json_num v.json cells_bad summary`" 2
    equal "still at depth one" "`json_num v.json column_depth summary`" 1
    run 0 "$XPAR" repair --in-place set.xpa
    same data.bin pristine.bin
  fi

  cd ..
}

step "the tool reports a cell geometry at all"
mkfile probe.bin 4194304
run 0 "$XPAR" create -s 1M -r 2 --dedup=none -o probe probe.bin
"$XPAR" info probe.xpa > info.txt 2> "$log"
if grep -q 'erasure budget is .* per column' info.txt ||
   grep -q 'the erasure unit is' info.txt; then ok
else bad "info does not explain the per-column erasure budget"; fi

one_config "matrix, GF(2^8), Z = 1 MiB"    8388608 \
           -s 1M -r 4 --codec=matrix --field=8
one_config "fft, GF(2^8), Z = 512 KiB"     4194304 \
           -s 512K -r 3 --codec=fft --field=8
one_config "matrix, GF(2^16), ragged tail" 2686976 \
           -s 256K -r 5 --codec=matrix --field=16
one_config "fft, GF(2^16), Z = 1 MiB"      8388608 \
           -s 1M -r 2 --codec=fft --field=16

#  Y = Z leaves no column structure.
one_config "matrix, GF(2^8), one cell per slice" 2097152 \
           -s 64K --cell=64K -r 3 --codec=matrix --field=8

if test "$XPAR_TEST_LEVEL" != quick; then
  one_config "matrix, GF(2^8), narrow cells" 4194304 \
             -s 128K -r 6 --codec=matrix --field=8
  one_config "fft, GF(2^8), deep recovery"   8388608 \
             -s 1M -r 7 --codec=fft --field=8
fi

summary
