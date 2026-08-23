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

# Property-based fault injection with outcomes predicted from geometry.
# Each corruption matrix is reproducible from XPAR_TEST_SEED.

. "${srcdir:-.}/lib.sh" 2> /dev/null || . "`dirname "$0"`/lib.sh"

rounds_for() {
  case $XPAR_TEST_LEVEL in
    quick) echo "$1" ;;
    full)  echo `expr $1 \* 3` ;;
    *)     echo `expr $1 \* 8` ;;
  esac
}

#  cells_depth <file-of-"slice,column"-lines>
#  Prints "<deepest column> <distinct cells>".
cells_depth() {
  sort -u "$1" | awk -F, '
    { d[$2]++ }
    END { m = 0
          for (c in d) if (d[c] > m) m = d[c]
          print m, NR }'
}

# A successful repair must reproduce the original bytes.
check_repair_outcome() {   # <status> <file> <pristine> <predicted 0|1> <what>
  _st=$1;  _got=$2;  _want=$3;  _pred=$4;  _what=$5
  if test "$_st" -eq 0; then
    if cmp -s "$_got" "$_want"; then ok
    else bad "$_what: repair exited 0 but the bytes are not the original"; fi
    if test "$_pred" -eq 0; then
      bad "$_what: repair succeeded past the recovery bound"
    else ok; fi
  else
    if test "$_pred" -eq 1; then
      bad "$_what: repair failed (status $_st) inside the recovery bound"
    else ok; fi
  fi
}

# Exact per-column profiles with known expected depth.

family_cells() {   # family_cells <label> <bytes> <rounds> <create options...>
  label=$1;  bytes=$2;  rounds=$3;  shift 3
  step "profiles: $label"

  rm -rf f1;  mkdir f1;  cd f1 || hard_error cd
  mkfile data.bin "$bytes"
  cp data.bin pristine.bin
  if ! run 0 "$XPAR" create --dedup=none --align=none -o set "$@" data.bin
  then cd ..;  return 0; fi
  read_geometry set.xpa
  note "Z=$Z S=$S Y=$Y K=$K R=$R"

  round=1
  while test "$round" -le "$rounds"; do
    cp pristine.bin data.bin
    : > cells.txt
    ops=
    profile=
    j=0
    while test "$j" -lt "$K"; do
      #  Zero happens often on purpose: an undamaged column must not be
      #  charged to the budget.
      want=`rnd \`expr $R + 2\``
      test "$want" -le "$S" || want=$S
      profile="$profile $want"
      i=0
      while test "$i" -lt "$want"; do
        #  Slices are taken from the front so the ragged last slice, whose
        #  trailing cells hold no bytes, is never asked for.
        ops="$ops cell=$i,$j"
        echo "$i,$j" >> cells.txt
        i=`expr $i + 1`
      done
      j=`expr $j + 1`
    done

    set -- `cells_depth cells.txt`
    depth=${1:-0};  cells=${2:-0}
    if test "$depth" -le "$R"; then pred=1;  else pred=0; fi
    what="round $round, profile [$profile ], depth $depth against R=$R"

    if test "$cells" -eq 0; then
      run 0 "$XPAR" verify set.xpa
      round=`expr $round + 1`
      continue
    fi

    # shellcheck disable=SC2086
    "$DAMAGE" data.bin -Z "$Z" -Y "$Y" -n 96 seed=$XPAR_TEST_SEED $ops ||
      hard_error "damage failed"

    "$XPAR" verify --json set.xpa > v.json 2> "$log"
    equal "$what: reported depth" "`json_num v.json column_depth summary`" \
          "$depth"
    equal "$what: reported cells" "`json_num v.json cells_bad summary`" \
          "$cells"
    if test "$pred" -eq 1; then
      equal "$what: verify status" \
            "`json_str v.json status summary`" repairable
      run 1 "$XPAR" verify set.xpa
    else
      equal "$what: verify status" \
            "`json_str v.json status summary`" unrepairable
      run 2 "$XPAR" verify set.xpa
    fi

    attempt "$XPAR" repair --in-place set.xpa || { cd ..;  return 0; }
    check_repair_outcome "$status" data.bin pristine.bin "$pred" "$what"
    if test "$status" -eq 0; then run 0 "$XPAR" verify --strong set.xpa; fi
    round=`expr $round + 1`
  done
  cd ..
}

# Unaligned bursts that may cross cell or slice boundaries.

family_bursts() {   # <label> <bytes> <rounds> <create options...>
  label=$1;  bytes=$2;  rounds=$3;  shift 3
  step "bursts: $label"

  rm -rf f2;  mkdir f2;  cd f2 || hard_error cd
  mkfile data.bin "$bytes"
  cp data.bin pristine.bin
  if ! run 0 "$XPAR" create --dedup=none --align=none -o set "$@" data.bin
  then cd ..;  return 0; fi
  read_geometry set.xpa
  note "Z=$Z S=$S Y=$Y K=$K R=$R L=$L"

  round=1
  while test "$round" -le "$rounds"; do
    cp pristine.bin data.bin
    : > cells.txt
    ops=
    bursts=`expr 1 + \` rnd 6\``
    b=0
    while test "$b" -lt "$bursts"; do
      #  Offsets are drawn in cell-sized steps plus a jitter, so a burst
      #  lands on a boundary far more often than uniform noise would.
      off=`expr \` rnd $S\` \* $Z + \` rnd $K\` \* $Y`
      off=`expr $off + \` rnd 3\` \* \( $Y - 8 \) / 2`
      len=`expr 1 + \` rnd 200\``
      test "$off" -lt "$L" || off=`expr $L - 1`
      if test `expr $off + $len` -gt "$L"; then len=`expr $L - $off`; fi
      ops="$ops rand=$off,$len"
      p=$off
      end=`expr $off + $len`
      while test "$p" -lt "$end"; do
        s=`expr $p / $Z`
        j=`expr \( $p % $Z \) / $Y`
        echo "$s,$j" >> cells.txt
        p=`expr $s \* $Z + \( $j + 1 \) \* $Y`
      done
      b=`expr $b + 1`
    done

    set -- `cells_depth cells.txt`
    depth=${1:-0};  cells=${2:-0}
    if test "$depth" -le "$R"; then pred=1;  else pred=0; fi
    what="round $round, $bursts bursts, depth $depth against R=$R"

    # shellcheck disable=SC2086
    "$DAMAGE" data.bin seed=$XPAR_TEST_SEED $ops || hard_error "damage failed"

    "$XPAR" verify --json set.xpa > v.json 2> "$log"
    equal "$what: reported depth" "`json_num v.json column_depth summary`" \
          "$depth"
    equal "$what: reported cells" "`json_num v.json cells_bad summary`" \
          "$cells"

    attempt "$XPAR" repair --in-place set.xpa || { cd ..;  return 0; }
    check_repair_outcome "$status" data.bin pristine.bin "$pred" "$what"
    round=`expr $round + 1`
  done
  cd ..
}

# Missing or damaged recovery volumes.

family_recovery() {   # family_recovery <label> <bytes> <create options...>
  label=$1;  bytes=$2;  shift 2
  step "recovery data: $label"

  rm -rf f3;  mkdir f3;  cd f3 || hard_error cd
  mkfile data.bin "$bytes"
  cp data.bin pristine.bin
  if ! run 0 "$XPAR" create --dedup=none --align=none -o set "$@" data.bin
  then cd ..;  return 0; fi
  read_geometry set.xpa
  note "Z=$Z S=$S Y=$Y K=$K R=$R"

  vols=`ls set.v*.xpa 2> /dev/null`
  if test -z "$vols"; then
    note "no separate recovery volumes in this layout"
    cd ..;  return 0
  fi
  cp set.xpa index.orig
  for v in $vols; do cp "$v" "$v.orig"; done

  #  Intact data with damaged recovery is not a data emergency, but it
  #  must never be reported as a clean set either.
  victim=`echo $vols | tr ' ' '\n' | head -1`
  "$DAMAGE" "$victim" "rand=200,4096" || hard_error "damage failed"
  run_any "0 1 2" "$XPAR" verify set.xpa
  run_any "0 1 2" "$XPAR" scrub --deep set.xpa
  same data.bin pristine.bin
  for v in $vols; do cp "$v.orig" "$v"; done

  #  A missing volume lowers the budget by exactly the slices it held.
  rm -f "$victim"
  run 0 "$XPAR" verify set.xpa
  run 0 "$XPAR" recover --volume="`basename "$victim"`" set.xpa
  exists "$victim"
  run 0 "$XPAR" verify set.xpa

  #  Damage the data to the full budget with a damaged recovery volume in
  #  place: whatever the tool decides, a zero status has to mean the
  #  bytes came back.
  cp pristine.bin data.bin
  ops=
  i=0
  while test "$i" -lt "$R"; do ops="$ops cell=$i,0";  i=`expr $i + 1`; done
  # shellcheck disable=SC2086
  "$DAMAGE" data.bin -Z "$Z" -Y "$Y" -n 96 $ops || hard_error "damage failed"
  "$DAMAGE" "$victim" "rand=300,8192" || hard_error "damage failed"
  attempt "$XPAR" repair --in-place set.xpa || { cd ..;  return 0; }
  if test "$status" -eq 0; then same data.bin pristine.bin
  else note "repair refused with a damaged recovery volume (status $status)"
       ok; fi

  cd ..
}

# Test matrix.

n=`rounds_for 5`

family_cells  "matrix, GF(2^8)"  8388608 "$n" \
              -s 1M -r 4 --codec=matrix --field=8
family_cells  "fft, GF(2^8)"     4194304 "$n" \
              -s 512K -r 3 --codec=fft --field=8
family_bursts "matrix, GF(2^8)"  8388608 "$n" \
              -s 1M -r 4 --codec=matrix --field=8
family_bursts "fft, GF(2^16)"    4194304 "$n" \
              -s 512K -r 3 --codec=fft --field=16

family_recovery "matrix, GF(2^8)" 4194304 -s 512K -r 4 --codec=matrix

if test "$XPAR_TEST_LEVEL" != quick; then
  family_cells  "matrix, GF(2^16), narrow cells" 4194304 "$n" \
                -s 128K -r 5 --codec=matrix --field=16
  family_bursts "matrix, GF(2^16), ragged tail"  2686976 "$n" \
                -s 256K -r 5 --codec=matrix --field=16
  family_recovery "fft, GF(2^8), ladder volumes" 8388608 \
                  -s 1M -r 6 --codec=fft --volumes=ladder
fi

summary
