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

#  Verify performance-sensitive batching and planner reporting.

. "${srcdir:-.}/lib.sh"

capture() {   # capture <outfile> <cmd>...
  _out=$1;  shift
  status=0
  "$@" > "$_out" 2> "$log" || status=$?
  if test "$status" -ge 128 || test "$status" -eq 8; then
    bad "$* : `explain_status $status` (status $status)"
    return 1
  fi
  return 0
}

#  Matching archives need the same output name in separate directories.

step "the armoured writer batches frames without moving a byte"

#  Batch size and worker count must not affect the archive.
mkdir -p b1 && cdto b1
mkfile p.bin 3000000
in=`pwd`/p.bin
rm -rf ref && mkdir ref
run 0 sh -c "cd ref && '$XPAR' create --reproducible --no-verify-after \
  --layout=armoured -s 16K -r 20% -j1 -m 512M -o a '$in'"
for opt in "-j1 -m 2M" "-j2 -m 8M" "-j4 -m 32M" "-j8 -m 64M" "-m 1G"; do
  rm -rf v && mkdir v
  #  Deliberate word splitting for option pairs.
  run 0 sh -c "cd v && '$XPAR' create --reproducible --no-verify-after \
    --layout=armoured -s 16K -r 20% $opt -o a '$in'"
  same v/a.xpa ref/a.xpa
done
run 0 "$XPAR" verify ref/a.xpa
cdto ..

step "batching preserves every armour parameter and its correction"

#  Batch boundaries must preserve codewords.
mkdir -p b2 && cdto b2
mkfile q.bin 2000000
in=`pwd`/q.bin
for opt in "--armour-field=8" "--armour-field=16" "--depth=4" \
           "--armour-t=32" "--burst=1K"; do
  for d in one many; do
    rm -rf $d && mkdir $d
    run 0 sh -c "cd $d && '$XPAR' create --reproducible --no-verify-after \
      --layout=armoured -s 16K -r 20% -m 256M $opt \
      `test $d = one && echo -j1 || echo -j8` -o a '$in'"
  done
  same one/a.xpa many/a.xpa
  run 0 "$XPAR" verify one/a.xpa
  damage one/a.xpa rand=100000,64
  run_any "0 1" "$XPAR" verify one/a.xpa
done
cdto ..

step "--layout=armoured refuses an armour level it cannot write"

#  The armoured layout has no unprotected form.
mkdir -p b3 && cdto b3
mkfile r.bin 200000
run 4 "$XPAR" create --layout=armoured --armour=none -r 10% -o n r.bin
run 4 "$XPAR" create --layout=armoured --armour=metadata -r 10% -o m r.bin
if test -e n.xpa || test -e m.xpa; then
  bad "rejected armour level wrote an archive"
else ok; fi
in=`pwd`/r.bin
for d in exp def; do
  rm -rf $d && mkdir $d
  run 0 sh -c "cd $d && '$XPAR' create --reproducible --layout=armoured \
    -r 10% `test $d = exp && echo --armour=all` -o a '$in'"
done
same exp/a.xpa def/a.xpa
#  Other layouts accept every level.
run 0 "$XPAR" create --armour=none -r 10% -o s1 r.bin
run 0 "$XPAR" create --armour=metadata -r 10% -o s2 r.bin
run 0 "$XPAR" create --layout=split --volumes=2 --armour=none -r 10% \
        -o s3 r.bin
cdto ..

step "the plan reports what -m does not bound"

mkdir -p b4 && cdto b4
mkfile s.bin 2000000
capture out "$XPAR" create -v --no-verify-after -m 8M -r 10% -o t s.bin
equal "create -v status" "$status" 0
if grep -q 'tables + buffers' "$log"; then ok
else bad "fixed memory cost missing"; fi
if grep -q 'of which -m bounds' "$log"; then ok
else bad "budgeted memory total missing"; fi
#  Decode reports include loaded volume images.
capture out2 "$XPAR" info -v t.xpa
if grep -q 'volume images' out2; then ok
else bad "volume-image memory missing"; fi
images=`sed -n 's/.*volume images \([0-9.]*\) *[KMGT]*i*B.*/\1/p' out2 |
        head -1`
if test -n "$images"; then ok
else bad "invalid volume-image size"; fi
capture out3 "$XPAR" create -v --no-verify-after --layout=armoured -m 8M \
        -r 10% -o u s.bin
equal "armoured create -v status" "$status" 0
if grep -q 'armour frames [1-9]' "$log"; then ok
else bad "armour batch not budgeted"; fi
cdto ..

step "a multi-pass plan says how much it will read"

mkdir -p b5 && cdto b5
mkfile u.bin 4000000
capture out "$XPAR" create --no-verify-after -v -s 256K -m 2M -r 200% \
        -o w u.bin
equal "many-pass create status" "$status" 0
if grep -q 'passes read .* total' "$log"; then ok
else bad "total read volume missing"; fi
if grep -q 'warning: .* passes read .* makes one pass' "$log"; then ok
else bad "one-pass budget warning missing"; fi
#  Verify the displayed budget using xpar's one-letter IEC syntax.
want=`sed -n 's/.*-m \([0-9.]*\) \([KMG]\)iB makes one pass.*/\1\2/p' \
      "$log" | head -1`
if test -n "$want"; then
  rm -f w.xpa w.v*.xpa
  capture out2 "$XPAR" create --no-verify-after -v -s 256K -m "$want" \
          -r 200% -o w u.bin
  if grep -q 'passes     : 1 ' "$log"; then ok
  else bad "-m $want was named as one-pass but is not"; fi
else bad "one-pass warning has no budget"; fi
#  A roomy budget stays silent.
rm -f w.xpa w.v*.xpa
capture out3 "$XPAR" create --no-verify-after -s 256K -m 512M -r 200% \
        -o w u.bin
if grep -q 'passes read' "$log"; then
  bad "one-pass plan emitted a pass warning"
else ok; fi
cdto ..

summary
