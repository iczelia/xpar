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

#  Regressions for defects found by audit. Each case states the contract
#  that was broken, so a rewrite that satisfies it differently still
#  passes and only a return of the original behaviour fails.

. "${srcdir:-.}/lib.sh"

#  Run a command with stdout captured to a file rather than the log, so a
#  verb that writes bytes can have both its status and its output judged.
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

step "extract --stdout must not emit unverified bytes"

#  --stdout took a shortcut past the manifest validation and the entry
#  content hash, so a corrupted stream was written out with status 0.
mkdir -p e1 && cd e1 || hard_error "cd e1"
mkfile data.bin 400000
cp data.bin pristine.bin
run 0 "$XPAR" create -r 4 -s 8K --layout=armoured -o set data.bin
rm -f data.bin
read_geometry set.xpa

#  More damaged slices than there is recovery to rebuild them.
ops=""
i=0
while test "$i" -lt 12; do
  ops="$ops rand=`expr 2048 + $i \* $Z \* 2`,64"
  i=`expr $i + 1`
done
"$DAMAGE" set.xpa $ops || hard_error "damage failed"

capture out.bin "$XPAR" extract --stdout set.xpa
if test "$status" -eq 0; then
  note "extract --stdout reported success; its bytes must then be correct"
  same out.bin pristine.bin
else
  note "extract --stdout refused the damaged stream (status $status)"
  #  Refusing after emitting is the same defect wearing a status code.
  equal "the refusal emitted nothing" "`wc -c < out.bin | tr -d ' '`" 0
fi
cd .. || hard_error cd

step "extract --stdout still works on an intact set"

mkdir -p e2 && cd e2 || hard_error "cd e2"
mkfile data.bin 400000
cp data.bin pristine.bin
run 0 "$XPAR" create -r 4 -s 8K --layout=armoured -o set data.bin
rm -f data.bin
capture out.bin "$XPAR" extract --stdout set.xpa
equal "intact extract status" "$status" 0
same out.bin pristine.bin
cd .. || hard_error cd

step "a substituted data volume is rewritten from chain-space offsets"

#  xpar_vol.stream_offset is relative to the generation, but xpar_vset_read
#  takes a chain-space offset. Without stream_base a generation past the
#  first read the wrong bytes, or refused the read outright.
mkdir -p v1 && cd v1 || hard_error "cd v1"
mkdir tree
mkfile tree/a.bin 60000 11
mkfile tree/b.bin 60000 22
run 0 "$XPAR" create -r 4 -s 8K --layout=split -o set -R tree
mkfile tree/c.bin 40000 33
run 0 "$XPAR" add -r 4 set.xpa -R tree

test -f set.g001.d00 || hard_error "split chain produced no set.g001.d00"
base=`"$XPAR" info --json set.xpa 2> /dev/null |
        sed -n 's/.*"stream_base":\([0-9][0-9]*\).*/\1/p' | head -1`
equal "generation 1 starts past the origin" "`test "${base:-0}" -gt 0 &&
                                              echo yes || echo no`" yes

#  The named volume is corrupt and an intact copy sits under another name,
#  which is what makes repair rewrite the named one.
cp set.g001.d00 orig.d00
cp set.g001.d00 spare.dat
"$DAMAGE" set.g001.d00 rand=100,512 || hard_error "damage failed"

run_any "0 1" "$XPAR" repair --in-place set.xpa
same set.g001.d00 orig.d00
run 0 "$XPAR" verify set.xpa
cd .. || hard_error cd

step "recover reproduces a volume the writer replicated into"

#  A critical group past the replication threshold is carried only by the
#  first volume, the last, and the power-of-two indices. recover has to
#  reach the same verdict as the writer: it thresholds on the armoured
#  size, and counts recovery volumes only, which a split LAYT interleaves
#  with data volumes. Getting either wrong drops the group silently.
mkdir -p r1 && cd r1 || hard_error "cd r1"
mkdir tree
mkfile tree/payload.bin 400000 44

#  The threshold has a 1 MiB floor, so only a large manifest crosses it.
pad=nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn
pad=$pad$pad$pad
i=0
while test "$i" -lt 3000; do
  printf x > "tree/f$i.$pad.txt"
  i=`expr $i + 1`
done

run 0 "$XPAR" create -r 16 -s 4K --volumes=8 --layout=split -o set -R tree

vols=`find . -maxdepth 1 -name 'set.v*' | sort`
test -n "$vols" || hard_error "no recovery volumes were written"

#  Only worth asserting if the group really did exceed the threshold, which
#  shows as some volumes carrying it and others not.
big=0;  small=0
for n in $vols; do
  if test `wc -c < "$n"` -gt 1000000; then big=`expr $big + 1`
  else small=`expr $small + 1`; fi
done
if test "$big" -eq 0 || test "$small" -eq 0; then
  bad "every volume replicated alike: the critical group no longer crosses
       the replication threshold, so this case proves nothing. Enlarge the
       manifest until some volumes carry the group and others do not."
else
  ok
fi

for n in $vols; do cp "$n" "orig-`basename $n`"; done
for n in $vols; do
  rm -f "$n"
  run 0 "$XPAR" recover --volume="`basename $n`" set.xpa
  same "$n" "orig-`basename $n`"
done
run 0 "$XPAR" verify set.xpa
cd .. || hard_error cd

step "the hand-recovery recipe explain prints actually recovers the data"

#  The recipe is the promise that the format survives the loss of this
#  tool, so it has to run. Piping into `dd count=1` did not: reading a
#  pipe, dd stops at the first short read and truncates the frame.
#  A frame narrower than a pipe buffer always arrives in one read, so
#  only a wide frame can catch the truncation. GF(2^8) frames are 223
#  bytes and never can; the size below is what makes GF(2^16) wide.
big_frame=no
for field in 8 16; do
  mkdir -p x$field && cd x$field || hard_error "cd x$field"
  mkfile p.bin 100000 55
  if run 0 "$XPAR" create -r 20% --layout=armoured \
                   --armour-field=$field --armour-t=16 -o p p.bin
  then
    "$XPAR" explain p.xpa 2> "$log" | sed -n '/^set -e$/,$p' > recipe.sh
    fd=`sed -n 's/.*plaintext bytes per frame = \([0-9][0-9]*\).*/\1/p' \
          recipe.sh | head -1`
    if test -n "$fd" && test "$fd" -gt 65536; then big_frame=yes; fi
    if test -s recipe.sh; then
      if "$XPAR_SH" recipe.sh > "$log" 2>&1; then
        same recovered.bin p.bin
      else
        bad "the GF(2^$field) recipe did not run to completion"
      fi
    else
      bad "explain printed no recipe for the GF(2^$field) archive"
    fi
  fi
  cd .. || hard_error cd
done
if test "$big_frame" = yes; then ok
else
  bad "no frame exceeded a pipe buffer, so the short-read case that the
       recipe used to hit was never exercised"
fi

step "prune: refuses a lossy removal, and performs a forced one"

#  prune had no coverage at all, though it is destructive and its -f
#  semantics decide whether entries survive.
mkdir -p p1 && cd p1 || hard_error "cd p1"
mkdir tree
mkfile tree/a.bin 80000 66
run 0 "$XPAR" create -r 4 -s 8K -o set -R tree
mkfile tree/a.bin 90000 77
run 0 "$XPAR" add -r 4 set.xpa -R tree
cp tree/a.bin pristine.bin

gens() {
  "$XPAR" info --json set.xpa 2> /dev/null |
    sed -n 's/.*"generations":\([0-9][0-9]*\).*/\1/p' | head -1
}
#  Contents, not just names: a dry run that rewrote a volume in place
#  would leave the listing identical. cksum is POSIX.
files() {
  find . -maxdepth 1 -name 'set*' | sort | while read _f; do
    printf '%s:%s ' "$_f" "`cksum < "$_f" | tr -d ' '`"
  done
}

equal "chain length before pruning" "`gens`" 2
snapshot=`files`

#  A generation a survivor still depends on is refused, and nothing moves.
run 4 "$XPAR" prune --dry-run --before=1 set.xpa
equal "dry run changed nothing" "`files`" "$snapshot"
run 4 "$XPAR" prune --before=1 set.xpa
equal "refusal changed nothing" "`files`" "$snapshot"
equal "chain length after refusal" "`gens`" 2
run 0 "$XPAR" verify set.xpa

#  --force accepts the loss, and what survives has to remain coherent.
run 0 "$XPAR" prune -f --before=1 set.xpa
equal "chain collapsed to one generation" "`gens`" 1
run 0 "$XPAR" verify set.xpa
#  A sidecar set protects the files in place, so the survivor is on disk.
same tree/a.bin pristine.bin
cd .. || hard_error cd

summary
