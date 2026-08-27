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

# Checksum-safety tests: success must reproduce the original bytes.
# CRC-preserving corruption remains detectable only through strong tags.

. "${srcdir:-.}/lib.sh" 2> /dev/null || . "`dirname "$0"`/lib.sh"

# never_false_success <status> <file> <pristine> <what>
never_false_success() {
  if test "$1" -ne 0; then ok;  return 0; fi
  if cmp -s "$2" "$3"; then ok
  else bad "$4: exited 0 with bytes that are not the original"; fi
}

# A forgery the stored checksums cannot see.

step "a CRC-preserving forgery is not mistaken for intact data"

mkdir forge;  cd forge || hard_error cd
mkfile data.bin 2097152
cp data.bin pristine.bin
run 0 "$XPAR" create -s 1M -r 2 --dedup=none --align=none -o set data.bin
read_geometry set.xpa
note "Z=$Z S=$S Y=$Y K=$K R=$R"

# The forged cell preserves both its cell and slice checksums.
"$DAMAGE" data.bin -Z "$Z" -Y "$Y" -k forge cell=0,3 ||
  hard_error "forge failed"
differs data.bin pristine.bin

"$XPAR" verify --json set.xpa > v.json 2> "$log"
equal "cells the checksums condemn" "`json_num v.json cells_bad summary`" 0
equal "entries the checksums condemn" \
      "`json_num v.json entries_damaged summary`" 1
equal "entries with damage no cell explains" \
      "`json_num v.json entries_opaque summary`" 1
equal "entries blamed on an alias" \
      "`json_num v.json entries_alias_only summary`" 0

# It must neither report clean nor promise an impossible repair.
run 2 "$XPAR" verify set.xpa
equal "verify status" "`json_str v.json status summary`" unrepairable
run 2 "$XPAR" verify --strong set.xpa
run_any "1 2" "$XPAR" scrub --deep set.xpa

attempt "$XPAR" repair --in-place set.xpa
never_false_success "$status" data.bin pristine.bin "repair of a forgery"
if test "$status" -eq 0; then
  bad "repair claimed to fix damage no checksum can localise"
else
  ok
  note "repair refused with status $status"
fi
grep -q 'strong tag' "$log" && ok || {
  #  Refusing for another stated reason is fine; refusing silently is not.
  test -s "$log" && ok || bad "repair refused without saying why"
}

# A strong tag must catch a forgery alongside repairable damage.
cp pristine.bin data.bin
"$DAMAGE" data.bin -Z "$Z" -Y "$Y" -k forge cell=1,5 ||
  hard_error "forge failed"
"$DAMAGE" data.bin -Z "$Z" -Y "$Y" -n 96 cell=1,9 ||
  hard_error "damage failed"
"$XPAR" verify --json set.xpa > v.json 2> "$log"
equal "the visible cell is the only one condemned" \
      "`json_num v.json cells_bad summary`" 1
attempt "$XPAR" repair --in-place set.xpa
never_false_success "$status" data.bin pristine.bin \
                    "repair of a slice carrying a forgery"
if test "$status" -eq 0; then
  bad "repair rebuilt a slice whose strong tag cannot match"
else
  ok
  note "the strong tag stopped the write back (status $status)"
fi
cd ..

# Keep alias-local damage distinct from checksum-invisible damage.

step "an alias-local difference is still reported as repairable"

mkdir alias;  cd alias || hard_error cd
mkdir tree
mkfile tree/a.bin 1048576
cp tree/a.bin tree/b.bin
cp -R tree tree.orig
run 0 "$XPAR" create -R --dedup=file -s 256K -r 3 -o set tree
"$DAMAGE" tree/b.bin "rand=100,64" || hard_error "damage failed"
"$XPAR" verify --json set.xpa > v.json 2> "$log"
equal "entries blamed on an alias" \
      "`json_num v.json entries_alias_only summary`" 1
equal "entries with damage no cell explains" \
      "`json_num v.json entries_opaque summary`" 0
run 1 "$XPAR" verify set.xpa
run 0 "$XPAR" repair --in-place set.xpa
same tree/b.bin tree.orig/b.bin
cd ..

# Corrupted checksum metadata must not be trusted or reported clean.

step "damaged metadata is never trusted"

mkdir meta;  cd meta || hard_error cd
mkfile data.bin 8388608
cp data.bin pristine.bin
# Small slices make checksum tables large enough to damage directly.
run 0 "$XPAR" create -s 256K -r 4 --dedup=none -o set data.bin
cp set.xpa index.orig
size=`wc -c < index.orig | tr -d ' '`
note "index volume is $size bytes"

case $XPAR_TEST_LEVEL in
  quick) spots=16 ;;
  full)  spots=48 ;;
  *)     spots=128 ;;
esac

spot=0
while test "$spot" -lt "$spots"; do
  cp index.orig set.xpa
  at=`expr \( $size \* $spot \) / $spots`
  test "$at" -lt `expr $size - 64` || at=`expr $size - 64`
  "$DAMAGE" set.xpa "seed=`expr $XPAR_TEST_SEED + $spot`" "rand=$at,48" ||
    hard_error "damage failed"

  # attempt() rejects internal errors and crashes.
  attempt "$XPAR" verify set.xpa
  vstatus=$status
  if test "$vstatus" -eq 0; then
    # Validate any clean report by extracting it.
    rm -rf out;  mkdir out
    attempt "$XPAR" extract --to out set.xpa
    if test "$status" -eq 0; then
      never_false_success 0 out/data.bin pristine.bin \
                          "extract after a clean verify at offset $at"
    else ok; fi
  else
    ok
  fi
  same data.bin pristine.bin
  spot=`expr $spot + 1`
done
cd ..

# Detect damaged recovery slices before using them.

step "damaged recovery is not decoded from"

mkdir rec;  cd rec || hard_error cd
mkfile data.bin 4194304
cp data.bin pristine.bin
run 0 "$XPAR" create -s 512K -r 4 --dedup=none --volumes=equal -o set data.bin
read_geometry set.xpa
vols=`ls set.v*.xpa 2> /dev/null`
if test -z "$vols"; then
  note "this layout keeps no separate recovery volumes"
else
  for v in $vols; do cp "$v" "$v.orig"; done
  victim=`echo $vols | tr ' ' '\n' | head -1`

  #  Damage inside the recovery payload, then spend the whole budget on
  #  the data.  Either the tool notices and refuses, or it decodes and is
  #  right; there is no third outcome.
  ops=
  i=0
  while test "$i" -lt "$R"; do ops="$ops cell=$i,0";  i=`expr $i + 1`; done
  # shellcheck disable=SC2086
  "$DAMAGE" data.bin -Z "$Z" -Y "$Y" -n 96 $ops || hard_error "damage failed"
  "$DAMAGE" "$victim" "rand=512,16384" || hard_error "damage failed"
  attempt "$XPAR" repair --in-place set.xpa
  never_false_success "$status" data.bin pristine.bin \
                      "repair against damaged recovery"
  note "status $status with a damaged recovery volume"

  #  scrub --deep recomputes the recovery, so it has to say so.
  cp pristine.bin data.bin
  for v in $vols; do cp "$v.orig" "$v"; done
  "$DAMAGE" "$victim" "rand=512,16384" || hard_error "damage failed"
  attempt "$XPAR" scrub --deep set.xpa
  if test "$status" -eq 0; then
    bad "scrub --deep called a set clean whose recovery does not recompute"
  else ok; fi
fi
cd ..

# Shape damage: truncation, extension and replacement.

step "a file of the wrong length is never called intact"

mkdir shape;  cd shape || hard_error cd
mkfile data.bin 2097152
cp data.bin pristine.bin
run 0 "$XPAR" create -s 256K -r 6 --dedup=none -o set data.bin

"$DAMAGE" data.bin "truncate=1900000" || hard_error "truncate failed"
run_any "1 2" "$XPAR" verify set.xpa
attempt "$XPAR" repair --in-place set.xpa
never_false_success "$status" data.bin pristine.bin "repair of a truncation"
if test "$status" -eq 0; then same data.bin pristine.bin; fi

cp pristine.bin data.bin
"$DAMAGE" data.bin "extend=4096" || hard_error "extend failed"
run_any "1 2" "$XPAR" verify set.xpa
attempt "$XPAR" repair --in-place set.xpa
never_false_success "$status" data.bin pristine.bin "repair of an extension"

# Same-length replacement exceeds this set's recovery capacity.
cp pristine.bin data.bin
mkfile other.bin 2097152 999999
cp other.bin data.bin
attempt "$XPAR" verify set.xpa
if test "$status" -eq 0; then bad "a wholly replaced file verified clean"
else ok; fi
attempt "$XPAR" repair --in-place set.xpa
never_false_success "$status" data.bin pristine.bin "repair of a replacement"
cd ..

# A dry run writes nothing.

step "a dry run changes no bytes"

mkdir dry;  cd dry || hard_error cd
mkfile data.bin 1048576
cp data.bin pristine.bin
run 0 "$XPAR" create -s 128K -r 4 --dedup=none -o set data.bin
"$DAMAGE" data.bin "rand=4096,512" || hard_error "damage failed"
cp data.bin damaged.bin
run_any "0 1" "$XPAR" repair --dry-run set.xpa
same data.bin damaged.bin
run 0 "$XPAR" repair --in-place --paranoid set.xpa
same data.bin pristine.bin
cd ..

# Volumes from another set are not recovery data.

step "a volume from another set is refused"

mkdir cross;  cd cross || hard_error cd
mkdir a b
( cd a && mkfile data.bin 1048576 1111 )
( cd b && mkfile data.bin 1048576 2222 )
( cd a && "$XPAR" create -s 128K -r 4 --dedup=none -o set data.bin ) \
  > "$log" 2>&1 || hard_error "create failed"
( cd b && "$XPAR" create -s 128K -r 4 --dedup=none -o set data.bin ) \
  > "$log" 2>&1 || hard_error "create failed"
cp a/data.bin a/pristine.bin
avol=`cd a && ls set.v*.xpa 2> /dev/null | head -1`
if test -z "$avol"; then
  note "no separate recovery volumes to swap"
else
  cp "b/$avol" "a/$avol"
  ( cd a && "$DAMAGE" data.bin "rand=8192,256" ) || hard_error "damage failed"
  cd a || hard_error cd
  attempt "$XPAR" repair --in-place set.xpa
  never_false_success "$status" data.bin pristine.bin \
                      "repair with a foreign recovery volume"
  note "status $status with a foreign volume in place"
  cd ..
fi
cd ..

# Auth-only sets must still distinguish localisable damage.

step "an authenticated set classifies damage the same way"

mkdir auth;  cd auth || hard_error cd
mkfile key.bin 32 4242
mkdir tree
mkfile tree/a.bin 1048576 11
mkfile tree/b.bin 262144 12
cp -R tree tree.orig
run 0 "$XPAR" create -R -s 256K -r 25% --auth-key=key.bin --auth-only \
    --dedup=none -o set tree
run 0 "$XPAR" verify --auth-key=key.bin set.xpa

# A missing key is an authentication failure, not a clean result.
run 6 "$XPAR" verify set.xpa

"$DAMAGE" tree/a.bin "rand=70000,64" || hard_error "damage failed"
"$XPAR" verify --auth-key=key.bin --json set.xpa > v.json 2> "$log"
equal "damage a cell explains" "`json_num v.json entries_opaque summary`" 0
if test "`json_num v.json cells_bad summary`" -ge 1; then ok
else bad "a keyed set did not localise ordinary damage"; fi
run 1 "$XPAR" verify --auth-key=key.bin set.xpa
run 0 "$XPAR" repair --auth-key=key.bin --in-place set.xpa
same tree/a.bin tree.orig/a.bin
cd ..

# Existing chains inherit their layout.

step "an owned-layout chain stays self-contained across add"

for layout in armoured split; do
  mkdir "own-$layout";  cd "own-$layout" || hard_error cd
  mkdir tree
  mkfile tree/a.bin 262144
  mkfile tree/b.bin 65536 2222
  cp -r tree tree.orig
  run 0 "$XPAR" create -s 32K -r 4 --layout="$layout" -o set -R tree
  mkfile tree/c.bin 131072 3333
  run 0 "$XPAR" add -r 4 set.xpa -R tree

  case $layout in
    split)    exists set.g001.d00 ;;
    armoured) exists set.g001.xpa ;;
  esac

  rm -rf tree
  run 0 "$XPAR" extract --to=out set.xpa
  same out/tree/a.bin tree.orig/a.bin
  same out/tree/b.bin tree.orig/b.bin
  exists out/tree/c.bin
  note "$layout: add kept the set extractable without the originals"
  cd ..
done

step "a chain refuses to change layout under it"

mkdir mixed;  cd mixed || hard_error cd
mkdir tree
mkfile tree/a.bin 262144
run 0 "$XPAR" create -s 32K -r 4 --layout=armoured -o set -R tree
mkfile tree/b.bin 65536 2222
run 4 "$XPAR" add -r 4 --layout=sidecar set.xpa -R tree
run 0 "$XPAR" add -r 4 --layout=armoured set.xpa -R tree
exists set.g001.xpa
cd ..

step "consolidate keeps the chain's layout"

mkdir flat;  cd flat || hard_error cd
mkdir tree
mkfile tree/a.bin 262144
cp -r tree tree.orig
run 0 "$XPAR" create -s 32K -r 4 --layout=armoured -o set -R tree
mkfile tree/b.bin 65536 2222
cp tree/b.bin tree.orig/b.bin
run 0 "$XPAR" add -r 4 set.xpa -R tree
run 0 "$XPAR" consolidate --replace set.xpa
rm -rf tree
run 0 "$XPAR" extract --to=out set.xpa
same out/tree/a.bin tree.orig/a.bin
same out/tree/b.bin tree.orig/b.bin
note "the collapsed set is still self-contained"
cd ..

# A substitute must not hide damage to the named data volume.

step "a substituted data volume is never reported as clean"

mkdir subst;  cd subst || hard_error cd

# Renamed volumes remain discoverable.
mkdir a;  cd a || hard_error cd
mkdir tree;  mkfile tree/a.bin 262144
"$XPAR" create -s 32K -r 6 --layout=split -o set -R tree > "$log" 2>&1 ||
  hard_error "create failed"
rm -rf tree
mv set.d00 renamed.bin
run 0 "$XPAR" verify set.xpa
cd ..

# An intact substitute must not make a damaged named volume clean.
mkdir b;  cd b || hard_error cd
mkdir tree;  mkfile tree/a.bin 262144
"$XPAR" create -s 32K -r 6 --layout=split -o set -R tree > "$log" 2>&1 ||
  hard_error "create failed"
rm -rf tree
cp set.d00 spare.bin
"$DAMAGE" set.d00 "rand=4096,512" || hard_error "damage failed"
differs set.d00 spare.bin
run 1 "$XPAR" verify set.xpa
run 0 "$XPAR" repair --in-place set.xpa
same set.d00 spare.bin
run 0 "$XPAR" verify set.xpa
note "the named volume was rewritten from the stream"
cd ..

# Every other verb has to cope with that state too: the stream is whole, so
# nothing may report an internal error over a volume that wants rewriting.
mkdir e;  cd e || hard_error cd
mkdir tree;  mkfile tree/a.bin 262144
"$XPAR" create -s 32K -r 6 --layout=split -o set -R tree > "$log" 2>&1 ||
  hard_error "create failed"
rm -rf tree
cp set.d00 spare.bin
"$DAMAGE" set.d00 "rand=4096,512" || hard_error "damage failed"
mkdir ex rv
run 0 "$XPAR" list set.xpa
run 0 "$XPAR" info set.xpa
run 0 "$XPAR" explain set.xpa
run 0 "$XPAR" extract --to=ex set.xpa
run 0 "$XPAR" addrecovery -r 10 set.xpa
run 0 "$XPAR" recover --volume=0 --to=rv set.xpa
run 1 "$XPAR" scrub set.xpa
note "no verb mistook a rewritable volume for a broken stream"
cd ..

# --to extracts the tree without rewriting the set.
mkdir d;  cd d || hard_error cd
mkdir tree;  mkfile tree/a.bin 262144
"$XPAR" create -s 32K -r 6 --layout=split -o set -R tree > "$log" 2>&1 ||
  hard_error "create failed"
rm -rf tree
cp set.d00 spare.bin
"$DAMAGE" set.d00 "rand=4096,512" || hard_error "damage failed"
cp set.d00 damaged.bin
run 0 "$XPAR" repair --to=out set.xpa
exists out/tree/a.bin
same set.d00 damaged.bin
note "--to wrote the tree and left the damaged volume where it was"
cd ..

# Repair a damaged volume without a substitute.
mkdir c;  cd c || hard_error cd
mkdir tree;  mkfile tree/a.bin 262144
"$XPAR" create -s 32K -r 6 --layout=split -o set -R tree > "$log" 2>&1 ||
  hard_error "create failed"
rm -rf tree
"$DAMAGE" set.d00 "rand=4096,512" || hard_error "damage failed"
run 1 "$XPAR" verify set.xpa
run 0 "$XPAR" repair --in-place set.xpa
run 0 "$XPAR" verify set.xpa
cd ..
cd ..

# Sidecar entries must remain reachable from the set directory.

step "a set can always find the data it just stored"

mkdir reach;  cd reach || hard_error cd
mkdir -p sub a/b/c away
mkfile sub/data.bin 131072
mkfile a/b/c/f.bin 65536 2222
mkfile flat.bin 65536 3333

# Preserve subdirectories for named files and recursive roots.
run 0 "$XPAR" create -s 32K -r 4 -o s1 sub/data.bin
run 0 "$XPAR" verify s1.xpa

run 0 "$XPAR" create -s 32K -r 4 -o s2 -R a/b/c
run 0 "$XPAR" verify s2.xpa

run 0 "$XPAR" create -s 32K -r 4 -o s3 flat.bin
run 0 "$XPAR" verify s3.xpa
run 0 "$XPAR" create -s 32K -r 4 -o s4 -R sub
run 0 "$XPAR" verify s4.xpa

# Refuse sidecar output that cannot reach its data.
run 4 "$XPAR" create -s 32K -r 4 -o away/s5 -R sub
if test -e away/s5.xpa; then bad "a refused create left a set behind"
else ok; fi

# Owned layouts may be written elsewhere.
run 0 "$XPAR" create -s 32K -r 4 --layout=armoured -o away/s6 -R sub
run 0 "$XPAR" create -s 32K -r 4 --layout=split -o away/s7 -R sub

# add refuses unreachable entries without changing the chain.
mkfile sub/more.bin 32768 4444
run 0 "$XPAR" add -r 4 s4.xpa -R sub
run 0 "$XPAR" verify --chain s4.xpa
# Parent paths are unreachable from the set directory.
mkfile ../outside.bin 32768 5555
run 4 "$XPAR" add -r 4 s4.xpa ../outside.bin
run 0 "$XPAR" verify --chain s4.xpa
note "unreachable names are refused before anything is written"
cd ..

# Repairs of missing, truncated, overlong, and damaged files must undo exactly.

step "undo journals restore every file state exactly"

mkdir shortread;  cd shortread || hard_error cd

# Start each case with a fresh set.
jrt() {
  rm -rf "$1";  mkdir -p "$1/tree";  cd "$1" || hard_error cd
  mkfile tree/a.bin 262144
  mkfile tree/b.bin 65536 2222
  cp -r tree keep
  "$XPAR" create -r 300% -s 16K -o s -R tree > "$log" 2>&1 ||
    hard_error "create failed"
}

# Undo removes a file created by repair.
jrt gone
rm -f tree/a.bin
run 0 "$XPAR" repair --in-place --keep-journal s.xpa
same tree/a.bin keep/a.bin
run 0 "$XPAR" undo s.xpa
if test -e tree/a.bin; then bad "undo kept a repair-created file"
else ok; fi
cd ..

# Undo restores a truncated file's length.
jrt cut
head -c 120000 keep/a.bin > tree/a.bin
run 0 "$XPAR" repair --in-place --keep-journal s.xpa
same tree/a.bin keep/a.bin
run 0 "$XPAR" undo s.xpa
equal "undo restored the truncated length" "`nbytes < tree/a.bin`" "120000"
cd ..

# Undo restores an overlong tail.
jrt long
cat keep/a.bin keep/b.bin > tree/a.bin
cp tree/a.bin long.keep
run 0 "$XPAR" repair --in-place --keep-journal s.xpa
same tree/a.bin keep/a.bin
run 0 "$XPAR" undo s.xpa
same tree/a.bin long.keep
cd ..

# Undo restores the damaged bytes.
jrt plain
"$DAMAGE" tree/a.bin "rand=4096,512" || hard_error "damage failed"
cp tree/a.bin damaged.keep
run 0 "$XPAR" repair --in-place --keep-journal s.xpa
same tree/a.bin keep/a.bin
run 0 "$XPAR" undo s.xpa
same tree/a.bin damaged.keep
cd ..

# --no-journal still repairs missing files.
jrt nojournal
rm -f tree/a.bin
run 0 "$XPAR" repair --in-place --no-journal s.xpa
same tree/a.bin keep/a.bin
cd ..
cd ..

# Content hashes validate rebuilt entries; one failure must not abort the tree.

step "a missing entry does not abandon the rest of the tree"

mkdir missing;  cd missing || hard_error cd
mkdir tree
mkfile tree/f1.bin 131072
mkfile tree/f2.bin 131072 2222
mkfile tree/f3.bin 131072 3333
mkfile tree/f4.bin 131072 4444
cp -r tree keep
run 0 "$XPAR" create -r 400% -s 16K -o base -R tree

damage_and_drop() {
  rm -rf tree;  cp -r keep tree
  "$DAMAGE" tree/f1.bin "rand=4096,64" || hard_error "damage failed"
  "$DAMAGE" tree/f4.bin "rand=8192,64" || hard_error "damage failed"
  rm -f tree/f2.bin
}

# --to rebuilds the missing entry.
damage_and_drop
run 0 "$XPAR" repair --to=out base.xpa
for f in f1 f2 f3 f4; do same "out/tree/$f.bin" "keep/$f.bin"; done
equal "no stage file was orphaned" "`find out -name '*.tmp' | nlines`" "0"

# --backup rebuilds it and backs up only existing files.
damage_and_drop
run 0 "$XPAR" repair --backup base.xpa
for f in f1 f2 f3 f4; do same "tree/$f.bin" "keep/$f.bin"; done
equal "backups kept for the two damaged files" "`ls tree/*.1 | nlines`" "2"
equal "no stage file was orphaned" "`find tree -name '*.tmp' | nlines`" "0"

# --in-place also rebuilds it.
damage_and_drop
run 0 "$XPAR" repair --in-place base.xpa
for f in f1 f2 f3 f4; do same "tree/$f.bin" "keep/$f.bin"; done
note "a missing entry is rebuilt by every destination"
cd ..

# Full-length components must leave room for staging suffixes.

step "full-length path components extract and repair"

mkdir longname;  cd longname || hard_error cd
long=`awk 'BEGIN{ s = "";  while (length(s) < 255) s = s "z";  print s }'`
equal "the test name is a full component" "`printf %s \"$long\" | nbytes`" "255"
mkdir tree
mkfile "tree/$long" 131072
cp -r tree keep
run 0 "$XPAR" create -r 300% -s 16K --layout=armoured -o arc -R tree
run 0 "$XPAR" extract --to=out arc.xpa
same "out/tree/$long" "keep/$long"
run 0 "$XPAR" create -r 300% -s 16K -o base -R tree
"$DAMAGE" "tree/$long" "rand=4096,64" || hard_error "damage failed"
run 0 "$XPAR" repair --to=rout base.xpa
same "rout/tree/$long" "keep/$long"
run 0 "$XPAR" repair --in-place base.xpa
same "tree/$long" "keep/$long"
staged=`find . -name '*.xpar-stage-*' -o -name '*.tmp' | nlines`
equal "nothing was left staged" "$staged" "0"
cd ..

# Windows and DOS always enforce their rules; --mangle does so here.

step "Windows naming rules reject backslashes"

mkdir winrules;  cd winrules || hard_error cd
mkdir tree
printf 'x' > 'tree/back\slash.bin'
mkfile tree/ok.bin 4096
run 0 "$XPAR" create -r 300% -s 16K --layout=armoured -o arc -R tree
run 0 "$XPAR" extract --to=plain arc.xpa
run 3 "$XPAR" extract --mangle --to=strict arc.xpa
note "Windows naming rules refuse backslashes"
cd ..

summary
