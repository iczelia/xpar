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

# A forgery the stored checksums cannot see.

step "a CRC-preserving forgery is not mistaken for intact data"

mkdir forge;  cdto forge
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
damage data.bin -Z "$Z" -Y "$Y" -n 96 cell=1,9
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

mkdir alias;  cdto alias
mkdir tree
mkfile tree/a.bin 1048576
cp tree/a.bin tree/b.bin
cp -R tree tree.orig
run 0 "$XPAR" create -R --dedup=file -s 256K -r 3 -o set tree
damage tree/b.bin "rand=100,64"
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

mkdir meta;  cdto meta
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
  damage set.xpa "seed=`expr $XPAR_TEST_SEED + $spot`" "rand=$at,48"

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

mkdir rec;  cdto rec
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
  damage data.bin -Z "$Z" -Y "$Y" -n 96 $ops
  damage "$victim" "rand=512,16384"
  attempt "$XPAR" repair --in-place set.xpa
  never_false_success "$status" data.bin pristine.bin \
                      "repair against damaged recovery"
  note "status $status with a damaged recovery volume"

  #  scrub --deep recomputes the recovery, so it has to say so.
  cp pristine.bin data.bin
  for v in $vols; do cp "$v.orig" "$v"; done
  damage "$victim" "rand=512,16384"
  attempt "$XPAR" scrub --deep set.xpa
  if test "$status" -eq 0; then
    bad "scrub --deep called a set clean whose recovery does not recompute"
  else ok; fi
fi
cd ..

# Shape damage: truncation, extension and replacement.

step "a file of the wrong length is never called intact"

mkdir shape;  cdto shape
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

mkdir dry;  cdto dry
mkfile data.bin 1048576
cp data.bin pristine.bin
run 0 "$XPAR" create -s 128K -r 4 --dedup=none -o set data.bin
damage data.bin "rand=4096,512"
cp data.bin damaged.bin
run_any "0 1" "$XPAR" repair --dry-run set.xpa
same data.bin damaged.bin
run 0 "$XPAR" repair --in-place --paranoid set.xpa
same data.bin pristine.bin
cd ..

# Volumes from another set are not recovery data.

step "a volume from another set is refused"

mkdir cross;  cdto cross
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
  ( cd a && damage data.bin "rand=8192,256" )
  cdto a
  attempt "$XPAR" repair --in-place set.xpa
  never_false_success "$status" data.bin pristine.bin \
                      "repair with a foreign recovery volume"
  note "status $status with a foreign volume in place"
  cd ..
fi
cd ..

# Auth-only sets must still distinguish localisable damage.

step "an authenticated set classifies damage the same way"

mkdir auth;  cdto auth
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

damage tree/a.bin "rand=70000,64"
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

mkdir mixed;  cdto mixed
mkdir tree
mkfile tree/a.bin 262144
run 0 "$XPAR" create -s 32K -r 4 --layout=armoured -o set -R tree
mkfile tree/b.bin 65536 2222
run 4 "$XPAR" add -r 4 --layout=sidecar set.xpa -R tree
run 0 "$XPAR" add -r 4 --layout=armoured set.xpa -R tree
exists set.g001.xpa
cd ..

step "consolidate keeps the chain's layout"

mkdir flat;  cdto flat
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

mkdir subst;  cdto subst

# Renamed volumes remain discoverable and repair restores recorded names.
mkdir a;  cdto a
mkdir tree;  mkfile tree/a.bin 262144
"$XPAR" create -s 32K -r 6 --layout=split -o set -R tree > "$log" 2>&1 ||
  hard_error "create failed"
rm -rf tree
mv set.d00 renamed.bin
run 1 "$XPAR" verify set.xpa
run 0 "$XPAR" repair --in-place set.xpa
exists set.d00
same set.d00 renamed.bin
run 0 "$XPAR" verify set.xpa
cd ..

# An intact substitute must not make a damaged named volume clean.
mkdir b;  cdto b
mkdir tree;  mkfile tree/a.bin 262144
"$XPAR" create -s 32K -r 6 --layout=split -o set -R tree > "$log" 2>&1 ||
  hard_error "create failed"
rm -rf tree
cp set.d00 spare.bin
damage set.d00 "rand=4096,512"
differs set.d00 spare.bin
run 1 "$XPAR" verify set.xpa
run 0 "$XPAR" repair --in-place set.xpa
same set.d00 spare.bin
run 0 "$XPAR" verify set.xpa
note "the named volume was rewritten from the stream"
cd ..

# Every other verb has to cope with that state too: the stream is whole, so
# nothing may report an internal error over a volume that wants rewriting.
mkdir e;  cdto e
mkdir tree;  mkfile tree/a.bin 262144
"$XPAR" create -s 32K -r 6 --layout=split -o set -R tree > "$log" 2>&1 ||
  hard_error "create failed"
rm -rf tree
cp set.d00 spare.bin
damage set.d00 "rand=4096,512"
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
mkdir d;  cdto d
mkdir tree;  mkfile tree/a.bin 262144
"$XPAR" create -s 32K -r 6 --layout=split -o set -R tree > "$log" 2>&1 ||
  hard_error "create failed"
rm -rf tree
cp set.d00 spare.bin
damage set.d00 "rand=4096,512"
cp set.d00 damaged.bin
run 0 "$XPAR" repair --to=out set.xpa
exists out/tree/a.bin
same set.d00 damaged.bin
note "--to wrote the tree and left the damaged volume where it was"
cd ..

# Repair a damaged volume without a substitute.
mkdir c;  cdto c
mkdir tree;  mkfile tree/a.bin 262144
"$XPAR" create -s 32K -r 6 --layout=split -o set -R tree > "$log" 2>&1 ||
  hard_error "create failed"
rm -rf tree
damage set.d00 "rand=4096,512"
run 1 "$XPAR" verify set.xpa
run 0 "$XPAR" repair --in-place set.xpa
run 0 "$XPAR" verify set.xpa
cd ..
cd ..

# Sidecar entries must remain reachable from the set directory.

step "a set can always find the data it just stored"

mkdir reach;  cdto reach
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

mkdir shortread;  cdto shortread

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
damage tree/a.bin "rand=4096,512"
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

mkdir missing;  cdto missing
mkdir tree
mkfile tree/f1.bin 131072
mkfile tree/f2.bin 131072 2222
mkfile tree/f3.bin 131072 3333
mkfile tree/f4.bin 131072 4444
cp -r tree keep
run 0 "$XPAR" create -r 400% -s 16K -o base -R tree

damage_and_drop() {
  rm -rf tree;  cp -r keep tree
  damage tree/f1.bin "rand=4096,64"
  damage tree/f4.bin "rand=8192,64"
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

mkdir longname;  cdto longname
long=`awk 'BEGIN{ s = "";  while (length(s) < 255) s = s "z";  print s }'`
mkdir tree
#  Probe the longest path used below.
mkdir -p rout/tree
can_hold "rout/tree/$long" && long_ok=yes || long_ok=no
rm -rf rout
if test "$long_ok" = yes; then
  n=`printf %s "$long" | nbytes`
  equal "the test name is a full component" "$n" "255"
  mkfile "tree/$long" 131072
  cp -r tree keep
  run 0 "$XPAR" create -r 300% -s 16K --layout=armoured -o arc -R tree
  run 0 "$XPAR" extract --to=out arc.xpa
  same "out/tree/$long" "keep/$long"
  run 0 "$XPAR" create -r 300% -s 16K -o base -R tree
  damage "tree/$long" "rand=4096,64"
  run 0 "$XPAR" repair --to=rout base.xpa
  same "rout/tree/$long" "keep/$long"
  run 0 "$XPAR" repair --in-place base.xpa
  same "tree/$long" "keep/$long"
  staged=`find . -name '*.xpar-stage-*' -o -name '*.tmp' | nlines`
  equal "nothing was left staged" "$staged" "0"
else
  note "255-byte components unsupported; skipped"
fi
cd ..

# Windows and DOS always enforce their rules; --strict-names does so here.

step "Windows naming rules reject backslashes"

mkdir winrules;  cdto winrules
mkdir tree
if can_hold 'tree/back\slash.bin'; then
  printf 'x' > 'tree/back\slash.bin'
  mkfile tree/ok.bin 4096
  run 0 "$XPAR" create -r 300% -s 16K --layout=armoured -o arc -R tree
  run 0 "$XPAR" extract --to=plain arc.xpa
  run 3 "$XPAR" extract --strict-names --to=strict arc.xpa
  note "Windows naming rules refuse backslashes"
else
  note "backslashes are native separators; skipped"
fi
cd ..

# --reproducible omits host metadata but preserves file modes.

step "--reproducible preserves permissions"

mkdir repro;  cdto repro
mkdir tree
mkfile tree/secret.bin 8192
mkfile tree/script.sh 8192 2222
chmod 600 tree/secret.bin 2> /dev/null
chmod 755 tree/script.sh  2> /dev/null

# Skip permission checks on hosts without file modes.
"$XPAR" create -r 300% -s 4K -o probe -R tree > "$log" 2>&1 ||
  hard_error "create failed"
if "$XPAR" list probe.xpa | grep -q '0600'; then
  run 0 "$XPAR" create -f -r 300% -s 4K --reproducible \
        --layout=armoured -o r -R tree
  if "$XPAR" list r.xpa | grep -q '0600'; then ok
  else bad "--reproducible dropped a file mode"; fi
  if "$XPAR" list r.xpa | grep -q '0755'; then ok
  else bad "--reproducible dropped the executable bit"; fi
  if "$XPAR" list r.xpa | grep -q '19[0-9][0-9]-\|20[0-9][0-9]-'; then
    bad "--reproducible kept a timestamp"
  else ok; fi
  run 0 "$XPAR" extract --to=out r.xpa
  if test -x out/tree/script.sh; then ok
  else bad "the extracted file lost its executable bit"; fi
else
  note "file modes unsupported; permission checks skipped"
fi

# Repeated runs remain byte-identical.
mkdir a b
cp -r tree a/;  cp -r tree b/
( cd a && "$XPAR" create -r 300% -s 4K --reproducible -o s -R tree ) \
  > "$log" 2>&1 || hard_error "create failed"
( cd b && "$XPAR" create -r 300% -s 4K --reproducible -o s -R tree ) \
  > "$log" 2>&1 || hard_error "create failed"
same a/s.xpa b/s.xpa
note "reproducible sets still match byte for byte"
cd ..

# --spool takes no argument, so it can never consume an input path; a
# directory goes to --spool-dir, which requires one.

step "--spool cannot consume an input path"

mkdir spool;  cdto spool
mkdir photos docs staging
mkfile photos/p.bin 8192
mkfile docs/d.bin 8192 2222
run 0 "$XPAR" create -r 300% -s 4K -R -o s --spool photos docs
run 0 "$XPAR" verify s.xpa
"$XPAR" list s.xpa > names.txt 2> "$log" || hard_error "list failed"
if grep -q 'photos/p.bin' names.txt; then ok
else bad "--spool consumed the path after it"; fi
if grep -q 'docs/d.bin' names.txt; then ok
else bad "docs was not protected"; fi

# Repeating it must stay harmless: an option that takes nothing can never
# put the argument count and the parse out of step.
run 0 "$XPAR" create -f -r 300% -s 4K -R -o s2 --spool --spool photos docs
run 0 "$XPAR" verify s2.xpa

# The directory form, and the spelling it replaced.
run 0 "$XPAR" create -f -r 300% -s 4K -R -o s3 --spool-dir=staging photos
run 4 "$XPAR" create -f -r 300% -s 4K -R -o s4 --spool-dir=nosuchdir photos
run 4 "$XPAR" create -f -r 300% -s 4K -R -o s5 --spool=staging photos
note "a directory goes to --spool-dir; --spool takes nothing"
cd ..

step "valid POSX tables load"

mkdir posx;  cdto posx
mkdir tree
mkfile tree/a.bin 16384
run 0 "$XPAR" create -r 300% -s 4K --preserve=+owner -o s -R tree
run 0 "$XPAR" list s.xpa
run 0 "$XPAR" verify s.xpa
run 0 "$XPAR" create -f -r 300% -s 4K --preserve=+owner \
      --layout=armoured -o a -R tree
run 0 "$XPAR" extract --to=out a.xpa
note "valid ownership records pass the count bound"
cd ..

# The inner code must reach the payload, which carries no packet checksum.

step "a correctable armoured archive extracts without repair"

mkdir inner;  cdto inner
mkdir tree
mkfile tree/f1.bin 70000
mkfile tree/f2.bin 70000 2222
mkfile tree/f3.bin 70000 3333
cp -r tree keep
run 0 "$XPAR" create -r 300% -s 4K --layout=armoured -o arc -R tree
cp arc.xpa pristine.xpa

#  Damage correctable payload regions at several offsets.
for off in 20000 100000 300000; do
  cp pristine.xpa arc.xpa
  damage arc.xpa "rand=$off,64"
  run 0 "$XPAR" verify arc.xpa
  rm -rf out
  run 0 "$XPAR" extract --to=out arc.xpa
  for f in f1 f2 f3; do same "out/tree/$f.bin" "keep/$f.bin"; done
  rm -rf rout
  run 0 "$XPAR" repair --to=rout arc.xpa
  for f in f1 f2 f3; do same "rout/tree/$f.bin" "keep/$f.bin"; done
done
note "verify, extract, and repair correct payload damage"

#  Chains use the same correction through generation sets.
mkfile tree/f4.bin 70000 4444
cp tree/f4.bin keep/f4.bin
cp pristine.xpa arc.xpa
run 0 "$XPAR" add -r 300% -s 4K arc.xpa -R tree
cp arc.xpa chain.xpa
damage arc.xpa "rand=20000,64"
run 0 "$XPAR" verify arc.xpa
rm -rf out
run 0 "$XPAR" extract --to=out arc.xpa
for f in f1 f2 f3 f4; do same "out/tree/$f.bin" "keep/$f.bin"; done
note "chained archives correct payload damage"
#  Restore a single-generation set.
rm -f arc.g001.xpa chain.xpa

#  Extract must report uncorrectable post-decoding damage.
cp pristine.xpa arc.xpa
damage arc.xpa "rand=100000,4096"
rm -rf out
run 2 "$XPAR" extract --to=out arc.xpa
#  Withhold damaged entries but extract intact ones.
kept=`find out -name 'f*.bin' | nlines`
if test "$kept" -lt 3; then ok
else bad "uncorrectable entry was written"; fi
for f in `find out -name 'f*.bin'`; do
  same "$f" "keep/`basename $f`"
done
cd ..

# An overlong file is damage, whatever the destination.

step "an overlong file is repaired by every destination"

mkdir overlong;  cdto overlong
mkfile data.bin 200000
cp data.bin pristine.bin
run 0 "$XPAR" create -r 300% -s 4K -o set data.bin

for dest in --backup --in-place --to=out; do
  rm -rf data.bin out;  cp pristine.bin data.bin;  rm -f data.bin.1
  printf 'JUNKJUNKJUNKJUNK' >> data.bin
  differs data.bin pristine.bin
  run 1 "$XPAR" verify set.xpa
  run 0 "$XPAR" repair $dest set.xpa
  case $dest in
    --to=*) same out/data.bin pristine.bin ;;
    *)      same data.bin pristine.bin
            #  Verify converges after repair.
            run 0 "$XPAR" verify set.xpa ;;
  esac
  if test "$dest" = --backup; then
    equal "--backup kept the overlong original" \
          "`ls data.bin.1 | nlines`" "1"
  fi
done

#  --dry-run changes nothing.
rm -f data.bin;  cp pristine.bin data.bin
printf 'JUNKJUNKJUNKJUNK' >> data.bin
run 0 "$XPAR" repair --dry-run set.xpa
equal "--dry-run preserved length" "`nbytes < data.bin`" "200016"

rm -f data.bin;  cp pristine.bin data.bin
"$XPAR" repair --in-place set.xpa > "$log" 2>&1
equal "intact set reports no damage" \
      "`grep -c 'no damage found' \"$log\"`" "1"
cd ..

# A journal names the set's files however the set was spelled.

step "undo accepts the set under a different spelling"

mkdir spell;  cdto spell
mkfile data.bin 200000
run 0 "$XPAR" create -r 300% -s 4K -o base data.bin
damage data.bin "rand=4096,64"
cp data.bin damaged.bin

#  Absolute at repair time, relative at undo time.
run 0 "$XPAR" repair --in-place --keep-journal "`pwd`/base.xpa"
differs data.bin damaged.bin
run 0 "$XPAR" undo base.xpa
same data.bin damaged.bin

#  Relative at repair time, absolute at undo time.
run 0 "$XPAR" repair --in-place --keep-journal base.xpa
run 0 "$XPAR" undo "`pwd`/base.xpa"
same data.bin damaged.bin

#  Reject journals outside their set directory.
run 0 "$XPAR" repair --in-place --keep-journal base.xpa
cd ..
run 3 "$XPAR" undo spell/base.xpa
cdto spell
note "alternate spellings accepted; other directories refused"
cd ..

# Data that merely looks like a v1 set is data, not a v1 set.

step "create protects files that only resemble xpar 1.x"

mkdir v1look;  cdto v1look
printf 'XPAS this is a text file\n' > notes.txt
printf 'XPAL and so is this\n'      > other.txt
run 0 "$XPAR" create -r 300% -s 4K -o out notes.txt other.txt
run 0 "$XPAR" verify out.xpa
#  Real v1 signatures remain invalid set inputs.
printf 'XPAS\001\002\003\004and the rest of a v1 shard' > shard.xpa
run 3 "$XPAR" verify shard.xpa
note "v1-like prefixes remain valid file data"
cd ..

# Case-folded collisions require their own sort order.

step "case-folded duplicates are found wherever they sort"

mkdir fold;  cdto fold
mkdir tree
if folds_case tree; then
  #  Colliding names cannot coexist here.
  note "case-folding filesystem; skipped"
else
  #  Bytewise these are not neighbours: R, R, Z sort before r.
  for n in README Readme.md Zebra.txt readme; do mkfile "tree/$n" 2048; done
  run 0 "$XPAR" create -r 300% -s 4K --layout=armoured -o a -R tree
  run 0 "$XPAR" extract --to=plain a.xpa
  run 3 "$XPAR" extract --strict-names --to=folded a.xpa
  equal "nothing was written from a colliding pair" \
        "`find folded -type f 2>/dev/null | nlines`" "0"
fi
cd ..

# A private file must never be briefly readable while it is written.

step "extracted modes are never wider than recorded"

mkdir modes;  cdto modes
mkdir tree
mkfile tree/secret.bin 65536
mkfile tree/public.bin 65536 2222
chmod 600 tree/secret.bin 2> /dev/null
chmod 644 tree/public.bin 2> /dev/null
run 0 "$XPAR" create -r 300% -s 4K --layout=armoured -o a -R tree
run 0 "$XPAR" extract --to=out a.xpa
#  Gate on what xpar recorded, not on what the shell can set: only the
#  former is what the next two checks are about.
if "$XPAR" list a.xpa | grep -q '0600'; then
  equal "the private file kept its mode" "`mode_of out/tree/secret.bin`" "600"
  equal "the public file kept its mode"  "`mode_of out/tree/public.bin`" "644"
  note "modes are restricted before extraction writes data"
else
  note "file modes unsupported; permission checks skipped"
fi
cd ..

# Replacing a journal requires --replace-journal, not -f.

step "a tagless set still gets journal-collision protection"

mkdir tagless;  cdto tagless
mkfile data.bin 200000
run 0 "$XPAR" create -r 300% -s 4K --slice-tag=none -o b data.bin
damage data.bin "rand=4096,64"
run 0 "$XPAR" repair --in-place -f --keep-journal b.xpa
exists b.xparundo
damage data.bin "rand=8192,64"
run 4 "$XPAR" repair --in-place -f --keep-journal b.xpa
run 0 "$XPAR" repair --in-place -f --keep-journal --replace-journal b.xpa
note "-f does not replace journals"
cd ..

# A generation must not quietly weaken the chain it joins.

step "add keeps the chain's integrity settings"

mkdir strength;  cdto strength
mkdir tree
mkfile tree/f1.bin 131072
run 0 "$XPAR" create -r 20% -s 4K --field=16 --slice-tag=16 -o s -R tree
mkfile tree/f2.bin 131072 2222
run 0 "$XPAR" add -r 20% s.xpa -R tree
field_of() {   # field_of <generation>
  "$XPAR" info --generation="$1" s.xpa 2> /dev/null |
    sed -n 's/.*matrix over GF(2^\([0-9]*\)).*/\1/p' | head -1
}
tag_of() {     # tag_of <generation>
  "$XPAR" info --generation="$1" s.xpa 2> /dev/null |
    sed -n 's/.*strong tag of \([0-9]*\) bytes.*/\1/p' | head -1
}
equal "the new generation kept the chain's field" "`field_of 1`" "`field_of 0`"
equal "the new generation kept the chain's tag length" \
      "`tag_of 1`" "`tag_of 0`"
equal "and that field is the one asked for" "`field_of 0`" "16"
equal "and that tag length is the one asked for" "`tag_of 0`" "16"
mkfile tree/f3.bin 131072 3333
run 0 "$XPAR" add -r 20% --slice-tag=8 s.xpa -R tree
equal "an explicit --slice-tag is still honoured" "`tag_of 2`" "8"
cd ..

# Report substitute volumes.

step "extract says when it read a substitute volume"

mkdir subst2;  cdto subst2
mkdir tree
mkfile tree/a.bin 400000
run 0 "$XPAR" create -r 100% -s 4K --layout=split -o sp -R tree
cp sp.d00 spare.bin
damage sp.d00 "rand=1000,4096"
"$XPAR" extract --to=out sp.xpa > /dev/null 2> "$log" || hard_error "extract failed"
same out/tree/a.bin tree/a.bin
equal "the substitution was reported" \
      "`grep -c 'intact copy found' \"$log\"`" "1"
cd ..

step "the spec's cell bound is enforced"

mkdir cells;  cdto cells
mkfile big.bin 4194304
#  One cell past the bound is refused by name, and the cell rule is decided
#  before any memory plan, so this holds at any -m.
"$XPAR" create -f -r 100% -s 512MB --cell=4096 -o bad big.bin > "$log" 2>&1
equal "past the bound is refused" "$?" "4"
equal "the refusal names the cell bound" \
      "`grep -c '65536 cells' \"$log\"`" "1"
#  At the bound, only host-dependent memory limits may refuse the plan.
"$XPAR" create -f -r 100% -s 268435456 --cell=4096 -m 512M -o edge big.bin \
  > "$log" 2>&1
rc=$?
equal "at the bound the cell rule does not fire" \
      "`grep -c '65536 cells' \"$log\"`" "0"
case $rc in 4) kind=usage ;; *) kind=other ;; esac
equal "at the bound nothing is refused as usage" "$kind" "other"
note "writers enforce K <= 65536"

#  Advice must replace the mutually exclusive -s with -b.
"$XPAR" create -f -r 100% -s 268435456 --cell=4096 -m 16M -o adv big.bin \
  > "$log" 2>&1
equal "a budget nothing can meet is refused" "$?" "7"
equal "the advice says to replace -s with -b" \
      "`grep -c 'replace -s with -b' \"$log\"`" "1"
sug=`sed -n 's/.*replace -s with -b \([0-9]*\).*/\1/p' "$log"`
"$XPAR" create -f -r 100% -b "$sug" --cell=4096 -m 16M -o adv2 big.bin \
  > "$log" 2>&1
equal "the advice can be followed exactly as given" "$?" "0"
cd ..

# explain reads a 384-byte prologue, not the whole archive.

step "explain reads only what it needs"

mkdir explain;  cdto explain
mkdir tree
mkfile tree/a.bin 262144
run 0 "$XPAR" create -r 20% -s 4K --layout=armoured -o a -R tree
run 0 "$XPAR" create -f -r 20% -s 4K -o s -R tree
"$XPAR" explain a.xpa > arm.txt 2> "$log" || hard_error "explain failed"
equal "the armoured recipe is printed" \
      "`grep -c 'armoured xpar archive' arm.txt`" "1"
#  Base names resolve to the volume actually read.
"$XPAR" explain a > base.txt 2> "$log" || hard_error "explain failed"
equal "a base name resolves to its volume" \
      "`grep -c 'a.xpa is an armoured' base.txt`" "1"
#  Packet-bearing volumes still require a full scan.
"$XPAR" explain s.xpa > side.txt 2> "$log" || hard_error "explain failed"
equal "a sidecar volume is explained too" \
      "`grep -c 'packet-bearing xpar volume' side.txt`" "1"
cd ..

# An untrusted archive must not be able to drive the terminal.

step "list shows control bytes rather than obeying them"

mkdir ansi;  cdto ansi
mkdir tree
mkfile tree/a.bin 4096
esc=`printf '\033'`
if ln -s "${esc}[41;97m PWNED ${esc}[0m" tree/evil 2> /dev/null; then
  run 0 "$XPAR" create -r 300% -s 4K --layout=armoured -o a -R tree
  "$XPAR" list a.xpa > out.txt 2> "$log" || hard_error "list failed"
  equal "no raw escape reached the output" \
        "`grep -c \"$esc\" out.txt`" "0"
  equal "the target is still shown, escaped" \
        "`grep -c 'x1B' out.txt`" "1"
  #  JSON remains escaped.
  "$XPAR" list --json a.xpa > out.json 2> "$log" || hard_error "list failed"
  equal "json escapes it too" "`grep -c 'u001b' out.json`" "1"
else
  note "control-byte symlink unsupported; skipped"
fi
cd ..

# A cell checksum and a slice checksum are not the same checksum.

step "a damaged cell table does not make a set unrepairable"

mkdir celltab;  cdto celltab
mkdir tree
mkfile tree/a.bin 400000
mkfile tree/b.bin 400000 2222
cp -r tree keep
run 0 "$XPAR" create -r 300% -s 64K --cell=4096 -o s -R tree
#  Corrupt the SLCL body so the reader drops the cell table.
slcl=`packet_body_at s.xpa SLCL`
if test -z "$slcl"; then
  hard_error "a set created with --cell=4096 has no SLCL packet"
fi
damage s.xpa "flip=$slcl,1"
damage tree/a.bin "rand=4096,64"
run 0 "$XPAR" repair --in-place s.xpa
same tree/a.bin keep/a.bin
note "slice fallback repaired without cell checksums"
cd ..

step "option bounds are enforced at both ends"

mkdir bounds;  cdto bounds
mkdir tree
mkfile tree/a.bin 100000
run 4 "$XPAR" create -r 20% -s 4K --dedup=chunk --dedup-chunk=1 -o d -R tree
run 4 "$XPAR" create -f -r 20% -s 4K --dedup=chunk --dedup-chunk=4095 -o d -R tree
run 0 "$XPAR" create -f -r 20% -s 4K --dedup=chunk --dedup-chunk=4096 -o d -R tree
equal "the refused run staged nothing" \
      "`find . -name '*.xpar-cache-*' | nlines`" "0"
#  Invalid recovery axes are usage errors.
run 4 "$XPAR" create -f -s 4K --codec=fft -r 24 --max-recovery=5000 -o m tree/a.bin
"$XPAR" create -f -s 4K --codec=fft -r 24 --max-recovery=5000 -o m tree/a.bin \
  > "$log" 2>&1
equal "and it says which knob to turn" \
      "`grep -c 'max-recovery' \"$log\"`" "1"
run 0 "$XPAR" create -f -s 4K --codec=fft -r 24 -o m tree/a.bin
cd ..

# A backslash is a separator only where the host says so.

step "a backslash in a name invents no directory"

mkdir bslash;  cdto bslash
mkdir tree
if can_hold 'tree/a\b.bin'; then
  printf 'x' > 'tree/a\b.bin'
  mkfile tree/plain.bin 4096
  run 0 "$XPAR" create -r 300% -s 4K --layout=armoured -o a -R tree
  run 0 "$XPAR" extract --to=out a.xpa
  equal "no parent directory was invented" \
        "`find out -type d -name a | nlines`" "0"
  exists 'out/tree/a\b.bin'
else
  note "backslashes are native separators here; skipped"
fi
cd ..

# All v1 mode flags receive migration guidance.

step "every 1.x mode flag is recognised"

mkdir v1flags;  cdto v1flags
for f in -Jse -Jsd -Jst -Jt -Je -Jd -We -Wd -Wt -Le -Ld -Lt -J -W -L; do
  "$XPAR" "$f" x.bin > out.txt 2>&1
  if grep -q '1.x mode flag' out.txt; then ok
  else bad "$f lacked migration guidance"; fi
done
note "bare and test-mode flags are recognized"
cd ..

# A self-contained chain carries its own data, so it can collapse alone.

step "a self-contained chain consolidates without its originals"

mkdir selfc;  cdto selfc
mkdir tree
mkfile tree/a.bin 150000
mkfile tree/b.bin 90000 2222
run 0 "$XPAR" create -r 50% -s 4K --layout=armoured -o s -R tree
mkfile tree/c.bin 100000 3333
run 0 "$XPAR" add -r 50% s.xpa -R tree
cp -r tree keep
rm -rf tree
run 0 "$XPAR" consolidate --replace s.xpa
equal "the chain collapsed to one generation" "`ls s*.xpa | nlines`" "1"
equal "nothing was left staged" \
      "`find . -maxdepth 1 -name '.xpar-consolidate-*' | nlines`" "0"
run 0 "$XPAR" extract --to=out s.xpa
for f in a b c; do same "out/tree/$f.bin" "keep/$f.bin"; done
note "the archive served as its own source"
cd ..

#  What the last run() printed; xpar reports on stderr.
said_safety() {   # said_safety <text>
  if grep -q "$1" "$log" 2> /dev/null; then echo yes;  else echo no; fi
}

step "the undo journal is private, fresh, and never a link somebody planted"

#  Journal creation must not follow links.
mkdir -p uj && cd uj
mkfile p.bin 300000 81
printf 'PRECIOUS\n' > victim.txt
cp victim.txt victim.keep
run 0 "$XPAR" create -s 4096 -r 30 -o set p.bin
damage p.bin rand=4096,64 rand=12288,64 rand=20480,64

if ln -s victim.txt set.xparundo 2> /dev/null; then
  run 4 "$XPAR" repair --in-place set.xpa
  same victim.txt victim.keep
  #  --replace-journal replaces the name; it does not write through it.
  run 0 "$XPAR" repair --in-place --replace-journal set.xpa
  same victim.txt victim.keep
else
  note "symbolic links unsupported; skipped"
  run 0 "$XPAR" repair --in-place set.xpa
fi

#  A kept journal is a copy of protected data and belongs to its owner.
damage p.bin rand=4096,64
run 0 "$XPAR" repair --in-place --keep-journal set.xpa
exists set.xparundo
equal "the journal is owner-only" "`mode_of set.xparundo`" 600
cd ..

step "an in-place repair restores the names and metadata it recreates"

#  Recreate parents and restore metadata for missing files.
mkdir -p ip && cd ip
mkdir -p tree/sub
mkfile tree/sub/x.bin 100000 82
chmod 600 tree/sub/x.bin
cp tree/sub/x.bin keep.bin
run 0 "$XPAR" create -R -s 4096 -r 30 -o s tree
rm -rf tree/sub

run 0 "$XPAR" repair --in-place s.xpa
same tree/sub/x.bin keep.bin
equal "the recorded mode came back" "`mode_of tree/sub/x.bin`" 600
equal "no journal was left behind" \
      "`find . -maxdepth 1 -name '*.xparundo' | nlines`" 0
run 0 "$XPAR" verify s.xpa
cd ..

step "names the manifest fully describes are recreated, not reported clean"

#  Recreate missing names that need no recovery data.
mkdir -p nm && cd nm
mkdir -p tree/d
mkfile tree/f.bin 50000 83
: > tree/empty.bin
run 0 "$XPAR" create -R -s 4096 -r 30 -o s tree
rm tree/empty.bin
rmdir tree/d

run 1 "$XPAR" verify s.xpa
run 0 "$XPAR" repair --in-place s.xpa
exists tree/empty.bin
exists tree/d
run 0 "$XPAR" verify s.xpa
cd ..

step "a dry run answers whether anything would change"

#  --exit-on-change must work with --dry-run.
mkdir -p dr && cd dr
mkfile p.bin 300000 84
run 0 "$XPAR" create -s 4096 -r 30 -o set p.bin
run 0 "$XPAR" repair --in-place --dry-run --exit-on-change set.xpa
damage p.bin rand=4096,64 rand=12288,64
run 1 "$XPAR" repair --in-place --dry-run --exit-on-change set.xpa
run 0 "$XPAR" repair --in-place --dry-run set.xpa
cd ..

step "verify says what an in-place repair of an untagged set will need"

#  Warn when untagged slices require forced in-place repair.
mkdir -p nt && cd nt
mkfile p.bin 300000 85
run 0 "$XPAR" create -s 4096 -r 30 --slice-tag=none -o set p.bin
damage p.bin rand=4096,64
run 1 "$XPAR" verify set.xpa
equal "the -f requirement is named" "`said_safety 'in-place repair needs'`" yes
run 4 "$XPAR" repair --in-place set.xpa
run 0 "$XPAR" repair -f --in-place set.xpa
cd ..

step "verify stages an armoured plaintext away from read-only media"

#  Use the host temp directory when environment variables are absent.
mkdir -p rostage && cd rostage
if perms_bite .; then
  mkfile data.bin 2097152 67
  run 0 "$XPAR" create -r 5% --layout=armoured -o arm data.bin
  mkdir ro && mv arm.xpa ro/ && chmod 555 ro
  run 0 env TMPDIR= TMP= TEMP= "$XPAR" verify -m 1M ro/arm.xpa
  equal "nothing was staged beside the archive" "`ls ro | nlines`" 1
  chmod 755 ro
else
  note "mode 555 is writable; skipping the read-only stage test"
fi
cd ..

summary
