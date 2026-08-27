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

step "--stdout corrects with the inner code, as --to does"

#  --stdout must apply lazy inner-code correction like directory extraction.
mkdir -p e3 && cd e3 || hard_error "cd e3"
mkfile data.bin 400000 88
cp data.bin pristine.bin
run 0 "$XPAR" create -r 20% --layout=armoured -o p data.bin
rm -f data.bin
"$DAMAGE" p.xpa rand=50000,32 || hard_error "damage failed"

rm -rf d1 && mkdir d1
run 0 "$XPAR" extract --to=d1 p.xpa
same d1/data.bin pristine.bin

capture out.bin "$XPAR" extract --stdout p.xpa
equal "stdout status on correctable damage" "$status" 0
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

# Require the group to cross the replication threshold.
big=0;  small=0
for n in $vols; do
  if test `wc -c < "$n"` -gt 1000000; then big=`expr $big + 1`
  else small=`expr $small + 1`; fi
done
if test "$big" -eq 0 || test "$small" -eq 0; then
  bad "replication threshold not crossed; enlarge the manifest"
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

step "recover rebuilds a volume with the set's own inner code"

#  recover must reuse the set's armour parameters, not CLI defaults.
mkdir -p a1 && cd a1 || hard_error "cd a1"
mkfile payload.bin 200000 91
for opt in --armour=none --armour-t=48 --armour-field=16 --armour-pct=5; do
  rm -f set.* orig.bin
  run 0 "$XPAR" create -r 4 -s 8K --layout=sidecar $opt -o set payload.bin
  v=`find . -maxdepth 1 -name 'set.v*' | head -1`
  if test -z "$v"; then
    bad "$opt: create wrote no recovery volume"
    continue
  fi
  cp "$v" orig.bin
  rm -f "$v"
  run 0 "$XPAR" recover --volume="`basename $v`" set.xpa
  same "$v" orig.bin
  run 0 "$XPAR" verify set.xpa
done
cd .. || hard_error cd

step "recover thresholds replication on the size as written"

#  Replication thresholds use the armoured critical-group size.
mkdir -p a2 && cd a2 || hard_error "cd a2"
mkdir tree
mkfile tree/payload.bin 400000 92
pad=nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn
i=0
while test "$i" -lt 400; do
  printf x > "tree/f$i.$pad.txt"
  i=`expr $i + 1`
done

#  High armour overhead straddles the replication threshold.
run 0 "$XPAR" create -r 16 -s 4K --volumes=8 --layout=split \
                     --armour-t=120 -o set -R tree

vols=`find . -maxdepth 1 -name 'set.v*' | sort`
test -n "$vols" || hard_error "no recovery volumes were written"
big=0;  small=0
for n in $vols; do
  if test `wc -c < "$n"` -gt 1000000; then big=`expr $big + 1`
  else small=`expr $small + 1`; fi
done
if test "$big" -eq 0 || test "$small" -eq 0; then
  bad "all volumes use the same replication; adjust --armour-t"
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

step "a superseded neighbour does not excuse damage in the same slice"

#  A superseded cell must not suppress damage to live cells in its slice.
mkdir -p s1 && cd s1 || hard_error "cd s1"
mkfile key.bin 32 7
mkdir tree
i=0
while test "$i" -lt 32; do
  mkfile "tree/f$i.bin" 4096 `expr 300 + $i`
  i=`expr $i + 1`
done
run 0 "$XPAR" create -R -s 64K --cell=4K -r 4 --auth-key=key.bin -o set tree
#  Require a slice that can hold both superseded and live cells.
"$XPAR" info --json --auth-key=key.bin set.xpa > g.json 2> "$log" ||
  hard_error "info --json failed on the keyed set"
zz=`tr ',' '\n' < g.json | sed -n 's/.*"slice_size":\([0-9][0-9]*\).*/\1/p' |
      head -1`
yy=`tr ',' '\n' < g.json | sed -n 's/.*"cell_bytes":\([0-9][0-9]*\).*/\1/p' |
      head -1`
equal "the slice holds several cells" "`test "${yy:-0}" -gt 0 &&
                                        test "$zz" -gt "$yy" &&
                                        echo yes || echo no`" yes

#  Replacing f0 supersedes it while leaving adjacent f1 live.
cp tree/f1.bin pristine.bin
mkfile tree/f0.bin 4096 99
run 0 "$XPAR" add -r 4 --auth-key=key.bin set.xpa -R tree
run 0 "$XPAR" verify --chain --fast --auth-key=key.bin set.xpa

"$DAMAGE" tree/f1.bin rand=100,64 || hard_error "damage failed"
differs tree/f1.bin pristine.bin
run_any "1 2" "$XPAR" verify --chain --fast --auth-key=key.bin set.xpa

#  Generation 0 owns and must report the damaged bytes.
"$XPAR" verify --chain --auth-key=key.bin --json set.xpa > v.json 2> "$log"
g0=`tr '{' '\n' < v.json |
      sed -n 's/.*"generation_result".*"generation":0,.*"status":"\([a-z]*\)".*/\1/p' |
      head -1`
equal "generation 0 placed the damage" "$g0" repairable
run 0 "$XPAR" repair --chain --in-place --auth-key=key.bin set.xpa
same tree/f1.bin pristine.bin
cd .. || hard_error cd

step "a truncated file is still scanned for the cells it damaged"

#  Missing or truncated aliases must mark cells even when dedup leaves the
#  canonical stream intact.
mkdir -p t1 && cd t1 || hard_error "cd t1"
mkdir tree
mkfile tree/a.bin 120000 21
cp tree/a.bin tree/b.bin
mkfile tree/c.bin 120000 22
run 0 "$XPAR" create -R -s 32K --cell=8K -r 8 --dedup=file -o set tree
cp tree/b.bin pristine.bin

dd if=pristine.bin of=tree/b.bin bs=1 count=60000 status=none ||
  hard_error "truncate failed"
differs tree/b.bin pristine.bin

run 1 "$XPAR" verify set.xpa
run 0 "$XPAR" repair --in-place set.xpa
same tree/b.bin pristine.bin
run 0 "$XPAR" verify set.xpa

#  A missing file takes the same branch.
rm -f tree/b.bin
run 1 "$XPAR" verify set.xpa
run 0 "$XPAR" repair --in-place set.xpa
same tree/b.bin pristine.bin
run 0 "$XPAR" verify set.xpa
cd .. || hard_error cd

step "a crafted packet key does not run the critical-group rebuild away"

#  A UINT64_MAX packet key must not wrap the rebuild cursor into a loop.
mkdir -p f1 && cd f1 || hard_error "cd f1"
mkfile payload.bin 200000 31
run 0 "$XPAR" create -r 4 -s 8K --layout=sidecar -o set payload.bin
v=`find . -maxdepth 1 -name 'set.v*' | head -1`
test -n "$v" || hard_error "create wrote no recovery volume"
cp "$v" orig.bin

#  Forge the only AUTH packet with an invalid short body.
for t in set.xpa `find . -maxdepth 1 -name 'set.v*'`; do
  "$FORGE" "$t" AUTH ffffffffffffffff || hard_error "forge failed on $t"
done
rm -f "$v"

#  Use the cap only if this build can start under it; sanitizers may not.
capped=no
if ( ulimit -v 2000000 ) 2> /dev/null &&
   ( ulimit -v 2000000; "$XPAR" --version ) > /dev/null 2>&1; then
  capped=yes
fi
if test "$capped" = yes; then
  status=0
  ( ulimit -v 2000000; "$XPAR" recover --volume="`basename $v`" set.xpa ) \
    > "$log" 2>&1 || status=$?
  equal "recover survived the crafted key" "$status" 0
  #  The forged packet changes the volume, but rebuilding stays bounded.
  if test -s "$v"; then
    grew=`expr \`wc -c < "$v"\` - \`wc -c < orig.bin\``
    equal "the rebuilt group stayed bounded" "`test "${grew:-999999}" -lt 65536 &&
                                               echo yes || echo no`" yes
  else
    bad "recover wrote no volume"
  fi
else
  note "address-space cap unsupported; skipping runaway test"
fi
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
  bad "no frame exceeded a pipe buffer; short-read path untested"
fi

step "--json --progress emits progress records, and --json alone does not"

#  --json --progress previously had no progress caller.
mkdir -p j1 && cd j1 || hard_error "cd j1"
#  Use enough data to pass progress throttling.
mkfile data.bin 40000000 51
run 0 "$XPAR" create -r 4 -s 1M --json --progress -o set data.bin
"$XPAR" verify --json --progress set.xpa > p.json 2> "$log"
n=`grep -c '"type":"progress"' p.json`
equal "verify --json --progress reported" "`test "${n:-0}" -gt 0 &&
                                            echo yes || echo no`" yes
equal "the record carries done and rate" \
      "`grep -c '"type":"progress".*"done":[0-9].*"rate_bps":[0-9]' p.json |
          { read c; test "$c" -gt 0 && echo yes || echo no; }`" yes

"$XPAR" verify --json set.xpa > q.json 2> "$log"
equal "--json alone stays silent" "`grep -c '"type":"progress"' q.json`" 0

"$XPAR" verify --progress set.xpa > /dev/null 2> h.txt
equal "--progress alone stays human" "`grep -c '"type":"progress"' h.txt`" 0
cd .. || hard_error cd

step "an unwritable destination exits with the I/O status"

#  Exercise the distinct I/O exit status (5).
mkdir -p w1 && cd w1 || hard_error "cd w1"
if perms_bite .; then
  mkfile data.bin 100000 41
  run 0 "$XPAR" create -r 20% --layout=armoured -o p data.bin
  mkdir ro && chmod 555 ro
  run 5 "$XPAR" extract --to=ro p.xpa
  run 5 "$XPAR" create -r 20% --layout=armoured -o ro/q data.bin
  chmod 755 ro
  run 0 "$XPAR" extract --to=ro p.xpa
  same ro/data.bin data.bin
else
  note "mode 555 is writable; skipping I/O test"
fi
cd .. || hard_error cd

step "a volume added later agrees with the ones already there"

#  Added volumes must reuse the set's replicated CRTR packet.
mkdir -p a4 && cd a4 || hard_error "cd a4"
mkfile p.bin 400000 83

for later in "addrecovery --reproducible -r 12" "addrecovery -r 12"; do
  rm -f set.* && cp p.bin d.bin
  run 0 "$XPAR" create --reproducible -r 4 -s 32K -o set d.bin
  run 0 "$XPAR" $later set.xpa
  #  Exercise chain readers that detect replicated-packet conflicts.
  run 0 "$XPAR" info set.xpa
  run 0 "$XPAR" repair --in-place set.xpa
  run 0 "$XPAR" verify set.xpa
done

rm -f set.* && cp p.bin d.bin
run 0 "$XPAR" create --reproducible -r 8 -s 32K -o set d.bin
v=`find . -maxdepth 1 -name 'set.v*' | head -1`
test -n "$v" || hard_error "create wrote no recovery volume"
rm -f "$v"
run 0 "$XPAR" recover --volume="`basename $v`" set.xpa
run 0 "$XPAR" info set.xpa
run 0 "$XPAR" repair --in-place set.xpa
cd .. || hard_error cd

step "creator disagreement is tolerated; other conflicts are fatal"

#  CRTR provenance may differ across volumes; other replicated packets may not.
mkdir -p a5 && cd a5 || hard_error "cd a5"
mkfile p.bin 200000 84
run 0 "$XPAR" create --reproducible -r 4 -s 32K --armour=none -o set p.bin
run 0 "$XPAR" info set.xpa

#  Conflicting CRTR provenance is accepted.
"$FORGE" set.xpa CRTR 78706172203939 || hard_error "forge failed"
run 0 "$XPAR" info set.xpa
run 0 "$XPAR" verify set.xpa
run 0 "$XPAR" repair --in-place set.xpa

#  Conflicting SETD remains fatal.
rm -f set.* && cp p.bin q.bin
run 0 "$XPAR" create --reproducible -r 4 -s 32K --armour=none -o set q.bin
"$FORGE" set.xpa SETD 00112233445566778899aabbccddeeff ||
  hard_error "forge failed"
run 3 "$XPAR" info set.xpa
cd .. || hard_error cd

step "addrecovery tops up every layout and the result still repairs"

#  Cover critical-group reuse across every layout.
mkdir -p a3 && cd a3 || hard_error "cd a3"
mkfile p.bin 400000 81
for lay in "--layout=sidecar" "--layout=split" "--layout=armoured"; do
  rm -f set.* && cp p.bin d.bin
  run 0 "$XPAR" create --reproducible -r 4 -s 32K $lay -o set d.bin
  before=`find . -maxdepth 1 -name 'set.v*' | wc -l | tr -d ' '`
  run 0 "$XPAR" addrecovery --reproducible -r 12 set.xpa
  run 0 "$XPAR" verify set.xpa
  after=`find . -maxdepth 1 -name 'set.v*' | wc -l | tr -d ' '`
  equal "$lay grew its recovery" "`test "$after" -ge "$before" &&
                                   echo yes || echo no`" yes

  #  Damage beyond the original redundancy; keep armoured hits codeword-sized.
  case "$lay" in
    *sidecar*) tgt=d.bin;                                       run_len=300 ;;
    *split*)   tgt=`find . -maxdepth 1 -name 'set.d*' | head -1`; run_len=300 ;;
    *)         tgt=set.xpa;                                     run_len=8 ;;
  esac
  test -n "$tgt" && test -f "$tgt" || hard_error "no damage target for $lay"
  bytes=`wc -c < "$tgt" | tr -d ' '`
  ops=""
  for pct in 40 52 64 76 88; do
    ops="$ops rand=`expr $bytes / 100 \* $pct`,$run_len"
  done
  "$DAMAGE" "$tgt" $ops > "$log" 2>&1 || hard_error "damage failed"
  run 0 "$XPAR" repair --in-place set.xpa
  run 0 "$XPAR" verify set.xpa
  case "$lay" in *sidecar*) same d.bin p.bin ;; esac
done
cd .. || hard_error cd

step "displaced data is found again rather than treated as damage"

#  Cover the shared misplaced-data search in verify and repair.
mkdir -p r2 && cd r2 || hard_error "cd r2"
mkfile p.bin 900000 71
cp p.bin pristine.bin
run 0 "$XPAR" create --reproducible -r 8 -s 64K -o s p.bin

#  Prepending produces one dominant displacement.
shift_by() {   # shift_by <bytes>
  cp pristine.bin p.bin
  dd if=/dev/zero of=pad.bin bs=1 count=$1 status=none ||
    hard_error "cannot build the pad"
  cat pad.bin pristine.bin > t.bin && mv t.bin p.bin
}

for n in 1 4096 65536; do
  shift_by $n
  differs p.bin pristine.bin
  run 0 "$XPAR" repair --in-place s.xpa
  same p.bin pristine.bin
done

#  Disabling resync leaves the shifted file damaged.
shift_by 4096
run_any "1 2" "$XPAR" repair --in-place --resync=off s.xpa
differs p.bin pristine.bin

#  Two displacements must be reported as ambiguous.
cp pristine.bin p.bin
dd if=/dev/zero of=pad1.bin bs=1 count=3 status=none
dd if=/dev/zero of=pad2.bin bs=1 count=9999 status=none
dd if=pristine.bin of=head.bin bs=1 count=400000 status=none
dd if=pristine.bin of=tail.bin bs=1 skip=400000 status=none
cat pad1.bin head.bin pad2.bin tail.bin > p.bin
"$XPAR" repair -v --in-place s.xpa > "$log" 2>&1
if grep -q "no dominant displacement" "$log"; then ok
else bad "ambiguous displacement was not reported"; fi
cd .. || hard_error cd

step "asking for no recovery means the same thing to create and to add"

# Zero requests no parity; positive fractions still round up to one.
mkdir -p j2 && cd j2 || hard_error "cd j2"
mkfile p.bin 200000 91
cp p.bin pristine.bin

for spec in 0 0% 0x; do
  rm -f set.*
  run 0 "$XPAR" create --reproducible -r $spec -s 32K -o set p.bin
  n=`"$XPAR" info --json set.xpa 2> "$log" | tr ',' '\n' |
       sed -n 's/.*"recovery":\([0-9][0-9]*\).*/\1/p' | head -1`
  equal "-r $spec asks for none and gets none" "${n:-x}" 0
  run 0 "$XPAR" verify set.xpa
done

#  It still detects damage; it simply cannot mend it.
"$DAMAGE" p.bin rand=1000,64 || hard_error "damage failed"
run 2 "$XPAR" verify set.xpa
run 2 "$XPAR" repair --in-place set.xpa

#  A fraction too small to name a slice is still a request for some.
cp pristine.bin p.bin
rm -f set.*
run 0 "$XPAR" create --reproducible -r 0.0001% -s 32K -o set p.bin
n=`"$XPAR" info --json set.xpa 2> "$log" | tr ',' '\n' |
     sed -n 's/.*"recovery":\([0-9][0-9]*\).*/\1/p' | head -1`
equal "a vanishing percentage rounds up to one" "${n:-x}" 1

# Add follows the same rule.
cp pristine.bin p.bin
rm -f set.*
run 0 "$XPAR" create --reproducible -r 4 -s 32K -o set p.bin
mkfile extra.bin 5000 92
run 0 "$XPAR" add --reproducible -r 0 set.xpa p.bin extra.bin
run 0 "$XPAR" verify --chain set.xpa
cd .. || hard_error cd

step "a lost cell table can be rebuilt from the slices that survive"

# Rebuild missing cell tables without scaling memory to the archive.
mkdir -p k1 && cd k1 || hard_error "cd k1"
mkfile p.bin 300000 93
cp p.bin pristine.bin
run 0 "$XPAR" create --armour=none --reproducible -s 16K --cell=4K -r 6 \
    -o set p.bin

#  Losing every copy of the table drops erasures back to whole slices.
for v in set.xpa set.v*.xpa; do
  test -f "$v" && "$DAMAGE" "$v" unpacket=SLCL > /dev/null
done
"$XPAR" scrub set.xpa > /dev/null 2> "$log" || :
if grep -q "no complete cell table survives" "$log"; then ok
else bad "a set with no SLCL packets did not report the fallback"; fi

run 0 "$XPAR" scrub --rebuild-cells set.xpa
if grep -q "rebuild-cells: wrote" "$log"; then ok
else bad "--rebuild-cells wrote nothing"; fi

"$XPAR" scrub set.xpa > /dev/null 2> "$log" || :
if grep -q "no complete cell table survives" "$log"; then
  bad "the rebuilt table did not take"
else ok; fi

# The rebuilt table restores cell-level repair.
"$DAMAGE" p.bin -Z 16384 -Y 4096 cell=3,1 || hard_error "damage failed"
run 1 "$XPAR" verify set.xpa
if grep -q "1 slice, 1 cell" "$log"; then ok
else bad "the rebuilt table did not narrow the damage to one cell"; fi
run 0 "$XPAR" repair --in-place set.xpa
same p.bin pristine.bin

# Unverified slices cannot seed a cell table.
cp pristine.bin p.bin
rm -f set.*
run 0 "$XPAR" create --armour=none --reproducible -s 16K --cell=4K -r 6 \
    -o set p.bin
for v in set.xpa set.v*.xpa; do
  test -f "$v" && "$DAMAGE" "$v" unpacket=SLCL > /dev/null
done
"$DAMAGE" p.bin rand=40000,64 || hard_error "damage failed"
"$XPAR" scrub --rebuild-cells set.xpa > /dev/null 2> "$log" || :
if grep -q "cannot seed a cell table" "$log"; then ok
else bad "--rebuild-cells seeded a table from a slice that does not verify"; fi
if grep -q "rebuild-cells: wrote" "$log"; then
  bad "--rebuild-cells wrote a table it could not seed"
else ok; fi
cd .. || hard_error cd

step "a verb written after its options is named as the mistake"

# Diagnose verbs placed after options.
mkdir -p k2 && cd k2 || hard_error "cd k2"
mkfile verify 60000 94
run 0 "$XPAR" create --reproducible -r 2 -s 16K -o s verify

run 4 "$XPAR" --json verify s.xpa
if grep -q "'verify' must come first" "$log"; then ok
else bad "a global option before the verb did not name the verb"; fi

# Diagnose misplaced verbs after verb-specific options too.
run 4 "$XPAR" -f create -o out verify
if grep -q "'create' must come first" "$log"; then ok
else bad "a verb option before the verb did not name the verb"; fi

# Correct ordering and -- still work.
run 0 "$XPAR" verify --json s.xpa
run 0 "$XPAR" --json -- s.xpa
cd .. || hard_error cd

step "explain names the file it was actually given"

# Use the resolved split-volume name in recipes.
mkdir -p i1 && cd i1 || hard_error "cd i1"
mkfile p.bin 200000 90
run 0 "$XPAR" create --reproducible -r 20% --layout=split -o photos p.bin
"$XPAR" explain photos > r.txt 2> "$log"
name=`sed -n 's/^in=//p' r.txt | head -1`
equal "the recipe reads the resolved name" "$name" "photos.xpa"
exists "$name"
cd .. || hard_error cd

step "--deep names the missing data rather than blaming the parity"

# Missing data must not be reported as bad parity.
mkdir -p h1 && cd h1 || hard_error "cd h1"
mkfile p.bin 400000 89
run 0 "$XPAR" create --reproducible -r 8 -s 32K -o set p.bin
run 0 "$XPAR" scrub --deep set.xpa
rm -f p.bin
"$XPAR" scrub --deep set.xpa > "$log" 2>&1
if grep -q "do not recompute from the data" "$log"; then
  bad "missing data was reported as bad parity"
else ok; fi
# Match the diagnostic prefix, not its wording.
if grep -q "^xpar: --deep: " "$log"; then ok
else bad "--deep did not say why it could not check the parity"; fi
cd .. || hard_error cd

step "the reader rejects what the format says it must"

# Reject reserved fields, reserved attribute bits and invalid generators.
mkdir -p g1 && cd g1 || hard_error "cd g1"
mkfile p.bin 200000 88
run 0 "$XPAR" create --reproducible -r 4 -s 32K --armour=none -o set p.bin
run 0 "$XPAR" info set.xpa

# STRM with a nonzero reserved field.
"$FORGE" set.xpa STRM 0000000000000000ffffffffffffffff ||
  hard_error "forge failed"
run 0 "$XPAR" info set.xpa
note "a malformed STRM is skipped rather than parsed"

# RCVS with a nonzero reserved field must not count as recovery.
rm -f set.* && cp p.bin q.bin
run 0 "$XPAR" create --reproducible -r 4 -s 32K --armour=none -o set q.bin
before=`"$XPAR" info --json set.xpa 2> "$log" | tr ',' '\n' |
          sed -n 's/.*"recovery":\([0-9][0-9]*\).*/\1/p' | head -1`
"$FORGE" set.xpa RCVS 0000000000000063ffffffffffffffff || hard_error "forge"
after=`"$XPAR" info --json set.xpa 2> "$log" | tr ',' '\n' |
         sed -n 's/.*"recovery":\([0-9][0-9]*\).*/\1/p' | head -1`
equal "a reserved RCVS field is not accepted" "$after" "$before"
run 0 "$XPAR" verify set.xpa
cd .. || hard_error cd

step "every armour field works on every layout"

# Cover every field/layout combination when parity uses defaults.
mkdir -p f2 && cd f2 || hard_error "cd f2"
mkfile p.bin 200000 87
for lay in --layout=sidecar --layout=split --layout=armoured; do
  for fld in "" --armour-field=8 --armour-field=16; do
    rm -f set.* && cp p.bin d.bin
    run 0 "$XPAR" create --reproducible -r 20% $lay $fld -o set d.bin
    run 0 "$XPAR" verify set.xpa
  done
done
cd .. || hard_error cd

step "the inner code corrects exactly what its parameters promise"

#  At depth 1, n corrupt bytes hit n symbols in one codeword; depth D spreads
#  them over D codewords. The outer code does not protect critical groups.
mkdir -p c1 && cd c1 || hard_error "cd c1"
mkfile p.bin 200000 61

# corrects <expected> <create options...>
corrects() {
  want=$1;  shift
  rm -f s.* clean.bin
  if ! attempt "$XPAR" create --reproducible -r 20% "$@" -o s p.bin; then
    return
  fi
  if test "$status" -ne 0; then
    bad "create $* exited $status"
    return
  fi
  cp s.xpa clean.bin
  got=0
  n=1
  while test "$n" -le `expr $want + 4`; do
    cp clean.bin s.xpa
    "$DAMAGE" s.xpa rand=600,$n > "$log" 2>&1 || hard_error "damage failed"
    "$XPAR" verify s.xpa > "$log" 2>&1 || break
    got=$n
    n=`expr $n + 1`
  done
  equal "$* corrects $want bytes of one frame" "$got" "$want"
}

#  GF(2^8): one symbol is one byte, so t bytes.
corrects 16 --armour-t=16
corrects 32 --armour-t=32
#  GF(2^16): one symbol is two bytes, so 2t bytes.
corrects 32 --armour-field=16 --armour-t=16
corrects 80 --armour-field=16 --armour-t=40
#  Depth D interleaves, so a burst of t*D symbols lands one per codeword.
corrects 64 --depth=4 --armour-t=16
cd .. || hard_error cd

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
