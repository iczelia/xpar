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

set_index=sets.xpa
set_g1_data=sets.g001.d00
set_volumes='sets.v*'
set_data_volumes='sets.d*'
disc_volumes='disc.v*.xpa'
disc_g1_volumes='disc.g001*.xpa'
bkup_g2_index=bkup.g002.xpa
data_g1_data=data.g001.d00
data_g1_volumes='data.g001.v*'
if xpar_config_defined XPAR_DOS; then
  set_index=SETS.XPA
  set_g1_data=SETG1.D00
  set_volumes='SETS.V??'
  set_data_volumes='SETS.D??'
  disc_volumes='DISC.V??'
  disc_g1_volumes='DISC.XG1 DISG1.V??'
  bkup_g2_index=BKUP.XG2
  data_g1_data=DATG1.D00
  data_g1_volumes='DATG1.V??'
fi

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

dos_stem() { printf '%.4s' "$1" | tr 'a-z' 'A-Z'; }

index_name() {   # index_name <stem> <generation>
  if xpar_config_defined XPAR_DOS; then
    _stem=`dos_stem "$1"`
    if test "$2" -eq 0; then printf '%s.XPA\n' "$_stem"
    else printf '%s.XG%s\n' "$_stem" "$2"; fi
  elif test "$2" -eq 0; then printf '%s.xpa\n' "$1"
  else printf '%s.g%03d.xpa\n' "$1" "$2"; fi
}

data_name() {   # data_name <stem> <generation> <ordinal>
  if xpar_config_defined XPAR_DOS; then
    _stem=`dos_stem "$1"`
    if test "$2" -eq 0; then printf '%s.D%02d\n' "$_stem" "$3"
    else printf '%.3sG%s.D%02d\n' "$_stem" "$2" "$3"; fi
  elif test "$2" -eq 0; then printf '%s.d%02d\n' "$1" "$3"
  else printf '%s.g%03d.d%02d\n' "$1" "$2" "$3"; fi
}

volume_files() {   # volume_files <stem> [generation]
  _gen=${2:-0}
  if xpar_config_defined XPAR_DOS; then
    _stem=`dos_stem "$1"`
    if test "$_gen" -eq 0; then ls "$_stem".V?? 2> /dev/null
    else ls "`printf '%.3s' "$_stem"`G$_gen".V?? 2> /dev/null; fi
  elif test "$_gen" -eq 0; then ls "$1".v*.xpa 2> /dev/null
  else ls "$1".g`printf '%03d' "$_gen"`.v*.xpa 2> /dev/null; fi
}

journal_name() {   # journal_name <stem> <generation>
  if xpar_config_defined XPAR_DOS; then
    _stem=`dos_stem "$1"`
    if test "$2" -eq 0; then printf '%s.XPU\n' "$_stem"
    else printf '%s.XU%s\n' "$_stem" "$2"; fi
  elif test "$2" -eq 0; then printf '%s.xparundo\n' "$1"
  else printf '%s.g%03d.xparundo\n' "$1" "$2"; fi
}

journal_files() {
  if xpar_config_defined XPAR_DOS; then ls *.XPU *.XU? 2> /dev/null
  else ls *.xparundo 2> /dev/null; fi
}

safe_g1_index=`index_name safe 1`
safe_g1_data=`data_name safe 1 0`
safe_g1_journal=`journal_name safe 1`
safe_data0=`data_name safe 0 0`
safe_data1=`data_name safe 0 1`
yset_data0=`data_name yset 0 0`

dos_manifest_padding() {
  count=$1; depth=$2; deep=tree
  i=0
  while test "$i" -lt "$depth"; do
    deep=$deep/d$i
    mkdir "$deep" || hard_error "cannot create DOS manifest fixture"
    i=$((i + 1))
  done
  i=0
  while test "$i" -lt "$count"; do
    if test $((i % 100)) -eq 0; then
      group=$((i / 100))
      leaf=$deep/g$group
      mkdir "$leaf" || hard_error "cannot create DOS manifest fixture"
    fi
    printf x > "$leaf/`printf 'f%07d' "$i"`" ||
      hard_error "cannot create DOS manifest fixture"
    i=$((i + 1))
  done
}

files() {
  find . -maxdepth 1 -name 'set*' | sort | while read _f; do
    printf '%s:%s ' "$_f" "`cksum < "$_f" | tr -d ' '`"
  done
}

said() {
  if grep -q "$1" "$log" 2> /dev/null; then echo yes;  else echo no; fi
}

#  Count RCVS, SLTG and SLCL packets outside an ARMG.
bare_packets() {   # bare_packets <volume>...
  _n=0
  for _v in "$@"; do
    for _t in RCVS SLTG SLCL; do
      _n=$((_n + `"$DAMAGE" "$_v" find=$_t | nlines`))
    done
  done
  echo "$_n"
}

step "extract --stdout must not emit unverified bytes"

#  --stdout took a shortcut past the manifest validation and the entry
#  content hash, so a corrupted stream was written out with status 0.
mkdir -p e1 && cdto e1
mkfile data.bin 400000
cp data.bin pristine.bin
run 0 "$XPAR" create -r 4 -s 8K --layout=armoured -o sets data.bin
rm -f data.bin
read_geometry sets.xpa

#  More damaged slices than there is recovery to rebuild them.
ops=""
i=0
while test "$i" -lt 12; do
  ops="$ops rand=$((2048 + i * Z * 2)),64"
  i=$((i + 1))
done
damage "$set_index" $ops

capture out.bin "$XPAR" extract --stdout sets.xpa
if test "$status" -eq 0; then
  note "extract --stdout reported success; its bytes must then be correct"
  same out.bin pristine.bin
else
  note "extract --stdout refused the damaged stream (status $status)"
  #  Refusing after emitting is the same defect wearing a status code.
  equal "the refusal emitted nothing" "`wc -c < out.bin | tr -d ' '`" 0
fi
cdto ..

step "extract --stdout still works on an intact set"

mkdir -p e2 && cdto e2
mkfile data.bin 400000
cp data.bin pristine.bin
run 0 "$XPAR" create -r 4 -s 8K --layout=armoured -o sets data.bin
rm -f data.bin
capture out.bin "$XPAR" extract --stdout sets.xpa
equal "intact extract status" "$status" 0
same out.bin pristine.bin
cdto ..

step "--stdout corrects with the inner code, as --to does"

#  --stdout must apply lazy inner-code correction like directory extraction.
mkdir -p e3 && cdto e3
mkfile data.bin 400000 88
cp data.bin pristine.bin
run 0 "$XPAR" create -r 20% --layout=armoured -o pack data.bin
rm -f data.bin
damage pack.xpa rand=50000,32

rm -rf d1 && mkdir d1
run 0 "$XPAR" extract --to=d1 pack.xpa
same d1/data.bin pristine.bin

capture out.bin "$XPAR" extract --stdout pack.xpa
equal "stdout status on correctable damage" "$status" 0
same out.bin pristine.bin
cdto ..

step "a substituted data volume is rewritten from chain-space offsets"

#  xpar_vol.stream_offset is relative to the generation, but xpar_vset_read
#  takes a chain-space offset. Without stream_base a generation past the
#  first read the wrong bytes, or refused the read outright.
mkdir -p v1 && cdto v1
mkdir tree
mkfile tree/a.bin 60000 11
mkfile tree/b.bin 60000 22
run 0 "$XPAR" create -r 4 -s 8K --layout=split -o sets -R tree
mkfile tree/c.bin 40000 33
run 0 "$XPAR" add -r 4 sets.xpa -R tree

test -f "$set_g1_data" || hard_error "split chain produced no generation data"
base=`"$XPAR" info --json sets.xpa 2> /dev/null |
        sed -n 's/.*"stream_base":\([0-9][0-9]*\).*/\1/p' | head -1`
equal "generation 1 starts past the origin" "`test "${base:-0}" -gt 0 &&
                                              echo yes || echo no`" yes

#  The named volume is corrupt and an intact copy sits under another name,
#  which is what makes repair rewrite the named one.
cp "$set_g1_data" orig.d00
cp "$set_g1_data" spare.dat
damage "$set_g1_data" rand=100,512

run_any "0 1" "$XPAR" repair --in-place sets.xpa
same "$set_g1_data" orig.d00
run 0 "$XPAR" verify sets.xpa
cdto ..

step "recover reproduces a volume the writer replicated into"

#  Recover must match the writer's armoured-size replication threshold.
mkdir -p r1 && cdto r1
mkdir tree
mkfile tree/payload.bin 400000 44

#  The threshold has a 1 MiB floor, so only a large manifest crosses it.
if xpar_config_defined XPAR_DOS; then
  dos_manifest_padding 4000 3
else
  pad=nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn
  pad=$pad$pad$pad
  i=0
  while test "$i" -lt 3000; do
    printf x > "tree/f$i.$pad.txt"
    i=$((i + 1))
  done
fi

run 0 "$XPAR" create -r 16 -s 4K --volumes=8 --layout=split -o sets -R tree

#  Keep the large source tree out of later DOS scans.
mv tree ../r1tree

vols=`ls $set_volumes 2> /dev/null | sort`
test -n "$vols" || hard_error "no recovery volumes were written"

# Require the group to cross the replication threshold.
big=0;  small=0
for n in $vols; do
  if test `wc -c < "$n"` -gt 1000000; then big=$((big + 1))
  else small=$((small + 1)); fi
done
if test "$big" -eq 0 || test "$small" -eq 0; then
  bad "replication threshold not crossed; enlarge the manifest"
else
  ok
fi

ncopy=0
for n in $vols; do cp "$n" "V$ncopy.BAK"; ncopy=$((ncopy + 1)); done
ncopy=0
for n in $vols; do
  rm -f "$n"
  run 0 "$XPAR" recover --volume="`basename $n`" sets.xpa
  same "$n" "V$ncopy.BAK"
  ncopy=$((ncopy + 1))
done
mv ../r1tree tree
run 0 "$XPAR" verify sets.xpa
cdto ..

step "recover rebuilds a volume with the set's own inner code"

#  recover must reuse the set's armour parameters, not CLI defaults.
mkdir -p a1 && cdto a1
mkfile payload.bin 200000 91
for opt in --armour=none --armour-t=48 --armour-field=16 --armour-pct=5; do
  rm -f sets.* orig.bin
  run 0 "$XPAR" create -r 4 -s 8K --layout=sidecar $opt -o sets payload.bin
  v=`ls $set_volumes 2> /dev/null | head -1`
  if test -z "$v"; then
    bad "$opt: create wrote no recovery volume"
    continue
  fi
  cp "$v" orig.bin
  rm -f "$v"
  run 0 "$XPAR" recover --volume="`basename $v`" sets.xpa
  same "$v" orig.bin
  run 0 "$XPAR" verify sets.xpa
done
cdto ..

step "recover thresholds replication on the size as written"

#  Replication thresholds use the armoured critical-group size.
mkdir -p a2 && cdto a2
mkdir tree
mkfile tree/payload.bin 400000 92
if xpar_config_defined XPAR_DOS; then
  dos_manifest_padding 300 3
else
  pad=nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn
  i=0
  while test "$i" -lt 400; do
    printf x > "tree/f$i.$pad.txt"
    i=$((i + 1))
  done
fi

#  High armour overhead straddles the replication threshold.
run 0 "$XPAR" create -r 16 -s 4K --volumes=8 --layout=split \
                     --armour-t=120 -o sets -R tree
mv tree ../a2tree

vols=`ls $set_volumes 2> /dev/null | sort`
test -n "$vols" || hard_error "no recovery volumes were written"
big=0;  small=0
for n in $vols; do
  if test `wc -c < "$n"` -gt 1000000; then big=$((big + 1))
  else small=$((small + 1)); fi
done
if test "$big" -eq 0 || test "$small" -eq 0; then
  bad "all volumes use the same replication; adjust --armour-t"
else
  ok
fi

ncopy=0
for n in $vols; do cp "$n" "V$ncopy.BAK"; ncopy=$((ncopy + 1)); done
ncopy=0
for n in $vols; do
  rm -f "$n"
  run 0 "$XPAR" recover --volume="`basename $n`" sets.xpa
  same "$n" "V$ncopy.BAK"
  ncopy=$((ncopy + 1))
done
mv ../a2tree tree
run 0 "$XPAR" verify sets.xpa
cdto ..

step "a superseded neighbour does not excuse damage in the same slice"

#  A superseded cell must not suppress damage to live cells in its slice.
mkdir -p s1 && cdto s1
mkfile key.bin 32 7
mkdir tree
i=0
while test "$i" -lt 32; do
  mkfile "tree/f$i.bin" 4096 "$((300 + i))"
  i=$((i + 1))
done
run 0 "$XPAR" create -R -s 64K --cell=4K -r 4 --auth-key=key.bin -o sets tree
#  Require a slice that can hold both superseded and live cells.
"$XPAR" info --json --auth-key=key.bin sets.xpa > g.json 2> "$log" ||
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
run 0 "$XPAR" add -r 4 --auth-key=key.bin sets.xpa -R tree
run 0 "$XPAR" verify --chain --fast --auth-key=key.bin sets.xpa

damage tree/f1.bin rand=100,64
differs tree/f1.bin pristine.bin
run_any "1 2" "$XPAR" verify --chain --fast --auth-key=key.bin sets.xpa

#  Generation 0 owns and must report the damaged bytes.
"$XPAR" verify --chain --auth-key=key.bin --json sets.xpa > v.json 2> "$log"
g0=`tr '{' '\n' < v.json |
      sed -n 's/.*"generation_result".*"generation":0,.*"status":"\([a-z]*\)".*/\1/p' |
      head -1`
equal "generation 0 placed the damage" "$g0" repairable
run 0 "$XPAR" repair --chain --in-place --auth-key=key.bin sets.xpa
same tree/f1.bin pristine.bin
cdto ..

step "a truncated file is still scanned for the cells it damaged"

#  Missing or truncated aliases must mark cells even when dedup leaves the
#  canonical stream intact.
mkdir -p t1 && cdto t1
mkdir tree
mkfile tree/a.bin 120000 21
cp tree/a.bin tree/b.bin
mkfile tree/c.bin 120000 22
run 0 "$XPAR" create -R -s 32K --cell=8K -r 8 --dedup=file -o sets tree
cp tree/b.bin pristine.bin

dd if=pristine.bin of=tree/b.bin bs=1 count=60000 2> /dev/null ||
  hard_error "truncate failed"
differs tree/b.bin pristine.bin

run 1 "$XPAR" verify sets.xpa
run 0 "$XPAR" repair --in-place sets.xpa
same tree/b.bin pristine.bin
run 0 "$XPAR" verify sets.xpa

#  A missing file takes the same branch.
rm -f tree/b.bin
run 1 "$XPAR" verify sets.xpa
run 0 "$XPAR" repair --in-place sets.xpa
same tree/b.bin pristine.bin
run 0 "$XPAR" verify sets.xpa
cdto ..

step "a crafted packet key does not run the critical-group rebuild away"

#  A UINT64_MAX packet key must not wrap the rebuild cursor into a loop.
mkdir -p f1 && cdto f1
mkfile payload.bin 200000 31
run 0 "$XPAR" create -r 4 -s 8K --layout=sidecar -o sets payload.bin
v=`ls $set_volumes 2> /dev/null | head -1`
test -n "$v" || hard_error "create wrote no recovery volume"
cp "$v" orig.bin

#  Forge the only AUTH packet with an invalid short body.
for t in "$set_index" `ls $set_volumes 2> /dev/null`; do
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
  ( ulimit -v 2000000; "$XPAR" recover --volume="`basename $v`" sets.xpa ) \
    > "$log" 2>&1 || status=$?
  equal "recover survived the crafted key" "$status" 0
  #  The forged packet changes the volume, but rebuilding stays bounded.
  if test -s "$v"; then
    grew=$((`wc -c < "$v"` - `wc -c < orig.bin`))
    equal "the rebuilt group stayed bounded" "`test "${grew:-999999}" -lt 65536 &&
                                               echo yes || echo no`" yes
  else
    bad "recover wrote no volume"
  fi
else
  note "address-space cap unsupported; skipping runaway test"
fi
cdto ..

step "explain's hand-recovery recipe recovers the data"

# Exercise short pipe reads with a GF(2^16) frame wider than 64 KiB.
big_frame=no
for field in 8 16; do
  mkdir -p x$field && cd x$field || hard_error "cd x$field"
  mkfile p.bin 100000 55
  if run 0 "$XPAR" create -r 20% --layout=armoured \
                   --armour-field=$field --armour-t=16 -o pack p.bin
  then
    "$XPAR" explain pack.xpa 2> "$log" | sed -n '/^set -e$/,$p' > recipe.sh
    if xpar_config_defined XPAR_DOS; then
      sed 's/P___\.XPA/pack.xpa/g' recipe.sh > recipe.host
      mv recipe.host recipe.sh
    fi
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
  cdto ..
done
if test "$big_frame" = yes; then ok
else
  bad "no frame exceeded a pipe buffer; short-read path untested"
fi

step "--json --progress emits progress records, and --json alone does not"

mkdir -p j1 && cdto j1
#  DOS uses less data but still emits a final progress update.
progress_bytes=40000000
xpar_config_defined XPAR_DOS && progress_bytes=2097152
mkfile data.bin "$progress_bytes" 51
run 0 "$XPAR" create -r 4 -s 1M --json --progress -o sets data.bin
"$XPAR" verify --json --progress sets.xpa > p.json 2> "$log"
n=`grep -c '"type":"progress"' p.json`
equal "verify --json --progress reported" "`test "${n:-0}" -gt 0 &&
                                            echo yes || echo no`" yes
equal "the record carries done and rate" \
      "`grep -c '"type":"progress".*"done":[0-9].*"rate_bps":[0-9]' p.json |
          { read c; test "$c" -gt 0 && echo yes || echo no; }`" yes

"$XPAR" verify --json sets.xpa > q.json 2> "$log"
equal "--json alone stays silent" "`grep -c '"type":"progress"' q.json`" 0

"$XPAR" verify --progress sets.xpa > /dev/null 2> h.txt
equal "--progress alone stays human" "`grep -c '"type":"progress"' h.txt`" 0
cdto ..

step "an unwritable destination exits with the I/O status"

#  Exercise the distinct I/O exit status (5).
mkdir -p w1 && cdto w1
if perms_bite .; then
  mkfile data.bin 100000 41
  run 0 "$XPAR" create -r 20% --layout=armoured -o pack data.bin
  mkdir ro && chmod 555 ro
  run 5 "$XPAR" extract --to=ro pack.xpa
  run 5 "$XPAR" create -r 20% --layout=armoured -o ro/roqt data.bin
  chmod 755 ro
  run 0 "$XPAR" extract --to=ro pack.xpa
  same ro/data.bin data.bin
else
  note "mode 555 is writable; skipping I/O test"
fi
cdto ..

step "a volume added later agrees with the ones already there"

#  Added volumes must reuse the set's replicated CRTR packet.
mkdir -p a4 && cdto a4
mkfile p.bin 400000 83

for later in "addrecovery --reproducible -r 12" "addrecovery -r 12"; do
  rm -f sets.* && cp p.bin d.bin
  run 0 "$XPAR" create --reproducible -r 4 -s 32K -o sets d.bin
  run 0 "$XPAR" $later sets.xpa
  #  Exercise chain readers that detect replicated-packet conflicts.
  run 0 "$XPAR" info sets.xpa
  run 0 "$XPAR" repair --in-place sets.xpa
  run 0 "$XPAR" verify sets.xpa
done

rm -f sets.* && cp p.bin d.bin
run 0 "$XPAR" create --reproducible -r 8 -s 32K -o sets d.bin
v=`ls $set_volumes 2> /dev/null | head -1`
test -n "$v" || hard_error "create wrote no recovery volume"
rm -f "$v"
run 0 "$XPAR" recover --volume="`basename $v`" sets.xpa
run 0 "$XPAR" info sets.xpa
run 0 "$XPAR" repair --in-place sets.xpa
cdto ..

step "creator disagreement is tolerated; other conflicts are fatal"

#  CRTR provenance may differ across volumes; other replicated packets may not.
mkdir -p a5 && cdto a5
mkfile p.bin 200000 84
run 0 "$XPAR" create --reproducible -r 4 -s 32K --armour=none -o sets p.bin
run 0 "$XPAR" info sets.xpa

#  Conflicting CRTR provenance is accepted.
"$FORGE" sets.xpa CRTR 78706172203939 || hard_error "forge failed"
run 0 "$XPAR" info sets.xpa
run 0 "$XPAR" verify sets.xpa
run 0 "$XPAR" repair --in-place sets.xpa

#  Conflicting SETD remains fatal.
rm -f sets.* && cp p.bin q.bin
run 0 "$XPAR" create --reproducible -r 4 -s 32K --armour=none -o sets q.bin
"$FORGE" sets.xpa SETD 00112233445566778899aabbccddeeff ||
  hard_error "forge failed"
run 3 "$XPAR" info sets.xpa
cdto ..

step "addrecovery tops up every layout and the result still repairs"

#  Cover critical-group reuse across every layout.
mkdir -p a3 && cdto a3
mkfile p.bin 400000 81
for lay in "--layout=sidecar" "--layout=split" "--layout=armoured"; do
  rm -f sets.* && cp p.bin d.bin
  run 0 "$XPAR" create --reproducible -r 4 -s 32K $lay -o sets d.bin
  before=`ls $set_volumes 2> /dev/null | wc -l | tr -d ' '`
  run 0 "$XPAR" addrecovery --reproducible -r 12 sets.xpa
  run 0 "$XPAR" verify sets.xpa
  after=`ls $set_volumes 2> /dev/null | wc -l | tr -d ' '`
  equal "$lay grew its recovery" "`test "$after" -ge "$before" &&
                                   echo yes || echo no`" yes

  #  Damage beyond the original redundancy; keep armoured hits codeword-sized.
  case "$lay" in
    *sidecar*) tgt=d.bin;                                       run_len=300 ;;
    *split*)   tgt=`ls $set_data_volumes 2> /dev/null | head -1`; run_len=300 ;;
    *)         tgt=sets.xpa;                                     run_len=8 ;;
  esac
  test -n "$tgt" && test -f "$tgt" || hard_error "no damage target for $lay"
  bytes=`wc -c < "$tgt" | tr -d ' '`
  ops=""
  for pct in 40 52 64 76 88; do
    ops="$ops rand=$((bytes / 100 * pct)),$run_len"
  done
  damage "$tgt" $ops > "$log" 2>&1
  run 0 "$XPAR" repair --in-place sets.xpa
  run 0 "$XPAR" verify sets.xpa
  case "$lay" in *sidecar*) same d.bin p.bin ;; esac
done
cdto ..

step "displaced data is found again rather than treated as damage"

#  Cover the shared misplaced-data search in verify and repair.
mkdir -p r2 && cdto r2
mkfile p.bin 900000 71
cp p.bin pristine.bin
run 0 "$XPAR" create --reproducible -r 8 -s 64K -o move p.bin

#  Prepending produces one dominant displacement.
shift_by() {   # shift_by <bytes>
  cp pristine.bin p.bin
  mkfile pad.bin "$1" 72
  cat pad.bin pristine.bin > t.bin && mv t.bin p.bin
}

for n in 1 4096 65536; do
  shift_by $n
  differs p.bin pristine.bin
  run 0 "$XPAR" repair --in-place move.xpa
  same p.bin pristine.bin
done

#  Disabling resync leaves the shifted file damaged.
shift_by 4096
run_any "1 2" "$XPAR" repair --in-place --resync=off move.xpa
differs p.bin pristine.bin

#  Confirm multiple strong displacement candidates.
cp pristine.bin p.bin
mkfile pad1.bin 3 73
mkfile pad2.bin 9999 74
dd if=pristine.bin of=head.bin bs=1 count=400000 2> /dev/null
dd if=pristine.bin of=tail.bin bs=1 skip=400000 2> /dev/null
cat pad1.bin head.bin pad2.bin tail.bin > p.bin
run_any "0 1" "$XPAR" verify move.xpa
run 0 "$XPAR" repair -v --in-place move.xpa
same p.bin pristine.bin
if grep -q "no dominant displacement" "$log"; then
  bad "the default gave up on an ambiguous displacement it can resolve"
else ok; fi

#  A file that never moved must not be reported as displaced.
cp pristine.bin p.bin
run 0 "$XPAR" verify --resync=always move.xpa
if grep -q "displaced slices" "$log"; then
  bad "an undisplaced file was reported as displaced"
else ok; fi
cdto ..

step "asking for no recovery means the same thing to create and to add"

# Zero requests no parity; positive fractions still round up to one.
mkdir -p j2 && cdto j2
mkfile p.bin 200000 91
cp p.bin pristine.bin

for spec in 0 0% 0x; do
  rm -f sets.*
  run 0 "$XPAR" create --reproducible -r $spec -s 32K -o sets p.bin
  n=`"$XPAR" info --json sets.xpa 2> "$log" | tr ',' '\n' |
       sed -n 's/.*"recovery":\([0-9][0-9]*\).*/\1/p' | head -1`
  equal "-r $spec asks for none and gets none" "${n:-x}" 0
  run 0 "$XPAR" verify sets.xpa
done

#  It still detects damage; it simply cannot mend it.
damage p.bin rand=1000,64
run 2 "$XPAR" verify sets.xpa
run 2 "$XPAR" repair --in-place sets.xpa

#  A fraction too small to name a slice is still a request for some.
cp pristine.bin p.bin
rm -f sets.*
run 0 "$XPAR" create --reproducible -r 0.0001% -s 32K -o sets p.bin
n=`"$XPAR" info --json sets.xpa 2> "$log" | tr ',' '\n' |
     sed -n 's/.*"recovery":\([0-9][0-9]*\).*/\1/p' | head -1`
equal "a vanishing percentage rounds up to one" "${n:-x}" 1

# Add follows the same rule.
cp pristine.bin p.bin
rm -f sets.*
run 0 "$XPAR" create --reproducible -r 4 -s 32K -o sets p.bin
mkfile extra.bin 5000 92
run 0 "$XPAR" add --reproducible -r 0 sets.xpa p.bin extra.bin
run 0 "$XPAR" verify --chain sets.xpa
cdto ..

step "a lost cell table can be rebuilt from the slices that survive"

# Rebuild missing cell tables without scaling memory to the archive.
mkdir -p k1 && cdto k1
mkfile p.bin 300000 93
cp p.bin pristine.bin
run 0 "$XPAR" create --armour=none --reproducible -s 16K --cell=4K -r 6 \
    -o sets p.bin

#  Losing every copy of the table drops erasures back to whole slices.
for v in "$set_index" $set_volumes; do
  test -f "$v" && "$DAMAGE" "$v" unpacket=SLCL > /dev/null
done
"$XPAR" scrub sets.xpa > /dev/null 2> "$log" || :
if grep -q "no complete cell table survives" "$log"; then ok
else bad "a set with no SLCL packets did not report the fallback"; fi

run 0 "$XPAR" scrub --rebuild-cells sets.xpa
if grep -q "rebuild-cells: wrote" "$log"; then ok
else bad "--rebuild-cells wrote nothing"; fi

"$XPAR" scrub sets.xpa > /dev/null 2> "$log" || :
if grep -q "no complete cell table survives" "$log"; then
  bad "the rebuilt table did not take"
else ok; fi

# The rebuilt table restores cell-level repair.
damage p.bin -Z 16384 -Y 4096 cell=3,1
run 1 "$XPAR" verify sets.xpa
if grep -q "1 slice, 1 cell" "$log"; then ok
else bad "the rebuilt table did not narrow the damage to one cell"; fi
run 0 "$XPAR" repair --in-place sets.xpa
same p.bin pristine.bin

# Unverified slices cannot seed a cell table.
cp pristine.bin p.bin
rm -f sets.*
run 0 "$XPAR" create --armour=none --reproducible -s 16K --cell=4K -r 6 \
    -o sets p.bin
for v in "$set_index" $set_volumes; do
  test -f "$v" && "$DAMAGE" "$v" unpacket=SLCL > /dev/null
done
damage p.bin rand=40000,64
"$XPAR" scrub --rebuild-cells sets.xpa > /dev/null 2> "$log" || :
if grep -q "cannot seed a cell table" "$log"; then ok
else bad "--rebuild-cells seeded a table from a slice that does not verify"; fi
if grep -q "rebuild-cells: wrote" "$log"; then
  bad "--rebuild-cells wrote a table it could not seed"
else ok; fi
cdto ..

step "a verb written after its options is named as the mistake"

# Diagnose verbs placed after options.
mkdir -p k2 && cdto k2
mkfile verify 60000 94
run 0 "$XPAR" create --reproducible -r 2 -s 16K -o safe verify

run 4 "$XPAR" --json verify safe.xpa
if grep -q "'verify' must come first" "$log"; then ok
else bad "a global option before the verb did not name the verb"; fi

# Diagnose misplaced verbs after verb-specific options too.
run 4 "$XPAR" -f create -o rslt verify
if grep -q "'create' must come first" "$log"; then ok
else bad "a verb option before the verb did not name the verb"; fi

# Correct ordering and -- still work.
run 0 "$XPAR" verify --json safe.xpa
run 0 "$XPAR" --json -- safe.xpa
cdto ..

step "an authenticated set finds displaced data the same way an open one does"

#  Keyed verification and repair must agree on displaced data.
mkdir -p k3 && cdto k3
mkfile p.bin 400000 95
cp p.bin pristine.bin
mkfile auth.key 40 96
run 0 "$XPAR" create --reproducible --auth-key=auth.key -r 20% -s 16K \
    -o sets p.bin

#  Push every slice forward by prepending bytes.
mkfile pad.bin 5000 97
cat pad.bin pristine.bin > p.bin

run 1 "$XPAR" verify --auth-key=auth.key sets.xpa
if grep -q "displaced slices" "$log"; then ok
else bad "a keyed set did not report displaced data"; fi

run 0 "$XPAR" repair --in-place --auth-key=auth.key sets.xpa
same p.bin pristine.bin
run 0 "$XPAR" verify --auth-key=auth.key sets.xpa
cdto ..

step "a lost hard-link name is damage repair can put back"

#  Repair must recreate aliases that verify reports as repairable.
mkdir -p k4 && cdto k4
mkdir tree
mkfile tree/a.bin 200000 98
mkfile tree/c.bin 50000 99
if xpar_hardlinks_work tree/a.bin tree/b.bin; then
  run 0 "$XPAR" create --reproducible -r 20% -s 16K -o sets -R tree
  rm -f tree/b.bin
  run 1 "$XPAR" verify sets.xpa
  run 0 "$XPAR" repair --in-place --keep-journal sets.xpa
  if grep -q "relinked 1 hard-link name" "$log"; then ok
  else bad "repair did not put the hard-link name back"; fi
  exists tree/b.bin
  run 0 "$XPAR" verify sets.xpa

  #  The name did not exist before the repair, so undo removes it.
  run 0 "$XPAR" undo sets.xpa
  if test -e tree/b.bin; then bad "undo left the recreated link behind"
  else ok; fi
else
  note "this filesystem has no hard links; skipped"
fi
cdto ..

step "a summary status means the same thing in every verb"

#  A verdict must use the same status with and without --chain.
mkdir -p k5 && cdto k5
mkfile p.bin 300000 100
run 0 "$XPAR" create --reproducible -r 4 -s 16K -o sets p.bin
mkfile extra.bin 30000 101
run 0 "$XPAR" add --reproducible -r 4 sets.xpa p.bin extra.bin
damage p.bin rand=1000,200000

word() {   # word <verb-and-flags...>
  "$XPAR" "$@" --json sets.xpa 2> /dev/null |
    tr ',' '\n' | sed -n 's/.*"status":"\([a-z-]*\)".*/\1/p' | tail -1
}
for verb in verify scrub; do
  one=`word $verb`
  all=`word $verb --chain`
  equal "$verb says the same word with and without --chain" "${one:-x}" \
        "${all:-y}"
  equal "$verb calls exit 2 unrepairable" "${one:-x}" unrepairable
done
cdto ..

step "consolidate --dry-run reports without doing the work"

#  Dry runs must not extract or stage an armoured archive.
mkdir -p k6 && cdto k6
consolidate_bytes=2000000
xpar_config_defined XPAR_DOS && consolidate_bytes=262144
mkfile p.bin "$consolidate_bytes" 102
run 0 "$XPAR" create --layout=armoured --reproducible -r 10% -s 64K \
    -o sets p.bin
mkfile extra.bin 100000 103
run 0 "$XPAR" add --reproducible -r 10% sets.xpa p.bin extra.bin

run 0 "$XPAR" consolidate --dry-run --output=rslt sets.xpa
if grep -q "reclaim" "$log"; then ok
else bad "the dry run reported no reclaim figure"; fi
#  Nothing was written: neither the output nor a staging directory.
if test -e rslt.xpa; then bad "--dry-run wrote its output"; else ok; fi
n=`/bin/ls -a | grep -c xpar-consolidate || :`
equal "no staging directory remains" "${n:-0}" 0

#  Confirm normal consolidation still works.
run 0 "$XPAR" consolidate --output=rslt sets.xpa
run 0 "$XPAR" verify rslt.xpa
cdto ..

step "explain uses the resolved input name"

# Use the resolved split-volume name in recipes.
mkdir -p i1 && cdto i1
mkfile p.bin 200000 90
run 0 "$XPAR" create --reproducible -r 20% --layout=split -o phot p.bin
"$XPAR" explain phot > r.txt 2> "$log"
name=`sed -n 's/^in=//p' r.txt | head -1`
if xpar_config_defined XPAR_DOS; then
  equal "the recipe reads the resolved name" "$name" "PHOT.XPA"
  name=phot.xpa
else
  equal "the recipe reads the resolved name" "$name" "phot.xpa"
fi
exists "$name"
cdto ..

step "--deep names the missing data rather than blaming the parity"

# Missing data must not be reported as bad parity.
mkdir -p h1 && cdto h1
mkfile p.bin 400000 89
run 0 "$XPAR" create --reproducible -r 8 -s 32K -o sets p.bin
run 0 "$XPAR" scrub --deep sets.xpa
rm -f p.bin
"$XPAR" scrub --deep sets.xpa > "$log" 2>&1
if grep -q "do not recompute from the data" "$log"; then
  bad "missing data was reported as bad parity"
else ok; fi
# Match the diagnostic prefix, not its wording.
if grep -q "^xpar: --deep: " "$log"; then ok
else bad "--deep did not say why it could not check the parity"; fi
cdto ..

step "the reader rejects what the format says it must"

# Reject reserved fields, reserved attribute bits and invalid generators.
mkdir -p g1 && cdto g1
mkfile p.bin 200000 88
run 0 "$XPAR" create --reproducible -r 4 -s 32K --armour=none -o sets p.bin
run 0 "$XPAR" info sets.xpa

# STRM with a nonzero reserved field.
"$FORGE" sets.xpa STRM 0000000000000000ffffffffffffffff ||
  hard_error "forge failed"
run 0 "$XPAR" info sets.xpa
note "a malformed STRM is skipped rather than parsed"

# RCVS with a nonzero reserved field must not count as recovery.
rm -f sets.* && cp p.bin q.bin
run 0 "$XPAR" create --reproducible -r 4 -s 32K --armour=none -o sets q.bin
before=`"$XPAR" info --json sets.xpa 2> "$log" | tr ',' '\n' |
          sed -n 's/.*"recovery":\([0-9][0-9]*\).*/\1/p' | head -1`
"$FORGE" sets.xpa RCVS 0000000000000063ffffffffffffffff || hard_error "forge"
after=`"$XPAR" info --json sets.xpa 2> "$log" | tr ',' '\n' |
         sed -n 's/.*"recovery":\([0-9][0-9]*\).*/\1/p' | head -1`
equal "a reserved RCVS field is not accepted" "$after" "$before"
run 0 "$XPAR" verify sets.xpa
cdto ..

step "every armour field works on every layout"

# Cover every field/layout combination when parity uses defaults.
mkdir -p f2 && cdto f2
mkfile p.bin 200000 87
for lay in --layout=sidecar --layout=split --layout=armoured; do
  for fld in "" --armour-field=8 --armour-field=16; do
    rm -f sets.* && cp p.bin d.bin
    run 0 "$XPAR" create --reproducible -r 20% $lay $fld -o sets d.bin
    run 0 "$XPAR" verify sets.xpa
  done
done
cdto ..

step "the inner code corrects exactly what its parameters promise"

#  At depth 1, n corrupt bytes hit n symbols in one codeword; depth D spreads
#  them over D codewords. The outer code does not protect critical groups.
mkdir -p c1 && cdto c1
mkfile p.bin 200000 61

# corrects <expected> <create options...>
corrects() {
  want=$1;  shift
  rm -f safe.* clean.bin
  if ! attempt "$XPAR" create --reproducible -r 20% "$@" -o safe p.bin; then
    return
  fi
  if test "$status" -ne 0; then
    bad "create $* exited $status"
    return
  fi
  cp safe.xpa clean.bin
  got=0
  #  Check the correction boundary and the next value.
  for n in "$want" "$((want + 1))"; do
    cp clean.bin safe.xpa
    damage safe.xpa rand=600,$n > "$log" 2>&1
    "$XPAR" verify safe.xpa > "$log" 2>&1
    grep -q '0 past the inner code' "$log" || break
    got=$n
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
cdto ..

step "prune: refuses a lossy removal, and performs a forced one"

#  prune had no coverage at all, though it is destructive and its -f
#  semantics decide whether entries survive.
mkdir -p p1 && cdto p1
mkfile a.bin 80000 66
mkfile keep.bin 1000 67
run 0 "$XPAR" create -r 4 -s 8K -o sets a.bin keep.bin
mkfile a.bin 90000 77
#  Let unchanged entries be inherited.
run 0 "$XPAR" add -r 4 --rescan=none sets.xpa a.bin
cp a.bin pristine.bin

gens() {
  "$XPAR" info --json sets.xpa 2> /dev/null |
    sed -n 's/.*"generations":\([0-9][0-9]*\).*/\1/p' | head -1
}
equal "chain length before pruning" "`gens`" 2
snapshot=`files`

#  A generation a survivor still depends on is refused, and nothing moves.
run 4 "$XPAR" prune --dry-run --before=1 sets.xpa
equal "dry run changed nothing" "`files`" "$snapshot"
run 4 "$XPAR" prune --before=1 sets.xpa
equal "refusal changed nothing" "`files`" "$snapshot"
equal "chain length after refusal" "`gens`" 2
#  Generation 0 cannot recover its superseded bytes, but nothing is
#  damaged, so the verdict is clean and the warning carries the news.
run 0 "$XPAR" verify sets.xpa
if grep -q 'count as erasures' "$log"
then ok
else bad "verify did not report generation 0's superseded slices"
fi

#  --force accepts the loss, and what survives has to remain coherent.
run 0 "$XPAR" prune -f --before=1 sets.xpa
equal "chain collapsed to one generation" "`gens`" 1
run 0 "$XPAR" verify sets.xpa
#  A sidecar set protects the files in place, so the survivor is on disk.
same a.bin pristine.bin
cdto ..

step "an unselected multi-generation set covers the whole ancestry"

#  Default verification must cover inherited bytes.
mkdir -p ch1 && cdto ch1
mkfile a.bin 400000 41
mkfile b.bin 200000 42
run 0 "$XPAR" create -s 4096 -r 20 -o sets a.bin
run 0 "$XPAR" add -s 4096 -r 20 sets.xpa a.bin b.bin
cp a.bin pristine.bin

#  Ten slices is well inside generation 0's twenty recovery slices.
damage a.bin rand=12288,64 rand=40960,64 rand=69632,64 rand=98304,64 \
             rand=126976,64 rand=155648,64 rand=184320,64 rand=212992,64 \
             rand=241664,64 rand=270336,64

run 1 "$XPAR" verify sets.xpa
equal "the newer pass names the owning generation" \
      "`said 'inherited from generation 0'`" yes
equal "no checksum-invisible verdict" "`said 'checksum-invisible'`" no
run 0 "$XPAR" repair --in-place sets.xpa
same a.bin pristine.bin
run 0 "$XPAR" verify sets.xpa
cdto ..

step "inherited damage past the owner's budget stays unrepairable"

#  Use the worst ancestry verdict and preserve undecodable files.
mkdir -p ch2 && cdto ch2
mkfile a.bin 400000 43
mkfile b.bin 200000 44
run 0 "$XPAR" create -s 4096 -r 8 -o sets a.bin
run 0 "$XPAR" add -s 4096 -r 20 sets.xpa a.bin b.bin

damage a.bin rand=4096,64 rand=12288,64 rand=20480,64 rand=28672,64 \
             rand=36864,64 rand=45056,64 rand=53248,64 rand=61440,64 \
             rand=69632,64 rand=77824,64 rand=86016,64 rand=94208,64
cp a.bin damaged.bin

run 2 "$XPAR" verify sets.xpa
run 2 "$XPAR" repair --in-place sets.xpa
same a.bin damaged.bin
cdto ..

step "a chain pass never truncates an entry a later generation owns"

#  A generation must not repair bytes it does not own.
mkdir -p ch3 && cdto ch3
mkfile a.bin 200000 45
mkfile b.bin 60000 46
run 0 "$XPAR" create -s 4096 -r 20 -o sets a.bin
run 0 "$XPAR" add -s 4096 -r 20 sets.xpa a.bin b.bin
cp b.bin pristine.bin

#  Insert a run so every slice after it is displaced and the file is long.
dd if=b.bin of=part1 bs=1000 count=30 2> /dev/null
dd if=b.bin of=part2 bs=1000 skip=30 2> /dev/null
mkfile pad 333 47
cat part1 pad part2 > b.bin
rm -f part1 part2 pad

run 0 "$XPAR" repair --in-place sets.xpa
same b.bin pristine.bin
run 0 "$XPAR" verify sets.xpa
cdto ..

step "superseded bytes count as erasures in the verdict that reports them"

#  Superseded ancestor cells are erasures for the decoder, so they are
#  reported and they deepen a column; with nothing damaged there is
#  nothing to decode and the verdict stays clean.
mkdir -p ch4 && cdto ch4
mkfile a.bin 400000 48
mkfile b.bin 200000 49
run 0 "$XPAR" create -s 4096 -r 20 -o sets a.bin b.bin
mkfile b.bin 200000 50
run 0 "$XPAR" add -s 4096 -r 20 sets.xpa a.bin b.bin

run 0 "$XPAR" verify sets.xpa
equal "the cause is named" "`said 'count as erasures'`" yes

#  Real damage has to be localised against that depth, and cannot be.
damage a.bin rand=1000,64
run 2 "$XPAR" verify sets.xpa
run 2 "$XPAR" repair --in-place --dry-run sets.xpa
cdto ..

step "repair never reports a clean tree while damage stands unlocalised"

#  Detect object-kind mismatches without cell evidence.
mkdir -p ch5 && cdto ch5
mkdir tree
mkfile tree/d.bin 200000 51
if symlinks_work d.bin tree/rel.lnk; then
  run 0 "$XPAR" create -r 10% -o sets -R tree
  rm tree/rel.lnk
  mkfile tree/rel.lnk 100 52
  run 2 "$XPAR" verify sets.xpa
  run 2 "$XPAR" repair --in-place sets.xpa
  equal "no clean verdict was printed" "`said 'no damage found'`" no
  mkdir out
  run 0 "$XPAR" repair --to out sets.xpa
  equal "--to rebuilt the link" \
        "`test -L out/tree/rel.lnk && echo yes || echo no`" yes
else
  note "symbolic links unsupported; skipped"
fi
cdto ..

step "-r delivers the redundancy it was asked for across the field bound"

#  Re-derive R whenever geometry changes.
mkdir -p rr && cdto rr
mkfile data.bin 1048576

#  GF(2^8) reaches the S+R limit with a small fixture.
check_recovery() {   # check_recovery <set> <wanted bytes> <what>
  read_geometry "$1"
  _got=$((R * Z))
  _slack=$((Z + $2 / 50))
  if test "$_got" -ge $(($2 - _slack)) &&
     test "$_got" -le $(($2 + _slack)); then ok
  else bad "$3: $_got recovery bytes, wanted about $2 (Z $Z, S $S, R $R)"
  fi
}

run 0 "$XPAR" create --field=8 -r 50% -o half data.bin
check_recovery half.xpa 524288 "-r 50%"
run 0 "$XPAR" verify half.xpa

run 0 "$XPAR" create --field=8 -r 2x -o twic data.bin
check_recovery twic.xpa 2097152 "-r 2x"
run 0 "$XPAR" verify twic.xpa

run 0 "$XPAR" create --field=8 -r 4M -o abst data.bin
check_recovery abst.xpa 4194304 "-r 4M"
run 0 "$XPAR" verify abst.xpa

#  A count is exact, so Z moves and R does not.
run 0 "$XPAR" create --field=8 -r 200 -o coun data.bin
read_geometry coun.xpa
equal "-r 200 kept the count" "$R" 200

#  Field overflow is a usage error that reports the limit.
run 4 "$XPAR" create --field=8 -r 1000x -o over data.bin
if grep -q 'Field limit' "$log"; then ok
else bad "the refusal does not state the reach the field still allows"; fi
cdto ..

step "a split volume of the wrong length is damaged, not absent"

#  A length mismatch must not discard the whole volume.
mkdir -p sl && cdto sl
mkfile f.bin 1048576 61
run 0 "$XPAR" create -r 20% -s 4K --layout=split --volumes=4 -o data f.bin

save_volumes() { rm -rf keepvol && mkdir keepvol && cp data.d0* keepvol/; }
restore_volumes() { cp keepvol/data.d0* .; }
save_volumes

#  Erasures must stay proportional to the bytes actually lost.
resized() {   # resized <what>
  run 1 "$XPAR" verify data.xpa
  if grep -q "unrepairable" "$log"; then
    bad "$1: a resized volume was treated as an absent one"
  else ok; fi
  run 0 "$XPAR" repair --in-place data.xpa
  cat data.d0* > joined.bin
  same joined.bin f.bin
  rm -f joined.bin
  restore_volumes
}

#  Shorter by one byte, by a whole slice, and longer by one byte.
"$DAMAGE" data.d01 "truncate=$((`cat data.d01 | nbytes` - 1))" > /dev/null ||
  hard_error "cannot shorten data.d01"
resized "one byte short"
"$DAMAGE" data.d01 "truncate=$((`cat data.d01 | nbytes` - 4096))" > /dev/null ||
  hard_error "cannot shorten data.d01 by a slice"
resized "one slice short"
"$DAMAGE" data.d01 extend=1 > /dev/null || hard_error "cannot extend data.d01"
resized "one byte long"
cdto ..

step "a missing split volume is not substituted by a same-length one"

#  Do not substitute a same-sized volume without matching certificates.
mkdir -p sw && cdto sw
split_bytes=3000000
xpar_config_defined XPAR_DOS && split_bytes=1048576
mkfile p.bin "$split_bytes" 62
mkdir set && mv p.bin set/p.bin && cp set/p.bin whole.bin
cdto set
run 0 "$XPAR" create -s 64K --cell=16K -r 25% --layout=split --volumes=4 \
    -o disc p.bin
rm -f p.bin disc.d00
run 1 "$XPAR" verify disc.xpa
if grep -q "is missing; using" "$log"; then
  bad "a same-length volume was accepted as a substitute"
else ok; fi
run 0 "$XPAR" repair --in-place disc.xpa
exists disc.d00
cat disc.d0* > joined.bin
same joined.bin ../whole.bin
cdto ../..

step "a renamed split volume is restored under its recorded name"

#  Repair must restore the recorded name after substitution.
mkdir -p sr && cdto sr
split_bytes=3000000
xpar_config_defined XPAR_DOS && split_bytes=1048576
mkfile p.bin "$split_bytes" 65
mkdir set && mv p.bin set/p.bin && cp set/p.bin whole.bin
cdto set
run 0 "$XPAR" create -s 64K --cell=16K -r 25% --layout=split --volumes=4 \
    -o disc p.bin
rm -f p.bin
mv disc.d01 other.bin
#  Put the wrong mapped volume at the missing name. Repair must release its
#  other hard-link name before replacing this one.
ln disc.d00 disc.d01 2> /dev/null ||
  note "this filesystem has no hard links; skipped aliased volume mapping"
run 1 "$XPAR" verify disc.xpa
if grep -q "restored under its recorded name" "$log"; then ok
else bad "verify called an incomplete layout clean"; fi
run 0 "$XPAR" repair --in-place disc.xpa
exists disc.d01
#  The user's own copy is read, never moved or removed.
exists other.bin
cat disc.d0* > joined.bin
same joined.bin ../whole.bin
rm -f joined.bin
run 0 "$XPAR" verify disc.xpa
if grep -q "is missing; using" "$log"; then
  bad "the substitution survived the repair"
else ok; fi
cdto ../..

step "extract reads a damaged split volume rather than calling it missing"

#  Extract must distinguish damaged volumes from missing ones.
mkdir -p xd && cdto xd
mkdir tree
mkfile tree/a.bin 200000 63
mkfile tree/b.bin 300000 64
run 0 "$XPAR" create --layout=split --volumes=3 -o safe -R tree
damage safe.d00 flip=1000,1
run 1 "$XPAR" verify safe.xpa
rm -rf out
attempt "$XPAR" extract --to=out safe.xpa
if test "$status" -eq 4; then
  bad "extract called a present volume missing"
else ok; fi
if grep -q "is missing" "$log"; then
  bad "extract reported a present volume as missing"
else ok; fi
#  The undamaged entry still has to come out.
same out/tree/b.bin tree/b.bin
cdto ..

step "a set never protects, or overwrites, its own volumes"

#  Never include a set's own outputs as inputs.
mkdir -p own && cdto own
mkfile a.bin 100000 71
run 0 "$XPAR" create -r 20% -o bkup -R .
run 0 "$XPAR" create -f -r 20% -o bkup -R .
run 0 "$XPAR" verify bkup.xpa
#  The report goes outside the tree so it is not an input itself.
"$XPAR" info --json bkup.xpa > "$work/own.json" 2> "$log"
equal "only the input is stored" "`json_num "$work/own.json" files set`" 1
#  A generation walks the same directory and must not ingest it either.
mkfile b.bin 1000 72
run 0 "$XPAR" add -r 20% bkup.xpa -R .
"$XPAR" info --json bkup.xpa > "$work/own.json" 2> "$log"
equal "the generation stored only the new input" \
      "`json_num "$work/own.json" files set`" 2
run 0 "$XPAR" verify bkup.xpa
cdto ..

step "the recovery spill never writes through a planted name"

#  Spill creation must not follow planted links.
mkdir -p spill && cdto spill
mkfile data.bin 16777216 73
echo VICTIM > victim.txt
cp victim.txt victim.orig
: > out.xpar-tmp-plain
#  A host that copies instead of linking still plants the name.
if symlinks_work victim.txt out.xpar-tmp; then linked=yes
else linked=no;  cp victim.txt out.xpar-tmp; fi
run 0 "$XPAR" create -o rslt -r 30% -m 4M -s 64K data.bin
same victim.txt victim.orig
exists out.xpar-tmp-plain
if test "$linked" = yes; then
  if test -L out.xpar-tmp; then ok
  else bad "the planted symlink was replaced"; fi
else
  same out.xpar-tmp victim.orig
  note "symbolic links unsupported; skipped"
fi
run 0 "$XPAR" verify rslt.xpa
cdto ..

step "an input that protects nothing is refused, not written"

#  Refuse inputs that produce no protected entries.
mkdir -p nothing && cdto nothing
mkdir sub
mkfile sub/x.bin 1000 74
run 4 "$XPAR" create -o setB sub
if test -e setB.xpa; then bad "a refused create still wrote a set"; else ok; fi
run 0 "$XPAR" create -o setB -R sub
run 0 "$XPAR" verify setB.xpa
if fifos_work p1; then
  run 4 "$XPAR" create -o er01 p1
  run 4 "$XPAR" create -o em02 --no-verify-after p1
  if test -e er01.xpa || test -e em02.xpa; then
    bad "a set with no entry was published"
  else ok; fi
else
  note "FIFOs unsupported; the empty-manifest case is skipped"
fi
cdto ..

step "a name in a diagnostic cannot drive the terminal"

#  Escape control bytes in diagnostic names.
mkdir -p ctrl && cdto ctrl
esc=`printf 'a\033[31mb'`
run 3 "$XPAR" verify "$esc"
if grep -q 'x1B' "$log"; then ok
else bad "the control byte was not escaped"; fi
if grep -q "`printf '\033'`" "$log"; then
  bad "a raw escape byte reached the diagnostic"
else ok; fi
cdto ..

step "an empty -o is a usage error"

#  Refuse an empty output name.
mkdir -p emptyo && cdto emptyo
mkfile f.bin 1000 75
run 4 "$XPAR" create -o '' f.bin
run 4 "$XPAR" create --output= f.bin
if test -e f.bin.xpa; then bad "an empty -o still named the set"; else ok; fi
run 0 "$XPAR" create -o safe f.bin
run 0 "$XPAR" verify safe.xpa
cdto ..

step "--scan=DIR finds volumes that are not beside the set"

#  All set-reading verbs must honor --scan.
mkdir -p scan1 && cdto scan1
mkfile f.bin 300000 91
cp f.bin pristine.bin
run 0 "$XPAR" create -r 30% -o disc f.bin
read_geometry disc.xpa
mkdir far
mv $disc_volumes far/
#  Without --scan the recovery is gone, and with it every slice is back.
run 0 "$XPAR" verify disc.xpa
if grep -q "not on disk" "$log"; then ok
else bad "the moved recovery volumes were not reported missing"; fi
run 0 "$XPAR" verify --scan=far disc.xpa
if grep -q "not on disk" "$log"; then
  bad "--scan did not find the volumes in far/"
else ok; fi
"$XPAR" verify --json --scan=far disc.xpa > got.json 2> "$log"
equal "recovery seen through --scan" \
      "`json_num got.json recovery_available summary`" "$R"
#  Every verb that opens a set has to take it, not just verify.
run 0 "$XPAR" scrub --scan=far disc.xpa
run 0 "$XPAR" info  --scan=far disc.xpa
run 0 "$XPAR" list  --scan=far disc.xpa
#  And repair must actually use what it finds.
damage f.bin -Z "$Z" -Y "$Y" -n 96 seed=$XPAR_TEST_SEED cell=3,0 cell=5,0
run 1 "$XPAR" verify --scan=far disc.xpa
run 0 "$XPAR" repair --scan=far --in-place disc.xpa
same f.bin pristine.bin
run 0 "$XPAR" verify --scan=far disc.xpa
#  A chain walk consults it as well, generation index volumes included.
mkfile g.bin 100000 92
run 0 "$XPAR" add -r 30% disc.xpa g.bin
mv $disc_g1_volumes far/
run 0 "$XPAR" verify --chain --scan=far disc.xpa
run 0 "$XPAR" list --chain --scan=far disc.xpa
cdto ..

step "add and consolidate keep the redundancy the chain already has"

#  Inherit redundancy when -r is omitted.
mkdir -p inh1 && cdto inh1
mkdir tree
mkfile tree/a.bin 200000 93
mkfile tree/b.bin 200000 94
run 0 "$XPAR" create -r 20% -R -o safe tree
read_geometry safe.xpa
base_r=$R;  base_s=$S
mkfile tree/c.bin 200000 95
run 0 "$XPAR" add safe.xpa -R tree
run 0 "$XPAR" verify --chain safe.xpa
#  R/S of the new generation must match the old ratio, not 5%.
"$XPAR" info --json --generation=1 safe.xpa > g1.json 2> "$log"
new_r=`json_num g1.json recovery set`
new_s=`json_num g1.json slices   set`
if test -n "$new_r" && test -n "$new_s" &&
   test "$((new_r * base_s))" -ge "$((base_r * new_s))"; then ok
else bad "add thinned the set: was $base_r/$base_s, now $new_r/$new_s"; fi
run 0 "$XPAR" consolidate --replace safe.xpa
run 0 "$XPAR" verify safe.xpa
read_geometry safe.xpa
if test "$((R * base_s))" -ge "$((base_r * S))"; then ok
else bad "consolidate thinned the set: was $base_r/$base_s, now $R/$S"; fi
cdto ..

step "an output base may not end in a generation suffix"

#  Reject output bases that collide with generation names.
mkdir -p gname && cdto gname
mkfile a.bin 50000 96
mkfile b.bin 50000 97
run 0 "$XPAR" create -o bkup a.bin
run 4 "$XPAR" create -o bkup.g001 b.bin
run 4 "$XPAR" create -o othr.g12 b.bin
if test -e bkup.g001.xpa; then
  bad "the colliding set was written anyway"
else ok; fi
run 0 "$XPAR" verify bkup.xpa
run 4 "$XPAR" add -o qset.g001 bkup.xpa b.bin
run 4 "$XPAR" consolidate -o wxyz.g007 bkup.xpa
#  A base that merely contains a 'g' is fine.
run 0 "$XPAR" create -o plan.gz b.bin
#  A forked chain has to name its branches, or --generation is unusable.
run 0 "$XPAR" create -o othr b.bin
cp othr.xpa "$bkup_g2_index"
run 4 "$XPAR" verify bkup.xpa
branch=`sed -n 's/.*--generation=\([0-9a-f][0-9a-f]*\).*/\1/p' "$log" | head -1`
if test -n "$branch"; then ok
else bad "the forked diagnostic named no branch to select"; fi
run 0 "$XPAR" verify --generation="$branch" bkup.xpa
cdto ..


step "split gen-1 volume damage repairs byte-identically"

#  Repair split volumes with nonzero stream bases.
mkdir -p sg && cdto sg
mkfile f1.bin 300000 41
mkfile f2.bin 250000 42
cp f1.bin f1.keep
cp f2.bin f2.keep
run 0 "$XPAR" create --layout=split --volumes=2 -r 25% -o data f1.bin
run 0 "$XPAR" add -r 25% data.xpa f1.bin f2.bin
cp "$data_g1_data" g1data.keep
rm -f f1.bin f2.bin
dv=$data_g1_data
test -n "$dv" || hard_error "gen-1 data volume not found"
damage "$dv" rand=1000,2000
#  Drop one gen-1 recovery volume so the codec must actually decode.
rv=`ls $data_g1_volumes 2> /dev/null | sort | head -1`
test -n "$rv" || hard_error "gen-1 recovery volume not found"
rm -f "$rv"
run 0 "$XPAR" repair --in-place data.xpa
run 0 "$XPAR" verify data.xpa
same "$data_g1_data" g1data.keep
rm -rf out
run 0 "$XPAR" extract --to out data.xpa
same out/f1.bin f1.keep
same out/f2.bin f2.keep
cdto ..

step "an unknown critical packet is refused by every reader"

#  Every reader must reject unknown critical packets.
mkdir -p uc && cdto uc
mkfile p.bin 200000 51
run 0 "$XPAR" create -r 20% -o safe p.bin
cp safe.xpa s.clean
for verb in list info explain scrub; do
  cp s.clean safe.xpa
  "$FORGE" safe.xpa ZZZZ 00112233 1 > "$log" 2>&1 || hard_error "forge failed"
  run 3 "$XPAR" $verb safe.xpa
done
cp s.clean safe.xpa
"$FORGE" safe.xpa ZZZZ 00112233 1 > "$log" 2>&1 || hard_error "forge failed"
run 3 "$XPAR" extract --to uout safe.xpa
cp s.clean safe.xpa
"$FORGE" safe.xpa ZZZZ 00112233 1 > "$log" 2>&1 || hard_error "forge failed"
run 3 "$XPAR" repair --in-place safe.xpa
#  A non-critical unknown packet is skipped everywhere.
cp s.clean safe.xpa
"$FORGE" safe.xpa ZZZZ 00112233 0 > "$log" 2>&1 || hard_error "forge failed"
run 0 "$XPAR" verify safe.xpa
run 0 "$XPAR" list safe.xpa
cdto ..

step "repair rewrites a stale volume from intact packet replicas"

#  Rewrite corrupt critical packets from replicas.
mkdir -p rr && cdto rr
mkfile p.bin 500000 61
run 0 "$XPAR" create --armour=none -r 20% -o safe p.bin
cp safe.xpa s.keep
off=`"$DAMAGE" safe.xpa find=SETD | head -1`
test -n "$off" || hard_error "no SETD in index"
#  Corrupt one byte of the index copy of SETD; the recovery volumes keep
#  intact replicas.
damage safe.xpa "rand=$((off + 4)),1"
run 0 "$XPAR" verify safe.xpa
grep -q 'replicas used' "$log" || bad "verify did not report replica use"
run 0 "$XPAR" repair --in-place safe.xpa
run 0 "$XPAR" verify safe.xpa
if grep -q 'replicas used' "$log"; then
  bad "verify still reports a stale volume after repair"
else ok; fi
same safe.xpa s.keep
cdto ..

step "the default armour still armours the metadata group"

#  Every layout accepts each armour level; non-archive default is metadata.
mkdir -p am && cdto am
mkfile p.bin 100000 71
run 0 "$XPAR" create --armour=all -r 10% -o arch p.bin
run 0 "$XPAR" create --layout=split --volumes=2 --armour=all -r 10% -o back p.bin
run 0 "$XPAR" create --layout=armoured --armour=all -r 10% -o code p.bin
run 0 "$XPAR" create --armour=metadata -r 10% -o data p.bin
run 0 "$XPAR" create -r 10% -o emit p.bin
#  Both the explicit metadata request and the default produce a readable
#  metadata-armoured set.
run 0 "$XPAR" verify arch.xpa
run 0 "$XPAR" verify back.xpa
run 0 "$XPAR" verify data.xpa
run 0 "$XPAR" verify emit.xpa
#  --armour-t stays valid on these layouts; it tunes the inner code.
run 0 "$XPAR" create --armour-t=24 -r 10% -o full p.bin

#  Info reports the armour level found on disk.
capture ia "$XPAR" info arch.xpa
equal "info reports level all" "`grep -c 'level all' ia`" 1
capture id "$XPAR" info data.xpa
equal "info reports level metadata" "`grep -c 'level metadata' id`" 1
capture ie "$XPAR" info emit.xpa
equal "the default is metadata" "`grep -c 'level metadata' ie`" 1
cdto ..

step "--armour=all wraps every recovery slice and table packet"

#  --armour=all wraps every RCVS; metadata armour leaves them bare.
mkdir -p aw && cdto aw
mkfile p.bin 400000 73
run 0 "$XPAR" create -r 20% -s 8K --slice-tag=16 --cell=4K --armour=all \
                     -o arch p.bin
run 0 "$XPAR" create -r 20% -s 8K --slice-tag=16 --cell=4K -o meta p.bin
equal "an --armour=all set leaves nothing bare" \
      "`bare_packets arch.xpa \`volume_files arch\``" 0
if test "`bare_packets meta.xpa \`volume_files meta\``" -gt 0; then ok
else bad "the metadata set was expected to store those packets bare"; fi
cdto ..

step "rot inside an --armour=all recovery volume costs no slices"

#  Damage one symbol per codeword, within t=16.
mkdir -p ai1 && cdto ai1
armour_bytes=1000000
xpar_config_defined XPAR_DOS && armour_bytes=400000
mkfile d.bin "$armour_bytes" 74
cp d.bin pristine.bin
run 0 "$XPAR" create -r 20% -s 8K --armour=all -o safe d.bin
v=`volume_files safe | sort | tail -1`
test -n "$v" || hard_error "no recovery volume was written"
ops=""
for off in `"$DAMAGE" "$v" find=ARMG`; do
  ops="$ops flip=$((off + 150)),1 flip=$((off + 1500)),1"
done
test -n "$ops" || hard_error "the recovery volume carries no ARMG packet"
damage "$v" $ops
#  Damage the data as well, within the outer code's budget.
damage d.bin rand=100000,4096 rand=300000,4096
run 1 "$XPAR" verify safe.xpa
equal "verify reports the inner corrections" \
      "`grep -c 'armoured regions:' "$log"`" 1
if grep -q 'recovery slices failed their checksum' "$log"; then
  bad "single-byte rot still cost recovery slices under --armour=all"
else
  ok
fi
run 0 "$XPAR" repair --in-place safe.xpa
equal "repair regenerated no recovery slices" \
      "`grep -c 'recovery slice.* regenerated' "$log"`" 0
same d.bin pristine.bin
run 0 "$XPAR" verify safe.xpa
cdto ..

step "the same rot costs slices when only the metadata is armoured"

#  The contrast that makes the level worth having.
mkdir -p ai2 && cdto ai2
armour_bytes=1000000
xpar_config_defined XPAR_DOS && armour_bytes=400000
mkfile d.bin "$armour_bytes" 74
cp d.bin pristine.bin
run 0 "$XPAR" create -r 20% -s 8K -o safe d.bin
v=`volume_files safe | sort | tail -1`
test -n "$v" || hard_error "no recovery volume was written"
ops=""
for off in `"$DAMAGE" "$v" find=RCVS`; do
  ops="$ops flip=$((off + 150)),1"
done
test -n "$ops" || hard_error "the recovery volume carries no RCVS packet"
damage "$v" $ops
damage d.bin rand=100000,4096 rand=300000,4096
run 1 "$XPAR" verify safe.xpa
equal "the unarmoured slices were lost" \
      "`grep -c 'recovery slices failed their checksum' "$log"`" 1
run 0 "$XPAR" repair --in-place safe.xpa
equal "repair had to regenerate them" \
      "`grep -c 'recovery slices regenerated' "$log"`" 1
same d.bin pristine.bin
cdto ..

step "wrapped slice tables survive the same rot"

#  --armour=all also protects SLTG and SLCL.
mkdir -p ai3 && cdto ai3
mkfile d.bin 900000 76
run 0 "$XPAR" create -r 20% -s 8K --slice-tag=16 --cell=4K --armour=all \
                     -o safe d.bin
ops=""
n=0
for off in `"$DAMAGE" safe.xpa find=ARMG`; do
  n=$((n + 1))
  #  The first ARMG in an index volume is the critical group.
  if test "$n" -gt 1; then
    ops="$ops flip=$((off + 64)),1 flip=$((off + 128)),1"
  fi
done
test -n "$ops" || hard_error "the index volume carries no wrapped table"
damage safe.xpa $ops
run 0 "$XPAR" verify safe.xpa
equal "the inner code repaired the tables" \
      "`grep -c 'armoured regions:' "$log"`" 1
cdto ..

step "maintenance keeps every packet wrapped"

#  Maintenance verbs inherit the stored armour level.
mkdir -p ai4 && cdto ai4
mkdir tree
mkfile tree/a.bin 300000 77
run 0 "$XPAR" create -r 10% -s 8K --armour=all -o safe -R tree
run 0 "$XPAR" addrecovery -r 20% safe.xpa
equal "addrecovery wraps its new slices" \
      "`bare_packets safe.xpa \`volume_files safe\``" 0
run 0 "$XPAR" verify safe.xpa
mkfile tree/b.bin 200000 78
run 0 "$XPAR" add safe.xpa -R tree
equal "add wraps the new generation" \
      "`bare_packets safe.xpa \`volume_files safe\` "$safe_g1_index" \
                      \`volume_files safe 1\``" 0
run 0 "$XPAR" verify safe.xpa
run 0 "$XPAR" consolidate safe.xpa -o keep
equal "consolidate wraps its output" \
      "`bare_packets keep.xpa \`volume_files keep\``" 0
run 0 "$XPAR" verify keep.xpa
v=`volume_files keep | sort | head -1`
test -n "$v" || hard_error "no recovery volume to recover"
cp "$v" orig.bin
rm -f "$v"
run 0 "$XPAR" recover --volume="`basename $v`" keep.xpa
same "$v" orig.bin
run 0 "$XPAR" verify keep.xpa
cdto ..

step "--armour=all writes the same bytes twice"

#  Determinism covers the wrapped packets as well as the group.
mkdir -p ai5 && cdto ai5
mkfile p.bin 300000 79
run 0 "$XPAR" create --reproducible -r 10% -s 8K --armour=all -o xset p.bin
mkdir keep && cp xset.xpa `volume_files xset` keep/
rm -f xset.xpa `volume_files xset`
run 0 "$XPAR" create --reproducible -r 10% -s 8K --armour=all -o xset p.bin
for f in xset.xpa `volume_files xset`; do same "$f" "keep/$f"; done
cdto ..

step "a same-name short volume is reported as relengthed, not substituted"

#  Report in-place length restoration separately from substitution.
mkdir -p rl && cdto rl
mkfile p.bin 400000 81
run 0 "$XPAR" create --layout=split --volumes=2 --armour=none -r 20% -o safe p.bin
dv=$safe_data0
test -n "$dv" || hard_error "split data volume not found"
"$DAMAGE" "$dv" extend=4096 > /dev/null || hard_error "extend failed"
capture rlout "$XPAR" repair --in-place safe.xpa
grep -q 'restored to its recorded length' "$log" ||
  grep -q 'restored to its recorded length' rlout ||
  bad "repair did not report a length restore"
run 0 "$XPAR" verify safe.xpa
cdto ..

step "a volume is unmapped before it is resized, replaced or unlinked"

#  Windows runners enforce the mapped-file restriction.
mkdir -p maplock && cdto maplock

#  A same-name volume restored to its recorded length: an ftruncate.
mkfile p.bin 400000 92
run 0 "$XPAR" create --layout=split --volumes=2 --armour=none -r 20% -o safe p.bin
"$DAMAGE" "$safe_data0" extend=4096 > /dev/null || hard_error "extend failed"
run 0 "$XPAR" repair --in-place safe.xpa
run 0 "$XPAR" verify safe.xpa

#  A renamed volume rewritten under its recorded name: a rename.
mv "$safe_data1" moved.bin
run 0 "$XPAR" repair --in-place safe.xpa
exists "$safe_data1"
run 0 "$XPAR" verify safe.xpa

#  A stale volume rewritten from replicas: a rename over a mapped volume.
mkfile q.bin 500000 93
run 0 "$XPAR" create --armour=none -r 20% -o test q.bin
off=`"$DAMAGE" test.xpa find=SETD | head -1`
test -n "$off" || hard_error "no SETD in index"
damage test.xpa "rand=$((off + 4)),1"
run 0 "$XPAR" repair --in-place test.xpa
run 0 "$XPAR" verify test.xpa
if grep -q 'replicas used' "$log"; then
  bad "the stale volume was not rewritten under a mapped-file lock"
else ok; fi

#  A second name for the same mapped volume must not keep its inode pinned.
if ln test.xpa test.v999+1.xpa 2> /dev/null; then
  off=`"$DAMAGE" test.xpa find=SETD | head -1`
  damage test.xpa "rand=$((off + 4)),1"
  run 0 "$XPAR" repair --in-place test.xpa
  run 0 "$XPAR" verify test.xpa
  rm -f test.v999+1.xpa
else
  note "this filesystem has no hard links; skipped aliased volume mapping"
fi

#  An armoured archive republished in place: a rename over the archive.
mkfile r.bin 300000 94
run 0 "$XPAR" create --layout=armoured --armour=all -r 20% -s 8K -o unit r.bin
damage unit.xpa rand=100000,64
run 0 "$XPAR" repair --in-place unit.xpa
run 0 "$XPAR" verify unit.xpa

cdto ..

step "repair regenerates recovery slices that no longer verify"

#  Re-encode recovery slices that have no replica.
mkdir -p rg && cdto rg
mkfile p.bin 500000 61
run 0 "$XPAR" create -r 20% --armour=none -o yset p.bin
mkdir keep && cp yset.xpa `volume_files yset` keep/
#  The widest volume of the ladder is the last one.
vol=
for v in `volume_files yset`; do vol=$v; done
test -n "$vol" || hard_error "no recovery volume"
off=`"$DAMAGE" "$vol" find=RCVS | head -1`
test -n "$off" || hard_error "no RCVS in $vol"
damage "$vol" "rand=$((off + 64)),128"

"$XPAR" verify yset.xpa > /dev/null 2> vw.txt
grep -q 'failed their checksum' vw.txt ||
  bad "verify does not warn about the damaged recovery slice"
grep -q 'xpar repair' vw.txt ||
  bad "the warning does not name xpar repair as the remedy"
#  The option the old warning advertised never existed.
run 4 "$XPAR" scrub --repair yset.xpa
run 1 "$XPAR" scrub --rewrite -f yset.xpa
grep -q 'does not rebuild recovery slices' "$log" ||
  bad "scrub --rewrite does not point at repair"

run 0 "$XPAR" repair --in-place yset.xpa
grep -q 'recovery slice' "$log" || bad "repair did not report regeneration"
for v in yset.xpa `volume_files yset`; do same "$v" "keep/$v"; done
"$XPAR" verify yset.xpa > /dev/null 2> va.txt
if grep -q 'protection is reduced' va.txt; then
  bad "verify still reports reduced protection after repair"
else ok; fi

#  The volume that carries the slice tables is regenerated the same way.
first=
for v in `volume_files yset`; do first=$v;  break; done
off=`"$DAMAGE" "$first" find=RCVS | head -1`
damage "$first" "rand=$((off + 64)),128"
run 0 "$XPAR" repair --in-place yset.xpa
same "$first" "keep/$first"
run 0 "$XPAR" verify yset.xpa

#  A recovery volume that is gone altogether comes back the same way.
rm -f "$vol"
run 0 "$XPAR" repair --in-place yset.xpa
same "$vol" "keep/$vol"
run 0 "$XPAR" verify yset.xpa

#  recover --volume must agree with repair, and must not clobber silently.
off=`"$DAMAGE" "$vol" find=RCVS | head -1`
damage "$vol" "rand=$((off + 64)),128"
run 4 "$XPAR" recover --volume="$vol" yset.xpa
run 0 "$XPAR" recover -f --volume="$vol" yset.xpa
same "$vol" "keep/$vol"
cdto ..

step "split repair regenerates recovery slices too"

#  The owned layouts take a different path through repair than sidecar.
mkdir -p rgs && cdto rgs
mkfile p.bin 500000 63
run 0 "$XPAR" create -r 20% --layout=split --armour=none -o yset p.bin
mkdir keep && cp yset.xpa "$yset_data0" `volume_files yset` keep/
vol=
for v in `volume_files yset`; do vol=$v; done
off=`"$DAMAGE" "$vol" find=RCVS | head -1`
test -n "$off" || hard_error "no RCVS in $vol"
damage "$vol" "rand=$((off + 64)),128"
run 0 "$XPAR" repair --in-place yset.xpa
grep -q 'recovery slice' "$log" || bad "split repair did not regenerate"
for v in yset.xpa "$yset_data0" `volume_files yset`; do
  same "$v" "keep/$v"
done
run 0 "$XPAR" verify yset.xpa
cdto ..

step "--scan supplies volumes, never the data root"

#  --scan must not change the protected data root.
mkdir -p scanroot && cdto scanroot
mkdir orig backup
mkfile orig/data.bin 400000 401
cdto orig
run 0 "$XPAR" create -r 20% -o safe data.bin
cdto ..
( cd orig && cp safe.xpa `volume_files safe` data.bin ../backup/ )
damage orig/data.bin rand=200000,64
run 1 "$XPAR" verify --scan="`pwd`/backup" orig/safe.xpa
run 0 "$XPAR" repair --scan="`pwd`/backup" --in-place orig/safe.xpa
same orig/data.bin backup/data.bin
cdto ..

step "a generation with no readable descriptor is damage, not absence"

#  Preserve generations whose descriptors are unreadable.
mkdir -p lostgen && cdto lostgen
mkdir t
mkfile t/a.bin 80000 421
cp t/a.bin t/b.bin
run 0 "$XPAR" create -r 30% --dedup=file -o safe -R t
cp t/a.bin t/c.bin
run 0 "$XPAR" add --dedup-scope=chain safe.xpa -R t
test -f "$safe_g1_index" || hard_error "add wrote no second generation"

#  This generation has no critical-group replica.
damage "$safe_g1_index" truncate=0
run 2 "$XPAR" verify safe.xpa
run 2 "$XPAR" consolidate --replace safe.xpa
exists safe.xpa
exists "$safe_g1_index"
run 2 "$XPAR" prune --generation=0 safe.xpa
"$XPAR" list safe.xpa > /dev/null 2> "$log"
grep -q descriptor "$log" || bad "list dropped the damaged generation silently"
cdto ..

step "a generation recovers from the replicas in its own volumes"

#  Recovery volumes replicate the generation's critical group.
mkdir -p replgen && cdto replgen
mkdir t
mkfile t/a.bin 80000 421
run 0 "$XPAR" create -r 30% -o safe -R t
mkfile t/c.bin 90000 99
run 0 "$XPAR" add -r 30% safe.xpa -R t
test -n "`volume_files safe 1`" || hard_error "the generation carries no recovery"
mkdir keep
cp "$safe_g1_index" keep/g1index.keep
rm -f "$safe_g1_index"
#  The generation still reads through its own replicas, and the lost index
#  volume is repairable damage that repair puts back byte for byte.
run 1 "$XPAR" verify safe.xpa
if xpar_config_defined XPAR_DOS; then missing_index=SAFE.XG1
else missing_index=safe.g001.xpa
fi
grep -q "index volume '$missing_index' is missing" "$log" && ok ||
  bad "verify did not name the lost index volume"
run 0 "$XPAR" repair --in-place safe.xpa
exists "$safe_g1_index"
same "$safe_g1_index" keep/g1index.keep
run 0 "$XPAR" verify safe.xpa
cdto ..

step "a split set's lost or renamed index volume comes back byte for byte"

#  An armoured archive is its own index, so only the sidecar and the split
#  layout can lose one on its own.
mkdir -p splitidx && cdto splitidx
mkfile l.bin 800000 61
run 0 "$XPAR" create --layout=split --volumes=3 -r 20% -o lvol l.bin
rv=`volume_files lvol | head -1`
test -n "$rv" || hard_error "the split set carries no recovery volume"
mkdir keep
cp lvol.xpa keep/lv.keep
rm -f lvol.xpa
run 1 "$XPAR" verify "$rv"
if xpar_config_defined XPAR_DOS; then missing_index=LVOL.XPA
else missing_index=lvol.xpa
fi
grep -q "index volume '$missing_index' is missing" "$log" ||
  bad "verify did not name the lost split index volume"
run 0 "$XPAR" repair --in-place "$rv"
exists lvol.xpa
same lvol.xpa keep/lv.keep
run 0 "$XPAR" verify lvol.xpa
cat lvol.d0* > joined.bin
same joined.bin l.bin

#  A renamed index volume goes back under its recorded name, and the file
#  someone else put there stays where it is.
rm -f joined.bin
mv lvol.xpa other.bin
run 1 "$XPAR" verify "$rv"
grep -qi "is missing; using 'other.bin'" "$log" ||
  bad "verify did not adopt the renamed split index volume"
run 0 "$XPAR" repair --in-place "$rv"
exists other.bin
same lvol.xpa keep/lv.keep
run 0 "$XPAR" verify lvol.xpa
cat lvol.d0* > joined.bin
same joined.bin l.bin
cdto ..

step "an unstorable manifest is refused before any output exists"

#  Reject duplicate stored names before publishing.
mkdir -p dupname && cdto dupname
mkfile a.bin 20000 411
run 4 "$XPAR" create --no-verify-after -r 30% -o code a.bin a.bin
if test -e code.xpa; then bad "create wrote an index no reader accepts"; else ok; fi
run 4 "$XPAR" create -r 30% -o cc02 a.bin a.bin
run 4 "$XPAR" create -r 30% -o cc03 a.bin a.bin
for d in .xpar-create-*; do
  test -e "$d" && bad "a failed create left the staging directory $d"
done
ok
cdto ..

step "add validates before it publishes a generation"

#  Validate additions before publishing a generation.
mkdir -p dupadd && cdto dupadd
mkdir t
mkfile t/a.bin 80000 421
run 0 "$XPAR" create -r 30% -o safe -R t
run 4 "$XPAR" add -r 30% safe.xpa t/a.bin t/a.bin
if test -e "$safe_g1_index"
then bad "add published a generation it then refused"
else ok
fi
run 0 "$XPAR" verify safe.xpa
cdto ..

step "--include restricts, and never re-admits an --exclude"

mkdir -p sel && cdto sel
mkdir -p tree/a
echo hi > tree/a/1.txt
echo secret > tree/secret.txt
run 0 "$XPAR" create -r 10% -o em02 --exclude='*secret*' --include='*.txt' -R tree
capture names "$XPAR" list em02.xpa
grep -q '1\.txt' names || bad "--include dropped a path it admitted"
if grep -q secret names
then bad "--include re-admitted a path --exclude removed"
else ok
fi
cdto ..

step "an interleave past -m is refused on every layout"

#  Every layout must reject an over-budget depth.
mkdir -p armdepth && cdto armdepth
mkfile d.bin 300000 431
run 4 "$XPAR" create -m 2M -r 20% --depth=32000 -o safe d.bin
run 4 "$XPAR" create -m 2M -r 20% --burst=100M -o sf02 d.bin
run 4 "$XPAR" create -m 2M -r 20% --depth=32000 --layout=split -o sf03 d.bin

#  Report an affordable depth.
"$XPAR" create -m 2M -r 20% --depth=32000 -o sf04 d.bin > /dev/null 2> "$log"
afford=`sed -n 's/.*affords --depth \([0-9][0-9]*\).*/\1/p' "$log" | head -1`
test -n "$afford" || hard_error "the refusal named no affordable depth"
run 0 "$XPAR" create -m 2M -r 20% --depth="$afford" -o sf05 d.bin
cdto ..

step "--burst delivers at least the tolerance it was asked for"

#  Burst tolerance must not be rounded below the request.
mkdir -p burst && cdto burst
mkfile d.bin 300000
for b in 127 255 511; do
  "$XPAR" create -f -r 20% --layout=armoured --burst=$b -o zset d.bin \
    > /dev/null 2> "$log" || bad "create --burst=$b failed"
  got=`sed -n 's/.*burst tolerance \([0-9][0-9]*\) bytes.*/\1/p' "$log" |
       head -1`
  test -n "$got" || hard_error "no burst tolerance was reported for $b"
  if test "$got" -ge "$b"
  then ok
  else bad "--burst=$b delivered only $got bytes"
  fi
done
cdto ..

step "--max-recovery widens the axis for the matrix codec too"

#  Reserve the matrix recovery axis requested by --max-recovery.
mkdir -p maxrec && cdto maxrec
mkfile d.bin 300000 1
run 0 "$XPAR" create -r 10 --max-recovery=500 --codec=matrix -o unit d.bin
run 0 "$XPAR" addrecovery -r 400 unit.xpa
run 0 "$XPAR" verify unit.xpa
cdto ..

step "--rescan=hash inherits the bytes of a metadata-only change"

mkdir -p metaonly && cdto metaonly
mkdir r
mkfile r/a.bin 90000 77
run 0 "$XPAR" create -r 30% -o rset -R r
touch r/a.bin
"$XPAR" add -r 30% --rescan=hash rset.xpa -R r > /dev/null 2> "$log" ||
  bad "add after touch failed"
grep -q '0 new stream bytes' "$log" || bad "a touch re-stored the whole file"
chmod 600 r/a.bin
"$XPAR" add -r 30% --rescan=hash rset.xpa -R r > /dev/null 2> "$log" ||
  bad "add after chmod failed"
grep -q '0 new stream bytes' "$log" || bad "a chmod re-stored the whole file"
run 0 "$XPAR" verify rset.xpa
cdto ..

step "an intact set is never unrepairable for superseded slices alone"

#  Superseded slices alone require no decoding.
mkdir -p supersede && cdto supersede
mkfile f1.bin 300000 1
mkfile f2.bin 250000 2
run 0 "$XPAR" create -r 15% -o sets f1.bin f2.bin
mkfile f2.bin 250000 22
run 0 "$XPAR" add sets.xpa f1.bin f2.bin
run 0 "$XPAR" verify sets.xpa
grep -q superseded "$log" || bad "verify did not warn about superseded slices"

#  Real damage against that same depth still exhausts the recovery.
damage f1.bin rand=1000,64
run 2 "$XPAR" verify sets.xpa
cdto ..

step "consolidate inherits the widest ratio in the chain"

#  Inherit the widest nonzero ratio in the chain.
mkdir -p ratio && cdto ratio
mkdir t
mkfile t/a.bin 80000 41
cp t/a.bin t/b.bin
run 0 "$XPAR" create -r 30% --dedup=file -o safe -R t
cp t/a.bin t/c.bin
run 0 "$XPAR" add --dedup-scope=chain safe.xpa -R t
run 0 "$XPAR" consolidate --replace safe.xpa
read_geometry safe.xpa
pct=$((100 * R / S))
if test "$pct" -ge 25
then ok
else bad "consolidate fell to R=$R of S=$S, $pct%"
fi
cdto ..

step "--base names the input or an ancestor of it, or is refused"

mkdir -p basedir && cdto basedir
mkdir -p tree/a
echo hi > tree/a/1.txt
run 0 "$XPAR" create -r 10% -o tree/emit --base="`pwd`/tree" -R tree
capture names "$XPAR" list tree/emit.xpa
if grep -q 'tree/a/1\.txt' names
then bad "--base naming the input directory did not strip it"
else ok
fi
grep -q 'a/1\.txt' names || bad "--base lost the entry altogether"
run 4 "$XPAR" create -f -r 10% -o em02 --base=/usr -R tree
cdto ..

step "a name the format cannot carry is skipped, not fatal"

mkdir -p ctrlname && cdto ctrlname
mkdir -p t2/a
echo hi > t2/a/1.txt
odd="t2/`printf 'ct\001rl'`"
if xpar_config_defined XPAR_DOS; then
  note "DOS cannot name a control byte; skipped"
elif can_hold "$odd"; then
  echo x > "$odd"
  "$XPAR" create -r 10% -o cc01 -R t2 > /dev/null 2> "$log" ||
    bad "one unstorable name aborted the whole create"
  grep -q 'skipp' "$log" || bad "the unstorable name was not reported"
  run 0 "$XPAR" verify cc01.xpa
  run 4 "$XPAR" create -f -r 10% --strict -o c01s -R t2
else
  note "this host cannot hold a control byte in a name"
fi
cdto ..

step "argument paths are normalised before they become stored names"

mkdir -p dotpath && cdto dotpath
mkdir -p t2/a t2/b
echo hi > t2/a/1.txt
for p in "t2//a/1.txt" "t2/./a/1.txt" "t2/b/../a/1.txt"; do
  run 0 "$XPAR" create -f -r 10% -o cc02 "$p"
  capture names "$XPAR" list cc02.xpa
  grep -q 't2/a/1\.txt' names || bad "'$p' was not stored as t2/a/1.txt"
done
cdto ..

step "armour parameters that would inflate the archive are refused"

#  Refuse disproportionate inner-code expansion.
mkdir -p armgrow && cdto armgrow
printf A > one.b
run 4 "$XPAR" create --layout=armoured --armour-t=32767 --armour-field=16 \
                     -r 10% -o zset one.b
if test -e zset.xpa; then bad "the refused archive was written anyway"; else ok; fi
if xpar_config_defined XPAR_DOS; then mkfile big.b 16
else mkfile big.b 300000
fi
run 4 "$XPAR" create --layout=armoured --armour-t=32767 --armour-field=16 \
                     -r 10% -o zz02 big.b

#  The refusal has to name a t the very same input accepts.
most=`sed -n 's/.*[^0-9]\([0-9][0-9]*\)[^0-9]*$/\1/p' "$log" | head -1`
test -n "$most" || hard_error "the refusal named no affordable --armour-t"
run 0 "$XPAR" create --layout=armoured --armour-t="$most" --armour-field=16 \
                     -r 10% -o ztag big.b
run 0 "$XPAR" verify ztag.xpa
run 4 "$XPAR" create -r 10% --armour-t=32767 --armour-field=8 -o zz03 big.b
cdto ..

step "an armoured create reports the one volume it wrote"

mkdir -p onevol && cdto onevol
mkfile d.bin 300000
"$XPAR" create -r 10% --layout=armoured -o zset d.bin > /dev/null 2> "$log" ||
  bad "armoured create failed"
grep -q 'in 1 volume' "$log" || bad "armoured create miscounted its volumes"
run 0 "$XPAR" verify zset.xpa
cdto ..

step "consolidate --dry-run writes nothing and needs no destination"

mkdir -p condry && cdto condry
mkdir t
mkfile t/a.bin 80000 41
run 0 "$XPAR" create -r 30% -o safe -R t
mkfile t/c.bin 40000 42
run 0 "$XPAR" add safe.xpa -R t
capture plan "$XPAR" consolidate --dry-run safe.xpa
if test "$status" -eq 0
then ok
else bad "consolidate --dry-run demanded a destination it does not use"
fi
exists "$safe_g1_index"

#  Count aliased extents once when estimating reclaimable bytes.
total=`sed -n 's/.*stream *: \([0-9][0-9]*\) bytes across.*/\1/p' plan`
live=`sed -n 's/.*bytes across the chain, \([0-9][0-9]*\) still.*/\1/p' plan`
test -n "$total" && test -n "$live" || hard_error "no dry-run stream line"
if test "$live" -le "$total"
then ok
else bad "consolidate --dry-run says $live of $total bytes are referenced"
fi
cdto ..

# Batch B regressions.

step "no verb blocks on a FIFO where a regular file belongs"

mkdir -p fifo && cdto fifo
if fifos_work probe.fifo; then
  rm -f probe.fifo
  mkfile d.bin 200000 500
  run 0 "$XPAR" create -r 20% -o back d.bin

  #  A FIFO named like the set.
  mkfifo v.xpa
  run_any "3 5" "$XPAR" verify v.xpa
  run_any "3 5" "$XPAR" info v.xpa
  run_any "3 5" "$XPAR" list v.xpa
  run_any "3 5" "$XPAR" explain v.xpa
  rm -f v.xpa

  #  A FIFO the scan directory offers as a volume.
  mkdir sc && mkfifo sc/back.v00+05.xpa
  run_any "1 2 5" "$XPAR" verify --scan="`pwd`/sc" back.xpa
  rm -rf sc

  #  A FIFO where an output volume would go.
  mkdir out && cdto out
  mkfile z.bin 40000 7
  mkfifo oset.xpa
  run 4 "$XPAR" create -r 10% -o oset z.bin
  cdto ..

  #  A FIFO where a protected file used to be.
  rm -f d.bin && mkfifo d.bin
  run_any "2 5" "$XPAR" repair --in-place back.xpa
  run_any "1 2 5" "$XPAR" verify back.xpa
  rm -f d.bin
  note "every verb refused the FIFO instead of blocking"
else
  note "this host has no usable FIFOs; skipped"
fi
cdto ..

step "verify judges by the recovery that survives, not by the layout"

mkdir -p survive && cdto survive
survive_bytes=1000000
xpar_config_defined XPAR_DOS && survive_bytes=262144
mkfile d.bin "$survive_bytes" 301
run 0 "$XPAR" create -r 20% -s 4096 -o back d.bin
victim=`volume_files back | tail -1`
vsize=`nbytes < "$victim"`
damage "$victim" "rand=200,$((vsize - 300))"
i=0
while test "$i" -lt 40; do
  damage d.bin "rand=$((i * 4096 + 100)),64"
  i=$((i + 1))
done
"$XPAR" verify --json back.xpa > v.json 2> "$log" && bad "verify called it clean"
avail=`json_num v.json recovery_available summary`
need=`json_num v.json recovery_needed summary`
if test -n "$avail" && test -n "$need" && test "$avail" -lt "$need"
then ok
else bad "verify counted $avail usable against $need needed"
fi
run 2 "$XPAR" verify back.xpa
run 2 "$XPAR" repair --in-place back.xpa
cdto ..

step "a dry run plans the names and the recovery a real run would write"

mkdir -p dryplan && cdto dryplan
mkdir tr
mkfile tr/f.bin 5000 1
: > tr/empty
mkdir tr/d
run 0 "$XPAR" create --preserve=all -r 20% -o safe -R tr
rm -f tr/empty && rmdir tr/d
run 1 "$XPAR" repair --dry-run --exit-on-change --in-place safe.xpa
grep -q 'would recreate' "$log" || bad "the dry run did not plan the names"
if test -e tr/empty; then bad "the dry run recreated a name"; else ok; fi
run 1 "$XPAR" repair --in-place --exit-on-change safe.xpa
exists tr/empty
exists tr/d

#  A rotted recovery volume is a planned action too.
mkdir -p rot && cdto rot
mkfile d.bin 300000 3
run 0 "$XPAR" create -r 20% -o safe d.bin
victim=`volume_files safe | tail -1`
before=`cat "$victim" | nbytes`
damage "$victim" "rand=2000,4096"
run 1 "$XPAR" repair --dry-run --exit-on-change --in-place safe.xpa
grep -q 'would be regenerated' "$log" ||
  bad "the dry run did not plan the recovery rewrite"
equal "the dry run wrote nothing" "`cat "$victim" | nbytes`" "$before"
run 1 "$XPAR" repair --in-place --exit-on-change safe.xpa
cdto ..

#  Owned layouts use the same recovery plan.
mkdir -p split && cdto split
mkfile d.bin 300000 5
run 0 "$XPAR" create --layout=split -r 2 -o safe d.bin
victim=`volume_files safe | tail -1`
mkdir keep && cp "$victim" keep/
rm -f "$victim"
run 0 "$XPAR" verify safe.xpa
run 1 "$XPAR" repair --dry-run --exit-on-change safe.xpa
grep -q 'would be regenerated' "$log" ||
  bad "the split dry run did not plan the recovery rewrite"
if test -e "$victim"; then bad "the dry run wrote a volume"; else ok; fi
run 0 "$XPAR" repair --dry-run safe.xpa
run 1 "$XPAR" repair --exit-on-change safe.xpa
exists "$victim"
same "$victim" "keep/$victim"
run 0 "$XPAR" verify safe.xpa
run 0 "$XPAR" repair --dry-run --exit-on-change safe.xpa
cdto ..
cdto ..

step "scrub --rewrite reports write failures"

mkdir -p rwro && cdto rwro
mkfile p.bin 100000 63
run 0 "$XPAR" create --armour=all -o arch p.bin
off=`"$DAMAGE" arch.xpa find=ARMG | head -1`
test -n "$off" || hard_error "no ARMG in arch.xpa"
damage arch.xpa "flip=$((off + 300)),6"
cp arch.xpa a.hurt
chmod 444 arch.xpa
if ( : >> arch.xpa ) 2> /dev/null; then
  note "mode 444 is writable; skipped the refused rewrite"
else
  run 5 "$XPAR" scrub --rewrite arch.xpa
  grep -q 'cannot open' "$log" || bad "the unwritable volume was not named"
  if grep -q 'refreshed' "$log"; then
    bad "a refused rewrite was still called a refresh"
  else ok; fi
  same arch.xpa a.hurt
fi
chmod 644 arch.xpa
run_any "0 1" "$XPAR" scrub --rewrite arch.xpa
grep -q 'refreshed' "$log" || bad "the rewrite did not report the refresh"
run 0 "$XPAR" scrub arch.xpa
grep -q ' 0 corrected' "$log" ||
  bad "the refreshed region still needs correction"
cdto ..

step "an interrupted addrecovery leaves a set every verb still reads"

mkdir -p halfadd && cdto halfadd
mkfile d.bin 300000 17
run 0 "$XPAR" create -r 2 -o safe d.bin
mkdir pre post
cp safe.xpa `volume_files safe` pre/
run 0 "$XPAR" addrecovery -r 4 safe.xpa
cp safe.xpa `volume_files safe` post/
new=`cd post && volume_files safe | while read f; do test -e ../pre/$f || echo $f; done`
test -n "$new" || hard_error "addrecovery added no volume"
reset_set() { rm -f safe.xpa `volume_files safe`; cp pre/* .; }

#  Before index replacement, its old layout makes replacements stale.
reset_set
( cd post && volume_files safe ) | while read b; do
  test -e "pre/$b" && cp "post/$b" "$b"
done
run 1 "$XPAR" verify safe.xpa
grep -q 'stale' "$log" || bad "stale volumes were not reported"
run 1 "$XPAR" repair --dry-run --exit-on-change --in-place safe.xpa
grep -q 'would be rewritten' "$log" ||
  bad "the dry run did not plan the stale rewrite"
run 0 "$XPAR" repair --in-place safe.xpa
grep -q 'stale volume' "$log" || bad "repair did not report the rewrite"
run 0 "$XPAR" verify safe.xpa
run 0 "$XPAR" addrecovery -r 4 safe.xpa
run 0 "$XPAR" verify safe.xpa
for f in $new; do exists "$f"; done

#  A new volume published before the index is also stale.
reset_set
for f in $new; do cp "post/$f" .; done
run 1 "$XPAR" verify safe.xpa
grep -q 'stale' "$log" || bad "the early volume was not called stale"
run 0 "$XPAR" repair --in-place safe.xpa
run 0 "$XPAR" verify safe.xpa
run 0 "$XPAR" addrecovery -r 4 safe.xpa
run 0 "$XPAR" verify safe.xpa

#  After index replacement, repair regenerates missing new volumes.
reset_set
cp post/* .
for f in $new; do rm -f "$f"; done
run 0 "$XPAR" verify safe.xpa
grep -q 'not on disk' "$log" || bad "the missing new volumes went unreported"
run 0 "$XPAR" repair --in-place safe.xpa
for f in $new; do exists "$f"; done
run 0 "$XPAR" verify safe.xpa
cdto ..

step "a nonconforming volume is reported, and repair restores it"

mkdir -p ragged && cdto ragged
mkfile d.bin 300000 21
run 0 "$XPAR" create -r 20% -o back d.bin
cp back.xpa b.keep
for op in truncate=1000 extend=4096; do
  cp b.keep back.xpa
  damage back.xpa "$op"
  run 1 "$XPAR" verify back.xpa
  grep -q 'nonconforming' "$log" || bad "$op was not reported"
  run 0 "$XPAR" repair --in-place back.xpa
  same back.xpa b.keep
  run 0 "$XPAR" verify back.xpa
done
cdto ..

step "a packet-bearing volume is found under any name and put back"

mkdir -p renamed && cdto renamed
mkfile d.bin 400000 31
run 0 "$XPAR" create -r 20% -o back d.bin
victim=`volume_files back | tail -1`
#  Keep the reference copy out of the set's own directory.
mkdir keep && cp "$victim" keep/orig
mv "$victim" zz
run 1 "$XPAR" verify back.xpa
grep -q "is missing; using 'zz'" "$log" ||
  bad "the renamed volume was not found by its header"
run 0 "$XPAR" repair --in-place back.xpa
exists "$victim"
same "$victim" keep/orig
run 0 "$XPAR" verify back.xpa
cdto ..

step "an armoured archive with no prologue says how to get it back"

mkdir -p noprol && cdto noprol
prologue_bytes=200000
test "$xpar_test_dos" != yes || prologue_bytes=40000
mkfile d.bin "$prologue_bytes" 13
run 0 "$XPAR" create --layout=armoured -r 20% -o arch d.bin
cp arch.xpa a.keep
damage arch.xpa "zero=0,384"
for v in verify repair scrub extract; do
  run 2 "$XPAR" "$v" arch.xpa
  grep -q 'recover-prologue' "$log" || bad "$v gave no hint"
done
run 2 "$XPAR" explain arch.xpa
grep -q 'prologue is gone' "$log" ||
  bad "explain did not recognise the armoured layout"
run 0 "$XPAR" recover-prologue arch.xpa
same arch.xpa a.keep
run 0 "$XPAR" verify arch.xpa
cdto ..

step "a copy that replaced a hard link is relinked or refused"

mkdir -p relink && cdto relink
mkdir t
mkfile t/a.bin 40000 5
if xpar_hardlinks_work t/a.bin t/b.bin; then
  run 0 "$XPAR" create -r 20% -o safe -R t
  rm -f t/b.bin && cp t/a.bin t/b.bin
  run 1 "$XPAR" verify safe.xpa
  run 0 "$XPAR" repair --in-place safe.xpa
  run 0 "$XPAR" verify safe.xpa

  #  A failed link preserves the copy and makes repair fail.
  rm -f t/b.bin && cp t/a.bin t/b.bin
  if perms_bite .; then
    chmod 555 t
    run 5 "$XPAR" repair --in-place safe.xpa
    grep -q 'cannot link' "$log" || bad "the refused link was not reported"
    grep -q 'failed to link' "$log" ||
      bad "the summary hid the refused link"
    if grep -q 'no damage found' "$log"; then
      bad "a refused link was reported as no damage"
    else ok; fi
    chmod 755 t
    exists t/b.bin
    same t/b.bin t/a.bin
    if ls t/b.bin.xpar-link-* > /dev/null 2>&1; then
      bad "a link stage was left behind"
    else ok; fi
    run 1 "$XPAR" verify safe.xpa
    run 0 "$XPAR" repair --in-place safe.xpa
    run 0 "$XPAR" verify safe.xpa
  else
    note "mode 555 is writable; skipped the refused link"
  fi

  #  Bytes that differ are never discarded to make the link.
  rm -f t/b.bin && mkfile t/b.bin 40000 6
  run 1 "$XPAR" verify safe.xpa
  run 2 "$XPAR" repair --in-place safe.xpa
  exists t/b.bin
else
  note "this filesystem has no hard links; skipped"
fi
cdto ..

step "a chain repair states one verdict, not one per generation"

mkdir -p chainmsg && cdto chainmsg
mkdir t
mkfile t/a.bin 40000 7
if xpar_hardlinks_work t/a.bin t/c.bin; then
  run 0 "$XPAR" create -r 20% -o safe -R t
  mkfile t/a.bin 50000 8
  rm -f t/c.bin && mkfile t/c.bin 40000 7
  run 0 "$XPAR" add -r 20% safe.xpa -R t
  attempt "$XPAR" repair --in-place safe.xpa
  if test "$status" -ne 0 && grep -q 'no damage found' "$log"
  then bad "a failing chain repair still said it found no damage"
  else ok
  fi
else
  note "this filesystem has no hard links; skipped"
fi
cdto ..

step "a file the host will not open is an I/O error, not an absence"

mkdir -p noread && cdto noread
mkfile d.bin 40000 3
run 0 "$XPAR" create -r 20% -o safe d.bin
if chmod 000 d.bin 2> /dev/null && ! ( : < d.bin ) 2> /dev/null; then
  run 5 "$XPAR" verify safe.xpa
  grep -q 'read failed' "$log" || bad "an unreadable file was called missing"
  chmod 644 d.bin
else
  note "this user can read mode 000 files; skipped"
  chmod 644 d.bin 2> /dev/null
fi
cdto ..

step "a write past the file-size limit ends the JSON stream properly"

mkdir -p fsz && cdto fsz
#  Large enough that a volume passes the limit whichever block size the
#  shell's ulimit counts in.
mkfile d.bin 4000000 9
if xpar_config_defined XPAR_DOS; then
  note "skipped because DOS cannot inherit file-size limits"
elif ( ulimit -f 100 ) 2> /dev/null; then
  status=0
  ( ulimit -f 100; "$XPAR" create --json -r 20% -o safe d.bin ) > out.json \
      2> "$log" || status=$?
  if test "$status" -eq 5; then ok
  else bad "a size-limited write exited $status, not 5"
  fi
  equal "the JSON stream ended with a summary" \
        "`json_str out.json status summary`" error
else
  note "this shell cannot set a file-size limit; skipped"
fi
cdto ..

step "extract applies every class --preserve=all names"

mkdir -p setid && cdto setid
mkdir tr
mkfile tr/f1 5000 441
if modes_work . && chmod 4755 tr/f1 2> /dev/null &&
   test "`mode_of tr/f1`" = 4755; then
  run 0 "$XPAR" create --preserve=all -r 10% --layout=split -o pvol -R tr
  run 0 "$XPAR" extract -f --preserve=all --to=b pvol.xpa
  equal "--preserve=all kept the set-ID bits" "`mode_of b/tr/f1`" 4755
  #  --require makes a class it could not apply fatal.
  run 5 "$XPAR" extract -f --preserve=mode --require=setid --to=c pvol.xpa
else
  note "this filesystem does not keep set-ID bits; skipped"
fi
cdto ..

step "the documented exit code is the one every verb returns"

mkdir -p codes && cdto codes
mkfile d.bin 40000 3
run 0 "$XPAR" create -r 20% -o safe d.bin
mkdir bare
run 3 "$XPAR" verify bare
run 3 "$XPAR" nosuch.xpa
run 4 "$XPAR" verfy
run 4 "$XPAR" addrecovery -r 100000 safe.xpa
run 6 "$XPAR" verify --auth-key=absent.key safe.xpa
cdto ..

step "--json keeps stdout for the machine and stderr for the reader"

mkdir -p jsonio && cdto jsonio
mkdir t
mkfile t/a.bin 40000 3
run 0 "$XPAR" create --layout=split -r 20% -o safe -R t
for v in list info explain; do
  "$XPAR" "$v" --json safe.xpa > out.json 2> "$log" ||
    bad "$v --json failed"
  if test -s out.json && test -s "$log"; then ok
  else bad "$v --json dropped one of its two streams"; fi
done
#  A name that is not UTF-8 still reaches a consumer intact.
raw=`printf 't/bad\377name'`
if xpar_config_defined XPAR_DOS; then
  note "DOS names cannot contain byte 0xFF; skipped"
elif can_hold "$raw"; then
  printf 'x' > "$raw"
  run 0 "$XPAR" create --layout=split -r 20% -o sf02 -R t
  "$XPAR" list --json sf02.xpa > out.json 2> "$log" || bad "list --json failed"
  grep -q '"name_hex":' out.json ||
    bad "a name that is not UTF-8 was emitted without its bytes"
  rm -f "$raw"
else
  note "this filesystem refuses the byte 0xFF in a name; skipped"
fi
cdto ..

step "a separate destination reports its repairs and writes what it can"

mkdir -p tosave && cdto tosave
mkdir t
mkfile t/a.bin 200000 3
mkfile t/b.bin 200000 4
cp -R t t.orig
run 0 "$XPAR" create -r 30% -s 4096 -o safe -R t
damage t/a.bin "rand=64,5000"
run 0 "$XPAR" repair --to=out safe.xpa
grep -q '1 entry repaired' "$log" ||
  bad "--to reported no entries repaired"
same out/t/a.bin t.orig/a.bin

#  Past the recovery, the entries that survive are still written out.
cp t.orig/a.bin t/a.bin
i=0
while test "$i" -lt 41; do
  damage t/a.bin "rand=200,$((i * 4096 + 50))"
  i=$((i + 1))
done
run 2 "$XPAR" repair --to=out2 safe.xpa
same out2/t/b.bin t.orig/b.bin
cdto ..

step "recover --to creates the directory it was given"

mkdir -p recdir && cdto recdir
mkfile d.bin 300000 11
run 0 "$XPAR" create -r 20% -o lvol d.bin
run 0 "$XPAR" recover --volume=0 --to=fresh lvol.xpa
if xpar_config_defined XPAR_DOS; then exists fresh/lvol.xpa
else exists fresh/lvol.xpa
fi
cdto ..

step "a generation whose data volume landed before its index still reads"

#  An interrupted add may publish data before its index.
mkdir -p splitadd && cdto splitadd
mkfile a.bin 200000 21
mkfile b.bin 100000 22
run 0 "$XPAR" create -q --layout=split -r 20% -o safe a.bin
run 0 "$XPAR" add -q -r 20% safe.xpa b.bin
exists "$safe_g1_data"
exists "$safe_g1_index"

#  Hand-build the interrupted publish: everything but the newest index.
mkdir hb
for f in safe.xpa "$safe_data0" "$safe_g1_data" \
         `volume_files safe` `volume_files safe 1`; do cp "$f" hb/; done
cdto hb
run 1 "$XPAR" verify safe.xpa
if xpar_config_defined XPAR_DOS; then missing_index=SAFE.XG1
else missing_index=safe.g001.xpa
fi
grep -q "$missing_index' is missing" "$log" ||
  bad "the missing newest index was not reported"
run 0 "$XPAR" repair --in-place safe.xpa
exists "$safe_g1_index"
run 0 "$XPAR" verify safe.xpa
rm -rf out
run 0 "$XPAR" extract --to=out safe.xpa
same out/a.bin ../a.bin
same out/b.bin ../b.bin
cdto ..
cdto ..

step "a refused pipe publish keeps the piped bytes"

#  A failed publish must preserve staged pipe input.
mkdir -p pipefail && cdto pipefail
if perms_bite .; then
  mkfile src.bin 300000 57
  mkdir -p out/sub
  chmod 555 out/sub
  run 5 "$XPAR" create -r 20% --spool --stdin-name=sub/data.bin -o out/qset \
      - < src.bin
  grep -q "set remains in" "$log" ||
    bad "the refusal did not say where the set is"
  if xpar_config_defined XPAR_DOS; then
    stage=`ls -d out/XCR* 2> /dev/null | head -1`
    stdin_name=STDIN.DAT
    index_name=QSET.XPA
  else
    stage=`ls -d out/.xpar-create-* 2> /dev/null | head -1`
    stdin_name=.stdin-data
    index_name=qset.xpa
  fi
  if test -n "$stage"; then
    exists "$stage/$stdin_name"
    same "$stage/$stdin_name" src.bin
    exists "$stage/$index_name"
  else
    bad "the staging directory named in the refusal is gone"
  fi
  #  Nothing reached a final name.
  test -e out/qset.xpa && bad "an index was published by a refused create"
  test -e out/sub/data.bin && bad "the pipe input was published"
  chmod 755 out/sub
else
  note "mode 555 is writable; skipping the refused-publish test"
fi
cdto ..

step "recover-prologue republishes the archive whole"

#  Publish all three prologue copies atomically.
mkdir -p prol2 && cdto prol2
prologue_bytes=200000
test "$xpar_test_dos" != yes || prologue_bytes=40000
mkfile d.bin "$prologue_bytes" 61
run 0 "$XPAR" create --layout=armoured -r 20% -o arch d.bin
cp arch.xpa a.keep
damage arch.xpa "zero=0,384"
run 0 "$XPAR" recover-prologue arch.xpa
same arch.xpa a.keep
run 0 "$XPAR" verify arch.xpa
equal "no staging or rollback residue" \
      "`ls arch.xpa.xpar-tmp-* arch.xpa.xpar-old-* 2> /dev/null | nlines`" 0
cdto ..

step "undo replays every generation's journal of a chain repair"

#  A chain repair creates one journal per affected generation.
mkdir -p chundo && cdto chundo
mkdir t
mkfile t/a.bin 120000 11
mkfile t/b.bin 80000 12
run 0 "$XPAR" create -R -r 30% -o safe t
mkfile extra.bin 30000 14
cat extra.bin >> t/a.bin
run 0 "$XPAR" add -r 30% safe.xpa -R t
cp t/a.bin pa.bin;  cp t/b.bin pb.bin
damage t/a.bin "rand=130000,64"
damage t/b.bin "rand=4096,64"
cp t/a.bin da.bin;  cp t/b.bin db.bin
run 0 "$XPAR" repair --in-place --keep-journal safe.xpa
same t/a.bin pa.bin
same t/b.bin pb.bin
exists safe.xparundo
exists "$safe_g1_journal"
run 0 "$XPAR" undo safe.xpa
same t/a.bin da.bin
same t/b.bin db.bin
equal "both journals were replayed and dropped" \
      "`journal_files | nlines`" 0

#  An interrupted chain repair leaves only the older generation's journal.
run 0 "$XPAR" repair --in-place --keep-journal safe.xpa
rm -f "$safe_g1_journal"
run 0 "$XPAR" undo safe.xpa
same t/b.bin db.bin
same t/a.bin pa.bin
equal "the surviving journal was replayed" \
      "`journal_files | nlines`" 0

#  --generation still replays exactly one.
cp da.bin t/a.bin
run 0 "$XPAR" repair --in-place --keep-journal safe.xpa
run 0 "$XPAR" undo --generation=1 safe.xpa
same t/a.bin da.bin
same t/b.bin pb.bin
equal "only the selected generation's journal was replayed" \
      "`journal_files | nlines`" 1
run 0 "$XPAR" undo safe.xpa
same t/b.bin db.bin
cdto ..

step "a journal that cannot be read is an I/O error, not an absent one"

#  An unreadable journal is an I/O error, not an absent journal.
mkdir -p junread && cdto junread
mkfile d.bin 200000 21
run 0 "$XPAR" create -r 50% -o safe d.bin
damage d.bin "rand=4096,64"
run 0 "$XPAR" repair --in-place --keep-journal safe.xpa
exists safe.xparundo
if chmod 000 safe.xparundo 2> /dev/null &&
   ! cat safe.xparundo > /dev/null 2>&1; then
  run 5 "$XPAR" undo safe.xpa
  if xpar_config_defined XPAR_DOS; then journal_name=SAFE.XPU
  else journal_name=safe.xparundo
  fi
  grep -q "$journal_name" "$log" || bad "the I/O error did not name the journal"
  chmod 600 safe.xparundo
else
  note "this host reads a mode 000 file; skipping the unreadable journal"
  chmod 600 safe.xparundo 2> /dev/null
fi
run 0 "$XPAR" undo safe.xpa
run 3 "$XPAR" undo safe.xpa
grep -qi "nothing to undo" "$log" ||
  bad "an absent journal was not reported as absent"
cdto ..

step "extract names the destination it could not publish"

#  A destination without a usable backup name is reported and uncounted.
mkdir -p exname && cdto exname
long=`awk 'BEGIN { s = "";  while (length(s) < 244) s = s "n";  print s }'`
mkdir t
mkdir -p out/t
if xpar_config_defined XPAR_DOS; then
  long_ok=no
elif can_hold "t/$long.bin" && can_hold "out/t/$long.bin"; then
  long_ok=yes
else
  long_ok=no
fi
if test "$long_ok" = yes; then
  mkfile "t/$long.bin" 20000 31
  mkfile t/other.bin 5000 32
  run 0 "$XPAR" create -R --layout=split -r 20% -o safe t
  : > "out/t/$long.bin"
  if ( : > "out/t/$long.bin.xpar-old-000" ) 2> /dev/null; then
    rm -f "out/t/$long.bin.xpar-old-000"
    note "this filesystem allows the longer backup name; skipping"
  else
    run 5 "$XPAR" extract -f --to=out safe.xpa
    grep -q "$long" "$log" || bad "the refusal did not name the file"
    grep -q "1 entry, 5000 bytes" "$log" ||
      bad "an entry that never reached its final name was counted"
  fi
else
  note "248-byte components unsupported at this test path; skipping"
fi
cdto ..

step "extract --stdout folds exit codes and reports like a directory extract"

#  --stdout honors --require, exit-code folding, and summaries.
mkdir -p exso && cdto exso
mkfile d.bin 60000 41
run 0 "$XPAR" create --layout=split --preserve=all -r 20% -o safe d.bin
capture out.bin "$XPAR" extract --stdout safe.xpa
equal "the stream extracted cleanly" "$status" 0
same out.bin d.bin
grep -q "1 entry, 60000 bytes" "$log" || bad "--stdout printed no summary"
capture out2.bin "$XPAR" extract --stdout --require=mode safe.xpa
equal "--require is fatal under --stdout" "$status" 5
grep -q "require" "$log" || bad "--require said nothing under --stdout"
same out2.bin d.bin
cdto ..

step "an input the host will not read is an error, not an unchanged entry"

#  Read failures are neither unchanged data nor forced mismatches.
mkdir -p unread && cdto unread
mkfile a.bin 100000 51
mkfile d.bin 50000 52
run 0 "$XPAR" create -r 20% -o safe a.bin d.bin
mkfile new.bin 30000 53
: > probe.bin
if chmod 000 probe.bin 2> /dev/null && ! cat probe.bin > /dev/null 2>&1; then
  rm -f probe.bin
  chmod 000 d.bin
  run 5 "$XPAR" add --rescan=hash -r 20% safe.xpa new.bin
  grep -q "d.bin" "$log" || bad "the refusal did not name the input"
  equal "nothing was published" \
        "`test -e "$safe_g1_index" && echo 1 || echo 0`" 0

  #  consolidate must not call it damage, with or without -f.
  chmod 600 d.bin
  run 0 "$XPAR" add -r 20% safe.xpa new.bin
  chmod 000 a.bin
  run 5 "$XPAR" consolidate --replace safe.xpa
  grep -q "a.bin" "$log" || bad "consolidate did not name the input"
  run 5 "$XPAR" consolidate --replace -f safe.xpa
  grep -q "could not be read" "$log" ||
    bad "--force treated a host refusal as damage to bake in"
  exists "$safe_g1_index"
  chmod 600 a.bin
else
  rm -f probe.bin
  note "this host reads a mode 000 file; skipping the unreadable-input test"
fi
cdto ..

step "an interrupted consolidate or prune heals from its maintenance journal"

#  Maintenance journals recover the window before a new index is published.
mpreload=
fault_shim "$work/faultio-maint.so" && mpreload=$fault_pre

if test -z "$mpreload"; then
  note "the fault shim cannot be preloaded here; skipping the crash walk"
else
  crash_at() {   # crash_at <rename ordinal> <cmd>...
    _k=$1;  shift
    status=0
    env XPAR_FI_PATH="$work" XPAR_FI_CRASH_RENAME="$_k" \
        LD_PRELOAD="$mpreload" "$@" > "$log" 2>&1 || status=$?
  }
  residue() { ls *.xparmaint *.xpar-old-[0-9]* *.xpar-prune-old-[0-9]* \
                 2> /dev/null | nlines; }
  tree_matches() {
    rm -rf out
    run 0 "$XPAR" extract --to=out safe.xpa
    for _f in $files; do same "out/$_f" "keep/`basename $_f`"; done
    rm -rf out
  }

  #  Verify recovery from a missing canonical set.
  heals() {   # heals <what>
    run 1 "$XPAR" verify safe.xpa
    grep -qi 'interrupted' "$log" ||
      bad "$1: verify did not report the interrupted operation"
    grep -q 'xparmaint' "$log" || bad "$1: verify did not name the journal"
    grep -q 'repair' "$log" || bad "$1: verify did not point at repair"
    run 0 "$XPAR" repair --in-place safe.xpa
    run 0 "$XPAR" verify safe.xpa
    tree_matches
    equal "$1: no journal or rollback name is left" "`residue`" 0
  }

  mkdir -p maintc && cdto maintc
  files="t/a.bin t/b.bin t/c.bin"
  mkdir tmpl && cdto tmpl
  mkdir t
  mkfile t/a.bin 120000 71
  mkfile t/b.bin 50000 72
  run 0 "$XPAR" create -R --layout=split -r 20% -o safe t
  mkfile t/c.bin 40000 73
  run 0 "$XPAR" add -r 20% safe.xpa -R t
  mkdir keep && cp t/a.bin t/b.bin t/c.bin keep/
  cdto ..

  #  A damaged journal is still evidence of an interrupted operation.  It
  #  must win discovery over the misleading missing-set fallback and remain
  #  available for operator recovery.
  k=1
  while test "$k" -le 30; do
    rm -rf corrupt && cp -R tmpl corrupt && cdto corrupt
    crash_at "$k" "$XPAR" consolidate --replace safe.xpa
    if test -f safe.xparmaint; then break; fi
    cdto ..
    k=$((k + 1))
  done
  equal "a rename fault reached the journalled window" \
        "`test -f safe.xparmaint && echo yes`" yes
  equal "the injected consolidate stopped" "$status" 97
  damage safe.xparmaint flip=0,1
  run 2 "$XPAR" repair --in-place safe.xpa
  grep -q 'cannot validate pending maintenance journal' "$log" ||
    bad "repair did not identify the corrupt maintenance journal"
  exists safe.xparmaint
  cdto ..

  #  Scan crash points inside the missing-set window.
  hits=0
  k=1
  while test "$hits" -lt 4 && test "$k" -le 30; do
    rm -rf run && cp -R tmpl run && cdto run
    crash_at "$k" "$XPAR" consolidate --replace safe.xpa
    if test -f safe.xparmaint && test ! -f safe.xpa; then
      hits=$((hits + 1))
      heals "consolidate crashed at rename $k"
      #  Retry the interrupted operation.
      run_any "0 4" "$XPAR" consolidate --replace safe.xpa
      run 0 "$XPAR" verify safe.xpa
    elif test -f safe.xparmaint; then
      #  The journal exists before any rename.
      run 0 "$XPAR" repair --in-place safe.xpa
      run 0 "$XPAR" verify safe.xpa
      equal "rename $k: no journal is left" "`residue`" 0
    fi
    cdto ..
    k=$((k + 1))
  done
  if test "$hits" -lt 4; then
    bad "only $hits consolidate crashes landed in the rename window"
  else ok; fi
  cdto ..

  #  Repeat the crash walk for prune.
  mkdir -p maintp && cdto maintp
  files="a.bin b.bin"
  mkdir tmpl && cdto tmpl
  mkfile a.bin 100000 74
  mkfile b.bin 40000 75
  run 0 "$XPAR" create --layout=split -r 20% -o safe a.bin b.bin
  mkfile a.bin 100000 76
  mkfile b.bin 40000 77
  run 0 "$XPAR" add -r 20% safe.xpa a.bin b.bin
  mkfile a.bin 100000 78
  mkfile b.bin 40000 79
  run 0 "$XPAR" add -r 20% safe.xpa a.bin b.bin
  mkdir keep && cp a.bin b.bin keep/
  cdto ..

  hits=0
  k=1
  while test "$hits" -lt 4 && test "$k" -le 30; do
    rm -rf run && cp -R tmpl run && cdto run
    crash_at "$k" "$XPAR" prune --before=1 safe.xpa
    if test -f safe.xparmaint && test ! -f safe.xpa; then
      hits=$((hits + 1))
      heals "prune crashed at rename $k"
    elif test -f safe.xparmaint; then
      run 0 "$XPAR" repair --in-place safe.xpa
      run 0 "$XPAR" verify safe.xpa
      equal "rename $k: no journal is left" "`residue`" 0
    fi
    cdto ..
    k=$((k + 1))
  done
  if test "$hits" -lt 4; then
    bad "only $hits prune crashes landed in the rename window"
  else ok; fi
  cdto ..

  #  No set or journal remains a not-found case.
  mkdir -p maintn && cdto maintn
  run 3 "$XPAR" verify nothing.xpa
  cdto ..
fi

summary
