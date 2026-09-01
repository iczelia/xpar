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

#  Host failures must leave a readable set that a rerun can complete.

. "${srcdir:-.}/lib.sh"

case `xpar_host` in
  *djgpp* | *msdos*) skip_all "LD_PRELOAD cannot instrument a DOS binary" ;;
esac

case `uname -s 2> /dev/null` in
  Linux) ;;
  *) skip_all "the fault shim needs Linux and LD_PRELOAD" ;;
esac
fault_shim "$work/faultio.so" ||
  skip_all "the fault shim cannot intercept this binary"
preload=$fault_pre

ino() { ls -i "$1" | awk '{print $1}'; }

#  inject <VAR=N>... -- <cmd>...: run under the shim with the given faults.
inject() {
  _env=
  while test "$1" != --; do _env="$_env $1";  shift; done
  shift
  env XPAR_FI_PATH="$work" LD_PRELOAD="$preload" $_env "$@"
}

step "a repair that stops part way is undone by its journal, then completes"

mkdir -p rp && cdto rp
mkdir t
mkfile t/a.bin 200000 1
mkfile t/b.bin 60000 2
: > t/empty
if ln t/a.bin t/link 2> /dev/null; then links=yes; else links=no; fi
run 0 "$XPAR" create -R -r 30% -o s t
mkdir keep && cp t/a.bin keep/a.bin
hurt() {
  cp keep/a.bin t/a.bin
  damage t/a.bin rand=4096,64 rand=40000,64
  cp t/a.bin dmg.bin
  rm -f t/empty s.xparundo
  if test $links = yes; then rm -f t/link && cp t/a.bin t/link; fi
}
healed() {
  same t/a.bin keep/a.bin
  exists t/empty
  run 0 "$XPAR" verify s.xpa
}

#  The data write itself fails: the journal restores the bytes it saved.
hurt
run 5 inject XPAR_FI_PWRITE=1 -- "$XPAR" repair --in-place s.xpa
exists s.xparundo
run 0 "$XPAR" undo s.xpa
same t/a.bin dmg.bin
run 0 "$XPAR" repair --in-place s.xpa
healed

#  Power loss right after the data write: the journal is complete.
hurt
run 97 inject XPAR_FI_CRASH_PWRITE=1 -- "$XPAR" repair --in-place s.xpa
exists s.xparundo
run 0 "$XPAR" undo s.xpa
same t/a.bin dmg.bin
run 0 "$XPAR" repair --in-place s.xpa
healed

#  A torn journal cannot prove whether protected writes began.  Keep it and
#  require the operator to replace it explicitly.
hurt
run 97 inject XPAR_FI_CRASH_WRITE=1 -- "$XPAR" repair --in-place s.xpa
same t/a.bin dmg.bin
run 3 "$XPAR" undo s.xpa
grep -q 'not an xpar repair journal\|incomplete or corrupt' "$log" ||
  bad "a torn journal was not reported"
exists s.xparundo
run 4 "$XPAR" repair --in-place s.xpa
exists s.xparundo
run 0 "$XPAR" repair --in-place --replace-journal s.xpa
healed

#  An undeletable spent journal must not replay or block repair.
hurt
run_any "0 5" inject XPAR_FI_UNLINK=1 -- "$XPAR" repair --in-place s.xpa
same t/a.bin keep/a.bin
run 0 "$XPAR" repair --in-place s.xpa
if test -s s.xparundo; then bad "a spent journal was left live"; else ok; fi

#  If relinking fails, repair the copy and retry the link next run.
if test $links = yes; then
  hurt
  run 5 inject XPAR_FI_LINK=1 -- "$XPAR" repair --in-place s.xpa
  grep -q 'cannot link' "$log" || bad "the failed link was not reported"
  exists t/link
  same t/link keep/a.bin
  run 0 "$XPAR" repair --in-place s.xpa
  equal "the copy became the link" "`ino t/link`" "`ino t/a.bin`"
  healed
else
  note "this filesystem has no hard links; skipped the link case"
fi

#  A data file the host cannot read is an I/O error, never damage.
cp keep/a.bin t/a.bin
: > t/empty
run 5 inject XPAR_FI_STICKY=1 XPAR_FI_PREAD=1 -- "$XPAR" verify s.xpa
run 5 inject XPAR_FI_STICKY=1 XPAR_FI_PREAD=1 -- "$XPAR" repair --in-place s.xpa
same t/a.bin keep/a.bin
if test -s s.xparundo; then bad "an I/O error left a live journal"; else ok; fi
cdto ..

step "an addrecovery that fails part way changes nothing"

mkdir -p ar && cdto ar
mkfile d.bin 300000 3
run 0 "$XPAR" create -r 2 -o s d.bin
mkdir before && cp s.xpa s.v*.xpa before/
run 5 inject XPAR_FI_RENAME=2 -- "$XPAR" addrecovery -r 4 s.xpa
for f in before/*; do same "$f" "`basename "$f"`"; done
equal "no new volume appeared" "`ls s.v*.xpa | wc -l | tr -d ' '`" 2
run 0 "$XPAR" verify s.xpa
#  Repair and rerun converge after a crash between renames.
for k in 1 2 3; do
  run 97 inject XPAR_FI_CRASH_RENAME=$k -- "$XPAR" addrecovery -r 4 s.xpa
  run_any "0 1" "$XPAR" verify s.xpa
  run 0 "$XPAR" repair --in-place s.xpa
  run 0 "$XPAR" addrecovery -r 4 s.xpa
  run 0 "$XPAR" verify s.xpa
  equal "four recovery volumes after crash point $k" \
        "`ls s.v*.xpa | wc -l | tr -d ' '`" 4
  rm -f s.xpa s.v*.xpa && cp before/* .
done
cdto ..

step "scrub --rewrite and create report the host failure they hit"

mkdir -p sc && cdto sc
mkfile p.bin 100000 5
run 0 "$XPAR" create --armour=all -r 20% -o a p.bin
off=`"$DAMAGE" a.xpa find=ARMG | head -1`
damage a.xpa "flip=$((off + 300)),6"
run 5 inject XPAR_FI_OPENW=1 -- "$XPAR" scrub --rewrite a.xpa
grep -q 'cannot open' "$log" || bad "the unwritable volume was not named"
cdto ..

mkdir -p cr && cdto cr
mkdir t
mkfile t/a.bin 150000 7
: > t/empty
n=`inject XPAR_FI_TRACE=1 -- "$XPAR" create -q -R -r 20% -o s t 2>&1 |
     grep -c '^FI RENAME'`
rm -f s.xpa s.v*.xpa
test "$n" -gt 0 || hard_error "create renamed nothing"
run 5 inject XPAR_FI_RENAME=$n -- "$XPAR" create -R -r 20% -o s t
if test -e s.xpa; then bad "a half-published set stands under its name"; else ok; fi
run 0 "$XPAR" create -f -R -r 20% -o s t
run 0 "$XPAR" verify s.xpa
cdto ..

step "a volume that fails under its mapping is an I/O error, not a defect"

mkdir -p bus && cdto bus
mkfile d.bin 300000 9
run 0 "$XPAR" create -r 20% -o s d.bin
mkdir keep && cp s.xpa s.v*.xpa keep/
#  Truncating a mapped file simulates a mapped-read fault.
attempt inject XPAR_FI_SHORT_MAP=s.v -- "$XPAR" verify s.xpa
if grep -q 'SHORT_MAP truncated' "$log"; then
  if test "$status" -eq 5; then ok
  else bad "a fault under a mapping exited $status, not 5"
  fi
  grep -q 'mapped read failed' "$log" ||
    bad "the failing file was not named"
  if grep -q 'report this bug' "$log"; then
    bad "a media error was reported as a defect"
  else ok
  fi
else
  note "the mapping was not truncated here; skipped"
fi
cp keep/* .
run 0 "$XPAR" verify s.xpa
cdto ..

summary
