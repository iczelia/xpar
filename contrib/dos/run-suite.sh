#!/bin/sh
# Run the DJGPP suite in one FreeDOS guest.

set -e

prog=${0##*/}
fail() { echo "$prog: $*" >&2; exit 1; }

dos_build=${1:?usage: run-suite.sh DOS-BUILD FREEDOS-ROOT [WORKDIR]}
freedos_root=${2:?FreeDOS root required}
suite_root=${3:-./dos-suite}

script_dir=`cd "$(dirname "$0")" && pwd`
top_srcdir=`cd "$script_dir/../.." && pwd`
dos_build=`cd "$dos_build" && pwd`
freedos_root=`cd "$freedos_root" && pwd`
mkdir -p "$suite_root"
suite_root=`cd "$suite_root" && pwd`

for tool in mcopy mformat mmd mpartition qemu-system-i386 timeout truncate; do
  command -v "$tool" > /dev/null || fail "$tool is required"
done
for file in xpar.exe tests/t_unit.exe tests/t_codec.exe \
            tests/t_central.exe tests/mkdata.exe tests/damage.exe \
            tests/forge.exe bench/timeit.exe config.h; do
  test -f "$dos_build/$file" || fail "$dos_build/$file not found"
done
for file in freedos.img bin/BASH.EXE bin/CWSDPMI.EXE bin/SYNC.EXE; do
  test -f "$freedos_root/$file" || fail "$freedos_root/$file not found"
done

: "${CC_FOR_DOS:=i586-pc-msdosdjgpp-gcc}"
command -v "$CC_FOR_DOS" > /dev/null || fail "$CC_FOR_DOS is required"

case ${XPAR_TEST_LEVEL:-quick} in
  quick|full|torture) test_level=${XPAR_TEST_LEVEL:-quick} ;;
  *) fail "invalid XPAR_TEST_LEVEL" ;;
esac
test_seed=${XPAR_TEST_SEED:-20260823}
case $test_seed in ''|*[!0-9]*) fail "invalid XPAR_TEST_SEED" ;; esac

boot=$suite_root/boot.img
disk=$suite_root/tests.img
mtools=$suite_root/mtools.conf
rm -f "$boot" "$disk" "$suite_root/result.log" "$suite_root/status.txt"
cp "$freedos_root/freedos.img" "$boot"
truncate -s 528482304 "$disk"

printf 'drive d: file="%s" partition=1\n' "$disk" > "$mtools"
MTOOLSRC=$mtools
MTOOLS_NO_VFAT=1
export MTOOLSRC MTOOLS_NO_VFAT
mpartition -I d:
mpartition -c -s 63 -h 16 -t 1023 d:
mformat -v XPARTEST d:
mmd d:/BIN d:/BLD d:/SRC d:/SRC/TESTS d:/SRC/BENCH \
    d:/SRC/CONTRIB d:/TMP

mcopy "$freedos_root"/bin/*.EXE d:/BIN/
"$CC_FOR_DOS" -Os -march=i386 -mtune=i386 \
  -o "$suite_root/EXIT.EXE" "$script_dir/qemu-exit.c"
mcopy "$suite_root/EXIT.EXE" d:/BIN/EXIT.EXE

mcopy "$dos_build/xpar.exe" d:/BLD/XPAR.EXE
mcopy "$dos_build/config.h" d:/BLD/CONFIG.H
mcopy "$dos_build/tests/t_unit.exe" d:/BLD/TUNIT.EXE
mcopy "$dos_build/tests/t_codec.exe" d:/BLD/TCODEC.EXE
mcopy "$dos_build/tests/t_central.exe" d:/BLD/TCENTRAL.EXE
mcopy "$dos_build/tests/mkdata.exe" d:/BLD/MKDATA.EXE
mcopy "$dos_build/tests/damage.exe" d:/BLD/DAMAGE.EXE
mcopy "$dos_build/tests/forge.exe" d:/BLD/FORGE.EXE
mcopy "$dos_build/bench/timeit.exe" d:/BLD/TIMEIT.EXE

for pair in \
  'lib.sh LIB.SH' 'central.sh CENTRAL.SH' 'faults.sh FAULTS.SH' \
  'safety.sh SAFETY.SH' 'sanity.sh SANITY.SH' 'build.sh BUILD.SH' \
  'regress.sh REGRESS.SH' 'perf.sh PERF.SH' 'bench.sh BENCH.SH' \
  'hostfaults.sh HOSTFLT.SH'; do
  set -- $pair
  mcopy "$top_srcdir/tests/$1" "d:/SRC/TESTS/$2"
done
mcopy "$top_srcdir/bench/lib.sh" d:/SRC/BENCH/LIB.SH
mcopy "$top_srcdir/bench/run.sh" d:/SRC/BENCH/RUN.SH
mcopy "$top_srcdir/tests/mkdata.c" d:/SRC/TESTS/MKDATA.C
mcopy "$top_srcdir/tests/damage.c" d:/SRC/TESTS/DAMAGE.C
mcopy "$top_srcdir/bench/timeit.c" d:/SRC/BENCH/TIMEIT.C
mcopy "$top_srcdir/contrib/ci-check.sh" d:/SRC/CONTRIB/CICHK.SH

if test -n "${XPAR_COMPAT:-}" && test -d "$XPAR_COMPAT"; then
  mmd d:/COMPAT
  for file in "$XPAR_COMPAT"/*; do
    test -e "$file" || continue
    mcopy -s "$file" d:/COMPAT/
  done
fi

tests=${XPAR_DOS_TESTS:-'CENTRAL.SH FAULTS.SH SAFETY.SH SANITY.SH BUILD.SH REGRESS.SH PERF.SH BENCH.SH HOSTFLT.SH'}
: > "$suite_root/TESTS.LST"
for test_script in $tests; do
  case $test_script in
    CENTRAL.SH|FAULTS.SH|SAFETY.SH|SANITY.SH|BUILD.SH|REGRESS.SH|PERF.SH|BENCH.SH|HOSTFLT.SH) ;;
    *) fail "unknown DOS test $test_script" ;;
  esac
  printf '%s\n' "$test_script" >> "$suite_root/TESTS.LST"
done
mcopy "$suite_root/TESTS.LST" d:/TESTS.LST

sed -e "s/@LEVEL@/$test_level/g" -e "s/@SEED@/$test_seed/g" \
  "$script_dir/guest-suite.sh" > "$suite_root/RUN.SH"
mcopy "$suite_root/RUN.SH" d:/RUN.SH
mcopy -o -i "$boot" "$script_dir/fdauto.bat" ::FDAUTO.BAT
mcopy -o -i "$boot" "$script_dir/fdconfig.sys" ::FDCONFIG.SYS

if test -r /dev/kvm && test -w /dev/kvm; then
  accel=kvm
  cpu=host
else
  accel=tcg
  cpu=max
fi
timeout_seconds=${XPAR_DOS_TIMEOUT_SECONDS:-14400}
case $timeout_seconds in ''|*[!0-9]*) fail "invalid DOS timeout" ;; esac

echo "$prog: booting FreeDOS with $accel"
qemu_status=0
timeout "$timeout_seconds" qemu-system-i386 \
  -machine pc,accel="$accel" -cpu "$cpu" -m 256 \
  -display none -monitor none -serial none -nic none -no-reboot \
  -device isa-debug-exit,iobase=0xf4,iosize=0x04 \
  -drive file="$boot",format=raw,if=floppy,readonly=on \
  -drive file="$disk",format=raw,if=ide,index=0,cache=writeback \
  -boot order=a,strict=on || qemu_status=$?

mcopy d:/RESULT.LOG "$suite_root/result.log" 2> /dev/null || true
mcopy d:/STATUS.TXT "$suite_root/status.txt" 2> /dev/null || true
test -f "$suite_root/result.log" && cat "$suite_root/result.log"

test "$qemu_status" -eq 1 || fail "QEMU exited with status $qemu_status"
test -f "$suite_root/status.txt" || fail "guest status missing"
status=`tr -d '\r\n' < "$suite_root/status.txt"`
case $status in 0) ;; 1) fail "the FreeDOS suite failed" ;;
  *) fail "invalid guest status $status" ;;
esac

echo "$prog: FreeDOS suite passed"
