#!/bin/sh
# Run the C suite in one FreeDOS guest.

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
for file in xpar.exe tests/t_suite.exe; do
  test -f "$dos_build/$file" || fail "$dos_build/$file not found"
done
for file in freedos.img bin/CWSDPMI.EXE bin/SYNC.EXE; do
  test -f "$freedos_root/$file" || fail "$freedos_root/$file not found"
done

: "${CC_FOR_DOS:=i586-pc-msdosdjgpp-gcc}"
command -v "$CC_FOR_DOS" > /dev/null || fail "$CC_FOR_DOS is required"

case ${XPAR_TEST_LEVEL:-quick} in
  quick|full|torture) test_level=${XPAR_TEST_LEVEL:-quick} ;;
  *) fail "invalid XPAR_TEST_LEVEL" ;;
esac
boot=$suite_root/boot.img
disk=$suite_root/tests.img
mtools=$suite_root/mtools.conf
rm -f "$boot" "$disk" "$suite_root/result.log" \
      "$suite_root/error.log" "$suite_root/status.txt"
cp "$freedos_root/freedos.img" "$boot"
truncate -s 528482304 "$disk"

printf 'drive d: file="%s" partition=1\n' "$disk" > "$mtools"
MTOOLSRC=$mtools
MTOOLS_NO_VFAT=1
export MTOOLSRC MTOOLS_NO_VFAT
mpartition -I d:
mpartition -c -s 63 -h 16 -t 1023 d:
mformat -v XPARTEST d:
mmd d:/BIN d:/BLD d:/SRC d:/SRC/FORMAT d:/TMP

mcopy "$freedos_root/bin/CWSDPMI.EXE" d:/BIN/CWSDPMI.EXE
mcopy "$freedos_root/bin/SYNC.EXE" d:/BIN/SYNC.EXE
"$CC_FOR_DOS" -Os -march=i386 -mtune=i386 \
  -o "$suite_root/EXIT.EXE" "$script_dir/qemu-exit.c"
mcopy "$suite_root/EXIT.EXE" d:/BIN/EXIT.EXE

mcopy "$dos_build/xpar.exe" d:/BLD/XPAR.EXE
mcopy "$dos_build/tests/t_suite.exe" d:/BLD/TSUITE.EXE
mcopy "$top_srcdir/tests/format/DATA.BIN" d:/SRC/FORMAT/DATA.BIN
mcopy "$top_srcdir/tests/format/m8.xpa" d:/SRC/FORMAT/M8.XPA
mcopy "$top_srcdir/tests/format/a16.xpa" d:/SRC/FORMAT/A16.XPA
mcopy "$top_srcdir/tests/format/f16a.xpa" d:/SRC/FORMAT/F16A.XPA

sed "s/@LEVEL@/$test_level/g" "$script_dir/fdauto.bat" \
  > "$suite_root/FDAUTO.BAT"
mcopy -o -i "$boot" "$suite_root/FDAUTO.BAT" ::FDAUTO.BAT
mcopy -o -i "$boot" "$script_dir/fdconfig.sys" ::FDCONFIG.SYS

if test -r /dev/kvm && test -w /dev/kvm; then
  accel=kvm
  cpu=host
else
  accel=tcg
  cpu=max
fi
timeout_seconds=${XPAR_DOS_TIMEOUT_SECONDS:-1800}
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
mcopy d:/ERROR.LOG "$suite_root/error.log" 2> /dev/null || true
mcopy d:/STATUS.TXT "$suite_root/status.txt" 2> /dev/null || true
test -f "$suite_root/result.log" && cat "$suite_root/result.log"
test -f "$suite_root/error.log" && cat "$suite_root/error.log"

test "$qemu_status" -eq 1 || fail "QEMU exited with status $qemu_status"
test -f "$suite_root/status.txt" || fail "guest status missing"
status=`tr -d '\r\n' < "$suite_root/status.txt"`
case $status in 0) ;; 1) fail "the FreeDOS suite failed" ;;
  *) fail "invalid guest status $status" ;;
esac

echo "$prog: FreeDOS suite passed"
