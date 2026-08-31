#!/bin/sh
# Run every test against a DJGPP build. Shell tests use native helpers, but
# every xpar command crosses the DOSBox-X boundary.

set -e

prog=$(basename "$0")
fail() { echo "$prog: $*" >&2; exit 1; }

dos_build=${1:?usage: run-suite.sh DOS-BUILD NATIVE-BUILD CWSDPMI [WORKDIR]}
native_build=${2:?native build required}
cwsdpmi=${3:?CWSDPMI.EXE required}
suite_root=${4:-./dos-suite}

script_dir=$(dirname "$0")
srcdir=$(cd "$script_dir" && pwd)
top_srcdir=$(cd "$srcdir/../.." && pwd)
for f in "$dos_build/xpar.exe" "$native_build/xpar" "$cwsdpmi"; do
  test -f "$f" || fail "$f not found"
done
command -v dosbox-x > /dev/null || fail "dosbox-x is not on PATH"

dos_build=$(cd "$dos_build" && pwd)
native_build=$(cd "$native_build" && pwd)
mkdir -p "$suite_root"
suite_root=$(cd "$suite_root" && pwd)
tools=$suite_root/tools
requests=$suite_root/requests
mkdir -p "$tools" "$requests"

cp "$dos_build/xpar.exe" "$tools/XPAR.EXE"
cp "$dos_build/tests/t_unit.exe" "$tools/TUNIT.EXE"
cp "$dos_build/tests/t_codec.exe" "$tools/TCODEC.EXE"
cp "$dos_build/tests/t_central.exe" "$tools/TCENTRAL.EXE"
cp "$cwsdpmi" "$tools/CWSDPMI.EXE"

: "${CC_FOR_DOS:=i586-pc-msdosdjgpp-gcc}"
command -v "$CC_FOR_DOS" > /dev/null || fail "$CC_FOR_DOS is not on PATH"
"$CC_FOR_DOS" -O2 -o "$tools/RUN2.EXE" "$srcdir/run2.c"

export DOSBOX_TOOLS="$tools"
export DOSBOX_REQUEST_ROOT="$requests"
export DOSBOX_EXEC_CONF="$srcdir/exec.conf"
export DOSBOX_MOUNT_ROOT="$top_srcdir"

echo "$prog: running DOS test programs"
"$srcdir/exec.sh" TUNIT.EXE
"$srcdir/exec.sh" TCODEC.EXE
"$srcdir/exec.sh" TCENTRAL.EXE

launcher=$suite_root/xpar-dos
cat > "$launcher" <<EOF
#!/bin/sh
exec "$srcdir/exec.sh" XPAR.EXE "\$@"
EOF
chmod +x "$launcher"

shell_tests=$(make -s -C "$native_build" print-shell-tests)
test -n "$shell_tests" || fail "the shell test list is empty"

echo "$prog: running the shell suite through DOSBox-X"
VERBOSE=1 make -C "$native_build" -j1 check \
  TESTS="$shell_tests" \
  XPAR_TEST_BINARY="$launcher" \
  XPAR_TEST_CONFIG_H="$dos_build/config.h" \
  XPAR_TEST_CC="$CC_FOR_DOS"
