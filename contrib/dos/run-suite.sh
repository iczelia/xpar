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
: "${DOSBOX_MOUNT_ROOT:=$top_srcdir}"
export DOSBOX_MOUNT_ROOT
export DOSBOX_NATIVE_XPAR="$native_build/xpar"
export DOSBOX_BRIDGE="$srcdir/name-bridge.sh"
export DOSBOX_EXEC="$srcdir/exec.sh"

echo "$prog: running DOS test programs"
"$srcdir/exec.sh" TUNIT.EXE
"$srcdir/exec.sh" TCODEC.EXE
"$srcdir/exec.sh" TCENTRAL.EXE

launcher=$srcdir/test-launcher.sh

# Independent shell tests can use separate emulators.  Four is enough to keep
# common CI runners busy without making DOSBox-X contend heavily for CPUs.
if test -z "${DOSBOX_JOBS:-}"; then
  DOSBOX_JOBS=`getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1`
  test "$DOSBOX_JOBS" -le 4 || DOSBOX_JOBS=4
fi
case $DOSBOX_JOBS in
  *[!0-9]*|'') fail "DOSBOX_JOBS must be a positive integer" ;;
  0) fail "DOSBOX_JOBS must be a positive integer" ;;
esac

server_list=$requests/servers
: > "$server_list"

escape_sed() { printf '%s' "$1" | sed 's/[&|]/\\&/g'; }
root_sed=$(escape_sed "$DOSBOX_MOUNT_ROOT")
tools_sed=$(escape_sed "$DOSBOX_TOOLS")
i=1
while test "$i" -le "$DOSBOX_JOBS"; do
  server=$requests/server.$i
  mkdir -p "$server"
  rmdir "$server/LOCK" 2>/dev/null || true
  rm -f "$server/READY" "$server/STARTED" "$server/STOP"
  cat > "$server/RUN.BAT" <<'EOF'
@ECHO OFF
ECHO READY>R:\STARTED
D:\RUN2.EXE --server R:\READY R:\STOP R:\COMMAND.BAT
EXIT
EOF
  server_sed=$(escape_sed "$server")
  sed -e "s|@ROOT@|$root_sed|g" \
      -e "s|@TOOLS@|$tools_sed|g" \
      -e "s|@REQUEST@|$server_sed|g" \
      "$DOSBOX_EXEC_CONF" > "$server/exec.conf"
  SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy} \
  SDL_AUDIODRIVER=${SDL_AUDIODRIVER:-dummy} \
    dosbox-x -conf "$server/exec.conf" -exit \
             > "$server/dosbox.log" 2>&1 &
  printf '%s\t%s\n' "$server" "$!" >> "$server_list"
  i=`expr "$i" + 1`
done
DOSBOX_SERVER_LIST=$server_list
export DOSBOX_SERVER_LIST
server_running=yes
stop_server() {
  test -n "$server_running" || return
  while read server pid; do : > "$server/STOP"; done < "$server_list"
  while read server pid; do wait "$pid" || true; done < "$server_list"
  server_running=
}
trap 'stop_server' EXIT
trap 'exit 1' HUP INT TERM

while read server pid; do
  n=0
  while test ! -f "$server/STARTED"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      tail -80 "$server/dosbox.log" | sed 's/^/dosbox-x: /' >&2
      fail "DOSBox-X worker exited during startup"
    fi
    n=`expr "$n" + 1`
    test "$n" -lt 300 || fail "DOSBox-X worker did not start"
    sleep 0.1
  done
done < "$server_list"

shell_tests=${DOSBOX_TESTS:-`make -s -C "$native_build" print-shell-tests`}
test -n "$shell_tests" || fail "the shell test list is empty"

echo "$prog: running the shell suite through $DOSBOX_JOBS DOSBox-X workers"
VERBOSE=1 make -C "$native_build" -j"$DOSBOX_JOBS" check \
  TESTS="$shell_tests" \
  XPAR_TEST_BINARY="$launcher" \
  XPAR_TEST_CONFIG_H="$dos_build/config.h" \
  XPAR_TEST_CC="$CC_FOR_DOS"
