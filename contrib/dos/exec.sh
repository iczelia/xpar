#!/bin/sh
# Run one program in DOSBox-X while sharing the test suite's working tree.

set -e

prog=$(basename "$0")
fail() { echo "$prog: $*" >&2; exit 99; }

guest=${1:?usage: exec.sh PROGRAM.EXE [ARGUMENT...]}
shift

: "${DOSBOX_MOUNT_ROOT:?DOSBOX_MOUNT_ROOT is required}"
: "${DOSBOX_TOOLS:?DOSBOX_TOOLS is required}"
: "${DOSBOX_REQUEST_ROOT:?DOSBOX_REQUEST_ROOT is required}"
: "${DOSBOX_EXEC_CONF:?DOSBOX_EXEC_CONF is required}"

command -v dosbox-x > /dev/null || fail "dosbox-x is not on PATH"
test -f "$DOSBOX_TOOLS/$guest" || fail "$DOSBOX_TOOLS/$guest not found"
test -f "$DOSBOX_TOOLS/RUN2.EXE" || fail "$DOSBOX_TOOLS/RUN2.EXE not found"

root=$(cd "$DOSBOX_MOUNT_ROOT" && pwd)
tools=$(cd "$DOSBOX_TOOLS" && pwd)
case $PWD in
  "$root")   guest_dir="\\" ;;
  "$root"/*) guest_dir=/${PWD#"$root"/} ;;
  *) fail "$PWD is outside the mounted tree $root" ;;
esac
guest_dir=$(printf '%s' "$guest_dir" | sed 's|/|\\|g')

mkdir -p "$DOSBOX_REQUEST_ROOT"
request=$(mktemp -d "$DOSBOX_REQUEST_ROOT/request.XXXXXX")
trap 'rm -rf "$request"' EXIT HUP INT TERM

# NUL separators preserve spaces and keep the COMMAND.COM line short.
: > "$request/ARGS.BIN"
for arg in "$@"; do
  case $arg in
    "$root")   arg="C:\\" ;;
    "$root"/*)
      arg=C:/${arg#"$root"/}
      arg=$(printf '%s' "$arg" | sed 's|/|\\|g')
      ;;
  esac
  printf '%s\0' "$arg" >> "$request/ARGS.BIN"
done

cat > "$request/RUN.BAT" <<EOF
@ECHO OFF
C:
CD "$guest_dir"
D:\\RUN2.EXE --status R:\\STATUS.TXT --args R:\\ARGS.BIN D:\\$guest > R:\\OUTPUT.BIN
EXIT
EOF

escape_sed() { printf '%s' "$1" | sed 's/[&|]/\\&/g'; }
root_sed=$(escape_sed "$root")
tools_sed=$(escape_sed "$tools")
request_sed=$(escape_sed "$request")
sed -e "s|@ROOT@|$root_sed|g" \
    -e "s|@TOOLS@|$tools_sed|g" \
    -e "s|@REQUEST@|$request_sed|g" \
    "$DOSBOX_EXEC_CONF" > "$request/exec.conf"

SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy} \
SDL_AUDIODRIVER=${SDL_AUDIODRIVER:-dummy} \
  dosbox-x -conf "$request/exec.conf" -exit \
           > "$request/dosbox.log" 2>&1 || {
    sed 's/^/dosbox-x: /' "$request/dosbox.log" >&2
    fail "dosbox-x exited non-zero"
  }

test -f "$request/STATUS.TXT" || {
  sed 's/^/dosbox-x: /' "$request/dosbox.log" >&2
  fail "$guest wrote no status"
}
test ! -f "$request/OUTPUT.BIN" || cat "$request/OUTPUT.BIN"
status=$(tr -d '\r\n' < "$request/STATUS.TXT")
case $status in *[!0-9]*|'') fail "$guest wrote invalid status '$status'" ;; esac
exit "$status"
