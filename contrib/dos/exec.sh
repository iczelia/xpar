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
server_lock=
release_server() {
  test -z "$server_lock" || rmdir "$server_lock" 2>/dev/null || true
  server_lock=
}
if test -n "${DOSBOX_SERVER_LIST:-}"; then
  while test -z "$server_lock"; do
    alive=0
    while read candidate candidate_pid; do
      if kill -0 "$candidate_pid" 2>/dev/null; then
        alive=`expr "$alive" + 1`
        if mkdir "$candidate/LOCK" 2>/dev/null; then
          DOSBOX_SERVER_DIR=$candidate
          DOSBOX_SERVER_PID=$candidate_pid
          server_lock=$candidate/LOCK
          break
        fi
      fi
    done < "$DOSBOX_SERVER_LIST"
    test "$alive" -gt 0 || fail "all DOSBox-X workers exited"
    test -n "$server_lock" || sleep 0.01
  done
  trap 'release_server' EXIT
  trap 'exit 1' HUP INT TERM
fi
if test -n "${DOSBOX_SERVER_DIR:-}"; then
  request=$DOSBOX_SERVER_DIR
  test -f "$request/STARTED" || fail "DOSBox-X worker is not ready"
  test ! -f "$request/READY" || fail "DOSBox-X worker is busy"
  rm -f "$request/STATUS.TXT" "$request/STDOUT.BIN" \
        "$request/STDERR.BIN" "$request/COMMAND.BAT"
else
  request=$(mktemp -d "$DOSBOX_REQUEST_ROOT/request.XXXXXX")
  trap 'rm -rf "$request"' EXIT HUP INT TERM
fi

# NUL separators preserve spaces and keep the COMMAND.COM line short.
: > "$request/ARGS.BIN"
output_next=
for arg in "$@"; do
  if test -f "${DOSBOX_NAME_STATE:-}"; then
    lookup=$arg
    case $lookup in "$PWD"/*) lookup=${lookup#"$PWD"/} ;; esac
    if test -n "$output_next"; then
      mapped=`awk -F '\t' -v v="$lookup" \
        '$1 == "B" && $3 == v { print $2; exit }' "$DOSBOX_NAME_STATE"`
      test -z "$mapped" || arg=$mapped
      output_next=
    else case $arg in
      -o | --output) output_next=yes ;;
      --output=*)
        value=${arg#--output=}
        case $value in "$PWD"/*) value=${value#"$PWD"/} ;; esac
        mapped=`awk -F '\t' -v v="$value" \
          '$1 == "B" && $3 == v { print $2; exit }' "$DOSBOX_NAME_STATE"`
        test -z "$mapped" || arg=--output=$mapped
        ;;
      --volume=*)
        value=${arg#--volume=}
        case $value in "$PWD"/*) value=${value#"$PWD"/} ;; esac
        mapped=`awk -F '\t' -v v="$value" \
          '$1 == "F" && $3 == v { print $2; exit }' "$DOSBOX_NAME_STATE"`
        test -z "$mapped" || arg=--volume=$mapped
        ;;
      *)
        mapped=`awk -F '\t' -v v="$lookup" \
          '$1 == "F" && $3 == v { print $2; exit }' "$DOSBOX_NAME_STATE"`
        test -z "$mapped" || arg=$mapped
        ;;
    esac
    fi
  fi
  case $arg in
    "$root")   arg="C:\\" ;;
    "$root"/*)
      arg=C:/${arg#"$root"/}
      arg=$(printf '%s' "$arg" | sed 's|/|\\|g')
      ;;
  esac
  printf '%s\0' "$arg" >> "$request/ARGS.BIN"
done

command_file=$request/RUN.BAT
test -z "${DOSBOX_SERVER_DIR:-}" || command_file=$request/COMMAND.BAT
cat > "$command_file" <<EOF
@ECHO OFF
C:
CD "$guest_dir"
D:\\RUN2.EXE --status R:\\STATUS.TXT --stdout R:\\STDOUT.BIN --stderr R:\\STDERR.BIN --args R:\\ARGS.BIN D:\\$guest
EOF

if test -n "${DOSBOX_SERVER_DIR:-}"; then
  : > "$request/READY"
  while test -f "$request/READY"; do
    if ! kill -0 "$DOSBOX_SERVER_PID" 2>/dev/null; then
      sed 's/^/dosbox-x: /' "$request/dosbox.log" >&2
      fail "DOSBox-X worker exited"
    fi
    sleep 0.01
  done
else
  cat >> "$request/RUN.BAT" <<EOF
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
fi

test -f "$request/STATUS.TXT" || {
  sed 's/^/dosbox-x: /' "$request/dosbox.log" >&2
  fail "$guest wrote no status"
}
test ! -f "$request/STDOUT.BIN" || cat "$request/STDOUT.BIN"
test ! -f "$request/STDERR.BIN" || cat "$request/STDERR.BIN" >&2
status=$(tr -d '\r\n' < "$request/STATUS.TXT")
case $status in *[!0-9]*|'') fail "$guest wrote invalid status '$status'" ;; esac
exit "$status"
