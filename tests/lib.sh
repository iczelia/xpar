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

# Shared shell harness. Failures accumulate; exits follow Automake (0/77/99).

prog=`basename "$0"`
checks=0
failures=0
phase="(starting up)"

: "${abs_top_builddir:=..}"
: "${abs_top_srcdir:=..}"
: "${XPAR_TEST_LEVEL:=quick}"

# Use the configured POSIX shell, not the login shell.
: "${XPAR_SH:=/bin/sh}"

# Override the printed seed to reproduce a failure.
: "${XPAR_TEST_SEED:=20260823}"

hard_error() {
  echo "$prog: $*" >&2
  if test -n "${log:-}" && test -s "$log"; then
    echo "$prog:   last command output:" >&2
    sed 's/^/  | /' "$log" >&2
  fi
  exit 99
}
skip_all()   { echo "$prog: SKIP: $*" >&2;  exit 77; }

if test -n "${XPAR:-}"; then
  :
elif test -x "$abs_top_builddir/xpar"; then
  XPAR=$abs_top_builddir/xpar
elif test -x "$abs_top_builddir/xpar.exe"; then
  XPAR=$abs_top_builddir/xpar.exe
else
  hard_error "no xpar binary; set XPAR or build first"
fi

: "${MKDATA:=$abs_top_builddir/tests/mkdata}"
: "${DAMAGE:=$abs_top_builddir/tests/damage}"
: "${FORGE:=$abs_top_builddir/tests/forge}"
test -x "$MKDATA" || test -x "$MKDATA.exe" || hard_error "$MKDATA not built"
test -x "$DAMAGE" || test -x "$DAMAGE.exe" || hard_error "$DAMAGE not built"
test -x "$FORGE"  || test -x "$FORGE.exe"  || hard_error "$FORGE not built"
test -x "$MKDATA" || MKDATA=$MKDATA.exe
test -x "$DAMAGE" || DAMAGE=$DAMAGE.exe
test -x "$FORGE"  || FORGE=$FORGE.exe

work=`pwd`/tw-$prog.$$
rm -rf "$work"
mkdir "$work" || hard_error "cannot create $work"
if test -z "${XPAR_TEST_KEEP:-}"; then
  trap 'cd /; rm -rf "$work"' EXIT
  trap 'cd /; rm -rf "$work"; echo "$prog: signalled" >&2; exit 143' \
       HUP INT TERM
else
  trap 'cd /; echo "$prog: kept $work" >&2' EXIT
  trap 'cd /; echo "$prog: kept $work" >&2; exit 143' HUP INT TERM
fi
log=$work/last.log
cd "$work" || hard_error "cannot enter $work"

echo "$prog: seed $XPAR_TEST_SEED, level $XPAR_TEST_LEVEL"

# Reporting.

step() { phase="$*";  echo;  echo "$prog: --- $* ---"; }

note() { echo "$prog:   $*"; }

ok() { checks=`expr $checks + 1`; }

bad() {
  checks=`expr $checks + 1`
  failures=`expr $failures + 1`
  echo "$prog: FAIL in $phase: $*" >&2
  if test -s "$log"; then sed 's/^/  | /' "$log" >&2; fi
}

summary() {
  echo
  echo "$prog: $checks checks, $failures failed"
  test "$failures" -eq 0 || exit 1
  exit 0
}

# Running xpar. Statuses above 128, or 8, are always treated as crashes.

explain_status() {
  case $1 in
    0) echo "clean" ;;
    1) echo "damaged, repairable" ;;
    2) echo "damage beyond the recovery data" ;;
    3) echo "not found, or not an xpar set" ;;
    4) echo "usage error" ;;
    5) echo "I/O error" ;;
    6) echo "authentication failure" ;;
    7) echo "no plan fits the memory ceiling" ;;
    8) echo "INTERNAL ERROR (a bug)" ;;
    12[89]|13[0-9]|1[4-8][0-9]|19[01])
       echo "CRASHED (signal `expr $1 - 128`)" ;;
    *) echo "unrecognised status" ;;
  esac
}

never_false_success() {   # <status> <file> <pristine> <what>
  if test "$1" -ne 0; then ok;  return 0; fi
  if cmp -s "$2" "$3"; then ok
  else bad "$4: exited 0 with bytes that are not the original"; fi
}

# Inject damage or stop if the helper fails.
damage() {   # damage <file> <op>...
  "$DAMAGE" "$@" > /dev/null || hard_error "damage failed: $*"
}

# Enter a directory, or stop.
cdto() {   # cdto <dir>
  cd "$1" || hard_error "cd $1"
}

# Run a command and save its status without asserting it.
attempt() {
  status=0
  "$@" > "$log" 2>&1 || status=$?
  if test "$status" -ge 128 || test "$status" -eq 8; then
    bad "$* : `explain_status $status` (status $status)"
    return 1
  fi
  return 0
}

# Run a command and require an exact status.
run() {
  want=$1;  shift
  attempt "$@" || return 1
  if test "$status" -ne "$want"; then
    bad "$*
       expected: $want (`explain_status $want`)
       got     : $status (`explain_status $status`)"
    return 1
  fi
  ok
  return 0
}

# Run a command and require one of several statuses.
run_any() {
  want=$1;  shift
  attempt "$@" || return 1
  for w in $want; do
    if test "$status" -eq "$w"; then ok;  return 0; fi
  done
  bad "$*
       expected one of: $want
       got            : $status (`explain_status $status`)"
  return 1
}

# Assertions.

same() {
  if cmp -s "$1" "$2"; then ok
  else bad "$1 and $2 differ"; fi
}

differs() {
  if cmp -s "$1" "$2"; then bad "$1 and $2 are identical, expected a change"
  else ok; fi
}

equal() {   # equal <what> <got> <want>
  if test "x$2" = "x$3"; then ok
  else bad "$1: got '$2', want '$3'"; fi
}

exists() {
  if test -e "$1"; then ok;  else bad "$1 does not exist"; fi
}

# BSD wc pads its output to a fixed width and GNU's does not, so a count
# compared as a string has to be stripped. These give a bare number anywhere.
nlines() { wc -l | tr -d ' '; }
nbytes() { wc -c | tr -d ' '; }

# GNU and BSD stat spell the mode differently; try each.
mode_of() {   # mode_of <path>
  stat -c '%a' "$1" 2> /dev/null && return 0
  stat -f '%Lp' "$1" 2> /dev/null && return 0
  echo "?"
}

# Whether chmod round-trips file modes.
modes_work() {   # modes_work <scratch dir>
  _f=$1/mode-probe
  rm -f "$_f"
  ( : > "$_f" ) 2> /dev/null || return 1
  chmod 600 "$_f" 2> /dev/null || { rm -f "$_f";  return 1; }
  _m=`mode_of "$_f"`
  rm -f "$_f"
  test "x$_m" = "x600"
}

# Whether the filesystem creates true symbolic links.
symlinks_work() {   # symlinks_work <target> <link>
  ln -s "$1" "$2" 2> /dev/null || return 1
  test -L "$2" && return 0
  rm -f "$2" 2> /dev/null
  return 1
}

xpar_host() {
  "$XPAR" --version 2> /dev/null |
    sed -n '1s/^xpar [^ ]* (\([^,)]*\).*/\1/p'
}

# Whether this xpar build can open FIFOs.
fifos_work() {   # fifos_work <path>
  case `xpar_host` in
    *mingw* | *cygwin* | *msys* | *windows* | *djgpp* | *msdos*) return 1 ;;
  esac
  mkfifo "$1" 2> /dev/null || return 1
  test -p "$1" && return 0
  rm -f "$1" 2> /dev/null
  return 1
}

# Whether mode 555 prevents this user from creating files.
perms_bite() {   # perms_bite <scratch dir>
  _d=$1/perm-probe
  rm -rf "$_d";  mkdir -p "$_d" || return 1
  chmod 555 "$_d" 2> /dev/null || { rm -rf "$_d";  return 1; }
  if ( : > "$_d/x" ) 2> /dev/null; then
    chmod 755 "$_d";  rm -rf "$_d";  return 1
  fi
  chmod 755 "$_d";  rm -rf "$_d";  return 0
}

# Test whether a directory folds case.
folds_case() {   # folds_case <dir>
  ( : > "$1/XparCaseProbe" ) 2> /dev/null || return 0
  if test -e "$1/xparcaseprobe"; then
    rm -f "$1/XparCaseProbe";  return 0
  fi
  rm -f "$1/XparCaseProbe" "$1/xparcaseprobe"
  return 1
}

packet_body_at() {   # packet_body_at <file> <TYPE>
  "$DAMAGE" "$1" "find=$2" 2> /dev/null | head -1
}

# Test whether shell and native helpers can create a path.
can_hold() {   # can_hold <path>
  ( : > "$1" ) 2> /dev/null || return 1
  test -f "$1" || { rm -f "$1" 2> /dev/null;  return 1; }
  rm -f "$1"
  "$MKDATA" 1 1 "$1" > /dev/null 2>&1 || { rm -f "$1" 2> /dev/null;  return 1; }
  test -f "$1" || { rm -f "$1" 2> /dev/null;  return 1; }
  rm -f "$1"
  return 0
}

# Read flat JSON Lines without adding an interpreter dependency.

# Select a record type because fields such as "status" are not unique.
json_of() {    # json_of <file> <type>
  grep '"type":"'"$2"'"' "$1" | head -1
}

json_num() {   # json_num <file> <key> [<type>]
  if test -n "${3:-}"; then json_of "$1" "$3"; else cat "$1"; fi |
    sed -n 's/.*"'"$2"'":\([0-9][0-9]*\).*/\1/p' | head -1
}

json_str() {   # json_str <file> <key> [<type>]
  if test -n "${3:-}"; then json_of "$1" "$3"; else cat "$1"; fi |
    sed -n 's/.*"'"$2"'":"\([^"]*\)".*/\1/p' | head -1
}

# Geometry of a set, as shell variables: Z, S, Y, K, R, L.
read_geometry() {   # read_geometry <set>
  "$XPAR" info --json "$1" > "$work/geom.json" 2> "$log" ||
    hard_error "info --json failed on $1"
  Z=`json_num "$work/geom.json" slice_size  set`
  S=`json_num "$work/geom.json" slices      set`
  Y=`json_num "$work/geom.json" cell_bytes  set`
  R=`json_num "$work/geom.json" recovery    set`
  L=`json_num "$work/geom.json" stream_length set`
  test -n "$Z" && test -n "$S" || hard_error "no geometry in info --json"
  if test -z "$Y" || test "$Y" -eq 0; then K=1;  Y=$Z
  else K=`expr \( $Z + $Y - 1 \) / $Y`; fi
}

# Deterministic, portable randomness for reproducible corruption matrices.

# Park-Miller RNG with 31-bit-safe arithmetic.
rng_state=`expr $XPAR_TEST_SEED % 2147483647`
test "$rng_state" -gt 0 || rng_state=1

# Return through $rnd so state changes persist outside a subshell.
rnd() {   # rnd <n> -> 0 .. n-1 in $rnd
  _hi=`expr $rng_state / 127773`
  _lo=`expr $rng_state % 127773`
  rng_state=`expr 16807 \* $_lo - 2836 \* $_hi`
  test "$rng_state" -gt 0 || rng_state=`expr $rng_state + 2147483647`
  # Avoid correlated low bits for small n.
  rnd=`expr \( $rng_state / 32768 \) % "$1"`
}

# Corpora.

mkfile() {   # mkfile <path> <bytes> [<seed>] [<pattern>]
  "$MKDATA" "${3:-$XPAR_TEST_SEED}" "$2" "$1" --pattern="${4:-random}" ||
    hard_error "mkdata failed for $1"
}

# Fault injection.

#  Build and validate the LD_PRELOAD fault shim. Sets fault_pre.
fault_shim() {   # fault_shim <so path>
  fault_pre=
  case `uname -s 2> /dev/null` in  Linux) ;;  *) return 1 ;;  esac
  test -f "${srcdir:-.}/faultio.c" || return 1
  ${CC:-cc} -shared -fPIC -O1 -o "$1" "${srcdir:-.}/faultio.c" -ldl \
    > /dev/null 2>&1 || return 1
  printf 'xxxxxxxxxxxxxxxx' > fault-probe.xpa
  #  Sanitized binaries may need their runtime preloaded first.
  _asan=`${CC:-cc} -print-file-name=libasan.so 2> /dev/null`
  for _p in "$1" "$_asan:$1"; do
    case $_p in  :*) continue ;;  esac
    if env XPAR_FI_TRACE=1 LD_PRELOAD="$_p" "$XPAR" verify fault-probe.xpa \
         2>&1 | grep -q '^FI '; then
      fault_pre=$_p
      rm -f fault-probe.xpa
      return 0
    fi
  done
  rm -f fault-probe.xpa
  return 1
}
