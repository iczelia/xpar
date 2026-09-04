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

# Shared benchmark harness. Setups restore state; checks validate work.

prog=`basename "$0"`

die() { echo "$prog: $*" >&2;  exit 1; }
warn() { echo "$prog: $*" >&2; }
say() { echo "$prog: $*"; }

bench_find_tools() {
  test -n "$top" || die "internal error: top is unset"
  if test -z "${xpar:-}"; then
    if   test -x "$top/xpar";     then xpar=$top/xpar
    elif test -x "$top/xpar.exe"; then xpar=$top/xpar.exe
    else die "xpar binary not found; use --xpar"
    fi
  fi
  case $xpar in
    /*|?:[/\\]*) ;;
    *)  xpar=`cd \`dirname "$xpar"\` && pwd`/`basename "$xpar"` ;;
  esac
  test -x "$xpar" || die "not executable: $xpar"
  bench_dos=no
  if test "${XPAR_DOS_TEST:-0}" = 1; then bench_dos=yes
  else
    case `"$xpar" --version 2> /dev/null | head -1` in
      *djgpp*|*msdos*) bench_dos=yes ;;
    esac
  fi

  mkdata=${MKDATA:-$top/tests/mkdata}
  damage=${DAMAGE:-$top/tests/damage}
  timeit=${TIMEIT:-$top/bench/timeit}
  for t in "$mkdata" "$damage" "$timeit"; do
    test -x "$t" || test -x "$t.exe" ||
      die "required tool not built: $t (run 'make bench-tools')"
  done
  test -x "$mkdata" || mkdata=$mkdata.exe
  test -x "$damage" || damage=$damage.exe
  test -x "$timeit" || timeit=$timeit.exe
}

file_bytes() {
  test -f "$1" || { echo 0;  return; }
  stat -c %s "$1" 2>/dev/null && return
  stat -f %z "$1" 2>/dev/null && return
  wc -c < "$1" | tr -d ' '
}

archive_bytes() {
  _t=0
  for _f in "$1"*; do
    test -f "$_f" || continue
    _t=$((_t + `file_bytes "$_f"`))
  done
  echo "$_t"
}

# Treat R * Z as recovery payload; remaining archive bytes are overhead.
nominal_payload() {   # nominal_payload <recovery-symbols> <symbol-bytes>
  archive_nominal=$(( $1 * $2 ))
}

account_archive() {   # account_archive <base> <recovery> <symbol-bytes>
  archive_total=`archive_bytes "$1"`
  nominal_payload "$2" "$3"
  archive_overhead=$((archive_total - archive_nominal))
  test "$archive_overhead" -ge 0 || archive_overhead=0
}

# Count recovery symbols encoded as +COUNT in PAR volume names.
par_recovery_blocks() {   # par_recovery_blocks <base> <ext>
  _n=0
  for _f in "$1".vol*."$2"; do
    test -f "$_f" || continue
    _c=${_f##*+}
    _c=${_c%%.*}
    case $_c in ''|*[!0-9]*) continue ;; esac
    # Strip zero padding to avoid octal interpretation.
    while :; do
      case $_c in 0?*) _c=${_c#0} ;; *) break ;; esac
    done
    _n=$((_n + _c))
  done
  echo "$_n"
}

tree_bytes() {
  find "$1" -type f -exec cat {} + 2>/dev/null | wc -c | tr -d ' '
}

# Read fields from typed JSON Lines records.

jof() { grep '"type":"'"$2"'"' "$1" 2>/dev/null | head -1; }

jnum() {
  if test -n "${3:-}"; then jof "$1" "$3"; else cat "$1" 2>/dev/null; fi |
    sed -n 's/.*"'"$2"'":\([0-9][0-9]*\).*/\1/p' | head -1
}

jstr() {
  if test -n "${3:-}"; then jof "$1" "$3"; else cat "$1" 2>/dev/null; fi |
    sed -n 's/.*"'"$2"'":"\([^"]*\)".*/\1/p' | head -1
}

jnum0() { _v=`jnum "$@"`;  echo "${_v:-0}"; }

# Load geometry into g_z, g_y, g_k, g_s, g_r, g_l, g_codec and g_field.

read_geometry() {
  "$xpar" info --json "$1" > "$geom_json" 2> /dev/null ||
    die "cannot read set geometry: $1"
  g_z=`jnum0 "$geom_json" slice_size set`
  g_s=`jnum0 "$geom_json" slices set`
  g_y=`jnum0 "$geom_json" cell_bytes set`
  g_r=`jnum0 "$geom_json" recovery set`
  g_l=`jnum0 "$geom_json" stream_length set`
  g_codec=`jstr "$geom_json" codec set`
  g_field=`jnum0 "$geom_json" field set`
  test "$g_y" -gt 0 || g_y=$g_z
  g_k=$(( (g_z + g_y - 1) / g_y ))
}

# Park-Miller RNG; return via $rnd to preserve state.

rng_state=1

rng_seed() {
  rng_state=$(( ${1:-1} % 2147483647 ))
  test "$rng_state" -gt 0 || rng_state=1
}

rnd() {   # rnd <n> -> 0 .. n-1 in $rnd
  _hi=$(( rng_state / 127773 ))
  _lo=$(( rng_state % 127773 ))
  rng_state=$(( 16807 * _lo - 2836 * _hi ))
  test "$rng_state" -gt 0 || rng_state=$(( rng_state + 2147483647 ))
  rnd=$(( (rng_state / 32768) % $1 ))
}

# Sync before each run; optionally drop caches.

settle() {
  _t0=`date +%s 2>/dev/null || echo 0`
  sync 2>/dev/null || true
  _t1=`date +%s 2>/dev/null || echo 0`
  sync_seconds=$((sync_seconds + _t1 - _t0))
  test "$cold" = drop || return 0
  if test -w /proc/sys/vm/drop_caches; then
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
  elif test -n "$drop_sudo"; then
    echo 3 | sudo -n tee /proc/sys/vm/drop_caches > /dev/null 2>&1 || true
  fi
}

bench_probe_cold() {
  drop_sudo=
  test "$cold" = drop || return 0
  if test -w /proc/sys/vm/drop_caches; then return 0; fi
  if echo 3 | sudo -n tee /proc/sys/vm/drop_caches > /dev/null 2>&1; then
    drop_sudo=1;  return 0
  fi
  warn "cannot drop caches; using warm mode"
  cold=none
}

bench_open_output() {
  sync_seconds=0
  mkdir -p "$out" || die "cannot create output directory: $out"
  out=`cd "$out" && pwd`
  work=$out/work
  rm -rf "$work"
  mkdir -p "$work" || die "cannot create work directory: $work"
  csv=$out/results.csv
  jsonl=$out/results.json
  env_json=$out/environment.json
  provenance_json=$out/provenance.json
  kernel_json=$out/kernels.json
  geom_json=$work/geom.json
  pre_json=$work/pre.json
  out_json=$work/out.json
  out_log=$work/out.log
  if test "$bench_dos" = yes; then
    jsonl=$out/result.jsn
    env_json=$out/env.json
    provenance_json=$out/prov.json
    kernel_json=$out/kernel.jsn
    geom_json=$work/geom.jsn
    pre_json=$work/pre.jsn
    out_json=$work/out.jsn
  fi
  cmdlog=$out/commands.log
  test -s "$csv" || {
    : > "$csv"
    printf 'run_id,experiment,tool,op,rep,seed,corpus,corpus_bytes,' >> "$csv"
    printf 'codec,field,slice_size,cell_bytes,slices,recovery_spec,' >> "$csv"
    printf 'recovery_slices,layout,jobs,damage,damaged_cells,' >> "$csv"
    printf 'damaged_slices,column_depth,column_groups,' >> "$csv"
    printf 'repaired_bytes,' >> "$csv"
    printf 'archive_bytes,nominal_payload_bytes,' >> "$csv"
    printf 'format_overhead_bytes,' >> "$csv"
    printf 'scan_bytes,elapsed_us,maxrss_kb,' >> "$csv"
    printf 'in_blocks,' >> "$csv"
    printf 'out_blocks,cold,status,expect,work_ok,safety,' >> "$csv"
    printf 'expected_unsupported,note\n' >> "$csv"
  }
  test -f "$jsonl" || : > "$jsonl"
  test -f "$cmdlog" || : > "$cmdlog"
  run_id=${run_id:-0}
  bad_rows=0
  refused_rows=0
}

reset_row() {
  f_experiment=;  f_tool=xpar;  f_op=;  f_corpus=;  f_corpus_bytes=0
  f_codec=;  f_field=0;  f_slice_size=0;  f_cell_bytes=0;  f_slices=0
  f_recovery_spec=;  f_recovery_slices=0;  f_layout=;  f_damage=
  f_damaged_cells=0;  f_damaged_slices=0;  f_column_depth=0
  f_column_groups=0;  f_repaired_bytes=0;  f_archive_bytes=0
  f_nominal_payload_bytes=0;  f_format_overhead_bytes=0
  f_scan_bytes=0;  f_expect=0;  f_unsupported=;  f_note=;  f_refusals=
  f_safety=default
}

emit_row() {
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,' \
    "$run_id" "$f_experiment" "$f_tool" "$f_op" "$rep" "$seed" \
    "$f_corpus" "$f_corpus_bytes" "$f_codec" "$f_field" "$f_slice_size" \
    "$f_cell_bytes" "$f_slices" "$f_recovery_spec" "$f_recovery_slices" \
    "$f_layout" "${jobs:-auto}" "$f_damage" "$f_damaged_cells" \
    "$f_damaged_slices" >> "$csv"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$f_column_depth" "$f_column_groups" "$f_repaired_bytes" \
    "$f_archive_bytes" "$f_nominal_payload_bytes" \
    "$f_format_overhead_bytes" \
    "$f_scan_bytes" "$m_us" "$m_rss" "$m_in" "$m_out" \
    "$cold" "$m_status" "$f_expect" "$work_ok" "$f_safety" \
    "$f_unsupported" "$f_note" >> "$csv"

  printf '{"run_id":%s,"experiment":"%s","tool":"%s","op":"%s","rep":%s,' \
    "$run_id" "$f_experiment" "$f_tool" "$f_op" "$rep" >> "$jsonl"
  printf '"seed":%s,"corpus":"%s","corpus_bytes":%s,"codec":"%s",' \
    "$seed" "$f_corpus" "$f_corpus_bytes" "$f_codec" >> "$jsonl"
  printf '"field":%s,"slice_size":%s,"cell_bytes":%s,"slices":%s,' \
    "$f_field" "$f_slice_size" "$f_cell_bytes" "$f_slices" >> "$jsonl"
  printf '"recovery_spec":"%s","recovery_slices":%s,"layout":"%s",' \
    "$f_recovery_spec" "$f_recovery_slices" "$f_layout" >> "$jsonl"
  printf '"jobs":"%s","damage":"%s","damaged_cells":%s,' \
    "${jobs:-auto}" "$f_damage" "$f_damaged_cells" >> "$jsonl"
  printf '"damaged_slices":%s,"column_depth":%s,"column_groups":%s,' \
    "$f_damaged_slices" "$f_column_depth" "$f_column_groups" >> "$jsonl"
  printf '"repaired_bytes":%s,"archive_bytes":%s,' \
    "$f_repaired_bytes" "$f_archive_bytes" >> "$jsonl"
  printf '"nominal_payload_bytes":%s,"format_overhead_bytes":%s,' \
    "$f_nominal_payload_bytes" "$f_format_overhead_bytes" >> "$jsonl"
  printf '"scan_bytes":%s,' "$f_scan_bytes" >> "$jsonl"
  printf '"elapsed_us":%s,"maxrss_kb":%s,"in_blocks":%s,"out_blocks":%s,' \
    "$m_us" "$m_rss" "$m_in" "$m_out" >> "$jsonl"
  printf '"cold":"%s","status":%s,"expect":%s,"work_ok":%s,' \
    "$cold" "$m_status" "$f_expect" "$work_ok" >> "$jsonl"
  printf '"safety":"%s",' "$f_safety" >> "$jsonl"
  printf '"expected_unsupported":"%s","note":"%s"}\n' \
    "$f_unsupported" "$f_note" >> "$jsonl"
}

# Setup runs before timing; check sets sig afterward. f_expect=-1 accepts
# any foreign-tool status. f_refusals lists unsupported configurations.

setup_none() { :; }
check_none() { sig=-; }

bench_measure() {   # <setup-fn> <check-fn> <command...>
  _setup=$1;  _check=$2;  shift 2
  #  Derive the safety mode from the command actually measured, so that
  #  a call site cannot opt out of the durability machinery without the
  #  row it produces saying so.
  f_safety=
  for _a in "$@"; do
    case $_a in
      --no-journal)      f_safety="$f_safety no-journal" ;;
      --no-verify-after) f_safety="$f_safety no-verify-after" ;;
    esac
  done
  f_safety=`printf '%s' "$f_safety" | sed 's/^ //; s/ /+/g'`
  test -n "$f_safety" || f_safety=default
  _sig0=
  rep=1
  while test "$rep" -le "$reps"; do
    run_id=$((run_id + 1))
    "$_setup" "$rep" ||
      die "setup failed: $f_experiment/$f_op repetition $rep"
    echo "# run $run_id  $f_experiment  $f_tool  $f_op  rep $rep" >> "$cmdlog"
    echo "$*" >> "$cmdlog"
    settle
    m_status=0
    "$timeit" "$work/timing" "$@" > "$out_json" 2> "$out_log" ||
      m_status=$?
    m_us=`sed -n 's/^elapsed_us=//p' "$work/timing"`
    m_rss=`sed -n 's/^maxrss_kb=//p' "$work/timing"`
    m_in=`sed -n 's/^in_blocks=//p' "$work/timing"`
    m_out=`sed -n 's/^out_blocks=//p' "$work/timing"`
    : "${m_us:=0}" "${m_rss:=0}" "${m_in:=0}" "${m_out:=0}"
    # Force an unexpected status in the second repetition.
    if test "${XPAR_BENCH_BREAK:-}" = status && test "$rep" -eq 2 &&
       test "$f_expect" -eq 0; then
      m_status=9
    fi

    sig=?
    work_ok=1
    "$_check" "$rep" || work_ok=0
    if test "$f_expect" -ge 0 && test "$m_status" -ne "$f_expect"; then
      work_ok=0
      warn "$f_experiment/$f_op repetition $rep: status $m_status\
 (expected $f_expect)"
      sed 's/^/  | /' "$work/out.log" | head -8 >&2
    fi
    if test "$rep" -eq 1; then _sig0=$sig
    elif test "x$sig" != "x$_sig0"; then
      work_ok=0
      warn "$f_experiment/$f_op repetition $rep differs from repetition 1"
      warn "  repetition 1: $_sig0"
      warn "  repetition $rep: $sig"
    fi
    # Only caller-declared statuses represent unsupported configurations.
    _refusal=no
    if test "$m_status" -ne 0; then
      test "$f_expect" -eq -1 && _refusal=yes
      for _s in $f_refusals; do
        test "$m_status" -eq "$_s" && _refusal=yes
      done
    fi
    if test "$work_ok" -eq 1; then :
    elif test "$_refusal" = yes; then
      # Preserve the refusal reason for result consumers.
      if test -z "$f_unsupported"; then
        f_unsupported=`head -1 "$out_log" 2>/dev/null |
          sed 's/^xpar: //' | tr -d ',"' | cut -c1-90`
        test -n "$f_unsupported" || f_unsupported="status $m_status"
      fi
      refused_rows=$((refused_rows + 1))
    else
      warn "$f_experiment/$f_op $f_tool repetition $rep: validation failed"
      bad_rows=$((bad_rows + 1))
    fi
    emit_row
    if test "$work_ok" -eq 1; then
      say "$f_experiment/$f_op ${f_tool} repetition $rep: ${m_us} us [$sig]"
    fi
    rep=$((rep + 1))
  done
}

jstr_of() {
  if test -z "$1"; then printf 'null'
  else
    printf '"%s"' \
      "`printf '%s' "$1" | sed 's/\\\\/\\\\\\\\/g; s/"/\\\\"/g; s/	/ /g'`"
  fi
}

bench_environment() {
  _cpu=
  if test -r /proc/cpuinfo; then
    _cpu=`sed -n 's/^model name[ 	]*: *//p' /proc/cpuinfo | head -1`
    test -n "$_cpu" ||
      _cpu=`sed -n 's/^Model[ 	]*: *//p' /proc/cpuinfo | head -1`
  elif command -v sysctl > /dev/null 2>&1; then
    _cpu=`sysctl -n machdep.cpu.brand_string 2>/dev/null || true`
  fi
  _cores=
  if command -v nproc > /dev/null 2>&1; then _cores=`nproc`
  elif command -v getconf > /dev/null 2>&1; then
    _cores=`getconf _NPROCESSORS_ONLN 2>/dev/null || true`
  fi
  _mem=
  test -r /proc/meminfo &&
    _mem=`sed -n 's/^MemTotal: *\([0-9]*\).*/\1/p' /proc/meminfo`
  _gov=
  test -r /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor &&
    _gov=`cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor`
  _turbo=
  test -r /sys/devices/system/cpu/intel_pstate/no_turbo &&
    _turbo=`cat /sys/devices/system/cpu/intel_pstate/no_turbo`
  _boost=
  test -r /sys/devices/system/cpu/cpufreq/boost &&
    _boost=`cat /sys/devices/system/cpu/cpufreq/boost`
  _smt=
  test -r /sys/devices/system/cpu/smt/active &&
    _smt=`cat /sys/devices/system/cpu/smt/active`
  _cc=
  test -r "$top/config.log" &&
    _cc=`sed -n "s/^ *CC='\(.*\)'$/\1/p" "$top/config.log" | head -1`
  test -n "$_cc" || _cc=${CC:-cc}
  _ccver=`$_cc --version 2>/dev/null | head -1 || true`
  _conf=
  test -r "$top/config.log" &&
    _conf=`sed -n 's/^  \$ \(.*configure.*\)$/\1/p' \
                  "$top/config.log" | head -1`
  _commit=
  if test -d "$top/.git" && command -v git > /dev/null 2>&1; then
    _commit=`cd "$top" && git rev-parse HEAD 2>/dev/null || true`
    _dirty=`cd "$top" && git status --porcelain 2>/dev/null | head -1`
    test -z "$_dirty" || _commit="$_commit+dirty"
  fi
  _ver=`"$xpar" --version 2>&1 | head -1`
  _fs=
  command -v stat > /dev/null 2>&1 &&
    _fs=`stat -f -c %T "$work" 2>/dev/null || true`
  _started=`date -u '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || true`
  _host=`hostname 2>/dev/null || true`
  _sudo=false
  test -z "$drop_sudo" || _sudo=true

  {
    printf '{\n'
    printf '  "schema": 2,\n'
    printf '  "started_utc": %s,\n'          "`jstr_of "$_started"`"
    printf '  "host": %s,\n'                 "`jstr_of "$_host"`"
    printf '  "xpar_version": %s,\n'         "`jstr_of "$_ver"`"
    printf '  "xpar_path": %s,\n'            "`jstr_of "$xpar"`"
    printf '  "git_commit": %s,\n'           "`jstr_of "$_commit"`"
    printf '  "configure": %s,\n'            "`jstr_of "$_conf"`"
    printf '  "cc": %s,\n'                   "`jstr_of "$_cc"`"
    printf '  "cc_version": %s,\n'           "`jstr_of "$_ccver"`"
    printf '  "uname": %s,\n'                "`jstr_of "\`uname -a\`"`"
    printf '  "cpu_model": %s,\n'            "`jstr_of "$_cpu"`"
    printf '  "cores": %s,\n'                "${_cores:-null}"
    printf '  "mem_total_kb": %s,\n'         "${_mem:-null}"
    printf '  "scaling_governor": %s,\n'     "`jstr_of "$_gov"`"
    printf '  "intel_pstate_no_turbo": %s,\n' "${_turbo:-null}"
    printf '  "cpufreq_boost": %s,\n'        "`jstr_of "$_boost"`"
    printf '  "smt_active": %s,\n'           "`jstr_of "$_smt"`"
    printf '  "filesystem": %s,\n'           "`jstr_of "$_fs"`"
    printf '  "workdir": %s,\n'              "`jstr_of "$work"`"
    printf '  "cache_mode": %s,\n'           "`jstr_of "$cold"`"
    printf '  "drop_caches_via_sudo": %s,\n' "$_sudo"
    printf '  "corpus_seed": %s,\n'          "$seed"
    printf '  "repetitions": %s,\n'          "$reps"
    printf '  "jobs": %s\n'                  "`jstr_of "$jobs"`"
    printf '}\n'
  } > "$env_json"
  bench_provenance
}

# Hash the exact binary and harness because tarball builds may lack Git data.

_sha() {
  test -r "$1" || { echo null;  return; }
  _h=`sha256sum "$1" 2>/dev/null | cut -c1-64` ||
    _h=`shasum -a 256 "$1" 2>/dev/null | cut -c1-64`
  test -n "$_h" && echo "\"$_h\"" || echo null
}

bench_provenance() {
  _out=$provenance_json
  {
    printf '{\n  "schema": 1,\n'
    printf '  "recorded_utc": %s,\n' \
      "`jstr_of \"\`date -u '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null\`\"`"
    _ver=`"$xpar" --version 2>&1 | head -1`
    printf '  "xpar_version": %s,\n' "`jstr_of \"$_ver\"`"
    printf '  "xpar_binary_sha256": %s,\n' "`_sha \"$xpar\"`"
    printf '  "source_tarball": %s,\n' "`jstr_of \"${XPAR_SOURCE:-}\"`"
    printf '  "source_tarball_sha256": %s,\n' \
      "`test -n \"${XPAR_SOURCE:-}\" && _sha \"$XPAR_SOURCE\" || echo null`"
    printf '  "harness": {\n'
    _first=1
    for _f in lib.sh run.sh experiments.sh competitors.sh mktree.sh \
              plot.py timeit.c; do
      test -r "$top/bench/$_f" || continue
      test "$_first" -eq 1 || printf ',\n'
      _first=0
      printf '    "bench/%s": %s' "$_f" "`_sha \"$top/bench/$_f\"`"
    done
    for _f in mkdata.c damage.c; do
      test -r "$top/tests/$_f" || continue
      printf ',\n    "tests/%s": %s' "$_f" "`_sha \"$top/tests/$_f\"`"
    done
    printf '\n  },\n'
    printf '  "competitors": {\n'
    _first=1
    if test -n "${competitors:-}" && test -r "$competitors"; then
      # Hash baseline binaries rather than trusting version strings.
      _bins=`sed -n 's/.*"binary":"\([^"]*\)".*/\1/p' "$competitors"`
      for _b in $_bins; do
        case $_b in *.js) continue ;; esac
        test -x "$_b" || continue
        test "$_first" -eq 1 || printf ',\n'
        _first=0
        printf '    "%s": %s' "$_b" "`_sha \"$_b\"`"
      done
      for _b in `sed -n 's/.*"binary":"[^"]* \([^"]*\.js\)".*/\1/p' \
                 "$competitors"`; do
        test -r "$_b" || continue
        test "$_first" -eq 1 || printf ',\n'
        _first=0
        printf '    "%s": %s' "$_b" "`_sha \"$_b\"`"
      done
    fi
    printf '\n  },\n'
    printf '  "tools": {\n'
    _first=1
    for _t in "$mkdata" "$damage" "$timeit"; do
      test "$_first" -eq 1 || printf ',\n'
      _first=0
      printf '    "%s": %s' "`basename \"$_t\"`" "`_sha \"$_t\"`"
    done
    printf '\n  }\n}\n'
  } > "$_out"
}

bench_finish() {
  say "results:"
  say "  CSV: $csv"
  say "  JSON: $jsonl"
  say "  environment: $env_json"
  say "  commands: $cmdlog"
  if test "${refused_rows:-0}" -gt 0; then
    say "$refused_rows refused parameter combinations recorded"
  fi
  if test "${bad_rows:-0}" -gt 0; then
    warn "$bad_rows measurements failed work validation"
    return 1
  fi
  return 0
}
