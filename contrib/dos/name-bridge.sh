#!/bin/sh
# Present DOS 8.3 set names under the names expected by the POSIX tests.

set -e

mode=${1:?usage: name-bridge.sh pre|post STATE [XPAR-ARGUMENT...]}
state=${2:?state file required}
shift 2
: "${DOSBOX_NATIVE_XPAR:?DOSBOX_NATIVE_XPAR is required}"

active=$state.active
tab=`printf '\tX'`; tab=${tab%X}

same_inode() {
  test "`ls -di "$1" | awk '{print $1}'`" = \
       "`ls -di "$2" | awk '{print $1}'`"
}

clear_dosbox_names() {
  for db in .DBLOCALFILE_ATR_*; do
    test -e "$db" || continue
    rm -f "$db"
  done
}

short_actual() {
  actual=$1
  test -e "$actual" && return 0
  case $1 in
    */*) _sa_dir=${1%/*}; _sa_leaf=${1##*/} ;;
    *)   _sa_dir=.;       _sa_leaf=$1 ;;
  esac
  _sa_want=`printf '%s' "$_sa_leaf" | tr 'a-z' 'A-Z'`
  for _sa_path in "$_sa_dir"/*; do
    test -e "$_sa_path" || continue
    _sa_have=`printf '%s' "${_sa_path##*/}" | tr 'a-z' 'A-Z'`
    if test "$_sa_have" = "$_sa_want"; then actual=$_sa_path; return 0; fi
  done
}

canonical_short() {
  case $1 in
    */*) _cs_dir=${1%/*}/; _cs_leaf=${1##*/} ;;
    *)   _cs_dir=;         _cs_leaf=$1 ;;
  esac
  printf '%s%s' "$_cs_dir" "`printf '%s' "$_cs_leaf" | tr 'a-z' 'A-Z'`"
}

touch "$state"

add_base() {
  base=$1
  case $base in "$PWD"/*) base=${base#"$PWD"/} ;; esac
  case $base in */*) dir=${base%/*}/; leaf=${base##*/} ;; *) dir=; leaf=$base ;; esac
  leaf=`printf '%s' "$leaf" | tr 'a-z' 'A-Z' | tr -c 'A-Z0-9_' '_'`
  leaf=`printf '%-4.4s' "${leaf}____" | tr ' ' '_'`
  short=$dir$leaf
  if awk -F '\t' -v b="$base" '$1 == "B" && $3 == b { found=1 } END { exit !found }' "$state"
  then return 0
  fi
  n=0
  while awk -F '\t' -v s="$short" '$1 == "B" && $2 == s { found=1 } END { exit !found }' "$state"; do
    short=$dir`printf '%.2s%02d' "$leaf" "$n"`
    n=`expr "$n" + 1`
    test "$n" -lt 100 || exit 2
  done
  printf 'B\t%s\t%s\n' "$short" "$base" >> "$state"
}

if test "$mode" != pre && test "$mode" != post; then
  exit 2
fi

# Remember requested output bases and set names used as inputs.
want=
for arg in "$@"; do
  if test -n "$want"; then add_base "$arg"; want=; continue; fi
  case $arg in
    -o | --output) want=yes ;;
    --output=*) add_base "${arg#--output=}" ;;
    --volume=*) ;;
    *.xpa | *.XPA)
      base=${arg%.[xX][pP][aA]}
      base=`printf '%s' "$base" | sed 's/\.g[0-9][0-9]*$//;s/\.v[0-9][0-9]*+[0-9][0-9]*$//'`
      add_base "$base"
      ;;
  esac
done

if test "$mode" = pre; then
  clear_dosbox_names
  : > "$active"
  while IFS="$tab" read kind short long; do
    test "$kind" = F || continue
    # The launcher translates arguments too, so DOS sees only recorded names.
    short_actual "$short"
    if test -e "$long"; then
      rm -f "$actual"
      mv "$long" "$short"
      printf '%s\t%s\n' "$short" "$long" >> "$active"
    else
      rm -f "$actual"
    fi
  done < "$state"
  exit 0
fi

add_file() {
  short=`canonical_short "$1"`; long=$2
  if awk -F '\t' -v s="$short" -v l="$long" \
    '$1 == "F" && $2 == s && $3 == l { found=1 } END { exit !found }' "$state"
  then return 0
  fi
  if awk -F '\t' -v s="$short" \
    '$1 == "F" && $2 == s { found=1 } END { exit !found }' "$state"
  then
    swap=$state.swap.$$
    awk -F '\t' -v OFS='\t' -v s="$short" -v l="$long" '
      $1 == "F" && $2 == s {
        if (!done) print "F", s, l
        done=1
        next
      }
      { print }
    ' "$state" > "$swap"
    mv "$swap" "$state"
    return 0
  fi
  printf 'F\t%s\t%s\n' "$short" "$long" >> "$state"
}

# Read each DOS layout with the native binary. It supplies recovery ranges,
# which are not encoded in an 8.3 recovery filename.
tmp=$state.info
awk -F '\t' '$1 == "B" { print $2 "\t" $3 }' "$state" |
while IFS="$tab" read shortbase base; do
  case $shortbase in */*) shortdir=${shortbase%/*}/ ;; *) shortdir= ;; esac
  gen=0
  while test "$gen" -le 9; do
    if test "$gen" -eq 0; then
      index=$shortbase.XPA
      prefix=$base
    else
      index=$shortbase.XG$gen
      prefix=$base.g`printf '%03d' "$gen"`
    fi
    short_actual "$index"
    if test -f "$actual" &&
       "$DOSBOX_NATIVE_XPAR" info --generation="$gen" "$actual" > "$tmp" 2>/dev/null
    then
      add_file "$index" "$prefix.xpa"
      awk '/^    recovery / { split($4, r, "\\.\\."); print $2, r[1], r[2] }
           /^    data / { print $2, "data", ++data - 1 }' "$tmp" > "$tmp.vol"
      maxfirst=0 maxcount=1
      while read name first last; do
        test "$first" = data && continue
        count=`expr "$last" - "$first" + 1`
        test "$first" -le "$maxfirst" || maxfirst=$first
        test "$count" -le "$maxcount" || maxcount=$count
      done < "$tmp.vol"
      wf=${#maxfirst}; test "$wf" -ge 2 || wf=2
      wc=${#maxcount}; test "$wc" -ge 2 || wc=2
      while read name first last; do
        name=$shortdir$name
        if test "$first" = data; then
          long=$prefix.d`printf '%02d' "$last"`
          add_file "$name" "$long"
          label=`printf '%s' "$name" | sed 's/\.D\([0-9][0-9]\)$/.L\1/'`
          add_file "$label" "$long.xpa"
        else
          count=`expr "$last" - "$first" + 1`
          long=$prefix.v`printf "%0${wf}d" "$first"`+`printf "%0${wc}d" "$count"`.xpa
          add_file "$name" "$long"
        fi
      done < "$tmp.vol"
    fi
    gen=`expr "$gen" + 1`
  done
  add_file "$shortbase.XPU" "$base.xparundo"
  add_file "$shortbase.XPM" "$base.xparmaint"
  add_file "$shortbase.XPI" "$base.xparidx"
  gen=1
  while test "$gen" -le 9; do
    add_file "$shortbase.XU$gen" "$base.g`printf '%03d' "$gen"`.xparundo"
    gen=`expr "$gen" + 1`
  done
done

# A short member missing after xpar ran was deleted. Otherwise publish its
# harness spelling and remove the extra hard-link name.
if test -f "$active"; then
  while IFS="$tab" read short long; do
    short_actual "$short"
    if test ! -e "$actual"; then rm -f "$long"; fi
  done < "$active"
fi
awk -F '\t' '$1 == "F" { print $2 "\t" $3 }' "$state" |
while IFS="$tab" read short long; do
  short_actual "$short"
  test -e "$actual" || continue
  if test -e "$long"; then
    if same_inode "$actual" "$long"; then
      rm -f "$actual"
    else
      rm -f "$long"
      mv "$actual" "$long"
    fi
  else
    mv "$actual" "$long"
  fi
done
rm -f "$active" "$tmp" "$tmp.vol"
clear_dosbox_names
