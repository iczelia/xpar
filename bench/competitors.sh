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

# Build benchmark baselines under --root and record them in competitors.json.

set -e

prog=`basename "$0"`

root=`pwd`/competitors
jobs=`nproc 2>/dev/null || echo 4`
only=
skip=
keep_src=

M4_VER=1.4.19
AUTOCONF_VER=2.72
AUTOMAKE_VER=1.17
LIBTOOL_VER=2.5.4
PAR2TURBO_TAG=v1.5.0
PAR2_TAG=v1.3.0
NODE_VER=

usage() {
  cat <<'EOF'
usage: competitors.sh [options]

  --root DIR     workspace and install root (default: ./competitors)
  --only LIST    build only listed tools
  --skip LIST    skip listed tools
  --jobs N       parallel make jobs
  --node VER     Node version (default: latest LTS)
  --keep-src     keep the unpacked sources
  -h, --help     show this help

All sources, tools, and install prefixes remain under --root. Build results
and source revisions are recorded in competitors.json.
EOF
  exit 0
}

while test $# -gt 0; do
  case $1 in
    --root) root=$2;  shift 2 ;;
    --only) only=$2;  shift 2 ;;
    --skip) skip=$2;  shift 2 ;;
    --jobs) jobs=$2;  shift 2 ;;
    --node) NODE_VER=$2;  shift 2 ;;
    --keep-src) keep_src=1;  shift ;;
    -h|--help) usage ;;
    *) echo "$prog: unknown option: $1 (try --help)" >&2;  exit 1 ;;
  esac
done

mkdir -p "$root"
root=`cd "$root" && pwd`
src=$root/src
opt=$root/opt
log=$root/log
mkdir -p "$src" "$opt" "$log"

# Keep caches and dotfiles under --root.
export HOME=$root/home
mkdir -p "$HOME"
export PATH=$opt/toolchain/bin:$opt/node/bin:$PATH

say()  { echo "$prog: $*"; }
warn() { echo "$prog: $*" >&2; }

wanted() {
  case " $skip " in *" $1 "*) return 1 ;; esac
  test -z "$only" && return 0
  case " $only " in *" $1 "*) return 0 ;; esac
  return 1
}

fetch() {   # fetch <url> <file>
  test -s "$src/$2" && return 0
  say "fetching $2"
  curl -fsSL --retry 3 --retry-delay 2 -o "$src/$2.part" "$1" || return 1
  mv "$src/$2.part" "$src/$2"
}

manifest=$root/competitors.json
: > "$manifest.tmp"

record() {   # record <name> <version> <source> <binary> <status> <note>
  printf '{"name":"%s","version":"%s","source":"%s","binary":"%s",' \
    "$1" "$2" "$3" "$4" >> "$manifest.tmp"
  printf '"status":"%s","note":"%s"}\n' "$5" "$6" >> "$manifest.tmp"
}

have_autotools() {
  command -v autoconf > /dev/null 2>&1 &&
  command -v automake > /dev/null 2>&1 &&
  command -v libtoolize > /dev/null 2>&1
}

build_gnu() {   # build_gnu <name> <version> <extra configure args>
  _n=$1;  _v=$2;  shift 2
  _d=$src/$_n-$_v
  test -d "$_d" || {
    fetch "https://ftp.gnu.org/gnu/$_n/$_n-$_v.tar.xz" "$_n-$_v.tar.xz" ||
      return 1
    tar -C "$src" -xf "$src/$_n-$_v.tar.xz"
  }
  say "building $_n $_v"
  (
    cd "$_d"
    ./configure --prefix="$opt/toolchain" "$@" > "$log/$_n.log" 2>&1
    make -j"$jobs" >> "$log/$_n.log" 2>&1
    make install >> "$log/$_n.log" 2>&1
  ) || { warn "build failed for $_n $_v; see $log/$_n.log";  return 1; }
}

provision_autotools() {
  if have_autotools; then
    say "using host Autotools"
    return 0
  fi
  say "building Autotools in $opt/toolchain"
  # m4 needs --enable-c++ on glibc 2.28 and newer.
  build_gnu m4 "$M4_VER" --enable-c++ || return 1
  build_gnu autoconf "$AUTOCONF_VER" || return 1
  build_gnu automake "$AUTOMAKE_VER" || return 1
  build_gnu libtool "$LIBTOOL_VER" || return 1
  have_autotools
}

pick_node() {
  test -z "$NODE_VER" || return 0
  _idx=https://nodejs.org/dist/index.json
  NODE_VER=`curl -fsSL --retry 3 "$_idx" 2>/dev/null |
    tr '}' '\n' | grep -v '"lts":false' |
    sed -n 's/.*"version":"\(v[0-9.]*\)".*/\1/p' | head -1`
  test -n "$NODE_VER" || NODE_VER=v22.20.0
}

provision_node() {
  command -v node > /dev/null 2>&1 && test -x "$opt/node/bin/node" && return 0
  pick_node
  case `uname -m` in
    x86_64|amd64) _arch=linux-x64 ;;
    aarch64|arm64) _arch=linux-arm64 ;;
    *) warn "unsupported Node architecture: `uname -m`";  return 1 ;;
  esac
  _t=node-$NODE_VER-$_arch.tar.xz
  fetch "https://nodejs.org/dist/$NODE_VER/$_t" "$_t" || return 1
  rm -rf "$opt/node"
  mkdir -p "$opt/node"
  tar -C "$opt/node" --strip-components=1 -xf "$src/$_t" || return 1
  test -x "$opt/node/bin/node"
}

build_par2turbo() {
  _p=$opt/par2-turbo
  _v=`echo "$PAR2TURBO_TAG" | sed 's/^v//'`
  _d=$src/par2cmdline-turbo-$_v
  test -d "$_d" || {
    _u=https://github.com/animetosho/par2cmdline-turbo/releases/download
    fetch "$_u/$PAR2TURBO_TAG/par2cmdline-turbo-$_v.tar.xz" \
          "par2cmdline-turbo-$_v.tar.xz" || return 1
    tar -C "$src" -xf "$src/par2cmdline-turbo-$_v.tar.xz"
  }
  say "building par2cmdline-turbo $_v"
  (
    cd "$_d"
    test -x configure || ./automake.sh > "$log/par2turbo.log" 2>&1
    ./configure --prefix="$_p" > "$log/par2turbo.log" 2>&1
    make -j"$jobs" >> "$log/par2turbo.log" 2>&1
    make install >> "$log/par2turbo.log" 2>&1
  ) || return 1
  test -x "$_p/bin/par2"
}

build_par2() {
  _p=$opt/par2
  _d=$src/par2cmdline
  test -d "$_d" ||
    git clone --depth 1 --branch "$PAR2_TAG" \
      https://github.com/Parchive/par2cmdline "$_d" > "$log/par2.log" 2>&1 ||
      return 1
  say "building par2cmdline $PAR2_TAG"
  (
    cd "$_d"
    ./automake.sh > "$log/par2.log" 2>&1
    ./configure --prefix="$_p" >> "$log/par2.log" 2>&1
    make -j"$jobs" >> "$log/par2.log" 2>&1
    make install >> "$log/par2.log" 2>&1
  ) || return 1
  test -x "$_p/bin/par2"
}

# Prefer CMake because linux/configure.ac repeats AM_INIT_AUTOMAKE.

build_par3() {
  _p=$opt/par3
  _d=$src/par3cmdline
  test -d "$_d" ||
    git clone --depth 1 https://github.com/Parchive/par3cmdline "$_d" \
      > "$log/par3.log" 2>&1 || return 1
  say "building par3cmdline"
  par3_how=cmake
  if command -v cmake > /dev/null 2>&1 &&
     cmake -S "$_d/src" -B "$_d/build" -DCMAKE_BUILD_TYPE=Release \
           -DCMAKE_INSTALL_PREFIX="$_p" > "$log/par3.log" 2>&1 &&
     cmake --build "$_d/build" -j "$jobs" >> "$log/par3.log" 2>&1 &&
     cmake --install "$_d/build" >> "$log/par3.log" 2>&1; then
    :
  else
    par3_how="autotools, configure.ac patched"
    test "$autotools_ok" -eq 1 || return 1
    sed -i.orig '/^AM_INIT_AUTOMAKE$/d' "$_d/linux/configure.ac"
    (
      cd "$_d/linux"
      autoreconf -fi >> "$log/par3.log" 2>&1
      ./configure --prefix="$_p" >> "$log/par3.log" 2>&1
      make -j"$jobs" >> "$log/par3.log" 2>&1
      make install >> "$log/par3.log" 2>&1
    ) || return 1
  fi
  test -x "$_p/bin/par3" || test -x "$_p/bin/par3cmdline"
}

build_parpar() {
  _d=$src/ParPar
  test -d "$_d" ||
    git clone --depth 1 https://github.com/animetosho/ParPar "$_d" \
      > "$log/parpar.log" 2>&1 || return 1
  say "building ParPar"
  export npm_config_cache=$root/npm-cache
  export npm_config_devdir=$root/node-gyp
  (
    cd "$_d"
    npm install --omit=dev --no-audit --no-fund > "$log/parpar.log" 2>&1
  ) || return 1
  test -f "$_d/bin/parpar.js"
}

git_head() {
  (cd "$1" && git rev-parse --short HEAD 2>/dev/null) || echo unknown
}

say "workspace: $root"
say "parallel jobs: $jobs"

autotools_ok=0
if wanted par2; then
  if provision_autotools; then autotools_ok=1
  else warn "Autotools unavailable; skipping par2cmdline"
  fi
fi

if wanted par2turbo; then
  if build_par2turbo; then
    record par2cmdline-turbo "$PAR2TURBO_TAG" \
      "github.com/animetosho/par2cmdline-turbo release tarball" \
      "$opt/par2-turbo/bin/par2" ok ""
    say "par2cmdline-turbo: $opt/par2-turbo/bin/par2"
  else
    record par2cmdline-turbo "$PAR2TURBO_TAG" \
      "github.com/animetosho/par2cmdline-turbo" "" failed \
      "see $log/par2turbo.log"
    warn "failed to build par2cmdline-turbo"
  fi
fi

if wanted par2; then
  if test "$autotools_ok" -eq 1 && build_par2; then
    record par2cmdline "$PAR2_TAG" \
      "github.com/Parchive/par2cmdline `git_head "$src/par2cmdline"`" \
      "$opt/par2/bin/par2" ok ""
    say "par2cmdline: $opt/par2/bin/par2"
  else
    record par2cmdline "$PAR2_TAG" "github.com/Parchive/par2cmdline" "" \
      failed "see $log/par2.log"
    warn "failed to build par2cmdline"
  fi
fi

if wanted par3; then
  if build_par3; then
    _b=$opt/par3/bin/par3
    test -x "$_b" || _b=$opt/par3/bin/par3cmdline
    record par3cmdline "`git_head "$src/par3cmdline"`" \
      "github.com/Parchive/par3cmdline" "$_b" ok "experimental; $par3_how"
    say "par3cmdline: $_b"
  else
    record par3cmdline unknown "github.com/Parchive/par3cmdline" "" failed \
      "see $log/par3.log"
    warn "failed to build par3cmdline"
  fi
fi

if wanted parpar; then
  if provision_node && build_parpar; then
    record parpar "`git_head "$src/ParPar"`" \
      "github.com/animetosho/ParPar (node $NODE_VER binary tarball)" \
      "$opt/node/bin/node $src/ParPar/bin/parpar.js" ok "encoder only"
    say "ParPar: node $src/ParPar/bin/parpar.js"
  else
    record parpar unknown "github.com/animetosho/ParPar" "" failed \
      "see $log/parpar.log"
    warn "failed to build ParPar"
  fi
fi

{
  printf '{\n  "schema": 1,\n  "root": "%s",\n  "built_utc": "%s",\n' \
    "$root" "`date -u '+%Y-%m-%dT%H:%M:%SZ'`"
  printf '  "host": "%s",\n  "tools": [\n' "`uname -srm`"
  sed '$!s/$/,/' "$manifest.tmp" | sed 's/^/    /'
  printf '  ]\n}\n'
} > "$manifest"
rm -f "$manifest.tmp"

test -n "$keep_src" || say "sources: $src"
say "manifest: $manifest"
cat "$manifest"
