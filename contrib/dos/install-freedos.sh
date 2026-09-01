#!/bin/sh
# Install the FreeDOS and DJGPP files used by the test guest.

set -e

prog=${0##*/}
fail() { echo "$prog: $*" >&2; exit 1; }

prefix=${1:?usage: install-freedos.sh PREFIX}
case $prefix in ''|/) fail "unsafe prefix" ;; esac

version=1.4
root=$prefix/share/xpar-dos
test -f "$root/version-$version" && test -f "$root/freedos.img" &&
  test -f "$root/bin/BASH.EXE" && exit 0

for tool in curl install sha256sum unzip; do
  command -v "$tool" > /dev/null || fail "$tool is required"
done

work=`mktemp -d "${TMPDIR:-/tmp}/xpar-freedos.XXXXXX"`
stage=$prefix.tmp.$$
trap 'rm -rf "$work" "$stage"' EXIT HUP INT TERM
mkdir -p "$work/download" "$work/djgpp" "$stage/share/xpar-dos/bin"

freedos=FD14-FloppyEdition.zip
freedos_url=https://www.ibiblio.org/pub/micro/pc-stuff/freedos/files/distributions/1.4
curl -fL "$freedos_url/$freedos" -o "$work/download/$freedos"
echo "45b1fa7c52dd996c3bfa5e352ffcd410781b952a6ad629f15a4c9ec4bbaefc5a  $work/download/$freedos" |
  sha256sum -c -

packages='6c52e315ac779ff2b9820af2526d5745ec5464b070ddcf322398f36eb4902968 v2gnu/bsh4428b.zip
83551a1a626de95d07dff15c5e5ab9abec7763d81185440b708f41830b8e781f v2gnu/fil41br3.zip
7b30845a84d2e4371c5a46da3ca297a92dab4940d7f398e23cd7389ed84289e7 v2gnu/shl2011br3.zip
380e926877ef9af2e20831df79c33c940885a270c8979600901ee7b1d97e0c53 v2gnu/txt20br3.zip
31966a09fb446998b924f379da7a52c1c04041122ef1a059f90e63b55ad2ca40 v2gnu/grep228b.zip
1d29f5d41e67cdaefe2acb306035b27b0a4f7fadfb7cd9052dbd68acb1cc0b99 v2gnu/gwk500b.zip
518c2b926eda90447df2f2c156b7b478ed9b469b8a2d76d3384eff38923da422 v2gnu/sed48b.zip
cce446d53dd3f0d6b5fa076454e8be21245290103bbd985a5ba593536a003e00 v2gnu/fnd4233br5.zip
d1167d5feb352d2fdea6d871db9974e2a460895c9435c027ac71a84f7d604f77 v2gnu/dif37b.zip
deacda0488e1cdd7c4a9f32fab45662b34c0ed6b2d7d4d13bc07041b62004a8c v2misc/csdpmi7b.zip'

echo "$packages" | while read sum path; do
  file=${path##*/}
  curl -fL "https://www.delorie.com/pub/djgpp/current/$path" \
    -o "$work/download/$file"
  echo "$sum  $work/download/$file" | sha256sum -c -
  unzip -q -o "$work/download/$file" -d "$work/djgpp"
done

unzip -j -q "$work/download/$freedos" 144m/x86BOOT.img \
  -d "$stage/share/xpar-dos"
mv "$stage/share/xpar-dos/x86BOOT.img" \
   "$stage/share/xpar-dos/freedos.img"

for file in "$work"/djgpp/bin/*.exe "$work"/djgpp/bin/*.EXE; do
  test -f "$file" || continue
  name=${file##*/}
  name=`printf '%s' "$name" | tr '[:lower:]' '[:upper:]'`
  install -m 644 "$file" "$stage/share/xpar-dos/bin/$name"
done
# Replace DJGPP link stubs for commands called directly.
install -m 644 "$work/djgpp/bin/bash.exe" \
  "$stage/share/xpar-dos/bin/SH.EXE"
install -m 644 "$work/djgpp/bin/gawk.exe" \
  "$stage/share/xpar-dos/bin/AWK.EXE"

: > "$stage/share/xpar-dos/version-$version"
mkdir -p "$prefix/share"
rm -rf "$root"
mv "$stage/share/xpar-dos" "$root"
rm -rf "$stage"
trap - EXIT HUP INT TERM
rm -rf "$work"
