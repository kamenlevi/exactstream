#!/bin/bash
# Freeze BigMoE's engine at a commit into snap/bigmoe-<sha7>/, the way snap/bigmoe-6c64148 was made (the code an
# EvalSuite run imports through EVALSUITE_BIGMOE_CODE; first use: tools/r3check.sh on BigMoE's layout patch). bigmoe/ comes
# from the commit (git show, not the working tree). The ext .so is not in git: it is copied from ~/BigMoE/ext/build/ and
# must match the sha256 prefix BigMoE sent with the commit; ext/build_next's .so is copied too if there is one. SOURCE
# records the commit, the bigmoe/ tree and both .so hashes.
#   tools/snap_bigmoe.sh <bigmoe commit> <ext/build .so sha256 prefix> "<what it is, why frozen>"
# SNAP_ROOT (default snap) puts the copy elsewhere, e.g. for a test.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
BM=$HOME/BigMoE
C=$(git -C "$BM" rev-parse --short=7 "$1^{commit}") || { echo "no commit $1 in ~/BigMoE"; exit 2; }
WANT=$2; WHY=$3
[ -n "$WANT" ] && [ -n "$WHY" ] || { echo "usage: tools/snap_bigmoe.sh <commit> <.so sha256 prefix> \"<why>\""; exit 2; }
D=${SNAP_ROOT:-snap}/bigmoe-$C
[ -e "$D" ] && { echo "$D exists"; exit 2; }
SO=$BM/ext/build/bigmoe_ext.cpython-312-darwin.so
GOT=$(shasum -a 256 "$SO" | cut -c1-12)
case "$GOT" in "$WANT"*) ;; *) echo "ext/build .so is $GOT, BigMoE sent $WANT: not freezing"; exit 3 ;; esac
mkdir -p "$D/ext/build"
for f in $(git -C "$BM" ls-tree -r --name-only "$C" -- bigmoe/); do
  mkdir -p "$D/$(dirname "$f")"
  git -C "$BM" show "$C:$f" > "$D/$f" || { echo "git show $C:$f failed"; exit 4; }
done
cp "$SO" "$D/ext/build/"
NX=$BM/ext/build_next/bigmoe_ext.cpython-312-darwin.so; NXS=""
if [ -f "$NX" ]; then
  mkdir -p "$D/ext/build_next"; cp "$NX" "$D/ext/build_next/"; NXS="; ext/build_next .so sha256 $(shasum -a 256 "$NX" | cut -c1-12)"
fi
echo "source ~/BigMoE $C (bigmoe/ tree $(git -C "$BM" rev-parse --short=7 "$C:bigmoe"), from git); ext/build .so sha256 $GOT$NXS; copied $(date '+%Y-%m-%d %H:%M') $WHY" > "$D/SOURCE"
cat "$D/SOURCE"
echo "$(find "$D/bigmoe" -type f | wc -l | tr -d ' ') files under bigmoe/"
