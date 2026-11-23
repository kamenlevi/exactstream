#!/bin/bash
# Build mtlprof.m into a versioned dylib (never rebuilt in place: a new name per build). Usage: ./build_mtlprof.sh <version>
set -euo pipefail
cd "$(dirname "$0")"
v="${1:?version}"
out="libmtlprof_v${v}.dylib"
[ -e "$out" ] && { echo "exists: $out (use a new version)"; exit 2; }
xcrun clang -dynamiclib -O2 -fno-objc-arc -fblocks -Wall -Wno-unused-function -o "$out" mtlprof.m \
  -framework Foundation -framework Metal -framework QuartzCore
echo "built $out"
