#!/bin/bash
# Build libmtlrec.dylib (Breakthrough, MODULE 1a). Never rebuild it in place while a process has it loaded:
# build to a new name, then move. Usage: bash scripts/m1/rec1a/build_mtlrec.sh
set -e
cd "$(dirname "$0")"
clang -fobjc-arc -O2 -dynamiclib -o libmtlrec.dylib.new mtlrec.m -framework Foundation -framework Metal -framework QuartzCore
if lsof libmtlrec.dylib >/dev/null 2>&1; then echo "libmtlrec.dylib is loaded by a process: left the new build as libmtlrec.dylib.new"; exit 3; fi
mv libmtlrec.dylib.new libmtlrec.dylib
md5 -q mtlrec.m libmtlrec.dylib
