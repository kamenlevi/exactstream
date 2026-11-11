#!/bin/bash
# install.sh: set up a clone of this package on an Apple-silicon Mac (64 GB).
#   bash install.sh --store <model dir> [--weights <dir>] [--offline [--find-links <wheel dir>]] [--python <python3.12>] [--jobs N]
# 1. Writes this clone's absolute path where the code says @ROOT@ (re-run after moving the clone: it rewrites the old path).
# 2. Creates .venv with the pinned packages (requirements.txt). --offline installs from the local uv cache only (no network);
#    --find-links adds a local wheel directory there (e.g. an mtplx-2.12.2 wheel, since uv caches no git checkout offline).
# 3. Builds the C++ extension into engine/ext/build (CMake, nanobind 2.15.0 from third_party/nanobind, MLX from the venv).
# 4. Links engine/models/bare -> the model dir (the Hugging Face files + the converted expert store records3/), and
#    weights/ -> the drafter fine-tune and the MTP head (l120_ft.safetensors, d22_b5.safetensors).
# 5. Checks that the block drafter's base, incoai/Qwen3.8-Flash-Next-DFlash2, is in the Hugging Face cache (HF_HOME) at the
#    revision every measured run loaded (run.py: DRAFTER; the engine process pins it); prints the fetch command if not.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd -P)
STORE= WEIGHTS= OFFLINE=0 PY=${PYTHON:-} JOBS=4 FL=
while [ $# -gt 0 ]; do
  case "$1" in
    --store) STORE=$2; shift 2 ;;
    --weights) WEIGHTS=$2; shift 2 ;;
    --offline) OFFLINE=1; shift ;;
    --find-links) FL=$2; shift 2 ;;
    --python) PY=$2; shift 2 ;;
    --jobs) JOBS=$2; shift 2 ;;
    *) echo "install.sh: unknown option $1"; exit 2 ;;
  esac
done
say() { echo "[install] $*"; }
[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || { echo "install.sh: needs an Apple-silicon Mac"; exit 2; }

# 1. root path
OLD=$(cat "$ROOT/.root" 2>/dev/null || echo @ROOT@)
if [ "$OLD" != "$ROOT" ]; then
  n=0
  while IFS= read -r -d '' f; do
    if grep -qF "$OLD" "$f"; then LC_ALL=C sed -i '' "s#$OLD#$ROOT#g" "$f"; n=$((n + 1)); fi
  done < <(find "$ROOT/engine" "$ROOT/kernels" "$ROOT/third_party" "$ROOT/drafters" "$ROOT/keepset" "$ROOT/harness" "$ROOT/convert" \
             -path "$ROOT/third_party/nanobind" -prune -o \( -name '*.py' -o -name '*.sh' \) -type f -print0)
  echo "$ROOT" > "$ROOT/.root"
  say "root path written into $n files ($OLD -> $ROOT)"
fi

# 2. venv + pinned packages
if [ -z "$PY" ]; then
  for c in python3.12 /opt/homebrew/opt/python@3.12/bin/python3.12 /usr/local/bin/python3.12; do
    command -v "$c" >/dev/null 2>&1 && { PY=$(command -v "$c"); break; }
  done
fi
[ -n "$PY" ] && "$PY" -c 'import sys; assert sys.version_info[:2] == (3, 12)' 2>/dev/null || { echo "install.sh: needs Python 3.12 (--python)"; exit 2; }
if [ ! -x "$ROOT/.venv/bin/python" ]; then
  if [ $OFFLINE = 1 ]; then
    uv venv -q -p "$PY" "$ROOT/.venv"
  else
    "$PY" -m venv "$ROOT/.venv"
  fi
fi
if [ $OFFLINE = 1 ]; then
  if [ -n "$FL" ]; then                                    # the git pin becomes the version pin, served by the wheel dir
    sed 's#^mtplx @ git+.*#mtplx==2.12.2#' "$ROOT/requirements.txt" > "$ROOT/.venv/requirements.offline.txt"
    uv pip install -q --offline --find-links "$FL" -p "$ROOT/.venv/bin/python" -r "$ROOT/.venv/requirements.offline.txt"
  else
    uv pip install -q --offline -p "$ROOT/.venv/bin/python" -r "$ROOT/requirements.txt"
  fi
else
  "$ROOT/.venv/bin/python" -m pip install -q -r "$ROOT/requirements.txt"
fi
"$ROOT/.venv/bin/python" - <<'EOF'
import importlib.metadata as m
want = {"mlx": "0.32.2", "mlx-metal": "0.32.2", "mlx-lm": "0.31.3", "mtplx": "2.12.2"}
got = {k: m.version(k) for k in want}
assert got == want, f"pinned versions differ: {got} != {want}"
print("[install] venv OK:", ", ".join(f"{k} {v}" for k, v in got.items()))
EOF

# 3. the extension
B="$ROOT/engine/ext/build"
cmake -S "$ROOT/engine/ext" -B "$B" -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$ROOT/.venv/bin/python" \
  -DNB_CMAKE_DIR="$ROOT/third_party/nanobind/cmake" > "$ROOT/engine/ext/cmake.log" 2>&1 || { tail -20 "$ROOT/engine/ext/cmake.log"; exit 3; }
cmake --build "$B" -j "$JOBS" > "$ROOT/engine/ext/build.log" 2>&1 || { tail -30 "$ROOT/engine/ext/build.log"; exit 3; }
ls "$B"/bigmoe_ext*.so > /dev/null && say "extension built: $(ls "$B"/bigmoe_ext*.so)"

# 4. model store + drafter weights
if [ -n "$STORE" ]; then
  STORE=$(cd "$STORE" && pwd -P)
  for f in config.json tokenizer.json model.safetensors.index.json mtp.safetensors records3/index.json; do
    [ -e "$STORE/$f" ] || { echo "install.sh: $STORE has no $f (see README: Weights)"; exit 4; }
  done
  mkdir -p "$ROOT/engine/models"; ln -sfn "$STORE" "$ROOT/engine/models/bare"
  say "store linked: engine/models/bare -> $STORE"
fi
if [ -n "$WEIGHTS" ]; then
  WEIGHTS=$(cd "$WEIGHTS" && pwd -P)
  mkdir -p "$ROOT/weights"
  for f in l120_ft.safetensors d22_b5.safetensors; do
    [ -e "$WEIGHTS/$f" ] || { echo "install.sh: $WEIGHTS has no $f"; exit 4; }
    ln -sfn "$WEIGHTS/$f" "$ROOT/weights/$f"
  done
  say "drafter weights linked from $WEIGHTS"
fi
# 5. the block drafter's base checkpoint: pinned revision, read from the Hugging Face cache (nothing is downloaded here)
if d=$("$ROOT/.venv/bin/python" -c "import sys; sys.path.insert(0, sys.argv[1]); import run; p = run.drafter_cached(); print(p or ''); sys.exit(0 if p else 1)" "$ROOT" 2>/dev/null); then
  say "drafter base in the Hugging Face cache at the pinned revision: $d"
else
  say "NOTE: the drafter base is not in the Hugging Face cache at the pinned revision. Fetch it once (1.2 GB):"
  say "  $("$ROOT/.venv/bin/python" -c "import sys; sys.path.insert(0, sys.argv[1]); import run; print('.venv/bin/hf download', run.DRAFTER[0], 'config.json model.safetensors --revision', run.DRAFTER[1])" "$ROOT")"
fi
say "done. Try: .venv/bin/python run.py --config exact --mem 45 --prompt 'Explain how a refrigerator works.'"
