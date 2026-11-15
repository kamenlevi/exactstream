#!/bin/sh
# Leap, Wed 2026-11-04 21:2x (the orchestrator 21:18 / 21:20): finish Fit12's TRL store layer 41 with the batched Metal
# encoder so screen 7 (s7_trl) runs tonight. Writes Fit12's exact files (8-bit views + metadata, byte-identical to
# fit_trl_store.py's: 9 / 9 checked on experts 10 / 50 / 90) into Leap's own dir; Fit12 copies them in (cp -n).
# Runs only after P-CAP's resume ends, beside nothing over 8 GB (NO-SWAP margin). Usage: sh trl41_finish.sh N  (N = first expert)
N=${1:?first expert index}
OUT=$HOME/Leap/data/trl41_for_fit12
PY=$HOME/BigMoE/.venv/bin/python
[ -e "$HOME/.local/share/quiet/PAUSE" ] || [ -e "$HOME/.local/share/quiet/HOLD" ] && { echo "PAUSE/HOLD set: not starting"; exit 2; }
mkdir -p "$OUT"
cd "$HOME/Leap" || exit 1
date
env VECLIB_MAXIMUM_THREADS=4 OMP_NUM_THREADS=4 B3_THREADS=4 $PY -B scripts/trl_encode.py 41 "$N"-511 --batch 8 \
    --hx "$HOME/Fit12/data/trl/L41/_hx.npz" --fit12-out "$OUT"
rc=$?
date
# whole-ness of the union (Fit12's store + this dir): 3 files per expert, 512 experts
$PY - "$OUT" <<'EOF'
import os, sys
out, ref = sys.argv[1], os.path.expanduser("~/Fit12/data/trl/L41")
names = [f"e{e:03d}_{p}_k{k}.safetensors" for e in range(512) for p, k in (("gate", 2), ("up", 2), ("down", 3))]
have_ref = sum(os.path.exists(os.path.join(ref, n)) for n in names)
have_out = sum(os.path.exists(os.path.join(out, n)) for n in names)
union = sum(os.path.exists(os.path.join(ref, n)) or os.path.exists(os.path.join(out, n)) for n in names)
print(f"L41 files: Fit12's store {have_ref} / 1536, Leap's dir {have_out}, union {union} / 1536 -> "
      f"{'WHOLE' if union == 1536 else 'MISSING ' + str(1536 - union)}")
EOF
exit $rc
