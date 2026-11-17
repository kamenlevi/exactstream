#!/bin/bash
# g1_pick.sh: writes results/G1_PICK for tools/g1_chain.sh (PREREG_lossygate G1's arming addendum), after NewIdeas' 5d
# pick at V2's verdict and, for comments-only, 5d's dated fallback line (Tue 00:5x). It refuses unless:
#   - the env is one of the eight pinned envs;
#   - V2's both-orders verdict (PASS or FAIL) is in results/CARD_t1cx45t1c1_rev_DONE;
#   - BigMoE's key test records a PASS for the env: data/r57/g1_keytest_g1b.json, or for g1c / g1cr the probe re-run's
#     data/r57/g1_keytest_g1bp.json (G1P_PROBE_DONE);
#   - no G1_PICK exists yet (a pick is read once; a second one needs a dated line and --replace).
#   tools/g1_pick.sh <env> "<why: NewIdeas HH:MM, 5d branch, RES by 5c, the line if any>" [--replace]
cd "$HOME/EvalSuite" || exit 1
E=${1:?the env: g1 g1r g1c g1cr g1cd g1cdr g1l g1lr}; WHY=${2:?why (NewIdeas time, the 5d branch, RES)}; P=results/G1_PICK
case "$E" in g1|g1r|g1c|g1cr|g1cd|g1cdr|g1l|g1lr) ;; *) echo "not a pinned env: $E"; exit 2;; esac
head -1 results/CARD_t1cx45t1c1_rev_DONE 2>/dev/null | grep -qE '^LOSSY tier1 \(CODE-ONLY mode\) \[2 orders pooled\]: (PASS|FAIL)' \
  || { echo "no both-orders PASS / FAIL in results/CARD_t1cx45t1c1_rev_DONE"; exit 3; }
[ -e "$P" ] && [ "${3:-}" != --replace ] && { echo "$P exists: $(head -1 "$P")"; exit 4; }
KT=$(python3 -I - "$E" <<'KT'
import json, os, sys
e, d = sys.argv[1], os.path.expanduser("~/BigMoE/data/r57")
src = [f"{d}/g1_keytest_g1bp.json"] if e in ("g1c", "g1cr") else [f"{d}/g1_keytest_g1b.json"]
for f in src:
    if not os.path.exists(f):
        continue
    envs = json.load(open(f)).get("envs") or {}
    for k, v in envs.items():
        if k.startswith(e + "45g1b") and v.get("verdict") == "PASS":
            print(f"PASS ({os.path.basename(f)}: {k})"); sys.exit(0)
print(f"no key-test PASS for {e} in {[os.path.basename(f) for f in src]}"); sys.exit(1)
KT
) || { echo "$KT"; exit 5; }
echo "G1 PICK $(date '+%m-%d %H:%M:%S'): $E ($WHY; key test $KT; V2: $(head -1 results/CARD_t1cx45t1c1_rev_DONE | cut -c1-90))" > "$P"
cat "$P"
