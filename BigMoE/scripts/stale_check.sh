#!/bin/bash
# THE STALE-SLOT CHECK on the 45 GB engine (the orchestrator 11-03 22:5x; Breakthrough's SCOREBOARD ee2b3b8; Leap's
# S7B-PROBE 50de28c / fb999be: inside ONE command buffer, a slot the GPU read and the CPU then rewrote can be re-read
# stale; across buffers it is fresh). Our "exact" claims compare the engine with its own reference, so a deterministic stale
# read would hide in both. Here the same engine runs with and without a command-buffer boundary before every MoE layer
# call (R57_CBB=1), every check's verify logits hashed (R57_LOGHASH=1), at C 160 (CAP 128, three prompts: code / IFEval /
# multilingual):
#   sc0 / sc1   the default, without / with the boundary
#   sc2 / sc3   the smallest sidecar pool the 16-row checks allow (C_l 32: its slots are reused fastest)
#   sc4 / sc5   the zero-copy path (BIGMOE_RECORDS_DIRECT: preadv straight into the slots the GPU reads)
# Line (pre-registered 22:5x): PASS = in all three pairs every token AND every check's logits hash equal; any difference =
# FAIL, to the orchestrator at once (every 45 GB number would then need the boundary).
# usage: inside turn45 (HAVE_LOCK=1 KEEP_LOCK=1), frozen copies (data/r57/snap/*_sc.*)
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
mkdir -p data/sc
LOG="$HOME/BigMoE/data/sc/sc.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
RUN="/bin/bash data/r57/snap/c45_runner_sc.sh"
export C45_SCREEN=data/r57/snap/r57_screen_sc.py C45_DEADLINE=${SC_DEADLINE:-2400} C45_OUT="$HOME/BigMoE/data/sc"
CC=${SC_C:-160}
KEYS="humaneval/HumanEval/68,ifeval/1220,multi/de3"
H="R57_LOGHASH=1"
say "stale-slot check start (pid $$, C $CC)"
R57_KEYS=$KEYS MG=${SC_MG:-3} C45_CAP=128 C45_ALARM=600 C45_REFTOK= C45_ARMS="sc0_$CC:$CC:$H sc1_$CC:$CC:$H,R57_CBB=1 sc2_$CC:$CC:$H,BIGMOE_TIER_CL=32 sc3_$CC:$CC:$H,BIGMOE_TIER_CL=32,R57_CBB=1 sc4_$CC:$CC:$H,BIGMOE_RECORDS_DIRECT=1 sc5_$CC:$CC:$H,BIGMOE_RECORDS_DIRECT=1,R57_CBB=1" $RUN
say "arms rc=$?"
"$HOME/BigMoE/.venv/bin/python" - "$CC" >> "$LOG" 2>&1 <<'PYEOF'
import json, os, sys
C = sys.argv[1]; R = os.path.expanduser("~/BigMoE/data/r57")
def load(t):
    try:
        return {r["key"]: r for r in json.load(open(os.path.join(R, f"{t}_{C}_runs.json"))) if not r.get("bad")}
    except Exception as e:
        return {"_err": str(e)}
def toks(t):
    try:
        return {k: v.get("gate+t") for k, v in json.load(open(os.path.join(R, f"{t}_{C}_tokens.json"))).items()}
    except Exception:
        return {}
ok_all = True
for a, b, what in (("sc0", "sc1", "default"), ("sc2", "sc3", "sidecar C_l 32"), ("sc4", "sc5", "zero-copy (direct)")):
    ra, rb, ta, tb = load(a), load(b), toks(a), toks(b)
    keys = sorted(set(ra) & set(rb) - {"_err"})
    tok_eq = all(ta.get(k) == tb.get(k) for k in keys)
    lh_eq = all(ra[k].get("lh") == rb[k].get("lh") and ra[k].get("lh") for k in keys)
    nchk = sum(len(ra[k].get("lh") or []) for k in keys)
    ok = bool(keys) and len(keys) == 3 and tok_eq and lh_eq
    ok_all &= ok
    print(f"{what}: {a} vs {b} (+ boundary): {len(keys)} prompts, tokens equal {tok_eq}, logits hashes equal {lh_eq} "
          f"({nchk} checks) -> {'PASS' if ok else 'FAIL'}")
print(f"STALE-SLOT CHECK: {'PASS: every 45 GB number stands' if ok_all else 'FAIL: tell the orchestrator at once'}")
PYEOF
say "stale-slot check end (verdict above)"
