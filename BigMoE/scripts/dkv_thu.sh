#!/bin/bash
# Thu ~08:00, step 3 of night_thu.sh (the orchestrator Wed 22:1x: DKV's ~10 min token check takes LOO time): Leap's DKV
# (the drafter's copy-free window cache; 2f726fb / d080232, drop-in 43d3ecf; pinned copy scripts/dkv.py = Leap 23919fb)
# in the exact default (R57_PRESET=45 as the runner sets it), R57_DKV=1 + R57_DKV_ADD=1 (the additive mask: the faster
# form, and it runs DKV's own path too), on the 14 ref14 prompts at C 224, every token vs REF2-T.
# Line (pre-registered Wed 22:1x, before arming): PASS = every prompt's tokens == REF2-T (ref14 gate+t, prefix) AND the
# copy-free path was taken (DKV COUNTS fast > 0) -> DKV goes into R57_PRESET=45 after a same-run speed A/B (on / off) in
# the next exact slot; FAIL = any token differs -> bisect (DKV alone vs + ADD), nothing adopted; VOID = the arm did not
# finish all 14 or fast == 0 (DKV never engaged) -> re-run. No speed claim here (no DKV-off control in this ticket).
# usage: DKVT_DEADLINE=hhmm bash scripts/dkv_thu.sh (inside night_thu.sh) -> data/dkvt/dkvt.log, data/dkvt/dkvt_report.txt
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "${DKVT_ROOT:-$HOME/BigMoE}" || exit 1
mkdir -p data/dkvt
LOG="$PWD/data/dkvt/dkvt.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
swap_gb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}'; }
CC=${DKVT_C:-224}   # night_thu.sh picks 224 or 160 by NO-SWAP (tokens do not depend on C; Thu 00:1x)
if [ -z "$DKVT_REPORT_ONLY" ]; then
  cp scripts/r57_screen.py data/r57/snap/r57_screen_dkvt.py && cp scripts/c45_runner.sh data/r57/snap/c45_runner_dkvt.sh || exit 2
  export C45_SCREEN=data/r57/snap/r57_screen_dkvt.py C45_DEADLINE=${DKVT_DEADLINE:-0830} C45_OUT="$PWD/data/dkvt"
  export C45_SWAP0=$(swap_gb)
  K="humaneval/HumanEval/68,humaneval/HumanEval/91,humaneval/HumanEval/16,humaneval/HumanEval/6,math500/test/prealgebra/1640.json"
  K="$K,math500/test/algebra/694.json,math500/test/algebra/1282.json,gsm8k/708,ifeval/1220,ifeval/1561,multi/de3,multi/zh15"
  K="$K,agent/bash18,agent/text23"
  say "DKV token check start (pid $$, C $CC, deadline $C45_DEADLINE, swap $C45_SWAP0 GB)"
  R57_KEYS=$K MG=${DKVT_MG:-10} C45_ALARM=900 C45_REFTOK="$HOME/BigMoE/data/d45/ref14_tokens.json:gate+t" \
    C45_ARMS="dkvt_$CC:$CC:R57_DKV=1,R57_DKV_ADD=1" /bin/bash data/r57/snap/c45_runner_dkvt.sh
  say "arm rc=$?"
fi
"${DKVT_PY:-$HOME/BigMoE/.venv/bin/python}" -I - "$CC" "$HOME/BigMoE/data/d45/ref14_tokens.json" >> "$LOG" 2>&1 <<'EOF'
import ast, json, math, re, sys
C, REF = sys.argv[1], sys.argv[2]
g = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")
ref = json.load(open(REF))
try:
    runs = json.load(open(f"data/r57/dkvt_{C}_runs.json")); toks = json.load(open(f"data/r57/dkvt_{C}_tokens.json"))
except OSError:
    runs, toks = [], {}
try:
    cnt = re.findall(r"DKV COUNTS (\{[^}]*\})", open(f"data/dkvt/dkvt_{C}.txt").read())
except OSError:
    cnt = []
fast = sum(ast.literal_eval(c).get("fast", 0) for c in cnt)  # one dict per screen process (the runner may split a run)
tk = lambda v: (v.get("gate+t+pf") or v.get("gate+t") or [[]])[0]
d = [k for k, v in toks.items() if not tk(v) or tk(v) != ref[k]["gate+t"][0][:len(tk(v))]]
n = len(toks)
v = ("FAIL" if d else "VOID" if n < 14 or fast == 0 else "PASS")
tw = [r["tpsw"] for r in runs if "PAUSED" not in r.get("bad", "")]
L = [f"DKV token check (data/dkvt; Thu morning ticket; R57_DKV=1 + R57_DKV_ADD=1 in the exact default, C {C}, 14 ref14 prompts)",
     f"tokens == REF2-T: {n - len(d)} / {n}{' DIFFER ' + str(d) if d else ''}; DKV path counts: {', '.join(cnt) or 'none logged'}",
     f"geo steady {g(tw):.1f} tok/s over {len(tw)} (no DKV-off control here: not a speed number)",
     f"Verdict (pre-registered Wed 22:1x): {v} (PASS: every token == REF2-T AND fast > 0 -> R57_PRESET=45 after a same-run "
     f"speed A/B; FAIL: bisect DKV vs + ADD; VOID: < 14 prompts or DKV never engaged -> re-run)"]
open("data/dkvt/dkvt_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
EOF
say "report rc=$? (data/dkvt/dkvt_report.txt)"
say "DKV token check end"
