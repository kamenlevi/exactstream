#!/bin/bash
# WHICH LEVER OF THE NEW DEFAULT COSTS AGENT? (the orchestrator 10:1x; tonight's slot, beside the C 256 headline). At C 224
# the adopted R57_PRESET=45 (0365ada) ran agent -9.9 % vs the old default (bash18 -16.6 %, text23 -2.6 %; d256 09:49).
# Leave-one-out on the new default, C 224, CAP 256, 5 agent prompts (bash18, text23 + bash20, edit03, write12: one of each
# kind), every token vs the old default's own run (exact levers: all arms must give the same tokens), order
#   old  new  -vtier  -chat  -lever1  -K83  -keepalive  new2  old2   (Wed 14:0x: the RAM levers first; the day run was cut)
#   -chat     R57_CHATSW=0 (no TL_INFER / resident MTP head; agent drafts on DFlash2 either way)
#   -vtier    R57_VTIER= (no volatile tier)          -keepalive  BIGMOE_SSD_KEEPALIVE_US=0     -K83  K83_EXACT=0
#   -lever1   BIGMOE_JOINT=0, BIGMOE_COPY_PAR=0 (popped), BIGMOE_SPIN_COHERENT=0, BIGMOE_POST_SYS=0
# Line (pre-registered Wed 4 Nov 10:2x, before arming; geo-mean of steady tok/s over the 5 prompts, heat-paused runs not
# timed; new / old = means of their two passes):
#   - the regression R = new / old. CULPRIT = every lever X whose removal gives (new - X) / new >= +5 %, largest first.
#   - no culprit and R >= 0.97 -> NO AGENT REGRESSION at 5 prompts (the -9.9 % was bash18's spread): the default stays;
#     no culprit and R < 0.97 -> INTERACTION (no single lever): pairs next night, the default stays meanwhile.
#   - FIX if the culprit is the chat switch or the volatile tier (request-level settings): a per-domain switch, OFF for
#     agent requests, kept for chat / code / math (the screen knows the type; a server needs the request type, as U3C's
#     code switch). If it is an ext-level lever (keepalive, K-83, lever 1: process-wide, read at library load): drop it
#     from R57_PRESET=45 until per-lever arms on all 14 prompts show its gain elsewhere beats its agent cost.
# usage: inside turn45 (HAVE_LOCK=1 KEEP_LOCK=1) -> data/aloo/aloo.log, data/aloo/aloo_report.txt (~30 min quiet)
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
mkdir -p data/aloo
LOG="$HOME/BigMoE/data/aloo/aloo.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
swap_gb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}'; }
cp scripts/r57_screen.py data/r57/snap/r57_screen_aloo.py && cp scripts/c45_runner.sh data/r57/snap/c45_runner_aloo.sh || exit 2
RUN="/bin/bash data/r57/snap/c45_runner_aloo.sh"
export C45_SCREEN=data/r57/snap/r57_screen_aloo.py C45_DEADLINE=${ALOO_DEADLINE:-2400} C45_OUT="$HOME/BigMoE/data/aloo"
export C45_SWAP0=$(swap_gb)
CC=${ALOO_C:-224}
K=${ALOO_KEYS:-agent/bash18,agent/text23,agent/bash20,agent/edit03,agent/write12}
L1="BIGMOE_JOINT=0,BIGMOE_COPY_PAR=0,BIGMOE_SPIN_COHERENT=0,BIGMOE_POST_SYS=0"
say "agent leave-one-out start (pid $$, C $CC, swap $C45_SWAP0 GB)"
R57_KEYS=$K MG=${ALOO_MG:-4} C45_ALARM=900 C45_REFTOK= C45_ARMS="aold_$CC:$CC:R57_D45=0,R57_REFTOK= anew_$CC:$CC:R57_REFTOK= \
avt_$CC:$CC:R57_VTIER=,R57_REFTOK= achat_$CC:$CC:R57_CHATSW=0,R57_REFTOK= al1_$CC:$CC:$L1,R57_REFTOK= \
ak83_$CC:$CC:K83_EXACT=0,R57_REFTOK= aka_$CC:$CC:BIGMOE_SSD_KEEPALIVE_US=0,R57_REFTOK= anew2_$CC:$CC:R57_REFTOK= aold2_$CC:$CC:R57_D45=0,R57_REFTOK=" $RUN
say "arms rc=$?"
.venv/bin/python -I - "$CC" >> "$LOG" 2>&1 <<'EOF'
import json, math, re, subprocess, sys
C = sys.argv[1]
A = {"aold": "old", "anew": "new", "achat": "-chat switch", "avt": "-volatile tier", "aka": "-keepalive", "ak83": "-K83",
     "al1": "-lever 1", "anew2": "new (2nd)", "aold2": "old (2nd)"}
runs, toks = {}, {}
for t in A:
    try:
        runs[t] = {r["key"]: r for r in json.load(open(f"data/r57/{t}_{C}_runs.json"))}
        toks[t] = json.load(open(f"data/r57/{t}_{C}_tokens.json"))
    except OSError:
        runs[t], toks[t] = {}, {}
g = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")
ok = lambda t, k: k in runs[t] and "PAUSED" not in runs[t][k].get("bad", "")
keys = list(runs["aold"]) or list(runs["anew"])
L = [f"AGENT LEAVE-ONE-OUT at C {C} (data/aloo; 5 agent prompts; the new default = R57_PRESET=45, 0365ada)"]
ref = toks["aold"]
for t, name in A.items():
    rs = runs[t]
    if not rs:
        L.append(f"{name:15s} MISSING"); continue
    same = sum(toks[t].get(k, {}).get("gate+t") == ref.get(k, {}).get("gate+t") for k in keys)
    L.append(f"{name:15s} geo steady {g([rs[k]['tpsw'] for k in rs if ok(t, k)]):5.1f} tok/s, whole run "
             f"{g([rs[k]['tps'] for k in rs]):5.1f}; tokens == old {same} / {len(keys)}; " +
             ", ".join(f"{k.split('/')[-1]} {rs[k]['tpsw']:.1f}" for k in keys if k in rs))
def side(ts):                                            # per prompt: the mean over the side's unpaused runs
    out = {}
    for k in keys:
        v = [runs[t][k]["tpsw"] for t in ts if ok(t, k)]
        if v:
            out[k] = sum(v) / len(v)
    return out
old, new = side(("aold", "aold2")), side(("anew", "anew2"))
ratio = lambda a, b: g([a[k] / b[k] for k in keys if k in a and k in b])
R = ratio(new, old)
L.append(f"regression R = new / old: {100 * (R - 1):+.1f} % (geo over the prompts)")
gain = {}
for t in ("achat", "avt", "aka", "ak83", "al1"):
    x = side((t,))
    if x:
        gain[t] = ratio(x, new)
        L.append(f"  removing {A[t][1:]:14s}: {100 * (gain[t] - 1):+.1f} % vs new")
cul = sorted([t for t in gain if gain[t] >= 1.05], key=lambda t: -gain[t])
exact = all(toks[t].get(k, {}).get("gate+t") == ref.get(k, {}).get("gate+t") for t in A if runs[t] for k in keys)
miss = [t for t in A if not runs[t]]                     # Wed 14:0x: the day run stopped on the swap rule after one arm
log = open("data/aloo/c45.log").read().split(" c45 start ")[-1]          # this run's arms only
st = re.findall(r" run (\w+?)_\d+ \(NO-SWAP (\d+) GB, swap ([\d.]+) GB\)", log)
end = float(subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.split()[5].rstrip("M")) / 1024
sw = [float(x[2]) for x in st] + [end]
L.append("swap per arm (GB grown from its start to the next arm's start or the report; NO-SWAP at its start): " + ", ".join(
    f"{st[i][0]} {sw[i + 1] - sw[i]:+.2f} ({st[i][1]} GB)" for i in range(len(st))) + ("; SWAP GUARD fired" if "SWAP GUARD" in log else ""))
if not old or not new or any(t in miss for t in ("achat", "avt", "aka", "ak83", "al1")):
    v = f"INCOMPLETE, no verdict: {len(A) - len(miss)} of {len(A)} arms ran (missing: {', '.join(A[t] for t in miss)})"
elif not exact:
    v = "LOOK: an arm's tokens differ from the old default's (an exact lever is not exact here)"
elif cul:
    fix = {"achat": "per-domain switch: the chat switch OFF for agent requests", "avt": "per-domain switch: the volatile tier "
           "OFF for agent requests"}
    v = "CULPRIT " + ", ".join(f"{A[t][1:]} ({100 * (gain[t] - 1):+.1f} %) -> " + fix.get(t, "drop it from R57_PRESET=45 until "
        "per-lever arms on all 14 prompts") for t in cul)
else:
    v = "NO AGENT REGRESSION at 5 prompts (the default stays)" if R >= 0.97 else "INTERACTION (no single lever; pairs next)"
if miss and not v.startswith("INCOMPLETE"):
    v += f" (provisional: one pass of {', '.join(A[t] for t in miss)} missing)"
L.append(f"Verdict (pre-registered 10:2x): {v}")
open("data/aloo/aloo_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
EOF
say "report rc=$? (data/aloo/aloo_report.txt)"
say "agent leave-one-out end"
