#!/bin/bash
# BY DAY, bare minimum (Kamen's build rules 15:1x: apps stay open until 23:00, tests fit in the compute left; NO-SWAP ~29 GB):
# today's first combined FAST number at the small-pool config, C 96 (R57_PRESET=45, lifetime ~23 GB by c45's formula),
# under a share lock (scripts/guarded.sh), all arms on ext/build_l2. Shared GPU: rough, relative within this run only.
#   FAST, 12 HumanEval, CAP 256, A-B-C: u3cb (U3C len) / allinb (+ D7b code + HC8 v4 + DENSE-N-RAM + hc_b) / allinfb (+ lever 2)
#   EXACT lever 2, W1B's six, A-B: l2cb (FC off) / l2fb (FC 4 / 2), every token vs ref14 (REF2-T)
# Wed 15:13: C 96 failed at once (a 16-row check needs up to 128 slots per layer: 119 > 82 with the per-layer sizes);
# the smallest by-day config is C 128 with uniform pools: BD_C=128 BD_EXTRA=R57_LAYSZ=0,BIGMOE_TIER_CL=64 BD_L2=0 BD_MG=2.
# usage: NEED=27 WAITUNTIL=1700 LOCKWHY="BigMoE: ... MEM 26 GB" bash scripts/guarded.sh 2400 data/bd96/bd96.out /bin/bash scripts/bd96_wed.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
mkdir -p data/bd96
LOG="$HOME/BigMoE/data/bd96/bd96.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
swap_gb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}'; }
cp scripts/r57_screen.py data/r57/snap/r57_screen_bd96.py && cp scripts/c45_runner.sh data/r57/snap/c45_runner_bd96.sh || exit 2
RUN="/bin/bash data/r57/snap/c45_runner_bd96.sh"
export C45_SCREEN=data/r57/snap/r57_screen_bd96.py C45_DEADLINE=${BD_DEADLINE:-1800} C45_OUT="$HOME/BigMoE/data/bd96" KEEP_LOCK=1
export C45_SWAP0=$(swap_gb)
CC=${BD_C:-96}
K="humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91,humaneval/HumanEval/6,humaneval/HumanEval/79"
K="$K,humaneval/HumanEval/19,humaneval/HumanEval/161,humaneval/HumanEval/17,humaneval/HumanEval/56,humaneval/HumanEval/141"
K="$K,humaneval/HumanEval/101,humaneval/HumanEval/63"
K6="humaneval/HumanEval/68,humaneval/HumanEval/91,math500/test/prealgebra/1640.json,math500/test/algebra/694.json"
K6="$K6,math500/test/algebra/1282.json,gsm8k/708"
B="BIGMOE_EXT_BUILD=build_l2${BD_EXTRA:+,$BD_EXTRA}"   # BD_EXTRA: e.g. R57_LAYSZ=0,BIGMOE_TIER_CL=64 (C 128 by day)
U="$B,R57_FAST=U3C,R57_U3C_LEN=1,R57_REFTOK="
A="$U,R57_D7B=code,R57_HC8=1,REF2_HC8_FAST=1,R57_DNR=1,R57_HCB=1"
FC="BIGMOE_FC_F=4,BIGMOE_FC_FS=2"
say "by-day C $CC start (pid $$, swap $C45_SWAP0 GB)"
R57_KEYS=$K MG=${BD_MG:-4} C45_ALARM=900 C45_ARMS="u3cb_$CC:$CC:$U allinb_$CC:$CC:$A allinfb_$CC:$CC:$A,$FC" $RUN
say "FAST arms rc=$?"
[ "${BD_L2:-1}" = 1 ] && R57_KEYS=$K6 MG=${BD_MG:-4} C45_ALARM=900 C45_REFTOK="$HOME/BigMoE/data/d45/ref14_tokens.json:gate+t" \
  C45_ARMS="l2cb_$CC:$CC:$B l2fb_$CC:$CC:$B,$FC" $RUN
say "lever 2 arms rc=$?"
.venv/bin/python -I - "$CC" >> "$LOG" 2>&1 <<'EOF'
import json, math, statistics as S, sys
C = sys.argv[1]
g = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")
ref = json.load(open("data/d45/ref14_tokens.json"))
T = ("u3cb", "allinb", "allinfb", "l2cb", "l2fb")
runs, chk, toks = {}, {}, {}
for t in T:
    try:
        runs[t] = {r["key"]: r for r in json.load(open(f"data/r57/{t}_{C}_runs.json"))}
        chk[t] = json.load(open(f"data/r57/{t}_{C}_checks.json")); toks[t] = json.load(open(f"data/r57/{t}_{C}_tokens.json"))
    except OSError:
        runs[t], chk[t], toks[t] = {}, [], {}
ok = lambda t, k: k in runs[t] and "PAUSED" not in runs[t][k].get("bad", "")
F = ("ck", "drf", "io", "svc", "wt", "bld")
L = [f"BY DAY C {C} (data/bd96; shared GPU, apps open: rough, relative within this run; ext/build_l2)"]
for t in T:
    rs = runs[t]
    if not rs:
        L.append(f"{t:8s} MISSING"); continue
    tw = [r["tpsw"] for k, r in rs.items() if ok(t, k)]
    md = {f: S.median([x for e in chk[t] if not e.get("paused") for x in e.get(f, [])[1:]] or [float("nan")]) for f in F}
    pf = lambda x: sum(r.get("pf", {}).get(x, 0) for r in rs.values())
    ex = "" if not t.startswith("l2") else (" tokens == REF2-T %d / %d;" % (sum(toks[t][k]["gate+t"] == ref[k]["gate+t"] for k in toks[t]), len(toks[t])))
    L.append(f"{t:8s}{ex} geo steady {g(tw):6.1f} tok/s over {len(tw)}; kept per check {g([r['kept'] for r in rs.values()]):.2f}; "
             f"SSD MB per kept {g([r['mb'] for r in rs.values()]):.1f}; lifetime max {max(r['mem'][-1] for r in rs.values()):.1f} GB; "
             "medians per check (ms): " + ", ".join(f"{f} {md[f]:.1f}" for f in F) +
             (f"; forecast reads {pf('fc_issued')}, used {pf('fc_hits')} ({pf('fc_late')} landing), dropped {pf('fc_drops')}"
              if pf("fc_issued") else ""))
def ratio(a, b):
    r = [runs[b][k]["tpsw"] / runs[a][k]["tpsw"] for k in runs[a] if ok(a, k) and ok(b, k)]
    return f"{100 * (g(r) - 1):+.1f} % over {len(r)}" if r else "n/a"
L.append(f"allin vs U3C: {ratio('u3cb', 'allinb')}; + lever 2 on all-in: {ratio('allinb', 'allinfb')}; "
         f"lever 2 on the exact default: {ratio('l2cb', 'l2fb')} (geo steady, per prompt)")
open("data/bd96/bd96_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
EOF
say "report rc=$? (data/bd96/bd96_report.txt)"
say "by-day C $CC end"
