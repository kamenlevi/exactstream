#!/bin/bash
> (a quote from the project owner, kept private)
# builds stacked on U3C, measured once against U3C on the 12 HumanEval at C 224, all arms on ext/build_l2 (one binary):
#   allin = U3C len (R57_FAST=U3C + R57_U3C_LEN=1) + D7b's 4-bit drafter on code (R57_D7B=code) + HC8-1P v4 (R57_HC8=1,
#           REF2_HC8_FAST=1: 8-bit trunk mixers, the one-pass read) + DENSE-N-RAM (R57_DNR=1) + hc_b (R57_HCB=1; idle under
#           HC8, which takes the HC read, kept for completeness);   allinf = allin + lever 2 stage 1 (BIGMOE_FC_F 4 / FS 2)
#   order u3c / allin / allinf / allin2 / u3c2 (A-B-C-B-A), CAP 256, FAST: no token check (R57_REFTOK=); then allind = the
#   dump for one map screen (R57_U3C_DUMP, HumanEval 68 / 16 / 91, CAP 128; not timed). Bisect only if it breaks.
# Reported (no pass line: a number for Kamen, and the parts of the cycle for the list toward ~50 ms per check): geo steady
# tok/s per arm and allin / u3c per prompt; median cycle, check, draft, SSD I/O, gate service, build ms per check (checks.json).
# usage: inside turn45 (HAVE_LOCK=1 KEEP_LOCK=1) -> data/allin/allin.log, data/allin/allin_report.txt (~15 min, MEM ~42 GB)
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
mkdir -p data/allin
LOG="$HOME/BigMoE/data/allin/allin.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
swap_gb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}'; }
cp scripts/r57_screen.py data/r57/snap/r57_screen_allin.py && cp scripts/c45_runner.sh data/r57/snap/c45_runner_allin.sh \
  && cp scripts/r57_screen_rlx.py data/r57/snap/r57_screen_allinr.py || exit 2
RUN="/bin/bash data/r57/snap/c45_runner_allin.sh"
export C45_SCREEN=data/r57/snap/r57_screen_allin.py C45_DEADLINE=${ALLIN_DEADLINE:-2250} C45_OUT="$HOME/BigMoE/data/allin"
export C45_SWAP0=$(swap_gb)
CC=${ALLIN_C:-224}
K="humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91,humaneval/HumanEval/6,humaneval/HumanEval/79"
K="$K,humaneval/HumanEval/19,humaneval/HumanEval/161,humaneval/HumanEval/17,humaneval/HumanEval/56,humaneval/HumanEval/141"
K="$K,humaneval/HumanEval/101,humaneval/HumanEval/63"
U="BIGMOE_EXT_BUILD=build_qom,R57_FAST=U3C,R57_U3C_LEN=1,R57_REFTOK="   # one binary for every arm (QOM + lever 2 inside)
# Wed 15:5x (the orchestrator agreed): the all-in is QOM-based (by day at C 160: QOM all-in +26.3 % vs U3C, bytes x0.77):
QA="BIGMOE_EXT_BUILD=build_qom,R57_REFTOK=,R57_FAST=QOM,R57_D17=1,R57_D7B=code,R57_HC8=1,REF2_HC8_FAST=1,R57_DNR=1,R57_HCB=1"
QA="$QA,REF2_SMALL_NATIVE=1,R57_D7BNAX=1,REF2_OOO_1C=1"   # + Leap def104e: D7b on qmm16c (-2 ms per code check; smoke 16:18 OK)                             # Leap's small-call native qmm (FAST, -1.15 ms); no REF2_OOO_1C (QOM's
Q8="BIGMOE_EXT_BUILD=build_qom,R57_REFTOK=,R57_FAST=QOM,R57_D17=1,R57_D8NAX=1,R57_HC8=1,REF2_HC8_FAST=1,R57_DNR=1"
Q8="$Q8,R57_HCB=1,REF2_SMALL_NATIVE=1,REF2_OOO_1C=1"   # + one-call OOO, QOM-aware (Leap 00961fc;
# smoke 17:09: ooo_1c_qom 432, qom_opt 41K)                 # qall8 = qall with Leap's 8-bit NAX drafter (2945bf7) in place of D7b
QF="$QA,BIGMOE_FC_F=4,BIGMOE_FC_FS=2,R57_ARMS=gate+t+pf,BIGMOE_PRED_LINES=4"   # -2 slots); qallf = + lever 2 (+pf posts the forecast)
say "all-in FAST start (pid $$, C $CC, swap $C45_SWAP0 GB)"
R57_KEYS=$K MG=${ALLIN_MG:-4} C45_ALARM=900 \
  C45_ARMS="u3c_$CC:$CC:$U qall_$CC:$CC:$QA qallf_$CC:$CC:$QF u3c2_$CC:$CC:$U" $RUN
say "arms rc=$?"
# Wed 19:1x (the orchestrator 18:0x / 18:4x: heat ran the map batch late, so the 20:20 by-day ticket was stopped and its
# relaxed check folded in here): MoonShots' relaxed acceptance (32edb12) on its own screen copy (scripts/r57_screen_rlx.py
# = the frozen screen + 7 asserted edits; the frozen screen itself is untouched):
#   rx0   = flag off, the exact default, 2 prompts CAP 64: every token == REF2-T (ref14 run 0 prefix), else no qallr;
#   qallr = qall + R57_RELAX_DELTA=0.5 (+ R57_RELAX_LOG) on the 12 HumanEval: FAST, quality = EvalSuite's RELAX verdict.
# qall8 (Q8, Leap's 8-bit NAX drafter) moves to the next slot: the window holds five timed arms, and l2_wed.sh follows.
C45_SCREEN=data/r57/snap/r57_screen_allinr.py R57_KEYS=humaneval/HumanEval/91,gsm8k/708 MG=3 C45_ALARM=600 C45_CAP=64 \
  C45_REFTOK="$HOME/BigMoE/data/d45/ref14_tokens.json:gate+t" C45_ARMS="rx0_$CC:$CC" $RUN
RX=$(.venv/bin/python -I - "$CC" 2>&1 <<'EOF'
import json, sys
C = sys.argv[1]
try:
    rs = json.load(open(f"data/r57/rx0_{C}_runs.json")); tk = json.load(open(f"data/r57/rx0_{C}_tokens.json"))
except OSError as e:
    print(f"FAIL no run files ({e})"); sys.exit()
ref = json.load(open("data/d45/ref14_tokens.json"))
tok = lambda v: (v.get("gate+t") or [[]])[0]                      # [run][token]: run 0, a prefix of ref14's run 0
d = [k for k, v in tk.items() if not tok(v) or tok(v) != ref[k]["gate+t"][0][:len(tok(v))]]
n = sum(len(tok(v)) for v in tk.values())
print(("OK" if len(rs) == 2 and n > 0 and not d else "FAIL") + f": runs {len(rs)}, {n} tokens, differ {d}")
EOF
)
say "relax flag-off check: $RX"
if [ "${RX:0:2}" = "OK" ]; then
  C45_SCREEN=data/r57/snap/r57_screen_allinr.py R57_KEYS=$K MG=${ALLIN_MG:-4} C45_ALARM=900 \
    C45_ARMS="qallr_$CC:$CC:$QA,R57_RELAX_DELTA=0.5,R57_RELAX_LOG=$HOME/BigMoE/data/allin/relax_accepts.jsonl" $RUN
  say "relaxed arm rc=$?"
else
  say "relaxed arm NOT run (the flag-off check failed)"
fi
# (no dump arm: QOM's quality is the map's screen 6 + EvalSuite at the end)
.venv/bin/python -I - "$CC" "${RX:-not run}" >> "$LOG" 2>&1 <<'EOF'
import json, math, statistics as S, sys
C = sys.argv[1]
g = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")
T = ("u3c", "qall", "qallf", "qallr", "u3c2")
runs, chk = {}, {}
for t in T:
    try:
        runs[t] = {r["key"]: r for r in json.load(open(f"data/r57/{t}_{C}_runs.json"))}
        chk[t] = json.load(open(f"data/r57/{t}_{C}_checks.json"))
    except OSError:
        runs[t], chk[t] = {}, []
ok = lambda t, k: k in runs[t] and "PAUSED" not in runs[t][k].get("bad", "")
L = [f"ALL-IN FAST, code, C {C} (data/allin; ext/build_qom): u3c = U3C len; qall = QOM + DRAFT17 + D7b code (on NAX) + "
     f"HC8-1P v4 + DENSE-N-RAM + hc_b + SMALL_NATIVE + OOO-1C; qallf = qall + lever 2 (FC 4 / 2); qallr = qall + relaxed "
     f"acceptance 0.5 nat (MoonShots 32edb12, own screen copy; flag-off check rx0: {sys.argv[2]}). Order u3c / qall / qallf "
     f"/ u3c2, then qallr. FAST: tokens differ by design."]
F = ("ck", "drf", "io", "svc", "wt", "bld")
for t in T:
    rs = runs[t]
    if not rs:
        L.append(f"{t:7s} MISSING"); continue
    tw = [r["tpsw"] for k, r in rs.items() if ok(t, k)]
    md = {f: S.median([x for e in chk[t] if not e.get("paused") for x in e.get(f, [])[1:]] or [float("nan")]) for f in F}
    pf = lambda x: sum(r.get("pf", {}).get(x, 0) for r in rs.values())
    L.append(f"{t:7s} geo steady {g(tw):6.1f} tok/s over {len(tw)}; kept per check {g([r['kept'] if isinstance(r['kept'], (int, float)) else max(1e-9, S.mean(r['kept'])) for r in rs.values()]):.2f}; "
             f"SSD MB per kept {g([r['mb'] for r in rs.values()]):.1f}; lifetime max {max(r['mem'][-1] for r in rs.values()):.1f} GB; "
             "medians per check (ms): " + ", ".join(f"{f} {md[f]:.1f}" for f in F) +
             (f"; forecast reads {pf('fc_issued')}, used {pf('fc_hits')}" if pf("fc_issued") else "") +
             (f"; QOM optional uses {pf('qom_opt')}, not read {pf('qom_skip')}" if pf("qom_opt") else ""))
    rl = [r["rlx"] for r in rs.values() if r.get("rlx")]
    if rl:
        L[-1] += (f"; relaxed accepts {sum(x['relaxed'] for x in rl)} in {sum(x['rows'] for x in rl)} rows "
                  f"(kept {sum(x['k'] for x in rl)} vs {sum(x['kx'] for x in rl)} exact)")
for side, ta, tb in (("qall", ("u3c", "u3c2"), ("qall",)), ("qallf", ("u3c", "u3c2"), ("qallf",)), ("qallr", ("u3c", "u3c2"), ("qallr",))):
    rat = []
    for k in runs["u3c"] or runs["u3c2"]:
        a = [runs[t][k]["tpsw"] for t in ta if ok(t, k)]; b = [runs[t][k]["tpsw"] for t in tb if ok(t, k)]
        if a and b:
            rat.append((sum(b) / len(b)) / (sum(a) / len(a)))
    L.append(f"{side} vs u3c: {100 * (g(rat) - 1):+.1f} % geo steady over {len(rat)} prompts" if rat else f"{side} vs u3c: n/a")
def pr(a, b):   # per prompt, geo: b / a
    r = [runs[b][k]["tpsw"] / runs[a][k]["tpsw"] for k in runs[a] if ok(a, k) and ok(b, k)]
    return f"{100 * (g(r) - 1):+.1f} % over {len(r)}" if r else "n/a"
L.append(f"relaxed 0.5 on the all-in, qallr vs qall: {pr('qall', 'qallr')} (qallr runs last: the order effect, u3c2 vs "
         f"u3c, is {pr('u3c', 'u3c2')})")
for k in runs["u3c"]:
    v = [f"{t} {runs[t][k]['tpsw']:.1f}" for t in T if ok(t, k)]
    L.append(f"  {k}: " + ", ".join(v))
open("data/allin/allin_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
EOF
say "report rc=$? (data/allin/allin_report.txt)"
say "all-in FAST end"
