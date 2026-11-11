#!/bin/bash
# Thu ~07:35, step 2 of night_thu.sh (the orchestrator Wed 21:3x): the C 224 FAST trio that the Wed 23:00 ticket could not
# run with Kamen's apps open (NO-SWAP ~41 GB; qallf needs ~47.4 GB at C 224 under peak + 4, scripts/eve_2300b.sh). Apps are
# closed in the morning ticket (MEM ~52 GB). Same arms and keys as scripts/allin_wed.sh, order u3ct / qallt / qallft:
#   u3ct = U3C len; qallt = QOM + DRAFT17 + D7b code (on NAX) + HC8-1P v4 + DENSE-N-RAM + hc_b + SMALL_NATIVE + OOO-1C;
#   qallft = qallt + lever 2 (forecast reads FC 4 / 2). 12 HumanEval, FAST (no token check), ~25 min.
# No u3c2 / relaxed arm here: the Wed 23:00 run has both at its own C. MG 10: each arm needs NO-SWAP >= 48.5 GB at its
# start (c45_runner: 38.0 + MG + 0.5 at C 224), else it is skipped, not swapped.
# Reported (no pass line: the C 224 number for Kamen): geo steady tok/s per arm, the ratios, medians per check.
# usage: C224T_DEADLINE=hhmm bash scripts/c224_thu.sh (inside night_thu.sh) -> data/c224t/c224t.log, data/c224t/c224t_report.txt
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "${C224T_ROOT:-$HOME/BigMoE}" || exit 1
mkdir -p data/c224t
LOG="$PWD/data/c224t/c224t.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
swap_gb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}'; }
CC=${C224T_C:-224}   # night_thu.sh picks C (and C224T_NOQF / C224T_MG) by NO-SWAP at its start (Thu 00:1x)
if [ -z "$C224T_REPORT_ONLY" ]; then
  cp scripts/r57_screen.py data/r57/snap/r57_screen_c224t.py && cp scripts/c45_runner.sh data/r57/snap/c45_runner_c224t.sh || exit 2
  export C45_SCREEN=data/r57/snap/r57_screen_c224t.py C45_DEADLINE=${C224T_DEADLINE:-0815} C45_OUT="$PWD/data/c224t"
  export C45_SWAP0=$(swap_gb)
  K="humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91,humaneval/HumanEval/6,humaneval/HumanEval/79"
  K="$K,humaneval/HumanEval/19,humaneval/HumanEval/161,humaneval/HumanEval/17,humaneval/HumanEval/56,humaneval/HumanEval/141"
  K="$K,humaneval/HumanEval/101,humaneval/HumanEval/63"
  U="BIGMOE_EXT_BUILD=build_qom,R57_FAST=U3C,R57_U3C_LEN=1,R57_REFTOK="
  QA="BIGMOE_EXT_BUILD=build_qom,R57_REFTOK=,R57_FAST=QOM,R57_D17=1,R57_D7B=code,R57_HC8=1,REF2_HC8_FAST=1,R57_DNR=1,R57_HCB=1"
  QA="$QA,REF2_SMALL_NATIVE=1,R57_D7BNAX=1,REF2_OOO_1C=1"
  QF="$QA,BIGMOE_FC_F=4,BIGMOE_FC_FS=2,R57_ARMS=gate+t+pf,BIGMOE_PRED_LINES=4"
  say "C $CC FAST trio start (pid $$, deadline $C45_DEADLINE, swap $C45_SWAP0 GB)"
  AR="u3ct_$CC:$CC:$U qallt_$CC:$CC:$QA qallft_$CC:$CC:$QF"
  [ -n "$C224T_NOQF" ] && AR="u3ct_$CC:$CC:$U qallt_$CC:$CC:$QA"   # no forecast-read arm when memory is short
  R57_KEYS=$K MG=${C224T_MG:-10} C45_ALARM=900 C45_ARMS="$AR" \
    /bin/bash data/r57/snap/c45_runner_c224t.sh
  say "arms rc=$?"
fi
"${C224T_PY:-$HOME/BigMoE/.venv/bin/python}" -I - "$CC" >> "$LOG" 2>&1 <<'EOF'
import json, math, statistics as S, sys
C = sys.argv[1]
g = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")
T = ("u3ct", "qallt", "qallft")
runs, chk = {}, {}
for t in T:
    try:
        runs[t] = {r["key"]: r for r in json.load(open(f"data/r57/{t}_{C}_runs.json"))}
        chk[t] = json.load(open(f"data/r57/{t}_{C}_checks.json"))
    except OSError:
        runs[t], chk[t] = {}, []
ok = lambda t, k: k in runs[t] and "PAUSED" not in runs[t][k].get("bad", "")
L = [f"C {C} FAST trio, code (data/c224t; Thu morning ticket, apps closed; ext/build_qom): u3ct = U3C len; qallt = the QOM "
     f"all-in (QOM + DRAFT17 + D7b code on NAX + HC8-1P v4 + DENSE-N-RAM + hc_b + SMALL_NATIVE + OOO-1C); qallft = qallt + "
     f"lever 2 (FC 4 / 2). Order u3ct / qallt / qallft. FAST: tokens differ by design."]
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
def pr(a, b):   # per prompt, geo: b / a
    r = [runs[b][k]["tpsw"] / runs[a][k]["tpsw"] for k in runs[a] if ok(a, k) and ok(b, k)]
    return f"{100 * (g(r) - 1):+.1f} % over {len(r)}" if r else "n/a"
L.append(f"qallt vs u3ct: {pr('u3ct', 'qallt')}; qallft vs u3ct: {pr('u3ct', 'qallft')}; lever 2 on the all-in, qallft vs "
         f"qallt: {pr('qallt', 'qallft')} (geo steady, per prompt; later arms run warmer: the order effect is the Wed 23:00 "
         f"run's u3c2 vs u3c)")
for k in runs["u3ct"]:
    v = [f"{t} {runs[t][k]['tpsw']:.1f}" for t in T if ok(t, k)]
    L.append(f"  {k}: " + ", ".join(v))
open("data/c224t/c224t_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
EOF
say "report rc=$? (data/c224t/c224t_report.txt)"
say "C $CC FAST trio end"
