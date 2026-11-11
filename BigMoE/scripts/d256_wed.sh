#!/bin/bash
# The adopted combined default (R57_PRESET=45 since 0365ada, item 5: ~+19 % exact at C 160) at its own C: C 256, the 45 GB
# default's C since 11-02 (63.6 tok/s steady at a 41.7 GB lifetime peak, C50). Old (R57_D45=0) vs new, A-B-B-A (each
# side's mean position 2.5), item 5's 14 prompts, CAP 256, every token vs ref14 (REF2-T).
# Line (pre-registered Wed 4 Nov 09:0x, before the run): KEEP the new default if every token == REF2-T AND new / old
# geo-mean steady tok/s >= +3 % (item 5's ADOPT bar) AND no domain < -3 %; LOOK (per-lever arms at C 256 before the night,
# the default stays until then) if a token differs or < +1 %; else BETWEEN. Also reported: absolute geo tok/s per arm, SSD
# MB per kept token, lifetime footprint (the 45 GB line), heat-paused runs not timed.
# usage: inside turn45 (HAVE_LOCK=1 KEEP_LOCK=1); freezes its own copies -> data/d256/d256.log, data/d256/d256_report.txt
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
mkdir -p data/d256
LOG="$HOME/BigMoE/data/d256/d256.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
swap_gb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}'; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
cp scripts/r57_screen.py data/r57/snap/r57_screen_d256.py && cp scripts/c45_runner.sh data/r57/snap/c45_runner_d256.sh || exit 2
RUN="/bin/bash data/r57/snap/c45_runner_d256.sh"
export C45_SCREEN=data/r57/snap/r57_screen_d256.py C45_DEADLINE=${D256_DEADLINE:-1230} C45_OUT="$HOME/BigMoE/data/d256"
K="humaneval/HumanEval/68,humaneval/HumanEval/91,humaneval/HumanEval/16,humaneval/HumanEval/6,math500/test/prealgebra/1640.json"
K="$K,math500/test/algebra/694.json,math500/test/algebra/1282.json,gsm8k/708,ifeval/1220,ifeval/1561,multi/de3,multi/zh15"
K="$K,agent/bash18,agent/text23"
CC=${D256_C:-256}   # Wed 09:19: C 256 swapped +2.3 GB by day (~11 GB of sessions resident): D256_C=224 by day, 256 at night
export C45_SWAP0=$(swap_gb)   # the 2 GB swap rule from THIS run's start (09:1x took it at queue time)
say "C $CC confirmation start (pid $$, NO-SWAP $(noswap) GB, swap $C45_SWAP0 GB)"
R57_KEYS=$K MG=${D256_MG:-4} C45_ALARM=900 C45_REFTOK="$HOME/BigMoE/data/d45/ref14_tokens.json:gate+t" \
  C45_ARMS="d256a_$CC:$CC:R57_D45=0 d256n_$CC:$CC d256n2_$CC:$CC d256a2_$CC:$CC:R57_D45=0" $RUN
say "arms rc=$?"
.venv/bin/python -I - "$CC" >> "$LOG" 2>&1 <<'EOF'
import json, math, sys
C = sys.argv[1]
ref = json.load(open("data/d45/ref14_tokens.json"))
dom = lambda k: "code" if k.startswith("humaneval") else "math" if k.startswith(("math500", "gsm8k")) else \
    "agent" if k.startswith("agent") else "chat"
runs, toks = {}, {}
for t in ("d256a", "d256n", "d256n2", "d256a2"):
    try:
        runs[t] = {r["key"]: r for r in json.load(open(f"data/r57/{t}_{C}_runs.json"))}
        toks[t] = json.load(open(f"data/r57/{t}_{C}_tokens.json"))
    except OSError:
        runs[t], toks[t] = {}, {}
g = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")
ok = lambda t, k: k in runs[t] and "PAUSED" not in runs[t][k].get("bad", "")
L = [f"THE ADOPTED DEFAULT AT C {C} (data/d256): old = R57_PRESET=45 R57_D45=0, new = R57_PRESET=45 (0365ada)"]
bad = {t: [k for k in ref if toks[t].get(k, {}).get("gate+t") != ref[k]["gate+t"]] for t in toks}
for t, b in bad.items():
    rs = runs[t]
    tw = [rs[k]["tpsw"] for k in rs if ok(t, k)]
    L.append(f"{t:7s} tokens {len(ref) - len(b)} / {len(ref)} == REF2-T" + (f" DIFFER/MISSING {b}" if b else "") +
             (f"; geo steady {g(tw):.1f} tok/s over {len(tw)}; SSD MB per kept {g([rs[k]['mb'] for k in rs]):.1f} (geo); "
              f"lifetime footprint max {max(rs[k]['mem'][-1] for k in rs):.1f} GB" if rs else ""))
    if rs:                                               # the orchestrator 09:2x: absolute geo steady per domain, per arm
        L.append("        by domain (geo steady tok/s): " + ", ".join(
            f"{d} {g([rs[k]['tpsw'] for k in rs if dom(k) == d and ok(t, k)]):.1f}" for d in ("code", "chat", "math", "agent")))
for side, ts in (("old", ("d256a", "d256a2")), ("new", ("d256n", "d256n2"))):   # both passes of a side together
    rr = [(k, runs[t][k]) for t in ts for k in runs[t] if ok(t, k)]
    if rr:
        L.append(f"{side} (both passes): geo steady {g([r['tpsw'] for _, r in rr]):.1f} tok/s; " + ", ".join(
            f"{d} {g([r['tpsw'] for k, r in rr if dom(k) == d]):.1f}" for d in ("code", "chat", "math", "agent")) +
            f"; lifetime footprint max {max(r['mem'][-1] for _, r in rr):.1f} GB")
rat, per, nop = [], {}, []
for k in ref:
    a = [runs[t][k]["tpsw"] for t in ("d256a", "d256a2") if ok(t, k)]
    b = [runs[t][k]["tpsw"] for t in ("d256n", "d256n2") if ok(t, k)]
    if a and b:
        x = (sum(b) / len(b)) / (sum(a) / len(a)); rat.append(x); per.setdefault(dom(k), []).append(x)
        L.append(f"  {k}: old {sum(a) / len(a):.1f} -> new {sum(b) / len(b):.1f} tok/s steady ({100 * (x - 1):+.1f} %)")
    else:
        nop.append(k)
G = g(rat); D = {d: g(v_) for d, v_ in per.items()}
L.append("geo-mean steady tok/s new vs old: %+.1f %% over %d prompts; by domain: %s" % (100 * (G - 1), len(rat),
         ", ".join(f"{d} {100 * (v_ - 1):+.1f} %" for d, v_ in sorted(D.items()))) + (f"; not timed: {nop}" if nop else ""))
exact = not any(bad.values())
v = ("LOOK (a token differs or an arm is missing)" if not exact else f"BETWEEN (only {len(rat)} timed: re-run)"
     if len(rat) < len(ref) - 2 else "KEEP the new default" if G >= 1.03 and min(D.values()) >= 0.97 else "LOOK" if G < 1.01
     else "BETWEEN")
L.append(f"Verdict (pre-registered 09:0x): {v} (KEEP: every token == REF2-T AND >= +3 % AND no domain < -3 %; LOOK: a "
         f"token differs or < +1 %)")
open(f"data/d256/d256_report_{C}.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
EOF
say "report rc=$? (data/d256/d256_report_$CC.txt)"
say "C $CC confirmation end"
