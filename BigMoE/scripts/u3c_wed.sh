#!/bin/bash
# WED_TESTS 5b, Breakthrough's P1 NARROWED TO CODE (Breakthrough Wed 07:26, FLOOR's 42 held-out windows, SCOREBOARD 45fb226;
# ~/Breakthrough/WED_PLAN.md 962b370: pmtc costs on multi, z +2.78; len holds on code, cached bytes x0.196 vs u8+t3's
# x0.272 [P]). U3C in len's form (R57_FAST=U3C + R57_U3C_LEN=1, built in 0ea5b8b): on drafter-unsure rows keep ranks 0-3
# plus the ranks 4-9 whose expert is warm for this request (warm = the prompt's own prefill footprint when the prompt has
# >= 128 tokens, else the footprint OR cap45 share >= 0.05) or in the check's union; sure rows = U3. Code requests only:
# chat keeps U3 (the prompts here are all code, so no domain switch is needed for the test). FAST: tokens differ by design;
# EvalSuite's task results decide. Needs 5a first (today's u3n == u3c: the copied OOO path exact).
# Arms (one ticket, u3_smoke's 12 HumanEval prompts, CAP 256, one C for every arm: C 224 if NO-SWAP >= 44 GB, else C 160):
#   u3n2         the null control again (R57_FAST=U3, R57_U3_THR=2, R57_U3_FORCE=1) on today's ext/build, which item 0
#                rebuilt after 5a: its tokens must equal 5a's exact u3c_160 on all 12 prompts, else stop
#   u3b / u3b2   R57_FAST=U3, the baseline (A-B-A-B: back-to-back passes favour the later one's warmer page cache)
#   u3l / u3l2   R57_FAST=U3C + R57_U3C_LEN=1
#   u3ld         U3C len on HumanEval/68 (367 tokens), /16 (103: the < 128 branch), /91 (144), CAP 128, with
#                R57_U3C_DUMP: Breakthrough's mask check (dm_u3c_replay.py). The dump costs time: not a timing arm.
# Line (pre-registered Wed 4 Nov 07:4x; the orchestrator / Breakthrough's line with chat -> code, since P1 is code-only now):
#   PASS = code SSD MB per kept token (geo-mean over the 12 prompts of mean(u3l, u3l2) / mean(u3b, u3b2)) <= 0.85 AND steady
#   tok/s (tpsw, same form) >= 1.00; KILL = the MB ratio > 0.95 OR tok/s < 0.97; else BETWEEN. Then EvalSuite same-scores.
# usage: inside turn45 (HAVE_LOCK=1 KEEP_LOCK=1); freezes its own copies (data/r57/snap/*_u3c.*) -> data/u3c/u3c.log,
#        data/u3c/u3c_report.txt, data/u3c/dump_<C>/
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
mkdir -p data/u3c
LOG="$HOME/BigMoE/data/u3c/u3c.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
PY="$HOME/BigMoE/.venv/bin/python"
# the orchestrator Wed 07:5x: after the morning block, Breakthrough's screen 4 (NewIdeas', ~20 min, shared) goes first; 5b
# starts only after Breakthrough's "screen 4 done" (rm data/u3c/HOLD then) and only if it still ends by 12:55
[ -e data/u3c/HOLD ] && { say "HOLD (the orchestrator 07:5x: Breakthrough's screen 4 first): 5b not started"; exit 4; }
# 5a first: today's u3n and u3c token files, 12 prompts each, equal on every token
$PY -I - >> "$LOG" 2>&1 <<'EOF' || { say "5a has not passed today (u3n != u3c, a file missing or older than 06:00): STOP"; exit 2; }
import json, os, sys, time
t0 = time.mktime(time.strptime(time.strftime("%Y-%m-%d") + " 06:00", "%Y-%m-%d %H:%M"))
f = {a: f"data/r57/{a}_160_tokens.json" for a in ("u3c", "u3n")}
if not all(os.path.exists(p) and os.path.getmtime(p) > t0 for p in f.values()):
    sys.exit(1)
a, b = (json.load(open(f[k])) for k in ("u3c", "u3n"))
bad = [k for k in a if a[k].get("gate+t") != b.get(k, {}).get("gate+t")]
print(f"5a check: u3n vs u3c (today, C 160): {len(a)} prompts, differ: {bad}")
sys.exit(1 if bad or len(a) < 12 else 0)
EOF
cp scripts/r57_screen.py data/r57/snap/r57_screen_u3c.py && cp scripts/c45_runner.sh data/r57/snap/c45_runner_u3c.sh || exit 2
RUN="/bin/bash data/r57/snap/c45_runner_u3c.sh"
export C45_SCREEN=data/r57/snap/r57_screen_u3c.py C45_DEADLINE=${U3C_DEADLINE:-2400} C45_OUT="$HOME/BigMoE/data/u3c"
a=$(noswap); CC=160; [ "$a" -ge 44 ] && CC=224
K="humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91,humaneval/HumanEval/6,humaneval/HumanEval/79"
K="$K,humaneval/HumanEval/19,humaneval/HumanEval/161,humaneval/HumanEval/17,humaneval/HumanEval/56,humaneval/HumanEval/141"
K="$K,humaneval/HumanEval/101,humaneval/HumanEval/63"
U="R57_FAST=U3,R57_REFTOK="
L="R57_FAST=U3C,R57_U3C_LEN=1,R57_REFTOK="
say "5b U3C len start (pid $$, C $CC at NO-SWAP $a GB)"
# the gate again on today's ext/build (item 0 rebuilt it after 5a ran): the null control u3n2 (U3 with no sure rows) must
# give 5a's exact u3c_160 tokens on all 12 prompts (exact tokens do not depend on C), else the U3 path is broken: stop
R57_KEYS=$K MG=${U3C_MG:-4} C45_ALARM=900 C45_REFTOK="$HOME/BigMoE/data/r57/u3c_160_tokens.json:gate+t" \
  C45_ARMS="u3n2_$CC:$CC:R57_FAST=U3,R57_U3_THR=2,R57_U3_FORCE=1" $RUN
$PY -I - "$CC" >> "$LOG" 2>&1 <<'EOF' || { say "u3n2 != 5a's u3c_160 on today's ext/build: 5b STOPPED"; exit 3; }
import json, sys
a = json.load(open("data/r57/u3c_160_tokens.json")); b = json.load(open(f"data/r57/u3n2_{sys.argv[1]}_tokens.json"))
bad = [k for k in a if a[k].get("gate+t") != b.get(k, {}).get("gate+t")]
print(f"gate: u3n2 (C {sys.argv[1]}, today's ext/build) vs 5a's u3c_160: {len(a)} prompts, differ: {bad}")
sys.exit(1 if bad or len(a) < 12 else 0)
EOF
R57_KEYS=$K MG=${U3C_MG:-4} C45_ALARM=900 C45_ARMS="u3b_$CC:$CC:$U u3l_$CC:$CC:$L u3b2_$CC:$CC:$U u3l2_$CC:$CC:$L" $RUN
say "arms rc=$?"
R57_KEYS=humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91 C45_CAP=128 MG=${U3C_MG:-4} C45_ALARM=600 \
  C45_ARMS="u3ld_$CC:$CC:$L,R57_U3C_DUMP=$HOME/BigMoE/data/u3c/dump_$CC" $RUN
say "dump arm rc=$? ($(ls data/u3c/dump_$CC 2>/dev/null | wc -l | tr -d ' ') files in data/u3c/dump_$CC)"
$PY -I - "$CC" >> "$LOG" 2>&1 <<'EOF'
import json, math, sys
C = sys.argv[1]; R = "data/r57"
runs = {}
for t in ("u3b", "u3l", "u3b2", "u3l2"):
    try:
        runs[t] = {r["key"]: r for r in json.load(open(f"{R}/{t}_{C}_runs.json"))}
    except OSError:
        runs[t] = {}
m = lambda ts, k, f: [runs[t][k][f] for t in ts if k in runs[t]]
# a heat pause inside a prompt (the run's "bad" says PAUSED) is not U3C's effect: that run is not timed (its bytes and kept
# tokens still count); the prompt is timed on the other run of that side, or left out of the tok/s mean
mt = lambda ts, k: [runs[t][k]["tpsw"] for t in ts if k in runs[t] and "PAUSED" not in runs[t][k].get("bad", "")]
g = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")
mean = lambda xs: sum(xs) / len(xs)
L = [f"5b U3C len on code vs U3 at C {C} (data/u3c; U3C = R57_FAST=U3C + R57_U3C_LEN=1)"]
rb, rt, rk, nop = [], [], [], []
keys = sorted(set().union(*[set(v) for v in runs.values()]))
for k in keys:
    b, l = m(("u3b", "u3b2"), k, "mb"), m(("u3l", "u3l2"), k, "mb")
    bt, lt = mt(("u3b", "u3b2"), k), mt(("u3l", "u3l2"), k)
    bk, lk = m(("u3b", "u3b2"), k, "kept"), m(("u3l", "u3l2"), k, "kept")
    if not (b and l):
        continue
    x, z = mean(l) / mean(b), mean(lk) / mean(bk)
    rb.append(x); rk.append(z)
    s = f"  {k}: SSD MB per kept {mean(b):.1f} -> {mean(l):.1f} (x{x:.3f}); kept per check x{z:.3f}; steady tok/s "
    if bt and lt:
        y = mean(lt) / mean(bt); rt.append(y)
        s += f"{mean(bt):.1f} -> {mean(lt):.1f} ({100 * (y - 1):+.1f} %)" + ("" if len(bt) == len(lt) == 2 else f" [{len(bt)} + {len(lt)} unpaused]")
    else:
        nop.append(k); s += "not timed (heat pause)"
    L.append(s)
GB, GT, GK = g(rb), g(rt), g(rk)
L.append(f"geo-mean: SSD MB per kept token x{GB:.3f} over {len(rb)} prompts, steady tok/s {100 * (GT - 1):+.1f} % over {len(rt)}, "
         f"kept per check x{GK:.3f}" + (f"; not timed: {nop}" if nop else ""))
v = ("PASS" if len(rb) == 12 and len(rt) >= 10 and GB <= 0.85 and GT >= 1.0 else "KILL" if GB > 0.95 or GT < 0.97
     else "BETWEEN")
L.append(f"Verdict (pre-registered): {v} (PASS: MB x <= 0.85 AND tok/s >= U3's, all 12 prompts with bytes and >= 10 timed; "
         f"KILL: MB x > 0.95 OR tok/s < -3 %; FAST, so EvalSuite's same-scores decide adoption)")
open("data/u3c/u3c_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
EOF
say "report rc=$? (data/u3c/u3c_report.txt)"
say "5b end"
