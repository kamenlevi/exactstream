"""Fit12 K12 read-out (Tue 3 Nov 23:0x; orchestrator 22:5x; PLAN.md 'T3R2' [K] CHECK + K12). CPU only.
Usage: k12_verdict.py [DIR (/Users/kamen/Fit12/data/scr/t3r2)] [TAG_A (r47)] [TAG_B (k12)]
A = T3R2's own run (no extra syncs), B = the K12 run: the same first 4 prompts, arms and order, with mx.eval at every MoE
block's entry and exit (fit_t3.py F12_K12SYNC=1). Same requests and evictions, so every (prompt, arm) must be bitwise equal:
tokens (*_toks.json), the logits hash (lsha, the arm's first line in TAG.txt) and the per-check log (*.json). A's MAIN-pass
run is used: void1_* when the prompt was paused and re-run (a PAUSE changes timings, not bits; the re-run has another
history), else the live files. B must show the K12 lines (wrapper on, > 0 synced calls, still installed at the end).
Verdict: PASS = every comparable pair equal (>= 1 pair); FAIL = any pair differs ([K]-type stale reads in the unsynced
run, or bits that depend on timing); VOID = no comparable pair or the K12 sync not confirmed."""
import json, os, re, sys

D = sys.argv[1] if len(sys.argv) > 1 else "/Users/kamen/Fit12/data/scr/t3r2"
TA = sys.argv[2] if len(sys.argv) > 2 else "r47"
TB = sys.argv[3] if len(sys.argv) > 3 else "k12"
KEYS = ["156", "42", "68", "91"]
ARMS = ["t80ck2", "t80ck1um3", "t80ck1u"]


def txt(tag):
    f = os.path.join(D, f"{tag}.txt")
    return open(f).read() if os.path.exists(f) else ""


def lsha(t, key, arm):                                   # the arm's FIRST line for this prompt (the main pass)
    m = re.search(rf"HumanEval/{key} {arm} theta .*?\| lsha (\w+)", t)
    return m.group(1) if m else None


def load(tag, key, arm, x, main):
    base = f"{tag}_humaneval_HumanEval_{key}_{arm}{x}.json"
    for f in ((["void1_" + base] if main else []) + [base]):
        p = os.path.join(D, f)
        if os.path.exists(p):
            return json.load(open(p)), f
    return None, None


ta, tb = txt(TA), txt(TB)
on = "K12: mx.eval at every MoE block's entry and exit" in tb
m = re.search(r"K12: (\d+) synced MoE block calls; wrapper still installed: (\w+)", tb)
ncalls, still = (int(m.group(1)), m.group(2) == "True") if m else (0, False)
print(f"{TB}: finished {'] done' in tb}; K12 wrapper on {on}; synced MoE block calls {ncalls}; still installed at the end {still}"
      + ("; ERRORS: " + [l for l in tb.splitlines() if 'Error' in l or 'Traceback' in l][-1][:160]
         if any('Error' in l or 'Traceback' in l for l in tb.splitlines()) else ""))
same, diff, skip = [], [], []
for key in KEYS:
    for arm in ARMS:
        a_t, fa = load(TA, key, arm, "_toks", True)
        b_t, fb = load(TB, key, arm, "_toks", False)
        if a_t is None or b_t is None:
            skip.append(f"{key}/{arm} ({'A' if a_t is None else 'B'} missing)")
            continue
        a_l, _ = load(TA, key, arm, "", True)
        b_l, _ = load(TB, key, arm, "", False)
        la, lb = lsha(ta, key, arm), lsha(tb, key, arm)
        ok_t, ok_l, ok_c = a_t == b_t, la is not None and la == lb, a_l == b_l
        if ok_t and ok_l and ok_c:
            same.append(f"{key}/{arm}")
        else:
            fd = next((i for i, (x, y) in enumerate(zip(a_t, b_t)) if x != y), min(len(a_t), len(b_t)) if len(a_t) != len(b_t) else None)
            diff.append(f"{key}/{arm}: tokens {'=' if ok_t else '!= (first difference at ' + str(fd) + ')'}, lsha {la} vs {lb}, "
                        f"per-check log {'=' if ok_c else '!='} (A file {fa})")
print(f"equal ({len(same)}): {', '.join(same)}")
if diff:
    print("DIFFERENT:\n  " + "\n  ".join(diff))
if skip:
    print(f"not comparable: {', '.join(skip)}")
if not (on and ncalls > 0 and still):
    v = "VOID (the K12 sync is not confirmed in " + TB + ".txt)"
elif diff:
    v = f"FAIL: {len(diff)} of {len(same) + len(diff)} pairs differ with a sync at every MoE layer -> [K]-type stale reads (or timing-dependent bits) in the unsynced T3 run"
elif same:
    v = f"PASS: {len(same)} of {len(same)} (prompt, arm) pairs bitwise equal (tokens + logits + per-check log) with a sync at every MoE layer -> no stale reads in T3's calls on these prompts"
else:
    v = "VOID (no comparable pair)"
print(f"K12 VERDICT: {v}")
