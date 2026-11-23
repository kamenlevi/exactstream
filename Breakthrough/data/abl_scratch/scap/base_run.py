"""Usage (BigMoE venv): python run_map.py <name> [<name> ...] [--max 12] [--skip-done] | report
Names: baseline, gate, any dm_ablations name (or dm_ablations.ablation's moe_reuse_{prev|lag<k>|chain}_L<a>_<b>),
ctx512 / ctx2048. Several names run in ONE process (one model load).
WIRED (Breakthrough, Mon 2026-11-02): 'gate' checks baseline.json against D-02a's stored top-1 (>= 99.9 %) and stops
the process (exit 3) if it fails, so nothing after it runs on a wrong wiring. --skip-done skips names whose
out/<name>.json exists (a restarted runner resumes). ctxN re-runs the baseline with the prompt cut to its last N tokens
(context-length sensitivity: answer positions keep their text, the model sees less prefix).
Night 2: every name is resolved before the model loads (a typo stops at once), and a run that raises is logged and
skipped so the rest of the ticket still runs."""
import sys, os, time, traceback
import dm_common as C
from dm_ablations import ABLATIONS, ablation
args = [a for a in sys.argv[1:]]
maxw = 12
if "--max" in args:
    i = args.index("--max"); maxw = int(args[i + 1]); del args[i:i + 2]
skip = "--skip-done" in args
names = [a for a in args if not a.startswith("--")]
if names == ["report"]: C.report(); sys.exit(0)
for n in names:
    if n not in ("baseline", "gate", "capture", "xcapture") and not n.startswith("ctx"): ablation(n)
if skip:
    names = [n for n in names if n == "gate" or not os.path.exists(os.path.join(C.OUT, n + ".json"))]
print(f"[dm] runs: {names}", flush=True)
model, tok = C.load_model(); C.discover_layers(model); C.install()
windows = C.load_windows(maxw)
for name in names:
    t0 = time.time()
    if name == "gate":
        ag, ok = C.gate()
        if not ok: sys.exit(3)
        continue
    try:
        if name.startswith("ctx"):
            n = int(name[3:]); cut = []
            for dom, nm, ids, off, gen in windows:
                a = gen + 1 - off; start = max(0, a - n)
                cut.append((dom, nm, ids[start:], off + start, gen))
            C.measure(model, cut, name, lambda: None)
        elif name == "baseline": C.measure(model, windows, "baseline", lambda: None)
        elif name == "capture": C.measure(model, windows, "capture", lambda: C.CAPR.update(on=True))   # DAY9 routing
        elif name == "xcapture": C.measure(model, windows, "xcapture", lambda: C.CAPR.update(on=True, x=True))   # 11-02 shadows
        else: C.measure(model, windows, name, ablation(name))
    except Exception:
        C.clear(); C.REC["on"] = False
        print(f"[dm] {name} FAILED after {time.time() - t0:.0f}s:\n{traceback.format_exc()}", flush=True)
        continue
    print(f"[dm] {name} done in {time.time() - t0:.0f}s", flush=True)
C.report()
