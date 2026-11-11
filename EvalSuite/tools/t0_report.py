"""T0's verdict (docs/PREREG_t0_2026-11-03.md). CPU only.
  .venv/bin/python tools/t0_report.py   -> results/t0.txt; last line T0: PASS | FAIL (memory) | KILL | INCOMPLETE
Modes (decision 4 against REF): T0 = HumanEval+ from b12llff4t0c, IFEval and math from b12llff4t0h (theta 0.95);
T0m8 (information) = math from b12llff4t0c (0.8). Memory: every T0 process's lifetime peak <= 12.00 GB and <= the
control's peak + 0.05 GB (PMIN12's b12llff4 processes). Information: pass-2 rates, agreement with b12llff."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import relax_report as RR

RES = RR.RES
C08, C095 = "b12llff4t0c", "b12llff4t0h"
MODES = {"T0": dict(humaneval=C08, ifeval=C095, math=C095), "T0m8": dict(humaneval=C08, ifeval=C095, math=C08)}
WANT = {C08: {"math": 50, "humaneval": 60, "agent": 29}, C095: {"ifeval": 100, "math": 50}}


def rows(v):
    p = os.path.join(RES, v, "decode.jsonl")
    return {r["key"]: r for r in map(json.loads, open(p))} if os.path.exists(p) else {}


def peaks(v):
    p = os.path.join(RES, v, "chunks.jsonl")
    out = []
    if os.path.exists(p):
        for c in map(json.loads, open(p)):                # decode.run's chunk record: the process's lifetime peak
            out.append(c.get("footprint_peak_gb"))
    return [x for x in out if x is not None]


def main():
    L = []
    R = {v: rows(v) for v in WANT}
    have = {v: {s: sum(r["set"] == s for r in R[v].values()) for s in WANT[v]} for v in WANT}
    L.append(f"T0 rows: {have}")
    complete = all(have[v][s] >= n for v in WANT for s, n in WANT[v].items())
    ctl = max(peaks("b12llff4") or [11.96])
    pk = {v: peaks(v) for v in WANT}
    mem_ok = all(x <= 12.00 and x <= ctl + 0.05 for v in WANT for x in pk[v]) and all(pk[v] for v in WANT)
    L.append(f"memory: lifetime peaks {pk}; control's max {ctl}; line <= 12.00 and <= {ctl + 0.05:.2f}: "
             f"{'PASS' if mem_ok else 'FAIL'}")
    for v in WANT:
        st = [r.get("t0") for r in R[v].values() if r.get("t0")]
        for s in WANT[v]:
            ss = [r["t0"] for r in R[v].values() if r["set"] == s and r.get("t0")]
            rate = sum(x.get("pass2", 0) for x in ss) / max(sum(x.get("steps", 0) for x in ss), 1)
            L.append(f"{v} {s}: pass-2 rows {100 * rate:.1f} % of steps ({len(ss)} rows)")
    verdict = {}
    for m, spec in MODES.items():
        d = os.path.join(RES, f"t0_mode_{m}")
        os.makedirs(d, exist_ok=True)
        rr = [r for s, v in spec.items() for r in R[v].values() if r["set"] == s]
        with open(os.path.join(d, "decode.jsonl"), "w") as f:
            for r in rr:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if len(rr) == 210:
            ok4, txt = RR.score(d, ref="ref")
            ok4b, txtb = RR.score(d, ref="b12llff")
            verdict[m] = ok4
            L.append(f"\n{m} vs REF (the verdict):\n{txt}\n{m} vs b12llff (information):\n{txtb}")
        else:
            L.append(f"\n{m}: {len(rr)}/210 rows: not scored")
    if not complete or "T0" not in verdict:
        line = "T0: INCOMPLETE"
    elif not verdict["T0"]:
        line = "T0: KILL (decision 4)"
    elif not mem_ok:
        line = "T0: FAIL (memory; decision 4 PASS)"
    else:
        line = "T0: PASS"
    L.append(line + (f"; T0m8 (math at 0.8): decision 4 {'PASS' if verdict.get('T0m8') else 'FAIL'}" if "T0m8" in verdict else ""))
    text = "\n".join(L)
    open(os.path.join(RES, "t0.txt"), "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
