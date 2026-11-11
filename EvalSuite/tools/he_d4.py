"""HumanEval+-only decision 4 (docs/PREREG_t80ck1u_2026-11-07.md): tools/d4x.py's rule on the core 60 alone, for arms that
decode no math / IFEval rows (d4x needs all three sets). Rows without text get it from their ids (tools/rows_text.py).
  .venv/bin/python tools/he_d4.py <arm run> <reference run> [<information reference> ...]
Last line `HE_D4: PASS | FAIL | DIFFERS (HIGHER)` against the first reference; exit 0 PASS / DIFFERS, 1 FAIL, 2 rows missing."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from evalsuite import decode_report as dr
import d4x, rows_text as rt


def load(run):
    rows = dr.load(run) or {}
    for r in rows.values():
        if r.get("text") is None:
            r["text"] = rt.text(r["completion_ids"])
    return rows


def main(arm, ref, *info):
    its = {k: it for k, it in dr.items().items() if it.get("set") == "humaneval"}
    A = load(arm)
    out = None
    for i, r in enumerate((ref,) + info):
        R = load(r)
        keys = sorted(k for k in its if k in A and k in R)
        if len(keys) < 60:
            print(f"humaneval: {len(keys)} of 60 rows in both {arm} and {r}")
            if i == 0:
                return 2
            continue
        v = d4x.line(f"humaneval vs {r}" + ("" if i == 0 else " (information)"),
                     dr.score_humaneval({k: A[k] for k in keys}, its), dr.score_humaneval({k: R[k] for k in keys}, its), keys)
        print(f"  token-identical to {r}: {sum(A[k]['completion_ids'] == R[k]['completion_ids'] for k in keys)} of {len(keys)}")
        out = v if i == 0 else out
    print(f"HE_D4: {out}")
    return 1 if out == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
