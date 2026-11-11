"""Decision 4 on one decode set (docs/PREREG_chat_gate_2026-11-08.md; tools/he_d4.py's form for any set): d4x's rule on the
set's core rows, for arms that decode only some sets (d4x needs all three). Rows without text get it from their ids.
  .venv/bin/python tools/set_d4.py <set> <arm run> <reference run> [<information reference> ...]
Last line `SET_D4 <set>: PASS | FAIL | DIFFERS (HIGHER)` against the first reference; exit 0 PASS / DIFFERS, 1 FAIL,
2 rows missing."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from evalsuite import decode_report as dr
import d4x, rows_text as rt

N = {"humaneval": 60, "math": 50, "ifeval": 100, "mgsm": 80, "ifbench": 150, "gpqad": 100, "lcb6": 131}   # card sets:
#   the gate keys of tools/card_keys.json (PREREG_card_gates_2026-11-08)


def load(run):
    rows = dr.load(run) or {}
    for r in rows.values():
        if r.get("text") is None:
            r["text"] = rt.text(r["completion_ids"])
    return rows


def main(s, arm, ref, *info):
    its = {k: it for k, it in dr.items().items() if it.get("set") == s}
    A = load(arm)
    out = None
    for i, r in enumerate((ref,) + info):
        R = load(r)
        keys = sorted(k for k in its if k in A and k in R)
        need = int(os.environ.get("SET_D4_N") or N[s])   # SET_D4_N: an extended set's full size (PREREG_card_gates am. 1)
        if len(keys) < need:
            print(f"{s}: {len(keys)} of {need} rows in both {arm} and {r}")
            if i == 0:
                return 2
            continue
        v = d4x.line(f"{s} vs {r}" + ("" if i == 0 else " (information)"),
                     dr.SCORERS[s]({k: A[k] for k in keys}, its), dr.SCORERS[s]({k: R[k] for k in keys}, its), keys)
        print(f"  token-identical to {r}: {sum(A[k]['completion_ids'] == R[k]['completion_ids'] for k in keys)} of {len(keys)}")
        out = v if i == 0 else out
    print(f"SET_D4 {s}: {out}")
    return 1 if out == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
