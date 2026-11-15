"""PREREG_lossygate amendment 2, step 1's unset-path check: the exact config's own tier-0 decode (with the REPLAY-capable
screen, R57_REPLAY unset) against the REF2-path exact references. Exact paths may part only at near-ties, so each key's
FIRST divergence must sit where the exact config's own margin (top-1 minus top-2, from step 2's merged REPLAY margins)
is below 0.5 nat. After a divergence the prefixes differ, so later positions are not compared. CPU only.
  .venv/bin/python tools/replay_unset_check.py <own run> <ref runs, comma list> <merged margins.json> <keys file>
Last line: `UNSET-PATH: PASS | FAIL (...)`; exit 0 / 1, 2 on missing rows."""
import json, os, sys

NEAR = 0.5


def rows(runs):
    out = {}
    for run in runs.split(","):                     # earlier runs win, as cert_report.load merges references
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", run, "decode.jsonl")
        for l in open(p):
            r = json.loads(l)
            out.setdefault(r["key"], r)
    return out


def main(own, refs, margins, keysf):
    keys = [k for k in open(keysf).read().split() if k]
    O, R, M = rows(own), rows(refs), json.load(open(margins))
    miss = [k for k in keys if k not in O or k not in R or k not in M]
    if miss:
        print(f"{len(miss)} keys without an own row, a reference row or margins: {miss[:5]}")
        print("UNSET-PATH: INCOMPLETE"); return 2
    same, near, far, unk = 0, [], [], []
    for k in keys:
        a, b = O[k]["completion_ids"], R[k]["completion_ids"]
        i = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), None)
        if i is None and len(a) == len(b):
            same += 1; continue
        if i is None:                               # one is a prefix of the other: the shorter one stopped (EOS / cap)
            i = min(len(a), len(b))
        m = M[k].get(str(i))
        (unk if m is None else near if m < NEAR else far).append((k, i, m))   # no margin there (e.g. past the own row's end)
    print(f"{same} of {len(keys)} keys token-identical to the references ({refs})")
    for k, i, m in near + far + unk:
        print(f"  {k}: first divergence at answer position {i}, the exact own margin "
              f"{'n/a' if m is None else f'{m:.3f}'} nat -> "
              f"{'near-tie' if (k, i, m) in near else 'NOT a near-tie' if (k, i, m) in far else 'UNKNOWN (no margin there)'}")
    if far:
        print(f"UNSET-PATH: FAIL ({len(far)} divergences away from a near-tie; the line is a margin < {NEAR} nat)")
        return 1
    if unk:
        print(f"UNSET-PATH: INCOMPLETE ({len(unk)} divergences without an own margin; read them by hand)")
        return 2
    print(f"UNSET-PATH: PASS ({same} identical; {len(near)} part at near-ties < {NEAR} nat)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(__doc__); sys.exit(2)
    sys.exit(main(*sys.argv[1:5]))
