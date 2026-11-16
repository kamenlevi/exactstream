"""The 1c-tiers unset-path check (PREREG_lossygate amendment 4: information, flagged if it fails): a base run on the new
snapshot against lx45cc1's run of the same order and split. Tokens must be identical on the sets outside QOM
(humanevalx 104, IFEval 100), as lx45cc1's two orders were; the QOM sets (humaneval, mbpp) are information. CPU only.
  .venv/bin/python tools/t1c_unset_path.py <run> [<reference run>]      # default reference: card_lx45cc1_t1code
Exit 0 = identical on every non-QOM key; 1 = a difference (flag it); 2 = rows missing."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
GATED, INFO = ("humanevalx", "ifeval"), ("humaneval", "mbpp")
N = {"humanevalx": 104, "ifeval": 100}


def rows(run):
    return {json.loads(l)["key"]: json.loads(l)["completion_ids"] for l in open(os.path.join(RES, run, "decode.jsonl"))}


def main(argv):
    if not argv:
        print(__doc__); return 2
    run, ref = argv[0], argv[1] if len(argv) > 1 else "card_lx45cc1_t1code"
    A, R = rows(run), rows(ref)
    ok, missing = True, False
    print(f"unset path: {run} vs {ref}")
    for s in GATED + INFO:
        ks = sorted(k for k in R if k.split("/")[0] == s)
        have = [k for k in ks if k in A]
        same = sum(A[k] == R[k] for k in have)
        diff = [k for k in have if A[k] != R[k]]
        line = f"  {s:10s} {same} / {len(have)} identical"
        if s in GATED:
            if len(have) != N[s]:
                missing = True; line += f" (MISSING: {N[s] - len(have)} of {N[s]} keys)"
            ok &= not diff
            line += "" if not diff else f"; DIFFER: {' '.join(diff[:8])}{' ...' if len(diff) > 8 else ''}"
        else:
            line += " (QOM set: information)"
        print(line)
    v = "INCOMPLETE" if missing else ("PASS" if ok else "FLAG")
    print(f"UNSET PATH: {v} (non-QOM sets {' + '.join(GATED)} must be identical)")
    return 2 if missing else (0 if ok else 1)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
