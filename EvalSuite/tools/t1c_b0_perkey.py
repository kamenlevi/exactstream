"""The 1c-tiers B0 read per key (docs/RESULT_t1c_2026-11-16.md): B0 (lx45t1c1, tier 0) vs lx45cc1 forward vs the exact refs
on the 40 tier-0 code keys, graded by tools/lossy_d4.py's own pairs() (imported read-only). CPU only.
  .venv/bin/python tools/t1c_b0_perkey.py > results/lossygate/t1c_b0_perkey_2026-11-16.txt
Prints every key where the three disagree or B0's tokens differ from lx45cc1's, then the arm-only / reference-only
counts of each lossy run vs the exact refs."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import lossy_d4 as L


def main():
    keys, sets, refs = L.PANEL["tier0_code"], ("humaneval", "mbpp"), ["ref2tnax", "ref2tnaxn.r3", "ref2tnaxn.r2"]
    Pb, mb, Ab, _ = L.pairs("card_lx45t1c1_t0code", refs, keys, sets)
    Pc, mc, Ac, _ = L.pairs("card_lx45cc1_t1code", refs, keys, sets)
    print("missing", mb, mc)
    tot = {"b0_only": 0, "ref_only_b0": 0, "cc1_only": 0, "ref_only_cc1": 0}
    for s in sets:
        ks, sb, sr = Pb[s]; _, sc, _ = Pc[s]
        for k in ks:
            e, b, c = sr[k][1], sb[k][1], sc[k][1]
            same = Ab[k]["completion_ids"] == Ac[k]["completion_ids"]
            tot["b0_only"] += b == 1 and e == 0; tot["ref_only_b0"] += b == 0 and e == 1
            tot["cc1_only"] += c == 1 and e == 0; tot["ref_only_cc1"] += c == 0 and e == 1
            if not (e == b == c) or not same:
                print(f"  {k:24s} exact {e}  B0 {b}  lx45cc1 {c}  B0 tokens == lx45cc1: {same}")
    print(f"on the 40 tier-0 keys: B0 vs exact: arm only {tot['b0_only']} / reference only {tot['ref_only_b0']}; "
          f"lx45cc1 fwd vs exact: arm only {tot['cc1_only']} / reference only {tot['ref_only_cc1']}")


if __name__ == "__main__":
    main()
