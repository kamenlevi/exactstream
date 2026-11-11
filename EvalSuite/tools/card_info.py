"""Information lines of a card arm (docs/PREREG_card_gates_2026-11-08 "Information (no verdict)"), CPU only: per benchmark
and order, the arm's and NAX6-N's scores (IFBench loose, the gate's metric, and strict); GPQA by domain; LCB by difficulty
and platform; rows token-identical to NAX6-N; agreement between the two orders; rows at the cap. LCB verdicts come from
results/lcb6_grades.jsonl (graded on a miss). A missing order is skipped.
  .venv/bin/python -B tools/card_info.py <tag> [<reference run>]"""
import collections, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from evalsuite import decode_report as dr
import rows_text as rt

SETS = ("ifbench", "gpqad", "lcb6")
CAP = {"ifbench": 1536, "gpqad": 2048, "lcb6": 2048}
DOMAIN = {"Organic Chemistry": "Chemistry", "Chemistry (general)": "Chemistry", "Inorganic Chemistry": "Chemistry",
          "Genetics": "Biology", "Molecular Biology": "Biology"}      # every other GPQA subject is Physics


def load(run):
    rows = dr.load(run) or {}
    for r in rows.values():
        if r.get("text") is None:
            r["text"] = rt.text(r["completion_ids"])
    return rows


def pct(v):
    return f"{100 * sum(v) / len(v):.1f}" if v else "-"


def main(tag, ref="ref2tnaxn.r6"):
    keys = [l.strip() for l in open(os.path.join(HERE, "..", "results", "card", "card_ref_keys.txt")) if l.strip()]
    its = dr.items()
    R = load(ref)
    A = {o: load(f"card_{tag}_{o}") for o in ("fwd", "rev")}
    A = {o: rows for o, rows in A.items() if rows}
    print(f"card {tag} vs {ref}: orders {sorted(A)}")
    for s in SETS:
        ks = [k for k in keys if k.startswith(s + "/") and k in R]
        sc = {"ref": dr.SCORERS[s]({k: R[k] for k in ks}, its)}
        for o, rows in A.items():
            sc[o] = dr.SCORERS[s]({k: rows[k] for k in ks if k in rows}, its)
        print(f"\n{s} (n {len(ks)}): " + "; ".join(f"{n} {pct([v[1] for v in sc[n].values()])} % ({len(sc[n])} rows)"
                                                for n in sc))
        if s == "ifbench":
            st = {"ref": dr.score_ifbench_strict({k: R[k] for k in ks}, its)}
            for o, rows in A.items():
                st[o] = dr.score_ifbench_strict({k: rows[k] for k in ks if k in rows}, its)
            print("  strict: " + "; ".join(f"{n} {pct([v[1] for v in st[n].values()])} %" for n in st))
        groups = {}
        if s == "gpqad":
            groups = {k: DOMAIN.get(its[k]["subject"], "Physics") for k in ks}
        elif s == "lcb6":
            groups = {k: f"{its[k]['difficulty']}" for k in ks}
            groups2 = {k: f"{its[k]['platform']}" for k in ks}
        for gmap in ([groups] if s == "gpqad" else [groups, groups2] if s == "lcb6" else []):
            for g in sorted(set(gmap.values())):
                gk = [k for k in ks if gmap[k] == g]
                print(f"  {g} (n {len(gk)}): " + "; ".join(
                    f"{n} {pct([sc[n][k][1] for k in gk if k in sc[n]])} %" for n in sc))
        for o, rows in A.items():
            both = [k for k in ks if k in rows]
            same = sum(rows[k]["completion_ids"] == R[k]["completion_ids"] for k in both)
            cap_a = sum(len(rows[k]["completion_ids"]) == CAP[s] for k in both)
            print(f"  {o}: token-identical to the reference {same} / {len(both)}; at the cap {cap_a} (reference "
                  f"{sum(len(R[k]['completion_ids']) == CAP[s] for k in both)})")
        if len(A) == 2:
            both = [k for k in ks if k in A["fwd"] and k in A["rev"]]
            tok = sum(A["fwd"][k]["completion_ids"] == A["rev"][k]["completion_ids"] for k in both)
            agree = sum(sc["fwd"][k][1] == sc["rev"][k][1] for k in both)
            print(f"  fwd vs rev: token-identical {tok} / {len(both)}; same score {agree} / {len(both)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
