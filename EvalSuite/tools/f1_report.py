"""The FAST batch's verdict (docs/PREREG_f1_2026-11-03.md, revisions 2-3): F1 + r8 + HC8-1P on its base.
  .venv/bin/python tools/f1_report.py <batch run> <control run> <core reference run> [<stage-2 reference run>]
  e.g. NAX6 base: ref2tnfh ref2tnax ref2tnax ref2tnax;  REF2-T base: ref2tfh ref2tf0 ref2t ref2t.r2
Gate 2: per row with >= 2 tokens, the batch ran fs4 on every eligible verify call (fatt_verify > 0,
verify_old_eligible 0), r8 (fatt_r8 > 0) and the 8-bit HC read (hc8_fast > 0), plus NAX6 on its base. The control
ran none of fs4 / r8 / HC8. Lines: decision 4 on the core 210 vs the core reference (tools/d4x.py, with stage 2), and
RULER 24 vs the control (at most 1 net item lost). Flags (no lines): agent agreement vs the floor's (floor vs ref);
copy breaks on the 11 edit turns. Last line `F1: PASS | FAIL | DIFFERS (HIGHER) | VOID | STAGE2 | INCOMPLETE`;
exit 0 PASS / DIFFERS, 1 FAIL, 5 VOID, 10 STAGE2, 2 rows missing."""
import json, os, sys
from difflib import SequenceMatcher
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from evalsuite import decode_report as dr
from evalsuite import data as D
import d4x

AT14 = os.path.join(HERE, "..", "results", "relax", "keys_at14.txt")


def calls(r):
    return r.get("ref2", {}).get("calls", {})


def gate2(B, Cn, nax):
    bad = []
    for k, r in B.items():
        if len(r["completion_ids"]) < 2:
            continue
        c = calls(r)
        need = {"fatt_verify": c.get("fatt_verify", 0) > 0, "verify_old_eligible == 0": c.get("verify_old_eligible", 0) == 0,
                "fatt_r8": c.get("fatt_r8", 0) > 0, "hc8_fast": c.get("hc8_fast", 0) > 0}
        if nax:
            need["nax6"] = c.get("nax6", 0) + c.get("nax6_ooo", 0) > 0
        miss = [n for n, ok in need.items() if not ok]
        if miss:
            bad.append((k, miss))
    for k, r in Cn.items():
        c = calls(r)
        on = [p for p in ("fatt_verify", "fatt_r8", "hc8_fast") if c.get(p)]
        if on:
            bad.append((k, ["control ran " + ", ".join(on)]))
    return bad


def ruler(B, Cn, its):
    keys = sorted(k for k in its if its[k].get("set") == "ruler" and k in B and k in Cn)
    if len(keys) < 24:
        return None, f"RULER: {len(keys)} of 24 rows in both"
    sb = dr.score_ruler({k: B[k] for k in keys}, its)
    sc = dr.score_ruler({k: Cn[k] for k in keys}, its)
    b = sum(1 for k in keys if sc[k][1] == 1 and sb[k][1] == 0)       # control only right
    c = sum(1 for k in keys if sc[k][1] == 0 and sb[k][1] == 1)       # batch only right
    ok = (b - c) <= 1
    return ("PASS" if ok else "FAIL"), (f"RULER (n 24): batch {sum(sb[k][1] for k in keys):.0f} vs control "
                                         f"{sum(sc[k][1] for k in keys):.0f} right, control only {b} / batch only {c}, "
                                         f"net lost {b - c} (line <= 1): {'PASS' if ok else 'FAIL'}")


def agreement(B, Cn, F, R, keys, label):
    have = [k for k in keys if k in B and k in Cn]
    ident = sum(B[k]["completion_ids"] == Cn[k]["completion_ids"] for k in have)
    sim = sum(SequenceMatcher(None, Cn[k]["completion_ids"], B[k]["completion_ids"], autojunk=False).ratio()
              for k in have) / max(len(have), 1)
    fh = [k for k in keys if k in F and k in R]
    fident = sum(F[k]["completion_ids"] == R[k]["completion_ids"] for k in fh)
    fsim = sum(SequenceMatcher(None, R[k]["completion_ids"], F[k]["completion_ids"], autojunk=False).ratio()
               for k in fh) / max(len(fh), 1)
    flag = len(have) == len(fh) and (ident < fident or sim < fsim)
    return flag, (f"agent {label} (n {len(have)}): identical {ident}, similarity {sim:.3f}; the floor (floor vs ref) "
                  f"{fident} / {fsim:.3f}" + ("  FLAG" if flag else ""))


def copy_breaks(B, Cn, its):
    tok = D.tokenizer()
    edits = sorted(k for k in its if k.startswith("agent/edit") and k in B and k in Cn)
    P = {k: tok.decode(its[k]["prompt_ids"]) for k in edits}

    def lines(text):
        seg, closed = text.split("</parameter>")[0], "</parameter>" in text
        parts = seg.split("\n")
        return [l for l in (parts if closed else parts[:-1]) if l.strip()]
    out, br = [], []
    for k in edits:
        a, b = lines(Cn[k]["text"]), lines(B[k]["text"])
        same = bool(a) and bool(b) and a[0] == b[0]
        if same and all(l in P[k] for l in a) and not all(l in P[k] for l in b):
            br.append(k)
        out.append(f"{k[6:]} control {sum(l in P[k] for l in a)}/{len(a)}, batch {sum(l in P[k] for l in b)}/{len(b)}"
                   + (" (same target)" if same else ""))
    return br, "copy check on the edit turns: " + "; ".join(out) + f" | copy breaks: {br or 'none'}"


def main(batch, ctrl, core, ifx=None):
    its = dr.items()
    B, Cn, F, R = dr.load(batch), dr.load(ctrl), dr.load("floor"), dr.load("ref")
    if B is None or Cn is None:
        print(f"no rows for {batch if B is None else ctrl}"); print("F1: INCOMPLETE"); return 2
    nax = os.path.basename(batch).startswith("ref2tn")
    bad = gate2(B, Cn, nax)
    print(f"gate 2 (switch counts per row): {'PASS' if not bad else 'FAIL on ' + str(bad[:6])}")
    if bad:
        print("F1: VOID"); return 5
    print(f"core lines (tools/d4x.py {batch} {core} {ifx or ''}):")
    d = d4x.main(batch, core, ifx)
    if d == 2:
        print("F1: INCOMPLETE"); return 2
    rv, rline = ruler(B, Cn, its)
    print(rline)
    if rv is None:
        print("F1: INCOMPLETE"); return 2
    at = open(AT14).read().split()
    all29 = sorted(k for k in its if its[k].get("set") == "agent")
    f1, l1 = agreement(B, Cn, F, R, at, "AT14")
    f2, l2 = agreement(B, Cn, F, R, all29, "all")
    print("flags (no lines):"); print("  " + l1); print("  " + l2)
    br, cl = copy_breaks(B, Cn, its)
    print("  " + cl)
    flags = [n for n, f in (("agent agreement under the floor's (AT14)", f1), ("agent agreement under the floor's (all)", f2),
                            ("copy break", bool(br))) if f]
    if d4x.VERDICT == "STAGE2":
        print(f"F1: STAGE2 (IFEval alone fails on the core; flags: {flags or 'none'})"); return 10
    core_v = d4x.VERDICT
    final = "FAIL" if (core_v == "FAIL" or rv == "FAIL") else "DIFFERS (HIGHER)" if core_v == "DIFFERS (HIGHER)" else "PASS"
    print(f"F1: {final} (core {core_v}, RULER {rv}; flags: {flags or 'none'})")
    return 1 if final == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:5]))
