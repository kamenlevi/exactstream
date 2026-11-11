"""Fit12 PREFILL BUG CHECK (Sun 8 Nov 08:4x; orchestrator 08:2x + Leap a521e91: MLX 0.32.2's sorted NAX gather leaves rows
unwritten when one call has > 32,767 rows and rows % 64 != 0; pre-registration: /Users/kamen/Fit12/PLAN_12GB_90_fit12.md
'PREFILL BUG CHECK'). The reader of /Users/kamen/Fit12/scripts/pf13.sh: arm ex, CAP 32, BigMoE's PFX prompts (1.6K / 4.7K /
7.9K), one process per arm:
  F = pf13f: the shipped grid (bt_b12 fast mode: PFR 'balanced', one MoE chunk per prompt up to 16K) + the gather_qmm audit
      + Leap's pad (an exposed sorted call is padded to x64) = the re-measure;
  K = pf13k: B12_PFR_GRID=keep, s_moe 3072: every MoE call <= 30,720 rows, so no call can reach the bug = the reference.
Per prompt: PFR's (P, s_moe), the audit (sorted gather calls, the largest row count, exposed, padded), prefill s / tok/s /
SSD GB / TTFT / lifetime; then F = K in the kept-row logits hash (lsha) and the 32 tokens, and F = pf12a / pf12b's tokens.
Usage: python -I -B pf13_read.py [DIR] [TAG_F] [TAG_K] [PF12_DIR]"""
import ast, json, os, re, sys

D = sys.argv[1] if len(sys.argv) > 1 else "/Users/kamen/Fit12/data/scr/pf13"
TF = sys.argv[2] if len(sys.argv) > 2 else "pf13f"
TK = sys.argv[3] if len(sys.argv) > 3 else "pf13k"
P12 = sys.argv[4] if len(sys.argv) > 4 else "/Users/kamen/Fit12/data/scr/chat12"


def read(d, tag):
    T = re.escape(tag)
    rpf = re.compile(rf"^\[{T}\] (\S+) ex PREFILL n (\d+) smoe (\d+) \| ([\d.]+) s, ([\d.]+) tok/s, SSD (\d+) MB \| lifetime ([\d.]+) GB"
                     rf"(?: \| pfr (\[.*?\]) \| gq (\{{.*\}}))?")
    rdec = re.compile(rf"^\[{T}\] (\S+) ex theta None .*?\| tok (\d+) sha (\w+) \| lsha (\w+) .*?\| prefill ([\d.]+) s, decode ([\d.]+) s,")
    out = {}
    p_ = os.path.join(d, f"{tag}.txt")
    if not os.path.exists(p_):
        return out
    for ln in open(p_):
        s = ln.rstrip("\n")
        m = rpf.match(s)
        if m:
            out.setdefault(m.group(1), {}).update(n=int(m.group(2)), pf=float(m.group(4)), tps=float(m.group(5)),
                                                  mb=int(m.group(6)), life=float(m.group(7)),
                                                  pfr=ast.literal_eval(m.group(8)) if m.group(8) else None,
                                                  gq=ast.literal_eval(m.group(9)) if m.group(9) else None)
            continue
        m = rdec.match(s)
        if m:
            out.setdefault(m.group(1), {}).update(tok=int(m.group(2)), sha=m.group(3), lsha=m.group(4), dec=float(m.group(6)))
    for k, r in out.items():
        kf = k.replace("/", "_")
        f = os.path.join(d, f"{tag}_{kf}_ex_toks.json")
        r["toks"] = json.load(open(f)) if os.path.exists(f) else None
        f = os.path.join(d, f"{tag}_{kf}_ex_ph.json")
        if os.path.exists(f) and "dec" in r:
            ph = json.load(open(f))
            e0, eN = max(p[6] for p in ph if p[1] == 0), max(p[6] for p in ph)
            r["ttft"] = r["pf"] + r["dec"] - (eN - e0)
    return out


F, K = read(D, TF), read(D, TK)
A12, B12 = read(P12, "pf12a"), read(P12, "pf12b")
print(f"PREFILL BUG CHECK ({D}): F = {TF} (shipped grid + audit + Leap's pad), K = {TK} (keep grid, s_moe 3072: <= 30,720 rows per call)")
print("  prompt          | n    | arm | PFR (P, s_moe) | sorted calls  max rows  exposed  padded | prefill s  tok/s  SSD GB  TTFT s  life GB")
for k in sorted(set(F) | set(K), key=lambda k: (F.get(k) or K.get(k))["n"]):
    for a, R in (("F", F), ("K", K)):
        r = R.get(k)
        if not r or "pf" not in r:
            print(f"  {k:15s} |      | {a}   | (missing)")
            continue
        g = r["gq"] or {}
        print(f"  {k:15s} | {r['n']:4d} | {a}   | {str(r['pfr']):14s} | {g.get('sorted', '-'):>12} {g.get('maxm', '-'):>9} "
              f"{g.get('exposed', '-'):>8} {g.get('padded', '-'):>7} | {r['pf']:9.2f} {r['tps']:6.1f} {r['mb'] / 1e3:7.1f} "
              f"{r.get('ttft', float('nan')):7.2f} {r['life']:8.3f}")
n_eq = n_l = n_12 = n = exp = 0
for k in F:
    if k not in K or "lsha" not in F[k] or "lsha" not in K[k]:
        continue
    n += 1
    n_l += F[k]["lsha"] == K[k]["lsha"]
    n_eq += F[k]["toks"] == K[k]["toks"]
    t12 = (A12.get(k) or {}).get("toks")
    n_12 += t12 is not None and F[k]["toks"] == t12 and F[k]["toks"] == (B12.get(k) or {}).get("toks")
    exp += (F[k]["gq"] or {}).get("exposed", 0) + (K[k]["gq"] or {}).get("exposed", 0)
print(f"BITWISE: F = K lsha {n_l} / {n}, tokens {n_eq} / {n}; F tokens = pf12a = pf12b {n_12} / {n}; exposed sorted calls (F + K) {exp}")
for k in sorted(F, key=lambda k: F[k].get("n", 0)):
    a12 = A12.get(k)
    if a12 and "pf" in a12 and "pf" in F[k]:
        print(f"  {k}: pf12a {a12['tps']:.1f} tok/s, TTFT {a12.get('ttft', float('nan')):.2f} s -> F {F[k]['tps']:.1f} tok/s, "
              f"TTFT {F[k].get('ttft', float('nan')):.2f} s (x{F[k]['tps'] / a12['tps']:.3f})")
json.dump(dict(F={k: {kk: v for kk, v in r.items() if kk != "toks"} for k, r in F.items()},
               K={k: {kk: v for kk, v in r.items() if kk != "toks"} for k, r in K.items()},
               lsha_eq=n_l, toks_eq=n_eq, pf12_eq=n_12, n=n, exposed=exp), open(os.path.join(D, "pf13_report.json"), "w"), indent=1)
