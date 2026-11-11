"""CERT's lines (docs/PREREG_cert_2026-11-08.md + amendment 1) -> results/cert/CERT_*.txt (fixed paths, quoted by the paper).
  .venv/bin/python tools/cert_report.py <arm forward run> <arm reverse run> <long-code run> [<chat-check run>]
1. Per set and order (6 lines): decision 4 vs NAX6-N's rows (tools/d4x.py's rule: delta > -5 pt AND McNemar p >= 0.05; a
   gain with p < 0.05 is DIFFERS (HIGHER)). HumanEval+ 164 = dec_humaneval 60 + dec_humanevalx 104; math 100 = dec_math
   50 + dec_mathx 50; MBPP 100 (tools/mbpp_score.py). Reference rows: results/ref2tnax (the core 60 + 50), ref2tnaxn.r3
   (humanevalx 104 + mathx 50), ref2tnaxn.r2 (MBPP 100). humanevalx / mathx items are scored as HumanEval+ / math (their
   set renamed for the scorers, as d4x's stage 2 does for ifevalx).
2. Pooled over the three sets and both orders (728 pairs): delta > -2 pt AND McNemar p >= 0.05.
1b. IFEval (amendment 3, Sat 23:3x: the shipped config's chat is not exact): decision 4 vs NAX6's IFEval rows
   (results/ref2tnax) per order, the 100 core prompts, no stage 2 (2 more lines; the pooled line keeps its 3 sets).
3. Long code: the 41 humanevalt rows, tfwin against the exact base: blunders (the base's p of a committed token < 0.01)
   <= 1 per 10K tokens.
4. The chat check (PREREG_cert amendment 2; the orchestrator Sat 05:5x): IFEval prompts through the v1 engine, interleaved
   with code prompts (request-boundary profile swaps); their tokens must equal NAX6's AR rows (results/ref2tnax) on every
   one -> CHAT-EXACT: YES | NO (results/cert/CERT_chat.txt). A check of its own, outside CERT's lines.
CERT PASS = all 10 lines pass (the 8 + IFEval's 2, amendment 3); DIFFERS (HIGHER) is named in the verdict, not a FAIL (d4x's convention). Information
(CERT_info.txt): token agreement between the orders, rows token-identical to NAX6-N per set (which sets run FAST
kernels), cap hits, the core 60 + 50 vs REF2-T. Rows without text get it
from their ids in memory (tools/rows_text.py's decode, equal to the decode stage's text on 520 / 520 reference rows)."""
import gzip, json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from evalsuite import data as D
from evalsuite import decode_report as dr
from evalsuite.report import mcnemar_p
import d4x, mbpp_score as mb, rows_text as rt

RES = os.path.join(ROOT, "results")
OUT = os.environ.get("CERT_OUT", os.path.join(RES, "cert"))     # CERT_OUT: a test elsewhere
N = {"humaneval": 164, "math": 100, "mbpp": 100, "ifeval": 100}
POOLED = ("humaneval", "math", "mbpp")      # the pre-registered pooled line (728 pairs); IFEval (amendment 3) is information there
NAME = {"humaneval": "HumanEval+", "math": "math", "mbpp": "MBPP", "ifeval": "IFEval"}
REFS = ("ref2tnax", "ref2tnaxn.r3", "ref2tnaxn.r2")
NLONG = 41


def items():
    its = {k: it for k, it in dr.items().items() if it.get("set") in ("humaneval", "math", "ifeval")}
    for name, as_ in (("humanevalx", "humaneval"), ("mathx", "math")):
        its.update({it["key"]: dict(it, set=as_) for it in D.load("dec_" + name)})
    its.update({it["key"]: dict(it, set="mbpp") for it in D.load("dec_mbpp")})
    return its


def load(*runs):
    out = {}
    for run in runs:
        for k, r in (dr.load(run) or {}).items():
            if r.get("text") is None:
                r["text"] = rt.text(r["completion_ids"])
            out.setdefault(k, r)
    return out


def score(rows, its, s, keys):
    sub = {k: rows[k] for k in keys}
    return mb.score(sub, its) if s == "mbpp" else dr.SCORERS[s](sub, its)


def line(name, sa, sr, keys):
    b = sum(1 for k in keys if sa[k][1] == 1 and sr[k][1] == 0)
    c = sum(1 for k in keys if sa[k][1] == 0 and sr[k][1] == 1)
    v, d, p = d4x.rule(b, c, len(keys))
    ca, cr = (100 * sum(x[k][1] for k in keys) / len(keys) for x in (sa, sr))
    return v, b, c, (f"{name} (n {len(keys)}): {ca:.1f} % vs {cr:.1f} %, delta {d:+.2f} pt, McNemar p {p:.4f} "
                     f"(arm only {b} / reference only {c}): {v}")


def longcode(run):
    p = os.path.join(RES, run, "decode.jsonl")
    rows = {r["key"]: r for r in map(json.loads, open(p))} if os.path.exists(p) else {}
    tok = dev = bl = sc = 0
    for k, r in rows.items():
        f = os.path.join(RES, run, "tfwin", k.replace("/", "__") + ".json.gz")
        if not os.path.exists(f):
            continue
        s = json.load(gzip.open(f, "rt"))
        out = r["completion_ids"]
        sc += 1
        tok += len(out)
        dev += sum(a != b for a, b in zip(s["am"], out))
        bl += sum(lp < math.log(0.01) for lp in s["lp"])
    r10 = 1e4 * bl / max(tok, 1)
    v = "INCOMPLETE" if sc < NLONG else "PASS" if r10 <= 1 else "FAIL"
    return v, (f"long code (the {NLONG} humanevalt rows, {run}): {sc} scored, {tok} tokens, blunders {bl} = {r10:.2f} per "
               f"10K (bar <= 1), deviations {100 * dev / max(tok, 1):.2f} %: {v}")


def chat_check(run):
    """the chat check: the run's IFEval rows vs NAX6's AR rows, token for token"""
    A, R = load(run), load("ref2tnax")
    ks = sorted(k for k in A if A[k].get("set") == "ifeval" and k in R)
    diff = [(k, next((i for i, (a, b) in enumerate(zip(A[k]["completion_ids"], R[k]["completion_ids"])) if a != b),
                     min(len(A[k]["completion_ids"]), len(R[k]["completion_ids"]))))
            for k in ks if A[k]["completion_ids"] != R[k]["completion_ids"]]
    typ = sorted({str(A[k].get("type")) for k in ks})
    v = "INCOMPLETE" if not ks else "YES" if not diff else "NO"
    return v, (f"CHAT-EXACT: {v}: {len(ks) - len(diff)} of {len(ks)} IFEval rows token-identical to NAX6's AR ({run})"
               + (f"; first divergences {', '.join(f'{k} @{i}' for k, i in diff[:8])}" if diff else "")
               + (f"; row types {typ}" if typ != ["None"] else ""))


def main(fwd, rev, lrun, crun=None):
    os.makedirs(OUT, exist_ok=True)
    its = items()
    R = load(*REFS)
    A = {"fwd": load(fwd), "rev": load(rev)}
    res, vl, info = {}, [], []
    B = Cc = n = B4 = C4 = n4 = 0
    for s in ("humaneval", "math", "mbpp", "ifeval"):
        want = sorted(k for k in its if its[k]["set"] == s)
        assert len(want) == N[s], f"{s}: {len(want)} items, not {N[s]}"
        txt = []
        for o in ("fwd", "rev"):
            keys = [k for k in want if k in A[o] and k in R]
            if len(keys) < N[s]:
                v, l = "INCOMPLETE", f"{NAME[s]} {o}: {len(keys)} of {N[s]} rows in both the arm and the reference: INCOMPLETE"
            else:
                sa, sr = score(A[o], its, s, keys), score(R, its, s, keys)
                v, b, c, l = line(f"{NAME[s]} {o}", sa, sr, keys)
                B4, C4, n4 = B4 + b, C4 + c, n4 + len(keys)
                if s in POOLED:
                    B, Cc, n = B + b, Cc + c, n + len(keys)
                cap = sum(A[o][k].get("finish") == "length" for k in keys)
                info.append(f"cap hits {NAME[s]} {o}: arm {cap}, reference {sum(R[k].get('finish') == 'length' for k in keys)}")
            res[(s, o)] = v
            txt.append(l)
            vl.append(l)
        open(os.path.join(OUT, f"CERT_{s}.txt"), "w").write("\n".join(txt) + "\n")
        for o in ("fwd", "rev"):
            ks = [k for k in want if k in A[o] and k in R]
            info.append(f"token-identical to NAX6-N, {NAME[s]} {o}: "
                        f"{sum(A[o][k]['completion_ids'] == R[k]['completion_ids'] for k in ks)} of {len(ks)}")
        same = [k for k in want if k in A["fwd"] and k in A["rev"]]
        info.append(f"token agreement {NAME[s]}: the two orders identical on "
                    f"{sum(A['fwd'][k]['completion_ids'] == A['rev'][k]['completion_ids'] for k in same)} of {len(same)}")
    if n == 2 * sum(N[s] for s in POOLED):
        d, p = 100.0 * (B - Cc) / n, mcnemar_p(B, Cc)
        pv = "DIFFERS (HIGHER)" if p < 0.05 and B > Cc else "PASS" if d > -2 and p >= 0.05 else "FAIL"
        pl = (f"pooled (n {n} pairs: 3 sets x 2 orders): delta {d:+.2f} pt, McNemar p {p:.4f} (arm only {B} / reference only "
              f"{Cc}; bar: delta > -2 pt AND p >= 0.05): {pv}")
    else:
        pv, pl = "INCOMPLETE", f"pooled: {n} of {2 * sum(N[s] for s in POOLED)} pairs: INCOMPLETE"
    res["pooled"] = pv
    vl.append(pl)
    open(os.path.join(OUT, "CERT_pooled.txt"), "w").write(pl + "\n")
    lv, ll = longcode(lrun)
    res["long code"] = lv
    vl.append(ll)
    open(os.path.join(OUT, "CERT_longcode.txt"), "w").write(ll + "\n")
    name = lambda k: k if isinstance(k, str) else f"{NAME[k[0]]} {k[1]}"
    fl = [name(k) for k, v in res.items() if v == "FAIL"]
    inc = [name(k) for k, v in res.items() if v == "INCOMPLETE"]
    hi = [name(k) for k, v in res.items() if v == "DIFFERS (HIGHER)"]
    final = (f"CERT: FAIL ({', '.join(fl)})" if fl else f"CERT: INCOMPLETE ({', '.join(inc)})" if inc else
             "CERT: PASS" + (f" (DIFFERS (HIGHER) on {', '.join(hi)})" if hi else ""))
    open(os.path.join(OUT, "CERT_VERDICT.txt"), "w").write("\n".join(vl + [final]) + "\n")
    if crun:
        cv, cl = chat_check(crun)
        open(os.path.join(OUT, "CERT_chat.txt"), "w").write(cl + "\n")
        info.insert(0, cl)
    if n4:
        info.insert(0, f"pooled with IFEval (information; amendment 3): n {n4}, delta {100.0 * (B4 - C4) / n4:+.2f} pt, McNemar p "
                       f"{mcnemar_p(B4, C4):.4f} (arm only {B4} / reference only {C4})")
    T = load("ref2t")      # information: the core 60 + 50 vs REF2-T
    for o in ("fwd", "rev"):
        for s in ("humaneval", "math"):
            keys = sorted(k for k in its if its[k]["set"] == s and k in T and k in A[o])
            if keys:
                info.append("vs REF2-T: " + line(f"{NAME[s]} {o} (core)", score(A[o], its, s, keys), score(T, its, s, keys), keys)[3])
    open(os.path.join(OUT, "CERT_info.txt"), "w").write("\n".join(info) + "\n")
    print("\n".join(vl + [final, ""] + info))
    return 1 if fl else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:5]))
