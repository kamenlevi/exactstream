"""REF2c-T IFX (docs/PREREG_ifx_2026-11-02.md): REF2c-T vs REF2-T on IFEval's other 441 prompts, in two stages.
  build         data/items/dec_ifevalx.jsonl (IFEval's 541 prompts minus the suite's 100, built like dec_ifeval) and
                results/ifx/{stage1,stage2,repro}_keys.txt
  repro <run>   the run's reproduction rows == its variant's saved rows? (exit 0 yes, 5 no, 3 not all there yet)
  stage1        decision 4's rule on the 220 stage-1 rows + the stopping rule (exit 10 STOP and not FAIL,
                11 STOP FAIL, 0 run stage 2, 2 rows missing)
  final         decision 4's rule on all 441 new rows (exit 0 PASS, 1 FAIL or DIFFERS (HIGHER), 2 rows missing)
  table, oc     the stopping rule's call for stage-1 outcomes, and its behaviour by simulation (for the PREREG)
The verdict uses the new rows only (the ifevalx rows of the scratch runs ref2ct.r2 / ref2t.r2): a fresh sample. The
100 core IFEval rows (results/ref2ct = Part 1, results/ref2t = R-58) triggered this test, so they are information
only (pooled 541). Decision 4's rule on unrounded numbers: delta > -5 points AND exact McNemar p >= 0.05.
Usage: .venv/bin/python tools/ifx.py <cmd> [...]"""
import hashlib, json, math, os, random, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from evalsuite import decode_report as dr
from evalsuite.report import mcnemar_p, paired_ci

D_IFX = os.path.join(ROOT, "results", "ifx")
ARMS = (("ref2ct", "ref2ct.r2"), ("ref2t", "ref2t.r2"))   # (the variant's saved core rows, this test's scratch run)
REPRO = ("ifeval/3409", "ifeval/2439")    # two short core rows each arm re-decodes first (VOID if not token-equal)
N1, N_ALL, M2 = 220, 441, 221             # new rows at stage 1 and at the end; stage-2 rows
SEED, STAGE1 = 1102, 220
STOP_PP = 0.95                            # stop at stage 1 when finishing has < 5 % predictive chance to flip the call


def keyfile(name):
    return os.path.join(D_IFX, f"{name}_keys.txt")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]


def build():
    from evalsuite import data as D
    raw = [json.loads(l) for l in open(os.path.join(D.RAW, "ifeval_input_data.jsonl"))]
    core = D.load("dec_ifeval")
    ck = {it["key"] for it in core}
    assert len(raw) == 541 and len(ck) == 100, (len(raw), len(ck))
    tok = D.tokenizer()
    bad = [it["key"] for it in core if D.chat(tok, it["prompt"]) != it["prompt_ids"]]
    assert not bad, f"the chat template no longer gives dec_ifeval's prompt ids: {bad[:3]}"
    rows = [dict(key=f"ifevalx/{r['key']}", set="ifevalx", prompt=r["prompt"],
                 instruction_id_list=r["instruction_id_list"], kwargs=r["kwargs"], user=r["prompt"],
                 prompt_ids=D.chat(tok, r["prompt"]), max_tokens=1024)
            for r in raw if f"ifeval/{r['key']}" not in ck]
    assert len(rows) == 441, len(rows)
    D.write("dec_ifevalx", rows)
    keys = sorted(r["key"] for r in rows)
    s1 = sorted(random.Random(SEED).sample(keys, STAGE1))
    s2 = sorted(set(keys) - set(s1))
    os.makedirs(D_IFX, exist_ok=True)
    for name, ks in (("stage1", s1), ("stage2", s2), ("repro", list(REPRO))):
        with open(keyfile(name), "w") as f:
            f.write("\n".join(ks) + "\n")
    for p in (os.path.join(D.ITEMS, "dec_ifevalx.jsonl"), keyfile("stage1"), keyfile("stage2"), keyfile("repro")):
        print(f"{os.path.relpath(p, ROOT)}: sha256 {sha(p)}")
    print(f"stage 1: {len(s1)} keys, stage 2: {len(s2)} keys")


def items():
    its = {k: it for k, it in dr.items().items() if it.get("set") == "ifeval"}
    from evalsuite import data as D
    its.update({it["key"]: dict(it, set="ifeval") for it in D.load("dec_ifevalx")})   # score_ifeval checks the set
    return its


def load_keys(name):
    return open(keyfile(name)).read().split()


def scored(xkeys):
    """per arm: {key: (instruction results, 1/0)} over the 100 core rows + the given ifevalx keys; None if rows miss"""
    its = items()
    core = sorted(k for k in its if k.startswith("ifeval/"))
    out, miss = {}, {}
    for v, run in ARMS:
        a, b = dr.load(v) or {}, dr.load(run) or {}
        rows = {k: a[k] for k in core if k in a}
        rows.update({k: b[k] for k in xkeys if k in b})
        miss[v] = len(core) + len(xkeys) - len(rows)
        out[v] = (rows, dr.score_ifeval(rows, its))
    return out, miss, core


def rule(b, c, n):
    """decision 4's rule on unrounded numbers: (verdict, delta in points, exact McNemar p)"""
    d, p = 100.0 * (b - c) / n, mcnemar_p(b, c)
    if p < 0.05 and b > c:
        return "DIFFERS (HIGHER)", d, p
    return ("PASS" if d > -5 and p >= 0.05 else "FAIL"), d, p


def kstar(t):
    """the largest k with exact two-sided McNemar p < 0.05 at b + c = t (min(b, c) <= k fails), -1 if none"""
    s, k = 0, -1
    for i in range(t // 2 + 1):
        s += math.comb(t, i)
        if 2 * s >= 0.05 * 2 ** t:
            break
        k = i
    return k


KSTAR = {}


def final_ok(b, c, n=N_ALL):
    """decision 4's rule on n rows does not say FAIL (PASS, or DIFFERS (HIGHER)): == rule(b, c, n)[0] != "FAIL" """
    t = b + c
    if t not in KSTAR:
        KSTAR[t] = kstar(t)
    return b > c or (min(b, c) > KSTAR[t] and 100.0 * (b - c) / n > -5)


def pp_ok(b1, c1, n1=N1, m=M2):
    """predictive probability that decision 4's rule on all n1 + m rows does not say FAIL: Jeffreys Dirichlet posterior
    over (only REF2c-T right, only REF2-T right, both the same) from the n1 pooled stage-1 rows; the m remaining rows'
    counts are Dirichlet-multinomial; exact sum (terms below e^-40 dropped)"""
    a = (b1 + 0.5, c1 + 0.5, n1 - b1 - c1 + 0.5)
    lg = math.lgamma
    base = lg(sum(a)) - lg(sum(a) + m) + lg(m + 1) - sum(lg(x) for x in a)
    tot = pas = 0.0
    for x in range(m + 1):
        for y in range(m + 1 - x):
            z = m - x - y
            lp = (base + lg(x + a[0]) - lg(x + 1) + lg(y + a[1]) - lg(y + 1) + lg(z + a[2]) - lg(z + 1))
            if lp < -40:
                continue
            w = math.exp(lp)
            tot += w
            if final_ok(b1 + x, c1 + y, n1 + m):
                pas += w
    return pas / tot


def call(b1, c1):
    """the stopping rule at stage 1: STOP <decision 4's verdict on the pooled 320> when finishing has a predictive
    chance below 5 % to flip FAIL / not FAIL, else RUN STAGE 2; and that predictive probability (of not FAIL)"""
    pp, v = pp_ok(b1, c1), rule(b1, c1, N1)[0]
    if (pp >= STOP_PP and v != "FAIL") or (pp <= 1 - STOP_PP and v == "FAIL"):
        return f"STOP {v}", pp                    # PASS, DIFFERS (HIGHER) or FAIL
    return "RUN STAGE 2", pp


def summary(sc, keys, label):
    (rc, sc_c), (rt, sc_t) = sc["ref2ct"], sc["ref2t"]
    b = sum(1 for k in keys if sc_c[k][1] > sc_t[k][1])
    c = sum(1 for k in keys if sc_t[k][1] > sc_c[k][1])
    n = len(keys)
    m, lo, hi = paired_ci([sc_c[k][1] - sc_t[k][1] for k in keys])
    v, d, p = rule(b, c, n)
    print(f"  {label} (n {n}): REF2c-T {100 * sum(sc_c[k][1] for k in keys) / n:.1f} % vs REF2-T "
          f"{100 * sum(sc_t[k][1] for k in keys) / n:.1f} %, delta {d:+.2f} [{100 * lo:+.1f}, {100 * hi:+.1f}] pt, "
          f"McNemar p {p:.4f} ({b}/{c}): {v}")
    return b, c, n, v


def info(sc, xkeys, core):
    (rc, sc_c), (rt, sc_t) = sc["ref2ct"], sc["ref2t"]
    print("\n== information (no pass line; the counts below are over the new rows)")
    summary(sc, core + xkeys, "pooled with the 100 core rows")
    summary(sc, core, "the 100 core rows (Part 1)")
    keys = xkeys
    same = sum(1 for k in keys if rc[k]["completion_ids"] == rt[k]["completion_ids"])
    print(f"  token-identical rows: {same}/{len(keys)}")
    for v, (rows, s) in (("REF2c-T", sc["ref2ct"]), ("REF2-T", sc["ref2t"])):
        trunc = sum(1 for k in keys if rows[k]["finish"] == "length")
        loops = sum(1 for k in keys if dr.looped(rows[k]["completion_ids"]))
        toks = sum(len(rows[k]["completion_ids"]) for k in keys)
        print(f"  {v}: truncations {trunc}, loops {loops}, generated tokens {toks}")
    miss = {}
    its = items()
    for v, (rows, s) in (("REF2c-T", sc["ref2ct"]), ("REF2-T", sc["ref2t"])):
        for k in keys:
            for iid, ok in zip(its[k]["instruction_id_list"], s[k][0]):
                if not ok:
                    miss.setdefault(iid.split(":")[0], {}).setdefault(v, 0)
                    miss[iid.split(":")[0]][v] += 1
    print("  instruction misses by group (REF2c-T / REF2-T): " +
          ", ".join(f"{g} {m.get('REF2c-T', 0)}/{m.get('REF2-T', 0)}" for g, m in sorted(miss.items())))
    for v, run in ARMS:
        rows = dr.load(run) or {}
        ks = [k for k in xkeys if k in rows]
        tok = sum(len(rows[k]["completion_ids"]) for k in ks)
        sec = sum(rows[k]["seconds"] - rows[k].get("mem", {}).get("paused_s", 0) for k in ks)
        life = max((rows[k].get("mem", {}).get("life", 0) for k in ks), default=0)
        print(f"  {run}: {len(ks)} new rows, {tok / max(sec, 1e-9):.2f} tok/s (rough, not an A/B), max lifetime "
              f"footprint at a row {life:.2f} GB")


def repro(run):
    v = run.split(".")[0]
    a, b = dr.load(v) or {}, dr.load(run) or {}
    if not all(k in b for k in REPRO):
        print(f"{run}: reproduction rows not all there yet")
        return 3
    bad = [k for k in REPRO if a[k]["completion_ids"] != b[k]["completion_ids"]]
    print(f"{run}: reproduction rows {', '.join(REPRO)}: " + (f"DIFFERENT on {bad}: VOID" if bad else "token-identical"))
    return 5 if bad else 0


def stage1():
    xk = load_keys("stage1")
    sc, miss, core = scored(xk)
    if any(miss.values()):
        print(f"IFX stage 1: rows missing {miss}")
        return 2
    print(f"IFX stage 1: the {len(xk)} stage-1 rows of the new IFEval prompts")
    b, c, n, v = summary(sc, xk, "new prompts, stage 1")
    what, pp = call(b, c)
    print(f"  stopping rule: predictive probability that decision 4 on all {N_ALL} does not say FAIL = {pp:.4f} -> {what}")
    info(sc, xk, core)
    print(f"IFX STAGE 1: {what} ({b}/{c} on n {n}: {v})")
    return 11 if what == "STOP FAIL" else 10 if what.startswith("STOP") else 0


def final():
    xk = load_keys("stage1") + load_keys("stage2")
    sc, miss, core = scored(xk)
    if any(miss.values()):
        print(f"IFX final: rows missing {miss}")
        return 2
    print(f"IFX final: all {len(xk)} new IFEval prompts")
    b, c, n, v = summary(sc, xk, "new prompts")
    info(sc, xk, core)
    print(f"IFX: {v} (decision 4's rule on the {n} new IFEval rows, {b}/{c})")
    return 0 if v == "PASS" else 1


def table():
    print(f"stopping rule at stage 1 (n {N1}; b/c = only REF2c-T right / only REF2-T right): call, predictive "
          f"P(not FAIL at {N_ALL}); P = stop PASS, D = stop DIFFERS (HIGHER), F = stop FAIL, 2 = run stage 2")
    for t in (8, 12, 16, 20, 24, 30):
        cells = []
        for gap in range(-8, 15, 2):
            if (t + gap) % 2 or abs(gap) > t:
                continue
            c1, b1 = (t + gap) // 2, (t - gap) // 2
            what, pp = call(b1, c1)
            cells.append(f"{b1}/{c1} {'2' if what == 'RUN STAGE 2' else what.split()[1][0]} {pp:.2f}")
        print(f"  b + c = {t}: " + "  ".join(cells))


def oc(sims=4000):
    """the rule's behaviour by simulation: new rows drawn with a discordance rate of 7 % (same-weights arms: 5-8 per
    100) and a true REF2c-T - REF2-T gap of 0 / -2 / -4 points"""
    rng = random.Random(SEED)
    print(f"operating characteristics ({sims} runs each; new rows: 7 % discordant)")
    for gap in (0.0, -0.02, -0.04):
        pb, pc = (0.07 + gap) / 2, (0.07 - gap) / 2
        draw = lambda m: (lambda u: (sum(x < pb for x in u), sum(pb <= x < pb + pc for x in u)))([rng.random() for _ in range(m)])
        stop_ok = stop_fail = ok = 0
        for _ in range(sims):
            b1, c1 = draw(STAGE1)
            what, pp = call(b1, c1)
            if what == "STOP FAIL":
                stop_fail += 1
            elif what.startswith("STOP"):
                stop_ok += 1
                ok += 1
            else:
                b2, c2 = draw(M2)
                ok += rule(b1 + b2, c1 + c2, N_ALL)[0] != "FAIL"
        print(f"  true gap {100 * gap:+.0f} pt: stops at stage 1 {100 * (stop_ok + stop_fail) / sims:.1f} % "
              f"(not FAIL {100 * stop_ok / sims:.1f} %, FAIL {100 * stop_fail / sims:.1f} %); "
              f"verdict not FAIL {100 * ok / sims:.1f} %")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build":
        build()
    elif cmd == "repro":
        sys.exit(repro(sys.argv[2]))
    elif cmd == "stage1":
        sys.exit(stage1())
    elif cmd == "final":
        sys.exit(final())
    elif cmd == "table":
        table()
    elif cmd == "oc":
        oc()
    else:
        sys.exit(__doc__)
