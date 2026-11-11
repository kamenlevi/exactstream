"""R-56 on REF2b verdict (docs/PREREG_r56b_2026-10-30.md) from results/r56b/{ref2b,ref2b_gate16}.jsonl and chunks.jsonl.
Per arm: PASS = on all 36 prompts every token equals REF2b's AR and every kept row is bitwise equal to the AR row (every
AR row compared), with the AR reproducing results/ref2b; FAIL = any token or row differs; VOID = the AR differs from
results/ref2b on a prompt (that prompt doesn't count); INCOMPLETE = prompts missing.
R-56c (ref2t_gate16, results/r56c) is the same against REF2-T's AR and results/ref2t. R-56d (ref2to_gate16,
results/r56d, docs/PREREG_r56d_2026-11-01.md) adds two lines to the PASS: every AR row (tokens and row hashes) equals
R-56c's AR (results/r56c/ar: REF2-T without OOO / the fused combine), and on every prompt OOO ran every tier call of the AR
and of the loop (NOT ENGAGED otherwise). R-56e (ref2cto_gate16, results/r56e, docs/PREREG_ref2ct_2026-11-01.md): REF2c-T
with OOO + the fused combine against its own AR and results/ref2ct; OOO as in R-56d, and both REF2c switches (att_single,
gdn_inproj) ran in the AR and in the loop of every prompt (NOT ENGAGED otherwise).
Usage: .venv/bin/python tools/r56b.py [ARM...] (default: ref2b ref2b_gate16) > results/r56b.txt; ref2t_gate16 ->
results/r56c.txt; ref2to_gate16 -> results/r56d.txt; ref2cto_gate16 -> results/r56e.txt"""
import json, os, statistics as S, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ARMS = {"ref2b": ("r56b", "ref2b", "REF2b"), "ref2b_gate16": ("r56b", "ref2b", "REF2b"),   # (results dir, reference
        "ref2t_gate16": ("r56c", "ref2t", "REF2-T"), "ref2to_gate16": ("r56d", "ref2t", "REF2-T"),   # rows, its name)
        "ref2cto_gate16": ("r56e", "ref2ct", "REF2c-T"),
        "ref2tnaxno_gate16": ("r56n", "ref2tnax", "NAX6-N")}   # PREREG_nax6 Part 2 (converted store; rows: Part 1)
NAXSW = ("nax6", "nax6_ooo", "nax_native")        # NAX6-N's calls (Part 2): nax6 or nax6_ooo, and nax_native
SW = ("att_single", "gdn_inproj")                 # REF2c's switches (R-56e)
keys = open(os.path.join(ROOT, "results", "r56", "keys.txt")).read().split()


def ar_vs_r56c(D, k):
    """(rows compared, rows equal, tokens equal) of this arm's AR file vs R-56c's for prompt k"""
    f = k.replace("/", "__") + ".json"
    a, b = (json.load(open(os.path.join(ROOT, "results", d, "ar", f))) for d in (D, "r56c"))
    return len(b["hash"]), sum(x == y for x, y in zip(a["hash"], b["hash"])) if len(a["hash"]) == len(b["hash"]) else 0, \
        a["toks"] == b["toks"]


for arm in sys.argv[1:] or ["ref2b", "ref2b_gate16"]:
    D_, ref_v, REFNAME = ARMS[arm]
    D = os.path.join(ROOT, "results", D_)
    cp = os.path.join(D, "chunks.jsonl")
    chunks = [json.loads(l) for l in open(cp)] if os.path.exists(cp) else []
    p = os.path.join(D, f"{arm}.jsonl")
    rows = {r["key"]: r for r in map(json.loads, open(p))} if os.path.exists(p) else {}
    void = {k: r for k, r in rows.items() if r["ar_vs_ref2b"] is not None}
    valid = {k: r for k, r in rows.items() if k not in void}
    bad = {k: r for k, r in valid.items() if not r["bits_equal"]}
    missing = [k for k in keys if k not in rows]
    r56n = arm == "ref2tnaxno_gate16"
    ooo, r56d, r56e = arm in ("ref2to_gate16", "ref2cto_gate16") or r56n, arm == "ref2to_gate16", arm == "ref2cto_gate16"
    nax_ok = lambda c: bool(c) and (c.get("nax6", 0) + c.get("nax6_ooo", 0)) > 0 and c.get("nax_native", 0) > 0
    nx = {k: (r.get("ar_ref2c"), {s: (r["per_check"] or {}).get(s) for s in NAXSW}) for k, r in rows.items()
          if not (nax_ok(r.get("ar_ref2c")) and nax_ok(r["per_check"]))} if r56n else {}   # Part 2: NAX6-N ran
    arc = {k: ar_vs_r56c(D_, k) for k in valid} if r56d else {}   # R-56d: the AR vs R-56c's AR
    ar_bad = {k: c for k, c in arc.items() if not (c[0] == c[1] and c[2])}
    idle = {k: r["calls"] for k, r in valid.items()           # R-56d / e: OOO on every tier call
            if not all(c and c[0] > 0 and c[1] == c[0] for c in (r["calls"]["ar"], r["calls"]["loop"]))} if ooo else {}
    sw = {k: (r.get("ar_ref2c"), {s: (r["per_check"] or {}).get(s) for s in SW}) for k, r in valid.items()
          if not all((r.get("ar_ref2c") or {}).get(s, 0) > 0 and (r["per_check"] or {}).get(s, 0) > 0 for s in SW)
          } if r56e else {}                       # R-56e: both REF2c switches ran in the AR and in the loop
    if r56n:                                      # Part 2's line 2: the AR leaving NAX6's Part 1 rows is a FAIL
        v = ("FAIL" if bad or void else "NOT ENGAGED" if idle or nx else "INCOMPLETE" if missing else "PASS")
    else:
        v = ("FAIL" if bad or ar_bad else f"VOID (the AR left results/{ref_v})" if void else "NOT ENGAGED" if idle or sw
             else "INCOMPLETE" if missing else "PASS")
    toks = sum(r["n"] for r in valid.values() if r["bits_equal"])
    cmp_, eq = sum(r["rows_cmp"] for r in valid.values()), sum(r["rows_eq"] for r in valid.values())
    print(f"{arm}: {v}: {len(valid) - len(bad)}/{len(keys)} prompts exact (every token and every kept row bitwise vs "
          f"{REFNAME}'s AR; {toks} tokens); kept rows bitwise {eq}/{cmp_}; {len(missing)} missing, {len(void)} void")
    if r56d:
        n_, e_ = sum(c[0] for c in arc.values()), sum(c[1] for c in arc.values())
        print(f"  AR (OOO + fused combine) vs R-56c's AR (REF2-T without them): {len(arc) - len(ar_bad)}/{len(arc)} prompts "
              f"equal; AR rows bitwise {e_}/{n_}")
    if r56e:
        print(f"  REF2c switches engaged: {len(valid) - len(sw)}/{len(valid)} prompts in the AR and the loop; AR calls "
              + ", ".join(f"{s} {sum((r.get('ar_ref2c') or {}).get(s, 0) for r in valid.values())}" for s in SW)
              + "; loop per check " + ", ".join(f"{s} {S.mean((r['per_check'] or {}).get(s, 0) for r in valid.values()):.2f}"
                                                for s in SW) if valid else "  REF2c switches engaged: no rows")
    if r56n:
        print(f"  the AR vs results/{ref_v} (NAX6's row-major Part 1 rows; NAX6-N == NAX6): {len(rows) - len(void)}/{len(rows)} "
              f"prompts equal")
        print(f"  NAX6-N engaged: {len(rows) - len(nx)}/{len(rows)} prompts (nax6 / nax6_ooo and nax_native in the AR and "
              f"the loop)")
    if ooo:
        tc = [sum(r["calls"][ph][i] for r in valid.values() if r["calls"][ph]) for ph in ("ar", "loop") for i in (0, 1)]
        print(f"  OOO engaged: {len(valid) - len(idle)}/{len(valid)} prompts on every tier call; tier calls / OOO calls: "
              f"AR {tc[0]} / {tc[1]}, loop {tc[2]} / {tc[3]}")
    for s in ("humaneval", "math", "ifeval"):
        rs = [r for r in valid.values() if r["set"] == s]
        print(f"  {s}: {sum(r['bits_equal'] for r in rs)}/{len(rs)} exact")
    long_ = [r for r in valid.values() if r["n_prompt"] + r["n"] >= 497]
    print(f"  prompts that pass 512 keys: {sum(r['bits_equal'] for r in long_)}/{len(long_)} exact; "
          f"over 1024 keys: {sum(r['bits_equal'] for r in long_ if r['n_prompt'] + r['n'] >= 1009)}/"
          f"{sum(1 for r in long_ if r['n_prompt'] + r['n'] >= 1009)}")
    for k, r in sorted(bad.items(), key=lambda kv: keys.index(kv[0])):
        print(f"  DIFF {k} (P {r['n_prompt']}): first bit difference {r['bdiff']}; first token difference {r['tdiff']}")
    for k, c in ar_bad.items():
        print(f"  AR DIFF {k}: rows bitwise {c[1]}/{c[0]} vs R-56c's AR, tokens equal {c[2]}")
    for k, c in idle.items():
        print(f"  NOT ENGAGED {k}: [tier calls, OOO calls] {c}")
    for k, c in nx.items():
        print(f"  NOT ENGAGED {k}: NAX6-N calls (AR, loop per check) {c}")
    for k, c in sw.items():
        print(f"  NOT ENGAGED {k}: REF2c switch calls (AR, loop per check) {c}")
    for k, r in void.items():
        print(f"  VOID {k}: the AR leaves results/{ref_v} at token {r['ar_vs_ref2b']}")
    if valid:
        pc = [r["per_check"] or {} for r in valid.values()]
        keys_pc = sorted({k for x in pc for k in x})
        print("  per check (means): " + ", ".join(f"{k} {S.mean(x.get(k, 0) for x in pc):.2f}" for k in keys_pc))
        print(f"  speed (information, quiet mode FAN_CAP {sorted({str(r['fan_cap']) for r in valid.values()})}, not an A/B): "
              + ", ".join(f"{s} {S.mean(r['tok_s'] for r in valid.values() if r['set'] == s and r['tok_s']):.1f} tok/s"
                          for s in ("humaneval", "math", "ifeval") if any(r["set"] == s for r in valid.values()))
              + f"; kept {S.mean(r['kept'] for r in valid.values() if r['checks']):.2f} per check; paused rows "
              f"{sum(r['paused'] for r in valid.values())}")
    for c in (c for c in chunks if c["arm"] == arm):
        print(f"  process {c['pid']} {c['start']}-{c['end']}: {c['rows']} rows, lifetime peak {c['life_gb']} GB, swap "
              f"{c['swap0']} -> {c['swap1']} GB, ref2.py {c['ref2_sha']}, GATE_N {c['gate_n']}, MAXK {c['maxk']}, ext "
              f"{c['ext']}, C {c['C']}" + (f", COMBINE {c.get('combine')}, OOO {c.get('ooo')}" if ooo else "")
              + (f", {c['ref2_code']} CFG {c['ref2_cfg']}" if c.get("ref2_cfg") else ""))
