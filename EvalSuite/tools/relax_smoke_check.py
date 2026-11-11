"""RELAX smoke checks (docs/PREREG_relax_2026-11-02.md, Runs). CPU only; reads results/relax_<arm>_smoke[mask]/.
  1. step 0 equal to its reference in every G process (step0/*.json: REF2-T, or REF2-T + HC8 for B and C)
  2. A's route rows == A's mask rows: tokens, and every committed row's hash (EVALSUITE_RELAX_HASH=1)
  3. the replay's step 0 bitwise, and the control's replay bitwise on every committed row (G = E without a gate)
  4. the levers engage: gated rows and a shrinking union on A, B, C, G; HC8 on in B and C. Information: checks where
     the rules kept more than the exact rule (E, F, C), rows open to the comment rule (B, C; HumanEval/26 has '#'
     comments), misses per check on a shared key vs the control
Last line: SMOKE: PASS | FAIL (exit 0 / 1)."""
import glob, gzip, json, os, sys

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
ok = True


def check(name, cond, info=""):
    global ok
    ok &= bool(cond)
    print(f"{'ok  ' if cond else 'FAIL'} {name}{(': ' + info) if info else ''}")


def jl(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


def arm(a, tag="smoke"):
    d = os.path.join(RES, f"relax_{a}_{tag}")
    rows = {r["key"]: r for r in jl(os.path.join(d, "decode.jsonl"))} if os.path.exists(os.path.join(d, "decode.jsonl")) else {}
    chk = {k: jl(os.path.join(d, "checks", k.replace("/", "__") + ".jsonl.gz")) for k in rows}
    rep = {}
    for k in rows:
        p = os.path.join(d, "replay", k.replace("/", "__") + ".jsonl.gz")
        if os.path.exists(p):
            rep[k] = jl(p)[1:]
    s0 = [json.load(open(p)) for p in glob.glob(os.path.join(d, "step0", "*.json"))]
    rc = jl(os.path.join(d, "replay_chunks.jsonl")) if os.path.exists(os.path.join(d, "replay_chunks.jsonl")) else []
    return dict(rows=rows, checks=chk, replay=rep, step0=s0, replay_chunks=rc)


A = {a: arm(a) for a in ("control", "A", "B", "E", "F", "C", "G", "T3ALL")}
A["Amask"] = arm("A", "smokemask")
for a, x in A.items():
    check(f"{a}: rows written", len(x["rows"]) >= 1, f"{len(x['rows'])} rows")
    if a == "T3ALL":                              # no exact rows for its plan: the tier took effect instead
        check("T3ALL: every row at plan 3333333333", all(r.get("tier") == "3333333333" for r in x["rows"].values()))
        continue
    check(f"{a}: step 0 equal to its reference in every process", x["step0"] and all(z["equal"] for z in x["step0"]),
          f"{len(x['step0'])} process(es)")
# 2. route == mask
d, m = A["A"], A["Amask"]
same_tok = all(d["rows"][k]["completion_ids"] == m["rows"][k]["completion_ids"] for k in d["rows"] if k in m["rows"])
hd = [h for k in d["rows"] for c in d["checks"][k] for h in c.get("h", [])]
hm = [h for k in m["rows"] for c in m["checks"].get(k, []) for h in c.get("h", [])]
check("A route == mask: tokens", same_tok and set(d["rows"]) == set(m["rows"]))
check("A route == mask: committed rows bitwise", hd and hd == hm, f"{sum(x == y for x, y in zip(hd, hm))}/{len(hd)} hashes")
# 3. the replay
for a in ("control", "A", "B", "T3ALL"):
    rc = A[a]["replay_chunks"]
    check(f"replay {a}: step 0 bitwise", rc and all(r["step0_bits"][0] == r["step0_bits"][1] > 0 for r in rc),
          str([r["step0_bits"] for r in rc]))
c = A["control"]
eq = sum(r.get("h_eq", 0) for k in c["replay"] for r in c["replay"][k])
n = sum(len(r["eam"]) for k in c["replay"] for r in c["replay"][k] if "h_eq" in r)
check("replay control: every committed row bitwise (no gate: G = E)", n > 0 and eq == n, f"{eq}/{n}")
dev = sum(x != y for k in c["replay"] for r in c["replay"][k]
          for x, y in zip(r["eam"], c["rows"][k]["completion_ids"][r["pos"]:r["pos"] + len(r["eam"])]))
check("replay control: no deviation from E's argmax", dev == 0, f"{dev}")
# 4. the gate engages
for a in ("A", "B", "C", "G"):
    x = A[a]
    g = [v for k in x["checks"] for cc in x["checks"][k] for v in cc["g"]]
    gated = [cc for k in x["checks"] for cc in x["checks"][k] if any(cc["g"])]
    shrink = sum(cc["ub"] < cc["ubx"] for cc in gated)
    check(f"{a}: gated rows", g.count(1) + g.count(2) > 0, f"cf {g.count(1)}, copy {g.count(2)} of {len(g)} rows")
    check(f"{a}: the union shrinks on gated checks", gated and shrink == len(gated), f"{shrink}/{len(gated)} checks")
for a in ("E", "F", "C"):
    x = A[a]
    more = sum(cc["k"] > cc["kx"] for k in x["checks"] for cc in x["checks"][k])
    print(f"info {a}: the relaxed rule kept more than the exact rule on {more} checks (few prompts: information)")
for a in ("B", "C"):
    x = A[a]
    ncm = sum(len(cc.get("cm", [])) for k in x["checks"] for cc in x["checks"][k])
    print(f"info {a}: the comment rule was open on {ncm} rows (few prompts: information)")
    check(f"{a}: HC8 on", all(r.get("hc8") for r in x["rows"].values()))
ctl, dd = A["control"], A["A"]
for k in set(ctl["rows"]) & set(dd["rows"]):
    mc = sum(cc["ms"] for cc in ctl["checks"][k]) / max(len(ctl["checks"][k]), 1)
    md = sum(cc["ms"] for cc in dd["checks"][k]) / max(len(dd["checks"][k]), 1)
    uc = sum(cc["ub"] for cc in ctl["checks"][k]) / max(len(ctl["checks"][k]), 1) / 1e9
    ud = sum(cc["ub"] for cc in dd["checks"][k]) / max(len(dd["checks"][k]), 1) / 1e9
    print(f"info {k}: misses per check control {mc:.1f} vs A {md:.1f}; union {uc:.2f} vs {ud:.2f} GB per check")
print(f"SMOKE: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
