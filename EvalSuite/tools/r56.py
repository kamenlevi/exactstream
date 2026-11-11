"""R-56 LIVE16 verdict (docs/PREREG_r56_2026-10-30.md) from results/r56/{ref2,stock}.jsonl and chunks.jsonl.
Per arm: PASS = all 36 prompts' tokens equal to the reference row (sha256 of the token list too) in processes whose
reference check reproduced; FAIL = any prompt differs; VOID = a process's reference check failed (its rows don't count);
INCOMPLETE = prompts missing. Information: AR logits rows bitwise (ref2), per-check path counts, speed (quiet mode),
memory. Usage: tools/r56.py [ARM...] (default: ref2 stock) > results/r56.txt"""
import json, os, statistics as S, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
D = os.path.join(ROOT, "results", "r56")
keys = open(os.path.join(D, "keys.txt")).read().split()
chunks = [json.loads(l) for l in open(os.path.join(D, "chunks.jsonl"))] if os.path.exists(os.path.join(D, "chunks.jsonl")) else []
for arm in sys.argv[1:] or ["ref2", "stock"]:
    p = os.path.join(D, f"{arm}.jsonl")
    rows = {r["key"]: r for r in map(json.loads, open(p))} if os.path.exists(p) else {}
    ch = [c for c in chunks if c["arm"] == arm]
    valid = {k: r for k, r in rows.items() if r["refcheck"]}
    bad = {k: r for k, r in valid.items() if not (r["equal"] and r["sha"] == r["ref_sha"])}
    void = [k for k, r in rows.items() if not r["refcheck"]]
    missing = [k for k in keys if k not in valid]
    v = ("FAIL" if bad else "VOID (a reference check failed)" if void else "INCOMPLETE" if missing else "PASS")
    print(f"{arm}: {v}: {len(valid) - len(bad)}/{len(keys)} prompts token-equal to {('results/ref2' if arm == 'ref2' else 'results/ref.r2')}"
          f" ({sum(r['n'] for r in valid.values() if r['equal'])} tokens); {len(missing)} missing, {len(void)} void")
    for s in ("humaneval", "math", "ifeval"):
        rs = [r for r in valid.values() if r["set"] == s]
        print(f"  {s}: {sum(r['equal'] for r in rs)}/{len(rs)} equal")
    u = [r for r in valid.values() if r["under128"]]
    print(f"  prompts under 128 tokens: {sum(r['equal'] for r in u)}/{len(u)} equal; at or over 128: "
          f"{sum(r['equal'] for r in valid.values() if not r['under128'])}/{len(valid) - len(u)}")
    for k, r in sorted(bad.items(), key=lambda kv: keys.index(kv[0])):
        print(f"  DIFF {k} (P {r['n_prompt']}): first at token {r['diff']['at']} of {r['ref_n']}: got {r['diff']['got']}, "
              f"reference {r['diff']['want']}; check row {r['diff'].get('row')}, logit gap to the reference's token "
              f"{r['diff'].get('gap')}, top-2 gap {r['diff'].get('top2')}")
    if arm == "ref2" and valid:
        cmp_, bad_ = sum(r["ar_cmp"] for r in valid.values()), sum(r["ar_bad"] for r in valid.values())
        pc = [r["per_check"] or {} for r in valid.values()]
        print(f"  AR logits rows bitwise (information): {cmp_ - bad_}/{cmp_}; per check: slot calls "
              f"{S.mean(x.get('slot_calls', 0) for x in pc):.1f}, SDPA splits {S.mean(x.get('sdpa_split', 0) for x in pc):.2f}")
    if valid:
        print(f"  speed (information, quiet mode FAN_CAP {sorted({str(r['fan_cap']) for r in valid.values()})}): "
              + ", ".join(f"{s} {S.mean(r['tok_s'] for r in valid.values() if r['set'] == s and r['tok_s']):.1f} tok/s"
                          for s in ("humaneval", "math", "ifeval") if any(r["set"] == s for r in valid.values()))
              + f"; kept {S.mean(r['kept'] for r in valid.values() if r['checks']):.2f} per check; "
              f"paused rows {sum(r['paused'] for r in valid.values())}")
    for c in ch:
        print(f"  process {c['pid']} {c['start']}-{c['end']}: {c['rows']} rows, reference check {c['refcheck']}, "
              f"lifetime peak {c['life_gb']} GB, swap {c['swap0']} -> {c['swap1']} GB, ref2.py {c['ref2_sha']}, "
              f"SCRATCH_KEEP {c['scratch_keep']}, C {c['C']}")
