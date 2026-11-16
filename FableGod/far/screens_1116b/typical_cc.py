"""THINK_1116b screen 3 (Fable FAR): typical (entropy) acceptance vs margin close-call on kdump's recorded misses.
Each record: the 125B's top-8 log-probs at the miss row (top8_lp[0]); d = the draft token, y = the 125B's top-1.
Margin rule delta: accept if lp(y) - lp(d) <= delta. Typical rule: accept if p(d) >= max(eps, delta_t * exp(-H)),
H the 125B's entropy over its top-8 (a lower bound on the true entropy, so the rule is slightly generous)."""
import json, math, sys
rec = [json.loads(l) for l in open("/Users/kamen/BigMoE/data/deepdraft/ceiling/kdump_" + (sys.argv[1] if len(sys.argv) > 1 else "heplus24") + ".jsonl")]
n = 0; acc = {}
for r in rec:
    ids, lp = r["top8_ids"][0], r["top8_lp"][0]
    if r["d"] not in ids: continue
    n += 1
    pd_ = math.exp(lp[ids.index(r["d"])]); margin = lp[0] - lp[ids.index(r["d"])]
    ps = [math.exp(x) for x in lp]; H = -sum(p * math.log(p) for p in ps if p > 0)
    for name, ok in [("margin d1", margin <= 1), ("margin d2", margin <= 2),
                     ("typical eps.1 dt.3", pd_ >= max(0.1, 0.3 * math.exp(-H))), ("typical eps.1 dt.5", pd_ >= max(0.1, 0.5 * math.exp(-H))),
                     ("typical eps.2 dt.3", pd_ >= max(0.2, 0.3 * math.exp(-H)))]:
        a = acc.setdefault(name, [0, 0, 0]); a[0] += ok; a[1] += ok and H < 1.0; a[2] += ok and margin > 2
print(f"misses with the draft in the 125B's top-8: {n} of {len(rec)}")
for k, (a, lowH, farm) in acc.items(): print(f"  {k:20s} accepts {a:4d} ({a/n:.2f} of misses); of them at 125B entropy < 1 nat: {lowH}; at margin > 2 nats: {farm}")
