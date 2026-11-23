#!/bin/bash
# results/screens/SETUP.md's test (pre-registered 20:02): control at 2e-6 vs s30 on the same windows; feat re-run if it holds.
cd "$(dirname "$0")/.."
LR=2e-6 ./screen.sh ctl - "control at 2e-6" 300 50 30 || exit 1
OK=$($HOME/BigMoE/.venv/bin/python - <<'PYEOF'
import json
g = json.load(open("results/gate_chain.json")); c = json.load(open("results/screens/_eval_ctl_s30_300_lr2e-6_50_30.json"))
ps, pc = g["per"]["s30|mbpp|chain"], c["per"]["ctl_s30_300_lr2e-6|mbpp|chain"]
k = sorted(pc)
r = lambda p, b: sum(p[x][b][0] for x in k) / sum(p[x][b][1] for x in k)
d23, dk = 100 * (r(pc, "23-30") - r(ps, "23-30")), r(pc, "kept") - r(ps, "kept")
ok = abs(d23) <= 1.5 and abs(dk) <= 0.15
open("results/screens/SETUP.md", "a").write(f"\nResult ({__import__('time').strftime('%H:%M')}): control at 2e-6 vs s30 on MBPP-50: 23-30 {100 * r(ps, '23-30'):.2f} -> {100 * r(pc, '23-30'):.2f} ({d23:+.2f} pt), kept/32 {r(ps, 'kept'):.3f} -> {r(pc, 'kept'):.3f} ({dk:+.3f}) -> {'HOLDS: 2e-6 is the screen default' if ok else 'FAILS: screens use 1e-6 (untested)'}\n")
print("yes" if ok else "no")
PYEOF
)
echo "setup check: $OK"
if [ "$OK" = yes ]; then
  LR=2e-6 ./screen.sh feat feat "t30f (feature loss to the true deep state at every depth) was NO EFFECT on the 125B. Here: MBPP 23-30 >= +2.0 pt over the control with sign p < 0.05 -> DISAGREES with the 125B: find why before trusting stand-in levers; within +-2 pt -> AGREES (NO EFFECT); <= -2.0 pt with p < 0.05 -> HURTS here (note it)." 300 50 30
fi
echo "setup check done"
