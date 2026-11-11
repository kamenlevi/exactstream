#!/bin/bash
# Fri 6 Nov 11:3x: the two pre-flights as ONE chain. At 11:25 they waited separately on the same clear Mac, started together
# (2 x 31 GB, swap +3 GB) and the swap guard stopped both. Order: D7BNAX_32=1 re-run, its token compare against the =2 run
# (09:27, data/pf3/d7v2/), then never-wait bit check 4. The orchestrator's rule for =2: use it if the 2-prompt share
# pre-flight is clean and matches =1 token for token; otherwise keep =1.
cd "$HOME/BigMoE" || exit 1
PD_D7=1 bash scripts/preflight_d7x.sh
mkdir -p data/pf3/d7v1
cp data/r57/f3k_160_runs.json data/r57/f3k_160_tokens.json data/r57/f3r_160_runs.json data/r57/f3r_160_tokens.json \
  data/pf3/d7v1/ 2>/dev/null
.venv/bin/python -I - >> data/pf3/pf3.log 2>&1 <<'EOF'
import json, time
ts = time.strftime("%m-%d %H:%M:%S")
ok = []
for a in ("f3k", "f3r"):
    try:
        t1 = json.load(open(f"data/pf3/d7v1/{a}_160_tokens.json"))
        t2 = json.load(open(f"data/pf3/d7v2/{a}_160_tokens.json"))
    except (OSError, ValueError) as e:
        print(ts, a, "=1 vs =2: no compare:", type(e).__name__, getattr(e, "filename", "")); ok.append(False); continue
    first = None
    for k in sorted(set(t1) | set(t2)):
        x, y = json.dumps(t1.get(k), sort_keys=True), json.dumps(t2.get(k), sort_keys=True)
        if x != y:
            first = (k, next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y))))
            break
    ok.append(first is None and len(t1) > 0)
    print(ts, a, "=1 vs =2:", f"MATCH token for token ({len(t1)} prompts)" if ok[-1] else f"DIFF first at {first} (json char)")
print(ts, "D7BNAX_32 verdict:", "=2 MATCHES =1 on both arms: use =2 if both arms ran CLEAN" if all(ok) else "keep =1")
EOF
bash scripts/preflight_nw.sh
