#!/bin/bash
# PF-B bisect, one more length (the chain found 10K / 15K / 20K exact, 30K not): 25K tokens, exact-32, R57_LOGHASH=1, PF-A vs
# PF-B at TOK 8192 (4 groups per layer, like 30K). Gate: no lock / engine, NO-SWAP >= 47; share lock; swap guard +2 GB.
A=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/aft
SP=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad
CL=/Users/kamen/Leap/release/cleanclone/final/clone; cd $CL; LOG=$A/chain.log
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
busy() { peer ls 2>/dev/null | grep "^- " | grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*" | grep -cE "^· (benchmarking|waiting for a quiet window)|MEM ([6-9]|[1-9][0-9]) GB"; }
say() { echo "$(date +%T) $*" | tee -a $LOG; }
[ "$(busy)" = 0 ] && [ "$(noswap)" -ge 47 ] || { say "bis25 SKIP: busy $(busy), NO-SWAP $(noswap)"; exit 0; }
trap 'peer unlock >/dev/null 2>&1' EXIT
peer lock share 15m --cpu 300 --gpu 100 "MEM 40 GB: Leap PF-B bisect 25K (exact-32, 2 runs, ~2 min)" > /dev/null 2>&1
.venv/bin/python - $SP/prefillbench/prompt_30k.txt $A <<'EOF' >> $LOG 2>&1
import sys
from transformers import AutoTokenizer
src, out = sys.argv[1:3]; t = open(src).read(); ins = "Summarize the main findings of the document above in five bullet points."
doc = t[:t.rindex(ins)].rstrip(); tk = AutoTokenizer.from_pretrained("engine/models/bare"); ids = tk.encode(doc, add_special_tokens=False)
open(f"{out}/prompt_25k.txt", "w").write(tk.decode(ids[:24975]) + "\n\n" + ins + "\n"); print("prompt_25k.txt written")
EOF
S0=$(swap)
for arm in pfa pfb8k; do
  E="--env R57_PFLM=1"; [ $arm = pfb8k ] && E="--env R57_PFLM=3 --env R57_PFLM_TOK=8192"
  .venv/bin/python run.py --config exact --mem 32 --prompt-file $A/prompt_25k.txt --max-tokens 64 --env R57_LOGHASH=1 $E > $A/bis_25k_$arm.out 2>&1 &
  P=$!
  while kill -0 $P 2>/dev/null; do [ $(( $(swap) - S0 )) -gt 2048 ] && { say "bis25 STOP: swap"; kill -TERM $(pgrep -P $P) $P; }; sleep 3; done
  wait $P; say "  bis_25k_$arm: $(grep -a -E '^whole decode' $A/bis_25k_$arm.out | cut -c1-150)"
done
.venv/bin/python - $A <<'EOF' | tee -a $LOG
import json, re, sys
A = sys.argv[1]
def files(n):
    tp = re.search(r"^tokens: ([^;]+)", open(f"{A}/{n}.out").read(), re.M).group(1).strip()
    return json.load(open(tp)), json.load(open(tp.replace("_tokens.json", "_runs.json")))[0]
def toks(x):
    if isinstance(x, list) and x and all(isinstance(t, int) for t in x): return x
    for v in (x.values() if isinstance(x, dict) else x if isinstance(x, list) else []):
        t = toks(v)
        if t: return t
(ta, ra), (tb, rb) = files("bis_25k_pfa"), files("bis_25k_pfb8k")
a, b = toks(ta), toks(tb); ha, hb = ra.get("lh") or [], rb.get("lh") or []
ft = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None); eq = sum(x == y for x, y in zip(ha, hb))
fh = next((i for i, (x, y) in enumerate(zip(ha, hb)) if x != y), None)
print(f"25k: P {ra.get('P')}; PF-B calls {rb.get('pflm', {}).get('calls')}; tokens {'EQUAL' if a == b else f'differ from token {ft + 1}'}; "
      f"hashes equal on {eq} of {min(len(ha), len(hb))}{'' if fh is None else f' (first difference at check {fh + 1})'}; peaks {ra['mem'][6]:.2f} / {rb['mem'][6]:.2f} GB")
EOF
