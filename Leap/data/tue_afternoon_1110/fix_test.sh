#!/bin/bash
# PF-B fix test (orchestrator 14:2x: research copy only, not the package): release/cleanclone/pfbfix/clone = pkg_110b + pflm.py's
# R57_PFLM_SEGALL=1 (router + shared expert per chunk slice inside PF-B's groups). exact-32, R57_LOGHASH=1, 64 tokens,
# TOK 8192. Prompts: a short-tail one (P = 5,412 = 5 x 1,024 + 292, the 30K tail size) and the 29,988-token one. Arms per
# prompt: PF-A (reference), PF-B without the fix, PF-B with it. Gate: no lock / engine, NO-SWAP >= 49; share lock; swap +2 GB.
A=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/aft
SP=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad
CL=/Users/kamen/Leap/release/cleanclone/pfbfix/clone; cd $CL; LOG=$A/fix.log
END=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-10 16:45:00" +%s)
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
busy() { peer ls 2>/dev/null | grep "^- " | grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*" | grep -cE "^· (benchmarking|waiting for a quiet window)|MEM ([6-9]|[1-9][0-9]) GB"; }
say() { echo "$(date +%T) $*" | tee -a $LOG; }
n=0; until [ "$(busy)" = 0 ] && [ "$(noswap)" -ge 49 ]; do n=$((n + 1)); [ $n -gt 40 ] && { say "SKIP: not clear (busy $(busy), NO-SWAP $(noswap))"; exit 0; }; sleep 30; done
trap 'peer unlock >/dev/null 2>&1' EXIT
peer lock share 40m --cpu 300 --gpu 100 "MEM 41 GB: Leap PF-B fix test (research copy, exact-32, 6 runs, ~8 min)" > /dev/null 2>&1
S0=$(swap); say "GO: NO-SWAP $(noswap) GB, swap $S0 MB"
.venv/bin/python - $SP/prefillbench/prompt_30k.txt $A <<'EOF' >> $LOG 2>&1
import sys
from transformers import AutoTokenizer
src, out = sys.argv[1:3]; t = open(src).read(); ins = "Summarize the main findings of the document above in five bullet points."
doc = t[:t.rindex(ins)].rstrip(); tk = AutoTokenizer.from_pretrained("engine/models/bare"); ids = tk.encode(doc, add_special_tokens=False)
target = 5400                                    # + 12 template tokens = P 5,412 = 5 x 1,024 + 292
for n in range(target - 30, target + 1):
    s = tk.decode(ids[:n]) + "\n\n" + ins + "\n"
    if len(tk.encode(s, add_special_tokens=False)) >= target:
        break
open(f"{out}/prompt_tail.txt", "w").write(s); print("prompt_tail.txt:", len(tk.encode(s, add_special_tokens=False)), "tokens before the template")
EOF
for P_ in tail 30k; do
  F=$([ $P_ = tail ] && echo $A/prompt_tail.txt || echo $SP/prefillbench/prompt_30k.txt)
  for arm in pfa pfb pfbfix; do
    [ $(( $(date +%s) + 240 )) -le $END ] || { say "SKIP $P_ $arm: time"; continue; }
    [ "$(noswap)" -ge 48 ] || { say "SKIP $P_ $arm: NO-SWAP $(noswap)"; continue; }
    case $arm in pfa) E="--env R57_PFLM=1";; pfb) E="--env R57_PFLM=3 --env R57_PFLM_TOK=8192";; pfbfix) E="--env R57_PFLM=3 --env R57_PFLM_TOK=8192 --env R57_PFLM_SEGALL=1";; esac
    .venv/bin/python run.py --config exact --mem 32 --prompt-file $F --max-tokens 64 --env R57_LOGHASH=1 $E > $A/fix_${P_}_$arm.out 2>&1 &
    P=$!
    while kill -0 $P 2>/dev/null; do [ $(( $(swap) - S0 )) -gt 2048 ] && { say "STOP: swap"; for c in $(pgrep -P $P); do kill -TERM $c; done; kill -TERM $P; }; sleep 3; done
    wait $P; say "  $P_ $arm: $(grep -a -E '^whole decode' $A/fix_${P_}_$arm.out | cut -c1-160)"
  done
done
.venv/bin/python - $A <<'EOF' | tee -a $LOG
import json, re, sys
A = sys.argv[1]
def files(n):
    try:
        tp = re.search(r"^tokens: ([^;]+)", open(f"{A}/{n}.out").read(), re.M).group(1).strip()
    except Exception:
        return None, None
    return json.load(open(tp)), json.load(open(tp.replace("_tokens.json", "_runs.json")))[0]
def toks(x):
    if isinstance(x, list) and x and all(isinstance(t, int) for t in x): return x
    for v in (x.values() if isinstance(x, dict) else x if isinstance(x, list) else []):
        t = toks(v)
        if t: return t
for P_ in ("tail", "30k"):
    ta, ra = files(f"fix_{P_}_pfa")
    if not ra: print(f"{P_}: no PF-A reference"); continue
    print(f"{P_}: P {ra.get('P')}; PF-A first token {ra['pfs']:.2f} s ({ra['P'] / ra['pfs']:.0f} tok/s), peak {ra['mem'][6]:.2f} GB")
    for arm in ("pfb", "pfbfix"):
        tb, rb = files(f"fix_{P_}_{arm}")
        if not rb: print(f"  {arm}: no result"); continue
        a, b = toks(ta), toks(tb); ha, hb = ra.get("lh") or [], rb.get("lh") or []
        ft = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None); eq = sum(x == y for x, y in zip(ha, hb))
        print(f"  {arm}: tokens {'EQUAL' if a == b else f'differ from token {ft + 1}'}; hashes equal on {eq} of {min(len(ha), len(hb))}; "
              f"first token {rb['pfs']:.2f} s ({rb['P'] / rb['pfs']:.0f} tok/s, x{ra['pfs'] / rb['pfs']:.2f} vs PF-A); peak {rb['mem'][6]:.2f} GB; calls {rb.get('pflm', {}).get('calls')}")
EOF
say "fix test end: NO-SWAP $(noswap) GB, swap $(swap) MB (start $S0)"
