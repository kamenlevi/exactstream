#!/bin/bash
> (a quote from the project owner, kept private)
# math run, one job over 5 GB at a time, all done by 16:45:
#   0. prompts: 3K / 15K / 20K tokens cut from the same paper text as prompt_10k / prompt_30k (no GPQA text);
#   1. 36 GB Mac simulation of the 24 GB package: a 28 GB mlock'ed ballast (ballast.py) + the engine's GPU working set capped
#      in-process at 27 GB (site/sitecustomize.py); run.py --mem 24 on BigMoE's 12 HumanEval keys, one ~3K-token prompt,
#      then a 3-turn chat.py --mem 24 conversation; swap guard +2 GB; the ballast is released whatever happens;
#   2. PF-A at 24 GB: chat.py --mem 24 with R57_PFLM=1 vs 0 (short + long prompt), answers / first token / peak;
#   3. PF-B length bisect: exact-32, R57_LOGHASH=1, 64 tokens, PF-A vs PF-B (TOK 8192, the published setting) at 10K / 15K
#      / 20K tokens; NO-SWAP >= 48 before each run.
SP=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad
A=$SP/aft; CL=/Users/kamen/Leap/release/cleanclone/final/clone; LOG=$A/chain.log; cd $CL
ep() { date -j -f "%Y-%m-%d %H:%M:%S" "$1" +%s; }
END=$(ep "2026-11-10 16:45:00")
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
others() { peer ls 2>/dev/null | grep "^- "; }
locks() { grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*"; }
busy() { others | locks | grep -cE "^· (benchmarking|waiting for a quiet window)|MEM ([6-9]|[1-9][0-9]) GB"; }
engines() { ps -axo pid=,comm=,args= | awk '$2 ~ /[Pp]ython/ && /r57_screen_|run_map[.]py|fit_onecall[.]py|[ \/](chat|serve)[.]py( |$)/ {printf "%s ", $1}'; }
say() { echo "$(date +%T) $*" | tee -a $LOG; }
desc() { local c; for c in $(pgrep -P "$1"); do desc "$c"; echo "$c"; done; }
BAL=""; LOCKED=0; S0=0
cleanup() { [ -n "$BAL" ] && kill -TERM $BAL 2>/dev/null && say "ballast released (pid $BAL)"; BAL=""
            [ $LOCKED = 1 ] && { peer unlock >/dev/null 2>&1; LOCKED=0; say "unlocked"; }; }
trap 'cleanup; say "exit (NO-SWAP $(noswap) GB, swap $(swap) MB)"' EXIT
trap 'say "stopped by a signal"; exit 1' INT TERM
# run1 <name> <cap s> <cmd...>: one engine process with the swap guard (+2 GB since S0) and a wall cap; sets STOPPED
run1() {
  local name=$1 cap=$2 T0 s; shift 2; STOPPED=""
  "$@" > $A/$name.out 2>&1 &
  local P=$!; T0=$(date +%s)
  while kill -0 $P 2>/dev/null; do
    s=$(swap)
    if [ $((s - S0)) -gt 2048 ]; then STOPPED="swap $s MB (start $S0)"; elif [ $(( $(date +%s) - T0 )) -gt $cap ]; then STOPPED="cap $cap s"; fi
    [ -n "$STOPPED" ] && { say "STOP $name: $STOPPED"; for c in $(desc $P); do kill -TERM $c 2>/dev/null; done; kill -TERM $P; break; }
    sleep 2
  done
  wait $P 2>/dev/null
  say "  $name: $(grep -a -E '^whole decode' $A/$name.out | tail -1 | cut -c1-200) | wall $(( $(date +%s) - T0 )) s | swap now $(swap) MB"
}
say "armed (pid $$): after BigMoE's W24g; tests 1-3; end 16:45"
n=0; until [ "$(busy)" = 0 ] && [ -z "$(engines)" ] && [ "$(noswap)" -ge 44 ]; do
  n=$((n + 1)); [ $(( $(date +%s) + 1500 )) -gt $END ] && { say "SKIP: not clear in time (busy $(busy), engines '$(engines)', NO-SWAP $(noswap))"; exit 0; }
  sleep 30
done
sleep 30; [ "$(busy)" = 0 ] && [ -z "$(engines)" ] || { say "SKIP: another job appeared"; exit 0; }
peer lock share 120m --cpu 300 --gpu 100 "MEM 53 GB: Leap afternoon tests (36 GB simulation = 28 GB ballast + the 24 GB engine; PF-A at 24 GB; PF-B bisect at 32 GB), one at a time" > /dev/null 2>&1 && LOCKED=1
S0=$(swap); say "GO: NO-SWAP $(noswap) GB, swap $S0 MB"

# ---- 0. prompts ----
.venv/bin/python - $SP/prefillbench/prompt_30k.txt $A <<'EOF' >> $LOG 2>&1
import sys
from transformers import AutoTokenizer
src, out = sys.argv[1:3]
t = open(src).read()
ins = "Summarize the main findings of the document above in five bullet points."
doc = t[:t.rindex(ins)].rstrip() if ins in t else t
tk = AutoTokenizer.from_pretrained("engine/models/bare")
ids = tk.encode(doc, add_special_tokens=False)
for name, n in (("3k", 2975), ("15k", 14975), ("20k", 19975)):
    body = tk.decode(ids[:n])
    open(f"{out}/prompt_{name}.txt", "w").write(body + "\n\n" + ins + "\n")
    print(f"prompt_{name}.txt: {len(tk.encode(body + chr(10) + chr(10) + ins, add_special_tokens=False))} tokens before the template")
EOF
HE=$(cat $SP/w24check/he_keys.txt)

# ---- 1. 36 GB Mac simulation, two ballast sizes ----
sim_case() {   # <label> <ballast GB>
  local L=$1 G=$2 N0
  [ $(( $(date +%s) + 420 )) -le $END ] || { say "SKIP sim $L: cannot end by 16:45"; return; }
  N0=$(noswap); /opt/homebrew/opt/python@3.12/bin/python3.12 -I $A/ballast.py $G > $A/ballast_$L.out 2>&1 &
  BAL=$!; for i in $(seq 1 90); do grep -q "locked\|failed" $A/ballast_$L.out && break; sleep 1; done
  say "test 1 case $L: ballast $G GB: $(cat $A/ballast_$L.out | tr '\n' ' '); NO-SWAP $N0 -> $(noswap) GB"
  if grep -q locked $A/ballast_$L.out; then
    S0=$(swap)
    export PYTHONPATH=$A/site SIM36_WS_GB=27
    run1 sim${L}_he12 600 .venv/bin/python run.py --config lossy --mem 24 --keys "$HE" --max-tokens 256
    [ -z "$STOPPED" ] && run1 sim${L}_long 600 .venv/bin/python run.py --config chat --mem 24 --prompt-file $A/prompt_3k.txt --max-tokens 256
    [ -z "$STOPPED" ] && { run1 sim${L}_chat 600 sh -c "printf '%s\n%s\n%s\n/exit\n' 'What is a mixture-of-experts language model? Answer in three sentences.' 'Write a short Python function for a top-2 router over 8 experts.' 'Explain in two sentences what that function does.' | .venv/bin/python chat.py --mem 24 --max-tokens 256"; ls -t out/chat_*.jsonl | head -1 > $A/sim${L}_chat.jsonl.path; }
    unset PYTHONPATH SIM36_WS_GB
    say "  case $L wired-limit lines: $(grep -h -a -o 'R57_WIRED: mx.set_wired_limit([^)]*)' $(ls -t out/cli_*.log out/chat_*.log 2>/dev/null | head -3) 2>/dev/null | sort | uniq -c | tr '\n' ' ')"
  fi
  kill -TERM $BAL 2>/dev/null; wait $BAL 2>/dev/null; BAL=""; sleep 10
  say "test 1 case $L done: ballast released; NO-SWAP $(noswap) GB, swap $(swap) MB (case start $S0)"
}
NB=$(noswap); GA=$(( NB - 25 )); [ $GA -lt 10 ] && GA=10
say "test 1: NO-SWAP $NB GB before any ballast -> case A ballast $GA GB (engine room ~25 GB), case B 28 GB"
sim_case A $GA
sim_case B 28

# ---- 2. PF-A at 24 GB (no ballast) ----
S0=$(swap)
if [ $(( $(date +%s) + 420 )) -le $END ] && [ "$(noswap)" -ge 35 ]; then
  LONG="$(cat $SP/pfachat/long_body.txt) Summarize the main findings of the document above in five bullet points."
  for arm in pfa default; do
    V=$([ $arm = pfa ] && echo 1 || echo 0)
    run1 pfa24_$arm 420 sh -c "printf '%s\n/clear\n%s\n/exit\n' 'Explain in three sentences how a refrigerator keeps food cold.' \"\$1\" | R57_PFLM=$V .venv/bin/python chat.py --mem 24 --max-tokens 256" _ "$LONG"
    ls -t out/chat_*.jsonl | head -1 > $A/pfa24_$arm.jsonl.path
  done
else say "SKIP test 2: time or NO-SWAP $(noswap)"; fi

# ---- 3. PF-B length bisect (exact-32) ----
for L in 10k 15k 20k; do
  F=$([ $L = 10k ] && echo $SP/prefillbench/prompt_10k.txt || echo $A/prompt_$L.txt)
  for arm in pfa pfb8k; do
    [ $(( $(date +%s) + 240 )) -le $END ] || { say "SKIP bisect $L $arm: cannot end by 16:45"; continue; }
    NEED=$([ $L = 10k ] && echo 44 || { [ $L = 15k ] && echo 46 || echo 47; }); [ "$(noswap)" -ge $NEED ] || { say "SKIP bisect $L $arm: NO-SWAP $(noswap) < $NEED"; continue; }
    E="--env R57_PFLM=1"; [ $arm = pfb8k ] && E="--env R57_PFLM=3 --env R57_PFLM_TOK=8192"
    run1 bis_${L}_$arm 360 .venv/bin/python run.py --config exact --mem 32 --prompt-file $F --max-tokens 64 --env R57_LOGHASH=1 $E
  done
done
cleanup
.venv/bin/python - $A <<'EOF' | tee -a $LOG
import json, os, re, statistics, sys
A = sys.argv[1]
def files(name):
    p = f"{A}/{name}.out"
    if not os.path.exists(p): return None, None
    m = re.search(r"^tokens: ([^;]+)", open(p, errors="replace").read(), re.M)
    if not m: return None, None
    tp = m.group(1).strip(); return json.load(open(tp)), json.load(open(tp.replace("_tokens.json", "_runs.json")))
def toks(x):
    if isinstance(x, list) and x and all(isinstance(t, int) for t in x): return x
    for v in (x.values() if isinstance(x, dict) else x if isinstance(x, list) else []):
        t = toks(v)
        if t: return t
print("== 1. 36 GB simulation")
for name in ("simA_he12", "simA_long", "simB_he12", "simB_long"):
    t, r = files(name)
    if r:
        n = sum(x["n"] for x in r); dw = sum(x["n"] / x["tps"] for x in r); pf = [x["pfs"] for x in r]
        pk = max(x["mem"][6] for x in r if x.get("mem") and len(x["mem"]) > 6)
        print(f"{name}: {len(r)} prompt(s), P {[x.get('P') for x in r][:1] if len(r) == 1 else 'HE12'}, {n} tokens, whole decode {n / dw:.1f} tok/s, "
              f"first token median {statistics.median(pf):.2f} s, lifetime peak {pk:.2f} GB")
        if name.endswith("_he12"):
            ref = json.load(open(os.path.expanduser("~/BigMoE/data/r57/w13eh_100_tokens.json")))
            fl = lambda d: {k: (list(v.values())[0][-1] if isinstance(v, dict) else v) for k, v in d.items()}
            a, b = fl(t), fl(ref); print(f"  tokens vs BigMoE w13eh (free memory): {sum(a.get(k) == v for k, v in b.items())}/{len(b)} identical")
    else:
        print(f"{name}: no result")
try:
    for i, r in enumerate(json.loads(l) for l in open(open(f"{A}/simA_chat.jsonl.path").read().strip())):
        print(f"sim_chat turn {i + 1}: P {r.get('P')}, {r['n']} tokens, whole decode {r.get('tps', 0):.1f} tok/s, first token {r.get('pfs')} s, "
              f"peak {r.get('peak')} GB, drafter {r.get('drafter')}, answer {r['answer'][:60]!r}")
except Exception as e:
    print("sim_chat: incomplete:", type(e).__name__, e)
print("== 2. PF-A at 24 GB")
try:
    R = {arm: [json.loads(l) for l in open(open(f"{A}/pfa24_{arm}.jsonl.path").read().strip())] for arm in ("pfa", "default")}
    for i, label in enumerate(("short", "long")):
        a, d = R["pfa"][i], R["default"][i]
        print(f"{label}: P {a.get('P')}; answers {'IDENTICAL' if a['answer'] == d['answer'] else 'DIFFER'} ({a['n']} / {d['n']} tokens); first token "
              f"PF-A {a.get('pfs')} vs default {d.get('pfs')} s; peak {a.get('peak')} vs {d.get('peak')} GB; decode {a.get('tps', 0):.1f} vs {d.get('tps', 0):.1f}")
except Exception as e:
    print("PF-A at 24 GB: incomplete:", type(e).__name__, e)
print("== 3. PF-B bisect (exact-32, TOK 8192 vs PF-A)")
for L in ("10k", "15k", "20k"):
    ta, ra = files(f"bis_{L}_pfa"); tb, rb = files(f"bis_{L}_pfb8k")
    if not (ra and rb): print(f"{L}: incomplete"); continue
    a, b = toks(ta), toks(tb); ha, hb = ra[0].get("lh") or [], rb[0].get("lh") or []
    ft = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    eq = sum(x == y for x, y in zip(ha, hb)); fh = next((i for i, (x, y) in enumerate(zip(ha, hb)) if x != y), None)
    print(f"{L}: P {ra[0].get('P')}; PF-B calls {rb[0].get('pflm', {}).get('calls')}; tokens {'EQUAL' if a == b else f'differ from token {ft + 1}'}; "
          f"hashes equal on {eq} of {min(len(ha), len(hb))}{'' if fh is None else f' (first difference at check {fh + 1})'}; "
          f"first token PF-A {ra[0].get('pfs'):.1f} / PF-B {rb[0].get('pfs'):.1f} s; peaks {ra[0]['mem'][6]:.2f} / {rb[0]['mem'][6]:.2f} GB")
EOF
say "chain end"
