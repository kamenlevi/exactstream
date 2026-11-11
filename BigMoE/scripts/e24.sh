#!/bin/bash
# Sat 7 Nov 17:4x (NewIdeas TABLE TONIGHT slot 5, accepted 17:3x): E24, the 24 GB EXACT number = W1B's 24 GB stack (scripts/w1b_24.sh's env:
# s10b, stock kernels, v3 tiers 4433333333 / 3c + COMBINE + OOO, GDN_EXACT, MAXK 80 + DUPROW chunking, C 128 = FRAC 0.25, M-01, JOINT,
# MTP stream, FR-Spec + Scout) + BIGMOE_WIRED=1 (mx.set_wired_limit: the only one of this week's exact-safe fixes that applies; KEEP,
# NAX, S2 and the seams are r57 / ref2-only), on Wed's ext/build_m01 (the binary W1B ran). S5's 12 HumanEval + MBPP12, N 256, two
# passes (e24a, e24b), one process each, after a 1-prompt pre-flight. S10B_X_IDX=999 drops the MoE-input traces (disk).
# REPORT (data/e24/e24_report.txt), per set: WHOLE DECODE (spec125's tok/s, pair-mean geo), END TO END (n / wall incl. prefill),
#   SSD MB per token, lifetime peak (<= 24.0 GB); full period n/a (s10b records no per-check clock).
#   Exact status (NewIdeas): "exact vs its own stock-kernel AR (proven Wed, w24 == w24t); tonight: pass 1 == pass 2 + agreement with
#   NAX6-N's AR, as information" (token sha per prompt; AR = EvalSuite ref2tnax / ref2tnaxn.r2 via data/e5/ar_ref_tokens.json).
# usage: NOTAFTER=hh:mm NOWAIT45=1 GAP0=0 GAP=0 MARGIN=4 bash scripts/turn45.sh 26 30 "..." /bin/bash scripts/e24.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "${E24_ROOT:-$HOME/BigMoE}" || exit 1
OUT="$PWD/data/e24"; mkdir -p "$OUT"
LOG="$OUT/e24.log"
PY="$HOME/BigMoE/.venv/bin/python"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
A=${E24_AVAIL:-$(avail_gb)}
awk -v a="$A" 'BEGIN {exit !(a >= 30)}' || { say "NO-SWAP $A GB < 30: not run"; exit 0; }
[ -n "$E24_DRYRUN" ] || sleep 20
A=${E24_AVAIL:-$(avail_gb)}
awk -v a="$A" 'BEGIN {exit !(a >= 30)}' || { say "NO-SWAP $A GB < 30 at the 20 s look-again: not run"; exit 0; }
[ -f ext/build_m01/bigmoe_ext.cpython-312-darwin.so ] || { say "no ext/build_m01: not run"; exit 2; }
KEYS="humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91,humaneval/HumanEval/6,humaneval/HumanEval/79"
KEYS="$KEYS,humaneval/HumanEval/19,humaneval/HumanEval/161,humaneval/HumanEval/17,humaneval/HumanEval/56,humaneval/HumanEval/141"
KEYS="$KEYS,humaneval/HumanEval/101,humaneval/HumanEval/63"
KEYS="$KEYS,mbpp/16,mbpp/100,mbpp/146,mbpp/161,mbpp/183,mbpp/208,mbpp/224,mbpp/246,mbpp/268,mbpp/296,mbpp/305,mbpp/319"
REF="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=1 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12"
X="$REF BIGMOE_DUPROW_CHUNK=5 BIGMOE_MAXK=80 BIGMOE_FRSPEC=1 BIGMOE_SCOUT_BUNDLE=1 MTPLX_QWEN4_VERIFY_GLUE=1 BIGMOE_MTP_STREAM=1"
V3="BIGMOE_TIER=4433333333 BIGMOE_TIER_GLU=3c BIGMOE_TIER_COMBINE=1 BIGMOE_OOO=1"
FULL="BIGMOE_GDN_EXACT=1 $V3 BIGMOE_TIER_PF_GROUP=64 BIGMOE_PF_NOLOGITS=1 BIGMOE_EVICT_LFU=0.99 BIGMOE_VERIFY_ROWS_SDPA=1"
A_ENV="BIGMOE_JOINT=1 BIGMOE_MTP_PF_GROUP=48 BIGMOE_M01=1"
[ -n "$E24_DRYRUN" ] && { echo "dry run: $(echo "$KEYS" | tr ',' '\n' | grep -c .) keys; env $(echo $X $FULL $A_ENV | wc -w | tr -d ' ') K=V + BIGMOE_WIRED=1"; exit 0; }
run() {   # $1 tag, $2 keys, $3 N
  local T="$OUT/$1"; rm -rf "$T"; mkdir -p "$T"
  while [ -e "$HOME/.local/share/quiet/PAUSE" ]; do sleep 10; done
  say "run $1 ($(echo "$2" | tr ',' '\n' | grep -c .) prompts, N $3)"
  perl -e 'alarm shift; exec @ARGV' 1500 env $X $FULL $A_ENV BIGMOE_WIRED=1 S10B_KEYS=$2 S10B_X_IDX=999 BIGMOE_EXT_BUILD=build_m01 \
    BIGMOE_S10B=1 BIGMOE_S10B_MISS=$T/miss.bin "$PY" scripts/s10b_trace.py 0.25 "$3" "$T" > "$OUT/$1.txt" 2>&1
  local rc=$?; say "done $1 rc=$rc"; return $rc
}
rm -f "$OUT/DONE"
say "E24 start (pid $$, NO-SWAP $A GB)"
run e24pf humaneval/HumanEval/68 32
grep -q '"n_tok"' "$OUT/e24pf/meta.json" 2>/dev/null && ! grep -qE 'Traceback|Error' "$OUT/e24pf.txt" || {
  say "PRE-FLIGHT NOT CLEAN ($(grep -m1 -E 'Error|Traceback' "$OUT/e24pf.txt" | cut -c1-160)): no E24"; touch "$OUT/DONE"; exit 0; }
grep -q 'BIGMOE_WIRED: mx.set_wired_limit' "$OUT/e24pf.txt" || { say "the wired limit did not engage: no E24"; touch "$OUT/DONE"; exit 0; }
say "PRE-FLIGHT CLEAN: the passes run"
run e24a "$KEYS" 256
run e24b "$KEYS" 256
"$PY" -I - "$OUT" "$KEYS" >> "$LOG" 2>&1 <<'EOF'
import hashlib, json, math, os, statistics as S, sys
D, KEYS = sys.argv[1], sys.argv[2].split(",")
KBD = {}                                           # s10b names prompts p<NN>_<domain><i>: domain + i -> EvalSuite's key (S10B_KEYS order)
for k in KEYS:
    KBD.setdefault({"humaneval": "humaneval", "mbpp": "mbpp"}[k.split("/")[0]], []).append(k)
g = lambda xs: (lambda ys: math.exp(sum(map(math.log, ys)) / len(ys)) if ys else float("nan"))([x for x in xs if x and x > 0])
AR = json.load(open(os.path.expanduser("~/BigMoE/data/e5/ar_ref_tokens.json")))
M = {}
for r in ("e24a", "e24b"):
    try:
        M[r] = {p["name"]: p for p in json.load(open(os.path.join(D, r, "meta.json")))["prompts"]}
    except (OSError, ValueError):
        M[r] = {}
L = [f"E24, the 24 GB EXACT number (data/e24): W1B's 24 GB stack (s10b, stock kernels, C 128, MAXK 80 + DUPROW) + BIGMOE_WIRED=1, "
     f"N 256, passes e24a / e24b ({len(M['e24a'])} / {len(M['e24b'])} prompts)"]
names = sorted(set(M["e24a"]) | set(M["e24b"]))
dom = {n: (M["e24a"].get(n) or M["e24b"].get(n))["domain"] for n in names}
key_of = {n: KBD[dom[n]][(M["e24a"].get(n) or M["e24b"].get(n))["idx"]] for n in names}
sets = {"12 HumanEval": [n for n in names if dom[n] == "humaneval"], "MBPP12": [n for n in names if dom[n] == "mbpp"]}
pk = max([p.get("lifetime_fp_gb", 0) for r in M.values() for p in r.values()] or [float("nan")])
for sname, ns in sets.items():
    pm = lambda f: [sum(f(M[r][n]) for r in M if n in M[r]) / sum(1 for r in M if n in M[r]) for n in ns if any(n in M[r] for r in M)]
    tps = pm(lambda p: p["tok_s"]); e2e = pm(lambda p: p["n_tok"] / p["wall"]); mb = pm(lambda p: p["disk_read"] / 1e6 / max(1, p["n_tok"]))
    pas = ", ".join(f"{g([M[r][n]['tok_s'] for n in ns if n in M[r]]):.1f}" for r in M)
    L.append(f"  {sname}: WHOLE DECODE {g(tps):.1f} tok/s (passes {pas}); END TO END {g(e2e):.1f}; SSD MB per token {S.mean(mb) if mb else float('nan'):.1f}; "
             f"tokens / task {S.median([M['e24a'][n]['n_tok'] for n in ns if n in M['e24a']] or [float('nan')]):.0f}; full period n/a")
same = sum(1 for n in names if n in M["e24a"] and n in M["e24b"] and M["e24a"][n]["sha"] == M["e24b"][n]["sha"])
both = sum(1 for n in names if n in M["e24a"] and n in M["e24b"])
arq = 0; arn = 0
for n in names:
    p = M["e24a"].get(n)
    k = key_of.get(n)
    if p is None or k not in AR:
        continue
    arn += 1
    ref = AR[k]["gate+t"][0][: p["n_tok"]]
    arq += hashlib.sha1(str(list(ref)).encode()).hexdigest()[:10] == p["sha"]
L.append(f"  lifetime peak {pk:.2f} GB (line <= 24.0: {'OK' if pk <= 24.0 else 'OVER'}); pass 1 == pass 2 on {same} / {both} prompts "
         f"(its stock-kernel determinism, as Wed's w24 == w24t); agreement with NAX6-N's AR (sha of AR[:n], information) {arq} / {arn}")
open(os.path.join(D, "e24_report.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
EOF
say "report rc=$? (data/e24/e24_report.txt)"
git add scripts/s10b_trace.py data/e24/e24.log data/e24/e24_report.txt data/e24/*.txt data/e24/e24a/meta.json data/e24/e24b/meta.json \
  data/e24/e24pf/meta.json 2>/dev/null
> (a quote from the project owner, kept private)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011DvdV6eUB4NgBbVkyVFPrT"
touch "$OUT/DONE"
say "E24 end"
