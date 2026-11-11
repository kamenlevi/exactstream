#!/bin/bash
# Sat 7 Nov 18:5x (NewIdeas item 2): E24m, E24's MBPP12 with an EOS stop, so the 24 GB exact MBPP cell is comparable with r57's rows.
# E24's stack (W1B's s10b 24 GB stack + BIGMOE_WIRED=1, Wed's ext/build_m01, C 128) + S10B_EOS=1 (spec125's stop_ids = r57's EOS set
# {248044, 248046}) + S10B_SAVE_IDS=1 (token ids in meta for a real AR comparison). MBPP12, N 256, two passes (e24ma, e24mb).
# REPORT (data/e24m/e24m_report.txt): WHOLE DECODE, END TO END, SSD MB per token, tokens per task, lifetime peak (<= 24.0), pass 1 ==
#   pass 2 (ids), agreement with NAX6-N's AR (ids, first min(n, AR) tokens; information: stock kernels differ bitwise).
# usage: NOTBEFORE=21:00 NOTAFTER=00:30 NOWAIT45=1 GAP0=0 GAP=0 MARGIN=4 bash scripts/turn45.sh 26 20 "..." /bin/bash scripts/e24m.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "${E24_ROOT:-$HOME/BigMoE}" || exit 1
OUT="$PWD/data/e24m"; mkdir -p "$OUT"
LOG="$OUT/e24m.log"
PY="$HOME/BigMoE/.venv/bin/python"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
A=${E24_AVAIL:-$(avail_gb)}
awk -v a="$A" 'BEGIN {exit !(a >= 30)}' || { say "NO-SWAP $A GB < 30: not run"; exit 0; }
[ -n "$E24_DRYRUN" ] || sleep 20
A=${E24_AVAIL:-$(avail_gb)}
awk -v a="$A" 'BEGIN {exit !(a >= 30)}' || { say "NO-SWAP $A GB < 30 at the 20 s look-again: not run"; exit 0; }
KEYS="mbpp/16,mbpp/100,mbpp/146,mbpp/161,mbpp/183,mbpp/208,mbpp/224,mbpp/246,mbpp/268,mbpp/296,mbpp/305,mbpp/319"
REF="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=1 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12"
X="$REF BIGMOE_DUPROW_CHUNK=5 BIGMOE_MAXK=80 BIGMOE_FRSPEC=1 BIGMOE_SCOUT_BUNDLE=1 MTPLX_QWEN4_VERIFY_GLUE=1 BIGMOE_MTP_STREAM=1"
V3="BIGMOE_TIER=4433333333 BIGMOE_TIER_GLU=3c BIGMOE_TIER_COMBINE=1 BIGMOE_OOO=1"
FULL="BIGMOE_GDN_EXACT=1 $V3 BIGMOE_TIER_PF_GROUP=64 BIGMOE_PF_NOLOGITS=1 BIGMOE_EVICT_LFU=0.99 BIGMOE_VERIFY_ROWS_SDPA=1"
A_ENV="BIGMOE_JOINT=1 BIGMOE_MTP_PF_GROUP=48 BIGMOE_M01=1"
[ -n "$E24_DRYRUN" ] && { echo "dry run: $(echo "$KEYS" | tr ',' '\n' | grep -c .) MBPP keys; S10B_EOS=1 S10B_SAVE_IDS=1 BIGMOE_WIRED=1"; exit 0; }
rm -f "$OUT/DONE"
say "E24m start (pid $$, NO-SWAP $A GB)"
for R in e24ma e24mb; do
  T="$OUT/$R"; rm -rf "$T"; mkdir -p "$T"
  while [ -e "$HOME/.local/share/quiet/PAUSE" ]; do sleep 10; done
  say "run $R"
  perl -e 'alarm shift; exec @ARGV' 1200 env $X $FULL $A_ENV BIGMOE_WIRED=1 S10B_EOS=1 S10B_SAVE_IDS=1 S10B_KEYS=$KEYS S10B_X_IDX=999 \
    BIGMOE_EXT_BUILD=build_m01 BIGMOE_S10B=1 BIGMOE_S10B_MISS=$T/miss.bin "$PY" scripts/s10b_trace.py 0.25 256 "$T" > "$OUT/$R.txt" 2>&1
  say "done $R rc=$?"
done
"$PY" -I - "$OUT" "$KEYS" >> "$LOG" 2>&1 <<'EOF'
import json, math, os, statistics as S, sys
D, KEYS = sys.argv[1], sys.argv[2].split(",")
g = lambda xs: (lambda ys: math.exp(sum(map(math.log, ys)) / len(ys)) if ys else float("nan"))([x for x in xs if x and x > 0])
AR = json.load(open(os.path.expanduser("~/BigMoE/data/e5/ar_ref_tokens.json")))
M = {}
for r in ("e24ma", "e24mb"):
    try:
        M[r] = {p["idx"]: p for p in json.load(open(os.path.join(D, r, "meta.json")))["prompts"]}
    except (OSError, ValueError):
        M[r] = {}
ix = sorted(set(M["e24ma"]) | set(M["e24mb"]))
pm = lambda f: [sum(f(M[r][i]) for r in M if i in M[r]) / sum(1 for r in M if i in M[r]) for i in ix]
pk = max([p.get("lifetime_fp_gb", 0) for r in M.values() for p in r.values()] or [float("nan")])
same = sum(1 for i in ix if i in M["e24ma"] and i in M["e24mb"] and M["e24ma"][i].get("ids") == M["e24mb"][i].get("ids"))
arq = 0
for i in ix:
    out, ref = (M["e24ma"].get(i) or {}).get("ids") or [], AR[KEYS[i]]["gate+t"][0]
    m = min(len(out), len(ref))
    arq += m > 0 and out[:m] == ref[:m]
L = [f"E24m: E24's stack (W1B's s10b 24 GB + wired) on MBPP12 WITH the EOS stop (data/e24m): passes e24ma / e24mb ({len(M['e24ma'])} / {len(M['e24mb'])})",
     f"  MBPP12: WHOLE DECODE {g(pm(lambda p: p['tok_s'])):.1f} tok/s (passes {', '.join(f'{g([p['tok_s'] for p in M[r].values()]):.1f}' for r in M)}); "
     f"END TO END {g(pm(lambda p: p['n_tok'] / p['wall'])):.1f}; SSD MB per token {S.mean(pm(lambda p: p['disk_read'] / 1e6 / max(1, p['n_tok']))) if ix else float('nan'):.1f}; "
     f"tokens / task {S.median(pm(lambda p: p['n_tok'])) if ix else float('nan'):.0f}",
     f"  lifetime peak {pk:.2f} GB (<= 24.0: {'OK' if pk <= 24.0 else 'OVER'}); pass 1 == pass 2 (ids) {same} / {len(ix)}; agreement with NAX6-N's AR "
     f"(ids over min(n, AR), information: stock kernels differ bitwise) {arq} / {len(ix)}"]
open(os.path.join(D, "e24m_report.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
EOF
say "report rc=$? (data/e24m/e24m_report.txt)"
git add scripts/s10b_trace.py data/e24m/e24m.log data/e24m/e24m_report.txt data/e24m/*.txt data/e24m/e24ma/meta.json data/e24m/e24mb/meta.json 2>/dev/null
> (a quote from the project owner, kept private)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011DvdV6eUB4NgBbVkyVFPrT"
touch "$OUT/DONE"
say "E24m end"
