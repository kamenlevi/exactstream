#!/bin/bash
# Sun 8 Nov 00:3x: PF-A smoke before CERT (scripts/pflm.py has never run): exact E on agent/edit10 (1,565 tokens = 2 chunks),
# CAP 16, R57_LOGHASH + R57_PFSTAT, three arms in one ticket: chunk-major with no-logits (pa0 = pe1024nl's config), R57_PFLM=1
# (pa1: full-E scratch, original ids), R57_PFLM=2 (pa2: compact copy). Report (data/pfa/pfa_report.txt): TTFT (pfs), tokens and
# per-check logits hashes vs pa0, the scratch counters (reads / reuses / calls), prefill SSD GB, lifetime peak.
# usage: AFTER=... NOTAFTER=hh:mm NOWAIT45=1 GAP0=0 GAP=0 MARGIN=4 bash scripts/turn45.sh 33 10 "..." /bin/bash scripts/pfa_smoke.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
D=data/pfa; mkdir -p $D
LOG="$PWD/$D/pfa.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
PY=$HOME/BigMoE/.venv/bin/python
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
END_E=$(date -j -f '%Y-%m-%d %H:%M' "$(date +%Y-%m-%d) ${PFA_END:-01:18}" +%s)
[ $END_E -lt $(date +%s) ] && END_E=$((END_E + 86400))
[ $((END_E - $(date +%s))) -gt 43200 ] && { say "PFA_END is > 12 h away (passed?): not run"; exit 0; }
A=$(avail_gb); awk -v a="$A" 'BEGIN {exit !(a >= 47.2)}' || { say "NO-SWAP $A GB < 47.2: not run"; exit 0; }
BUN="BIGMOE_PAUSE_EVERY_MS=50,R57_S0CARRY=1,REF2_SB_FIRST=1,REF2_PLE_LATE=1,REF2_COMMIT_SUB=9,R57_DQ1=1"
E="R57_NAX=1,REF2_KEEP_CACHES=1,R57_WIRED=1,$BUN,R57_PFCHUNK=1024,R57_PFNOLOG=1"
FR=$(bash scripts/deep_freeze.sh pfa) || { say "freeze failed: $FR"; exit 2; }
SD=data/r57/snap/pfa
grep -q 'R57_PFLM' "$SD/r57_screen_pfa.py" && [ -f "$SD/pflm.py" ] || { say "the frozen screen has no R57_PFLM / pflm.py: not run"; exit 2; }
git add data/r57/snap/frozen_pfa ':!data/r57/snap/frozen_pfa/**/__pycache__/**' "$SD" ":!$SD/__pycache__/**" data/r57/snap/c45_runner_pfa.sh \
> (a quote from the project owner, kept private)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011DvdV6eUB4NgBbVkyVFPrT" || { say "snapshot commit failed: not run"; exit 3; }
SNAP=$(git rev-parse --short HEAD)
export C45_OUT="$PWD/$D" C45_SWAP0=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}') C45_ALARM=300 MG=4 \
  C45_DEADLINE=$(date -r $END_E +%H%M) C45_SCREEN="$SD/r57_screen_pfa.py" C45_REFTOK= R57_QOM_SETS=humaneval,mbpp
say "PF-A smoke start (pid $$, NO-SWAP $A GB, snapshot $SNAP)"
for a in "pa0:$E" "pa1:$E,R57_PFLM=1" "pa2:$E,R57_PFLM=2"; do
  t=${a%%:*}; e=${a#*:}
  [ $(( $(date +%s) + 120 )) -le $END_E ] || { say "$t skipped: it could not end before $(date -r $END_E +%H:%M)"; continue; }
  rm -f data/r57/${t}_256_* $D/${t}_256.txt
  R57_KEYS=agent/edit10 C45_CAP=16 R57_PFSTAT=1 R57_LOGHASH=1 C45_ARMS="${t}_256:256:$e" /bin/bash data/r57/snap/c45_runner_pfa.sh
  say "$t rc=$?"
done
"$PY" -I - "$SNAP" >> "$LOG" 2>&1 <<'EOF'
import json, re, sys
peak = {}
for ln in open("data/pfa/c45.log"):
    m = re.search(r"done (\S+) rc=\d+:.*lifetime peak ([\d.]+) GB", ln)
    if m:
        peak[m.group(1)] = float(m.group(2))
def load(t):
    try:
        return ({r["key"]: r for r in json.load(open(f"data/r57/{t}_256_runs.json"))},
                {k: list(v.values())[0][0] for k, v in json.load(open(f"data/r57/{t}_256_tokens.json")).items()})
    except (OSError, ValueError, KeyError):
        return {}, {}
k = "agent/edit10"
R0, T0 = load("pa0")
L = [f"PF-A smoke (data/pfa; snapshot {sys.argv[1]}; exact E, agent/edit10 1565 tokens = 2 chunks of <= 1024, CAP 16, quiet)"]
for t in ("pa0", "pa1", "pa2"):
    R, T = load(t)
    r, r0 = R.get(k) or {}, R0.get(k) or {}
    if not r.get("pfs"):
        L.append(f"  {t}: MISSING (see data/pfa/{t}_256.txt)"); continue
    h, h0 = r.get("lh") or [], r0.get("lh") or []
    L.append(f"  {t}: TTFT {r['pfs']:.2f} s ({r['P'] / r['pfs']:.0f} tok/s); tokens {'==' if T.get(k) == T0.get(k) else 'DIFFER from'} pa0's; "
             f"logits hash == {sum(1 for a_, b_ in zip(h, h0) if a_ == b_)}/{min(len(h), len(h0))}; prefill SSD "
             f"{(r.get('pfst') or {}).get('gb', float('nan')):.1f} GB; scratch {r.get('pflm')}; lifetime peak {peak.get(f'{t}_256', float('nan')):.2f} GB")
open("data/pfa/pfa_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
EOF
say "report rc=$? ($D/pfa_report.txt)"
git add $D data/r57/pa0_256_* data/r57/pa1_256_* data/r57/pa2_256_* 2>/dev/null
> (a quote from the project owner, kept private)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011DvdV6eUB4NgBbVkyVFPrT"
say "PF-A smoke end"
