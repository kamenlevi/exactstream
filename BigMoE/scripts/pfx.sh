#!/bin/bash
# Sun 8 Nov (the orchestrator Sat 23:4x: Kamen's MUST, prefill >= 500 tok/s, 1,000 better; the top engine task). PFX, the chunk-size step of
# expert-once prefill. Today's prefill (BIGMOE_PF_PREFILL / PF1, bigmoe/stream.py:134) streams, per prefill CALL and per layer, all of the next
# layer's non-resident experts into a staging buffer; the screen calls the model per 1,024-row chunk, so a 7.9K prompt re-streams ~half of every
# layer's experts 8 times (N45: TTFT 64 s, 124 tok/s). One call over the whole prompt streams each expert about once (~34 GB at C 256).
# New deep-frozen snapshot (R57_PFCHUNK knob). C 256, CAP 64, agent/edit10 (1565), agent/write13 (4730), agent/write11 (7932), one process per arm:
#   EXACT E (R57_PRESET=45 + NAX6-N + KEEP + wired + the Sunday bundle) at R57_PFCHUNK 1024 / 2048 / 4096 / 8192, then LOSSY L (S5''s config)
#   at 8192 (FLAG 2: L already peaks 45.1 GB at 1024 on the 7.9K prompt).
# REPORT (data/pfx/pfx_report.txt): per arm and prompt TTFT (pfs) and prefill tok/s; lifetime peak per arm (line 45.0); decode tokens
#   identical to the 1024 arm's (exactness across chunk sizes; the full-length R-56 audit vs NAX6-N comes after); PF1's split printout.
# usage: AFTER=... NOTAFTER=hh:mm NOWAIT45=1 GAP0=0 GAP=0 MARGIN=4 bash scripts/turn45.sh 33 40 "..." /bin/bash scripts/pfx.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "${PFX_ROOT:-$HOME/BigMoE}" || exit 1
D=data/pfx; mkdir -p $D
LOG="$PWD/$D/pfx.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
PY=${PFX_PY:-$HOME/BigMoE/.venv/bin/python}
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
A=${PFX_AVAIL:-$(avail_gb)}
awk -v a="$A" 'BEGIN {exit !(a >= 47.2)}' || { say "NO-SWAP $A GB < 47.2: not run"; exit 0; }
[ -n "$PFX_DRYRUN" ] || sleep 20
A=${PFX_AVAIL:-$(avail_gb)}
awk -v a="$A" 'BEGIN {exit !(a >= 47.2)}' || { say "NO-SWAP $A GB < 47.2 at the 20 s look-again: not run"; exit 0; }
BUN="BIGMOE_PAUSE_EVERY_MS=50,R57_S0CARRY=1,REF2_SB_FIRST=1,REF2_PLE_LATE=1,REF2_COMMIT_SUB=9,R57_DQ1=1"
E="R57_NAX=1,REF2_KEEP_CACHES=1,R57_WIRED=1,$BUN"
L=$(grep -v '^#' data/stack/s5p_env.txt | grep . | tail -1 | sed 's/,R57_QOM_SETS=humaneval,mbpp$//')
KP="agent/edit10,agent/write13,agent/write11"
FR=$(bash scripts/deep_freeze.sh pfx) || { say "freeze failed: $FR"; exit 2; }
SD=data/r57/snap/pfx
grep -q 'R57_PFCHUNK' "$SD/r57_screen_pfx.py" || { say "the frozen screen has no R57_PFCHUNK: not run"; exit 2; }
grep -q 'R57_PFLM' "$SD/r57_screen_pfx.py" && [ -f "$SD/pflm.py" ] || { say "the frozen screen has no R57_PFLM / pflm.py: not run"; exit 2; }
[ -n "$PFX_DRYRUN" ] && { echo "dry run: $FR; E $(echo "$E" | tr ',' '\n' | grep -c =) K=V, L $(echo "$L" | tr ',' '\n' | grep -c =) K=V"; exit 0; }
git add data/r57/snap/frozen_pfx ':!data/r57/snap/frozen_pfx/**/__pycache__/**' "$SD" ":!$SD/__pycache__/**" data/r57/snap/c45_runner_pfx.sh \
> (a quote from the project owner, kept private)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011DvdV6eUB4NgBbVkyVFPrT" || { say "snapshot commit failed: not run"; exit 3; }
SNAP=$(git rev-parse --short HEAD)
RUN="/bin/bash data/r57/snap/c45_runner_pfx.sh"
export C45_OUT="$PWD/$D" C45_SWAP0=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}') C45_ALARM=2400 \
  MG=4 C45_DEADLINE=$(date -v+120M +%H%M) C45_SCREEN="$SD/r57_screen_pfx.py" C45_REFTOK= R57_QOM_SETS=humaneval,mbpp R57_MODMAN=1
rm -f $D/DONE
say "PFX start (pid $$, NO-SWAP $A GB, snapshot $SNAP)"
for a in "pe1024:$E,R57_PFCHUNK=1024" "pe1024nl:$E,R57_PFCHUNK=1024,R57_PFNOLOG=1" "plm1:$E,R57_PFCHUNK=1024,R57_PFNOLOG=1,R57_PFLM=1" "plm2:$E,R57_PFCHUNK=1024,R57_PFNOLOG=1,R57_PFLM=2" "pe2048:$E,R57_PFCHUNK=2048" "pe4096:$E,R57_PFCHUNK=4096" "pe8192:$E,R57_PFCHUNK=8192" "pe8192nl:$E,R57_PFCHUNK=8192,R57_PFNOLOG=1" "pl8192:$L,R57_PFCHUNK=8192"; do
  t=${a%%:*}; e=${a#*:}; rm -f data/r57/${t}_256_* $D/${t}_256.txt
  R57_KEYS=$KP C45_CAP=64 R57_PFSTAT=1 R57_LOGHASH=1 C45_ARMS="${t}_256:256:$e" $RUN
  say "$t rc=$?"
done
"$PY" -I - "$SNAP" >> "$LOG" 2>&1 <<'EOF'
import json, re, sys
snap = sys.argv[1]
peak = {}
for ln in open("data/pfx/c45.log"):
    m = re.search(r"done (\S+) rc=\d+:.*lifetime peak ([\d.]+) GB", ln)
    if m:
        peak[m.group(1)] = float(m.group(2))
def load(t):
    try:
        return ({r["key"]: r for r in json.load(open(f"data/r57/{t}_256_runs.json"))},
                {k: list(v.values())[0][0] for k, v in json.load(open(f"data/r57/{t}_256_tokens.json")).items()})
    except (OSError, ValueError, KeyError):
        return {}, {}
R0, T0 = load("pe1024")
L = [f"PFX (data/pfx; snapshot {snap}; C 256, CAP 64; quiet): prefill TTFT (pfs: prefill + setup + the first token) and tok/s by R57_PFCHUNK"]
for t in ("pe1024", "pe1024nl", "plm1", "plm2", "pe2048", "pe4096", "pe8192", "pe8192nl", "pl8192"):
    R, T = load(t)
    cells = []
    for k in ("agent/edit10", "agent/write13", "agent/write11"):
        r = R.get(k)
        if r and r.get("pfs"):
            same = "==" if T.get(k) == T0.get(k) and T0.get(k) is not None else "DIFF"
            h, h0 = r.get("lh") or [], (R0.get(k) or {}).get("lh") or []
            hq = f"{sum(1 for a_, b_ in zip(h, h0) if a_ == b_)}/{min(len(h), len(h0))}"
            cells.append(f"{k.split('/')[-1]} P {r['P']}: {r['pfs']:.1f} s = {r['P'] / r['pfs']:.0f} tok/s (tokens {same} 1024's; logits hash == {hq})")
        else:
            cells.append(f"{k.split('/')[-1]}: MISSING")
    splits = [ln.strip()[:150] for ln in open(f"data/pfx/{t}_256.txt", errors="ignore") if "PF1 prefill split" in ln] if R else []
    L.append(f"  {t}: lifetime peak {peak.get(f'{t}_256', float('nan')):.2f} GB; " + "; ".join(cells))
    for s_ in splits[-3:]:
        L.append(f"      {s_}")
R8, T8 = load("pe8192")
k = "agent/write13"
if R0.get(k, {}).get("pfs") and R8.get(k, {}).get("pfs"):           # NewIdeas 00:1x, pre-registered before the run
    x = R0[k]["pfs"] / R8[k]["pfs"]; eq = T8.get(k) == T0.get(k); pk = peak.get("pe8192_256", 99.0)
    v = "PASS" if x >= 2.5 and eq and pk <= 45.0 else "KILL" if x < 1.5 else "BETWEEN"
    L.append(f"Verdict PFX (pre-registered, NewIdeas 00:1x): {v}: write13 (4.7K) TTFT x{x:.2f} at R57_PFCHUNK 8192 vs 1024; decode tokens "
             f"{'==' if eq else 'DIFFER from'} the 1024 arm's; lifetime peak {pk:.2f} GB (PASS >= x2.5 AND tokens == AND peak <= 45.0; KILL < x1.5)")
    for kk in ("agent/edit10", "agent/write11"):
        if R0.get(kk, {}).get("pfs") and R8.get(kk, {}).get("pfs"):
            L.append(f"  beside: {kk.split('/')[-1]} x{R0[kk]['pfs'] / R8[kk]['pfs']:.2f}; tokens {'==' if T8.get(kk) == T0.get(kk) else 'DIFFER'}")
else:
    L.append("Verdict PFX: MISSING (write13 at 1024 or 8192)")
for t, nm in (("plm1", "R57_PFLM=1 (full-E scratch, original ids)"), ("plm2", "R57_PFLM=2 (compact copy = today's call shape)")):
    Rl, Tl = load(t)                                     # PF-A (Sun 00:3x): layer-major over the 1024 chunks, each layer's experts read once
    for kk in ("agent/edit10", "agent/write13", "agent/write11"):
        r, r0 = Rl.get(kk) or {}, R0.get(kk) or {}
        if r.get("pfs") and r0.get("pfs"):
            h, h0 = r.get("lh") or [], r0.get("lh") or []
            L.append(f"PF-A {t} {kk.split('/')[-1]}: TTFT x{r0['pfs'] / r['pfs']:.2f} vs pe1024 ({r['P'] / r['pfs']:.0f} tok/s); decode tokens "
                     f"{'==' if Tl.get(kk) == T0.get(kk) else 'DIFFER from'} pe1024's; logits hash == "
                     f"{sum(1 for a_, b_ in zip(h, h0) if a_ == b_)}/{min(len(h), len(h0))}; scratch {r.get('pflm')}; {nm}")
    L.append(f"PF-A {t}: lifetime peak {peak.get(f'{t}_256', float('nan')):.2f} GB (line 45.0)")
L.append("PF-D price (R57_PFSTAT: experts resident at prefill start, which today's prefill re-reads from SSD; base = MSB-3 pool, side = LSB):")
for t in ("pe1024", "pe8192", "plm1", "plm2"):
    R, _ = load(t)
    for kk in ("agent/edit10", "agent/write13", "agent/write11"):
        s = (R.get(kk) or {}).get("pfst")
        if s:
            L.append(f"  {t} {kk.split('/')[-1]}: prefill SSD {s['gb']:.1f} GB; resident at start: base {s['res_base']} / side {s['res_side']} "
                     f"expert-layers (of 48 x 512 = 24576)")
open("data/pfx/pfx_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
EOF
say "report rc=$? ($D/pfx_report.txt)"
git add $D $(for t in pe1024 pe1024nl plm1 plm2 pe2048 pe4096 pe8192 pe8192nl pl8192; do echo data/r57/${t}_256_*; done) 2>/dev/null
> (a quote from the project owner, kept private)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011DvdV6eUB4NgBbVkyVFPrT"
touch $D/DONE
say "PFX end"
