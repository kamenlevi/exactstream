#!/bin/bash
# Sun 8 Nov 08:5x. PFB: PF-B on top of PF-A (Leap 08:2x; Kamen's MUST prefill >= 500 tok/s, 1,000 better) + NewIdeas' CT1 (chat prefill
# price of PF-D). PFX: PF-A (R57_PFLM=1) is exact and x1.9 / x2.9 / x3.4 (195 / 368 / 441 tok/s at 1.6K / 4.7K / 7.9K), compute-bound past 1.6K.
# R57_PFLM=3 (scripts/pflm.py _split_layers): each layer split at its MoE; every chunk's attention part goes to the GPU (async), the CPU
# reads all the layer's experts meanwhile (R57_PFLM_AHEAD), then one padded MoE call per <= R57_PFLM_TOK tokens (Leap's fix for MLX
# 0.32.2's sorted-gather row bug past 32,767 rows), then each chunk's residual write.
# Arms (EXACT E, C 256, CAP 64, the 3 PFX agent prompts, R57_PFSTAT + R57_LOGHASH, one process each):
#   pa1 = PF-A (control, PFX's plm1); pb3 = PF-B (TOK 4096, read-ahead); pb3n = PF-B without the read-ahead (the MoE batching alone);
#   pb38 = PF-B at TOK 8192 (one MoE call per layer up to 8K tokens); pb3c = PF-B + mx.clear_cache() per layer (the 45.0 line).
# CT1 (NewIdeas 08:19, pre-registered in ~/NewIdeas QUEUE c5a7e2c): zs (M1-B + COH + SB1) on the 12 IFEval prompts, CAP 16, R57_PFSTAT:
#   chat prefill SSD bytes, pool residency at prefill start, and the routed expert-layers that were already resident (PF-D's exact price).
# REPORT (data/pfb/pfb_report.txt): TTFT / tok/s per arm and prompt, x vs pa1, decode tokens and logits hash == pa1, lifetime peak, scratch
#   counters; CT1 per prompt and mean.
# usage: AFTER=... NOTAFTER=hh:mm NOWAIT45=1 GAP0=0 GAP=0 MARGIN=4 bash scripts/turn45.sh 33 20 "..." /bin/bash scripts/pfb.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "${PFB_ROOT:-$HOME/BigMoE}" || exit 1
D=data/pfb; mkdir -p $D
LOG="$PWD/$D/pfb.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
PY=${PFB_PY:-$HOME/BigMoE/.venv/bin/python}
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
A=${PFB_AVAIL:-$(avail_gb)}
awk -v a="$A" 'BEGIN {exit !(a >= 47.2)}' || { say "NO-SWAP $A GB < 47.2: not run"; exit 0; }
[ -n "$PFB_DRYRUN" ] || sleep 20
A=${PFB_AVAIL:-$(avail_gb)}
awk -v a="$A" 'BEGIN {exit !(a >= 47.2)}' || { say "NO-SWAP $A GB < 47.2 at the 20 s look-again: not run"; exit 0; }
BUN="BIGMOE_PAUSE_EVERY_MS=50,R57_S0CARRY=1,REF2_SB_FIRST=1,REF2_PLE_LATE=1,REF2_COMMIT_SUB=9,R57_DQ1=1"
E="R57_NAX=1,REF2_KEEP_CACHES=1,R57_WIRED=1,$BUN"
L=$(grep -v '^#' data/stack/s5p_env.txt | grep . | tail -1 | sed 's/,R57_QOM_SETS=humaneval,mbpp$//')
ZS="$L,R57_QOM_MTP=2,REF2_NAX_COH=1,R57_SB1=1"
KP="agent/edit10,agent/write13,agent/write11"
KI=$(sed -n 1p data/n45/keys.txt)
FR=$(bash scripts/deep_freeze.sh pfb) || { say "freeze failed: $FR"; exit 2; }
SD=data/r57/snap/pfb
grep -q '_tap_out' "$SD/r57_screen_pfb.py" && grep -q '_split_layers' "$SD/pflm.py" && grep -q 'routed_res_base' "$SD/r57_screen_pfb.py" \
  || { say "the frozen screen / pflm.py has no PF-B (_tap_out, _split_layers) or CT1 (routed_res_base): not run"; exit 2; }
[ -n "$PFB_DRYRUN" ] && { echo "dry run: $FR; E $(echo "$E" | tr ',' '\n' | grep -c =) K=V, ZS $(echo "$ZS" | tr ',' '\n' | grep -c =) K=V, KI $(echo "$KI" | tr ',' '\n' | grep -c .)"; exit 0; }
git add data/r57/snap/frozen_pfb ':!data/r57/snap/frozen_pfb/**/__pycache__/**' "$SD" ":!$SD/__pycache__/**" data/r57/snap/c45_runner_pfb.sh \
> (a quote from the project owner, kept private)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011DvdV6eUB4NgBbVkyVFPrT" || { say "snapshot commit failed: not run"; exit 3; }
SNAP=$(git rev-parse --short HEAD)
RUN="/bin/bash data/r57/snap/c45_runner_pfb.sh"
export C45_OUT="$PWD/$D" C45_SWAP0=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}') C45_ALARM=2400 \
  MG=4 C45_DEADLINE=$(date -v+60M +%H%M) C45_SCREEN="$SD/r57_screen_pfb.py" C45_REFTOK= R57_QOM_SETS=humaneval,mbpp R57_MODMAN=1
rm -f $D/DONE
say "PFB start (pid $$, NO-SWAP $A GB, snapshot $SNAP)"
P0="R57_PFCHUNK=1024,R57_PFNOLOG=1"
for a in "pa1:$E,$P0,R57_PFLM=1" "pb3:$E,$P0,R57_PFLM=3" "pb3n:$E,$P0,R57_PFLM=3,R57_PFLM_AHEAD=0" "pb38:$E,$P0,R57_PFLM=3,R57_PFLM_TOK=8192" \
         "pb3c:$E,$P0,R57_PFLM=3,R57_PFLM_CC=1"; do
  t=${a%%:*}; e=${a#*:}; rm -f data/r57/${t}_256_* $D/${t}_256.txt
  R57_KEYS=$KP C45_CAP=64 R57_PFSTAT=1 R57_LOGHASH=1 C45_ARMS="${t}_256:256:$e" $RUN
  say "$t rc=$?"
done
rm -f data/r57/ct1_256_* $D/ct1_256.txt                  # CT1: the chat prompts' prefill (NewIdeas), TTFT + PFSTAT only
R57_KEYS=$KI C45_CAP=16 R57_PFSTAT=1 R57_QOM_SETS=humaneval,mbpp,ifeval,multi C45_ARMS="ct1_256:256:$ZS" $RUN
say "ct1 rc=$?"
rm -f data/r57/fb0r_256_* $D/fb0r_256.txt                # FU's fb0 (L at C 256, CAP 16) again: the orchestrator's publish job ran beside
R57_KEYS=$KP C45_CAP=16 C45_ARMS="fb0r_256:256:$L" $RUN  # FU's quiet lock 09:24:44-09:29 (fb0 + the start of fb1): TTFT reference for fb4 / fb5
say "fb0r rc=$?"
"$PY" -I - "$SNAP" >> "$LOG" 2>&1 <<'EOF'
import json, re, sys, statistics as S
snap = sys.argv[1]
peak = {}
for ln in open("data/pfb/c45.log"):
    m = re.search(r"done (\S+) rc=\d+:.*lifetime peak ([\d.]+) GB", ln)
    if m:
        peak[m.group(1)] = float(m.group(2))
def load(t):
    try:
        return ({r["key"]: r for r in json.load(open(f"data/r57/{t}_256_runs.json"))},
                {k: list(v.values())[0][0] for k, v in json.load(open(f"data/r57/{t}_256_tokens.json")).items()})
    except (OSError, ValueError, KeyError):
        return {}, {}
R0, T0 = load("pa1")
L = [f"PFB (data/pfb; snapshot {snap}; exact E, C 256, CAP 64; quiet): prefill TTFT (pfs) and tok/s; x vs pa1 (PF-A); decode tokens and "
     f"logits hash vs pa1; lifetime peak (line 45.0)"]
for t, nm in (("pa1", "PF-A (R57_PFLM=1, control)"), ("pb3", "PF-B (PFLM=3, TOK 4096, read-ahead)"), ("pb3n", "PF-B without the read-ahead"),
              ("pb38", "PF-B at TOK 8192"), ("pb3c", "PF-B + clear_cache per layer")):
    R, T = load(t)
    cells = []
    for k in ("agent/edit10", "agent/write13", "agent/write11"):
        r, r0 = R.get(k), R0.get(k) or {}
        if r and r.get("pfs"):
            same = "==" if T.get(k) == T0.get(k) and T0.get(k) is not None else "DIFF"
            h, h0 = r.get("lh") or [], r0.get("lh") or []
            x = f"x{r0['pfs'] / r['pfs']:.2f}" if r0.get("pfs") else "x?"
            cells.append(f"{k.split('/')[-1]} P {r['P']}: {r['pfs']:.1f} s = {r['P'] / r['pfs']:.0f} tok/s ({x}; tokens {same}; logits hash "
                         f"{sum(1 for a_, b_ in zip(h, h0) if a_ == b_)}/{min(len(h), len(h0))}; scratch {r.get('pflm')})")
        else:
            cells.append(f"{k.split('/')[-1]}: MISSING")
    L.append(f"  {t} {nm}: lifetime peak {peak.get(f'{t}_256', float('nan')):.2f} GB")
    L += [f"      {c}" for c in cells]
R, _ = load("ct1")
rows = [(k, r) for k, r in sorted(R.items()) if r.get("pfst")]
L.append(f"CT1 (NewIdeas; zs = M1-B + COH + SB1 on the 12 IFEval prompts, CAP 16; lifetime peak {peak.get('ct1_256', float('nan')):.2f} GB): per prompt "
         "P, TTFT, prefill SSD GB, expert-layers resident at start (base / side, of 24576), routed expert-layers and those resident (base / side)")
for k, r in rows:
    s = r["pfst"]
    L.append(f"  {k}: P {r['P']}, TTFT {r['pfs']:.2f} s, SSD {s['gb']:.1f} GB, resident base {s['res_base']} / side {s['res_side']}, routed "
             f"{s.get('routed')} of which resident base {s.get('routed_res_base')} / side {s.get('routed_res_side')}")
if rows:
    f = lambda key: S.mean([r["pfst"].get(key) or 0 for _, r in rows])
    rt = sum(r["pfst"].get("routed") or 0 for _, r in rows)
    L.append(f"  MEAN: TTFT {S.mean([r['pfs'] for _, r in rows]):.2f} s, SSD {f('gb'):.1f} GB, base residency at start {f('res_base') / 24576:.1%}, "
             f"routed expert-layers {f('routed'):.0f}, resident share of the routed: base "
             f"{sum(r['pfst'].get('routed_res_base') or 0 for _, r in rows) / max(rt, 1):.1%} / side "
             f"{sum(r['pfst'].get('routed_res_side') or 0 for _, r in rows) / max(rt, 1):.1%}")
try:                                                     # FU's arms: their lifetime peaks are in data/fu/c45.log
    for ln in open("data/fu/c45.log"):
        m = re.search(r"done (\S+) rc=\d+:.*lifetime peak ([\d.]+) GB", ln)
        if m:
            peak.setdefault(m.group(1), float(m.group(2)))
except OSError:
    pass
L.append("FU re-check (the orchestrator's publish job ran beside FU's quiet lock 09:24:44-09:29, over fb0): L at C 256, CAP 16, TTFT s by prompt")
for t, nm in (("fb0", "FU's L at C 256 (perturbed)"), ("fb0r", "the same, re-run in PFB"), ("fb4", "FU's L + PF-A"), ("fb5", "FU's L + PF-A + CACHE_GB 2")):
    R, _ = load(t)
    tt = {k.split("/")[-1]: round(r.get("pfs") or float("nan"), 1) for k, r in sorted(R.items())}
    L.append(f"  {t} {nm}: lifetime peak {peak.get(f'{t}_256', float('nan')):.2f} GB; TTFT {tt}")
open("data/pfb/pfb_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
EOF
say "report rc=$? ($D/pfb_report.txt)"
git add $D $(for t in pa1 pb3 pb3n pb38 pb3c ct1 fb0r; do echo data/r57/${t}_256_*; done) 2>/dev/null
> (a quote from the project owner, kept private)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011DvdV6eUB4NgBbVkyVFPrT"
touch $D/DONE
say "PFB end"
