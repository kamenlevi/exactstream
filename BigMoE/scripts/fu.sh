#!/bin/bash
# Sun 8 Nov (the orchestrator Sat 23:3x: two follow-ups for the publish config, AFTER CERT): FU, on a new deep-frozen snapshot (fu), C 256.
#   L = S5''s lossy config (data/stack/s5p_env.txt: v1' + wired + the Sunday bundle)
# A. FLAG 1 BISECT: which non-QOM lever breaks chat exactness (N45: L == NAX6-N's AR on only 6 / 24 IFEval runs, EB 24 / 24).
#    12 IFEval (N45's keys), one pass each: fa0 = L; fa1 = L - R57_FASTTAX; fa2 = L - REF2_HC8_FAST; fa3 = L - R57_SVH - R57_DSUB;
#    fa4 = L - REF2_DN_KS; fa5 = L - R57_RELAX_SPAN (+ its rule); fa6 = L - all of these. Report: runs == AR (of 12), whole decode.
#    A single lever whose removal gives 12 / 12 is the cause (then: can it switch off at the chat request boundary; its speed cost).
# B. FLAG 2: L peaked 45.1 GB on agent/write11 (7.9K tokens). Candidate fixes on the 3 agent prompts (CAP 16): fb0 = L at C 256;
#    fb1 = L at C 248; fb2 = L + R57_PFCHUNK=512; fb3 = L + R57_CACHE_GB=2 (mx.set_cache_limit) -> lifetime peak + TTFT.
#    Then decode cost on 12 HumanEval (one pass each): fh0 = L at 256, fh1 = L at 248, fh3 = L + R57_CACHE_GB=2.
# usage: AFTER='<CERT_DONE newer than Sun 01:30>' NOTAFTER=hh:mm NOWAIT45=1 GAP0=0 GAP=0 MARGIN=4 bash scripts/turn45.sh 33 50 "..." /bin/bash scripts/fu.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "${FU_ROOT:-$HOME/BigMoE}" || exit 1
D=data/fu; mkdir -p $D
LOG="$PWD/$D/fu.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
PY=${FU_PY:-$HOME/BigMoE/.venv/bin/python}
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
A=${FU_AVAIL:-$(avail_gb)}
awk -v a="$A" 'BEGIN {exit !(a >= 47.2)}' || { say "NO-SWAP $A GB < 47.2: not run"; exit 0; }
[ -n "$FU_DRYRUN" ] || sleep 20
A=${FU_AVAIL:-$(avail_gb)}
awk -v a="$A" 'BEGIN {exit !(a >= 47.2)}' || { say "NO-SWAP $A GB < 47.2 at the 20 s look-again: not run"; exit 0; }
L=$(grep -v '^#' data/stack/s5p_env.txt | grep . | tail -1 | sed 's/,R57_QOM_SETS=humaneval,mbpp$//')
drop() { echo "$L" | tr ',' '\n' | grep -vxE "$1" | paste -sd, -; }
FA1=$(drop 'R57_FASTTAX=1'); FA2=$(drop 'REF2_HC8_FAST=1'); FA3=$(drop 'R57_SVH=1|R57_DSUB=1'); FA4=$(drop 'REF2_DN_KS=1')
FA5=$(drop 'R57_RELAX_SPAN=1|R57_RELAX_SPAN_RULE=R2:2:0.25')
FA6=$(drop 'R57_FASTTAX=1|REF2_HC8_FAST=1|R57_SVH=1|R57_DSUB=1|REF2_DN_KS=1|R57_RELAX_SPAN=1|R57_RELAX_SPAN_RULE=R2:2:0.25')
for v in "$FA1" "$FA2" "$FA3" "$FA4" "$FA5" "$FA6"; do [ "$v" != "$L" ] || { say "a bisect arm drops nothing (s5p_env changed?): not run"; exit 2; }; done
KI=$(sed -n 1p data/n45/keys.txt)
KP="agent/edit10,agent/write13,agent/write11"
K="humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91,humaneval/HumanEval/6,humaneval/HumanEval/79"
K="$K,humaneval/HumanEval/19,humaneval/HumanEval/161,humaneval/HumanEval/17,humaneval/HumanEval/56,humaneval/HumanEval/141"
K="$K,humaneval/HumanEval/101,humaneval/HumanEval/63"
FR=$(bash scripts/deep_freeze.sh fu) || { say "freeze failed: $FR"; exit 2; }
SD=data/r57/snap/fu
[ -n "$FU_DRYRUN" ] && { echo "dry run: $FR; L $(echo "$L" | tr ',' '\n' | grep -c =) K=V; arms drop: $(for v in "$FA1" "$FA2" "$FA3" "$FA4" "$FA5" "$FA6"; do echo -n "$(( $(echo "$L" | tr ',' '\n' | grep -c =) - $(echo "$v" | tr ',' '\n' | grep -c =) )) "; done)"; exit 0; }
git add data/r57/snap/frozen_fu ':!data/r57/snap/frozen_fu/**/__pycache__/**' "$SD" ":!$SD/__pycache__/**" data/r57/snap/c45_runner_fu.sh \
> (a quote from the project owner, kept private)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011DvdV6eUB4NgBbVkyVFPrT" || { say "snapshot commit failed: not run"; exit 3; }
SNAP=$(git rev-parse --short HEAD)
RUN="/bin/bash data/r57/snap/c45_runner_fu.sh"
export C45_OUT="$PWD/$D" C45_SWAP0=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}') C45_ALARM=2400 \
  MG=4 C45_DEADLINE=$(date -v+150M +%H%M) C45_SCREEN="$SD/r57_screen_fu.py" R57_QOM_SETS=humaneval,mbpp R57_MODMAN=1
rm -f $D/DONE
say "FU start (pid $$, NO-SWAP $A GB, snapshot $SNAP)"
if [ -n "$FU_DO_A" ]; then                              # Sun 00:3x: part A is superseded by CHS1 (chs1_report: fa1-fa6 on the same 12 IFEval,
for a in "fa0:$L" "fa1:$FA1" "fa2:$FA2" "fa3:$FA3" "fa4:$FA4" "fa5:$FA5" "fa6:$FA6"; do   # fa2 dropping R57_HC8 too) + CHS3's forward
  t=${a%%:*}; e=${a#*:}; rm -f data/r57/${t}_256_* $D/${t}_256.txt   # bisect; its tags would overwrite chs1's fa* files: off unless FU_DO_A=1
  R57_KEYS=$KI R57_REFTOK="$PWD/data/n45/ar_ref_tokens.json:gate+t" C45_ARMS="${t}_256:256:$e" $RUN
  say "$t rc=$?"
done
else
  say "part A skipped (superseded by CHS1 data/chs/chs1_report.txt + CHS3)"
fi
for a in "fb0:256:$L" "fb1:248:$L" "fb2:256:$L,R57_PFCHUNK=512" "fb3:256:$L,R57_CACHE_GB=2" "fb4:256:$L,R57_PFLM=1" "fb5:256:$L,R57_PFLM=1,R57_CACHE_GB=2"; do   # fb4 / fb5 (Sun 08:0x): PF-A on L + the cache-limit trim (PFX plm1 peaked 45.10 GB on write11)
  t=${a%%:*}; r=${a#*:}; c=${r%%:*}; e=${r#*:}; rm -f data/r57/${t}_${c}_* $D/${t}_${c}.txt
  R57_KEYS=$KP C45_CAP=16 C45_ARMS="${t}_$c:$c:$e" $RUN
  say "$t rc=$?"
done
for a in "fh0:256:$L" "fh1:248:$L" "fh3:256:$L,R57_CACHE_GB=2"; do
  t=${a%%:*}; r=${a#*:}; c=${r%%:*}; e=${r#*:}; rm -f data/r57/${t}_${c}_* $D/${t}_${c}.txt
  R57_KEYS=$K C45_ARMS="${t}_$c:$c:$e" $RUN
  say "$t rc=$?"
done
"$PY" -I - "$SNAP" >> "$LOG" 2>&1 <<'EOF'
import json, math, re, statistics as S, sys
snap = sys.argv[1]
g = lambda xs: (lambda ys: math.exp(sum(map(math.log, ys)) / len(ys)) if ys else float("nan"))([x for x in xs if x and x > 0])
peak = {}
for ln in open("data/fu/c45.log"):
    m = re.search(r"done (\S+) rc=\d+:.*lifetime peak ([\d.]+) GB", ln)
    if m:
        peak[m.group(1)] = float(m.group(2))
AR = json.load(open("data/n45/ar_ref_tokens.json"))
def load(t, c):
    try:
        return ({r["key"]: r for r in json.load(open(f"data/r57/{t}_{c}_runs.json"))},
                {k: list(v.values())[0][0] for k, v in json.load(open(f"data/r57/{t}_{c}_tokens.json")).items()})
    except (OSError, ValueError, KeyError):
        return {}, {}
names = {"fa0": "L (baseline)", "fa1": "L - FASTTAX", "fa2": "L - HC8_FAST", "fa3": "L - SVH - DSUB", "fa4": "L - DN_KS",
         "fa5": "L - RELAX_SPAN", "fa6": "L - all six"}
import os
L = [f"FU (data/fu; snapshot {snap}; C 256 unless noted; quiet)", "A. FLAG 1 bisect, 12 IFEval, runs == NAX6-N's AR (one pass each):"]
if not os.environ.get("FU_DO_A"):
    L.append("  skipped: superseded by CHS1 (data/chs/chs1_report.txt) + CHS3 (data/chs/chs3_report.txt)")
    names = {}
for t, nm in names.items():
    R, tk = load(t, 256)
    eq, dif = 0, []
    for k, o in tk.items():
        r = AR[k]["gate+t"][0]; m = min(len(o), len(r))
        if m > 0 and o[:m] == r[:m]:
            eq += 1
        else:
            dif.append(f"{k.split('/')[-1]}@{next((i for i in range(m) if o[i] != r[i]), m)}")
    L.append(f"  {t} {nm:16s}: == AR {eq} / {len(tk)}; whole decode {g([r['tps'] for r in R.values()]):.1f} tok/s; first diffs {dif[:6]}")
L.append("B. FLAG 2: agent prompts (CAP 16): lifetime peak, TTFT per prompt (s); then 12 HumanEval whole decode for the fixes")
for t, c, nm in (("fb0", 256, "L at C 256"), ("fb1", 248, "L at C 248"), ("fb2", 256, "L + PFCHUNK 512"), ("fb3", 256, "L + CACHE_GB 2"), ("fb4", 256, "L + PF-A (R57_PFLM=1)"), ("fb5", 256, "L + PF-A + CACHE_GB 2")):
    R, _ = load(t, c)
    tt = {k.split("/")[-1]: round(r.get("pfs") or float("nan"), 1) for k, r in R.items()}
    L.append(f"  {t} {nm:16s}: lifetime peak {peak.get(f'{t}_{c}', float('nan')):.2f} GB (line 45.0); TTFT {tt}")
for t, c, nm in (("fh0", 256, "L at C 256"), ("fh1", 248, "L at C 248"), ("fh3", 256, "L + CACHE_GB 2")):
    R, _ = load(t, c)
    L.append(f"  {t} {nm:16s}: 12 HumanEval WHOLE DECODE {g([r['tps'] for r in R.values()]):.1f} tok/s; lifetime peak {peak.get(f'{t}_{c}', float('nan')):.2f} GB")
open("data/fu/fu_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
EOF
say "report rc=$? ($D/fu_report.txt)"
git add $D $(for t in fa0 fa1 fa2 fa3 fa4 fa5 fa6; do echo data/r57/${t}_256_*; done) data/r57/fb0_256_* data/r57/fb1_248_* data/r57/fb2_256_* \
  data/r57/fb3_256_* data/r57/fb4_256_* data/r57/fb5_256_* data/r57/fh0_256_* data/r57/fh1_248_* data/r57/fh3_256_* 2>/dev/null
> (a quote from the project owner, kept private)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011DvdV6eUB4NgBbVkyVFPrT"
touch $D/DONE
say "FU end"
