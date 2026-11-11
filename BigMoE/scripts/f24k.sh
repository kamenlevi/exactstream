#!/bin/bash
# F24k (Sat 19:xx; Leap c176a5ad: REF2_TIER_CHUNK=1 row-chunks a tier call whose expert union exceeds the pool; bitwise per Leap 43/43): F24 on
# r57 with the chunking, from f24.sh. New deep-frozen snapshot f24k (ref2 with REF2_TIER_CHUNK), probe C 100 / 96 (Leap: C 112 peaked
# 25.4 GB), the chunked path's exact check at the chosen C (3 HE == NAX6-N AR), then the FAST passes. Leap's caveat: every 16-row check
# takes the synchronous chunked path (no gate overlap): ~29 tok/s at CAP 64, so F24 may land below E24's exact 41.3.
# Sat 7 Nov 17:4x (NewIdeas TABLE TONIGHT slot 6, accepted 17:3x): F24, the 24 GB LOSSY number = r57 v1' + wired + the Sunday bundle
# (S5''s config, its committed snapshot data/r57/snap/s5p) at the largest C whose lifetime peak is <= 24.0 GB, R57_VTIER off (its
# 7.45 GB volatile tier would not exist on a 24 GB Mac). r57's exact path needs C >= ~160 (16 rows x top-10 per layer call, else the ext
# aborts "no free slot"); v1''s FAST QOM only demand-loads ranks 0-1, so the code prompts fit a smaller pool, but a large union can abort.
# 1. PROBE: 12 HumanEval, CAP 256, at C 112 then C 104: lifetime peak + abort ("no free slot" / rc != 0) -> C24 = the largest clean C
#    with peak <= 24.0 GB. None -> stop (F24 not measurable on r57 tonight).
# 2. PASSES at C24: 12 HumanEval x 2, MBPP12 x 2 (one process each).
# REPORT (data/f24k/f24_report.txt): per set WHOLE DECODE (headline), full period, end to end, check 0, cycle, kept, tokens per task,
#   SSD MB per kept token, lifetime peak (<= 24.0), aborts, live outside modules (0). Writes data/stack/f24_env.txt for EvalSuite's
#   CODE-ONLY gate (NewIdeas 17:3x: HumanEval+ both orders + long code vs NAX6-N's ref2tnax; math / IFEval out).
# usage: NOTAFTER=hh:mm NOWAIT45=1 GAP0=0 GAP=0 MARGIN=4 bash scripts/turn45.sh 28 30 "..." /bin/bash scripts/f24.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "${F24_ROOT:-$HOME/BigMoE}" || exit 1
D=data/f24k; mkdir -p $D
LOG="$PWD/$D/f24.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
PY=${F24_PY:-$HOME/BigMoE/.venv/bin/python}
SD=data/r57/snap/f24k
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
A=${F24_AVAIL:-$(avail_gb)}
awk -v a="$A" 'BEGIN {exit !(a >= 32)}' || { say "NO-SWAP $A GB < 32: not run"; exit 0; }
[ -n "$F24_DRYRUN" ] || sleep 20
A=${F24_AVAIL:-$(avail_gb)}
awk -v a="$A" 'BEGIN {exit !(a >= 32)}' || { say "NO-SWAP $A GB < 32 at the 20 s look-again: not run"; exit 0; }
B=$(grep -v '^#' data/stack/cert_env.txt | grep . | tail -1 | sed 's/,R57_QOM_SETS=humaneval,mbpp$//')
echo "$B" | tr ',' '\n' | grep -qx 'R57_WIRED=1' || { say "cert_env is not v1' + wired: not run"; exit 2; }
F="$B,BIGMOE_PAUSE_EVERY_MS=50,R57_S0CARRY=1,REF2_SB_FIRST=1,REF2_PLE_LATE=1,REF2_COMMIT_SUB=9,R57_DQ1=1,R57_VTIER=,REF2_TIER_CHUNK=1"
if [ -n "$F24K_C24" ]; then                              # Sat 21:1x: REUSE the committed f24k snapshot (the one the probes and the
  git diff --quiet HEAD -- "$SD" data/r57/snap/frozen_f24k data/r57/snap/c45_runner_f24k.sh 2>/dev/null || { say "f24k snapshot differs from its commit: not run"; exit 2; }
  SNAP=$(git log -1 --format=%h -- "$SD/r57_screen_f24k.py")   # exact check ran on), skip the probes and the exact check, run the passes
  [ -n "$F24_DRYRUN" ] && { echo "dry run (reuse): snapshot $SNAP, C24 $F24K_C24"; exit 0; }
else
FR=$(bash scripts/deep_freeze.sh f24k) || { say "freeze failed: $FR"; exit 2; }
grep -q 'REF2_TIER_CHUNK' data/r57/snap/frozen_f24k/Leap_scripts/ref2.py || { say "the frozen ref2 has no REF2_TIER_CHUNK: not run"; exit 2; }
[ -n "$F24_DRYRUN" ] && { echo "dry run: $FR; F $(echo "$F" | tr ',' '\n' | grep -c =) K=V"; exit 0; }
git add data/r57/snap/frozen_f24k ':!data/r57/snap/frozen_f24k/**/__pycache__/**' "$SD" ":!$SD/__pycache__/**" data/r57/snap/c45_runner_f24k.sh \
> (a quote from the project owner, kept private)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011DvdV6eUB4NgBbVkyVFPrT" || { say "snapshot commit failed: not run"; exit 3; }
SNAP=$(git rev-parse --short HEAD)
fi
RUN="/bin/bash data/r57/snap/c45_runner_f24k.sh"
export C45_OUT="$PWD/$D" C45_SWAP0=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}') C45_ALARM=2400 \
  MG=4 C45_DEADLINE=$(date -v+120M +%H%M) C45_SCREEN="$SD/r57_screen_f24k.py" C45_REFTOK= R57_QOM_SETS=humaneval,mbpp R57_MODMAN=1
K="humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91,humaneval/HumanEval/6,humaneval/HumanEval/79"
K="$K,humaneval/HumanEval/19,humaneval/HumanEval/161,humaneval/HumanEval/17,humaneval/HumanEval/56,humaneval/HumanEval/141"
K="$K,humaneval/HumanEval/101,humaneval/HumanEval/63"
KM="mbpp/16,mbpp/100,mbpp/146,mbpp/161,mbpp/183,mbpp/208,mbpp/224,mbpp/246,mbpp/268,mbpp/296,mbpp/305,mbpp/319"
rm -f $D/DONE $D/C24
say "F24 start (pid $$, NO-SWAP $A GB, snapshot $SNAP)"
C24="$F24K_C24"
for c in $([ -z "$F24K_C24" ] && echo 100 96); do
  t=f24kp$c; rm -f data/r57/${t}_${c}_* $D/${t}_${c}.txt
  R57_KEYS=$K C45_ARMS="${t}_$c:$c:$F" $RUN; rc=$?
  pk=$(grep -oE "done ${t}_$c rc=[0-9]+:.*lifetime peak [0-9.]+" $D/c45.log | tail -1 | grep -oE '[0-9.]+$')
  ab=$(grep -c 'no free slot' $D/${t}_$c.txt 2>/dev/null)
  n=$(grep -o '"key"' data/r57/${t}_${c}_runs.json 2>/dev/null | wc -l | tr -d ' ')
  say "probe C $c: rc $rc, runs ${n:-0} / 12, lifetime peak ${pk:-?} GB, aborts ${ab:-?}"
  if [ "$rc" = 0 ] && [ "${n:-0}" = 12 ] && [ "${ab:-1}" = 0 ] && awk -v p="${pk:-99}" 'BEGIN {exit !(p <= 24.0)}'; then
    [ -z "$C24" ] && C24=$c
  fi
done
[ -n "$C24" ] || { say "NO C passed the probe (peak <= 24.0, no abort, 12 / 12): F24 not measurable on r57 tonight"; touch $D/DONE; exit 0; }
echo "$C24" > $D/C24
# the chunked path's exactness at C24 (Leap: the official proof): the NAX6-N exact env + REF2_TIER_CHUNK=1, 3 HumanEval, CAP 64,
# tokens == NAX6-N's one-token AR (E5's file); every 16-row check there takes the chunked sync path
t=f24kx
if [ -z "$F24K_C24" ]; then
rm -f data/r57/${t}_${C24}_* $D/${t}_${C24}.txt
R57_KEYS="humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91" C45_CAP=64 R57_REFTOK="$PWD/data/e5/ar_ref_tokens.json:gate+t" \
  C45_ARMS="${t}_$C24:$C24:R57_NAX=1,REF2_KEEP_CACHES=1,R57_WIRED=1,R57_VTIER=,REF2_TIER_CHUNK=1" $RUN
fi
XP=$("$PY" -I -c 'import json, sys
ar = json.load(open(sys.argv[1])); tk = json.load(open(sys.argv[2])); cn = json.load(open(sys.argv[3]))
eq = sum(1 for k, v in tk.items() if (lambda o, r: len(o) > 0 and o[:min(len(o), len(r))] == r[:min(len(o), len(r))])(list(v.values())[0][0], ar[k]["gate+t"][0]))
print("%d / %d prompts == AR; tier_chunk calls %d, groups %d" % (eq, len(tk), cn.get("tier_chunk", 0), cn.get("tier_chunk_groups", 0)))' \
  data/e5/ar_ref_tokens.json data/r57/${t}_${C24}_tokens.json data/r57/${t}_${C24}_counts.json 2>&1)
say "EXACT CHECK at C $C24 (chunked path): $XP"
echo "$XP" | grep -q '^3 / 3 prompts == AR' || { say "the chunked path is NOT exact at C $C24: no F24 passes"; touch $D/DONE; exit 0; }
say "C24 = $C24: the passes run"
for a in "f24kh1:$K" "f24kh2:$K" "f24km1:$KM" "f24km2:$KM"; do
  t=${a%%:*}; k=${a#*:}; rm -f data/r57/${t}_${C24}_* $D/${t}_${C24}.txt
  R57_KEYS=$k C45_ARMS="${t}_$C24:$C24:$F" $RUN
  say "$t rc=$?"
done
"$PY" -I - "$C24" "$(date +%F)" "$SNAP" >> "$LOG" 2>&1 <<'EOF'
import json, math, re, statistics as S, sys
C, day, snap = sys.argv[1:4]
g = lambda xs: (lambda ys: math.exp(sum(map(math.log, ys)) / len(ys)) if ys else float("nan"))([x for x in xs if x and x > 0])
peak, ab = {}, {}
for ln in open("data/f24k/c45.log"):
    m = re.search(r"done (\S+) rc=\d+:.*lifetime peak ([\d.]+) GB", ln)
    if m:
        peak[m.group(1)] = float(m.group(2))
SETS = {"12 HumanEval": ("f24kh1", "f24kh2"), "MBPP12": ("f24km1", "f24km2")}
runs, live, miss = {}, 0, []
for t in sum(SETS.values(), ()):
    try:
        ab[t] = sum(1 for ln in open(f"data/f24k/{t}_{C}.txt", errors="ignore") if "no free slot" in ln)
        R = {r["key"]: r for r in json.load(open(f"data/r57/{t}_{C}_runs.json"))}
        for c in json.load(open(f"data/r57/{t}_{C}_checks.json")):
            r = R.get(c["key"])
            if r is None:
                continue
            dec, c0 = r["n"] / r["tps"], c.get("c0") or []
            r["fp"] = sum(c["kept"][2:]) / (dec - c0[2] / 1e3) if len(c0) > 2 and dec > c0[2] / 1e3 else None
            r["e2e"] = r["n"] / (dec + r["pfs"]) if r.get("pfs") is not None else None
            r["ck0"] = c["bld"][0] + c["wt"][0] if c["bld"] else None
            r["kpc"] = S.mean(c["kept"][2:]) if len(c["kept"]) > 2 else None
        runs[t] = R
        live += sum(1 for _, f in json.load(open(f"data/r57/{t}_{C}_modules.json")) if "/BigMoE/data/r57/snap/" not in f)
    except (OSError, ValueError, KeyError):
        miss.append(t); runs[t] = {}
probes = [ln.strip() for ln in open("data/f24k/f24.log") if "probe C" in ln]
L = [f"F24k, the 24 GB LOSSY number (row-chunked): r57 v1' + wired + the Sunday bundle, R57_VTIER off, C {C} (data/f24k; F24k snapshot {snap}; quiet); "
     f"arms missing: {miss or 'none'}; live outside modules: {live}"] + [f"  {p.split(' ', 2)[2]}" for p in probes]
pk_all = 0.0
for sname, ps in SETS.items():
    pm = lambda f: [sum(f(runs[t][k]) for t in ps if k in runs[t] and f(runs[t][k]) is not None) /
                    max(1, sum(1 for t in ps if k in runs[t] and f(runs[t][k]) is not None)) for k in set().union(*[runs[t] for t in ps])]
    pk = max([peak.get(f"{t}_{C}", 0.0) for t in ps]) or float("nan"); pk_all = max(pk_all, pk)
    pas = ", ".join(f"{g([r['tps'] for r in runs[t].values()]):.1f}" for t in ps)
    L.append(f"  {sname}: WHOLE DECODE {g(pm(lambda r: r['tps'])):.1f} tok/s ({pas}); FULL PERIOD {g(pm(lambda r: r.get('fp'))):.1f}; END TO END "
             f"{g(pm(lambda r: r.get('e2e'))):.1f}; check 0 {S.median([x for x in pm(lambda r: r.get('ck0'))] or [float('nan')]):.0f} ms; cycle "
             f"{S.median([r['cyc'] for t in ps for r in runs[t].values() if isinstance(r.get('cyc'), (int, float))] or [float('nan')]):.1f} ms; "
             f"kept {g(pm(lambda r: r.get('kpc'))):.2f}; tokens/task {S.median(pm(lambda r: r['n']) or [float('nan')]):.0f}; SSD MB per kept token "
             f"{S.mean([r['mb'] for t in ps for r in runs[t].values() if r.get('mb') is not None] or [float('nan')]):.1f}; lifetime peak {pk:.2f} GB; "
             f"aborts {sum(ab.get(t, 0) for t in ps)}")
clean = not miss and not live and pk_all <= 24.0 and not any(ab.values())
L.append(f"CLEAN: {clean} (all 4 passes, 0 live modules, lifetime peak {pk_all:.2f} GB <= 24.0, no abort)")
open("data/f24k/f24_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
if clean:
    open("data/f24k/CLEAN", "w").write("ok\n")
EOF
say "report rc=$? ($D/f24_report.txt)"
if [ -f $D/CLEAN ]; then
  MEM=$(awk -v c="$C24" 'BEGIN {printf "%d", 38.0 + (c - 224) * 0.1325 + 4.5 + 0.99}')
  printf '%s\n' "# F24 config for EvalSuite's CODE-ONLY gate (BigMoE f24k.sh, snapshot $SNAP): Sat $C24, MEM $MEM GB, screen data/r57/snap/f24k/r57_screen_f24k.py; v1' + wired + the Sunday bundle at the 24 GB budget with Leap's row-chunked tier path (REF2_TIER_CHUNK=1; R57_VTIER off). C $C24 (EvalSuite: your 'Sat <C>' parser reads 3 digits). One comma-separated env line:" \
    "$F,R57_QOM_SETS=humaneval,mbpp" > data/stack/f24_env.txt
  git add data/stack/f24_env.txt
fi
git add $D data/r57/f24kp100_100_* data/r57/f24kp96_96_* data/r57/f24kh1_* data/r57/f24kh2_* data/r57/f24km1_* data/r57/f24km2_* 2>/dev/null
> (a quote from the project owner, kept private)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011DvdV6eUB4NgBbVkyVFPrT"
touch $D/DONE
say "F24 end"
