#!/bin/bash
> (a quote from the project owner, kept private)
# NewIdeas' board, QUEUE 7788571 / a9d645a). BigMoE is the strike's A/B runner: every chat candidate runs here on N45's 12 IFEval
# prompts at C 256, one process per arm, on a new deep-frozen snapshot, then EvalSuite's IFEval gate.
# Per arm (data/chs/<batch>_report.txt): WHOLE DECODE (pooled n / decode wall, and the geo-mean of the runs), kept per check, rows
# per check, ms per check (ck = the check wall; drf = the draft, outside ck; cycle = ck + drf), misses per check, svc (ms of gate
# service on calls with misses = the miss-stall bound), io (SSD busy ms), stl (layer calls that waited, a COUNT), bld / wt (build
# incl. the per-layer router waits / the GPU tail), MB SSD per kept token, lifetime peak (line 45.0), runs == NAX6-N's AR (prefix).
# Arms: CHS_ARMS="tag|qom_sets|ENV=V,ENV=V;..." (qom_sets = R57_QOM_SETS, commas allowed there). Default = batch chs1:
#   c0 = L (S5''s lossy, data/stack/s5p_env.txt); q1 = L + QOM on chat (M1); r1 = L + R57_CHAT_RULE=R1 (K1, arm E's rule; HC8 stays
#   on: E + HC8 failed IFEval 84 vs 90 on 11-05, so r1 is a speed reading); fa1-fa6 = C4's bisect (L minus FASTTAX / HC8 (R57_HC8
#   AND REF2_HC8_FAST: R57_HC8 alone sets ref2's hc8_fast) / SVH + DSUB / DN_KS / RELAX_SPAN / all); e0 = E (exact + bundle);
#   c0b = L again (drift); qr1 = QOM on chat + R1.
# CHS_END=HH:MM (default 01:18; epoch from today's date): no arm starts unless it can end before it (CERT 01:30 never moves).
# usage: NOTAFTER=hh:mm NOWAIT45=1 GAP0=0 GAP=0 MARGIN=4 bash scripts/turn45.sh 33 40 "..." /bin/bash scripts/chs.sh
# chsb.sh (Sun 00:2x) = chs.sh (left unedited while chs1 ran it) + CHS_ARMS placeholders @L / @E + engagement counters per arm
# (FAST gated rows, QOM skip / opt, lever 2's forecast reads issued / hits).
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "${CHS_ROOT:-$HOME/BigMoE}" || exit 1
B=${CHS_BATCH:-chs1}
D=data/chs; mkdir -p $D
LOG="$PWD/$D/chs.log"
say() { echo "$(date '+%m-%d %H:%M:%S') [$B] $*" >> "$LOG"; }
PY=${CHS_PY:-$HOME/BigMoE/.venv/bin/python}
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
END_E=$(date -j -f '%Y-%m-%d %H:%M' "$(date +%Y-%m-%d) ${CHS_END:-01:18}" +%s)
[ $END_E -lt $(date +%s) ] && END_E=$((END_E + 86400))
[ $((END_E - $(date +%s))) -gt 43200 ] && { say "CHS_END ${CHS_END:-01:18} is > 12 h away (passed?): not run"; exit 0; }
A=${CHS_AVAIL:-$(avail_gb)}
awk -v a="$A" 'BEGIN {exit !(a >= 47.2)}' || { say "NO-SWAP $A GB < 47.2: not run"; exit 0; }
[ -n "$CHS_DRYRUN" ] || sleep 20
A=${CHS_AVAIL:-$(avail_gb)}
awk -v a="$A" 'BEGIN {exit !(a >= 47.2)}' || { say "NO-SWAP $A GB < 47.2 at the 20 s look-again: not run"; exit 0; }
L=$(grep -v '^#' data/stack/s5p_env.txt | grep . | tail -1 | sed 's/,R57_QOM_SETS=humaneval,mbpp$//')
drop() { echo "$L" | tr ',' '\n' | grep -vxE "$1" | paste -sd, -; }
BUN="BIGMOE_PAUSE_EVERY_MS=50,R57_S0CARRY=1,REF2_SB_FIRST=1,REF2_PLE_LATE=1,REF2_COMMIT_SUB=9,R57_DQ1=1"
E="R57_NAX=1,REF2_KEEP_CACHES=1,R57_WIRED=1,$BUN"
QC="humaneval,mbpp"; QI="humaneval,mbpp,ifeval"
ALL='R57_FASTTAX=1|R57_HC8=1|REF2_HC8_FAST=1|R57_SVH=1|R57_DSUB=1|REF2_DN_KS=1|R57_RELAX_SPAN=1|R57_RELAX_SPAN_RULE=R2:2:0.25'
ARMS=${CHS_ARMS:-"c0|$QC|$L;q1|$QI|$L;fa1|$QC|$(drop 'R57_FASTTAX=1');fa2|$QC|$(drop 'R57_HC8=1|REF2_HC8_FAST=1');fa3|$QC|$(drop 'R57_SVH=1|R57_DSUB=1');fa4|$QC|$(drop 'REF2_DN_KS=1');fa5|$QC|$(drop 'R57_RELAX_SPAN=1|R57_RELAX_SPAN_RULE=R2:2:0.25');fa6|$QC|$(drop "$ALL");e0|$QC|$E;c0b|$QC|$L;r1|$QC|$L,R57_CHAT_RULE=R1;qr1|$QI|$L,R57_CHAT_RULE=R1"}
[ -f "$D/${B}_extra.txt" ] && ARMS="$ARMS;$(grep -v '^#' "$D/${B}_extra.txt" | grep . | paste -sd';' -)"   # Sun 01:2x: arms added
ARMS=${ARMS//@L/$L}; ARMS=${ARMS//@E/$E}                # after queueing (data/chs/<batch>_extra.txt) ride along; @L / @E as in CHS_ARMS
nk() { echo "$1" | tr ',' '\n' | grep -c =; }
IFS=';' read -ra AL <<< "$ARMS"
for a in "${AL[@]}"; do                                  # every bisect arm must drop something (s5p_env changed?)
  t=${a%%|*}; e=${a##*|}
  case $t in fa*) [ "$(nk "$e")" -lt "$(nk "$L")" ] || { say "$t drops nothing: not run"; exit 2; };; esac
done
KI=$(sed -n 1p data/n45/keys.txt)
[ "$(echo "$KI" | tr ',' '\n' | grep -c '^ifeval/')" = 12 ] || { say "data/n45/keys.txt line 1 is not 12 IFEval keys: not run"; exit 2; }
FR=$(bash scripts/deep_freeze.sh $B) || { say "freeze failed: $FR"; exit 2; }
SD=data/r57/snap/$B
grep -q 'R57_QOM_MTP' "$SD/r57_screen_$B.py" || { say "the frozen screen has no R57_QOM_MTP: not run"; exit 2; }
if [ -n "$CHS_DRYRUN" ]; then
  echo "dry run: $FR; L $(nk "$L") K=V; END $(date -r $END_E '+%a %H:%M')"
  for a in "${AL[@]}"; do t=${a%%|*}; r=${a#*|}; q=${r%%|*}; e=${r#*|}; echo "  $t: QOM_SETS=$q, $(nk "$e") K=V ($(( $(nk "$e") - $(nk "$L") )) vs L)"; done
  exit 0
fi
git add data/r57/snap/frozen_$B ":!data/r57/snap/frozen_$B/**/__pycache__/**" "$SD" ":!$SD/__pycache__/**" data/r57/snap/c45_runner_$B.sh \
> (a quote from the project owner, kept private)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011DvdV6eUB4NgBbVkyVFPrT" || { say "snapshot commit failed: not run"; exit 3; }
SNAP=$(git rev-parse --short HEAD)
RUN="/bin/bash data/r57/snap/c45_runner_$B.sh"
export C45_OUT="$PWD/$D" C45_SWAP0=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}') C45_ALARM=600 \
  MG=4 C45_DEADLINE=$(date -r $END_E +%H%M) C45_SCREEN="$SD/r57_screen_$B.py" R57_MODMAN=1 \
  C45_REFTOK="$PWD/data/n45/ar_ref_tokens.json:gate+t"      # exact arms compare in-run vs NAX6-N's AR (L's arms carry R57_REFTOK=)
rm -f $D/${B}_DONE
say "start (pid $$, NO-SWAP $A GB, snapshot $SNAP, end $(date -r $END_E +%H:%M), arms: $(for a in "${AL[@]}"; do echo -n "${a%%|*} "; done))"
TAGS=""
for a in "${AL[@]}"; do
  t=${a%%|*}; r=${a#*|}; q=${r%%|*}; e=${r#*|}
  [ $(( $(date +%s) + ${CHS_ARMS_S:-150} )) -le $END_E ] || { say "$t skipped: it could not end before $(date -r $END_E +%H:%M)"; continue; }
  rm -f data/r57/${t}_256_* $D/${t}_256.txt
  R57_QOM_SETS=$q R57_KEYS=$KI C45_ARMS="${t}_256:256:$e" $RUN
  say "$t rc=$?"; TAGS="$TAGS $t"
done
"$PY" -I - "$D" "$SNAP" "$B" "$TAGS" >> "$LOG" 2>&1 <<'EOF'
import json, math, re, statistics as S, sys
D, snap, B, tags = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split()
g = lambda xs: (lambda ys: math.exp(sum(map(math.log, ys)) / len(ys)) if ys else float("nan"))([x for x in xs if x and x > 0])
peak = {}
for ln in open(f"{D}/c45.log"):
    m = re.search(r"done (\S+) rc=\d+:.*lifetime peak ([\d.]+) GB", ln)
    if m:
        peak[m.group(1)] = float(m.group(2))
AR = json.load(open("data/n45/ar_ref_tokens.json"))
L = [f"CHS {B} (data/chs; snapshot {snap}; 12 IFEval (N45's keys), C 256, one pass per arm, quiet): WHOLE DECODE pooled (geo of runs);"
     " per check means: kept, rows, ck / drf / cycle ms, misses, svc / io ms, stl (count), bld / wt ms; MB SSD per kept token;"
     " lifetime peak; runs == NAX6-N's AR; relaxed accepts (rule R1 / span)"]
base = None
for t in tags:
    try:
        R = [r for r in json.load(open(f"data/r57/{t}_256_runs.json")) if r["key"].startswith("ifeval")]
        C = [c for c in json.load(open(f"data/r57/{t}_256_checks.json")) if c["key"].startswith("ifeval")]
        T = {k: list(v.values())[0][0] for k, v in json.load(open(f"data/r57/{t}_256_tokens.json")).items()}
    except (OSError, ValueError, KeyError, IndexError) as e:
        L.append(f"  {t}: MISSING ({type(e).__name__})"); continue
    if not R:
        L.append(f"  {t}: no IFEval runs"); continue
    pooled = sum(r["n"] for r in R) / sum(r["n"] / r["tps"] for r in R)
    base = base or pooled
    agg = {k: [] for k in ("kept", "rows", "ck", "drf", "miss", "svc", "io", "stl", "bld", "wt", "mb")}
    for c in C:
        for k in agg:
            agg[k] += c.get(k) or []
    m = lambda k: S.mean(agg[k]) if agg[k] else float("nan")
    mbk = sum(agg["mb"]) / max(sum(agg["kept"]), 1e-9)
    eq = 0
    for k, o in T.items():
        r_ = AR[k]["gate+t"][0]; n_ = min(len(o), len(r_))
        eq += int(n_ > 0 and o[:n_] == r_[:n_])
    rl = sum(((r.get("fast") or {}).get("k", 0) - (r.get("fast") or {}).get("kx", 0)) for r in R)
    sp = sum((r.get("spn") or {}).get("relaxed", 0) for r in R)
    bad = sum(1 for r in R if r.get("bad"))
    pf_ = {k: sum((r.get("pf") or {}).get(k, 0) for r in R) for k in ("qom_skip", "qom_opt", "fc_issued", "fc_hits")}
    gt_ = sum((r.get("fast") or {}).get("gated", 0) for r in R)
    L.append(f"  {t:4s}: WHOLE DECODE {pooled:6.1f} tok/s ({100 * (pooled / base - 1):+5.1f} % vs first) geo {g([r['tps'] for r in R]):.1f}; "
             f"kept {m('kept'):.2f} rows {m('rows'):.2f}; ck {m('ck'):.1f} drf {m('drf'):.1f} cycle {m('ck') + m('drf'):.1f} ms; "
             f"miss {m('miss'):.1f} svc {m('svc'):.1f} io {m('io'):.1f} stl {m('stl'):.1f}; bld {m('bld'):.1f} wt {m('wt'):.1f}; "
             f"{mbk:.1f} MB/kept; peak {peak.get(f'{t}_256', float('nan')):.2f} GB; == AR {eq}/{len(T)}; relaxed R1 {rl} span {sp}; "
             f"gated rows {gt_}; qom skip/opt {pf_['qom_skip']}/{pf_['qom_opt']}; fc issued/hits {pf_['fc_issued']}/{pf_['fc_hits']}"
             + (f"; BAD {bad}" if bad else ""))
open(f"{D}/{B}_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
EOF
say "report rc=$? ($D/${B}_report.txt)"
git add $D $(for t in $TAGS; do echo data/r57/${t}_256_*; done) 2>/dev/null
> (a quote from the project owner, kept private)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011DvdV6eUB4NgBbVkyVFPrT"
touch $D/${B}_DONE
say "end"
