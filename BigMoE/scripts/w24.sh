#!/bin/bash
> (a quote from the project owner, kept private)
# config in the package by Wed, installable on a 36 GB Mac). The shipped lossy config L + COH (data/stack/lcoh_env.txt) below a
# 24.0 GB lifetime peak ON THE FAST PATH.
# Measured (records3 index, 48 layers): a base slot costs 0.0991 GB, a sidecar slot 0.0346 GB; the rest of the lifetime peak is
# ~10.1-10.6 GB (F24k: C 100 = 94 base + 112 sidecar slots, 13.19 GB pools, peak 23.7 GB). A call takes the sync-free gate only
# if its ids (rows x top-10) <= the base slots, or <= 92 (bigmoe/stream.py:76), so F24k's 16-row checks ran on the slow sync
# path: 231 MB SSD per kept token on HumanEval vs 18.1 at C 160 (C30), 484 ms cycles. W24 moves F24k's bytes into the base pool
# (BIGMOE_TIER_CM / BIGMOE_TIER_CL) and caps every check at R rows (R57_ROWCAP, data/r57/snap/w24's screen = rel1's + that cap)
# with R x 10 <= base slots and R x 2 (the 4-bit ranks) <= sidecar slots. R57_VTIER off: volatile memory beside the footprint
# would not exist on a 24 GB machine. REF2_TIER_CHUNK=1 only as a net (a wider call row-chunks instead of aborting); the
# report shows the path counters (tier_calls / tier_chunk must stay 0 for a fast-path number).
# Arms, in order (a cut night keeps the first): w12 = 120 base / 34 sidecar, rows 12 (13.07 GB pools); w10 = 104 / 76, rows 10
# (12.94 GB); w8 = 100 / 90, rows 8 (13.02 GB). Sets: 12 HumanEval + 12 MBPP (E5 / S5' B's keys in run order) + 12 IFEval
# (data/n45/keys.txt line 1), CAP 256, one pass each.
# usage: W24_ARMS="w12 w10 w8" NOWAIT45=1 GAP0=0 GAP=0 MARGIN=4 bash scripts/turn45.sh 24 60 "..." /bin/bash scripts/w24.sh
#        W24_DRY=1 bash scripts/w24.sh   (no GPU: checks the snapshot, env and keys, prints the arms)
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
D=data/w24; mkdir -p $D
RUN=${W24_RUN:-w24}                                    # W24b: W24_RUN=w24b -> data/w24/w24b_report.txt
LOG="$PWD/$D/w24.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; [ -z "$W24_DRY" ] || echo "$*"; }
PY=$HOME/BigMoE/.venv/bin/python
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
SD=data/r57/snap/w24; SCR=$SD/r57_screen_w24.py; RS=data/r57/snap/c45_runner_rel1.sh; FR=data/r57/snap/frozen_rel1
[ -f "$SCR" ] && [ -f "$RS" ] && [ -d "$FR" ] || { say "no w24 snapshot (screen, runner, frozen tree): not run"; exit 2; }
git diff --quiet HEAD -- "$SD" "$FR" "$RS" && [ -z "$(git ls-files --others --exclude-standard -- "$SD" "$FR" "$RS")" ] \
  || { say "snapshot w24 differs from its commit: not run"; exit 2; }
SNAPC=$(git log -1 --format=%h -- "$SD")
# the live planes_engine kernel files the screen imports must equal frozen_rel1's copies, before and after (as REL ROWS)
MSD=$HOME/MoonShots/patches/planes_engine; FMS=$FR/Leap_third_party/ms_gdn
ms_same() { local f; for f in tierglu.py gdn_exact.py gdn_exact_patch.py; do
  [ -s "$FMS/$f" ] && [ "$(md5 -q "$MSD/$f" 2>/dev/null)" = "$(md5 -q "$FMS/$f")" ] || return 1; done; }
ms_same || { say "the live planes_engine kernel files differ from the frozen copies in $FMS: not run"; exit 2; }
LC=$(grep -v '^#' data/stack/lcoh_env.txt | grep . | tail -1 | sed 's/,R57_QOM_SETS=humaneval,mbpp$//')
echo "$LC" | grep -q 'REF2_NAX_COH=1' || { say "lcoh_env.txt's line has no REF2_NAX_COH=1: not run"; exit 2; }
keys_of() { "$PY" -I -c 'import json, sys; print(",".join(r["key"] for r in json.load(open(sys.argv[1]))))' "$1"; }
KH=$(keys_of data/r57/e5h1_256_runs.json); KM=$(keys_of data/r57/e5m1_256_runs.json); KI=$(sed -n 1p data/n45/keys.txt); KA=$(sed -n 2p data/n45/keys.txt)   # W24g (Tue 13:3x): the 12 math keys (n45 line 2)
for k in "$KH:humaneval" "$KM:mbpp" "$KI:ifeval"; do
  s=${k##*:}; l=${k%:*}
  [ "$(echo "$l" | tr ',' '\n' | grep -c "^$s/")" = 12 ] || { say "the $s keys are not 12 $s keys: not run"; exit 2; }
done
armenv() { case $1 in   # base slots, sidecar slots, rows
  w12) echo "120 34 12";; w10) echo "104 76 10";; w8) echo "100 90 8";; w12s) echo "120 26 12";;   # w12s: -0.28 GB (opt-in)
  w12pf) echo "120 34 12 R57_ARMS=gate+t+pf,BIGMOE_PRED_LINES=4";;   # + the ranked L+1 look-ahead (PF_B 96 needs 4 prediction lines)
  w12r) echo "120 34 12";;   # W24b: w12 again (noise)
  w12t) echo "120 34 12 BIGMOE_TIER_PF_GROUP=64,R57_CACHE_GB=0.25";;   # W24b: w12 + the prefill-peak trim
  w13t) echo "130 26 13 BIGMOE_TIER_PF_GROUP=64,R57_CACHE_GB=0.25";;   # W24b: +10 base slots and 13 rows, paid by the trim
  w16u) echo "120 34 12 R57_FAST=U3W,R57_GATEREQ=1";;   # U3 at thr 0 on the code sets, 16 rows on the gate (742758e); chat capped
  # W24c (Mon 9 Nov 22:5x): the drafter diet. R57_D7B=all loads ONLY the 4-bit DFlash2 drafter; with the gated =code the
  # bf16 one (l120_ft, 1.14 GB) stays resident for math / agent prompts only (code already drafts on the 4-bit one). Its
  # GB pays for a 14th row (+10 base, +2 sidecar slots = +1.06 GB) or 11 more base slots. Env only: snapshot w24 as gated.
  w13c|w13d) echo "130 26 13 BIGMOE_TIER_PF_GROUP=64,R57_CACHE_GB=0.25";;   # W24c: w13t again, as gated (control, first + last)
  w13a) echo "130 26 13 BIGMOE_TIER_PF_GROUP=64,R57_CACHE_GB=0.25,R57_D7B=all";;   # W24c: w13t + the diet alone (its GB)
  w14a|w14b) echo "140 28 14 BIGMOE_TIER_PF_GROUP=64,R57_CACHE_GB=0.25,R57_D7B=all";;   # W24c: 14 rows paid by the diet (x2)
  w13b) echo "141 26 13 BIGMOE_TIER_PF_GROUP=64,R57_CACHE_GB=0.25,R57_D7B=all";;   # W24c: the diet's GB as 11 base slots
  # W24d (Mon 23:5x, the orchestrator: option B): w13b vs the 14-row config, same-run ABBA on HumanEval (w14c w13e w13f w14d)
  w14c|w14d) echo "140 28 14 BIGMOE_TIER_PF_GROUP=64,R57_CACHE_GB=0.25,R57_D7B=all";;   # W24d: the 14-row config (= card_w24c_env.txt)
  w13e|w13f) echo "141 26 13 BIGMOE_TIER_PF_GROUP=64,R57_CACHE_GB=0.25,R57_D7B=all";;   # W24d: w13b again, twice
  # W24f (Tue 09:0x, the orchestrator: the slot after Leap's package check): MBPP + IFEval speed, w13b vs w13t, same-run ABBA
  tf1|tf2) echo "130 26 13 BIGMOE_TIER_PF_GROUP=64,R57_CACHE_GB=0.25";;   # W24f: w13t as gated in W24b (control, first + last)
  bf1|bf2) echo "141 26 13 BIGMOE_TIER_PF_GROUP=64,R57_CACHE_GB=0.25,R57_D7B=all";;   # W24f: w13b, the card-gated 24 GB config
  bg1|bg2) echo "141 26 13 BIGMOE_TIER_PF_GROUP=64,R57_CACHE_GB=0.25,R57_D7B=all";;   # W24g (Tue 13:3x, the orchestrator): w13b on the 12 math prompts, twice
  *) return 1;; esac; }
ARMS=${W24_ARMS:-w12 w16u w12pf w10 w8}
for a in $ARMS; do
  read -r cm cl rw ex <<< "$(armenv $a)" || { say "unknown arm $a: not run"; exit 2; }
  [ $((rw * 10)) -le $cm ] && [ $((rw * 2)) -le $cl ] || { say "$a: rows $rw do not fit $cm / $cl: not run"; exit 2; }
done
if [ -n "$W24_DRY" ]; then
  for a in $ARMS; do read -r cm cl rw ex <<< "$(armenv $a)"
    say "dry: $a = base $cm, sidecar $cl, rows $rw${ex:+, $ex}, pools $(awk -v m=$cm -v l=$cl 'BEGIN {printf "%.2f", m * 0.0991 + l * 0.0346}') GB"; done
  say "dry run: snapshot w24 ($SNAPC), L + COH line $(echo "$LC" | tr ',' '\n' | grep -c =) K=V, live kernel files == frozen_rel1's"
  exit 0
fi
A=$(avail_gb)
awk -v a="$A" 'BEGIN {exit !(a >= 28.0)}' || { say "NO-SWAP $A GB < 28.0: not run"; exit 0; }
export C45_OUT="$PWD/$D" C45_SWAP0=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}') C45_ALARM=2400 \
  MG=4 C45_DEADLINE=${W24_DEADLINE:-$(date -v+75M +%H%M)} C45_SCREEN="$SCR" C45_REFTOK= R57_MODMAN=1
rm -f $D/DONE
say "W24 start (pid $$, NO-SWAP $A GB, snapshot w24 = $SNAPC, arms $ARMS)"
TAGS=""
for a in $ARMS; do
  read -r cm cl rw ex <<< "$(armenv $a)"
  X="R57_VTIER=,BIGMOE_TIER_CM=$cm,BIGMOE_TIER_CL=$cl,R57_ROWCAP=$rw,REF2_TIER_CHUNK=1${ex:+,$ex}"
  for s in "h:$KH" "m:$KM" "i:$KI" "a:$KA"; do
    case " ${W24_SETS:-h m i} " in *" ${s%%:*} "*) ;; *) continue;; esac   # W24_SETS: a subset of h m i (W24b: h m)
    t=${a}${s%%:*}_100; k=${s#*:}
    rm -f data/r57/${t}_* $D/${t}.txt
    R57_KEYS=$k C45_CAP=256 R57_LOGHASH=1 R57_QOM_SETS=humaneval,mbpp C45_ARMS="$t:100:$LC,$X" /bin/bash "$RS"
    say "$t rc=$?"; TAGS="$TAGS $t"
  done
done
"$PY" -I - "$SNAPC" "$RUN" $TAGS > $D/${RUN}_report.txt <<'EOF'
import json, re, statistics as S, sys
snap, run, tags = sys.argv[1], sys.argv[2], sys.argv[3:]
L = [f"{run.upper()} (data/w24; snapshot w24 = {snap}; quiet): L + COH (data/stack/lcoh_env.txt) + R57_VTIER off + base / sidecar slots + "
     f"R57_ROWCAP rows, CAP 256, one pass. Pooled as the number check does: whole decode (sum n / sum n / tps), whole request "
     f"(sum n / (decode + first token time)), first token median (pfs), lifetime peak (line 24.0). FAST PATH = tier_calls 0 and "
     f"tier_chunk 0 (every check on the sync-free gate)."]
for t in tags:
    try:
        R = json.load(open(f"data/r57/{t}_runs.json"))
        C = json.load(open(f"data/r57/{t}_counts.json"))
    except (OSError, ValueError):
        L.append(f"  {t}: MISSING"); continue
    log = open(f"data/w24/{t}.txt", errors="ignore").read()
    ab = len(re.findall(r"no free slot|experts > C=", log))
    n = sum(r["n"] for r in R); dec = sum(r["n"] / r["tps"] for r in R); pf = [r["pfs"] for r in R]
    peak = max((r.get("mem") or [0])[-1] for r in R); cuts = sum(1 for r in R if r.get("cut"))
    kept = [r["kept"] for r in R if isinstance(r.get("kept"), (int, float))]
    cyc = [r["cyc"] for r in R if isinstance(r.get("cyc"), (int, float))]
    mb = [r["mb"] for r in R if isinstance(r.get("mb"), (int, float))]
    fast = C.get("tier_calls", 0) == 0 and C.get("tier_chunk", 0) == 0
    qs = sum((r.get("pf") or {}).get("qom_skip", 0) for r in R); qo = sum((r.get("pf") or {}).get("qom_opt", 0) for r in R)
    qd = f"QOM drops {100 * qs / qo:.1f} % of optional uses (C 256 REL ROWS: HE 11.4, MBPP 12.2)" if qo else "no QOM uses"
    hz = sum(1 for r in R if "PAUSED" in str(r.get("bad") or ""))   # heat freezes (fanwatch SIGSTOP) inside a row
    pk_ = {r["key"] for r in R if "PAUSED" in str(r.get("bad") or "")}
    try:                                                 # W24b: the fast path judged on the MEASURED checks (the process
        CK = json.load(open(f"data/r57/{t}_checks.json"))   # counters include warm-ups): checks over 300 ms in unpaused runs
        slow = sum(1 for c in CK if c["key"] not in pk_ for b, w in zip(c["bld"], c["wt"]) if b + w > 300)
        nck = sum(len(c["bld"]) for c in CK if c["key"] not in pk_)
    except (OSError, ValueError, KeyError):
        slow, nck = -1, 0
    Rn = [r for r in R if r["key"] not in pk_]
    wn = f"{sum(r['n'] for r in Rn) / sum(r['n'] / r['tps'] for r in Rn):.1f}" if Rn and pk_ else "-"
    L.append(f"  {t}: prompts {len(R)}, tokens {n}, WHOLE DECODE {n / dec:.1f}, whole request {n / (dec + sum(pf)):.1f} tok/s, first "
             f"token median {S.median(pf):.2f} s, lifetime peak {peak:.2f} GB{' (OVER 24.0)' if peak > 24.0 else ''}; kept per check "
             f"{S.mean(kept) if kept else float('nan'):.2f}, cycle median {S.median(cyc) if cyc else float('nan'):.1f} ms, SSD MB per kept "
             f"token {S.mean(mb) if mb else float('nan'):.1f}; path: ooo_1c {C.get('ooo_1c', 0)}, tier_calls {C.get('tier_calls', 0)}, "
             f"tier_chunk {C.get('tier_chunk', 0)} (process, incl. warm-ups); measured checks over 300 ms {slow} of {nck}; {qd}; rows with a heat "
             f"freeze {hz} (whole decode without them {wn}); aborts {ab}; cut rows {cuts}"
             f"{' (DO NOT QUOTE)' if cuts or ab else ''}")
open(f"data/w24/{run}_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
EOF
say "report rc=$? ($D/${RUN}_report.txt)"
if ms_same; then echo "  live kernel files == frozen_rel1's copies before and after" >> $D/${RUN}_report.txt
else echo "  WARNING: the live planes_engine kernel files changed during the run" >> $D/${RUN}_report.txt; fi
touch $D/DONE
git add $D $(for t in $TAGS; do ls data/r57/${t}_* 2>/dev/null; done) 2>/dev/null
> (a quote from the project owner, kept private)

$(cat $D/${RUN}_report.txt)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011DvdV6eUB4NgBbVkyVFPrT" || say "data commit failed"
say "W24 end"
