#!/bin/bash
# Mon 9 Nov 00:4x. REL ROWS: the shipped lossy config L + COH (data/stack/lcoh_env.txt == the package's configs/lossy-45.env and
# lossy-32.env, Leap 855cfa2) had no speed rows and no reference tokens: L's table rows and its certificate are L before COH, and
# L alone is run-to-run nondeterministic on chat. This measures it from a frozen release snapshot (REL=rel0 | rel1, made by
# scripts/deep_freeze.sh and committed): C 256 on the table's four sets (12 HumanEval and 12 MBPP = E5 / S5' B's keys in their run
# order, 12 IFEval and 12 math = data/n45/keys.txt lines 1 / 2), CAP 256, one pass, then C 160 on the 12 IFEval. Tokens = Leap's
# references for lossy-45 / lossy-32 (data/r57/rl*_tokens.json); the report pools rows as the number check does.
# usage: REL=rel1 NOWAIT45=1 GAP0=0 GAP=0 MARGIN=4 bash scripts/turn45.sh 43 15 "..." /bin/bash scripts/rel_rows.sh
#        REL=rel0 REL_DRY=1 bash scripts/rel_rows.sh   (no GPU: checks the snapshot, env and keys, prints the arms)
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
REL=${REL:-}
[ -n "$REL" ] || { echo "REL=rel0 or rel1 needed"; exit 2; }
D=data/rel; mkdir -p $D
LOG="$PWD/$D/rel_rows.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; [ -z "$REL_DRY" ] || echo "$*"; }   # returns 0 (Mon 09:5x: was 1 outside dry runs, so the script ended rc=1)
PY=$HOME/BigMoE/.venv/bin/python
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
SD=data/r57/snap/$REL; SCR=$SD/r57_screen_$REL.py; RS=data/r57/snap/c45_runner_$REL.sh
[ -f "$SCR" ] && [ -f "$RS" ] && [ -d data/r57/snap/frozen_$REL ] || { say "no frozen snapshot $REL (screen, runner, frozen tree): not run"; exit 2; }
git diff --quiet HEAD -- "$SD" "data/r57/snap/frozen_$REL" "$RS" \
  && [ -z "$(git ls-files --others --exclude-standard -- "$SD" "data/r57/snap/frozen_$REL" "$RS")" ] \
  || { say "snapshot $REL differs from its commit: not run"; exit 2; }
SNAPC=$(git log -1 --format=%h -- "$SD")
# Mon 07:2x (the package smoke): lossy / chat runs import tierglu.py, gdn_exact.py and gdn_exact_patch.py LIVE from
# ~/MoonShots/patches/planes_engine (bigmoe/tier.py _tier_glu puts it first on sys.path) and R57_MODMAN's manifest misses them.
# The freeze holds byte-identical copies (Leap_third_party/ms_gdn): run only while the live files equal them, before and after.
MSD=$HOME/MoonShots/patches/planes_engine; FMS=data/r57/snap/frozen_$REL/Leap_third_party/ms_gdn
ms_same() { local f; for f in tierglu.py gdn_exact.py gdn_exact_patch.py; do
  [ -s "$FMS/$f" ] && [ "$(md5 -q "$MSD/$f" 2>/dev/null)" = "$(md5 -q "$FMS/$f")" ] || return 1; done; }
ms_same || { say "the live planes_engine kernel files differ from the frozen copies in $FMS: not run"; exit 2; }
MSMD5=$(cd "$MSD" && for f in tierglu.py gdn_exact.py gdn_exact_patch.py; do echo -n "$f $(md5 -q $f | cut -c1-8), "; done)
LC=$(grep -v '^#' data/stack/lcoh_env.txt | grep . | tail -1 | sed 's/,R57_QOM_SETS=humaneval,mbpp$//')
echo "$LC" | grep -q 'REF2_NAX_COH=1' || { say "lcoh_env.txt's line has no REF2_NAX_COH=1: not run"; exit 2; }
PKG=$HOME/Leap/release/src/configs/lossy-45.env
if [ -f "$PKG" ]; then                                   # the package's line must be this one (+ the QOM sets it keeps inline)
  P=$(grep -v '^#' "$PKG" | grep -v '^C=' | grep . | tail -1 | sed 's/,R57_QOM_SETS=humaneval,mbpp$//')
  [ "$P" = "$LC" ] || { say "the package's lossy-45.env line differs from lcoh_env.txt: not run"; exit 2; }
fi
keys_of() { "$PY" -I -c 'import json, sys; print(",".join(r["key"] for r in json.load(open(sys.argv[1]))))' "$1"; }
KH=$(keys_of data/r57/e5h1_256_runs.json); KM=$(keys_of data/r57/e5m1_256_runs.json)
KI=$(sed -n 1p data/n45/keys.txt); KT=$(sed -n 2p data/n45/keys.txt)
for k in "$KH:humaneval" "$KM:mbpp" "$KI:ifeval" "$KT:math500"; do
  s=${k##*:}; l=${k%:*}
  [ "$(echo "$l" | tr ',' '\n' | grep -c "^$s/")" = 12 ] || { say "the $s keys are not 12 $s keys: not run"; exit 2; }
done
ARMS="rlh_256:256:$KH rlm_256:256:$KM rli_256:256:$KI rlt_256:256:$KT rli_160:160:$KI rlr_256:256:$KI"
# rlr_256 (Mon 07:3x, Leap's question): rli_256 again in its own process, same keys and order = is L + COH run-to-run identical
# for the same history? (COH made E's one-call path and M1-B repeat token for token: CHS4 ycoh / ycoh2, CHS5 z0 / z0b; L + COH untested)
if [ -n "$REL_DRY" ]; then
  say "dry run: snapshot $REL ($SNAPC), L + COH line $(echo "$LC" | tr ',' '\n' | grep -c =) K=V, live kernel files == frozen ($MSMD5), arms: $(for a in $ARMS; do echo -n "${a%%:*} "; done)"
  exit 0
fi
A=$(avail_gb)
awk -v a="$A" 'BEGIN {exit !(a >= 47.2)}' || { say "NO-SWAP $A GB < 47.2: not run"; exit 0; }
sleep 20
A=$(avail_gb)
awk -v a="$A" 'BEGIN {exit !(a >= 47.2)}' || { say "NO-SWAP $A GB < 47.2 at the 20 s look-again: not run"; exit 0; }
export C45_OUT="$PWD/$D" C45_SWAP0=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}') C45_ALARM=2400 \
  MG=4 C45_DEADLINE=$(date -v+30M +%H%M) C45_SCREEN="$SCR" C45_REFTOK= R57_MODMAN=1
rm -f $D/DONE
say "REL ROWS start (pid $$, NO-SWAP $A GB, snapshot $REL = $SNAPC)"
TAGS=""
for a in $ARMS; do
  t=${a%%:*}; r=${a#*:}; c=${r%%:*}; k=${r#*:}
  rm -f data/r57/${t}_* $D/${t}.txt
  R57_KEYS=$k C45_CAP=256 R57_LOGHASH=1 R57_QOM_SETS=humaneval,mbpp C45_ARMS="$t:$c:$LC" /bin/bash "$RS"
  say "$t rc=$?"; TAGS="$TAGS $t"
done
"$PY" -I - "$REL" "$SNAPC" $TAGS > $D/rel_rows_report.txt <<'EOF'
import json, os, statistics as S, sys
rel, snap, tags = sys.argv[1], sys.argv[2], sys.argv[3:]
AR = {}
for f in ("data/e5/ar_ref_tokens.json", "data/n45/ar_ref_tokens.json"):
    AR.update(json.load(open(f)))
def first(v):
    while isinstance(v, dict):
        v = next(iter(v.values()))
    while isinstance(v, list) and v and isinstance(v[0], list):
        v = v[0]
    return v
L = [f"REL ROWS (data/rel; snapshot {rel} = {snap}; quiet): L + COH (data/stack/lcoh_env.txt, the package's lossy-45 / lossy-32), "
     f"CAP 256, one pass. Pooled as the number check does: whole decode (sum n / sum n / tps), whole request (sum n / (decode + "
     f"first token time)), first token median (pfs), lifetime peak (line 45.0 at C 256, 32.0 at C 160)."]
TOK = {}
for t in tags:
    try:
        R = json.load(open(f"data/r57/{t}_runs.json"))
        T = {k: first(v) for k, v in json.load(open(f"data/r57/{t}_tokens.json")).items()}
    except (OSError, ValueError):
        L.append(f"  {t}: MISSING"); continue
    TOK[t] = T
    n = sum(r["n"] for r in R); dec = sum(r["n"] / r["tps"] for r in R); pf = [r["pfs"] for r in R]
    peak = max((r.get("mem") or [0])[-1] for r in R); cuts = sum(1 for r in R if r.get("cut"))
    eq = sum(1 for k, o in T.items() if k in AR and o[:min(len(o), len(first(AR[k])))] == first(AR[k])[:min(len(o), len(first(AR[k])))])
    L.append(f"  {t}: prompts {len(R)}, tokens {n}, whole decode {n / dec:.1f}, whole request {n / (dec + sum(pf)):.1f} tok/s, "
             f"first token median {S.median(pf):.2f} s (mean {S.mean(pf):.2f}), lifetime peak {peak:.2f} GB, == AR {eq}/{len(T)}; "
             f"cut rows {cuts} (the screen's 600 s budget{': DO NOT QUOTE this arm' if cuts else ''}), rows ended before the "
             f"cap {sum(1 for r in R if r['n'] < 256 and not r.get('cut'))}")
if "rli_256" in TOK and "rli_160" in TOK:
    a, b = TOK["rli_256"], TOK["rli_160"]
    L.append(f"  IFEval tokens C 256 == C 160: {sum(1 for k in a if a[k] == b.get(k))}/{len(a)}")
if "rli_256" in TOK and "rlr_256" in TOK:                # the repeat: the same keys in the same order, its own process
    a, b = TOK["rli_256"], TOK["rlr_256"]
    d = [next((i for i, (x, y) in enumerate(zip(a[k], b.get(k) or [])) if x != y), None) for k in a]
    L.append(f"  REPEAT rli_256 == rlr_256 (own processes, same keys and order): {sum(1 for k in a if a[k] == b.get(k))}/{len(a)} "
             f"prompts token-identical; first differing token per differing prompt: {[x for x in d if x is not None] or 'none'}")
open("data/rel/rel_rows_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
EOF
say "report rc=$? ($D/rel_rows_report.txt)"
if ms_same; then echo "  live kernel files read from the planes_engine tree == the frozen copies before and after: $MSMD5" >> $D/rel_rows_report.txt
else echo "  WARNING: the live planes_engine kernel files changed during the run (no longer == $FMS)" >> $D/rel_rows_report.txt; fi
touch $D/DONE
git add $D $(for t in $TAGS; do ls data/r57/${t}_* 2>/dev/null; done) 2>/dev/null
> (a quote from the project owner, kept private)

$(cat $D/rel_rows_report.txt)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011DvdV6eUB4NgBbVkyVFPrT" || say "data commit failed"
say "REL ROWS end"
