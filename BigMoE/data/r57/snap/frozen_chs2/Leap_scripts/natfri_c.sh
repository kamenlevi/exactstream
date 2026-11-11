#!/bin/bash
# NATIVE-FRI phase C (Leap, Thu 2026-11-05 19:0x; TABLE_350 'NATIVE-FRI' AMENDMENT, written before arming): the 45 GB default
# on the converted records3 (Dc: 3c-N, OOO two phases) vs the same + R57_NAX=1 (Nc: NAX6-N, nax_native), Dc / Nc / Nc / Dc
# on the 14 ref14 prompts at the largest C of 224 / 208 / 192 / 176 / 160 / 144 / 128 that fits NO-SWAP (lower rungs Thu
# 23:4x, BigMoE's fri_night ladder: the orchestrator expects NO-SWAP ~33-38 GB tonight). Kamen 15:0x: R57_PRESET=45 moves to NAX6-N
# "if NATIVE-FRI passes".
# Runs BigMoE's scripts/c45_runner.sh and scripts/r57_screen.py as COPIES made at run time in ~/Leap/data/natfri/snap
# (read-only use of BigMoE's files). The screen copy's OUT line is patched to ~/Leap/data/natfri/r57, so the arms' files
# land here. Only turn45's shared log line is written in ~/BigMoE.
# Lines (pre-registered): Nc vs Dc = pair-mean geo steady tok/s (tpsw) over the prompts both passes of both arms kept:
#   PASS >= +6 %, KILL < +2 %, else BETWEEN. Dc tokens != REF2-T (ref14 gate+t, prefix) on any prompt: BUG.
#   Nc pass 1 != pass 2 on any prompt: VOID. Fewer than 12 prompts in both passes of an arm: VOID.
# usage (attached harness background task; turn45 takes the quiet ticket, AFTER waits for BigMoE's Friday ticket):
#   NOTBEFORE=00:32 NOTAFTER=12:00 WAITMEM=44 GAP0=0 GAP=0 MARGIN=4 AFTER="sh $HOME/Leap/scripts/natfri_after.sh" \
#     bash ~/BigMoE/scripts/turn45.sh 40 45 "Leap: NATIVE-FRI phase C ..." /bin/bash ~/Leap/scripts/natfri_c.sh
#   NATFRI_DRY=1 bash natfri_c.sh          copies + patches + prints the plan; runs nothing
#   NATFRI_REPORT_ONLY=1 bash natfri_c.sh C  the report from finished arms (NATFRI_DIR for another folder)
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
D=${NATFRI_DIR:-$HOME/Leap/data/natfri}
mkdir -p "$D/r57" "$D/snap"
LOG="$D/natfri.log"
PY=${NATFRI_PY:-$HOME/BigMoE/.venv/bin/python}
REF=$HOME/BigMoE/data/d45/ref14_tokens.json
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
swap_gb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}'; }
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
K="humaneval/HumanEval/68,humaneval/HumanEval/91,humaneval/HumanEval/16,humaneval/HumanEval/6,math500/test/prealgebra/1640.json"
K="$K,math500/test/algebra/694.json,math500/test/algebra/1282.json,gsm8k/708,ifeval/1220,ifeval/1561,multi/de3,multi/zh15"
K="$K,agent/bash18,agent/text23"                                  # dkv_thu.sh's ref14 keys (BigMoE 6adad93+)

report() {   # report C: tokens, speed, verdict -> $D/natfri_report.txt
  "$PY" -I - "$1" "$D" "$REF" "$K" <<'EOF'
import json, math, os, sys
C, D, REF, K = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
def load(t, kind):
    try:
        return json.load(open(os.path.join(D, "r57", f"{t}_{C}_{kind}.json")))
    except (OSError, ValueError):
        return None
tok = lambda v: (v.get("gate+t") or [[]])[0] if v else []
ref = json.load(open(REF))
T = {a: load(a, "tokens") or {} for a in ("nfdc", "nfnc", "nfncb", "nfdcb")}
R = {a: {r["key"]: r for r in (load(a, "runs") or [])} for a in T}
ok = lambda a, k: k in R[a] and "PAUSED" not in str(R[a][k].get("bad", ""))
L = [f"NATIVE-FRI phase C (Leap; TABLE_350 'NATIVE-FRI' AMENDMENT Thu 19:0x), C {C}, ref14 prompts, Dc / Nc / Nc / Dc",
     "Dc = the 45 GB default on the converted records3 (3c-N); Nc = + R57_NAX=1 (NAX6-N). Data: " + os.path.join(D, "r57")]
bug, void = [], []
for a in ("nfdc", "nfdcb"):                                      # Dc == REF2-T (prefix: a run may stop early)
    diff = [k for k in K if k in T[a] and (not tok(T[a][k]) or tok(T[a][k]) != ref[k]["gate+t"][0][:len(tok(T[a][k]))])]
    n = sum(1 for k in K if k in T[a])
    L.append(f"{a}: tokens == REF2-T on {n - len(diff)} / {n} prompts" + (f"; DIFFER {diff}" if diff else ""))
    bug += diff
same = [k for k in K if k in T["nfnc"] and k in T["nfncb"]]
nd = [k for k in same if tok(T["nfnc"][k]) != tok(T["nfncb"][k])]
L.append(f"Nc pass 1 == pass 2 on {len(same) - len(nd)} / {len(same)} prompts" + (f"; DIFFER {nd}" if nd else ""))
if nd:
    void.append("Nc not deterministic")
for a in T:
    if sum(ok(a, k) for k in K) < 12:
        void.append(f"{a}: {sum(ok(a, k) for k in K)} prompts")
def pm(a, b, k):
    v = [R[t][k]["tpsw"] for t in (a, b) if ok(t, k)]
    return sum(v) / len(v) if len(v) == 2 else None
g = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")
ks = [k for k in K if pm("nfdc", "nfdcb", k) and pm("nfnc", "nfncb", k)]
dc = {k: pm("nfdc", "nfdcb", k) for k in ks}
nc = {k: pm("nfnc", "nfncb", k) for k in ks}
r = g([nc[k] / dc[k] for k in ks]) if ks else float("nan")
L.append(f"steady tok/s (tpsw, pair means, geo over {len(ks)} prompts both arms kept): Dc {g(list(dc.values())):.2f}, "
         f"Nc {g(list(nc.values())):.2f} -> Nc / Dc = {r:.4f} ({100 * (r - 1):+.1f} %)")
dom = {}
for k in ks:
    dom.setdefault(k.split("/")[0], []).append(nc[k] / dc[k])
L.append("per domain Nc / Dc: " + ", ".join(f"{d} {100 * (g(v) - 1):+.1f} % ({len(v)})" for d, v in dom.items()))
L.append("order effect (Dc pass 2 / pass 1, Nc pass 2 / pass 1): "
         + ", ".join(f"{100 * (g([R[b][k]['tpsw'] / R[a][k]['tpsw'] for k in ks]) - 1):+.1f} %" for a, b in
                     (("nfdc", "nfdcb"), ("nfnc", "nfncb"))))
for a in T:
    try:
        fp = [x for x in open(os.path.join(D, f"{a}_{C}.txt")) if "footprint now" in x]
        L.append(f"{a}: {fp[-1].strip()[:110] if fp else 'no footprint line'}")
    except OSError:
        L.append(f"{a}: no screen output")
if bug:
    v = "BUG (Dc tokens differ from REF2-T: 3c-N on the converted store is not REF2-T)"
elif void or not ks:
    v = "VOID (" + "; ".join(void or ["no prompt in both arms"]) + ")"
else:
    v = "PASS" if r >= 1.06 else ("KILL" if r < 1.02 else "BETWEEN")
L.append(f"Verdict (pre-registered Thu 19:0x: PASS Nc / Dc >= +6 %, KILL < +2 %, else BETWEEN; BUG / VOID rules): {v}")
open(os.path.join(D, "natfri_report.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
EOF
}

if [ -n "$NATFRI_REPORT_ONLY" ]; then report "${1:?C}"; exit $?; fi
A=${NATFRI_AVAIL:-$(avail_gb)}
CC=${NATFRI_C:-}
if [ -z "$CC" ]; then
  # NATFRI_MARGIN (Fri 01:5x, after run 1's swap guard at C 224 with 0.5 GB to spare): GB kept beyond the rung's need
  for c in 224:44.3 208:42.2 192:40.0 176:37.9 160:35.7 144:33.6 128:31.4; do awk -v a="$A" -v n="${c#*:}" -v m="${NATFRI_MARGIN:-0}" 'BEGIN {exit !(n + m <= a)}' && { CC=${c%%:*}; break; }; done
fi
[ -z "$CC" ] && { say "NO-SWAP $A GB < 31.4 + margin ${NATFRI_MARGIN:-0}: nothing fits, nothing runs"; exit 0; }
SRC=$HOME/BigMoE/scripts/r57_screen.py
FS="$D/snap/r57_screen_natfri.py"; RUN="$D/snap/c45_runner_natfri.sh"
ANCHOR='OUT = os.path.expanduser("~/BigMoE/data/r57"); os.makedirs(OUT, exist_ok=True)'
[ "$(grep -cF "$ANCHOR" "$SRC")" = 1 ] || { say "the screen's OUT line moved (anchor not found once): nothing runs"; exit 2; }
sed 's#^OUT = os.path.expanduser("~/BigMoE/data/r57")#OUT = os.path.expanduser("~/Leap/data/natfri/r57")#' "$SRC" > "$FS.tmp" \
  && [ "$(grep -c '^OUT = os.path.expanduser("~/Leap/data/natfri/r57")' "$FS.tmp")" = 1 ] \
  && [ "$(grep -c 'BigMoE/data/r57")' "$FS.tmp")" = 0 ] && mv "$FS.tmp" "$FS" \
  && cp "$HOME/BigMoE/scripts/c45_runner.sh" "$RUN.tmp" && mv "$RUN.tmp" "$RUN" || { say "copy / patch failed: nothing runs"; exit 2; }
say "NATIVE-FRI phase C start (pid $$, NO-SWAP $A GB -> C $CC, swap $(swap_gb) GB); screen $(md5 -q "$SRC") (BigMoE $(git -C "$HOME/BigMoE" log -1 --format=%h -- scripts/r57_screen.py)), runner $(md5 -q "$RUN")"
ARMS="nfdc_$CC:$CC nfnc_$CC:$CC:R57_NAX=1,R57_REFTOK= nfncb_$CC:$CC:R57_NAX=1,R57_REFTOK= nfdcb_$CC:$CC"
if [ -n "$NATFRI_DRY" ]; then
  say "DRY: C45_SCREEN=$FS C45_OUT=$D R57_KEYS=<14 ref14> C45_REFTOK=$REF:gate+t C45_ALARM=900 MG=${NATFRI_MG:-4}"
  say "DRY: C45_ARMS=\"$ARMS\" /bin/bash $RUN   (then the report)"
  exit 0
fi
export C45_SCREEN="$FS" C45_OUT="$D" C45_SWAP0=$(swap_gb) C45_ALARM=900 MG=${NATFRI_MG:-4}
export C45_DEADLINE=${NATFRI_DEADLINE:-$(date -v+50M +%H%M)}
R57_KEYS=$K C45_REFTOK="$REF:gate+t" C45_ARMS="$ARMS" /bin/bash "$RUN"
say "arms rc=$? (deadline $C45_DEADLINE)"
report "$CC" >> "$LOG" 2>&1
say "report rc=$? ($D/natfri_report.txt)"
say "NATIVE-FRI phase C end"
