#!/bin/bash
# PF-D on multi-chunk prompts (BigMoE, Tue 10 Nov 00:3x; the orchestrator's question via Leap 00:2x: "Is PF-D safe on prompts
# longer than the ~101-token IFEval prompts it was measured on?"). PF-D was only ever checked on single-chunk prompts (12 N45
# IFEval, WED_TESTS Sun 17:23-18:4x); a long prompt runs it once per 1,024-row chunk with its cross-call staging (STAGE /
# GRAVE / LAST) and the AHEAD read-ahead. Here: the published chat-45 config through the release package's run.py
# (~/Leap/release/cleanclone/final/clone, PF-D on by default: R57_PFD=1, V=4, AHEAD=1) on Leap's 9,988- and 29,988-token
# prompts (~/Leap/data/prefill_1109/prompt_{10k,30k}.txt), 64 new tokens, R57_LOGHASH=1:
#   nopfd = R57_PFD=0 (the stock chunk-major prefill), pfd = as published, check = as published + R57_PFD_CHECK=1 (every
#   assembled scratch row vs tier._read_rebuilt's; timing void). EXACT iff pfd and check give nopfd's tokens (and logits hashes
#   where recorded) and the check arm counts 0 differing rows (bad_A / bad_B / bad_C).
# Order: 10k nopfd, 10k pfd, 10k check, 30k nopfd, 30k pfd, 30k check. Gates per run (Leap's bench_prefill.sh): last start
# (10K 01:08, 30K 01:00), every run capped so all ends by 01:18 (the w24 card gate may start 01:25); no chat.py / serve.py;
# NO-SWAP >= the run's need; stop on swap +2 GB or Kamen back. Needs a held quiet lock (turn45.sh).
P=$HOME/BigMoE/data/pfd_long; mkdir -p $P/out
CL=$HOME/Leap/release/cleanclone/final/clone; cd $CL || exit 1
PR=$HOME/Leap/data/prefill_1109
ep() { date -j -f "%Y-%m-%d %H:%M:%S" "$1" +%s; }
END=$(ep "2026-11-10 01:18:00"); L10=$(ep "2026-11-10 01:08:00"); L30=$(ep "2026-11-10 01:00:00")
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
idle() { ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print int($NF/1000000000); exit}'; }
other() { ps -axo pid=,comm=,args= | awk '$2 ~ /[Pp]ython/ && /[ \/](chat|serve)[.]py( |$)/ {printf "%s ", $1}'; }
killtree() { for c in $(pgrep -P $1); do killtree $c $2; done; kill -$2 $1 2>/dev/null; }
say() { echo "$(date +%T) $*" | tee -a $P/summary.txt; }
S0=$(swap); say "PF-D long-prompt check start: swap $S0 MB, NO-SWAP $(noswap) GB"
for cell in "10k nopfd" "10k pfd" "10k check" "30k nopfd" "30k pfd" "30k check"; do
  set -- $cell; SZ=$1; AR=$2; TAG=${SZ}_${AR}
  ENV="--env R57_LOGHASH=1"; [ $AR = nopfd ] && ENV="$ENV --env R57_PFD=0"; [ $AR = check ] && ENV="$ENV --env R57_PFD_CHECK=1"
  if [ $SZ = 10k ]; then LS=$L10; NEED=48; else LS=$L30; NEED=50; fi
  [ "$(date +%s)" -ge "$LS" ] && { say "SKIP $SZ $AR: past its last start"; continue; }
  [ -n "$(other)" ] && { say "STOP: a chat.py / serve.py is running ($(other))"; break; }
  A=$(noswap); [ "$A" -lt "$NEED" ] && { say "SKIP $SZ $AR: NO-SWAP $A GB < $NEED GB"; continue; }
  CAP=$(( END - $(date +%s) )); [ $CAP -gt 600 ] && CAP=600
  say "run $SZ $AR (NO-SWAP $A GB, need $NEED; swap $(swap) MB; cap $CAP s)"
  .venv/bin/python run.py --config chat --mem 45 --prompt-file $PR/prompt_$SZ.txt --max-tokens 64 --out $P/out $ENV > $P/run_$TAG.out 2>&1 &
  RP=$!; T0=$(date +%s); SMAX=$S0; ACT=0; WHY=""; PZ=""
  while kill -0 $RP 2>/dev/null; do
    s=$(swap); [ "$s" -gt "$SMAX" ] && SMAX=$s
    i=$(idle); if [ "$i" -lt 5 ]; then ACT=$((ACT + 1)); else ACT=0; fi
    [ -e ~/.local/share/quiet/PAUSE ] && PZ=", heat PAUSE seen"
    if [ $((s - S0)) -gt 2048 ]; then WHY="swap $s MB (start $S0)"
> (a quote from the project owner, kept private)
    elif [ $(( $(date +%s) - T0 )) -gt $CAP ]; then WHY="cap $CAP s"
    fi
    [ -n "$WHY" ] && { say "STOP $SZ $AR: $WHY"; killtree $RP TERM; break; }
    sleep 3
  done
  wait $RP 2>/dev/null
  L=$(grep -a -E "^whole decode [0-9.]+ tok/s" $P/run_$TAG.out | tail -1)
  TK=$(grep -a -o "^tokens: [^;]*" $P/run_$TAG.out | tail -1 | cut -c9-); echo "$TK" > $P/tokens_$TAG
  say "  $SZ $AR: ${L:-no result (see run_$TAG.out)} | wall $(( $(date +%s) - T0 )) s | swap max $SMAX MB$PZ"
  case "$WHY" in swap*|Kamen*) break;; esac
done
/opt/homebrew/opt/python@3.12/bin/python3.12 -I - "$P" <<'EOF' | tee -a $P/summary.txt
import json, os, sys
P = sys.argv[1]
def load(sz, ar):
    f = os.path.join(P, f"tokens_{sz}_{ar}")
    path = open(f).read().strip() if os.path.exists(f) else ""
    if not (path and os.path.exists(path)):
        return None
    T = json.load(open(path))                          # keyed by the run's own prompt key (multi/cli_<time>): compare values
    rp = path.replace("_tokens.json", "_runs.json")
    R = json.load(open(rp)) if os.path.exists(rp) else []
    return list(T.values()), R, [r.get("lh") for r in R if r.get("lh") is not None]
for sz in ("10k", "30k"):
    got = {ar: load(sz, ar) for ar in ("nopfd", "pfd", "check")}
    ref = got["nopfd"]
    for ar in ("pfd", "check"):
        g = got[ar]
        if ref is None or g is None:
            print(f"{sz} {ar}: not checked (runs present: {', '.join(k for k, v in got.items() if v) or 'none'})")
            continue
        same = g[0] == ref[0]
        hs = "no logits hashes recorded" if not (g[2] or ref[2]) else ("logits hashes == nopfd" if g[2] == ref[2] else "logits hashes DIFFER")
        pfd = [r.get("pfd") for r in g[1] if r.get("pfd")]
        st = pfd[-1] if pfd else {}
        print(f"{sz} {ar}: tokens {'== nopfd' if same else 'DIFFER from nopfd'}, {hs}; PF-D rows A {st.get('A')} / B {st.get('B')} / "
              f"C {st.get('C')}, calls {st.get('calls')}, fallback {st.get('fallback')}, checked {st.get('checked')}, "
              f"bad A/B/C {st.get('bad_A')}/{st.get('bad_B')}/{st.get('bad_C')}")
EOF
