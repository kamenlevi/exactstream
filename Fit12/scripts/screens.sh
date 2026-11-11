#!/bin/sh
# Fit12 rapid-screen runner (Mon 26 Oct 2026; pacer RULES.md 'Rapid screening'): the screens in $SCREENS, one after
# another, each under its own 10-min quiet lock taken here; run detached (nohup, ppid 1). Pass / kill lines are written
# first in ~/Fit12/PLAN.md. Before an arm: no PAUSE, SSD <= 66 C, enough lock left; after each grant the GPU must be idle
# (else unlock, wait 2 min, re-queue; 3 tries). During an arm: SSD >= 73 C, swap +2 GB, a lost lock or 600 s stop it; a
# PAUSE seen during an arm is logged (that arm's timings are void). Log: /Users/kamen/Fit12/logs/summary.txt ("SCR");
# outputs /Users/kamen/Fit12/data/scr/<screen>/<arm>.txt. Launch:
#   cd /Users/kamen/Fit12 && SCREENS="cmp" CLAUDE_PID=23445 nohup sh /Users/kamen/Fit12/scripts/screens.sh >/dev/null 2>&1 &
F=$HOME/Fit12; LOG=$F/logs/summary.txt; PY=$HOME/BigMoE/.venv/bin/python; PAUSEF=$HOME/.local/share/quiet/PAUSE
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH" PYTHONDONTWRITEBYTECODE=1
LOCKF=$HOME/.claude-peers/locks/${CLAUDE_PID:?CLAUDE_PID}.json
cd "$F" || exit 1
log() { echo "$(date +%H:%M:%S) SCR $*" >> "$LOG"; }
kill_tree() { for c in $(pgrep -P "$1"); do kill_tree "$c" "$2"; done; kill -"$2" "$1" 2>/dev/null; }   # RULES.md 27 Oct: whole tree
keepalive() { $PY -I -S -c '
import json, os, sys, time
p = sys.argv[1]
try: d = json.load(open(p))
except Exception: sys.exit(1)
if d.get("mode") == "share": sys.exit(0 if d.get("until", 0) > time.time() else 1)   # slock (R-45): a reservation, no lapse
if d.get("mode") != "quiet" or d.get("until", 0) <= time.time(): sys.exit(1)
d["active_at"] = time.time(); t = "%s.tmp%d" % (p, os.getpid()); json.dump(d, open(t, "w")); os.replace(t, p)' "$LOCKF"; }
left() { $PY -I -S -c 'import json, sys, time; print(int(json.load(open(sys.argv[1])).get("until", 0) - time.time()))' "$LOCKF" 2>/dev/null || echo 0; }
ssd_t() { tail -1 "$HOME/Library/Logs/health.csv" | awk -F, -v now="$(date +%s)" '{ if (now - $1 < 180) printf "%d", $6 }'; }
swapmb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6)}'; }
gpu() { peer top 2>/dev/null | head -1 | grep -o 'GPU [0-9]*%' | grep -o '[0-9]*'; }
top1() { ps -Ao pcpu,comm -r | sed -n 2p | awk '{n = split($2, a, "/"); print $1 "% " a[n]}'; }

HEALTH=$HOME/.local/share/quiet/HEALTH.md
act() { grep -q '^\*\*Verdict: ACT' "$HEALTH" 2>/dev/null; }
ok_time() { [ -e "$F/data/NIGHT_OK" ] && return 0   # Kamen Thu 00:27: tests may run tonight until his stop (PAUSE still holds)
  m=$(date +%H:%M | awk -F: '{print $1 * 60 + $2}'); [ "$m" -ge 481 ] && [ $((m * 60 + $1)) -le 86340 ]; }
ramp() { pgrep -f '^/bin/zsh [^ ]*soft_end\.sh' >/dev/null; }   # 27 Oct: soft_end.sh's 15-min resume ramp SIGSTOPs arms too
lid_closed() { ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes; }   # lid closed = pause
gate() {  # gate NEED_S (batch 2): wait while PAUSE exists, the health monitor says ACT, or NEED_S from now would pass 23:59
  w=0     # (silent night 00:01-08:01 unless data/NIGHT_OK; Breakthrough's and Leap's runners wait the same way); (Fri 23:3x) lid
  while [ -e "$PAUSEF" ] || act || ! ok_time "$1" || ramp || lid_closed; do
    [ $w -eq 0 ] && log "waiting: PAUSE $([ -e "$PAUSEF" ] && echo on || echo off), $(grep -m1 '^\*\*Verdict' "$HEALTH" | cut -c3-60), hours $(ok_time "$1" && echo ok || echo no), resume ramp $(ramp && echo on || echo off), lid $(lid_closed && echo closed || echo open)"
    w=$((w + 1)); sleep 60
  done
  [ $w -gt 0 ] && log "gate open after $w min"
  return 0
}

q45() {  # (30 Oct; NewIdeas 02:3x, Kamen 21:1x '45 GB first'): print the first 45 GB ticket waiting in the quiet queue
  for qf in "$HOME"/.claude-peers/queue/*.json; do   # (Leap, EvalSuite, BigMoE items), not counting this session's own
    [ -f "$qf" ] || continue
    case "$qf" in *-"${CLAUDE_PID}".json) continue ;; esac
    if grep -qiE '"why": "(Leap|evalsuite|BigMoE)|S45|45 GB|R-5[5-7]' "$qf"; then
      sed 's/.*"why": "\([^"]*\)".*/\1/' "$qf" | cut -c1-90; return 0
    fi
  done
  return 1
}
lock() {  # lock "why" [DUR (10m)] [NEED_S (600)] [GPU_MAX % (5)]: a quiet lock with an idle GPU after the grant, outside
  # PAUSE / ACT / night. Memory-only screens pass GPU_MAX 30 (29 Oct): desktop / game use (8-40 %) leaves their numbers alone,
  # while a detached decode (80-97 %) still blocks them. A grant while a 45 GB item waits behind it is given back (q45; up to
  # 40 yields, then the screen stops), and the request is made again once no 45 GB ticket waits.
  y=0 try=0
  while [ $try -lt 6 ]; do   # (30 Oct 19:0x) 6 tries, was 3: Kamen's own desktop use can keep the GPU busy for a while
    try=$((try + 1))
    gate "${3:-600}"
    h=; if [ -n "$F12_HELD" ]; then   # (30 Oct) the runner was launched under the session's own grant (the peers hook blocks
      F12_HELD=                       # launches while others hold or wait): use that grant once if NEED_S is left, else drop it
      if keepalive && [ "$(left)" -ge "${3:-600}" ]; then h=1; else peer unlock >/dev/null 2>&1; fi
    fi
    if [ -n "$h" ]; then log "using the session's held quiet lock ($(left) s left) for '$1'"
    else PEER_QUEUE_TIMEOUT=10800 peer lock quiet --queue "${2:-10m}" "$1" >> "$LOG" 2>&1 || { log "no lock: $1"; return 1; }; fi
    if act || ! ok_time "${3:-600}"; then log "lock came at ACT or too late for '$1': released"; peer unlock >/dev/null 2>&1; continue; fi
    if [ -z "$5" ] && w=$(q45); then   # a 5th argument (45 GB items themselves): no yield
      peer unlock >/dev/null 2>&1; y=$((y + 1)); try=$((try - 1))
      [ $y -gt 40 ] && { log "45 GB items kept waiting behind '$1' ($y yields): giving up"; return 1; }
      log "45 GB item waits behind me ($w): lock given back (yield $y)"
      while q45 >/dev/null; do sleep 60; done
      continue
    fi
    sleep 3; g=$(gpu); gp=100; gn=1   # (30 Oct 19:0x) wait up to ~2 min for 2 samples in a row <= GPU_MAX, 5 s apart (was the
    while { [ "${g:-0}" -gt "${4:-5}" ] || [ "$gp" -gt "${4:-5}" ]; } && [ $gn -lt 24 ]; do   # lowest of 4 in 12 s): a grant's
      gp=${g:-0}; sleep 5; g=$(gpu); gn=$((gn + 1))         # first seconds can catch the previous holder's exit, a UI burst or
    done                                                    # Kamen's desktop app (45 % at the 19:02 grant)
    if [ "${g:-0}" -le "${4:-5}" ] && [ "$gp" -le "${4:-5}" ]; then SW0=$(swapmb); log "lock ($(left) s): $1 | GPU $gp, ${g:-?} % (sample $gn), top $(top1), SSD $(ssd_t) C, swap $SW0 MB"; return 0; fi
    log "GPU $g % after the grant (top $(top1)): unlock, retry in 2 min"; peer unlock >/dev/null 2>&1; sleep 120
  done
  return 1
}
unlock() { peer unlock >/dev/null 2>&1; log "lock released (SSD $(ssd_t) C, swap $(swapmb) MB)"; }
fmem() { vm_stat | awk '/page size of/ {ps = $8} /^Pages (free|inactive|speculative|purgeable):/ {gsub("\\.", "", $NF); n += $NF} END {printf "%d", n * ps / 1e9}'; }
slock() {  # slock "why" [DUR (15m)] [NEED_S (600)] [NOT_BEFORE HHMM (0801)]: a shared lock for memory-only runs (R-45, NewIdeas
  nb=${4:-0801} w=0 r=0 on=0   # 28 Oct), after NOT_BEFORE, outside PAUSE / ACT / night / ramp, once `peer wait` says nothing
  while :; do             # blocks (others' quiet locks and queued tickets go first) and >= 16 GB are free; gives up at 21:00
    gate "${3:-600}"      # of the day it started trying (a launch the evening before waits for NOT_BEFORE first)
    hm=$(date +%H%M)      # (Sat 31 Oct 00:5x) NOT_BEFORE 0000 = any hour, no 21:00 give-up (no night gap unless announced);
    if [ $on -eq 0 ]; then   # 5th argument NEED_GB (16) on NO-SWAP (avail_gb; the shared rule is the peak + 8 GB)
      if [ "$nb" = "0000" ] || { [ "$hm" -ge "$nb" ] && [ "$hm" -lt 2100 ]; }; then on=1
      else [ $w -eq 0 ] && log "share lock for '$1': waiting until $nb"; w=1; sleep 60; continue; fi
    fi
    [ "$nb" != "0000" ] && [ "$hm" -ge 2100 ] && { log "share lock for '$1': not granted by 21:00: giving up"; return 1; }
    peer wait --timeout 30m >/dev/null 2>&1
    gate "${3:-600}"
    fg=$(avail_gb)
    if [ "${fg:-0}" -lt "${5:-16}" ]; then
      [ $((r % 10)) -eq 0 ] && log "share lock for '$1': ${fg:-?} GB no-swap (< ${5:-16}): retrying every 2 min"; r=$((r + 1)); sleep 120; continue
    fi
    if out=$(peer lock share "${2:-15m}" --cpu 400 --gpu 80 "$1" 2>&1); then
      SW0=$(swapmb); log "share lock ($(left) s): $1 | GPU $(gpu) %, top $(top1), SSD $(ssd_t) C, swap $SW0 MB, $fg GB free"; return 0
    fi
    [ $((r % 10)) -eq 0 ] && log "share lock refused for '$1': $(echo "$out" | sed -n 2p | cut -c1-120)"; r=$((r + 1)); sleep 60
  done
}

arm() {  # arm SCREEN TAG NEED_S "ENV ..." SCRIPT [CAP_S (600)]: one fresh process "SCRIPT TAG", output data/scr/SCREEN/TAG.txt
  scr=$1 tag=$2 need=$3 E=$4 S=$5 cap=${6:-600}; O=$F/data/scr/$scr; mkdir -p "$O"
  i=0; while t=$(ssd_t); { [ -z "$t" ] || [ "$t" -gt 66 ]; } && [ $i -lt 30 ]; do sleep 10; i=$((i + 1)); [ $((i % 6)) -eq 0 ] && keepalive; done
  t=$(ssd_t); if [ -z "$t" ] || [ "$t" -gt 66 ]; then log "$scr/$tag: SSD '${t:-unknown}' C after 5 min: skipped"; return 3; fi
  while [ -e "$PAUSEF" ] || ramp; do sleep 5; keepalive; done
  if [ "$(left)" -lt "$need" ]; then log "$scr/$tag: < $need s of lock left: skipped"; return 4; fi
  log "$scr/$tag start (SSD $t C, GPU $(gpu) %, top $(top1); $E)"
  env $E $PY -B "$S" ${ARGS:-"$tag"} > "$O/$tag.txt" 2>&1 &   # ARGS (S45b-2): the script's own arguments instead of TAG
  P=$!; j=0; rc=0; pz=0
  while kill -0 $P 2>/dev/null; do
    sleep 5; j=$((j + 5)); t=$(ssd_t); [ -e "$PAUSEF" ] && pz=1
    if [ -n "$t" ] && [ "$t" -ge 73 ]; then log "$scr/$tag: SSD $t C: stopping"; rc=73; fi
    [ $(( $(swapmb) - SW0 )) -gt 2048 ] && { log "$scr/$tag: swap +2 GB: stopping"; rc=127; }
    [ $((j % 60)) -eq 0 ] && { keepalive || { log "$scr/$tag: lock lost: stopping"; rc=125; }; }
    [ $j -ge $cap ] && { log "$scr/$tag: TIMEOUT $cap s"; rc=124; }
    [ $rc -ne 0 ] && { kill_tree $P TERM; sleep 5; kill_tree $P KILL; break; }
  done
  wait $P; w=$?; [ $rc -eq 0 ] && rc=$w
  log "$scr/$tag done (rc $rc, $j s, PAUSE seen $pz, swap $(swapmb) MB, SSD $(ssd_t) C): $(grep ' RESULT ' "$O/$tag.txt" | sed 's/.* RESULT //' | cut -c1-90 | tr '\n' ';')$(grep -E 'Error|Traceback' "$O/$tag.txt" | tail -1 | cut -c1-200)"
  return $rc
}

scr_cmp() {  # PLAN.md 'Rapid screens, batch 1' CMP
  lock "Fit12 CMP screen: per-prompt C_m, MTP path, 2.6K + 8.5K (+32K peak), <= 10 min" || return 1
  X="F12_V3F=$F/v3p F12_PFR=1 F12_PFR_CAP=16384 F12_NDEC=96"
  arm cmp cm51 150 "$X F12_PROMPTS=p2k,p8k B12_CM=51" $F/scripts/fit_cmp.py &&
  arm cmp cm59 150 "$X F12_PROMPTS=p2k,p8k B12_CM=59" $F/scripts/fit_cmp.py &&
  arm cmp cm51_32k 200 "$X F12_PROMPTS=p32k B12_CM=51" $F/scripts/fit_cmp.py
  unlock
}

scr_cmpr() {  # PLAN.md 'Rapid screens, batch 1' CMPR: repeat at 8.5K + Breakthrough's v3l (read-only)
  lock "Fit12 CMPR screen: repeat of CMP at 8.5K + v3l cross-check (hashes), ~3 min" || return 1
  X="F12_PFR=1 F12_PFR_CAP=16384 F12_NDEC=96 F12_PROMPTS=p8k F12_LAYERS=1"
  arm cmp r51 90 "$X F12_V3F=$F/v3p B12_CM=51" $F/scripts/fit_cmp.py &&
  arm cmp r59 90 "$X F12_V3F=$F/v3p B12_CM=59" $F/scripts/fit_cmp.py &&
  arm cmp l51 90 "$X F12_V3F=$HOME/Breakthrough/v3l B12_CM=51" $F/scripts/fit_cmp.py &&
  arm cmp l59 90 "$X F12_V3F=$HOME/Breakthrough/v3l B12_CM=59" $F/scripts/fit_cmp.py
  unlock
}

scr_mov() {  # PLAN.md 'Rapid screens, batch 1' MOV: no second copy of the prefill state, 32K
  lock "Fit12 MOV screen: MTP decode without the second prefill-state copy, 32K, ~2 min" || return 1
  arm mov m51_32k 200 "F12_V3F=$F/v3p F12_PFR=1 F12_PFR_CAP=16384 F12_NDEC=96 F12_PROMPTS=p32k B12_CM=51 F12_MOVE=1" $F/scripts/fit_cmp.py
  unlock
}

scr_movc() {  # MOV's control (PLAN.md AMENDMENT 17:56): the same arm without F12_MOVE, same hash definition
  lock "Fit12 MOVC: control for the MOV screen (32K, ~2 min)" || return 1
  arm mov c51_32k 200 "F12_V3F=$F/v3p F12_PFR=1 F12_PFR_CAP=16384 F12_NDEC=96 F12_PROMPTS=p32k B12_CM=51 F12_LAYERS=1" $F/scripts/fit_cmp.py
  unlock
}

scr_cmp2() {  # PLAN.md CMP2: C_m 59 with C_l held at 20 (and C_m 51 / C_l 24), 8.5K
  lock "Fit12 CMP2 screen: C_m vs C_l bit test at 8.5K, ~1.5 min" || return 1
  X="F12_V3F=$F/v3p F12_PFR=1 F12_PFR_CAP=16384 F12_NDEC=96 F12_PROMPTS=p8k F12_LAYERS=1"
  arm cmp x59l20 90 "$X B12_CM=59 B12_CL=20" $F/scripts/fit_cmp.py &&
  arm cmp x51l24 90 "$X B12_CM=51 B12_CL=24" $F/scripts/fit_cmp.py
  unlock
}

scr_cmp3() {  # PLAN.md CMP3 diagnosis: single-row passes wait for every rank (Fit12 copy of bt_b12)
  lock "Fit12 CMP3 diagnosis: single-row wait rule vs C_m at 8.5K, ~1.5 min" || return 1
  X="F12_V3F=$F/v3p F12_PFR=1 F12_PFR_CAP=16384 F12_NDEC=96 F12_PROMPTS=p8k F12_LAYERS=1 F12_BTC=1 F12_ALLWAIT=1 B12_CL=20"
  arm cmp w51 90 "$X B12_CM=51" $F/scripts/fit_cmp.py &&
  arm cmp w59 90 "$X B12_CM=59" $F/scripts/fit_cmp.py
  unlock
}

scr_mov64() {  # PLAN.md MOV64: the MTP path + MOV at ~64K, C_m 51
  lock "Fit12 MOV64 screen: MTP path + MOV at 64K (peak), ~4.5 min" || return 1
  arm mov m51_64k 330 "F12_V3F=$F/v3p F12_PFR=1 F12_PFR_CAP=16384 F12_NDEC=96 F12_PROMPTS=p64k B12_CM=51 B12_CL=20 F12_MOVE=1" $F/scripts/fit_cmp.py
  unlock
}

scr_aw() {  # PLAN.md AW: the ALLWAIT change (cost, tokens, bitwise C_m invariance), 256 tokens, MOV on
  lock "Fit12 AW screen: single-row wait-all rule, cost + exactness, ~5 min" || return 1
  X="F12_V3F=$F/v3p F12_PFR=1 F12_PFR_CAP=16384 F12_NDEC=256 F12_MOVE=1 B12_CL=20 F12_BTC=1"
  arm aw s51 100 "$X F12_PROMPTS=p2k,p8k B12_CM=51" $F/scripts/fit_cmp.py &&
  arm aw a51 100 "$X F12_PROMPTS=p2k,p8k B12_CM=51 F12_ALLWAIT=1" $F/scripts/fit_cmp.py &&
  arm aw a59 100 "$X F12_PROMPTS=p2k,p8k B12_CM=59 F12_ALLWAIT=1" $F/scripts/fit_cmp.py &&
  arm aw a65 60 "$X F12_PROMPTS=p2k B12_CM=65 F12_ALLWAIT=1" $F/scripts/fit_cmp.py
  unlock
}

scr_hcb() {  # PLAN.md HCB: bf16 hyper-connection mixers (B12_HC=0) vs HC8 on the 12 GB stack (MOV + ALLWAIT)
  lock "Fit12 HCB screen: bf16 HC mixers at 12 GB (peak + speed), ~2.5 min" || return 1
  X="F12_V3F=$F/v3p F12_PFR=1 F12_PFR_CAP=16384 F12_NDEC=256 F12_MOVE=1 B12_CL=20 F12_BTC=1 F12_ALLWAIT=1 B12_CM=51 F12_PROMPTS=p2k,p8k"
  arm hcb h8 100 "$X B12_HC=8" $F/scripts/fit_cmp.py &&
  arm hcb h0 100 "$X B12_HC=0" $F/scripts/fit_cmp.py
  unlock
}

scr_pfm() {  # PLAN.md PFM: PFR on vs off on the MTP path (state hashes)
  lock "Fit12 PFM screen: PFR on the MTP path, on vs off (hashes + prefill), ~2.5 min" || return 1
  X="F12_V3F=$F/v3p F12_PFR_CAP=16384 F12_NDEC=256 F12_MOVE=1 B12_CL=20 F12_BTC=1 F12_ALLWAIT=1 B12_CM=51 F12_PROMPTS=p2k,p8k"
  arm pfm pf0 100 "$X F12_PFR=0" $F/scripts/fit_cmp.py &&
  arm pfm pf1 100 "$X F12_PFR=1" $F/scripts/fit_cmp.py
  unlock
}

scr_mem2() {  # PLAN.md MEM2: resident-memory census of the 12 GB stack
  lock "Fit12 MEM2 screen: memory census of the 12 GB stack, ~1.5 min" || return 1
  arm mem2 census 90 "F12_V3F=$F/v3p F12_PFR=0 F12_MOVE=1 B12_CL=20 B12_CM=51 F12_BTC=1 F12_ALLWAIT=1" $F/scripts/fit_mem2.py
  unlock
}

scr_aw1k() {  # PLAN.md AW1K: ALLWAIT over 1,024 tokens at 8.5K
  lock "Fit12 AW1K screen: wait-all rule over 1,024 tokens, ~4 min" || return 1
  X="F12_V3F=$F/v3p F12_PFR=1 F12_PFR_CAP=16384 F12_NDEC=1024 F12_MOVE=1 B12_CL=20 B12_CM=51 F12_BTC=1 F12_PROMPTS=p8k"
  arm aw1k s51 150 "$X" $F/scripts/fit_cmp.py &&
  arm aw1k a51 150 "$X F12_ALLWAIT=1" $F/scripts/fit_cmp.py
  unlock
}

scr_mvc() {  # PLAN.md MVC: MOV alone (shipped rule), C_m 51 vs 59, fresh p8k
  lock "Fit12 MVC diagnosis: MOV alone vs C_m at 8.5K, ~1.5 min" || return 1
  X="F12_V3F=$F/v3p F12_PFR=1 F12_PFR_CAP=16384 F12_NDEC=96 F12_MOVE=1 B12_CL=20 F12_PROMPTS=p8k F12_LAYERS=1"
  arm cmp v51 90 "$X B12_CM=51" $F/scripts/fit_cmp.py &&
  arm cmp v59 90 "$X B12_CM=59" $F/scripts/fit_cmp.py
  unlock
}

scr_gap() {  # PLAN.md 'Rapid screens, batch 2' R-04 GAP: the engine's small expert kernels after idle gaps (gapclock port)
  lock "Fit12 R-04 GAP screen: expert-kernel microbench after idle gaps, ~1 min" 10m 300 || return 1
  arm gap g1 90 "" $F/scripts/fit_gap.py 300
  unlock
}

scr_kvc() {  # R-02 KVC: QSA buffers sized once (mtplx qsa_rows_target) vs stock growth, 32K decode peak
  lock "Fit12 R-02 KVC screen: QSA KV sized once vs stock growth at 32K, ~4 min" 10m 300 || return 1
  X="F12_V3F=$F/v3p F12_PFR=1 F12_PFR_CAP=16384 F12_NDEC=96 F12_MOVE=1 B12_CL=20 F12_BTC=1 F12_ALLWAIT=1 B12_CM=51 F12_PROMPTS=p32k"
  arm kvc k0 130 "$X" $F/scripts/fit_cmp.py 240 &&
  arm kvc k1 130 "$X F12_TGT=16" $F/scripts/fit_cmp.py 240
  unlock
}

scr_warm() {  # R-03 WARM bound: p8k twice (16, then 256 tokens); the 2nd decode cold (PFR on) vs warm (PFR off)
  lock "Fit12 R-03 WARM screen: warm vs cold pools at decode start, 8.5K, ~3 min" 10m 300 || return 1
  X="F12_V3F=$F/v3p F12_PFR_CAP=16384 F12_NDECS=16,256 F12_MOVE=1 B12_CL=20 F12_BTC=1 F12_ALLWAIT=1 B12_CM=51 F12_PROMPTS=p8k,p8k"
  arm warm w0 110 "$X F12_PFR=1" $F/scripts/fit_cmp.py 240 &&
  arm warm w1 110 "$X F12_PFR=0" $F/scripts/fit_cmp.py 240
  unlock
}

scr_hcx() {  # R-14 HC12 step 1: the price of bt_b12's HC8 reads + writes (all run twice, tokens equal), 3 paired rounds
  lock "Fit12 R-14 HC price screen: HC reads doubled vs stock, 8.5K, 3 paired rounds, ~5 min" 10m 420 || return 1
  X="F12_V3F=$F/v3p F12_PFR=1 F12_PFR_CAP=16384 F12_NDEC=256 F12_MOVE=1 B12_CL=20 F12_BTC=1 F12_ALLWAIT=1 B12_CM=51 F12_PROMPTS=p8k"
  for r in 1 2 3; do
    arm hcx h1_$r 70 "$X" $F/scripts/fit_cmp.py 150 && arm hcx h2_$r 70 "$X F12_HCX=1" $F/scripts/fit_cmp.py 150 || break
  done
  unlock
}

scr_laysz() {  # R-28 LAYSZ-X: CPU replay (1 core), per-layer pool sizes under LRU / LFU, 4 / 3 and 16 / 10 checks
  lock "Fit12 R-28 LAYSZ-X: CPU replay, 1 core, per-layer pool sizes, ~4 min" 10m 300 || return 1
  arm laysz x1 60 "VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1" $F/scripts/fit_laysz.py 540
  unlock
}

scr_pos2b() {  # pos-2 follow-up (27 Oct): the prefill logits of a repeated p8k saved at both positions, fold stack vs stock
  LEAK=1
  lock "Fit12 pos-2 follow-up: repeated 8.5K prompt, fold stack vs stock, logits saved, ~3 min" 10m 300 || return 1
  O=$F/data/scr/pos2b; mkdir -p $O
  X="F12_V3F=$F/v3p F12_NDECS=16,16 B12_CL=20 B12_CM=51 F12_PROMPTS=p8k,p8k F12_SAVE_LOGITS=$O"
  arm pos2b s1 110 "$X F12_PFR=1 F12_PFR_CAP=16384 F12_BTC=1 F12_ALLWAIT=1 F12_MOVE=1" $F/scripts/fit_cmp.py 240 &&
  arm pos2b s0 110 "$X F12_PFR=0 F12_BTC=0 F12_ALLWAIT=0 F12_MOVE=0" $F/scripts/fit_cmp.py 240; r=$?
  unlock
  [ $r -eq 0 ] && $PY -B $F/scripts/fit_pos2.py $O 2>&1 | while read -r l; do log "$l"; done
  LEAK=1; [ $r -eq 0 ] && tail -20 "$LOG" | grep -q -E "\[pos2b\] verdict: (PRE-EXISTING|NONE)" && LEAK=0   # NewIdeas 20:47
  log "pos2b: $([ $LEAK -eq 0 ] && echo 'R-01 may run (fresh = warm judged on sha, kv, head)' || echo 'R-01 waits: bisect first')"
  return $r
}

scr_grow() {  # GROW (27 Oct, after R-01; Breakthrough's split): 32K, the same prompt twice vs two different prompts
  lock "Fit12 GROW: 32K memory growth per prompt, same vs different prompt, 2 arms, ~7 min" 10m 480 || return 1
  X="F12_V3F=$F/v3p F12_PFR=1 F12_PFR_CAP=16384 F12_NDEC=32 F12_MOVE=1 B12_CL=20 F12_BTC=1 F12_ALLWAIT=1 B12_CM=51"
  arm grow g1 230 "$X F12_PROMPTS=c32k_1,c32k_1" $F/scripts/fit_cmp.py 300 &&
  arm grow g2 230 "$X F12_PROMPTS=c32k_1,c32k_2" $F/scripts/fit_cmp.py 300
  unlock
}

scr_gfix() {  # R-42 GROWFIX (28 Oct): 5 x 32K in one process on bt_b12's 31b63af defaults; stock, then a trim between prompts
  for a in s t; do
    lock "Fit12 R-42 GROWFIX arm $a: 4 x 32K prompts in one process, memory per prompt, ~9 min" 10m 560 || return 1
    arm gfix $a 540 "F12_PROMPTS=c32k_1,c32k_2,c32k_3,c32k_4 $([ $a = t ] && echo F12_TRIM=1 || echo F12_TRIM=0)" $F/scripts/fit_gfix.py 590; r=$?
    unlock
    [ $r -eq 0 ] || return $r
  done
}

scr_creep() {  # R-45 CREEPMAP (29 Oct): 8 of the suite's math rows to EOS in one process on fold-exact (bt_b12 c3f957cba6), memory only
  lock "Fit12 R-45 CREEPMAP: 8 math rows in one process, memory only, ~9 min" 15m 620 || return 1   # orchestrator 00:28: quiet lock, now
  arm creep c1 560 "" $F/scripts/fit_creep.py 600; r=$?
  unlock
  return $r
}

scr_creepf() {  # R-45F (29 Oct): R-45's harness on fold-fast (b12llff), the same 8 math rows, gc between rows
  lock "Fit12 R-45F: fold-fast, 8 math rows in one process, memory only, ~8 min" 15m 620 || return 1
  arm creep f1 560 "F12_ARM=ff" $F/scripts/fit_creep.py 600; r=$?
  unlock
  return $r
}

scr_creepr() {  # R-45R (29 Oct, NewIdeas GO 01:4x): who owns the residual after gc; fold-exact rows 1-3 with tracemalloc
  lock "Fit12 R-45R: fold-exact, 3 math rows with tracemalloc, memory only, ~6 min" 15m 620 || return 1
  arm creep r1 400 "F12_TRACE=1 F12_NROWS=3" $F/scripts/fit_creep.py 600; r=$?
  unlock
  return $r
}

scr_creeps() {  # R-45S (29 Oct; Breakthrough's split): the native residual after gc, the same row x3 vs three different rows
  lock "Fit12 R-45S: same row x3 vs 3 different rows, memory only, ~5 min" 15m 400 30 || return 1
  arm creep s3 200 "F12_ROWS=3,3,3 F12_VMALL=1" $F/scripts/fit_creep.py 300; r=$?
  [ $r -eq 0 ] && { arm creep d3 200 "F12_ROWS=3,10,12 F12_VMALL=1" $F/scripts/fit_creep.py 300; r=$?; }
  unlock
  return $r
}

scr_creept() {  # R-45T MALLOCSTACKS (29 Oct; NewIdeas GO 19:1x): the call stacks of the large blocks each request keeps
  lock "Fit12 R-45T: malloc stacks of the per-request blocks, memory only, ~8 min" 15m 600 30 || return 1
  arm creep t3 540 "F12_ROWS=3,10,12 F12_VMALL=1 F12_MHIST=1 MallocStackLogging=lite F12_BT=$F/pinned/bt_b12.py" \
    $F/scripts/fit_creep.py 840; r=$?
  unlock
  return $r
}

scr_r45u() {  # R-45U V3O RESIDUE (Sat 31 Oct; NewIdeas GO 00:5x, low priority, shared GPU; PLAN.md 'R-45U'): the call stacks
  # of the ~3-6 MB each request still keeps after gc on v3o; malloc_history after rows 3 and 6 of 2,3,4,2,3,4 (no new content
  # in the second pass), memory only, no timing
  slock "Fit12 R-45U: malloc stacks of v3o's per-request residue (12 GB, memory only, no timing), ~10 min" 20m 900 0801 22 || return 1   # (NIGHT MODE 01:4x: 12 GB screens by day)
  arm r45u u6 900 "F12_ROWS=2,3,4,2,3,4 F12_MHIST=1 F12_MHIST_ROWS=3,6 F12_VMALL=1 MallocStackLogging=lite F12_HOT_MB=16 F12_BT=$F/pinned/b677/bt_b12.py F12_ENGINE=v3o F12_OUT=$F/data/scr/r45u" \
    $F/scripts/fit_creep.py 1140; r=$?
  unlock
  return $r
}

bigproc() {  # (Sun 1 Nov) other processes over 8 GB of physical footprint (another model would make a memory number unclean);
  # top's MEM, since ps' RSS misses MLX's Metal buffers (an MLX model process shows ~1 GB RSS)
  top -l 1 -n 8 -o mem -stats pid,command,mem 2>/dev/null | awk -v me="$$" 'NR > 1 && $1 ~ /^[0-9]+$/ && $1 != me {
    m = $NF; v = m + 0; g = (m ~ /G/) ? v : (m ~ /M/) ? v / 1024 : 0; if (g > 8) printf "%s %.1f GB; ", $2, g }'
}
scr_m5() {  # M-5 DECISION-5 MEMORY CHECK (Sun 1 Nov; NewIdeas 01:0x, FIRST; PLAN.md 'M-5'): lifetime peak of the live 12 GB
  # default (cf8bd7840a) on R-41C's 16 math rows, then its 29 agent rows, one process each, memory only (shared locks)
  E0="F12_ARM=live F12_BT=$F/pinned/cf8b/bt_b12.py F12_OUT=$F/data/scr/m5"
  for set in math agent; do
    nr=16; [ $set = agent ] && nr=29
    w=0; while b=$(bigproc) && [ -n "$b" ] && [ $w -lt 60 ]; do [ $w -eq 0 ] && log "m5/$set: waiting, big process running: $b"; w=$((w + 1)); sleep 60; done
    [ -n "$(bigproc)" ] && { log "m5/$set: a big process still runs after 60 min: re-queued"; requeue m5; return 1; }
    slock "Fit12 M-5: lifetime peak of the live 12 GB default, $nr $set rows in one process (memory only, no timing), ~25 min" 35m 1500 0000 20 || return 1
    arm m5 $set 1500 "$E0 F12_SET=$set F12_NROWS=$nr" $F/scripts/fit_creep.py 2040; r=$?
    unlock
    b=$(bigproc); [ -n "$b" ] && log "m5/$set: big process seen at the end: $b (check the arm's swap / timing)"
    [ $r -eq 0 ] || return $r
  done
  return 0
}

scr_m5b() {  # M-5b BISECT (Sun 1 Nov; NewIdeas 07:5x GO; PLAN.md 'M-5b'): which piece of decision 5's bundle (eaff23d) raises
  # the math rows' in-row peak: rows 8 (warm-up), 3, 13 per arm, one process each, memory only (shared locks), resumable;
  # stage 2 (F: AW 0, G: CM 0) only if E's rows 3 / 13 maxima are >= 0.5 GB under A's
  O=$F/data/scr/m5b; LV="F12_ARM=live F12_BT=$F/pinned/cf8b/bt_b12.py"
  for a in A B C D E F G; do
    case $a in
      A) E1="$LV";;
      B) E1="F12_ARM=ff3 F12_BT=$F/pinned/b677/bt_b12.py";;
      C) E1="$LV F12_SETENV=B12_ENGINE=v3o";;
      D) E1="$LV F12_HOT_MB=1024";;
      E) E1="$LV F12_SETENV=B12_MODE=exact";;
      F) E1="$LV F12_SETENV=B12_FOLD_AW=0";;
      G) E1="$LV F12_SETENV=B12_FOLD_CM=0";;
    esac
    grep -q "] done" $O/$a.txt 2>/dev/null && continue
    if [ $a = F ]; then
      d=$(m5b_drop A E); log "m5b: rows 3 / 13 mean max, A - E = $d GB"
      awk -v d="$d" 'BEGIN { exit !(d >= 0.5) }' || { log "m5b: stage 2 not run (A - E $d < 0.5 GB)"; return 0; }
    fi
    w=0; while b=$(bigproc) && [ -n "$b" ] && [ $w -lt 60 ]; do [ $w -eq 0 ] && log "m5b/$a: waiting, big process running: $b"; w=$((w + 1)); sleep 60; done
    [ -n "$(bigproc)" ] && { log "m5b/$a: a big process still runs after 60 min: re-queued"; requeue m5b; return 1; }
    slock "Fit12 M-5b arm $a: math memory bisect of the live 12 GB default, 3 rows in one process (memory only, no timing), ~5 min" 12m 400 0000 21 || return 1
    arm m5b $a 400 "F12_OUT=$O F12_SET=math F12_ROWS=8,3,13 $E1" $F/scripts/fit_creep.py 720; r=$?
    unlock
    b=$(bigproc); [ -n "$b" ] && log "m5b/$a: big process seen at the end: $b (check this arm before using it)"
    [ $r -eq 0 ] || return $r
  done
  return 0
}
scr_m5c() {  # M-5b stage 3 (Sun 1 Nov 11:4x; PLAN.md 'M-5b stage 3'): which R-53 part (F12_R53A / B / C_MB = 0 on the live
  # file), same rows and harness as scr_m5b, outputs beside its arms in data/scr/m5b; resumable
  O=$F/data/scr/m5b; LV="F12_ARM=live F12_BT=$F/pinned/cf8b/bt_b12.py"
  for a in H I J; do
    case $a in
      H) E1="$LV F12_R53A=0";;
      I) E1="$LV F12_R53B=0";;
      J) E1="$LV F12_R53C_MB=0";;
    esac
    grep -q "] done" $O/$a.txt 2>/dev/null && continue
    w=0; while b=$(bigproc) && [ -n "$b" ] && [ $w -lt 60 ]; do [ $w -eq 0 ] && log "m5c/$a: waiting, big process running: $b"; w=$((w + 1)); sleep 60; done
    [ -n "$(bigproc)" ] && { log "m5c/$a: a big process still runs after 60 min: re-queued"; requeue m5c; return 1; }
    slock "Fit12 M-5b stage 3 arm $a: which R-53 part raises the 12 GB default's short-prompt prefill peak, 3 rows (memory only), ~3 min" 12m 400 0000 21 || return 1
    arm m5b $a 400 "F12_OUT=$O F12_SET=math F12_ROWS=8,3,13 $E1" $F/scripts/fit_creep.py 720; r=$?
    unlock
    b=$(bigproc); [ -n "$b" ] && log "m5c/$a: big process seen at the end: $b (check this arm before using it)"
    [ $r -eq 0 ] || return $r
    log "m5c: rows 3 / 13 mean max, A - $a = $(m5b_drop A $a) GB"
  done
  return 0
}
scr_m5d() {  # M-5d (Sun 1 Nov 11:5x; Breakthrough's ask; PLAN.md 'M-5d'): the 16-row math lifetime peak of the new live
  # default 8db71cc537 (v3p + F12_R53A=0; first planned on 31e4333196 = v3o), pinned, one process, memory only (shared lock), as M-5's math arm
  O=$F/data/scr/m5d
  grep -q "] done" $O/math.txt 2>/dev/null && return 0
  w=0; while b=$(bigproc) && [ -n "$b" ] && [ $w -lt 60 ]; do [ $w -eq 0 ] && log "m5d: waiting, big process running: $b"; w=$((w + 1)); sleep 60; done
  [ -n "$(bigproc)" ] && { log "m5d: a big process still runs after 60 min: re-queued"; requeue m5d; return 1; }
  slock "Fit12 M-5d: 16-row math lifetime peak of the new live 12 GB default (v3p + R-53a off), one process (memory only, no timing), ~20 min" 30m 1300 0000 21 || return 1
  arm m5d math 1300 "F12_ARM=live F12_BT=$F/pinned/8db7/bt_b12.py F12_OUT=$O F12_SET=math F12_NROWS=16" $F/scripts/fit_creep.py 1800; r=$?
  unlock
  b=$(bigproc); [ -n "$b" ] && log "m5d: big process seen at the end: $b (check the arm before using it)"
  return $r
}
m5b_drop() {  # rows 3 and 13 (the arm's rows 2 and 3): mean row max of arm $1 minus arm $2, GB
  for x in $1 $2; do grep -o "RESULT row [23] |.*footprint start [0-9.]* max [0-9.]*" $F/data/scr/m5b/$x.txt 2>/dev/null |
    awk '{s += $NF; n++} END {printf "%.3f\n", (n == 2 ? s / 2 : -99)}'; done | paste -s - | awk '{printf "%.3f", (($1 < 0 || $2 < 0) ? -99 : $1 - $2)}'
}

scr_grow2() {  # GROW2 (27 Oct): the same 32K prompt twice with PFR off (is PFR's release / re-create the growth?)
  lock "Fit12 GROW2: 32K same prompt twice, PFR off, 1 arm, ~4 min" 10m 300 || return 1
  arm grow g3 230 "F12_V3F=$F/v3p F12_PFR=0 F12_NDEC=32 F12_MOVE=1 B12_CL=20 F12_BTC=1 F12_ALLWAIT=1 B12_CM=51 F12_PROMPTS=c32k_1,c32k_1" $F/scripts/fit_cmp.py 300
  unlock
}

scr_pos2c() {  # pos-2 detail (27 Oct): which vocab ids differ in pos2b's saved logits (CPU numpy, ~2 s)
  lock "Fit12 pos-2 detail: numpy look at saved logits, CPU ~2 s" 10m 60 || return 1
  $PY -B $F/scripts/fit_pos2.py $F/data/scr/pos2b ids 2>&1 | grep "pos2c" | while read -r l; do log "$l"; done
  unlock
}

scr_pos2() {  # pos-2 check (27 Oct): R-03's repeated p8k had a new 'logits' hash at pos 2 (tokens, kv, head equal). Is the
  # prefill's own logits hash (lpre, taken before the decode) repeat-invariant? MOV on and off. Sets LEAK for scr_conf.
  LEAK=1
  lock "Fit12 pos-2 check: repeated 8.5K prompt, logits hash before / after decode, MOV on / off, ~3 min" 10m 300 || return 1
  X="F12_V3F=$F/v3p F12_PFR=1 F12_PFR_CAP=16384 F12_NDECS=16,16 B12_CL=20 F12_BTC=1 F12_ALLWAIT=1 B12_CM=51 F12_PROMPTS=p8k,p8k"
  arm pos2 m1 110 "$X F12_MOVE=1" $F/scripts/fit_cmp.py 240 && arm pos2 m0 110 "$X F12_MOVE=0" $F/scripts/fit_cmp.py 240; r=$?
  unlock
  LEAK=0
  for a in m1 m0; do
    o=$F/data/scr/pos2/$a.txt; n=$(grep -c ' RESULT ' $o); l=$(grep ' RESULT ' $o | grep -o 'lpre [0-9a-f]*' | sort -u | wc -l)
    [ "$n" -eq 2 ] && [ "$l" -eq 1 ] || LEAK=1
    log "pos2 $a: $n RESULT lines, $l distinct lpre; logits by position: $(grep ' RESULT ' $o | grep -o 'RESULT logits [0-9a-f]*' | cut -c15- | tr '\n' ' ')"
  done
  log "pos2: $([ $LEAK -eq 0 ] && echo 'lpre repeat-invariant in both arms: conf runs' || echo 'lpre moved or lines missing: conf skipped')"
  return $r
}

scr_conf() {  # R-01 CONF: per length a C_m 51 reference, then C_m(L) from its highest clean peak; prompts [1,2,3,4,5,1]
  [ "${LEAK:-0}" -eq 0 ] || { log "conf skipped: the pos-2 check did not pass"; return 1; }
  X="F12_V3F=$F/v3p F12_PFR=1 F12_PFR_CAP=16384 F12_NDEC=256 F12_MOVE=1 B12_CL=20 F12_BTC=1 F12_ALLWAIT=1"
  for L in 8k 2k 16k 32k; do
    case $L in 2k) d=10m need=240;; 8k) d=10m need=330;; 16k) d=15m need=480;; *) d=20m need=840;; esac
    PR="c${L}_1,c${L}_2,c${L}_3,c${L}_4,c${L}_5,c${L}_1"
    lock "Fit12 R-01 CONF $L: C_m 51 reference, 6 prompts x 256 tokens" $d $((need + 120)) || return 1
    arm conf r51_$L $need "$X B12_CM=51 F12_PROMPTS=$PR" $F/scripts/fit_cmp.py $((need + 120)); rc=$?
    unlock
    nr=$(grep -c ' RESULT ' $F/data/scr/conf/r51_$L.txt)
    pk=$(grep ' RESULT ' $F/data/scr/conf/r51_$L.txt | sed 's/.*clean peak \([0-9.]*\) GB.*/\1/' | sort -n | tail -1)
    [ $rc -eq 0 ] && [ "$nr" -eq 6 ] && [ -n "$pk" ] || { log "conf $L: reference rc $rc, $nr RESULT lines: no test arm"; continue; }
    cm=$(awk -v p="$pk" 'BEGIN { c = 51 + int((11.85 - p) / 0.104); print (c > 80 ? 80 : c) }')
    log "conf $L: reference peak $pk GB -> C_m $cm"
    [ "$cm" -gt 51 ] || continue
    lock "Fit12 R-01 CONF $L: C_m $cm, 6 prompts x 256 tokens" $d $((need + 120)) || return 1
    arm conf t${cm}_$L $need "$X B12_CM=$cm F12_PROMPTS=$PR" $F/scripts/fit_cmp.py $((need + 120))
    unlock
  done
}

if [ -z "$SCREENS_LIB" ]; then   # sourced by runq.sh (SCREENS_LIB=1): functions only
nohup sh $F/scripts/awake.sh $$ >/dev/null 2>&1 &   # awake only while an arm or lock wait runs (27 Oct: caffeinate -w held a quiet window awake)
log "runner start: SCREENS '$SCREENS' (pid $$)"
for s in $SCREENS; do scr_$s || log "screen $s stopped early"; done
log "runner end"
fi

scr_r53() {  # R-53 SCRATCH12 A/B (30 Oct; NewIdeas GO Thu 20:50; PLAN.md 'R-53 A/B pre-registration' + amendment): the live
  # prefill (A = ~/Breakthrough/v3n, read-only) vs the scratch-reuse copy (B = ~/Fit12/v3n53), bt_b12 23b42972ef pinned,
  # p8k + p16k, 32 spec125 tokens each; order A B | B A in two 15-min quiet locks (B1 doubles as the patch's smoke test)
  E0="F12_BT=$F/pinned/b23/bt_b12.py"
  lock "Fit12 R-53 (1/2): prefill scratch reuse A/B, live vs patched, p8k + p16k, ~6 min" 15m 600 || return 1
  arm r53 a1 200 "$E0 F12_V3F=$HOME/Breakthrough/v3n" $F/scripts/fit_scr53.py 360; r=$?
  [ $r -eq 0 ] && { arm r53 b1 200 "$E0 F12_V3F=$F/v3n53" $F/scripts/fit_scr53.py 360; r=$?; }
  unlock
  [ $r -eq 0 ] || return $r
  lock "Fit12 R-53 (2/2): prefill scratch reuse A/B, patched vs live, p8k + p16k, ~6 min" 15m 600 || return 1
  arm r53 b2 200 "$E0 F12_V3F=$F/v3n53" $F/scripts/fit_scr53.py 360; r=$?
  [ $r -eq 0 ] && { arm r53 a2 200 "$E0 F12_V3F=$HOME/Breakthrough/v3n" $F/scripts/fit_scr53.py 360; r=$?; }
  unlock
  return $r
}

cool() {  # cool [MAX_C (55)] [WAIT_S (300)] (R-53C, 30 Oct): wait until fanwatch's die max <= MAX_C at thermal state 0, so every
  # timed arm starts from the same thermal state (at FAN_CAP 3500 the die heats ~25 C in 2 min of load); keepalive meanwhile.
  # Its variables are cool-only names: runq.sh keeps the screen's name in $s (30 Oct: an 's' here logged 'screen 0').
  ci=0
  while :; do
    cl=$(tail -40 "$HOME/Library/Logs/fanwatch.log" 2>/dev/null | grep '^[0-9][0-9]:[0-9][0-9]:[0-9][0-9] die max' | tail -1)
    cdie=$(echo "$cl" | awk '{printf "%d", $4}'); cst=$(echo "$cl" | awk '{print $14}')
    [ -n "$cdie" ] && [ "$cdie" -le "${1:-55}" ] && [ "$cst" = 0 ] && { log "cool: die $cdie C, state $cst (${ci}0 s)"; return 0; }
    [ $ci -ge $(( ${2:-300} / 10 )) ] && { log "cool: die ${cdie:-?} C, state ${cst:-?} after ${2:-300} s: going on"; return 1; }
    sleep 10; ci=$((ci + 1)); [ $((ci % 6)) -eq 0 ] && keepalive
  done
}

scr_r53d() {  # R-53D (Sun 1 Nov; NewIdeas 12:4x; PLAN.md 'R-53D'): v3o vs v3p + F12_R53A=0 prefill on the live file 8db71cc537
  # (its fast defaults: F12_LIVE=1), p8k + p16k, A1 B1 | B2 A2, cool starts, shared locks (timings: shared GPU, rough); resumable
  E0="F12_BT=$F/pinned/8db7/bt_b12.py F12_LIVE=1 F12_PROMPTS=p8k,p16k"
  EA="$E0 F12_V3F=$HOME/Breakthrough/v3o"; EB="$E0 F12_V3F=$HOME/Breakthrough/v3p F12_R53A=0"
  for h in 1 2; do
    if [ $h = 1 ]; then t1=a1 e1=$EA t2=b1 e2=$EB; else t1=b2 e1=$EB t2=a2 e2=$EA; fi
    grep -q "] done" $F/data/scr/r53d/$t2.txt 2>/dev/null && continue
    night45 2220 && { log "r53d: half $h not started: from 22:20 the night's 45 GB queue needs the memory (NewIdeas 20:0x); re-queue by hand"; return 0; }
    es_wait 300 || { log "r53d: EvalSuite decode still running after 5 h: re-queued"; requeue r53d; return 1; }
    w=0; while b=$(bigproc) && [ -n "$b" ] && [ $w -lt 60 ]; do [ $w -eq 0 ] && log "r53d: waiting, big process running: $b"; w=$((w + 1)); sleep 60; done
    [ -n "$(bigproc)" ] && { log "r53d: a big process still runs after 60 min: re-queued"; requeue r53d; return 1; }
    slock "Fit12 R-53D ($h/2): v3o vs v3p + R-53a off, p8k + p16k prefill, cool starts, ~8-15 min (shared, rough)" 25m 600 0000 21 || return 1
    cool 55 300; arm r53d $t1 150 "$e1" $F/scripts/fit_scr53.py 420; r=$?
    [ $r -eq 0 ] && night45 2235 && { log "r53d: $t2 not started (the night's 45 GB queue): re-queue by hand"; r=9; }
    [ $r -eq 0 ] && { cool 55 300; arm r53d $t2 150 "$e2" $F/scripts/fit_scr53.py 420; r=$?; }
    unlock
    [ $r -eq 0 ] || return $r
  done
  return 0
}

night45() {  # night45 HHMM (Sun 1 Nov 20:0x, NewIdeas): true from HHMM to 07:00, when the night's 45 GB quiet queue needs every GB
  t=$(date +%H%M); [ "$t" -ge "$1" ] || [ "$t" -lt 0700 ]
}

scr_r53c() {  # R-53C (30 Oct; PLAN.md 'R-53C confirmation lines'): R-53's A/B again with cool starts, p8k + p16k, A1 B1 | B2 A2
  E0="F12_BT=$F/pinned/b677/bt_b12.py F12_PROMPTS=p8k,p16k"   # (19:2x amendment) on the live engine: v3o vs v3o53, bt_b12 677c6558b6
  memgate 20 || { requeue r53c; return 1; }   # (21:1x) GB available: the 12 GB line's peak + 8
  lock "Fit12 R-53C (1/2): prefill scratch reuse confirmation, cool starts, p8k + p16k, ~8 min" 15m 600 || return 1
  cool 55 300; arm r53c a1 150 "$E0 F12_V3F=$HOME/Breakthrough/v3o" $F/scripts/fit_scr53.py 300; r=$?
  [ $r -eq 0 ] && { cool 55 300; arm r53c b1 150 "$E0 F12_V3F=$F/v3o53" $F/scripts/fit_scr53.py 300; r=$?; }
  unlock
  [ $r -eq 0 ] || return $r
  memgate 20 || { requeue r53c; return 1; }
  lock "Fit12 R-53C (2/2): prefill scratch reuse confirmation, cool starts, p8k + p16k, ~8 min" 15m 600 || return 1
  cool 55 300; arm r53c b2 150 "$E0 F12_V3F=$F/v3o53" $F/scripts/fit_scr53.py 300; r=$?
  [ $r -eq 0 ] && { cool 55 300; arm r53c a2 150 "$E0 F12_V3F=$HOME/Breakthrough/v3o" $F/scripts/fit_scr53.py 300; r=$?; }
  unlock
  return $r
}

scr_r46() {  # R-46 HOTCAP (30 Oct; NewIdeas GO 02:3x; PLAN.md 'R-46 HOTCAP GO'): n-gram hot cap 1024 vs 1 MB, math rows 1-4
  E0="F12_BT=$F/pinned/b677/bt_b12.py F12_ENGINE=v3o F12_ROWS=1,2,3,4 F12_OUT=$F/data/scr/r46"   # (19:2x) live engine
  memgate 20 || { requeue r46; return 1; }    # (21:1x) GB available: the 12 GB line's peak + 8
  lock "Fit12 R-46 (1/2): n-gram hot cap 1024 vs 1 MB, 4 math rows per arm, ~9 min" 15m 600 || return 1
  cool 55 300; arm r46 a1 240 "$E0 F12_HOT_MB=1024" $F/scripts/fit_creep.py 420; r=$?
  [ $r -eq 0 ] && { cool 55 300; arm r46 b1 240 "$E0 F12_HOT_MB=1" $F/scripts/fit_creep.py 420; r=$?; }
  unlock
  [ $r -eq 0 ] || return $r
  memgate 20 || { requeue r46; return 1; }
  lock "Fit12 R-46 (2/2): n-gram hot cap 1 vs 1024 MB, 4 math rows per arm, ~9 min" 15m 600 || return 1
  cool 55 300; arm r46 b2 240 "$E0 F12_HOT_MB=1" $F/scripts/fit_creep.py 420; r=$?
  [ $r -eq 0 ] && { cool 55 300; arm r46 a2 240 "$E0 F12_HOT_MB=1024" $F/scripts/fit_creep.py 420; r=$?; }
  unlock
  return $r
}

mp_free() { memory_pressure 2>/dev/null | awk '/free percentage/ {sub("%", "", $NF); print int($NF)}'; }
avail_gb() {  # NO-SWAP GB (orchestrator Fri 30 Oct 23:5x, standing rule in RULES.md): free + speculative + purgeable +
  # file-backed pages (vm_stat). Gate: >= the lifetime peak + 4 GB under a quiet lock (after the grant and a 30 s settle),
  # + 8 GB under a shared one. Was AVAILABLE (+ inactive, which holds anonymous app pages only swap or compression free:
  # Leap's C 224 load passed it at 45.0 GB and tripped the swap guard, 23:51); before that memory_pressure's free share
  vm_stat | awk '/page size of/ {ps = $8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v = $NF; sub("[.]", "", v); s += v} END {printf "%d", s * ps / 1e9}'
}
memgate() {  # memgate NEED_GB [TRIES (23)]: wait (2-min steps) until avail_gb >= NEED_GB = the expected lifetime peak + 8 GB
  mw=0       # (FAST INTEGRATION; Kamen's apps count). Was memory_pressure's free share until Fri 21:1x
  while [ "$(avail_gb)" -lt "$1" ]; do
    [ $mw -eq 0 ] && log "memgate: $(avail_gb) GB no-swap < $1 GB: waiting"; mw=$((mw + 1))
    [ $mw -gt "${2:-23}" ] && return 1
    sleep 120
  done
  return 0
}
requeue() { grep -qx "$1" "$F/data/queue.txt" 2>/dev/null || echo "$1" >> "$F/data/queue.txt"; log "queue: $1 put back at the end"; }
s45t_held() {  # (Fri 23:5x) BigMoE's S45-T holds a quiet lock now
  now=$(date +%s)
  for lf in "$HOME"/.claude-peers/locks/*.json; do
    [ -f "$lf" ] || continue
    grep -q '"mode": "quiet"' "$lf" || continue
    u=$(sed -n 's/.*"until": \([0-9]*\).*/\1/p' "$lf" | head -1); { [ -n "$u" ] && [ "$u" -gt "$now" ]; } || continue
    grep -q '"why": "[^"]*S45-T' "$lf" && return 0
  done
  return 1
}
scr_w45t() {  # (Fri 23:5x, orchestrator 23:48 order: Leap's S45-STACK, BigMoE's cap45 at C 160, BigMoE's S45-T, then Fit12's
  # 45 GB screens) wait until BigMoE's S45-T lock has come and gone, at most 4 h; turn45 then adds the 15-min gap
  t0=$(date +%s); seen=0; log "w45t: waiting for BigMoE's S45-T to run first (orchestrator 23:48 order)"
  while [ $(( $(date +%s) - t0 )) -lt 14400 ]; do
    if s45t_held; then [ $seen -eq 0 ] && log "w45t: BigMoE's S45-T is running"; seen=1
    elif [ $seen -eq 1 ]; then log "w45t: BigMoE's S45-T lock is gone"; date +%s > "$F/data/last45"; return 0; fi
    sleep 60
  done
  log "w45t: 4 h without BigMoE's S45-T coming and going: going on"; return 0
}
l45() {  # (Fri 23:3x, orchestrator addendum) print another session's HELD 45 GB quiet lock (q45's patterns), if any
  now=$(date +%s)
  for lf in "$HOME"/.claude-peers/locks/*.json; do
    [ -f "$lf" ] || continue
    case "$lf" in */"${CLAUDE_PID}".json) continue ;; esac
    grep -q '"mode": "quiet"' "$lf" || continue
    u=$(sed -n 's/.*"until": \([0-9]*\).*/\1/p' "$lf" | head -1); { [ -n "$u" ] && [ "$u" -gt "$now" ]; } || continue
    if grep -qiE '"why": "(Leap|evalsuite|BigMoE)|S45|45 GB|R-5[5-7]' "$lf"; then
      sed 's/.*"why": "\([^"]*\)".*/\1/' "$lf" | cut -c1-90; return 0
    fi
  done
  return 1
}
turn45() {  # turn45 [GAP_S (900)] (Fri 23:3x, orchestrator addendum): one 45 GB ticket at a time across all sessions, 15 min
  # apart, so EvalSuite's REF2b chunks get the gaps. Waits while another session queues or holds one, and until GAP_S have
  # passed since the last one ended (data/last45: my own 45 GB arms' ends, and any other's last sighting here).
  cs=$(cat "$F/data/last45" 2>/dev/null || echo 0); tw=0
  while :; do
    if w=$(q45) || w=$(l45); then
      cs=$(date +%s); echo "$cs" > "$F/data/last45"; [ $tw -eq 0 ] && log "45 GB turn: waiting behind '$w'"; tw=1
    elif [ $(( $(date +%s) - cs )) -ge "${1:-900}" ]; then break
    elif [ $tw -eq 0 ]; then log "45 GB turn: the last one ended $(( $(date +%s) - cs )) s ago, waiting for ${1:-900} s"; tw=1; fi
    sleep 60
  done
  [ $tw -gt 0 ] && log "45 GB turn: mine now"
  return 0
}

scr_s45b2() {  # S45b-2 (30 Oct; NewIdeas 16:1x, a 45 GB item; PLAN.md 'S45b-2'): Leap's S45 live loop (copy) at C 224 / 256 / 288
  for c in 224 256 288; do   # (19:0x) resumable: a C with its s45_speed file is skipped; a swap trip stops every later run too
    [ -f $F/data/s45b2/s45_speed_c$c.txt ] && { log "s45b2/c$c: done before: skipped"; continue; }
    ls $F/data/s45b2/swaptrip_c* >/dev/null 2>&1 && { log "s45b2/c$c: an earlier point tripped the swap guard: sweep stopped"; return 0; }
    m=50; [ $c -eq 256 ] && m=53; [ $c -eq 288 ] && m=57   # (21:1x) GB available: the peak (40.5 / ~44.7 / ~48.9) + 8, and >= 50 for ~45 GB loads (brief)
    k=0
    while :; do   # (23:3x, orchestrator addendum) queue first, memory after the grant: a queued ticket stops new shared runs
      turn45 900  # and the grant waits for the running ones, so memory drains for the ticket
      lock "Fit12 S45b-2: Leap's S45 live loop (REF2b) at C $c, 6 prompts, ~9 min (45 GB item for NewIdeas)" 15m 600 5 noyield || return 1
      sleep 30                                          # the previous holder's memory drains
      [ "$(avail_gb)" -ge $m ] && break
      k=$((k + 1)); log "s45b2/c$c: $(avail_gb) GB no-swap < $m GB 30 s after the grant: released ($k), queue again in 2 min"
      unlock; [ $k -ge 8 ] && { requeue s45b2; return 1; }; sleep 120
    done
    ARGS="$c 256 speed" arm s45b2 c$c 560 "" $F/leap_s45/r40s3k_screen_f12.py 600; r=$?
    unlock; date +%s > "$F/data/last45"
    [ $r -eq 127 ] && { touch $F/data/s45b2/swaptrip_c$c; log "s45b2/c$c: swap trip: sweep stopped"; return 0; }
    [ $r -eq 0 ] || return $r
  done
  return 0
}

scr_r59() {  # R-59 DENSE16 (30 Oct; NewIdeas QUEUE R-59, a 45 GB item; PLAN.md 'R-59 DENSE16'): per-stage profile of a warm REF2b
  # 16-row check at C 224 (in place + op groups), one quiet lock. Not below C 160: the loader aborts when one call needs more
  # experts than C, and a 16-row check needs up to 160 per layer (HELD with S45b-2 from Fri 23:3x until memory is back)
  k=0
  while :; do   # (23:3x, orchestrator addendum) queue first, memory after the grant (GB available: peak 40.5 + 8, >= 50)
    [ -e "$F/data/NIGHT_FIFO" ] || turn45 900   # (Sat 01:4x NIGHT MODE: one quiet ticket at a time, FIFO, no gap; NO-SWAP gate peak + 4 = 45 GB)
    lock "Fit12 R-59: per-stage profile of a warm REF2b 16-row check at C 224 (45 GB item for NewIdeas), ~6 min" 15m 600 5 noyield || return 1
    sleep 30
    [ "$(avail_gb)" -ge 45 ] && break
    k=$((k + 1)); log "r59: $(avail_gb) GB no-swap < 45 GB 30 s after the grant: released ($k), queue again in 2 min"
    unlock; [ $k -ge 8 ] && { requeue r59; return 1; }; sleep 120
  done
  ARGS="224 10" arm r59 c224 400 "" $F/scripts/fit_r59.py 540; r=$?
  unlock; date +%s > "$F/data/last45"
  return $r
}

scr_sg1() {  # SG12 step 1a (Sun 1 Nov; PLAN.md 'SG12 short-prompt grouped prefill'): v3p (live) vs v3q + F12_SG12=1 on M-5d's
  # binding math prompts (m10 m13 m3 m15: 95 / 69 / 98 / 78 tokens), live file 8db71cc537 (F12_LIVE=1): KV / h / logits / sha32
  # equal, and the prefill-phase peaks; arms a b, one shared lock, after the day's one SSD-heavy run (bigproc); no timing claim
  E0="F12_BT=$F/pinned/8db7/bt_b12.py F12_LIVE=1 F12_PROMPTS=m10,m13,m3,m15"
  grep -q "] done" $F/data/scr/sg1/b.txt 2>/dev/null && return 0
  es_wait 300 || { log "sg1: EvalSuite decode still running after 5 h: re-queued"; requeue sg1; return 1; }
  w=0; while b=$(bigproc) && [ -n "$b" ] && [ $w -lt 60 ]; do [ $w -eq 0 ] && log "sg1: waiting, big process running: $b"; w=$((w + 1)); sleep 60; done
  [ -n "$(bigproc)" ] && { log "sg1: a big process still runs after 60 min: re-queued"; requeue sg1; return 1; }
  slock "Fit12 SG12 step 1a: v3p vs v3q (short-prompt grouped prefill), 4 math prompts, hashes + memory, ~8 min (shared)" 20m 600 0000 21 || return 1
  arm sg1 a 150 "$E0 F12_V3F=$HOME/Breakthrough/v3p" $F/scripts/fit_scr53.py 300; r=$?
  [ $r -eq 0 ] && { arm sg1 b 150 "$E0 F12_V3F=$F/v3q F12_SG12=1" $F/scripts/fit_scr53.py 300; r=$?; }
  unlock
  return $r
}

es_wait() {  # es_wait [MAX_MIN (300)] (Sun 1 Nov 15:2x): until EvalSuite's decode runs (runner 37 PMIN12, the day's one SSD-heavy
  # run, EvalSuite 14:51) are gone 5 minutes in a row: its ~17-min agent processes restart with short gaps, which bigproc missed
  n=0 w=0 said=0
  while [ $n -lt 5 ]; do
    if pgrep -f "evalsuite.stages decode" >/dev/null 2>&1; then n=0; else n=$((n + 1)); fi
    [ $n -eq 0 ] && [ $said -eq 0 ] && { log "es_wait: EvalSuite decode running: waiting for 5 idle minutes"; said=1; }
    w=$((w + 1)); [ $w -gt "${1:-300}" ] && return 1
    sleep 60
  done
  [ $said -eq 1 ] && log "es_wait: EvalSuite idle 5 min"
  return 0
}

scr_ht1() {  # HT1 (Mon 2 Nov, PLAN.md 'HT1'): HumanEval+ prefill memory trace on the live 12 GB default (pinned 8db71cc537),
  # 5 rows in one process, memory only (shared lock)
  O=$F/data/scr/ht1
  grep -q "MEMTRACE row 5" $O/he.txt 2>/dev/null && return 0
  es_wait 120 || { log "ht1: EvalSuite decode still running after 2 h: re-queued"; requeue ht1; return 1; }
  w=0; while b=$(bigproc) && [ -n "$b" ] && [ $w -lt 60 ]; do [ $w -eq 0 ] && log "ht1: waiting, big process running: $b"; w=$((w + 1)); sleep 60; done
  [ -n "$(bigproc)" ] && { log "ht1: a big process still runs after 60 min: re-queued"; requeue ht1; return 1; }
  w=0; while o=$(bigrun) && [ -n "$o" ] && [ $w -lt 90 ]; do
    [ $w -eq 0 ] && log "ht1: waiting, another session's run (one > 5 GB job at a time by day, orchestrator 20:50): $o"; w=$((w + 1)); sleep 60; done
  [ -n "$o" ] && { log "ht1: another session's run still listed after 90 min: re-queued"; requeue ht1; return 1; }
  slock "Fit12 HT1: HumanEval+ prefill memory trace, MEM 12 GB, memory only, ~5 min (shared)" 15m 400 0000 21 || return 1
  arm ht1 he 400 "F12_ARM=live F12_BT=$F/pinned/8db7/bt_b12.py F12_OUT=$O F12_SET=humaneval F12_REFARM=b12llff4 F12_KEYS=HumanEval/68,HumanEval/156,HumanEval/45,HumanEval/42,HumanEval/8 F12_MEMTRACE=1" $F/scripts/fit_creep.py 900; r=$?
  unlock
  b=$(bigproc); [ -n "$b" ] && log "ht1: big process seen at the end: $b (check the arm before using it)"
  return $r
}

bigrun() {  # (Mon 2 Nov 21:2x) another session's listed run that may exceed ~5 GB (the day rule: one such job at a time);
  # runs marked CPU only, '<= n GB' or 'MEM n GB' with n <= 5 do not count
  # (Tue 3 Nov 14:4x) match peer ls's run annotation only: ' run (' also matched status text ("NOT run (memory") and
  # held T3P on an idle session
  peer ls 2>/dev/null | grep -F 'shared run (others may work)' | grep -v '^You:' | grep -v ojox-12 \
    | grep -v -E 'CPU only|<= ?[0-5](\.[0-9]+)? GB|MEM [0-5](\.[0-9]+)? GB' | cut -c1-90 | head -1
}

scr_t0g() {  # T0 gates (Mon 2 Nov, PLAN.md 'T0'): two-pass decode vs the exact AR decode on the live file, 3 prompts x 6 arms
  O=$F/data/scr/t0g
  grep -q "] done" $O/g1.txt 2>/dev/null && return 0
  es_wait 120 || { log "t0g: EvalSuite decode still running after 2 h: re-queued"; requeue t0g; return 1; }
  w=0; while o=$(bigrun) && [ -n "$o" ] && [ $w -lt 90 ]; do
    [ $w -eq 0 ] && log "t0g: waiting, another session's run (one > 5 GB job at a time by day): $o"; w=$((w + 1)); sleep 60; done
  [ -n "$o" ] && { log "t0g: another session's run still listed after 90 min: re-queued"; requeue t0g; return 1; }
  slock "Fit12 T0 gates: two-pass decode vs exact AR, MEM 10 GB, 3 prompts, ~6 min (shared)" 15m 600 0000 21 || return 1
  arm t0g g1 600 "F12_OUT=$O" $F/scripts/fit_t0.py 900; r=$?
  unlock
  return $r
}

scr_sgb0() {  # SGB step 0 (Mon 2 Nov, PLAN.md 'SGB'): balanced-group micro test on synthetic experts, < 5 GB
  O=$F/data/scr/sgb0
  grep -q "^VERDICT" $O/s0.txt 2>/dev/null && return 0
  slock "Fit12 SGB step 0: balanced prefill groups micro test, MEM 4 GB, ~2 min (shared)" 10m 200 0000 12 || return 1
  arm sgb0 s0 200 "" $F/scripts/sgb_micro.py 420; r=$?
  unlock
  return $r
}

scr_sgb0b() {  # SGB step 0 re-run on the greedy split (Mon 2 Nov 22:0x; information: v3r's amended _sgb_groups rule)
  O=$F/data/scr/sgb0
  grep -q "^VERDICT" $O/s0b.txt 2>/dev/null && return 0
  slock "Fit12 SGB step 0b: greedy-split micro test, MEM 4 GB, ~2 min (shared)" 10m 200 0000 12 || return 1
  arm sgb0 s0b 200 "" $F/scripts/sgb_micro.py 420; r=$?
  unlock
  return $r
}

scr_t2g() {  # T2 gates (Mon 2 Nov 22:1x, PLAN.md 'T1 + T2'): two-pass MTP check vs spec125's exact decode, 3 prompts x 7 arms
  O=$F/data/scr/t2g
  grep -q "] done" $O/g1.txt 2>/dev/null && return 0
  es_wait 120 || { log "t2g: EvalSuite decode still running after 2 h: re-queued"; requeue t2g; return 1; }
  w=0; while o=$(bigrun) && [ -n "$o" ] && [ $w -lt 90 ]; do
    [ $w -eq 0 ] && log "t2g: waiting, another session's run (one > 5 GB job at a time by day): $o"; w=$((w + 1)); sleep 60; done
  [ -n "$o" ] && { log "t2g: another session's run still listed after 90 min: re-queued"; requeue t2g; return 1; }
  slock "Fit12 T2 gates: two-pass MTP check vs exact, MEM 12 GB, 3 prompts, ~10 min (shared)" 25m 900 0000 21 || return 1
  arm t2g g1 900 "F12_OUT=$O" $F/scripts/fit_t2.py 1500; r=$?
  unlock
  return $r
}

scr_t2s() {  # T2s (Mon 2 Nov 23:0x, PLAN.md 'T2s'): the skip-pass-1 rule, 5 prompts, information
  O=$F/data/scr/t2s
  grep -q "] done" $O/s1.txt 2>/dev/null && return 0
  es_wait 120 || { log "t2s: EvalSuite decode still running after 2 h: re-queued"; requeue t2s; return 1; }
  w=0; while o=$(bigrun) && [ -n "$o" ] && [ $w -lt 90 ]; do
    [ $w -eq 0 ] && log "t2s: waiting, another session's run (one > 5 GB job at a time by day): $o"; w=$((w + 1)); sleep 60; done
  [ -n "$o" ] && { log "t2s: another session's run still listed after 90 min: re-queued"; requeue t2s; return 1; }
  slock "Fit12 T2s: two-pass MTP check + skip rule, MEM 12 GB, 5 prompts, ~8 min (shared)" 20m 700 0000 21 || return 1
  arm t2s s1 700 "F12_OUT=$O F12_KEYS=humaneval/HumanEval/156,humaneval/HumanEval/42,ifeval/,math500/test/number_theory/459.json,math500/test/algebra/509.json F12_T2ARMS=ref:spec,t80:0.8,t80s:0.8s,t95s:0.95s" $F/scripts/fit_t2.py 1200; r=$?
  unlock
  return $r
}

scr_t3g() {  # T3 gates (Mon 2 Nov 23:4x, PLAN.md 'T3'): DFlash2 L1-20 (D7b) 16-row two-pass checks on B12, 4 prompts x 7 arms
  O=$F/data/scr/t3g
  grep -q "] done" $O/g1.txt 2>/dev/null && return 0
  es_wait 120 || { log "t3g: EvalSuite decode still running after 2 h: re-queued"; requeue t3g; return 1; }
  w=0; while o=$(bigrun) && [ -n "$o" ] && [ $w -lt 90 ]; do
    [ $w -eq 0 ] && log "t3g: waiting, another session's run (one > 5 GB job at a time by day): $o"; w=$((w + 1)); sleep 60; done
  [ -n "$o" ] && { log "t3g: another session's run still listed after 90 min: re-queued"; requeue t3g; return 1; }
  slock "Fit12 T3 gates: DFlash2 D7b 16-row two-pass on B12, MEM 12 GB, 4 prompts, ~15 min (shared)" 25m 1000 0000 21 || return 1
  arm t3g g1 1000 "F12_OUT=$O F12_CMCL=47,32" $F/scripts/fit_t3.py 1500; r=$?
  unlock
  return $r
}

mapbusy() {  # (Tue 3 Nov 01:5x) Breakthrough's decision map waits to start or runs: run_day9.sh lives, or its watcher
  # (wait_map / wait_map2) lives with one of tonight's outputs (xcapture, t3_all, t4_all) still missing
  pgrep -f "run_day9\.sh" >/dev/null && return 0
  pgrep -f "wait_map2?\.sh" >/dev/null || return 1
  for n in xcapture t3_all t4_all; do [ -e "$HOME/FableGod/decision_map/out/$n.json" ] || return 0; done
  return 1
}

scr_t3t() {  # T3 timing (Tue 3 Nov, PLAN.md 'T3 timing'): quiet lock, 7 prompts x 6 arms, 256 tokens
  # (Tue 01:5x) tonight's slot (NewIdeas 01:4x): after Breakthrough's decision map and Leap's small screens, before EvalSuite's
  # IFX (GO or 05:30) and MTP Head's 06:00 chain. Before 15:00: no quiet ticket while the map waits or runs (its guard stops a
  # running map for any other session's quiet window, held or waited for), no start or grant from 05:05 (left for a day slot),
  # no bigrun wait (the day rule is for Kamen's day). From 15:00: the bigrun wait as before. Then NO-SWAP >= 16 GB (peak ~11.4 + 4).
  O=$F/data/scr/t3t
  grep -q "] done" $O/q1.txt 2>/dev/null && return 0
  es_wait 120 || { log "t3t: EvalSuite decode still running after 2 h: re-queued"; requeue t3t; return 1; }
  if [ "$(date +%H%M)" -lt 1500 ]; then
    w=0; while mapbusy && [ "$(date +%H%M)" -lt 0505 ]; do
      [ $w -eq 0 ] && log "t3t: waiting for Breakthrough's decision map to finish (NewIdeas 01:4x order)"; w=$((w + 1)); sleep 60; done
    [ $w -gt 0 ] && log "t3t: map wait over after $w min"
    hm=$(date +%H%M); [ "$hm" -ge 0505 ] && { log "t3t: $hm, past tonight's 05:05 start limit (IFX 05:30, MTP Head 06:00): left for a day slot"; return 1; }
  else
    w=0; while o=$(bigrun) && [ -n "$o" ] && [ $w -lt 90 ]; do
      [ $w -eq 0 ] && log "t3t: waiting, another session's run: $o"; w=$((w + 1)); sleep 60; done
    [ -n "$o" ] && { log "t3t: another session's run still listed after 90 min: re-queued"; requeue t3t; return 1; }
  fi
  memgate 16 || { log "t3t: NO-SWAP under 16 GB for 46 min: left for a day slot"; return 1; }
  lock "Fit12 T3 timing: DFlash2 D7b two-pass at 12 GB, MEM 12 GB, quiet, ~20 min" 30m 1500 || return 1
  hm=$(date +%H%M); if [ "$hm" -ge 0505 ] && [ "$hm" -lt 1500 ]; then
    log "t3t: lock came at $hm, past 05:05: released, left for a day slot"; unlock; return 1; fi
  arm t3t q1 1500 "F12_OUT=$O F12_CMCL=47,32 F12_CAP=256 F12_T3ARMS=ex,g0,g0k2,t80c,t80ck2,t80csk2 F12_KEYS=humaneval/HumanEval/156,humaneval/HumanEval/42,humaneval/HumanEval/68,humaneval/HumanEval/91,math500/test/number_theory/459.json,math500/test/prealgebra/1640.json,ifeval/" $F/scripts/fit_t3.py 1700; r=$?
  unlock
  return $r
}

scr_t3r() {  # T3R (PLAN.md 'T3R', PROPOSED Tue 3 Nov 14:5x): P1-K1 / M3 re-test on the shared-text metric, 8 code prompts,
  # one process at C_m 47; quiet, MEM 12 GB; queue only with NewIdeas' OK and the orchestrator's slot
  O=$F/data/scr/t3r
  grep -q "] done" $O/r47.txt 2>/dev/null && return 0
  [ "$F12_ES_OK" = 1 ] || es_wait 120 || { log "t3r: EvalSuite decode still running after 2 h: left"; return 1; }
  w=0; while o=$(bigrun) && [ -n "$o" ] && [ $w -lt 90 ]; do
    [ $w -eq 0 ] && log "t3r: waiting, another session's run: $o"; w=$((w + 1)); sleep 60; done
  [ -n "$o" ] && { log "t3r: another session's run still listed after 90 min: re-queued"; requeue t3r; return 1; }
  memgate 16 || { log "t3r: NO-SWAP under 16 GB for 46 min: left"; return 1; }
  # (Tue 14:5x) the orchestrator's slot: Wed 00:30-00:45, first in line, BigMoE at 00:45: no lock request before 00:30
  # (23:00-00:29 waits), none used if granted after 00:36; the process is capped at 780 s (ends <= ~00:44)
  w=0; while hm=$(date +%H%M); [ "$hm" -lt 0030 ] || [ "$hm" -ge 2300 ]; do
    [ $w -eq 0 ] && log "t3r: waiting for the 00:30 slot"; w=1; sleep 20; done
  lock "Fit12 T3R: P1-K1 / 3-bit pass-1 re-test (shared-text metric) at 12 GB, MEM 12 GB, quiet, ~8 min, slot 00:30-00:45" 15m 600 || return 1
  # (Tue 15:0x, orchestrator) the GPU must be clear by 00:44: every process ends by DL = 00:43:30 (TIMEOUT kill + 5 s)
  hm=$(date +%H%M); DL=$(date -j -f "%H:%M:%S" "00:43:30" +%s); cr=$((DL - $(date +%s)))
  if [ "$hm" -gt 0036 ] || [ $cr -lt 480 ]; then log "t3r: lock came at $hm ($cr s to 00:43:30), too late for the slot: released"; unlock; return 1; fi
  [ $cr -gt 780 ] && cr=780
  K=humaneval/HumanEval
  arm t3r r47 420 "F12_OUT=$O F12_CAP=256 F12_PHLOG=1 F12_ALT=1 F12_K1U_KU=4 F12_KEYS=$K/156,$K/42,$K/68,$K/91,$K/16,$K/6,$K/79,$K/19 F12_T3ARMS=ex,t80ck2,t80ck2m3,t80ck1um3,t80ck1u F12_CMCL=47,32" $F/scripts/fit_t3.py $cr; r=$?
  # (Tue 15:0x) T3R-PF (PLAN.md): P2-PREFETCH, exact, a second ~3-min process, only if r47 ended rc 0 by 00:38
  hm=$(date +%H%M); st=$(ssd_t); cp=$((DL - $(date +%s)))   # arm() would wait up to 5 min for SSD <= 66 C: start
  if [ $r -eq 0 ] && [ "$hm" -le 0038 ] && [ $cp -ge 250 ] && [ -n "$st" ] && [ "$st" -le 62 ] && [ ! -e "$PAUSEF" ]; then   # p47 only at <= 62 C
    arm t3r p47 240 "F12_OUT=$O F12_CAP=256 F12_PHLOG=1 F12_ALT=1 F12_K1U_KU=4 F12_PF2_B=6 F12_KEYS=$K/156,$K/42,$K/68,$K/91 F12_T3ARMS=t80ck2,t80ck2pf,t80ck1u,t80ck1upf F12_CMCL=47,32 BIGMOE_CALLLOG=$O/p47_calls.csv" $F/scripts/fit_t3.py 240
    log "t3r: p47 (P2-PREFETCH) rc $?"
  else log "t3r: p47 (P2-PREFETCH) skipped: r47 rc $r at $hm, SSD ${st:-?} C (needs rc 0 by 00:38, SSD <= 62 C, no PAUSE)"; fi
  unlock
  return $r
}

scr_t3p() {  # T3P + P1-K1 (Tue 3 Nov, PLAN.md 'T3P' + 'T3P amendment'): pass-1 per-layer trace + bytes per phase, C_m 47
  # vs 51, and NewIdeas' P1-K1 arm t80ck1u beside t80ck2 (orchestrator 10:3x: this lock, <= ~25 min in all); quiet, 2
  # processes. (10:3x) The session starts the runner when the slot is open (15:07, or earlier once MTP Head's chain ends
  # by ~14:40), so no clock rule here; the bigrun wait (one job over ~5 GB at a time) and NO-SWAP >= 16 GB stay.
  O=$F/data/scr/t3p
  grep -q "] done" $O/c51.txt 2>/dev/null && return 0
  # (14:4x) F12_ES_OK=1: the session saw EvalSuite's decode idle for the last 5+ min (a restart after a fix)
  [ "$F12_ES_OK" = 1 ] || es_wait 120 || { log "t3p: EvalSuite decode still running after 2 h: left"; return 1; }
  w=0; while o=$(bigrun) && [ -n "$o" ] && [ $w -lt 90 ]; do
    [ $w -eq 0 ] && log "t3p: waiting, another session's run: $o"; w=$((w + 1)); sleep 60; done
  [ -n "$o" ] && { log "t3p: another session's run still listed after 90 min: re-queued"; requeue t3p; return 1; }
  memgate 16 || { log "t3p: NO-SWAP under 16 GB for 46 min: left"; return 1; }
  lock "Fit12 T3P + P1-K1: pass-1 per-layer trace + NewIdeas' P1-K1 timing at 12 GB, C_m 47 vs 51, MEM 12 GB, quiet, ~15 min" 25m 900 || return 1
  E0="F12_OUT=$O F12_CAP=256 F12_PHLOG=1 F12_ALT=1 F12_K1U_KU=4 F12_KEYS=humaneval/HumanEval/156,humaneval/HumanEval/42,humaneval/HumanEval/68,humaneval/HumanEval/91"
  arm t3p c47 420 "$E0 F12_T3ARMS=ex,t80ck2,t80ck2m3,t80ck1um3,t80ck1u F12_CMCL=47,32 BIGMOE_CALLLOG=$O/c47_calls.csv" $F/scripts/fit_t3.py 720; r=$?
  A=ex,t80ck2,t80ck2m3,t80ck1um3,t80ck1u
  case $r in
    0) ;;
    1) A=ex,t80ck2; log "t3p: c47 rc 1 (script error): c51 runs T3P's own arms ($A)";;
    *) unlock; return $r;;
  esac
  arm t3p c51 420 "$E0 F12_T3ARMS=$A F12_CMCL=51,32 BIGMOE_CALLLOG=$O/c51_calls.csv" $F/scripts/fit_t3.py 720; r2=$?
  unlock
  [ $r -eq 0 ] && return $r2
  return $r
}

scr_sgb1() {  # SGB step 1 (PLAN.md 'SGB'): v3p vs v3r + F12_SGB=1 hashes (8 prompts), then HT1's rows' memory on v3r
  O=$F/data/scr/sgb1
  grep -q "] done" $O/m.txt 2>/dev/null && return 0
  grep -q "^VERDICT PASS" $F/data/scr/sgb0/s0.txt 2>/dev/null || { log "sgb1: step 0 has no PASS: skipped"; return 0; }
  es_wait 120 || { log "sgb1: EvalSuite decode still running after 2 h: re-queued"; requeue sgb1; return 1; }
  w=0; while o=$(bigrun) && [ -n "$o" ] && [ $w -lt 90 ]; do
    [ $w -eq 0 ] && log "sgb1: waiting, another session's run (one > 5 GB job at a time by day): $o"; w=$((w + 1)); sleep 60; done
  [ -n "$o" ] && { log "sgb1: another session's run still listed after 90 min: re-queued"; requeue sgb1; return 1; }
  slock "Fit12 SGB step 1: balanced prefill groups on the model, hashes + memory, MEM 12 GB, ~10 min (shared)" 20m 600 0000 21 || return 1
  E0="F12_BT=$F/pinned/8db7/bt_b12.py F12_LIVE=1 F12_PROMPTS=m10,m13,m3,m15,h156,h45,h42,h8"
  arm sgb1 a 300 "$E0 F12_V3F=$HOME/Breakthrough/v3p" $F/scripts/fit_scr53.py 420; r=$?
  [ $r -eq 0 ] && { arm sgb1 b 300 "$E0 F12_V3F=$F/v3r F12_SGB=1" $F/scripts/fit_scr53.py 420; r=$?; }
  [ $r -eq 0 ] && { arm sgb1 m 300 "F12_ARM=live F12_BT=$F/pinned/8db7/bt_b12.py F12_OUT=$O F12_SET=humaneval F12_REFARM=b12llff4 F12_KEYS=HumanEval/68,HumanEval/156,HumanEval/45,HumanEval/42,HumanEval/8 F12_MEMTRACE=1 F12_V3F=$F/v3r F12_SGB=1" $F/scripts/fit_creep.py 900; r=$?; }
  unlock
  return $r
}

scr_sg1b() {  # SG12 step 1b (PLAN.md 'SG12'): M-5d's 16-row math lifetime peak on v3q + F12_SG12=1 (pinned 8db71cc537), one
  # process, memory only (shared lock); runs only if step 1a's arms have equal kv / h / logits / sha32 on all 4 prompts
  O=$F/data/scr/sg1b
  grep -q "] done" $O/math.txt 2>/dev/null && return 0
  if ! grep -q " RESULT kv" $F/data/scr/sg1/a.txt 2>/dev/null && grep -qx sg1 "$F/data/queue.txt" 2>/dev/null; then  # (19:3x) sg1 timed out
    { echo sg1; echo sg1b; grep -vx -e sg1 -e sg1b "$F/data/queue.txt"; } > "$F/data/queue.tmp" && mv "$F/data/queue.tmp" "$F/data/queue.txt"
    log "sg1b: step 1a has not run yet (sg1 re-queued): queue reordered, sg1 and sg1b first"; return 1
  fi
  ha=$(grep -h " RESULT kv" $F/data/scr/sg1/a.txt 2>/dev/null | awk '{print $2, $5, $7, $9, $11}')
  hb=$(grep -h " RESULT kv" $F/data/scr/sg1/b.txt 2>/dev/null | awk '{print $2, $5, $7, $9, $11}')
  na=$(printf "%s\n" "$ha" | grep -c .)
  if [ "$na" -ne 4 ] || [ "$ha" != "$hb" ]; then log "sg1b: step 1a has not 4 equal prompts (a: $na lines): skipped"; return 0; fi
  es_wait 300 || { log "sg1b: EvalSuite decode still running after 5 h: re-queued"; requeue sg1b; return 1; }
  w=0; while b=$(bigproc) && [ -n "$b" ] && [ $w -lt 60 ]; do [ $w -eq 0 ] && log "sg1b: waiting, big process running: $b"; w=$((w + 1)); sleep 60; done
  [ -n "$(bigproc)" ] && { log "sg1b: a big process still runs after 60 min: re-queued"; requeue sg1b; return 1; }
  slock "Fit12 SG12 step 1b: 16 math rows on v3q + SG12 (short-prompt grouped prefill), lifetime peak vs M-5d, memory only, ~20 min" 30m 1300 0000 21 || return 1
  arm sg1b math 1300 "F12_ARM=live F12_BT=$F/pinned/8db7/bt_b12.py F12_OUT=$O F12_SET=math F12_NROWS=16 F12_V3F=$F/v3q F12_SG12=1" $F/scripts/fit_creep.py 1800; r=$?
  unlock
  b=$(bigproc); [ -n "$b" ] && log "sg1b: big process seen at the end: $b (check the arm before using it)"
  return $r
}

scr_t3rn() {  # T3R NOW (Tue 3 Nov 19:0x; the orchestrator relaying Kamen: small breakthrough tests run now, after MoonShots'
  # M-28 and Leap's WC2 in the quiet FIFO; PLAN.md 'T3R NOW'): scr_t3r's two processes without the 00:30 slot timing, same
  # arms / prompts / lines; output data/scr/t3rn (the slot's data/scr/t3r stays clean). Gates: NO-SWAP >= 16 GB before the
  # request and again after the grant + 30 s settle; GPU <= 20 % at the grant (Kamen's apps open; logged); swap +2 GB guard
  # (arm); r47 cap 780 s; p47 only on r47 rc 0, SSD <= 62 C, no PAUSE, cap 240 s. A failed gate leaves the 00:30 slot armed.
  O=$F/data/scr/t3rn
  grep -q "] done" $O/r47.txt 2>/dev/null && return 0
  [ "$F12_ES_OK" = 1 ] || es_wait 120 || { log "t3rn: EvalSuite decode still running after 2 h: left (00:30 slot stays armed)"; return 1; }
  w=0; while o=$(bigrun) && [ -n "$o" ] && [ $w -lt 30 ]; do
    [ $w -eq 0 ] && log "t3rn: waiting, another session's run: $o"; w=$((w + 1)); sleep 60; done
  [ -n "$o" ] && { log "t3rn: another session's run still listed after 30 min: left (00:30 slot stays armed)"; return 1; }
  memgate 16 3 || { log "t3rn: NO-SWAP $(avail_gb) GB < 16 GB for 6 min: left (00:30 slot stays armed)"; return 1; }
  lock "Fit12 T3R now: P1-K1 / 3-bit pass-1 re-test + P2-PREFETCH at 12 GB, MEM 12 GB, quiet, ~11 min (after M-28, WC2)" 20m 900 20 \
    || { log "t3rn: no lock: left (00:30 slot stays armed)"; return 1; }
  sleep 30; i=0; while a=$(avail_gb); [ "$a" -lt 16 ] && [ $i -lt 6 ]; do sleep 10; i=$((i + 1)); done
  if [ "$a" -lt 16 ]; then log "t3rn: NO-SWAP $a GB < 16 GB after the grant: released (00:30 slot stays armed)"; unlock; return 1; fi
  log "t3rn: NO-SWAP $a GB after the grant (+30 s), GPU $(gpu) %"
  K=humaneval/HumanEval
  arm t3rn r47 420 "F12_OUT=$O F12_CAP=256 F12_PHLOG=1 F12_ALT=1 F12_K1U_KU=4 F12_KEYS=$K/156,$K/42,$K/68,$K/91,$K/16,$K/6,$K/79,$K/19 F12_T3ARMS=ex,t80ck2,t80ck2m3,t80ck1um3,t80ck1u F12_CMCL=47,32" $F/scripts/fit_t3.py 780; r=$?
  st=$(ssd_t)
  if [ $r -eq 0 ] && [ -n "$st" ] && [ "$st" -le 62 ] && [ ! -e "$PAUSEF" ]; then
    arm t3rn p47 240 "F12_OUT=$O F12_CAP=256 F12_PHLOG=1 F12_ALT=1 F12_K1U_KU=4 F12_PF2_B=6 F12_KEYS=$K/156,$K/42,$K/68,$K/91 F12_T3ARMS=t80ck2,t80ck2pf,t80ck1u,t80ck1upf F12_CMCL=47,32 BIGMOE_CALLLOG=$O/p47_calls.csv" $F/scripts/fit_t3.py 240
    log "t3rn: p47 (P2-PREFETCH) rc $?"
  else log "t3rn: p47 (P2-PREFETCH) skipped: r47 rc $r, SSD ${st:-?} C (needs rc 0, SSD <= 62 C, no PAUSE)"; fi
  unlock
  return $r
}

scr_t3r2() {  # T3R2 (NewIdeas' pre-registration, QUEUE 4dbe21a; orchestrator Tue 19:2x: Wed ~15:00, first in line, or right after
  # Breakthrough's map arms if they did not run tonight; T0 follows; PLAN.md 'T3R2'): t80ck1u vs t80ck2 (+ t80ck1um3 INFO) on
  # T3R's 8 HumanEval prompts + the next 8 by index (161 17 56 141 101 63 114 77), pooled shared-text verdict; then T3R-PF
  # exactly as registered (fd081ae) on its gates. Runner rules: start only at SSD <= 55 C and no PAUSE (<= 8 min under the
  # lock); a PAUSE inside a prompt's arms voids them, re-run once at the end (fit_t3.py F12_RERUN); the SSD per arm is logged.
  # MOVED (orchestrator Tue 22:3x): Wed 13:05 (FLOOR's deadline 13:00), quiet, MEM 12 GB, HARD STOP 13:40 (Kamen on the laptop
  # from ~13:45). Everything here ends by T2DL 13:39:00: the lock is requested only with >= 15 min to it, r47 starts only with
  # >= 10 min (it takes ~7-9), p47 only with >= 260 s; each cap is cut to T2DL - 20 s. at_screen.sh's STOP 13:30 / BACK 13:39:30
  # are the backstops (a queued ticket, a PAUSE or a yield that outlasts the slot).
  O=${F12_T3R2_O:-$F/data/scr/t3r2}   # (F12_T3R2_O: the dry test's scratch dir)
  grep -q "] done" $O/r47.txt 2>/dev/null && return 0
  T2DL=$(date -j -f %Y%m%d%H%M%S "${F12_T3R2_DL:-20261104133900}" +%s)
  t2dl() { echo $((T2DL - $(date +%s))); }
  es_wait 30 || { log "t3r2: EvalSuite decode still running after 30 min: left"; return 1; }
  while [ "$(date +%H%M%S)" -lt "${F12_T3R2_AT:-130500}" ]; do sleep 10; done   # FLOOR's deadline 13:00: the GPU is free at 13:05
  w=0; while o=$(bigrun) && [ -n "$o" ] && [ $w -lt 30 ] && [ "$(t2dl)" -ge 900 ]; do
    [ $w -eq 0 ] && log "t3r2: waiting, another session's run: $o"; w=$((w + 1)); sleep 60; done
  [ -n "$o" ] && { log "t3r2: another session's run still listed ($w min): left"; return 1; }
  memgate 16 3 || { log "t3r2: NO-SWAP $(avail_gb) GB < 16 GB for 6 min: left"; return 1; }
  [ "$(t2dl)" -lt 900 ] && { log "t3r2: $(t2dl) s left to the 13:39 deadline before the lock request (< 900): left"; return 1; }
  lock "Fit12 T3R2: P1-K1 timing re-test on 16 code prompts + P2-PREFETCH at 12 GB, MEM 12 GB, quiet, ~20 min (orchestrator: Wed 13:05, hard stop 13:40)" 30m 900 10 \
    || { log "t3r2: no lock: left"; return 1; }
  sleep 30; i=0; while a=$(avail_gb); [ "$a" -lt 16 ] && [ $i -lt 6 ]; do sleep 10; i=$((i + 1)); done
  if [ "$a" -lt 16 ]; then log "t3r2: NO-SWAP $a GB < 16 GB after the grant: released"; unlock; return 1; fi
  i=0; while st=$(ssd_t); { [ -z "$st" ] || [ "$st" -gt 55 ] || [ -e "$PAUSEF" ]; } && [ $i -lt 48 ] && [ "$(t2dl)" -ge 610 ]; do
    sleep 10; i=$((i + 1)); [ $((i % 6)) -eq 0 ] && keepalive; done
  if [ -z "$st" ] || [ "$st" -gt 55 ] || [ -e "$PAUSEF" ]; then
    log "t3r2: SSD ${st:-?} C / PAUSE $([ -e "$PAUSEF" ] && echo on || echo off) after ${i}0 s under the lock ($(t2dl) s to the deadline): released (start needs <= 55 C, no PAUSE)"; unlock; return 1; fi
  if [ "$(t2dl)" -lt 600 ]; then log "t3r2: $(t2dl) s left to the 13:39 deadline (< 600 for r47): released"; unlock; return 1; fi
  log "t3r2: start conditions: NO-SWAP $a GB, SSD $st C, no PAUSE, GPU $(gpu) % (waited ${i}0 s; $(t2dl) s to the deadline)"
  K=humaneval/HumanEval
  c=$(( $(t2dl) - 20 )); [ $c -gt 960 ] && c=960
  arm t3r2 r47 420 "F12_OUT=$O F12_CAP=256 F12_PHLOG=1 F12_ALT=1 F12_RERUN=1 F12_K1U_KU=4 F12_KEYS=$K/156,$K/42,$K/68,$K/91,$K/16,$K/6,$K/79,$K/19,$K/161,$K/17,$K/56,$K/141,$K/101,$K/63,$K/114,$K/77 F12_T3ARMS=t80ck2,t80ck1um3,t80ck1u F12_CMCL=47,32" $F/scripts/fit_t3.py $c; r=$?
  # T3R-PF exactly as registered: only on r47 rc 0, then SSD <= 62 C and no PAUSE (waits <= 3 min), cap 240 s; a PAUSE inside voids its timings
  i=0; while st=$(ssd_t); { [ -z "$st" ] || [ "$st" -gt 62 ] || [ -e "$PAUSEF" ]; } && [ $r -eq 0 ] && [ $i -lt 18 ] && [ "$(t2dl)" -ge 260 ]; do
    sleep 10; i=$((i + 1)); [ $((i % 6)) -eq 0 ] && keepalive; done
  if [ $r -eq 0 ] && [ -n "$st" ] && [ "$st" -le 62 ] && [ ! -e "$PAUSEF" ] && [ "$(t2dl)" -ge 260 ]; then
    c=$(( $(t2dl) - 20 )); [ $c -gt 240 ] && c=240
    arm t3r2 p47 240 "F12_OUT=$O F12_CAP=256 F12_PHLOG=1 F12_ALT=1 F12_K1U_KU=4 F12_PF2_B=6 F12_KEYS=$K/156,$K/42,$K/68,$K/91 F12_T3ARMS=t80ck2,t80ck2pf,t80ck1u,t80ck1upf F12_CMCL=47,32 BIGMOE_CALLLOG=$O/p47_calls.csv" $F/scripts/fit_t3.py $c
    log "t3r2: p47 (T3R-PF) rc $?"
  else log "t3r2: p47 (T3R-PF) skipped: r47 rc $r, SSD ${st:-?} C, PAUSE $([ -e "$PAUSEF" ] && echo on || echo off), $(t2dl) s to the deadline (needs rc 0, <= 62 C, no PAUSE, >= 260 s)"; fi
  # K12 (orchestrator 22:5x; PLAN.md 'T3R2' [K] CHECK): Leap's [K] A/B, bitwise only: T3R2's first 4 prompts, arms and order
  # again with mx.eval at every MoE block's entry and exit (F12_K12SYNC=1), here if >= 6 min remain (r47's main pass reached
  # 156, SSD <= 66 C, no PAUSE), cap <= 330 s, the deadline - 20 s and the lock - 15 s; else scr_k12 at 13:45 (shared lock).
  st=$(ssd_t); lf=$(left)
  if grep -q 'PAIR humaneval/HumanEval/156: ' $O/r47.txt 2>/dev/null && ! grep -q '] done' $O/k12.txt 2>/dev/null \
     && [ "$(t2dl)" -ge 360 ] && [ "${lf:-0}" -ge 315 ] && [ -n "$st" ] && [ "$st" -le 66 ] && [ ! -e "$PAUSEF" ]; then
    c=$(( $(t2dl) - 20 )); [ $c -gt 330 ] && c=330; [ $c -gt $((lf - 15)) ] && c=$((lf - 15))
    arm t3r2 k12 300 "F12_OUT=$O F12_CAP=256 F12_PHLOG=1 F12_ALT=1 F12_K1U_KU=4 F12_K12SYNC=1 F12_KEYS=$K/156,$K/42,$K/68,$K/91 F12_T3ARMS=t80ck2,t80ck1um3,t80ck1u F12_CMCL=47,32" $F/scripts/fit_t3.py $c
    log "t3r2: k12 rc $?: $($PY -B $F/scripts/k12_verdict.py $O r47 k12 2>&1 | tail -1 | cut -c1-170)"
  else log "t3r2: k12 not run in the slot ($(t2dl) s to the deadline, lock ${lf:-?} s, SSD ${st:-?} C, PAUSE $([ -e "$PAUSEF" ] && echo on || echo off); needs >= 360 / >= 315 s, <= 66 C, no PAUSE, r47 past 156): scr_k12 at 13:45"; fi
  unlock
  return $r
}

scr_k12() {  # K12 alone (orchestrator 22:5x: first thing in Kamen's day when it did not fit in the 13:05 slot; ~5 min, any
  # lock): the same run as in scr_t3r2 under a shared lock (bitwise only: timings do not matter), NO-SWAP >= 20 GB (peak + 8)
  O=${F12_T3R2_O:-$F/data/scr/t3r2}
  grep -q "] done" $O/k12.txt 2>/dev/null && return 0
  grep -q 'PAIR humaneval/HumanEval/156: ' $O/r47.txt 2>/dev/null || { log "k12: T3R2 has no main-pass data for 156 to compare with: left"; return 1; }
  slock "Fit12 K12: [K] stale-slot A/B on 4 code prompts, bitwise only, MEM 12 GB, ~5 min (shared; orchestrator 22:5x)" 12m 420 0000 20 || return 1
  K=humaneval/HumanEval
  arm t3r2 k12 300 "F12_OUT=$O F12_CAP=256 F12_PHLOG=1 F12_ALT=1 F12_K1U_KU=4 F12_K12SYNC=1 F12_KEYS=$K/156,$K/42,$K/68,$K/91 F12_T3ARMS=t80ck2,t80ck1um3,t80ck1u F12_CMCL=47,32" $F/scripts/fit_t3.py 480; r=$?
  log "k12: rc $r: $($PY -B $F/scripts/k12_verdict.py $O r47 k12 2>&1 | tail -1 | cut -c1-170)"
  unlock
  return $r
}
