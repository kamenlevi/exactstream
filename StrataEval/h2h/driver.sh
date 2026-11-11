#!/bin/bash
# driver.sh - the Strata head-to-head (Sat 7 Nov 23:5x, for the orchestrator; Kamen OK'd Strata + MTP). Normally started by
# waiter.sh under its quiet lock; H2H_DRY=1 H2H_EXE=<fake engine> H2H_LIMIT=2 runs it without the model (no lock, no pack).
#
# Arms, one warm engine process per arm and pass (run_arm.py: a discarded warm-up, then the 48 measure prompts of
# prompts/manifest.json in order: 12 HumanEval + 12 MBPP (S5's), 12 IFEval + 12 math (N45's)), passes in palindrome order:
#   A  Strata's default: the Q8_0 hyper-connection read (lossy; MAC.md: mean KL 0.0007 vs BF16), no MTP: --spec 2 --suffix-draft 0
#   B  A + the MTP draft layer: --mtp ~/StrataEval/Strata-data/mtp/rt (mtp-q2_0.gguf beside it)
#   C  exact: STRATA_HC_Q8=0 (the BF16 read), no MTP
#   D  C + MTP
# Common: setup's Metal arguments for this 64 GB Mac (strata_control/mac_fit.engine_args: the experts stream from the SSD,
# STRATA_METAL_STREAM=1, --expert-cache 64, --prefill auto, --max-context 32768, --mmap-experts), greedy, 256 new tokens, EOS
# stop, every prompt read from the start (--prompt-cache 0, --conversation-cache-mib 0, ckpt=0).
# Resumable: an arm+pass whose results/<tag>.done exists is skipped. Lock time: with H2H_LOCK_UNTIL (epoch) and H2H_LOCK_WHY
# from waiter.sh, a process that may not fit in the lock's time extends it first (`peer lock quiet <N>m`, refused while another
# session waits in line); if it cannot, the driver stops (exit 6) and waiter.sh re-queues and resumes.
# Exit: 0 all done; 3 swap guard; 4 paused (PAUSE / lid / soft_end); 6 lock time ran out; 7 prerequisites; 8 pack failed;
#       9 finished with some arm+pass incomplete.
set -u
H=/Users/kamen/StrataEval/h2h
SRC=/Users/kamen/StrataEval/src
D=/Users/kamen/StrataEval/Strata-data
M=/Users/kamen/StrataEval/models/IQ3_XXS
EXE=${H2H_EXE:-$SRC/build-metal/strata}
OUT=${H2H_OUT:-$H/results}
PY=${H2H_PY:-$SRC/.venv/bin/python}
PACK=${H2H_PACK:-$D/packs/iq3_xxs}
SH1=$M/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
SH2=$M/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf
RT=$D/mtp/rt
MTPG=$D/mtp/mtp-q2_0.gguf
MTP_SHA=a22206a4d74eafbf31063a84610292535fb3b49af9a339963817fd04aed81e4d
ARMS=${H2H_ARMS:-"A B C D D C B A"}
DRY=${H2H_DRY:-0}
LIMIT=${H2H_LIMIT:-0}
EST=${H2H_EST_S:-1200}                       # seconds one arm+pass may take (first guess; then 1.15 x the longest seen)
mkdir -p "$OUT"
LOG=$OUT/driver.log
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
swapmb() { sysctl -n vm.swapusage | awk '{gsub("M", "", $6); printf "%d", $6}'; }
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
lid_shut() {
  ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes || return 1
  (( $(system_profiler SPDisplaysDataType -detailLevel mini 2>/dev/null | grep -c "Resolution:") > 1 )) && return 1
  return 0
}
not_calm() {
  [ -e "$HOME/.local/share/quiet/PAUSE" ] && { echo "PAUSE file"; return 0; }
  ps -Ao command | grep -qE '^/bin/zsh [^ ]*soft_end\.sh' && { echo "soft_end.sh"; return 0; }
  lid_shut && { echo "lid closed, no external display"; return 0; }
  return 1
}

say "driver start (pid $$; arms: $ARMS; dry $DRY; limit $LIMIT; out $OUT; lock until ${H2H_LOCK_UNTIL:-none})"
SWAP0=$(swapmb)

# ---- prerequisites (skipped in a dry run)
if [ "$DRY" != 1 ]; then
  v=$("$EXE" --version 2>&1); [ "$v" = "strata 0.1.58-metal" ] || { say "STOP 7: $EXE --version says '$v'"; exit 7; }
  while read -r name size oid; do
    got=$(stat -f %z "$M/$name" 2>/dev/null || echo 0)
    [ "$got" = "$size" ] || { say "STOP 7: $name is $got B, not $size"; exit 7; }
  done < "$M/files.txt"
  case "$ARMS" in *B*|*D*)
    [ "$(stat -f %z "$MTPG" 2>/dev/null)" = 889017216 ] || { say "STOP 7: $MTPG missing or not 889017216 B"; exit 7; }
    [ "$(shasum -a 256 "$MTPG" | cut -d' ' -f1)" = "$MTP_SHA" ] || { say "STOP 7: $MTPG sha256 is not the packed one"; exit 7; }
    [ -f "$RT/draft_vocab.bin" ] || { say "STOP 7: $RT/draft_vocab.bin missing"; exit 7; } ;;
  esac
  if ! [ -f "$PACK/native_experts.txt" ] || ! [ -f "$PACK/tokenizer/vocab.json" ]; then
    say "pack: building $PACK (tools/iq_pack.py on shard 1; setup's step 6, no --compat-bf16 for this family)"
    t0=$(date +%s)
    STRATA_GGUF_PY=$SRC/third_party/llama.cpp/gguf-py nice -n 10 "$PY" "$SRC/tools/iq_pack.py" --gguf "$SH1" --out "$PACK" \
      > "$OUT/pack.log" 2>&1
    rc=$?
    { [ $rc = 0 ] && [ -f "$PACK/native_experts.txt" ] && [ -f "$PACK/tokenizer/vocab.json" ]; } ||
      { say "STOP 8: the pack failed (rc $rc; $OUT/pack.log): $(tail -2 "$OUT/pack.log" | tr '\n' ' ')"; exit 8; }
    say "pack: done in $(( $(date +%s) - t0 )) s, $(du -sh "$PACK" | cut -f1)"
  fi
fi

BASE=(--pack "$PACK" --native "$SH1" --ple-gguf "$SH2" --expert-profile "$SRC/data/expert-profile.bin" --expert-cache 64
      --prefill auto --max-context 32768 --mmap-experts --spec 2 --suffix-draft 0 --prompt-cache 0 --conversation-cache-mib 0)
json_args() { "$PY" -I -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "$@"; }

# (macOS /bin/bash is 3.2: no associative arrays; an empty array is expanded as ${a[@]+"${a[@]}"} under set -u)
pass_at() { echo $ARMS | tr ' ' '\n' | head -n "$1" | grep -cx "$2"; }   # the pass of the arm at position $1 (1-based)
todo_left() {   # arm+pass runs still without a .done, from position $1 (1-based) on
  local i=0 n=0 a
  for a in $ARMS; do
    i=$((i + 1))
    [ $i -ge "$1" ] && [ ! -f "$OUT/$a$(pass_at $i "$a").done" ] && n=$((n + 1))
  done
  echo $n
}
idx=0
for arm in $ARMS; do
  idx=$((idx + 1)); pas=$(pass_at $idx "$arm"); tag=$arm$pas
  if [ -f "$OUT/$tag.done" ]; then say "$tag: done earlier, skipped"; continue; fi
  if why=$(not_calm); then say "STOP 4: $why before $tag (waiter.sh unlocks, waits and resumes)"; exit 4; fi
  s=$(swapmb); [ $(( s - SWAP0 )) -le 2048 ] || { say "STOP 3: swap grew $(( s - SWAP0 )) MB since the start"; exit 3; }
  if [ "$DRY" != 1 ] && [ -n "${H2H_LOCK_UNTIL:-}" ]; then
    left=$(( H2H_LOCK_UNTIL - $(date +%s) ))
    if [ "$left" -lt "$EST" ]; then
      need=$(( ( $(todo_left $idx) * EST ) / 60 + 10 )); [ "$need" -gt 180 ] && need=180
      if peer lock quiet "${need}m" "${H2H_LOCK_WHY:-Strata head-to-head}" >> "$LOG" 2>&1; then
        H2H_LOCK_UNTIL=$(( $(date +%s) + need * 60 )); export H2H_LOCK_UNTIL
        say "lock extended by ${need} min (until $(date -r "$H2H_LOCK_UNTIL" '+%H:%M')) before $tag"
      else
        say "STOP 6: $left s of the lock left, $tag may need $EST s, and the lock could not be extended (someone waits)"; exit 6
      fi
    fi
  fi
  if [ "$DRY" != 1 ]; then
    want=${H2H_WANT_AVAIL:-47.0}   # 11-08: lower for a run that sets its arena outright (STRATA_METAL_ARENA_GB)
    a0=$(avail_gb); w=0
    while awk -v a="$(avail_gb)" -v m="$want" 'BEGIN {exit !(a < m)}' && [ $w -lt 10 ]; do sleep 30; w=$((w + 1)); done
    a1=$(avail_gb)
    awk -v a="$a1" -v m="$want" 'BEGIN {exit !(a < m)}' && say "$tag: WARNING only $a1 GB available (wanted $want): Strata's arena will be smaller"
  else
    a0=$(avail_gb); a1=$a0
  fi
  case $arm in
    A) envs=(env -u STRATA_HC_Q8 STRATA_METAL_STREAM=1); extra=() ;;
    B) envs=(env -u STRATA_HC_Q8 STRATA_METAL_STREAM=1); extra=(--mtp "$RT") ;;
    C) envs=(env STRATA_HC_Q8=0 STRATA_METAL_STREAM=1); extra=() ;;
    D) envs=(env STRATA_HC_Q8=0 STRATA_METAL_STREAM=1); extra=(--mtp "$RT") ;;
    *) say "unknown arm $arm"; exit 7 ;;
  esac
  ARGS=$(json_args "${BASE[@]}" ${extra[@]+"${extra[@]}"})
  mt="no MTP"; [ ${#extra[@]} -gt 0 ] && mt="${extra[*]}"
  say "$tag: start ($arm; ${envs[*]:1}; $mt; available $a1 GB; swap $(swapmb) MB)"
  t0=$(date +%s)
  "${envs[@]}" "$PY" -I "$H/run_arm.py" --arm "$arm" --pass "$pas" --exe "$EXE" --manifest "${H2H_MANIFEST:-$H/prompts/manifest.json}" \
    --out "$OUT" --engine-args "$ARGS" --limit "$LIMIT" >> "$LOG" 2>&1
  rc=$?
  dt=$(( $(date +%s) - t0 ))
  say "$tag: rc $rc in $dt s"
  [ $(( dt * 115 / 100 )) -gt "$EST" ] && EST=$(( dt * 115 / 100 ))
  case $rc in
    0) ;;
    3) say "STOP 3: the swap guard stopped $tag"; exit 3 ;;
    4) say "STOP 4: paused during $tag (waiter.sh unlocks, waits and resumes)"; exit 4 ;;
    *) say "$tag: failed (rc $rc); one retry"
       "${envs[@]}" "$PY" -I "$H/run_arm.py" --arm "$arm" --pass "$pas" --exe "$EXE" --manifest "${H2H_MANIFEST:-$H/prompts/manifest.json}" \
         --out "$OUT" --engine-args "$ARGS" --limit "$LIMIT" >> "$LOG" 2>&1
       rc=$?; say "$tag: retry rc $rc"
       case $rc in 3) exit 3 ;; 4) exit 4 ;; esac ;;
  esac
done
n=0; for f in "$OUT"/*.done; do [ -e "$f" ] && n=$((n + 1)); done
want=$(echo $ARMS | wc -w | tr -d ' ')
if [ "$n" -ge "$want" ]; then
  date '+%Y-%m-%d %H:%M:%S' > "$OUT/DRIVER_DONE"
  say "H2H DRIVER DONE: $n / $want arm+pass runs"
  exit 0
fi
say "H2H DRIVER INCOMPLETE: $n / $want arm+pass runs done"
exit 9
