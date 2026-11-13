#!/bin/bash
# EvalSuite runner 63 (attached to a live evalsuite session): the 45 GB P2 same-scores gate (docs/PREREG_p2gate45_2026-11-13.md),
# runner 62 (the 24 GB P2 gate) with switches for a same-night control arm. Runs one arm per call:
#   CARD_P2=off: the control arm (chat45 as shipped): the 217 keys of tools/p2_keys.json, then the 5 controls in their own
#     process, all in results/card_<tag>_fwd; P2 must be off on every kept run (runs.json p2 empty); no verdict ->
#     results/CARD_<tag>_DONE "CONTROL ...".
#   Both arms run the 5-key control process with R57_WARM=0 (the warm-up prompt is <= 256 tokens: it would go through P2).
#   CARD_P2=on (default): the P2 arm, as runner 62, scored vs CARD_REF (the control arm's run); the 5 controls in
#     card_<tag>_ctl must keep P2 off (p2 = 0), and with CARD_CTL_IDENT=1 equal the reference's tokens (else FAIL).
# CARD_FBLOB = the registered git blob of CARD_FENV (checked at the start; the env files predate the registration, so
# the blob replaces runner 62's written-after test). CARD_MEM overrides the env comment's MEM (45: 49 GB NO-SWAP per
# process). CARD_RESUME=1 continues; CARD_CUTOFF = the window's end (YYYY-mm-dd HH:MM); dry run CARD_DRY=1.
# Stop it by pid only, and its children too; then peer unlock.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
TAG=${CARD_TAG:?set CARD_TAG (the arm tag named in the env file comment, or ref for the reference stage)}
LOG=results/day63_${TAG}_runner.log
AF=results/card
mkdir -p "$AF"
DONE=results/CARD_${TAG}_DONE
P2MODE=${CARD_P2:-on}; case "$P2MODE" in on|off) ;; *) echo "CARD_P2 is on or off"; exit 2;; esac
RUNF=card_${TAG}_fwd; RUNC=card_${TAG}_ctl; REFRUN=${CARD_REF:-card_w24_fwd}   # the reference run (PREREG_p2gate45)
[ "$P2MODE" = off ] && RUNC=$RUNF   # the control arm keeps its controls in its one run (the P2 arm's reference)
STAGE=${CARD_STAGE:-arm}; PERPROC=20; CAP=2048; NK=217; NC=5
KF=$AF/p2_keys.txt; CF=$AF/p2_ctl_keys.txt   # tools/p2_keys.json: the gate keys and the controls, one per line
EXT=""   # runner 62: no extension (CARD_EXTEND is runner 61's)
XF=$AF/card_ext_keys.txt; RKF=$AF/card_ref_keys.txt; RNK=$NK   # the extension keys; the reference's keys (gate + extension)
EARLIEST=0   # runner 63: the env files predate the registration; CARD_FBLOB pins them instead
CUTOFF=$(date -j -f "%Y-%m-%d %H:%M:%S" "${CARD_CUTOFF:?set CARD_CUTOFF to the end of the GPU window, YYYY-mm-dd HH:MM}:00" +%s) \
  || { echo "CARD_CUTOFF does not parse as YYYY-mm-dd HH:MM"; exit 2; }
export EVALSUITE_DEADLINE=$((CUTOFF + 1800))
export PEER_QUEUE_TIMEOUT=21600
R3IDX=$HOME/BigMoE/models/bare/records3/index.json
BM=$HOME/BigMoE
FENV=${CARD_FENV:-}   # the arm stage needs it (checked at the start); never an apostrophe inside ${} text
DRY=${CARD_DRY:-0}
[ "$DRY" = 1 ] && LOG=results/day63_dry_${TAG}.log
UNSETS=(-u EVALSUITE_BIGMOE_CODE -u EVALSUITE_RELAX_VARIANT -u EVALSUITE_STAGE_ENV -u R57_KEYS -u R57_TAG -u BIGMOE_TIER
  -u BIGMOE_TIER_GLU -u BIGMOE_TIER_CL -u BIGMOE_TIER_COMBINE -u BIGMOE_OOO -u BIGMOE_MAXK)
STAGEENV="BIGMOE_MAXK=160,BIGMOE_TIER=4433333333,BIGMOE_TIER_GLU=3c,BIGMOE_TIER_CL=112,BIGMOE_TIER_COMBINE=1,BIGMOE_OOO=1"
PY=.venv/bin/python
EST_TF=2700
MODE=""; FZ=""; SNAPP=(); QS=""
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
void() { say "$1: VOID: runner stops"; echo "CARD $TAG: VOID ($1)" > "$DONE"; exit 5; }
lid_shut() {
  ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes || return 1
  (( $(system_profiler SPDisplaysDataType -detailLevel mini 2>/dev/null | grep -c "Resolution:") > 1 )) && return 1
  return 0
}
LASTWAIT=0
waitlog() { [ $(( $(date +%s) - LASTWAIT )) -ge 600 ] && { say "$*"; LASTWAIT=$(date +%s); }; }
QPAUSE=${CARD_PAUSE_FILE:-$HOME/.local/share/quiet/PAUSE}   # CARD_PAUSE_FILE: tests only
calm() {
  while pgrep -f '^/bin/zsh [^ ]*soft_end\.sh' >/dev/null || [ -e "$QPAUSE" ] || lid_shut; do
    waitlog "waiting: soft_end.sh, PAUSE or a closed lid"
    sleep 30
  done
}
full_converted() {   # every one of the 48 layers carries "layout", and index.json has not changed for 2 min
  [ $(( $(date +%s) - $(stat -f %m "$R3IDX") )) -ge 120 ] || return 1
  python3 -c 'import json, sys
L = json.load(open(sys.argv[1]))["layers"]
sys.exit(0 if len(L) == 48 and all("layout" in v for v in L.values()) else 1)' "$R3IDX"
}
store_ok() { local i; for i in 1 2 3 4; do full_converted && return 0; sleep 60; done; return 1; }
mem_busy() {    # another session's shared run with MEM > 5 GB (our own reasons start "evalsuite: MEM"); amendment 12b:
  # with CARD_YIELD_GB=12 (the w24 arm) only a run above 12 GB or a second run above 5 GB (one <= 12 GB neighbour may run)
  peer ls 2>/dev/null | grep "shared run" | grep -v "evalsuite: MEM" | grep -oE "MEM [0-9]+ GB" \
    | awk -v y="${CARD_YIELD_GB:-5}" '$2 > 5 {n++} $2 > y {f = 1} END {exit !(f || n > 1)}'
}
freegb() { .venv/bin/python -c "import sys; sys.path.insert(0, '.'); from evalsuite.cli import free_gb; print(int(free_gb()))"; }
swapgb() { sysctl -n vm.swapusage | awk '{gsub("M", "", $6); print int($6 / 1024 + 0.5)}'; }
SWAP0=$(swapgb)
rebase_swap() { # after waiting for another session (our process had ended): swap grown meanwhile is theirs (RISK 1, 11-09)
  local s; s=$(swapgb); [ "$s" = "$SWAP0" ] || say "swap base $SWAP0 -> $s GB (re-read after $1)"; SWAP0=$s
}
alive() { [ -f "results/q$1.pid" ] && kill -0 "$(cat "results/q$1.pid")" 2>/dev/null; }
fz_ok() {       # every file of the freeze still has the sha FREEZE.json records
  python3 - "$FZ" <<'FZ'
import hashlib, json, sys
fz = sys.argv[1]
F = json.load(open(f"{fz}/FREEZE.json"))["files"]
sys.exit(1 if any(hashlib.sha256(open(f"{fz}/{p}", "rb").read()).hexdigest()[:12] != s for p, s in F.items()) else 0)
FZ
}
bm_pin() {      # snapshot: BigMoE's bigmoe/ and the snapshot's paths == $BMC, nothing untracked there, the .so; freeze: FZ intact
  [ "$MODE" = freeze ] && { fz_ok; return; }
  git -C "$BM" diff --quiet "$BMC" -- bigmoe "${SNAPP[@]}" 2>/dev/null || return 1
  [ -z "$(git -C "$BM" status --porcelain -- bigmoe "${SNAPP[@]}" 2>/dev/null)" ] || return 1
  [ "$(shasum -a 256 "$BM"/ext/"$EB"/*.so | cut -c1-12)" = "$QOMSO" ]
}
bm_loaded() {   # a process's manifest. snapshot: every module it loaded from ~/BigMoE (data/r57/snap too) equals its file at
                # $BMC. freeze: every module it loaded from the freeze has FREEZE.json's sha. Live outside files: a NOTE
                # (the manifest pins them across processes)
  python3 - "$1" "$BM" "$BMC" "$MODE" "$FZ" <<'MAN'
import hashlib, json, os, subprocess, sys
mods, bm, c, mode, fz = json.load(open(sys.argv[1]))["modules"], *sys.argv[2:6]
H = os.path.expanduser("~")
F = json.load(open(f"{fz}/FREEZE.json"))["files"] if mode == "freeze" else {}
bad, n, live = [], 0, []
for name, (f, sha) in sorted(mods.items()):
    if mode == "freeze" and f.startswith(fz + "/"):
        n += 1
        if F.get(f[len(fz) + 1:]) != sha:
            bad.append(f[len(fz) + 1:])
    elif mode == "snapshot" and f.startswith(bm + "/") and not f.endswith(".so"):
        rel = f[len(bm) + 1:]
        r = subprocess.run(["git", "-C", bm, "show", f"{c}:{rel}"], capture_output=True)
        n += 1
        if r.returncode or hashlib.sha256(r.stdout).hexdigest()[:12] != sha:
            bad.append(rel)
    elif any(f.startswith(f"{H}/{t}/") for t in ("BigMoE", "Leap", "TrainLab", "Breakthrough")) and not f.endswith(".so"):
        live.append(f[len(H) + 1:])
ref = "the freeze" if mode == "freeze" else c
print(f"outside modules loaded ({mode}): {n}, equal to {ref}: {n - len(bad)}" + (f"; DIFFERENT or untracked: {bad}" if bad else "")
      + (f"; NOTE live outside files (pinned by the manifest): {live}" if live else ""))
sys.exit(1 if bad else 0)
MAN
}
pin() {
  { shasum snap/leap-faff8e5/*.py snap/leap-52cf2c9/*.py snap/leap-52cf2c9/ms_gdn/* tools/r57_wrap.py tools/r57rows.py \
      tools/d4x.py tools/mbpp_score.py tools/freeze_r57.py tools/rows_text.py tools/he_d4.py tools/set_d4.py "$SCREEN" results/drift/keys_T.txt data/items/dec_humanevalt.jsonl \
      data/items/dec_mbpp.jsonl data/items/dec_mgsm.jsonl data/items/dec_ifbench.jsonl data/items/dec_gpqad.jsonl data/items/dec_lcb6.jsonl tools/card_keys.json data/items/dec_multi.jsonl data/items/dec_math.jsonl data/items/dec_ifeval.jsonl data/items/dec_humaneval.jsonl
    (cd snap/bigmoe-da8e7a6 && find bigmoe ext -type f ! -path '*__pycache__*' | sort | xargs shasum)
    (cd "$HOME/MoonShots/patches/planes_engine" && ls *.py | sort | xargs shasum)
    find evalsuite -name '*.py' ! -path '*__pycache__*' | sort | xargs shasum
    ls "$HOME/BigMoE/.venv/lib/python3.12/site-packages" | grep -E '^(mlx|mlx_lm|mlx_metal|mtplx)-.*dist-info$'; } 2>&1 \
    | shasum | cut -c1-12
}
REFMAN=""        # the stage's first process manifest: every later process of the stage must load the same outside code
REFMAN1=""       # the first stage's: later stages must load the same shas on the modules they share
T_='("/BigMoE/", "/Leap/", "/TrainLab/", "/Breakthrough/", "/MoonShots/")'
ext_pin() {     # every outside module of the reference manifest still has its sha
  [ -n "$REFMAN" ] || return 0
  python3 - "$REFMAN" <<EXT
import hashlib, json, sys
m = json.load(open(sys.argv[1]))["modules"]
bad = [f for f, s in m.values() if any(t in f for t in $T_) and hashlib.sha256(open(f, "rb").read()).hexdigest()[:12] != s]
sys.exit(1 if bad else 0)
EXT
}
man_same() {    # manifest $2 vs reference $1: the same outside modules with the same shas ($3 = shared: the shared ones only)
  python3 - "$1" "$2" "${3:-all}" <<MS
import json, sys
ext = lambda p: {k: s for k, (f, s) in json.load(open(p))["modules"].items() if any(t in f for t in $T_)}
a, b = ext(sys.argv[1]), ext(sys.argv[2])
ks = set(a) & set(b) if sys.argv[3] == "shared" else set(b) if sys.argv[3] == "subset" else set(a) | set(b)   # subset: every module loaded here = the reference's
d = sorted(k for k in ks if a.get(k) != b.get(k))
print(f"outside modules: {len(b)}, compared {len(ks)} ({sys.argv[3]}), equal: {len(ks) - len(d)}" + (f"; DIFFERENT: {d}" if d else "")
      + (f"; only in this stage: {sorted(set(b) - set(a))}" if sys.argv[3] == "shared" and set(b) - set(a) else "")
      + (f"; not loaded here: {sorted(set(a) - set(b))}" if sys.argv[3] == "subset" and set(a) - set(b) else ""))
sys.exit(1 if d else 0)
MS
}
parse_env() {   # file -> ENVARR: its last non-comment line, comma-split; a token without '=' continues the previous value
  ENVARR=()       # (BigMoE Fri 11:0x: R57_QOM_SETS=humaneval,mbpp,math,ifeval,multi sits inside the comma list)
  while IFS= read -r -d '' x; do ENVARR+=("$x"); done < <(python3 - "$1" <<'PE'
import sys
line = [l for l in open(sys.argv[1]).read().splitlines() if l and not l.startswith("#")][-1]
out = []
for t in line.split(","):
    if "=" in t or not out:
        out.append(t)
    else:
        out[-1] += "," + t
if not all("=" in t for t in out):
    sys.exit(1)
sys.stdout.write("\0".join(out) + "\0")
PE
)
  [ "${#ENVARR[@]}" -gt 0 ]
}
gate() {        # label, MEM GB, seconds the step is estimated / measured to take: every check before a GPU step
  local w=0
  calm
  while mem_busy; do
    [ $((w % 10)) = 0 ] && say "$1: another session's MEM job holds a shared lock: waiting (day rule)"
    w=$((w + 1)); sleep 60; calm
  done
  [ "$w" -gt 0 ] && rebase_swap "a wait for another session's MEM job"
  [ "$(pin)" = "$PIN0" ] || { say "code changed: pin $PIN0 -> $(pin): runner stops"; exit 7; }
  [ "${BMPIN:-0}" = 1 ] && ! bm_pin && { say "BigMoE's code is not $BMC / the freeze ($MODE): runner stops"; exit 7; }
  [ "${BMPIN:-0}" = 1 ] && ! ext_pin && { say "a module the stack loads changed vs $REFMAN: runner stops"; exit 7; }
  store_ok || { say "records3 is not fully converted: runner stops"; exit 8; }
  [ "$(swapgb)" -le $(( SWAP0 + 2 )) ] || { say "swap $(swapgb) GB > $SWAP0 + 2 at the start: runner stops"; exit 3; }
  [ $(( $(date +%s) + $3 )) -le "$CUTOFF" ] || { say "$1: does not fit before the cutoff ($(( $3 / 60 )) min): runner stops"; exit 6; }
  w=0
  until [ "$(freegb)" -ge $(( $2 + 8 )) ]; do
    [ $((w % 10)) = 0 ] && say "$1: waiting for memory: $(freegb) GB NO-SWAP (needs $(( $2 + 8 )))"
    w=$((w + 1)); sleep 60; calm
    [ $(( $(date +%s) + $3 )) -le "$CUTOFF" ] || { say "$1: no memory before the cutoff: runner stops"; exit 6; }
  done
}
share_lock() {  # reason, lease minutes, resources (default the stack's): retry while a quiet lock runs
  local r w=0
  while :; do
    r=$(peer lock share "${2}m" ${3:---cpu 300 --gpu 60} "$1" 2>&1)
    if [ $? = 0 ]; then
      if [ -e "$QPAUSE" ]; then   # a PAUSE that began as we asked (amendment 6): release, wait it out, ask again
        peer unlock > /dev/null 2>&1; say "PAUSE right after the shared lock: released it"; calm; continue
      fi
      say "shared lock: $1"; [ "$w" -gt 0 ] && rebase_swap "a yield"; return 0
    fi
    echo "$r" | grep -q "quiet" || { say "share lock refused: $(echo "$r" | tail -1)"; }
    [ $((w % 10)) = 0 ] && say "share lock refused (a quiet lock runs?): waiting"
    w=$((w + 1)); sleep 60; calm
    [ "$w" -lt 240 ] || { say "no share lock in 4 h: runner stops"; exit 3; }
  done
}
tfwin() {       # run, label: tfwin chunks until every row of the run is scored
  local run=$1 b n rc t0 d np=0
  while :; do
    n=$(ls "results/$run/tfwin" 2>/dev/null | grep -c '\.json\.gz$')
    [ "$n" -ge "$(grep -c . "results/$run/decode.jsonl")" ] && return 0
    gate "tfwin $2" 34 "$EST_TF"
    t0=$(date +%s)
    EVALSUITE_STAGE_ENV="$STAGEENV" EVALSUITE_BIGMOE_CODE="$HOME/EvalSuite/snap/bigmoe-da8e7a6" EVALSUITE_RELAX_VARIANT=ref2tnaxn \
      ./bin/evalsuite gpu tfwin ref --run "$run" --self ref2tnax --chunk-minutes 45 --C 192 --need 34 --minutes 60 --share \
      >> "$LOG" 2>&1; rc=$?
    d=$(( $(date +%s) - t0 )); [ "$d" -gt "$EST_TF" ] && EST_TF=$d
    b=$n; n=$(ls "results/$run/tfwin" 2>/dev/null | grep -c '\.json\.gz$')
    say "tfwin $2: chunk rc=$rc ($n rows scored, $((d / 60)) min)"
    [ "$rc" = 5 ] && void "tfwin $2: the self-check"
    if [ "$n" -le "$b" ]; then
      tail -40 results/gpu_stage.log | grep -q "Traceback" && { say "tfwin $2: the stage failed (results/gpu_stage.log): runner stops"; exit 4; }
      np=$((np + 1)); [ "$np" -ge 3 ] && { say "tfwin $2: three chunks without progress: runner stops"; exit 3; }
      sleep 60
    fi
  done
}
fast() {        # tag, keys, label: one stack process (all its keys), under this runner's share lock
  local tag=$1 keys=$2 t0 d rc
  BMPIN=1 gate "CHAT $3" "$((MEM - 4))" "${STEP:-900}"     # needs MEM + 4 GB NO-SWAP free (BigMoE's own margin)
  rm -f "$BM/data/r57/${tag}_"*
  share_lock "evalsuite: MEM $MEM GB: $PHASE, $3" "$(( ${STEP:-900} / 30 ))"   # the lease: 2x the step (30 min at HARD 600)
  t0=$(date +%s)
  env "${UNSETS[@]}" "${STACKENV[@]}" "${EXTRAENV[@]}" R57_PRESET=45 R57_REPS=1 R57_WARM="${WARM:-1}" R57_TAG="$tag" R57_KEYS="$keys" \
    "$BM/.venv/bin/python" tools/r57_wrap.py "$SCREEN" "$AF/${tag}_modules.json" "$C" "$CAP" > "$AF/${tag}.log" 2>&1; rc=$?
  peer unlock > /dev/null 2>&1
  d=$(( $(date +%s) - t0 ))
  say "CHAT $3: rc=$rc ($((d / 60)) min)"
  grep -q "no free slot" "$AF/${tag}.log" && { say "CHAT $3: a 'no free slot' abort: INCOMPLETE: runner stops"; echo "CARD $TAG: INCOMPLETE (a 'no free slot' abort in ${tag})" > "$DONE"; exit 6; }
  cp "$BM/data/r57/${tag}_"* "$AF/" 2>/dev/null
  [ -s "$AF/${tag}_tokens.json" ] || { say "CHAT $3: no tokens (results/chat/${tag}.log)"; return 1; }
  bm_loaded "$AF/${tag}_modules.json" >> "$LOG" 2>&1 || void "CHAT $3: a loaded outside module is not $BMC's / the freeze's"
  if [ -z "$REFMAN" ]; then
    REFMAN="$AF/${tag}_modules.json"; say "reference manifest: $REFMAN"
    if [ -z "$REFMAN1" ]; then REFMAN1=$REFMAN
    else man_same "$REFMAN1" "$REFMAN" shared >> "$LOG" 2>&1 || void "CHAT $3: its shared modules differ from the first stage's ($REFMAN1)"; fi
  else man_same "$REFMAN" "$AF/${tag}_modules.json" "$([ "${WARM:-1}" = 0 ] && echo subset || echo all)" >> "$LOG" 2>&1 || void "CHAT $3: its modules differ from $REFMAN"; fi   # PREREG_p2gate45 1b: without the warm-up, a module only it loads (qmm32) is absent
  QSETS="$QS" python3 - "$AF/${tag}_runs.json" <<'ENG' >> "$LOG" 2>&1 || void "CHAT $3: QOM not engaged on its sets"
import json, os, sys
runs = json.load(open(sys.argv[1]))
S = {s for s in os.environ.get("QSETS", "").split(",") if s}     # the env's QOM sets (chat ones included for chat candidates)
cut = [r["key"] for r in runs if r.get("cut")]      # cut at the screen's time budget: not rows (r57rows leaves them out; Mon 11-09)
code = [r for r in runs if str(r.get("key", "")).split("/")[0] in S and not r.get("cut")]
qom = [r["key"] for r in code if (r.get("pf") or {}).get("qom_opt", 0) > 0]
print(f"engaged: QOM optional uses on {len(qom)} of {len(code)} runs in the QOM sets {sorted(S)}; types "
      f"{sorted({str(r.get('type')) for r in runs})}; inert {sum(1 for r in runs if r.get('inert'))}, relaxed "
      f"{sum(1 for r in runs if r.get('rlx'))}, span {sum(1 for r in runs if r.get('spn'))} runs"
      + (f"; cut at the time budget (not rows): {' '.join(cut)}" if cut else ""))
sys.exit(0 if len(qom) == len(code) else 1)
ENG
  P2MODE="$P2MODE" P2SHORT="$KF" P2CTL="$CF" python3 - "$AF/${tag}_runs.json" <<'P2E' >> "$LOG" 2>&1 || void "P2 $3: P2 not as registered (runs.json p2, mode $P2MODE)"
import json, os, sys
runs = json.load(open(sys.argv[1]))
short, ctl = set(open(os.environ["P2SHORT"]).read().split()), set(open(os.environ["P2CTL"]).read().split())
kept = [r for r in runs if not r.get("cut")]
other = [r["key"] for r in kept if r["key"] not in short | ctl]
if os.environ.get("P2MODE") == "off":                # the control arm: P2 off on every kept run
    lit = [r["key"] for r in kept if r.get("p2")]
    print(f"P2 off: {len(kept)} kept runs; p2 set on {lit}; outside {other}")
    sys.exit(1 if lit or other else 0)
on = [r["key"] for r in kept if r["key"] in short and (r.get("p2") or 0) > 0]
off = [r["key"] for r in kept if r["key"] in ctl and r.get("p2") == 0]
bad = [r["key"] for r in kept if (r["key"] in short and r["key"] not in on) or (r["key"] in ctl and r["key"] not in off)]
other = [r["key"] for r in kept if r["key"] not in short | ctl]
print(f"P2 engaged: p2 > 0 on {len(on)} short keys, p2 = 0 on {len(off)} controls, of {len(kept)} kept runs; wrong {bad}; outside {other}")
sys.exit(1 if bad or other else 0)
P2E
  return $rc
}
missing() {     # run, comma list -> the keys not yet in results/<run>/decode.jsonl
  python3 -c 'import json, os, sys
p = os.path.join("results", sys.argv[1], "decode.jsonl")
have = {json.loads(l)["key"] for l in open(p)} if os.path.exists(p) else set()
print(",".join(k for k in sys.argv[2].split(",") if k not in have))' "$1" "$2"
}
order() {       # run, label, keys (comma list): chunks of 10 per process; any missed keys go into the next process
  local run=$1 lab=$2 all=$3 i=0 m tag tries=0
  while :; do
    m=$(missing "$run" "$all"); [ -z "$m" ] && break
    i=$((i + 1)); tries=$((tries + 1)); [ "$tries" -le 30 ] || { say "$lab: 30 processes without all rows: runner stops"; exit 3; }
    tag="${TAGP}${lab}${i}"
    fast "$tag" "$(echo "$m" | tr ',' '\n' | head -"$PERPROC" | paste -sd, -)" "$lab process $i"
    [ -s "$AF/${tag}_tokens.json" ] && $PY tools/r57rows.py "$AF/${tag}_tokens.json" "$run" --append >> "$LOG" 2>&1
  done
  [ "$IFECOPY" = 1 ] && python3 - "$run" <<'IFE' >> "$LOG" 2>&1
import json, sys
run = sys.argv[1]
p = f"results/{run}/decode.jsonl"
have = {json.loads(l)["key"] for l in open(p)}
add = [r for r in map(json.loads, open("results/ref2tnax/decode.jsonl")) if r["set"] == "ifeval" and r["key"] not in have]
with open(p, "a") as f:
    for r in add:
        f.write(json.dumps(dict(r, source="NAX6's own row (chat stays EXACT; PREREG_s4)")) + "\n")
print(f"results/{run}: + {len(add)} IFEval rows from results/ref2tnax")
IFE
  $PY tools/rows_text.py "$run" >> "$LOG" 2>&1 || { say "$lab: rows_text.py failed: runner stops"; exit 4; }   # the scorers read text
  say "$lab: $(grep -c . "results/$run/decode.jsonl") rows"
}
count() { [ -f "$1" ] && grep -c . "$1" || echo 0; }
refcount() {    # keys of the file (default: the reference's keys) present in results/$REFRUN
  python3 - "$REFRUN" "${1:-$RKF}" <<'RC'
import json, os, sys
p = f"results/{sys.argv[1]}/decode.jsonl"
have = {json.loads(l)["key"] for l in open(p)} if os.path.exists(p) else set()
print(len(have & set(open(sys.argv[2]).read().split())))
RC
}
refdecode() {   # run, sets, keys file, C, need GB, max rows: one chunk of the reference decode stage (runner 53 / 60's env)
  ( export EVALSUITE_BIGMOE_CODE="$HOME/EvalSuite/snap/bigmoe-da8e7a6"
    unset BIGMOE_TIER BIGMOE_TIER_GLU BIGMOE_TIER_CL BIGMOE_TIER_COMBINE BIGMOE_OOO BIGMOE_EXT_BUILD BIGMOE_MAXK EVALSUITE_RELAX_VARIANT
    EVALSUITE_STAGE_ENV="BIGMOE_TIER=4433333333,BIGMOE_TIER_GLU=3c,BIGMOE_TIER_CL=112" ./bin/evalsuite gpu decode "$1" --sets "$2" \
      --keys "$3" --max-rows "$6" --chunk-minutes 45 --share --C "$4" --need "$5" --minutes 60 ) >> "$LOG" 2>&1
}
linecheck() {   # C, need: 12 IFEval rows at this C == ref2tnax's (made at C 192); the verdict is kept per C. The run is
  #               ref2tnaxn.r<C> (the stage takes scratch runs as <variant>.r<digits> only)
  local c=$1 ok=$AF/linecheck_C$1.ok bad=$AF/linecheck_C$1.bad lk=$AF/linecheck_keys.txt run=ref2tnaxn.r$1 rc
  [ -f "$ok" ] && return 0; [ -f "$bad" ] && return 1
  python3 -c 'import json; print("\n".join(sorted(json.loads(l)["key"] for l in open("data/items/dec_ifeval.jsonl"))[:12]))' > "$lk"
  say "line check: 12 IFEval rows at C $c -> results/$run (need $2 GB)"
  refdecode "$run" ifeval "$lk" "$c" "$2" 12; rc=$?; say "line check: chunk rc=$rc"
  if [ "$rc" != 0 ] && [ "$rc" != 75 ] && [ ! -s "results/$run/decode.jsonl" ]; then
    say "line check: the decode failed with no row (rc $rc; results/gpu_stage.log): runner stops"; exit 4
  fi
  python3 - "$run" "$lk" <<'LC' > "$AF/linecheck_C$c.txt" 2>&1
import json, sys
ks = open(sys.argv[2]).read().split()
a = {r["key"]: r["completion_ids"] for r in map(json.loads, open(f"results/{sys.argv[1]}/decode.jsonl"))}
b = {r["key"]: r["completion_ids"] for r in map(json.loads, open("results/ref2tnax/decode.jsonl"))}
same = [k for k in ks if k in a and a[k] == b[k]]
print(f"{len(same)} of {len(ks)} rows token-identical to ref2tnax ({sum(k in a for k in ks)} decoded)")
sys.exit(0 if len(same) == len(ks) else 1)
LC
  local rc=$?
  say "line check C $c: $(cat "$AF/linecheck_C$c.txt" | tail -1)"
  if [ "$rc" = 0 ]; then touch "$ok"; return 0; fi
  [ "$(grep -c . "results/$run/decode.jsonl" 2>/dev/null || echo 0)" -ge 12 ] && touch "$bad"   # a short chunk: retry later
  return 1
}
quiet_held() { peer ls 2>/dev/null | grep -q "benchmarking: quiet lock until"; }
cutoff_or_stop() { [ "$(date +%s)" -lt "$CUTOFF" ] || { say "reference: $1 by the cutoff: runner stops ($(refcount) of $RNK rows; rerun to resume)"; exit 6; }; }
yield_check() { # amendment 2: Google Chrome closed while Strata has not ended = Strata is about to take the Mac
  local t_s
  strata_done && return 0
  if [ "${CARD_WAIT_STRATA:-0}" = 1 ]; then   # amendment 3: Strata runs with Chrome open
    say "reference: CARD_WAIT_STRATA=1: yielding to Strata ($(refcount) of $RNK rows); no new chunk until WAITER END, PFB, no quiet lock"
  else
    pgrep -x "Google Chrome" > /dev/null && return 0
    say "reference: Google Chrome is not running: yielding to Strata ($(refcount) of $RNK rows); no new chunk until WAITER END, PFB, no quiet lock"
  fi
  until strata_done; do waitlog "reference: yielding: waiting for Strata's DRIVER_DONE / WAITER END: DONE"; sleep 60; cutoff_or_stop "Strata not done"; done
  t_s=$(date +%s); say "reference: Strata ended; waiting for pids ${CARD_AFTER_PIDS:-none} (PFB; at most ${CARD_AFTER_MAXWAIT:-60} min)"
  until pids_gone || [ $(( $(date +%s) - t_s )) -ge $(( ${CARD_AFTER_MAXWAIT:-60} * 60 )) ]; do sleep 60; cutoff_or_stop "PFB not done"; done
  sleep 120     # a job that starts on the same signal takes its lock first
  while quiet_held; do waitlog "reference: a quiet lock is held: waiting"; sleep 60; cutoff_or_stop "a quiet lock still held"; done
  SWAP0=$(swapgb)
  say "reference: resuming after Strata$(pids_gone || echo " (pids ${CARD_AFTER_PIDS} still alive after the cap)"); swap base now $SWAP0 GB"
}
refstage() {    # NAX6-N's AR on the gate keys -> results/$REFRUN (runner 60's stage 0); below C 192 only after its line check
  local est=2100 np=0 b t0 d rc fg c need
  [ "$(refcount)" -ge "$RNK" ] && { say "reference: $REFRUN has its $RNK rows"; return 0; }
  while [ "$(refcount)" -lt "$RNK" ]; do
    calm
    yield_check
    while quiet_held; do waitlog "reference: a quiet lock is held: waiting"; sleep 60; cutoff_or_stop "a quiet lock still held"; done
    while mem_busy; do waitlog "reference: waiting: another session's shared run (MEM > 5 GB)"; sleep 60; calm; done
    [ "$(swapgb)" -le $(( SWAP0 + 2 )) ] || { say "reference: swap $(swapgb) GB > $SWAP0 + 2 (the start): runner stops"; exit 3; }
    [ $(( $(date +%s) + est )) -le "$CUTOFF" ] || { say "reference: no chunk fits before the cutoff: runner stops ($(refcount) of $RNK rows; rerun to resume)"; exit 6; }
    fg=$(freegb); c=192; need=36; [ "$fg" -ge 40 ] || { c=160; need=32; }; [ "$fg" -ge 36 ] || { c=128; need=27; }
    if [ "$c" != 192 ] && ! linecheck "$c" "$need"; then
      waitlog "reference: C $c has no passed line check: waiting for 40 GB free (C 192)"; sleep 120; continue
    fi
    b=$(refcount); t0=$(date +%s)
    refdecode "$REFRUN" ifbench,gpqad,lcb6 "$RKF" "$c" "$need" "${CARD_CHUNK_ROWS:-10}"; rc=$?
    d=$(( $(date +%s) - t0 )); [ "$d" -gt "$est" ] && est=$d
    say "reference: chunk rc=$rc at C $c: $(refcount)/$RNK rows ($((d / 60)) min)"
    if [ "$(refcount)" -le "$b" ]; then
      np=$((np + 1)); [ "$np" -ge 3 ] && { say "reference: three chunks without a new row: runner stops"; exit 3; }; sleep 60
    else np=0; fi
  done
}
# ------------------------------------------------ start
mkdir -p "$AF"
python3 - "$KF" "$CF" <<'PK' || { say "tools/p2_keys.json does not build: runner stops"; exit 2; }
import json, sys
d = json.load(open("tools/p2_keys.json"))
ks = [k for v in d["keys"].values() for k in v]
assert len(ks) == len(set(ks)) == 217 and len(d["keys"]["ifbench"]) == 150 and len(d["keys"]["gpqad"]) == 67, len(ks)
assert len(d["control"]) == 5 and not set(d["control"]) & set(ks), d["control"]
open(sys.argv[1], "w").write("\n".join(ks) + "\n")
open(sys.argv[2], "w").write("\n".join(d["control"]) + "\n")
PK
grep . "$KF" "$CF" -h > "$RKF"; RNK=$(sort -u "$RKF" | grep -c .)
[ "$RNK" = $((NK + NC)) ] || { say "the gate keys and controls overlap or miss ($RNK): runner stops"; exit 2; }
[ "$P2MODE" = off ] || [ "$(refcount)" = "$RNK" ] || [ "${CARD_REF_LATER:-0}" = 1 ] || { say "the reference ($REFRUN) does not hold all $RNK keys ($(refcount)): runner stops"; exit 2; }
strata_done() { # amendment 4: a finished Strata only (DRIVER_DONE, or a last waiter line "WAITER END: DONE"), or the release file
  [ -e "$HOME/StrataEval/h2h/results/DRIVER_DONE" ] || [ -e "$AF/STRATA_RELEASE" ] \
    || tail -1 "$HOME/StrataEval/h2h/logs/waiter.log" 2>/dev/null | grep -q "WAITER END: DONE"
}
pids_gone() { local p_; for p_ in ${CARD_AFTER_PIDS:-}; do kill -0 "$p_" 2>/dev/null && return 1; done; return 0; }
[ "$STAGE" = arm ] || { say "runner 63 has the arm stage only: runner stops"; exit 2; }
[ -n "$FENV" ] || { say "the arm stage needs CARD_FENV: runner stops"; exit 2; }
[ -z "$EXT" ] || [ "${CARD_RESUME:-0}" = 1 ] || { say "CARD_EXTEND continues an arm: it needs CARD_RESUME=1: runner stops"; exit 2; }
if [ "$DRY" != 1 ]; then
if [ "${CARD_RESUME:-0}" != 1 ]; then
  for r in "$RUNF" "$RUNC"; do [ -e "results/$r" ] && { echo "results/$r exists: move it aside, or CARD_RESUME=1 to continue it"; exit 2; }; done
fi
echo $$ > results/q63_${TAG}.pid; trap 'rm -f results/q63_${TAG}.pid' EXIT
rm -f "$DONE"
say "runner 63 (the P2 gate 45, $TAG, P2 $P2MODE) start (pid $$)$([ "${CARD_RESUME:-0}" = 1 ] && echo ', a resume'): waits for $FENV (written after Wed 14:33); cutoff $(date -r "$CUTOFF" '+%m-%d %H:%M')"
while [ ! -s "$FENV" ] || [ "$(stat -f %m "$FENV" 2>/dev/null || echo 0)" -lt "$EARLIEST" ] \
    || [ $(( $(date +%s) - $(stat -f %m "$FENV" 2>/dev/null || echo 0) )) -lt 120 ]; do
  sleep 60; [ "$(date +%s)" -lt "$CUTOFF" ] || { say "no env file by the cutoff: runner stops"; exit 6; }
done
else say "DRY run on $FENV"; fi
FB=$(git -C "$BM" hash-object "$FENV" 2>/dev/null)
[ -n "${CARD_FBLOB:-}" ] && [ "$FB" = "$CARD_FBLOB" ] || { say "the env file's blob ($FB) is not the registered CARD_FBLOB (${CARD_FBLOB:-unset}): runner stops"; exit 2; }
say "env file $FENV: blob $FB (registered)"
ENVLINE=$(grep -v '^#' "$FENV" | grep . | tail -1)
parse_env "$FENV" || { say "the env file's env line does not parse as K=V pairs: runner stops"; exit 2; }
STACKENV=("${ENVARR[@]}")
COMMENT=$(grep '^#' "$FENV" | paste -sd' ' -)
[ "$DRY" = 1 ] || echo "$COMMENT" | tr -d '-' | grep -qi "$TAG" || { say "the env file's comment does not name the candidate '$TAG' (CHAT_TAG): runner stops"; exit 2; }
C=$(echo "$COMMENT" | grep -oE '(Sat|Sun|Mon|C) [12][0-9][0-9]' | head -1 | awk '{print $2}'); C=${C:-256}
case "$C" in 100|160|176|192|208|224|240|256) ;; *) say "C '$C' from the comment is not a release C: runner stops"; exit 2;; esac   # 100: W24 (24 GB)
MEM=$(echo "$COMMENT" | grep -oE 'MEM [0-9]+' | head -1 | awk '{print $2}'); MEM=${MEM:-47}; MEM=${CARD_MEM:-$MEM}   # PREREG_p2gate45: 45 (49 GB NO-SWAP)
EB=$(echo "$ENVLINE" | tr ',' '\n' | grep '^BIGMOE_EXT_BUILD=' | cut -d= -f2); EB=${EB:-build}
BMC=$(git -C "$BM" rev-parse --short HEAD); QOMSO=$(shasum -a 256 "$BM"/ext/"$EB"/*.so | cut -c1-12)
SNAP=$(echo "$COMMENT" | grep -oE 'data/r57/snap/([A-Za-z0-9_.-]+/)*r57_screen[A-Za-z0-9_.-]*\.py' | head -1)
[ -z "$SNAP" ] && echo "$COMMENT" | grep -qi 'snap' && { say "the comment names a snapshot but no data/r57/snap/[dir/]r57_screen*.py: runner stops"; exit 2; }
if [ -n "$SNAP" ]; then
  MODE=snapshot; SCREEN=$BM/$SNAP
  [ -f "$SCREEN" ] || { say "no $SCREEN: runner stops"; exit 2; }
  SNAPP=("$SNAP" $(grep -oE 'data/r57/snap/([A-Za-z0-9_.-]+/)*frozen_[A-Za-z0-9_]+' "$SCREEN" | sort -u))
  [ "$(dirname "$SNAP")" != data/r57/snap ] && SNAPP+=("$(dirname "$SNAP")")
  bm_pin || { say "the snapshot (${SNAPP[*]}) or bigmoe/ is not BigMoE $BMC (uncommitted or untracked files), or the .so: runner stops"; exit 7; }
else
  MODE=freeze; FZ=${S4_FZBASE:-$HOME/EvalSuite/$AF}/fz_$(date +%m%d_%H%M%S)
  python3 tools/freeze_r57.py now "$FZ" "$EB" >> "$LOG" 2>&1 || { say "the freeze failed: runner stops"; exit 2; }
  SCREEN=$FZ/BigMoE/scripts/r57_screen_fast.py
fi
say "env: $(echo "$COMMENT" | cut -c1-220)"
for s_ in ifbench gpqad lcb6; do
  grep -q "\"$s_\": *\"$s_\"" "$SCREEN" || { say "the screen's SETOF does not map $s_ ($SCREEN): runner stops"; exit 2; }
done
say "config env (${#STACKENV[@]} keys): $ENVLINE"
say "C $C (MEM $MEM); BigMoE $BMC; ext $EB .so $QOMSO; code: $MODE${SNAP:+ $SNAP}${FZ:+ $FZ} (${SNAPP[*]:1}); screen $(shasum -a 256 "$SCREEN" | cut -c1-12)"
PIN0=$(pin)
say "code pin $PIN0"
QS=$(printf '%s\n' "${STACKENV[@]}" | grep '^R57_QOM_SETS=' | tail -1); QS=${QS#R57_QOM_SETS=}
# amendment 12a: the screen stops a process at R57_HARD s of running time (default 600; the w24 env sets 1200). A process
# takes HARD + its load, the cut row and teardown (762 s at most at HARD 600, Mon 11-09), so each is priced HARD + 300 s
# before the cutoff (900 s at 600, as before) and its shared lock lease is 2x that
HARDS=$(printf '%s\n' "${STACKENV[@]}" | grep '^R57_HARD=' | tail -1); HARDS=${HARDS#R57_HARD=}; HARDS=${HARDS:-600}
case "$HARDS" in *[!0-9]*) say "R57_HARD '$HARDS' is not whole seconds: runner stops"; exit 2;; esac
STEP=$((HARDS + 300))
say "R57_HARD $HARDS s: each process is priced $STEP s before the cutoff; shared lock lease $((STEP / 30)) min"
case "${CARD_YIELD_GB:-5}" in ''|*[!0-9]*) say "CARD_YIELD_GB '${CARD_YIELD_GB}' is not whole GB: runner stops"; exit 2;; esac
say "each process start yields to another session's shared MEM run above ${CARD_YIELD_GB:-5} GB, or to two above 5 GB"
KEYS=$(LC_ALL=C sort "$KF" | paste -sd, -)     # the card gate's forward order (sorted keys), restricted to P2's keys
CKEYS=$(LC_ALL=C sort "$CF" | paste -sd, -)
[ "$DRY" = 1 ] && { say "DRY: QOM sets ${QS:-unset}; $(echo "$KEYS" | tr ',' '\n' | grep -c .) keys + $(echo "$CKEYS" | tr ',' '\n' | grep -c .) controls; reference $REFRUN $(refcount) of $RNK; pin $PIN0; bm_pin $(bm_pin && echo OK || echo FAILS); stops before the first GPU step"; exit 0; }
# ------------------------------------------------ the arm: P2's keys in the forward order, then the controls
PHASE="P2 gate $TAG"; EXTRAENV=(); TAGP0="es_cd${TAG}_"; TAGP="${TAGP0}$(date +%d%H%M)_"; IFECOPY=0; REFMAN=""
[ "${CARD_RESUME:-0}" = 1 ] && REFMAN=$(ls -tr "$AF"/${TAGP0}*_modules.json 2>/dev/null | head -1)
[ -n "$REFMAN" ] && say "resume: reference manifest $REFMAN"
order "$RUNF" fwd "$KEYS"
WARM=0   # PREREG_p2gate45: the 5-key control process without the warm-up (its gsm8k prompt is <= 256 tokens, so in the
#   P2 arm it would go through P2 and change the expert-cache state the controls meet); then neither arm engages P2 there
order "$RUNC" ctl "$CKEYS"
if [ "$P2MODE" = off ]; then
  n_=$(python3 -c 'import json, sys; print(len({json.loads(l)["key"] for l in open(sys.argv[1])}))' "results/$RUNF/decode.jsonl")
  echo "CONTROL $TAG: $n_ of $RNK rows in $RUNF (P2 off on every kept run)" > "$DONE"
  say "the control arm: $(cat "$DONE")"; say "runner 63 end"; exit 0
fi
[ "$(refcount)" = "$RNK" ] || { say "the reference ($REFRUN) holds $(refcount) of $RNK keys: no verdict: runner stops"; echo "P2 GATE $TAG: INCOMPLETE (the reference holds $(refcount) of $RNK keys)" > "$DONE"; exit 6; }
share_lock "evalsuite: P2 gate verdict scoring (CPU only: graders)" 60 "--cpu 150 --gpu 0"
F=""; I=""; S=""
for s_ in ifbench gpqad; do
  n_=150; [ "$s_" = gpqad ] && n_=67
  SET_D4_N="$n_" $PY tools/set_d4.py "$s_" "$RUNF" "$REFRUN" > "results/card_${TAG}_${s_}_d4.txt" 2>&1
  l=$(grep "^$s_ vs $REFRUN " "results/card_${TAG}_${s_}_d4.txt" | tail -1)
  say "P2 $TAG $s_ (n $n_): ${l:-missing}"
  case "$l" in
    "") I="$I $s_" ;;
    *": PASS"|*": DIFFERS (HIGHER)") ;;
    *) F="$F $s_" ;;
  esac
  S="$S; $s_$(echo "$l" | sed -E 's/^[^:]*://' | cut -c1-70)"
done
CTL=$(python3 - "$RUNC" "$REFRUN" "$CF" <<'CT'
import json, sys
load = lambda r: {json.loads(l)["key"]: json.loads(l) for l in open(f"results/{r}/decode.jsonl")}
A, R, ks = load(sys.argv[1]), load(sys.argv[2]), open(sys.argv[3]).read().split()
same = [k for k in ks if k in A and k in R and A[k]["completion_ids"] == R[k]["completion_ids"]]
print(f"{len(same)} of {len(ks)}")
CT
)
say "P2 $TAG controls token-identical to the card: $CTL"
peer unlock > /dev/null 2>&1
V=PASS
[ "${CARD_CTL_IDENT:-1}" = 1 ] && { [ "$CTL" = "$NC of $NC" ] || F="$F control($CTL)"; }   # PREREG_p2gate45: identity required only if the control is deterministic
[ -n "$I" ] && V="INCOMPLETE ($I )"; [ -n "$F" ] && V="FAIL ($F )"
echo "P2 GATE $TAG: $V vs $REFRUN$S; controls $CTL identical (identity $([ "${CARD_CTL_IDENT:-1}" = 1 ] && echo required || echo information))" > "$DONE"
say "the P2 verdict: $(cat "$DONE")"
peer say --to "${CARD_NOTIFY:-AI/LLM project recommendations}" "EvalSuite P2 gate $(date +%H:%M): $(cut -c1-230 "$DONE")" > /dev/null 2>&1
say "runner 63 end"
