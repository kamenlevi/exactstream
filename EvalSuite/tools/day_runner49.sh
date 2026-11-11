#!/bin/bash
# EvalSuite runner 49, the Fri -> Sat night (attached to a live evalsuite session), after BigMoE's stacked palindrome
# (NewIdeas' GPU ORDER, 3d82ace):
# 1. The FAST stack's same-scores run (docs/PREREG_stack_2026-11-06.md + amendment 1): the code profile BigMoE writes to
#    ~/BigMoE/data/stack/code_env.txt (stackr or the stack, QOM-based, NAX6-N) on the core math 50 + HumanEval+ 60, in two
#    prompt orders, chunks of 10 prompts per process (the screen's HARD = 600 s), at the C of BigMoE's run ("Sat <C>" in
#    the file's comment); the screen is tonight's snapshot data/r57/snap/r57_screen_sat.py. IFEval stays EXACT: NAX6's own
#    rows (results/ref2tnax). tools/d4x.py vs REF2-T per order -> results/stack_{fwd,rev}_d4.txt, results/STACK_DONE.
# 2. The code-only drift top-up (PREREG_relax_drift amendment 2): RELAX's B on the 41 humanevalt keys + REF2-T's replay
#    (the converted store, ref2tf0, as Friday's runner 45), then tools/relax_report.py's top-up line -> results/TOPUP_DONE.
# 2b. The stack's CODE env on all 41 humanevalt rows (blunders, tfwin; NewIdeas Fri 13:3x), + the QOM-off arm if the
#    afternoon smoke said > 1/10K (results/topsmoke_report.txt) -> results/stack_ht_report.txt.
# 3. MBPP (docs/PREREG_mbpp_2026-11-06.md + amendment 3): NAX6-N's AR on dec_mbpp (ref2tnaxn.r2, done Fri 16:46), then
#    the stack WITHOUT CD16 (NewIdeas Fri 16:0x: CD16 KILLed on GOAL's MBPP12), a coverage check, last
#    (stack_env.txt + R57_QOM_SETS ... mbpp) in two orders, tools/mbpp_score.py d4 per order -> results/MBPP_DONE.
# Waits for code_env.txt (written tonight, after BigMoE's report), BigMoE's sat_night.sh gone and the day rule. Every GPU
# step: the gates of runners 47 / 48 (pins: EvalSuite's code + BigMoE's HEAD at the start + the ext .so + every external
# module of the first stack process's manifest (Leap / TrainLab / Breakthrough / MoonShots, Fri 09:3x); the store; swap;
# memory; END-bounded by 09:30). Stop it by pid only, and its children too; then peer unlock.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day49_runner.log
AF=results/stack
mkdir -p "$AF"
EARLIEST=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-07 00:30" +%s)
CUTOFF=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-07 09:30" +%s)
export EVALSUITE_DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-07 10:00" +%s)
export PEER_QUEUE_TIMEOUT=21600
R3IDX=$HOME/BigMoE/models/bare/records3/index.json
BM=$HOME/BigMoE
CENV=$BM/data/stack/code_env.txt
SCREEN=$BM/data/r57/snap/r57_screen_sat.py
UNSETS=(-u EVALSUITE_BIGMOE_CODE -u EVALSUITE_RELAX_VARIANT -u EVALSUITE_STAGE_ENV -u R57_KEYS -u R57_TAG -u BIGMOE_TIER
  -u BIGMOE_TIER_GLU -u BIGMOE_TIER_CL -u BIGMOE_TIER_COMBINE -u BIGMOE_OOO -u BIGMOE_MAXK)
RLXENV="BIGMOE_MAXK=160,BIGMOE_TIER=4433333333,BIGMOE_TIER_GLU=3c,BIGMOE_TIER_CL=112,BIGMOE_TIER_COMBINE=1,BIGMOE_OOO=1"
STAGEENV=$RLXENV
PY=.venv/bin/python
EST_TF=2700      # tfwin's chunk estimate (unset before Sat 04:0x: bash dropped the tfwin call on the arithmetic error)
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
lid_shut() {
  ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes || return 1
  (( $(system_profiler SPDisplaysDataType -detailLevel mini 2>/dev/null | grep -c "Resolution:") > 1 )) && return 1
  return 0
}
LASTWAIT=0
waitlog() { [ $(( $(date +%s) - LASTWAIT )) -ge 600 ] && { say "$*"; LASTWAIT=$(date +%s); }; }
calm() {
  while pgrep -f '^/bin/zsh [^ ]*soft_end\.sh' >/dev/null || [ -e "$HOME/.local/share/quiet/PAUSE" ] || lid_shut; do
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
mem_busy() {    # another session's shared run with MEM > 5 GB (our own reasons start "evalsuite: MEM")
  peer ls 2>/dev/null | grep "shared run" | grep -v "evalsuite: MEM" | grep -oE "MEM [0-9]+ GB" | awk '$2 > 5 {f = 1} END {exit !f}'
}
freegb() { .venv/bin/python -c "import sys; sys.path.insert(0, '.'); from evalsuite.cli import free_gb; print(int(free_gb()))"; }
swapgb() { sysctl -n vm.swapusage | awk '{gsub("M", "", $6); print int($6 / 1024 + 0.5)}'; }
SWAP0=$(swapgb)
bm_pin() {      # BigMoE's bigmoe/ == $BMC (tracked), nothing untracked there, the build_qom .so. Not scripts/*.py (Sat 01:17:
                # a live edit of r57_screen_fast.py stopped the run; the run executes the snapshot screen, and ext_pin
                # pins every module a process loaded, live scripts/ included)
  git -C "$BM" diff --quiet "$BMC" -- bigmoe 2>/dev/null || return 1
  [ -z "$(git -C "$BM" status --porcelain -- bigmoe 2>/dev/null)" ] || return 1
  [ "$(shasum -a 256 "$BM"/ext/"$EB"/*.so | cut -c1-12)" = "$QOMSO" ]
}
bm_loaded() {   # manifest: every module the FAST process loaded from ~/BigMoE equals its file at 6c08636 (else VOID)
  python3 - "$1" "$BM" "$BMC" <<'MAN'
import hashlib, json, subprocess, sys
mods, bm, c = json.load(open(sys.argv[1]))["modules"], sys.argv[2], sys.argv[3]
bad, n = [], 0
for name, (f, sha) in sorted(mods.items()):
    if not f.startswith(bm + "/") or f.endswith(".so") or f.startswith(bm + "/data/"):   # data/r57/snap: BigMoE's own freeze
        continue                                                                          # (untracked; the manifest pin covers it)
    rel = f[len(bm) + 1:]
    r = subprocess.run(["git", "-C", bm, "show", f"{c}:{rel}"], capture_output=True)
    n += 1
    if r.returncode or hashlib.sha256(r.stdout).hexdigest()[:12] != sha:
        bad.append(rel)
print(f"BigMoE modules loaded: {n}, equal to {c}: {n - len(bad)}" + (f"; DIFFERENT or untracked: {bad}" if bad else ""))
sys.exit(1 if bad else 0)
MAN
}
pin() {
  { shasum snap/leap-faff8e5/*.py snap/leap-52cf2c9/*.py snap/leap-52cf2c9/ms_gdn/* tools/r57_wrap.py tools/r57rows.py \
      tools/d4x.py tools/relax_report.py tools/mbpp_score.py tools/rows_text.py "$SCREEN" results/drift/keys_T.txt data/items/dec_humanevalt.jsonl \
      data/items/dec_mbpp.jsonl \
      data/items/dec_math.jsonl data/items/dec_ifeval.jsonl data/items/dec_humaneval.jsonl
    (cd snap/bigmoe-da8e7a6 && find bigmoe ext -type f ! -path '*__pycache__*' | sort | xargs shasum)
    (cd "$HOME/MoonShots/patches/planes_engine" && ls *.py | sort | xargs shasum)
    find evalsuite -name '*.py' ! -path '*__pycache__*' | sort | xargs shasum
    ls "$HOME/BigMoE/.venv/lib/python3.12/site-packages" | grep -E '^(mlx|mlx_lm|mlx_metal|mtplx)-.*dist-info$'; } 2>&1 \
    | shasum | cut -c1-12
}
REFMAN=""        # the first stack process's manifest: every later process must load the same external code
ext_pin() {     # every external module of the reference manifest still has its sha (BigMoE, Leap, TrainLab, Breakthrough, MoonShots)
  [ -n "$REFMAN" ] || return 0
  python3 - "$REFMAN" <<'EXT'
import hashlib, json, sys
m = json.load(open(sys.argv[1]))["modules"]
T = ("/BigMoE/", "/Leap/", "/TrainLab/", "/Breakthrough/", "/MoonShots/")
bad = [f for f, s in m.values() if any(t in f for t in T) and hashlib.sha256(open(f, "rb").read()).hexdigest()[:12] != s]
sys.exit(1 if bad else 0)
EXT
}
man_same() {    # a process's manifest: the same external modules, same shas, as the reference (else VOID)
  python3 - "$REFMAN" "$1" <<'MS'
import json, sys
T = ("/BigMoE/", "/Leap/", "/TrainLab/", "/Breakthrough/", "/MoonShots/")
ext = lambda p: {k: s for k, (f, s) in json.load(open(p))["modules"].items() if any(t in f for t in T)}
a, b = ext(sys.argv[1]), ext(sys.argv[2])
d = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
print(f"external modules: {len(b)}, equal to the reference: {len(b) - len(d)}" + (f"; DIFFERENT: {d}" if d else ""))
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
  [ "$(pin)" = "$PIN0" ] || { say "code changed: pin $PIN0 -> $(pin): runner stops"; exit 7; }
  [ "${BMPIN:-0}" = 1 ] && ! bm_pin && { say "BigMoE's tree is not $BMC / $EB (git diff or status, .so sha): runner stops"; exit 7; }
  [ "${BMPIN:-0}" = 1 ] && ! ext_pin && { say "a module the stack loads changed vs $REFMAN: runner stops"; exit 7; }
  store_ok || { say "records3 is not fully converted: runner stops"; exit 8; }
  [ "$(swapgb)" -le $(( SWAP0 + 2 )) ] || { say "swap $(swapgb) GB > $SWAP0 + 2 at the start: runner stops"; exit 3; }
  [ $(( $(date +%s) + $3 )) -le "$CUTOFF" ] || { say "$1: does not fit before 15:00 ($(( $3 / 60 )) min): runner stops"; exit 6; }
  w=0
  until [ "$(freegb)" -ge $(( $2 + 8 )) ]; do
    [ $((w % 10)) = 0 ] && say "$1: waiting for memory: $(freegb) GB NO-SWAP (needs $(( $2 + 8 )))"
    w=$((w + 1)); sleep 60; calm
    [ $(( $(date +%s) + $3 )) -le "$CUTOFF" ] || { say "$1: no memory before the cutoff: runner stops"; exit 6; }
  done
}
share_lock() {  # reason, lease minutes: retry while a quiet lock runs (exit 3)
  local r w=0
  while :; do
    r=$(peer lock share "${2}m" --cpu 300 --gpu 60 "$1" 2>&1); [ $? = 0 ] && { say "shared lock: $1"; return 0; }
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
    [ "$rc" = 5 ] && { say "tfwin $2: VOID (the self-check): runner stops"; exit 5; }
    if [ "$n" -le "$b" ]; then
      tail -40 results/gpu_stage.log | grep -q "Traceback" && { say "tfwin $2: the stage failed (results/gpu_stage.log): runner stops"; exit 4; }
      np=$((np + 1)); [ "$np" -ge 3 ] && { say "tfwin $2: three chunks without progress: runner stops"; exit 3; }
      sleep 60
    fi
  done
}

fast() {        # tag, keys, label: one stack process (all its keys), under this runner's share lock
  local tag=$1 keys=$2 t0 d rc
  BMPIN=1 gate "stack $3" "$((MEM - 4))" 900          # needs MEM + 4 GB NO-SWAP free: BigMoE's own margin for this config (MG 4);
                                              # the night's order runs it alone
  rm -f "$BM/data/r57/${tag}_"*
  share_lock "evalsuite: MEM $MEM GB: $PHASE, $3" 30
  t0=$(date +%s)
  env "${UNSETS[@]}" "${CURENV[@]}" "${EXTRAENV[@]}" R57_PRESET=45 R57_REPS=1 R57_WARM=1 R57_TAG="$tag" R57_KEYS="$keys" \
    "$BM/.venv/bin/python" tools/r57_wrap.py "$SCREEN" "$AF/${tag}_modules.json" "$C" 1536 > "$AF/${tag}.log" 2>&1; rc=$?
  peer unlock > /dev/null 2>&1
  d=$(( $(date +%s) - t0 ))
  say "stack $3: rc=$rc ($((d / 60)) min)"
  cp "$BM/data/r57/${tag}_"* "$AF/" 2>/dev/null
  [ -s "$AF/${tag}_tokens.json" ] || { say "stack $3: no tokens (results/stack/${tag}.log)"; return 1; }
  bm_loaded "$AF/${tag}_modules.json" >> "$LOG" 2>&1 || { say "stack $3: a loaded BigMoE module is not $BMC's: VOID: runner stops"; exit 5; }
  if [ -z "$REFMAN" ]; then REFMAN="$AF/${tag}_modules.json"; say "reference manifest: $REFMAN"
  else man_same "$AF/${tag}_modules.json" >> "$LOG" 2>&1 || { say "stack $3: its modules differ from $REFMAN: VOID: runner stops"; exit 5; }; fi
  python3 - "$AF/${tag}_runs.json" <<'ENG' >> "$LOG" 2>&1 || { say "stack $3: QOM not engaged on code: VOID: runner stops"; exit 5; }
import json, sys
runs = json.load(open(sys.argv[1]))
code = [r for r in runs if str(r.get("key", "")).startswith(("humaneval/", "humanevalt/", "mbpp/"))]
qom = [r["key"] for r in code if (r.get("pf") or {}).get("qom_opt", 0) > 0]
print(f"engaged: QOM optional uses on {len(qom)} of {len(code)} code (HumanEval / humanevalt / MBPP) runs; types "
      f"{sorted({str(r.get('type')) for r in runs})}; inert {sum(1 for r in runs if r.get('inert'))}, relaxed "
      f"{sum(1 for r in runs if r.get('rlx'))}, span {sum(1 for r in runs if r.get('spn'))} runs")
sys.exit(0 if len(qom) == len(code) else 1)
ENG
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
    while [ -e "$AF/${TAGP}${lab}${i}_tokens.json" ] || [ -e "$AF/${TAGP}${lab}${i}_modules.json" ]; do i=$((i + 1)); done   # a restart
    tag="${TAGP}${lab}${i}"
    fast "$tag" "$(echo "$m" | tr ',' '\n' | head -10 | paste -sd, -)" "$lab process $i"
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
        f.write(json.dumps(dict(r, source="NAX6's own row (chat stays EXACT; PREREG_stack amendment 1)")) + "\n")
print(f"results/{run}: + {len(add)} IFEval rows from results/ref2tnax")
IFE
  say "$lab: $(grep -c . "results/$run/decode.jsonl") rows"
}
# ------------------------------------------------ the relax top-up: runner 45's chunk logic
pick_c() {      # the gate needs C >= MAXK 160: C 192 (need 34) or 160 (need 30) by free memory, else wait (no lock held)
  local fg w=0
  while :; do
    fg=$(freegb)
    if [ "$fg" -ge 38 ]; then RC_C=192; RC_NEED=34; return 0; fi
    if [ "$fg" -ge 34 ]; then RC_C=160; RC_NEED=30; return 0; fi
    [ $((w % 10)) = 0 ] && say "waiting for memory: $fg GB NO-SWAP (C 160 needs 34)"
    w=$((w + 1)); sleep 60; calm
    [ $(( $(date +%s) + 3000 )) -le "$CUTOFF" ] || { say "no relax chunk fits before 09:30 while waiting for memory: runner stops"; exit 6; }
  done
}
have() {   # rows of arm $1 among the keys in file $2
  [ -f "results/relax_$1/decode.jsonl" ] || { echo 0; return; }
  python3 -c 'import json, sys
k = set(open(sys.argv[2]).read().split())
print(sum(json.loads(l)["key"] in k for l in open(sys.argv[1])))' "results/relax_$1/decode.jsonl" "$2"
}
replayed() {   # replays of arm $1 among the keys in file $2
  python3 -c 'import os, sys
k = set(open(sys.argv[2]).read().split())
d = sys.argv[1]
print(sum(f[:-9].replace("__", "/") in k for f in (os.listdir(d) if os.path.isdir(d) else []) if f.endswith(".jsonl.gz")))' \
    "results/relax_$1/replay" "$2"
}
RLX_EST=3000
rchunk() {      # label, stage args...: one relax share-lock chunk (runner 45's), END-bounded
  local lab=$1 t0 d; shift
  gate "$lab" 34 "$RLX_EST"
  pick_c
  t0=$(date +%s)
  EVALSUITE_STAGE_ENV="$RLXENV" EVALSUITE_BIGMOE_CODE="$HOME/EvalSuite/snap/bigmoe-da8e7a6" EVALSUITE_RELAX_VARIANT=ref2tf0 \
    ./bin/evalsuite gpu "$@" --chunk-minutes 45 --C "$RC_C" --need "$RC_NEED" --minutes 60 --share >> "$LOG" 2>&1; RC=$?
  d=$(( $(date +%s) - t0 )); [ "$d" -gt "$RLX_EST" ] && RLX_EST=$d
  say "$lab: chunk rc=$RC at C $RC_C ($((d / 60)) min)"
  [ "$RC" = 5 ] && { say "$lab: VOID (step 0): runner stops"; exit 5; }
}
# ------------------------------------------------ start
grep -q '"humanevalt"' evalsuite/relax.py || { say "relax.py lacks humanevalt: run pending/apply_humanevalt.py first"; exit 2; }
echo $$ > results/q49.pid; trap 'rm -f results/q49.pid' EXIT
say "runner 49 start (pid $$); waits for $CENV (tonight), sat_night.sh gone, then the stack run and the top-up"
while [ "$(date +%s)" -lt "$EARLIEST" ] || [ ! -s "$CENV" ] || [ "$(stat -f %m "$CENV" 2>/dev/null || echo 0)" -lt "$EARLIEST" ] \
    || pgrep -qf 'sat_night\.sh'; do
  sleep 60
  [ "$(date +%s)" -lt "$CUTOFF" ] || { say "no code_env.txt by 09:30: runner stops"; exit 6; }
done
ENVLINE=$(grep -v '^#' "$CENV" | grep . | tail -1)
parse_env "$CENV" || { say "code_env.txt's env line does not parse as K=V pairs: runner stops"; exit 2; }
STACKENV=("${ENVARR[@]}")
C=$(grep -o 'Sat [0-9]*' "$CENV" | head -1 | awk '{print $2}'); C=${C:-224}
case "$C" in 224) MEM=46;; 208) MEM=44;; 192) MEM=42;; 176) MEM=40;; 160) MEM=38;; *) say "C '$C' unknown: runner stops"; exit 2;; esac
EB=$(echo "$ENVLINE" | tr ',' '\n' | grep '^BIGMOE_EXT_BUILD=' | cut -d= -f2); EB=${EB:-build}
BMC=$(git -C "$BM" rev-parse --short HEAD); QOMSO=$(shasum -a 256 "$BM"/ext/"$EB"/*.so | cut -c1-12)
[ -f "$SCREEN" ] || { say "no tonight's screen snapshot $SCREEN: runner stops"; exit 2; }
say "code_env: $(grep '^#' "$CENV" | head -1 | cut -c1-200)"
say "stack env (${#STACKENV[@]} keys): $ENVLINE"
say "C $C (MEM $MEM); BigMoE $BMC; ext $EB .so $QOMSO; screen $(shasum -a 256 "$SCREEN" | cut -c1-12) (live: $(shasum -a 256 "$BM/scripts/r57_screen_fast.py" | cut -c1-12))"
PIN0=$(pin)
say "code pin $PIN0"
KEYS=$(python3 -c 'import json
r = [json.loads(l) for l in open("results/ref2t/decode.jsonl")]
print(",".join(sorted(x["key"] for x in r if x["set"] in ("math", "humaneval"))))')
RKEYS=$(echo "$KEYS" | tr ',' '\n' | tail -r | paste -sd, -)
say "the stack: $(echo "$KEYS" | tr ',' '\n' | grep -c .) keys per order"
PHASE="the stack's same-scores run"; CURENV=("${STACKENV[@]}"); EXTRAENV=(); TAGP=es_st_; IFECOPY=1
[ -s "$AF/es_st_fwd1_modules.json" ] && [ -s results/stack_fwd/decode.jsonl ] && { REFMAN="$AF/es_st_fwd1_modules.json"; say "restart: reference manifest $REFMAN"; }
order stack_fwd fwd "$KEYS"
order stack_rev rev "$RKEYS"
V=""
for o in fwd rev; do
  $PY tools/rows_text.py "stack_$o" >> "$LOG" 2>&1      # r57rows writes ids only; d4x's scorers read text (Sat 01:2x)
  $PY tools/d4x.py "stack_$o" ref2t ref2t.r2 > "results/stack_${o}_d4.txt" 2>&1
  ./bin/evalsuite report --variant "stack_$o" --decode --ref ref2t > "results/report_stack_${o}_vs_ref2t.txt" 2>&1
  say "stack $o: $(grep '^D4X:' "results/stack_${o}_d4.txt" | tail -1)"
  V="$V $o: $(grep '^D4X:' "results/stack_${o}_d4.txt" | tail -1)"
done
case "$V" in *FAIL*|*STAGE2*|*none*) W=FAIL ;; *) W=PASS ;; esac
grep -q "D4X:" results/stack_fwd_d4.txt && grep -q "D4X:" results/stack_rev_d4.txt || W=INCOMPLETE
echo "STACK: $W ($V )" > results/STACK_DONE
say "the stack's same-scores verdict: $(cat results/STACK_DONE)"
# ------------------------------------------------ the code-only drift top-up
KT=results/drift/keys_T.txt
while [ "$(have B "$KT")" -lt "$(grep -c . "$KT")" ]; do
  b=$(have B "$KT"); rchunk "B top-up" relax ref --arm B --keys "$KT"
  [ "$(have B "$KT")" -gt "$b" ] || { tail -40 results/gpu_stage.log | grep -q Traceback && { say "B top-up: the stage failed: runner stops"; exit 4; }; }
done
while [ "$(replayed B "$KT")" -lt "$(have B "$KT")" ]; do
  b=$(replayed B "$KT"); rchunk "replay B top-up" relax-replay ref --arm B
  [ "$(replayed B "$KT")" -gt "$b" ] || { tail -40 results/gpu_stage.log | grep -q Traceback && { say "replay B top-up: the stage failed: runner stops"; exit 4; }; }
done
$PY tools/relax_report.py --arms B --out results/relax_topup_report.txt > /dev/null 2>&1
say "the top-up: $(grep '^== code-only top-up' results/relax_topup_report.txt | sed 's/^== //' | cut -c1-260)"
touch results/TOPUP_DONE
# ------------------------------------------------ the stack's CODE env on all 41 humanevalt rows (NewIdeas Fri 13:3x): blunders
TKEYS=$(grep . results/drift/keys_T.txt | paste -sd, -)
REFMAN_SS=$REFMAN; REFMAN=""
PHASE="the stack on humanevalt"; CURENV=("${STACKENV[@]}"); EXTRAENV=(R57_QOM_SETS=humaneval,mbpp,humanevalt); TAGP=es_ht_; IFECOPY=0
order stack_ht fwd "$TKEYS"
tfwin stack_ht "stack humanevalt"
HT="stack_ht"
if grep -q "ALSO runs the QOM-off" results/topsmoke_report.txt 2>/dev/null; then   # the smoke's trigger (amendment 3)
  REFMAN=""; EXTRAENV=(R57_QOM_SETS=humaneval,mbpp); TAGP=es_hq_
  order stack_htq fwd "$TKEYS"
  tfwin stack_htq "stack humanevalt, QOM off"
  HT="stack_ht stack_htq"
fi
python3 - $HT <<'HTR' > results/stack_ht_report.txt 2>&1
import gzip, json, math, sys
for run in sys.argv[1:]:
    rows = {r["key"]: r for r in map(json.loads, open(f"results/{run}/decode.jsonl"))}
    tok = dev = bl = 0
    for k in rows:
        s = json.load(gzip.open(f"results/{run}/tfwin/{k.replace('/', '__')}.json.gz", "rt"))
        out = rows[k]["completion_ids"]
        tok += len(out); dev += sum(a != b for a, b in zip(s["am"], out)); bl += sum(lp < math.log(0.01) for lp in s["lp"])
    print(f"STACK_HT {run}: {len(rows)} rows, {tok} tokens, blunders {bl} = {1e4 * bl / max(tok, 1):.2f} per 10K, "
          f"deviations {100 * dev / max(tok, 1):.2f} %")
HTR
say "$(grep '^STACK_HT' results/stack_ht_report.txt | paste -sd' ' -)"
REFMAN=$REFMAN_SS
# ------------------------------------------------ MBPP (docs/PREREG_mbpp_2026-11-06.md): NAX6-N's AR, then the stack with CD16
count() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
MREF=ref2tnaxn.r2
TIERENV="BIGMOE_TIER=4433333333,BIGMOE_TIER_GLU=3c,BIGMOE_TIER_CL=112"
DEST=3000
while [ "$(count results/$MREF/decode.jsonl)" -lt 100 ]; do
  b=$(count "results/$MREF/decode.jsonl")
  gate "MBPP reference" 34 "$DEST"; pick_c
  t0=$(date +%s)
  EVALSUITE_STAGE_ENV="$TIERENV" EVALSUITE_BIGMOE_CODE="$HOME/EvalSuite/snap/bigmoe-da8e7a6" \
    ./bin/evalsuite gpu decode "$MREF" --sets mbpp --chunk-minutes 45 --share --C "$RC_C" --need "$RC_NEED" --minutes 60 \
    >> "$LOG" 2>&1; RC=$?
  d=$(( $(date +%s) - t0 )); [ "$d" -gt "$DEST" ] && DEST=$d
  say "MBPP reference: chunk rc=$RC ($(count "results/$MREF/decode.jsonl")/100 rows, $((d / 60)) min)"
  [ "$(count "results/$MREF/decode.jsonl")" -gt "$b" ] || { tail -40 results/gpu_stage.log | grep -q Traceback && { say "MBPP reference: the stage failed: runner stops"; exit 4; }; }
done
SENV=$BM/data/stack/stack_env.txt
SLINE=$(grep -v '^#' "$SENV" | grep . | tail -1)
parse_env "$SENV" || { say "stack_env.txt's env line does not parse as K=V pairs: runner stops"; exit 2; }
MBENV=("${ENVARR[@]}")
echo "$SLINE" | tr ',' '\n' | grep -q '^R57_INERT=CD16$' && say "NOTE: stack_env.txt still carries R57_INERT=CD16 (NewIdeas dropped it Fri 16:0x)"
QS=$(printf '%s\n' "${MBENV[@]}" | grep '^R57_QOM_SETS=' | tail -1)
case ",${QS#R57_QOM_SETS=}," in *,mbpp,*) EXTRAENV=() ;; *) EXTRAENV=(R57_QOM_SETS=humaneval,ifeval,multi,mbpp) ;; esac
say "MBPP arm: stack_env.txt ($(grep '^#' "$SENV" | head -1 | cut -c1-120)); QOM sets ${QS:-unset}${EXTRAENV:+ -> ${EXTRAENV[0]}}"
REFMAN_STACK=$REFMAN; REFMAN=""
PHASE="the MBPP coverage check (the stack without CD16)"; CURENV=("${MBENV[@]}")
TAGP=es_mb_; IFECOPY=0
MKEYS=$(python3 -c 'import json
r = [json.loads(l) for l in open("data/items/dec_mbpp.jsonl")]
print(",".join(x["key"] for x in sorted(r, key=lambda x: x["task_id"])))')
MRKEYS=$(echo "$MKEYS" | tr ',' '\n' | tail -r | paste -sd, -)
order mbpp_stack_fwd fwd "$MKEYS"
order mbpp_stack_rev rev "$MRKEYS"
python3 - "$REFMAN_STACK" "$REFMAN" <<'CM' >> "$LOG" 2>&1
import json, sys
T = ("/BigMoE/", "/Leap/", "/TrainLab/", "/Breakthrough/", "/MoonShots/")
ext = lambda p: {k: s for k, (f, s) in json.load(open(p))["modules"].items() if any(t in f for t in T)}
a, b = ext(sys.argv[1]), ext(sys.argv[2])
d = sorted(k for k in set(a) & set(b) if a[k] != b[k])
print(f"the stack run and the MBPP run share {len(set(a) & set(b))} outside modules; differing: {d or 'none'}; only in MBPP: "
      f"{sorted(set(b) - set(a)) or 'none'}")
CM
MV=""
for o in fwd rev; do
  $PY tools/rows_text.py "mbpp_stack_$o" >> "$LOG" 2>&1
  $PY tools/mbpp_score.py d4 "mbpp_stack_$o" "$MREF" > "results/mbpp_${o}_d4.txt" 2>&1
  say "MBPP $o: $(grep '^mbpp (n' "results/mbpp_${o}_d4.txt" | tail -1) -> $(grep '^MBPP:' "results/mbpp_${o}_d4.txt" | tail -1)"
  MV="$MV $o: $(grep '^MBPP:' "results/mbpp_${o}_d4.txt" | tail -1)"
done
case "$MV" in *FAIL*) MW=FAIL ;; *) MW=PASS ;; esac
grep -q "^MBPP:" results/mbpp_fwd_d4.txt && grep -q "^MBPP:" results/mbpp_rev_d4.txt || MW=INCOMPLETE
echo "MBPP: $MW ($MV )" > results/MBPP_DONE
say "the MBPP verdict: $(cat results/MBPP_DONE)"
say "runner 49 end"
