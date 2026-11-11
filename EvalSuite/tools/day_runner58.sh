#!/bin/bash
# EvalSuite runner 58, Sat 2026-11-07 (attached to a live evalsuite session): the C30 CODE-ONLY gate
# (docs/PREREG_night_gates_2026-11-07.md gate 2): BigMoE's data/stack/c30_env.txt (S5''s config at C 160, VTIER off;
# MEM 35 GB; the committed snapshot screen its comment names). After runner 57 ends:
# 1. Same scores: HumanEval+ 60 in two prompt orders, 10 prompts per process; tools/he_d4.py vs NAX6-N (results/ref2tnax)
#    per order (REF2-T as information); PASS = both orders.
# 2. Long code: every second key of keys_T (21 rows), one order, QOM on humanevalt; tfwin vs NAX6-N; <= 1 blunder / 10K.
# -> results/C30_DONE (C30: PASS | FAIL | INCOMPLETE | VOID). Runner 54's (S4's) code checks, manifests and gates; a
# "no free slot" abort = INCOMPLETE (no retry); no step starts that would end after 00:45 (CERT 01:30 is never
# delayed). Dry run: C30_DRY=1 [C30_FENV=<file>]. Stop it by pid only, and its children too; then peer unlock.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day58_runner.log
AF=results/c30
mkdir -p "$AF"
EARLIEST=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-07 18:00" +%s)   # c30_env.txt was written 18:50
CUTOFF=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-08 00:45" +%s)
export EVALSUITE_DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-08 01:15" +%s)
export PEER_QUEUE_TIMEOUT=21600
R3IDX=$HOME/BigMoE/models/bare/records3/index.json
BM=$HOME/BigMoE
FENV=${C30_FENV:-$BM/data/stack/c30_env.txt}
DRY=${C30_DRY:-0}
[ "$DRY" = 1 ] && LOG=results/day58_dry.log
UNSETS=(-u EVALSUITE_BIGMOE_CODE -u EVALSUITE_RELAX_VARIANT -u EVALSUITE_STAGE_ENV -u R57_KEYS -u R57_TAG -u BIGMOE_TIER
  -u BIGMOE_TIER_GLU -u BIGMOE_TIER_CL -u BIGMOE_TIER_COMBINE -u BIGMOE_OOO -u BIGMOE_MAXK)
STAGEENV="BIGMOE_MAXK=160,BIGMOE_TIER=4433333333,BIGMOE_TIER_GLU=3c,BIGMOE_TIER_CL=112,BIGMOE_TIER_COMBINE=1,BIGMOE_OOO=1"
PY=.venv/bin/python
EST_TF=2700
MODE=""; FZ=""; SNAPP=()
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
void() { say "$1: VOID: runner stops"; echo "C30: VOID ($1)" > results/C30_DONE; exit 5; }
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
      tools/d4x.py tools/mbpp_score.py tools/freeze_r57.py tools/rows_text.py tools/he_d4.py "$SCREEN" results/drift/keys_T.txt data/items/dec_humanevalt.jsonl \
      data/items/dec_mbpp.jsonl data/items/dec_math.jsonl data/items/dec_ifeval.jsonl data/items/dec_humaneval.jsonl
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
ks = set(a) & set(b) if sys.argv[3] == "shared" else set(a) | set(b)
d = sorted(k for k in ks if a.get(k) != b.get(k))
print(f"outside modules: {len(b)}, compared {len(ks)} ({sys.argv[3]}), equal: {len(ks) - len(d)}" + (f"; DIFFERENT: {d}" if d else "")
      + (f"; only in this stage: {sorted(set(b) - set(a))}" if sys.argv[3] == "shared" and set(b) - set(a) else ""))
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
  [ "${BMPIN:-0}" = 1 ] && ! bm_pin && { say "BigMoE's code is not $BMC / the freeze ($MODE): runner stops"; exit 7; }
  [ "${BMPIN:-0}" = 1 ] && ! ext_pin && { say "a module the stack loads changed vs $REFMAN: runner stops"; exit 7; }
  store_ok || { say "records3 is not fully converted: runner stops"; exit 8; }
  [ "$(swapgb)" -le $(( SWAP0 + 2 )) ] || { say "swap $(swapgb) GB > $SWAP0 + 2 at the start: runner stops"; exit 3; }
  [ $(( $(date +%s) + $3 )) -le "$CUTOFF" ] || { say "$1: does not fit before 00:45 ($(( $3 / 60 )) min): runner stops"; exit 6; }
  w=0
  until [ "$(freegb)" -ge $(( $2 + 8 )) ]; do
    [ $((w % 10)) = 0 ] && say "$1: waiting for memory: $(freegb) GB NO-SWAP (needs $(( $2 + 8 )))"
    w=$((w + 1)); sleep 60; calm
    [ $(( $(date +%s) + $3 )) -le "$CUTOFF" ] || { say "$1: no memory before the cutoff: runner stops"; exit 6; }
  done
}
share_lock() {  # reason, lease minutes: retry while a quiet lock runs
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
  BMPIN=1 gate "C30 $3" "$((MEM - 4))" 900     # needs MEM + 4 GB NO-SWAP free (BigMoE's own margin)
  rm -f "$BM/data/r57/${tag}_"*
  share_lock "evalsuite: MEM $MEM GB: $PHASE, $3" 30
  t0=$(date +%s)
  env "${UNSETS[@]}" "${STACKENV[@]}" "${EXTRAENV[@]}" R57_PRESET=45 R57_REPS=1 R57_WARM=1 R57_TAG="$tag" R57_KEYS="$keys" \
    "$BM/.venv/bin/python" tools/r57_wrap.py "$SCREEN" "$AF/${tag}_modules.json" "$C" 1536 > "$AF/${tag}.log" 2>&1; rc=$?
  peer unlock > /dev/null 2>&1
  d=$(( $(date +%s) - t0 ))
  say "C30 $3: rc=$rc ($((d / 60)) min)"
  grep -q "no free slot" "$AF/${tag}.log" && { say "C30 $3: a 'no free slot' abort: INCOMPLETE: runner stops"; echo "C30: INCOMPLETE (a 'no free slot' abort in ${tag})" > results/C30_DONE; exit 6; }
  cp "$BM/data/r57/${tag}_"* "$AF/" 2>/dev/null
  [ -s "$AF/${tag}_tokens.json" ] || { say "C30 $3: no tokens (results/c30/${tag}.log)"; return 1; }
  bm_loaded "$AF/${tag}_modules.json" >> "$LOG" 2>&1 || void "C30 $3: a loaded outside module is not $BMC's / the freeze's"
  if [ -z "$REFMAN" ]; then
    REFMAN="$AF/${tag}_modules.json"; say "reference manifest: $REFMAN"
    if [ -z "$REFMAN1" ]; then REFMAN1=$REFMAN
    else man_same "$REFMAN1" "$REFMAN" shared >> "$LOG" 2>&1 || void "C30 $3: its shared modules differ from the first stage's ($REFMAN1)"; fi
  else man_same "$REFMAN" "$AF/${tag}_modules.json" >> "$LOG" 2>&1 || void "C30 $3: its modules differ from $REFMAN"; fi
  python3 - "$AF/${tag}_runs.json" <<'ENG' >> "$LOG" 2>&1 || void "C30 $3: QOM not engaged on code"
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
        f.write(json.dumps(dict(r, source="NAX6's own row (chat stays EXACT; PREREG_s4)")) + "\n")
print(f"results/{run}: + {len(add)} IFEval rows from results/ref2tnax")
IFE
  $PY tools/rows_text.py "$run" >> "$LOG" 2>&1 || { say "$lab: rows_text.py failed: runner stops"; exit 4; }   # the scorers read text
  say "$lab: $(grep -c . "results/$run/decode.jsonl") rows"
}
# ------------------------------------------------ start: c30_env.txt, then runner 57 out of the way
if [ "$DRY" != 1 ]; then
for r in c30_fwd c30_rev c30_ht; do [ -e "results/$r" ] && { echo "results/$r exists: move it aside first"; exit 2; }; done
echo $$ > results/q58.pid; trap 'rm -f results/q58.pid' EXIT
rm -f results/C30_DONE
say "runner 58 (the C30 code gate) start (pid $$); waits for $FENV (written after 18:00) and runner 57"
while [ ! -s "$FENV" ] || [ "$(stat -f %m "$FENV" 2>/dev/null || echo 0)" -lt "$EARLIEST" ] \
    || [ $(( $(date +%s) - $(stat -f %m "$FENV" 2>/dev/null || echo 0) )) -lt 120 ]; do
  sleep 60; [ "$(date +%s)" -lt "$CUTOFF" ] || { say "no c30_env.txt by 00:45: runner stops"; exit 6; }
done
while alive 57; do waitlog "waiting for runner 57 (the 12 GB gate) to end"; sleep 60; done
else say "DRY run on $FENV"; fi
ENVLINE=$(grep -v '^#' "$FENV" | grep . | tail -1)
parse_env "$FENV" || { say "c30_env.txt's env line does not parse as K=V pairs: runner stops"; exit 2; }
STACKENV=("${ENVARR[@]}")
COMMENT=$(grep '^#' "$FENV" | paste -sd' ' -)
C=$(echo "$COMMENT" | grep -oE '(Sat|C) [12][0-9][0-9]' | head -1 | awk '{print $2}'); C=${C:-160}
case "$C" in 160|176|192|208|224|240|256) ;; *) say "C '$C' from the comment is not a release C: runner stops"; exit 2;; esac
MEM=$(echo "$COMMENT" | grep -oE 'MEM [0-9]+' | head -1 | awk '{print $2}'); MEM=${MEM:-35}
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
say "c30_env: $(echo "$COMMENT" | cut -c1-220)"
say "config env (${#STACKENV[@]} keys): $ENVLINE"
say "C $C (MEM $MEM); BigMoE $BMC; ext $EB .so $QOMSO; code: $MODE${SNAP:+ $SNAP}${FZ:+ $FZ} (${SNAPP[*]:1}); screen $(shasum -a 256 "$SCREEN" | cut -c1-12)"
PIN0=$(pin)
say "code pin $PIN0"
QS=$(printf '%s\n' "${STACKENV[@]}" | grep '^R57_QOM_SETS=' | tail -1); QS=${QS#R57_QOM_SETS=}
KEYS=$(python3 -c 'import json
print(",".join(sorted(json.loads(l)["key"] for l in open("data/items/dec_humaneval.jsonl"))))')
RKEYS=$(echo "$KEYS" | tr ',' '\n' | tail -r | paste -sd, -)
HKEYS=$(awk 'NR % 2 == 1' results/drift/keys_T.txt | paste -sd, -)
case ",$QS," in *,humanevalt,*) HEXTRA=() ;; *) HEXTRA=("R57_QOM_SETS=${QS:+$QS,}humanevalt") ;; esac
[ "$DRY" = 1 ] && { say "DRY: QOM sets ${QS:-unset}; HE+ $(echo "$KEYS" | tr ',' '\n' | grep -c .) per order; long $(echo "$HKEYS" | tr ',' '\n' | grep -c .)${HEXTRA:+ (${HEXTRA[0]})}; pin $PIN0; bm_pin $(bm_pin && echo OK || echo FAILS); stops before the first GPU step"; exit 0; }
# ------------------------------------------------ 1. same scores: HumanEval+ 60, two orders, vs NAX6-N
PHASE="C30 same scores"; EXTRAENV=(); TAGP=es_c30_; IFECOPY=0
order c30_fwd fwd "$KEYS"
order c30_rev rev "$RKEYS"
SSF=""; SSI=""
for o in fwd rev; do
  $PY tools/he_d4.py "c30_$o" ref2tnax ref2t > "results/c30_${o}_d4.txt" 2>&1
  l=$(grep '^humaneval vs ref2tnax ' "results/c30_${o}_d4.txt" | tail -1)
  say "C30 $o: ${l:-missing}"
  case "$l" in "") SSI="$SSI $o" ;; *": PASS"|*": DIFFERS (HIGHER)") ;; *) SSF="$SSF $o" ;; esac
done
SS=PASS; [ -n "$SSI" ] && SS="INCOMPLETE ($SSI )"; [ -n "$SSF" ] && SS="FAIL ($SSF )"
say "C30 same scores: $SS"
# ------------------------------------------------ 2. long code: every second key of keys_T, one order, tfwin
PHASE="C30 long code"; EXTRAENV=("${HEXTRA[@]}"); TAGP=es_c30h_; IFECOPY=0; REFMAN=""
order c30_ht fwd "$HKEYS"
tfwin c30_ht "C30 long code"
python3 - <<'HTR' > results/c30_ht_report.txt 2>&1
import gzip, json, math
rows = {r["key"]: r for r in map(json.loads, open("results/c30_ht/decode.jsonl"))}
tok = dev = bl = 0
for k in rows:
    s = json.load(gzip.open(f"results/c30_ht/tfwin/{k.replace('/', '__')}.json.gz", "rt"))
    out = rows[k]["completion_ids"]
    tok += len(out); dev += sum(a != b for a, b in zip(s["am"], out)); bl += sum(lp < math.log(0.01) for lp in s["lp"])
r = 1e4 * bl / max(tok, 1)
print(f"C30_HT: {len(rows)} rows, {tok} tokens, blunders {bl} = {r:.2f} per 10K, deviations {100 * dev / max(tok, 1):.2f} %: "
      f"{'PASS' if r <= 1 else 'FAIL'}")
HTR
HT=$(grep '^C30_HT' results/c30_ht_report.txt | tail -1)
say "${HT:-C30_HT: no report (results/c30_ht_report.txt)}"
# ------------------------------------------------ the verdict
F=""; I=""
case "$SS" in PASS) ;; INCOMPLETE*) I="$I same-scores" ;; *) F="$F same-scores" ;; esac
case "$HT" in *": PASS") ;; *": FAIL") F="$F long-code" ;; *) I="$I long-code" ;; esac
V=PASS; [ -n "$I" ] && V="INCOMPLETE ($I )"; [ -n "$F" ] && V="FAIL ($F )"
echo "C30: $V; same scores (HE+ vs NAX6-N) $SS; ${HT:-long code missing}" > results/C30_DONE
say "the C30 verdict: $(cat results/C30_DONE)"
peer say --to ojox-3f "EvalSuite C30 $(date +%H:%M): $(cut -c1-230 results/C30_DONE)" > /dev/null 2>&1
say "runner 58 end"
