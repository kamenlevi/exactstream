#!/bin/bash
# EvalSuite runner 55, the Sat -> Sun night (attached to a live evalsuite session): CERT, the release certificate for the
# frozen 45 GB FAST config (docs/PREREG_cert_2026-11-08.md + amendments 1-2). Waits for BigMoE's
# ~/BigMoE/data/stack/cert_env.txt (written at the ~20:00 FREEZE; final_env.txt's format: 'Sat <C>', 'MEM <n> GB' and
# 'screen data/r57/snap/<dir>/<file>.py' in its comment) and for runners 53 / 54 to end. Then:
# 0. NAX6-N's reference rows on humanevalx + mathx (ref2tnaxn.r3, 154): the decode stage tops up what runner 53 left.
# 1. The config on CERT's 464 rows (HumanEval+ 60 + 104, math 50 + 50, MBPP 100, IFEval 100: amendment 3, Sat 23:3x)
#    in two prompt orders, 10 prompts per process; R57_QOM_SETS + humanevalx (amendment 1; math / mathx / ifeval stay
#    as the env has them).
# 1b. The chat check (amendment 2): 12 IFEval prompts interleaved with 12 humanevalx prompts, 8 per process, so the
#    engine swaps profiles at request boundaries; the IFEval rows must equal NAX6's AR token for token.
# 2. Long code: the 41 humanevalt rows, one order, + humanevalt in the QOM sets; tfwin against NAX6-N.
# -> tools/cert_report.py: results/cert/CERT_{humaneval,math,mbpp,pooled,longcode,VERDICT,chat,info}.txt and
# results/CERT_DONE. The code, the manifests and the gates are runner 54's (snapshot or freeze; each stage's first
# process manifest is its reference, later stages must match the first stage on shared modules); no step starts that
# would end after Sun 10:00; nothing starts before the slot (CERT_START, default "2026-11-08 01:30"). Dry run: CERT_DRY=1 CERT_FENV=<file> [S4_FZBASE=<dir>]. Stop it by pid only, and its
# children too; then peer unlock.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day55_runner.log
AF=results/cert
mkdir -p "$AF"
EARLIEST=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-07 12:00" +%s)   # cert_env.txt is written at the ~20:00 freeze
START=$(date -j -f "%Y-%m-%d %H:%M" "${CERT_START:-2026-11-08 01:30}" +%s)   # PREREG_cert's slot: Sun ~01:30-10:00 (NewIdeas may move it)
CUTOFF=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-08 10:00" +%s)     # Kamen's Sunday starts ~10:00
export EVALSUITE_DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-08 10:30" +%s)
export PEER_QUEUE_TIMEOUT=21600
R3IDX=$HOME/BigMoE/models/bare/records3/index.json
BM=$HOME/BigMoE
FENV=${CERT_FENV:-$BM/data/stack/cert_env.txt}
DRY=${CERT_DRY:-0}
[ "$DRY" = 1 ] && LOG=results/day55_dry.log
UNSETS=(-u EVALSUITE_BIGMOE_CODE -u EVALSUITE_RELAX_VARIANT -u EVALSUITE_STAGE_ENV -u R57_KEYS -u R57_TAG -u BIGMOE_TIER
  -u BIGMOE_TIER_GLU -u BIGMOE_TIER_CL -u BIGMOE_TIER_COMBINE -u BIGMOE_OOO -u BIGMOE_MAXK)
STAGEENV="BIGMOE_MAXK=160,BIGMOE_TIER=4433333333,BIGMOE_TIER_GLU=3c,BIGMOE_TIER_CL=112,BIGMOE_TIER_COMBINE=1,BIGMOE_OOO=1"
TIERENV="BIGMOE_TIER=4433333333,BIGMOE_TIER_GLU=3c,BIGMOE_TIER_CL=112"
RREF=ref2tnaxn.r3
PY=.venv/bin/python
EST_TF=2700
CHUNK=10
MODE=""; FZ=""; SNAPP=()
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
void() { say "$1: VOID: runner stops"; echo "CERT: VOID ($1)" > results/CERT_DONE; exit 5; }
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
      tools/d4x.py tools/mbpp_score.py tools/freeze_r57.py tools/rows_text.py tools/cert_report.py "$SCREEN" results/drift/keys_T.txt data/items/dec_humanevalt.jsonl \
      data/items/dec_mbpp.jsonl data/items/dec_math.jsonl data/items/dec_ifeval.jsonl data/items/dec_humaneval.jsonl \
      data/items/dec_humanevalx.jsonl data/items/dec_mathx.jsonl
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
  [ $(( $(date +%s) + $3 )) -le "$CUTOFF" ] || { say "$1: does not fit before Sun 10:00 ($(( $3 / 60 )) min): runner stops"; exit 6; }
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
  BMPIN=1 gate "CERT $3" "$((MEM - 4))" 900     # needs MEM + 4 GB NO-SWAP free (BigMoE's own margin)
  rm -f "$BM/data/r57/${tag}_"*
  share_lock "evalsuite: MEM $MEM GB: $PHASE, $3" 30
  t0=$(date +%s)
  env "${UNSETS[@]}" "${STACKENV[@]}" "${EXTRAENV[@]}" R57_PRESET=45 R57_REPS=1 R57_WARM=1 R57_TAG="$tag" R57_KEYS="$keys" \
    "$BM/.venv/bin/python" tools/r57_wrap.py "$SCREEN" "$AF/${tag}_modules.json" "$C" 1536 > "$AF/${tag}.log" 2>&1; rc=$?
  peer unlock > /dev/null 2>&1
  d=$(( $(date +%s) - t0 ))
  say "CERT $3: rc=$rc ($((d / 60)) min)"
  cp "$BM/data/r57/${tag}_"* "$AF/" 2>/dev/null
  [ -s "$AF/${tag}_tokens.json" ] || { say "CERT $3: no tokens (results/cert/${tag}.log)"; return 1; }
  bm_loaded "$AF/${tag}_modules.json" >> "$LOG" 2>&1 || void "CERT $3: a loaded outside module is not $BMC's / the freeze's"
  if [ -z "$REFMAN" ]; then
    REFMAN="$AF/${tag}_modules.json"; say "reference manifest: $REFMAN"
    if [ -z "$REFMAN1" ]; then REFMAN1=$REFMAN
    else man_same "$REFMAN1" "$REFMAN" shared >> "$LOG" 2>&1 || void "CERT $3: its shared modules differ from the first stage's ($REFMAN1)"; fi
  else man_same "$REFMAN" "$AF/${tag}_modules.json" >> "$LOG" 2>&1 || void "CERT $3: its modules differ from $REFMAN"; fi
  python3 - "$AF/${tag}_runs.json" <<'ENG' >> "$LOG" 2>&1 || void "CERT $3: QOM not engaged on code"
import json, sys
runs = json.load(open(sys.argv[1]))
code = [r for r in runs if str(r.get("key", "")).startswith(("humaneval/", "humanevalx/", "humanevalt/", "mbpp/"))]
qom = [r["key"] for r in code if (r.get("pf") or {}).get("qom_opt", 0) > 0]
print(f"engaged: QOM optional uses on {len(qom)} of {len(code)} code (HumanEval / humanevalx / humanevalt / MBPP) runs; types "
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
order() {       # run, label, keys (comma list): chunks of $CHUNK per process; any missed keys go into the next process
  local run=$1 lab=$2 all=$3 i=0 m tag tries=0 cap
  cap=$(( $(echo "$all" | tr ',' '\n' | grep -c .) / CHUNK + 10 ))   # (was 30: CERT's 464 keys need 47; stopped Sun 02:22)
  while :; do
    m=$(missing "$run" "$all"); [ -z "$m" ] && break
    i=$((i + 1)); tries=$((tries + 1)); [ "$tries" -le "$cap" ] || { say "$lab: $cap processes without all rows: runner stops"; exit 3; }
    while [ -e "$AF/${TAGP}${lab}${i}_tokens.json" ] || [ -e "$AF/${TAGP}${lab}${i}_modules.json" ]; do i=$((i + 1)); done   # a restart
    tag="${TAGP}${lab}${i}"
    fast "$tag" "$(echo "$m" | tr ',' '\n' | head -"$CHUNK" | paste -sd, -)" "$lab process $i"
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
        f.write(json.dumps(dict(r, source="NAX6's own row (chat stays EXACT)")) + "\n")
print(f"results/{run}: + {len(add)} IFEval rows from results/ref2tnax")
IFE
  $PY tools/rows_text.py "$run" >> "$LOG" 2>&1 || { say "$lab: rows_text.py failed: runner stops"; exit 4; }   # the scorers read text
  say "$lab: $(grep -c . "results/$run/decode.jsonl") rows"
}
count() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
# ------------------------------------------------ start: cert_env.txt, then runners 53 / 54 out of the way
if [ "$DRY" != 1 ]; then
[ "${CERT_RESUME:-0}" = 1 ] || for r in cert_fwd cert_rev cert_ht cert_chat; do [ -e "results/$r" ] && { echo "results/$r exists: move it aside first (or CERT_RESUME=1)"; exit 2; }; done
echo $$ > results/q55.pid; trap 'rm -f results/q55.pid' EXIT
rm -f results/CERT_DONE
say "runner 55 (CERT) start (pid $$); waits for $FENV (written after Sat 12:00), then runners 53 / 54"
while [ ! -s "$FENV" ] || [ "$(stat -f %m "$FENV" 2>/dev/null || echo 0)" -lt "$EARLIEST" ] \
    || [ $(( $(date +%s) - $(stat -f %m "$FENV" 2>/dev/null || echo 0) )) -lt 120 ]; do   # 2 quiet minutes after the write
  sleep 60
  [ $(( $(date +%s) + 4 * 3600 )) -le "$CUTOFF" ] || { say "no cert_env.txt by Sun 06:00: runner stops"; exit 6; }
done
say "cert_env.txt in (written $(date -r "$(stat -f %m "$FENV")" +%H:%M))"
while alive 53 || alive 54 || [ "$(date +%s)" -lt "$START" ]; do waitlog "waiting for runner 53 / 54 to end and the slot ($(date -r "$START" '+%a %H:%M'))"; sleep 60; done
else say "DRY run on $FENV"; fi
ENVLINE=$(grep -v '^#' "$FENV" | grep . | tail -1)
parse_env "$FENV" || { say "cert_env.txt's env line does not parse as K=V pairs: runner stops"; exit 2; }
STACKENV=("${ENVARR[@]}")
COMMENT=$(grep '^#' "$FENV" | paste -sd' ' -)
C=$(echo "$COMMENT" | grep -oE '(Sat|C) [12][0-9][0-9]' | head -1 | awk '{print $2}'); C=${C:-256}
case "$C" in 160|176|192|208|224|240|256) ;; *) say "C '$C' from the comment is not a release C: runner stops"; exit 2;; esac
MEM=$(echo "$COMMENT" | grep -oE 'MEM [0-9]+' | head -1 | awk '{print $2}'); MEM=${MEM:-47}
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
python3 - "$SCREEN" <<'SET' >> "$LOG" 2>&1 || { say "the screen's SETOF lacks one of CERT's sets: runner stops"; exit 2; }
import re, sys
s = open(sys.argv[1]).read()
m = re.search(r"SETOF = \{(.*?)\}", s, re.S)
need = ("humaneval", "humanevalx", "math500", "gsm8k", "mathx", "mbpp", "humanevalt", "ifeval")
miss = [k for k in need if not m or f'"{k}":' not in m.group(1)]
print(f"screen SETOF covers CERT's sets: {'yes' if not miss else 'NO, missing ' + str(miss)}")
sys.exit(1 if miss else 0)
SET
say "cert_env: $(echo "$COMMENT" | cut -c1-220)"
say "config env (${#STACKENV[@]} keys): $ENVLINE"
say "C $C (MEM $MEM); BigMoE $BMC; ext $EB .so $QOMSO; code: $MODE${SNAP:+ $SNAP}${FZ:+ $FZ} (${SNAPP[*]:1}); screen $(shasum -a 256 "$SCREEN" | cut -c1-12)"
PIN0=$(pin)
say "code pin $PIN0"
QS=$(printf '%s\n' "${STACKENV[@]}" | grep '^R57_QOM_SETS=' | tail -1); QS=${QS#R57_QOM_SETS=}
case ",$QS," in *,humanevalx,*) Q1="$QS" ;; *) Q1="${QS:+$QS,}humanevalx" ;; esac
case ",$Q1," in *,humanevalt,*) Q2="$Q1" ;; *) Q2="$Q1,humanevalt" ;; esac
KEYS=$(python3 -c 'import json
ks = [json.loads(l)["key"] for n in ("dec_humaneval", "dec_humanevalx", "dec_math", "dec_mathx", "dec_mbpp", "dec_ifeval") for l in open(f"data/items/{n}.jsonl")]
assert len(ks) == 464 == len(set(ks)), len(ks)
print(",".join(sorted(ks)))') || { say "CERT's 464 keys do not build: runner stops"; exit 2; }
RKEYS=$(echo "$KEYS" | tr ',' '\n' | tail -r | paste -sd, -)
CKEYS=$(python3 -c 'import json
f = sorted(json.loads(l)["key"] for l in open("data/items/dec_ifeval.jsonl"))[:12]
h = sorted(json.loads(l)["key"] for l in open("data/items/dec_humanevalx.jsonl"))[:12]
print(",".join(k for p in zip(h, f) for k in p))')
TKEYS=$(grep . results/drift/keys_T.txt | paste -sd, -)
[ "$DRY" = 1 ] && { say "DRY: QOM sets ${QS:-unset} -> $Q1 (stage 1, chat check) / $Q2 (long code); keys $(echo "$KEYS" | tr ',' '\n' | grep -c .) + chat $(echo "$CKEYS" | tr ',' '\n' | grep -c .) + long $(echo "$TKEYS" | tr ',' '\n' | grep -c .); refs $(count results/$RREF/decode.jsonl)/154; pin $PIN0; bm_pin $(bm_pin && echo OK || echo FAILS); stops before the first GPU step"; exit 0; }
# ------------------------------------------------ 0. NAX6-N's reference rows on humanevalx + mathx (runner 53's top-up)
DEST=3000; NP=0
while [ "$(count results/$RREF/decode.jsonl)" -lt 154 ]; do
  b=$(count "results/$RREF/decode.jsonl")
  gate "CERT refs" 28 "$DEST"
  fg=$(freegb); RC=192; NEED=36; [ "$fg" -ge 40 ] || { RC=160; NEED=32; }
  t0=$(date +%s)
  EVALSUITE_STAGE_ENV="$TIERENV" EVALSUITE_BIGMOE_CODE="$HOME/EvalSuite/snap/bigmoe-da8e7a6" \
    ./bin/evalsuite gpu decode "$RREF" --sets humanevalx,mathx --chunk-minutes 45 --share --C "$RC" --need "$NEED" --minutes 60 \
    >> "$LOG" 2>&1; rc=$?
  d=$(( $(date +%s) - t0 )); [ "$d" -gt "$DEST" ] && DEST=$d
  say "CERT refs: chunk rc=$rc at C $RC: $(count "results/$RREF/decode.jsonl")/154 rows ($((d / 60)) min)"
  if [ "$(count "results/$RREF/decode.jsonl")" -le "$b" ]; then
    tail -40 results/gpu_stage.log | grep -q Traceback && { say "CERT refs: the stage failed (results/gpu_stage.log): runner stops"; exit 4; }
    NP=$((NP + 1)); [ "$NP" -ge 3 ] && { say "CERT refs: three chunks without a new row: runner stops"; exit 3; }; sleep 60
  else NP=0; fi
done
# ------------------------------------------------ 1. the config on CERT's 364 rows, two orders
say "CERT: $(echo "$KEYS" | tr ',' '\n' | grep -c .) keys per order; QOM sets $Q1"
PHASE="CERT"; EXTRAENV=("R57_QOM_SETS=$Q1"); TAGP=es_ct_; IFECOPY=0; CHUNK=10
[ -s "$AF/es_ct_fwd1_modules.json" ] && [ -s results/cert_fwd/decode.jsonl ] && { REFMAN="$AF/es_ct_fwd1_modules.json"; REFMAN1=$REFMAN; say "restart: reference manifest $REFMAN"; }
order cert_fwd fwd "$KEYS"
order cert_rev rev "$RKEYS"
# ------------------------------------------------ 1b. the chat check: IFEval interleaved with code, 8 per process
PHASE="CERT chat check"; EXTRAENV=("R57_QOM_SETS=$Q1"); TAGP=es_cc_; CHUNK=8; REFMAN=""
order cert_chat fwd "$CKEYS"
# ------------------------------------------------ 2. long code: the 41 humanevalt rows, one order, tfwin
PHASE="CERT long code"; EXTRAENV=("R57_QOM_SETS=$Q2"); TAGP=es_cl_; CHUNK=10; REFMAN=""
order cert_ht fwd "$TKEYS"
tfwin cert_ht "CERT long code"
# ------------------------------------------------ the report
$PY tools/cert_report.py cert_fwd cert_rev cert_ht cert_chat > results/cert/CERT_report.log 2>&1
echo "$(tail -1 results/cert/CERT_VERDICT.txt 2>/dev/null || echo 'CERT: no verdict (results/cert/CERT_report.log)'); $(cat results/cert/CERT_chat.txt 2>/dev/null)" > results/CERT_DONE
say "the CERT verdict: $(cat results/CERT_DONE)"
peer say --to ojox-3f "EvalSuite CERT $(date +%H:%M): $(cut -c1-230 results/CERT_DONE) (results/cert/)" > /dev/null 2>&1
say "runner 55 end"
