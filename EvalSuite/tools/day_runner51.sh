#!/bin/bash
# EvalSuite runner 51, Fri 2026-11-06 (attached to a live evalsuite session): the long-code smoke (NewIdeas Fri 12:5x,
# QUEUE b6e90b5; docs/PREREG_relax_drift_2026-11-05.md amendment 3): tonight's CODE env (BigMoE's sat_night stack_env:
# QA + STACK, QOM on the code sets + humanevalt) on 6 humanevalt rows (HumanEval+ + a pytest module, max 1536), one
# order, one process on the LIVE tree (its module manifest is the provenance), then tfwin vs NAX6-N. The line: blunders
# > 1 per 10K -> tonight's top-up also runs the QOM-off CODE arm. Runner 48's gates. Stop it by pid only.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day51_runner.log
AF=results/agentfast
mkdir -p "$AF"
CUTOFF=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-06 15:00" +%s)
FULL_BY=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-06 12:30" +%s)
export EVALSUITE_DEADLINE=$CUTOFF
export PEER_QUEUE_TIMEOUT=21600
R3IDX=$HOME/BigMoE/models/bare/records3/index.json
BM=$HOME/BigMoE
BMC=fce0e55
REFMAN=results/agentfast/es_af_fwd5_modules.json   # the reference manifest: Leap 03a63c5, BigMoE fce0e55 (Fri 09:3x)
FZ=$HOME/EvalSuite/snap/r57frz-af1                 # tools/freeze_r57.py FZ REFMAN 03a63c5 fce0e55 (Fri 09:4x): the outside code, frozen
SCREEN=$FZ/BigMoE/scripts/r57_screen_fast.py
QOMSO=ff69fa24a745
KEYS="agentlong/edit00,agentlong/edit01,agentlong/edit02,agentlong/edit03,agentlong/edit04,agentlong/edit05,agentlong/edit06,agentlong/edit07,agentlong/edit08,agentlong/edit09,agentlong/edit10,agentlong/write11,agentlong/write12,agentlong/write13,agentlong/write14,agentlong/write15,agentlong/write16"
RKEYS=$(echo "$KEYS" | tr ',' '\n' | tail -r | paste -sd, -)
FASTENV=(R57_PRESET=45 R57_REPS=1 R57_WARM=1 R57_REFTOK= BIGMOE_EXT_BUILD=build_qom R57_FAST=QOM R57_D17=1 R57_D7B=code
  R57_HC8=1 REF2_HC8_FAST=1 R57_DNR=1 R57_HCB=1 REF2_SMALL_NATIVE=1 R57_D7BNAX=1 REF2_OOO_1C=1
  R57_QOM_SETS=humaneval,ifeval,multi,agentlong R57_NAX=1 REF2_NAX_1CQ=1)
UNSETS=(-u EVALSUITE_BIGMOE_CODE -u EVALSUITE_RELAX_VARIANT -u EVALSUITE_STAGE_ENV -u R57_INERT -u R57_NOCOM
  -u R57_QOM_SPLIT -u R57_RELAX_DELTA -u R57_RELAX_GATE -u R57_RELAX_SPAN -u R57_RELAX_V2 -u R57_RELAX_LOG -u R57_KEYS
  -u R57_TAG -u BIGMOE_TIER -u BIGMOE_TIER_GLU -u BIGMOE_TIER_CL -u BIGMOE_TIER_COMBINE -u BIGMOE_OOO -u BIGMOE_MAXK)
STAGEENV="BIGMOE_MAXK=160,BIGMOE_TIER=4433333333,BIGMOE_TIER_GLU=3c,BIGMOE_TIER_CL=112,BIGMOE_TIER_COMBINE=1,BIGMOE_OOO=1"
PY=.venv/bin/python
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
bm_pin() {      # BigMoE's scripts/*.py and bigmoe/ == 6c08636 (tracked), no untracked .py there, the build_qom .so
  git -C "$BM" diff --quiet "$BMC" -- 'scripts/*.py' bigmoe 2>/dev/null || return 1
  [ -z "$(git -C "$BM" status --porcelain -- 'scripts/*.py' bigmoe 2>/dev/null)" ] || return 1
  [ "$(shasum -a 256 "$BM"/ext/build_qom/*.so | cut -c1-12)" = "$QOMSO" ]
}
bm_loaded() {   # manifest: every module the FAST process loaded from ~/BigMoE equals its file at 6c08636 (else VOID)
  python3 - "$1" "$BM" "$BMC" <<'MAN'
import hashlib, json, subprocess, sys
mods, bm, c = json.load(open(sys.argv[1]))["modules"], sys.argv[2], sys.argv[3]
bad, n = [], 0
for name, (f, sha) in sorted(mods.items()):
    if not f.startswith(bm + "/") or f.endswith(".so"):
        continue
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
  { shasum snap/leap-52cf2c9/*.py snap/leap-52cf2c9/ms_gdn/* tools/agentfast.py tools/r57_wrap.py tools/r57rows.py \
      data/items/dec_agentlong.jsonl \
      data/items/dec_math.jsonl data/items/dec_ifeval.jsonl data/items/dec_humaneval.jsonl
    (cd snap/bigmoe-da8e7a6 && find bigmoe ext -type f ! -path '*__pycache__*' | sort | xargs shasum)
    (cd "$HOME/MoonShots/patches/planes_engine" && ls *.py | sort | xargs shasum)
    find evalsuite -name '*.py' ! -path '*__pycache__*' | sort | xargs shasum
    ls "$HOME/BigMoE/.venv/lib/python3.12/site-packages" | grep -E '^(mlx|mlx_lm|mlx_metal|mtplx)-.*dist-info$'; } 2>&1 \
    | shasum | cut -c1-12
}
ext_pin() {     # every external module of the reference manifest still has its sha (BigMoE, Leap, TrainLab, Breakthrough, MoonShots)
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
frz_pin() {     # the frozen tree unchanged (FREEZE.json's shas) and the live-but-pinned modules (MoonShots' tierglu) as REFMAN
  python3 - "$FZ/FREEZE.json" "$REFMAN" <<'FP'
import hashlib, json, sys
fz, ref = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))["modules"]
sh = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()[:12]
bad = [n for n, r in fz["modules"].items() if r.get("frozen", 1) is not None and sh(r["path"]) != r["sha"]]
bad += [n for n, (f, s) in ref.items() if "/MoonShots/" in f and sh(f) != s]
bad += [] if sh(fz["screen"]) == fz["screen_sha"] else ["screen"]
print("frozen tree / pinned live modules: " + ("OK" if not bad else f"CHANGED {bad}"))
sys.exit(1 if bad else 0)
FP
}
frz_loaded() {  # a process's manifest: every outside module from the frozen tree with FREEZE.json's sha, tierglu as REFMAN,
  python3 - "$FZ/FREEZE.json" "$REFMAN" "$1" <<'FL'   # and the same outside modules as REFMAN (none missing, none extra)
import json, sys
fz, ref, got = (json.load(open(a)) for a in sys.argv[1:4])
T = ("/BigMoE/", "/Leap/", "/TrainLab/", "/Breakthrough/", "/MoonShots/")
ext = lambda m: {k: v for k, v in m.items() if any(t in v[0] for t in T)}
r, g = ext(ref["modules"]), ext(got["modules"])
bad = sorted(set(r) ^ set(g))
for k, (f, s) in g.items():
    row = fz["modules"].get(k, {})
    if "/MoonShots/" in f:
        bad += [] if s == r.get(k, [None, None])[1] else [k]
    elif not f.startswith(fz["fz"] + "/") or s != row.get("sha"):
        bad.append(k)
print(f"outside modules loaded: {len(g)}, from the freeze as recorded: {len(g) - len(bad)}" + (f"; NOT: {bad}" if bad else ""))
sys.exit(1 if bad else 0)
FL
}
gate() {        # label, MEM GB, seconds the step is estimated / measured to take: every check before a GPU step
  local w=0
  calm
  while mem_busy; do
    [ $((w % 10)) = 0 ] && say "$1: another session's MEM job holds a shared lock: waiting (day rule)"
    w=$((w + 1)); sleep 60; calm
  done
  [ "$(pin)" = "$PIN0" ] || { say "code changed: pin $PIN0 -> $(pin): runner stops"; exit 7; }
  frz_pin >> "$LOG" 2>&1 || { say "the frozen tree or a pinned live module changed: runner stops"; exit 7; }
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
EST_FAST=900; EST_TF=3300
fast() {        # tag, keys, label: one FAST process (all its keys), under this runner's share lock
  local tag=$1 keys=$2 t0 d rc
  gate "FAST $3" 39 "$EST_FAST"                    # BigMoE's C 224 FAST-NAX screens: lifetime peaks 38.4-38.5 GB
  rm -f "$BM/data/r57/${tag}_"*
  share_lock "evalsuite: MEM 39 GB: agent check, plain FAST $3" 90
  t0=$(date +%s)
  env "${UNSETS[@]}" "${FASTENV[@]}" R57_TAG="$tag" R57_KEYS="$keys" \
    EVALSUITE_BIGMOE_CODE="$FZ/BigMoE" "$BM/.venv/bin/python" tools/r57_wrap.py "$SCREEN" "$AF/${tag}_modules.json" 224 1024 \
    > "$AF/${tag}.log" 2>&1; rc=$?
  peer unlock > /dev/null 2>&1
  d=$(( $(date +%s) - t0 ))
  say "FAST $3: rc=$rc ($((d / 60)) min)"
  cp "$BM/data/r57/${tag}_"* "$AF/" 2>/dev/null
  [ "$rc" = 0 ] && [ -s "$AF/${tag}_tokens.json" ] || { say "FAST $3 failed (results/agentfast/${tag}.log)"; return 1; }
  frz_loaded "$AF/${tag}_modules.json" >> "$LOG" 2>&1 || { say "FAST $3: an outside module did not come from the freeze: VOID: runner stops"; exit 5; }
  python3 - "$AF/${tag}_runs.json" "$AF/${tag}_modules.json" <<'ENG' >> "$LOG" 2>&1 || { say "FAST $3: not engaged (QOM / draft17): VOID: runner stops"; exit 5; }
import json, sys
runs = [r for r in json.load(open(sys.argv[1])) if str(r.get("key", "")).startswith("agentlong/")]
mods = json.load(open(sys.argv[2]))["modules"]
qom = [r["key"] for r in runs if (r.get("pf") or {}).get("qom_opt", 0) > 0]
print(f"engaged: QOM optional uses on {len(qom)} of {len(runs)} agentlong runs; draft17 loaded: {'draft17' in mods}; "
      f"types {sorted({r.get('type') for r in runs})}")
sys.exit(0 if runs and len(qom) == len(runs) and "draft17" in mods else 1)
ENG
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
SMOKEENV=(BIGMOE_EXT_BUILD=build_qom R57_REFTOK= R57_FAST=QOM R57_D17=1 R57_D7B=code R57_HC8=1 REF2_HC8_FAST=1 R57_DNR=1
  R57_HCB=1 REF2_SMALL_NATIVE=1 R57_D7BNAX=1 REF2_OOO_1C=1 R57_NAX=1 REF2_NAX_1CQ=1 REF2_DN_KS=1 REF2_GDN_LAZY=1 D7BNAX_32=1
  R57_INERT=CD16 R57_FASTTAX=1 R57_QOM_SETS=humaneval,mbpp,humanevalt)
SKEYS="humanevalt/HumanEval/0,humanevalt/HumanEval/4,humanevalt/HumanEval/8,humanevalt/HumanEval/12,humanevalt/HumanEval/16,humanevalt/HumanEval/20"
RUN=topsmoke_stack
TAG=es_ts_stack
grep -q '"humanevalt"' "$BM/scripts/r57_screen_fast.py" || { say "BigMoE's screen has no humanevalt in SETOF: runner stops"; exit 2; }
PIN0=$(pin)
echo $$ > results/q51.pid; trap 'rm -f results/q51.pid' EXIT
say "runner 51 start (pid $$); code pin $PIN0; BigMoE $(git -C "$BM" rev-parse --short HEAD), Leap $(git -C "$HOME/Leap" rev-parse --short HEAD); the live tree"
if [ "$(grep -c . "results/$RUN/decode.jsonl" 2>/dev/null || echo 0)" -lt 6 ]; then
  w=0
  calm
  while mem_busy; do [ $((w % 10)) = 0 ] && say "another session's MEM job holds a shared lock: waiting (day rule)"; w=$((w + 1)); sleep 60; calm; done
  [ "$(pin)" = "$PIN0" ] || { say "code changed: runner stops"; exit 7; }
  store_ok || { say "records3 is not fully converted: runner stops"; exit 8; }
  [ $(( $(date +%s) + 1800 )) -le "$CUTOFF" ] || { say "the smoke does not fit before 15:00: runner stops"; exit 6; }
  until [ "$(freegb)" -ge 50 ]; do say "waiting for memory: $(freegb) GB NO-SWAP (needs 50)"; sleep 60; calm; done
  share_lock "evalsuite: MEM 42 GB: the long-code smoke (tonight's CODE env, 6 humanevalt)" 30
  t0=$(date +%s)
  env "${UNSETS[@]}" "${SMOKEENV[@]}" R57_PRESET=45 R57_REPS=1 R57_WARM=1 R57_TAG="$TAG" R57_KEYS="$SKEYS" \
    "$BM/.venv/bin/python" tools/r57_wrap.py "$BM/scripts/r57_screen_fast.py" "$AF/${TAG}_modules.json" 224 1536 \
    > "$AF/${TAG}.log" 2>&1; rc=$?
  peer unlock > /dev/null 2>&1
  say "FAST smoke: rc=$rc ($(( ($(date +%s) - t0) / 60 )) min)"
  cp "$BM/data/r57/${TAG}_"* "$AF/" 2>/dev/null
  [ -s "$AF/${TAG}_tokens.json" ] || { say "no tokens (results/agentfast/${TAG}.log): runner stops"; exit 4; }
  python3 - "$AF/${TAG}_runs.json" <<'ENG' >> "$LOG" 2>&1 || { say "QOM not engaged on the humanevalt rows: VOID: runner stops"; exit 5; }
import json, sys
runs = [r for r in json.load(open(sys.argv[1])) if str(r.get("key", "")).startswith("humanevalt/")]
qom = [r["key"] for r in runs if (r.get("pf") or {}).get("qom_opt", 0) > 0]
print(f"engaged: QOM optional uses on {len(qom)} of {len(runs)} humanevalt runs; inert {sum(1 for r in runs if r.get('inert'))} runs; "
      f"types {sorted({str(r.get('type')) for r in runs})}")
sys.exit(0 if runs and len(qom) == len(runs) else 1)
ENG
  $PY tools/r57rows.py "$AF/${TAG}_tokens.json" "$RUN" >> "$LOG" 2>&1
fi
tfwin "$RUN" "smoke"
python3 - "$RUN" <<'REP' > results/topsmoke_report.txt 2>&1
import gzip, json, math, os, sys
run = sys.argv[1]
rows = {r["key"]: r for r in map(json.loads, open(f"results/{run}/decode.jsonl"))}
tok = dev = bl = 0
for k in sorted(rows):
    s = json.load(gzip.open(f"results/{run}/tfwin/{k.replace('/', '__')}.json.gz", "rt"))
    out = rows[k]["completion_ids"]
    d = sum(a != b for a, b in zip(s["am"], out)); b_ = sum(lp < math.log(0.01) for lp in s["lp"])
    tok += len(out); dev += d; bl += b_
    print(f"{k}: {len(out)} tokens ({rows[k]['finish']}), {d} not the reference's argmax ({100 * d / len(out):.2f} %), {b_} blunder(s)")
per = 1e4 * bl / tok
print(f"\nTOPSMOKE: {bl} blunders in {tok} tokens = {per:.2f} per 10K; deviations {100 * dev / tok:.2f} % -> "
      + ("> 1/10K: tonight's top-up ALSO runs the QOM-off CODE arm" if per > 1 else "<= 1/10K: no extra arm"))
REP
say "$(tail -1 results/topsmoke_report.txt)"
touch results/TOPSMOKE_DONE
say "runner 51 end"
