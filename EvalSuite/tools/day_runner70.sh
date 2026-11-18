#!/bin/bash
# EvalSuite runner 70 = runner 69 (below, unchanged) + CARD_AR=1: a re-certification's rows (docs/PREREG_refc_2026-11-18.md,
# Part A; BigMoE 11:59 Wed: EvalSuite runs them on BigMoE's r57 screen): CARD_KEYSFILE's decode keys (data/items), as given,
# in ONE order, under the share lock, with no verdict (the PREREG's own tools grade the rows). CARD_TIER=1, CARD_SCOPE=all,
# the forward order; CARD_C sets C (Part A: 192, as NAX6's AR). The run is card_<tag>_ar; every process's runs.json goes
# into results/<run>/engage_runs.txt (the contract's per-run checks read it); DONE: 'AR ROWS <tag>: ...'. Its logs and
# pid file are day70_ / q70_, so it never collides with runner 69's.
# EvalSuite runner 69 = runner 68 with the card tier for the lossy stack (PREREG_lossygate, the stack's amendment; the
# orchestrator Tue 06:4x / 06:5x: the FULL card gate, lcb6 in every lever's sets). Runner 68 is unchanged for tiers 0 / 1;
# this file adds, and its logs are results/day70_*:
#   CARD_TIER=2, the card gate (LOSSY_GATE section 4; docs/PREREG_card_gates_2026-11-08.md's lines and its amendment 1):
#   - The keys: tools/card_keys.json's gate keys (IFBench 150, GPQA Diamond 100, LCB v6 131) of the sets in CARD_SETS
#     (default all three), or every key of CARD_KEYSFILE; with CARD_EXTEND="<sets>" (and CARD_RESUME=1) also those sets'
#     "extend" keys, which the reference must hold already (runner 61's ref stage).
#   - Both prompt orders in one call, as runner 61: forward = the keys sorted, then reverse, in their own runs
#     card_<tag>_t2<scope>_fwd / _rev; 20 prompts per process; the screen's CAP, each row cut to its item's max_tokens.
#   - The verdict, per benchmark and order vs CARD_REF (NAX6-N: ref2tnaxn.r6): tools/set_d4.py's decision 4. A benchmark
#     PASSES only in both orders; the gate PASSES when every judged benchmark does. Amendment 1: exactly one order failing
#     on the delta alone (McNemar p >= 0.05) is PENDING (extend that set: CARD_EXTEND); any other failure is a FAIL.
#   - CARD_IDENTSETS="<sets>" with CARD_IDENT_REF="<base fwd run>+<base rev run>": those sets are judged by token identity
#     with the base's same-order rows instead (the switch-off check), on every key of the set.
#   - In the code scope the chat sets are IFBench and GPQA: each engagement term must read 0 there; LCB v6 is the code
#     side (as IFEval and the code sets are in tier 1).
# EvalSuite runner 68 = runner 67 with two changes for res1c's envs (PREREG_lossygate 5d, dated Tue 01:2x; NewIdeas 01:15):
#   1. QOM's validity rule under 1c (amendment 4b) with res1c: on the sets in both R57_QOM_SETS and R57_TIERS_SETS,
#      pf.qom_opt must EQUAL lossy_check.opt_post (res1c's RES / COMQOM optional uses; QOM's own keep stays replaced by
#      1c). With res1c unset, opt_post is absent, so the rule is runner 65's "== 0" exactly. res_opt + com_opt is printed
#      beside opt_post per process (reported, never ruled).
#   2. Whole-run engagement terms: a CARD_ENGAGE term ending "@run" must be > 0 summed over the run's kept code answers
#      (its processes listed in results/<run>/engage_runs.txt, resumes included; the latest record of a key counts) and
#      0 on chat. Checked after each process (the EARLY check: once the run holds >= 32 kept code answers, a sum still
#      at 0 is VOID) and before every verdict (the FULL check). Each process still checks the chat side of every term;
#      an unmarked term keeps runner 67's rule (> 0 on every kept code run). A new file, because runner 67 is
#      registered. Its logs are results/day68_*.
# EvalSuite runner 67 = runner 66 (runner 65 + CARD_GATE) with one change, the engagement check takes several terms
# (PREREG_lossygate amendment 5d; G1 = 1c + int8 dense, G2 adds more): CARD_ENGAGE is a comma list of field[:key+key]
# terms and EACH must hold (> 0 on every kept code run, 0 on chat), so a dead mechanism cannot hide behind a live one in
# a sum. One term is runner 66's check exactly. A new file, because runners 65 / 66 serve armed or registered chains.
# Its logs are results/day67_*.
# EvalSuite runner 66 = runner 65 with one change, the CHAT step's NO-SWAP gate (PREREG_lossygate amendment 4e, the
# orchestrator 11-16 11:3x fallback): CARD_GATE (whole GB, 41-49), inherited from the caller's environment, replaces
# MEM + 4. Unset, it is MEM + 4 as in runner 65. MEM (the share lock's "MEM <n> GB") and every other check and stop
# are runner 65's. A new file, because runner 65 serves armed chains. Its logs are results/day66_*.
# EvalSuite runner 65 = runner 64 with one change, QOM's validity rule under 1c (PREREG_lossygate amendment 4b): with
# R57_TIERS set in the env, the runs of the sets in both R57_QOM_SETS and R57_TIERS_SETS need pf.qom_opt == 0 (1c replaced
# QOM's keep in the check, never stacked on it); every other QOM set keeps pf.qom_opt > 0. A new file, because runner 64
# serves armed chains. Its logs are results/day65_*.
# EvalSuite runner 64 (attached to a live evalsuite session): the lossy-check gate (~/FableGod/deepdraft/LOSSY_GATE.md,
# docs/PREREG_lossygate_*), runner 63's code checks, manifests, gates and yields with the panel of tools/lossy_panel.json:
#   CARD_TIER=1 (default): the 364-key tier-1 panel (HumanEval+ 164, MBPP 100, IFEval 100) in its round-robin order; after
#     the first 182 keys, tools/lossy_d4.py interim (STOP-FAIL at pooled z <= -2.8); then the rest and lossy_d4.py tier1.
#   CARD_TIER=0: the 40-key mini-panel, then lossy_d4.py tier0 (KILL / GO).
# CARD_REF = the reference run(s), a comma list merged in order (exact: ref2tnax,ref2tnaxn.r3,ref2tnaxn.r2; or a cached
# base run). CARD_ENGAGE = the runs.json field that proves the variant engaged (> 0 on every kept run; empty for a base
# arm); for a dict field, field:key+key sums only those keys (Breakthrough's lossy_check:e3+edrop+lmsl_rows).
# CARD_REPLAY (tier 0 only): exact = the exact config's REPLAY self-check, writing the merged margins; <margins.json> =
# a variant's REPLAY pass (A, F_hi), run before the mini-panel under its own run name (<run>_rpl). CARD_ENGAGE_MODE=any
# (accept rules): the counter must be > 0 summed over each process's code runs (and 0 on chat) instead of on every run.
# CARD_REPLAY_ROWS = the
# run(s) whose rows are replayed (default CARD_REF): the exact config's OWN cached tier-0 run, for every pass. CARD_FBLOB = the env file's registered git blob. CARD_TREES = "path=tree,..." (BigMoE HEAD tree hashes) and
# CARD_SCREEN_SHA / CARD_SO_SHA (12 hex) pin the registered code at the start. CARD_MEM overrides the comment's MEM.
# CARD_RESUME=1 continues; CARD_CUTOFF = the window's end (YYYY-mm-dd HH:MM); dry run CARD_DRY=1. -> results/CARD_<tag>_DONE
# Stop it by pid only, and its children too; then peer unlock.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
TAG=${CARD_TAG:?set CARD_TAG (the arm tag named in the env file comment, or ref for the reference stage)}
ORDER=${CARD_ORDER:-fwd}; case "$ORDER" in fwd|rev) ;; *) echo "CARD_ORDER is fwd or rev"; exit 2;; esac
OSUF=""; [ "$ORDER" = rev ] && OSUF=_rev; TG=${TAG}${OSUF}   # amendment 1c: the reverse order gets its own run and files
LOG=results/day70_${TG}_runner.log
AF=results/card
mkdir -p "$AF"
DONE=results/CARD_${TG}_DONE
TIER=${CARD_TIER:-1}; case "$TIER" in 0|1|2) ;; *) echo "CARD_TIER is 0, 1 or 2"; exit 2;; esac
[ "$TIER" = 2 ] && [ "$ORDER" = rev ] && { echo "CARD_TIER=2 runs both orders in one call: leave CARD_ORDER unset"; exit 2; }
SCOPE=${CARD_SCOPE:-all}; case "$SCOPE" in all|code) ;; *) echo "CARD_SCOPE is all or code"; exit 2;; esac   # code = CODE-ONLY mode
IDENT=""; [ "${CARD_IFEVAL_IDENT:-0}" = 1 ] && IDENT=ident   # code scope vs a cached base run: IFEval tokens must be identical
ENGAGE=${CARD_ENGAGE:-}; REFS=${CARD_REF:?set CARD_REF to the reference run(s), a comma list}; REFRUN=$REFS
RUNTERMS=$(echo "$ENGAGE" | tr ',' '\n' | grep '@run$' | paste -sd, -)   # runner 68: the whole-run terms
RUNF=card_${TAG}_t${TIER}${SCOPE/all/}${OSUF}; RUNC=$RUNF; RUNR=""
CHATSETS=ifeval; [ "$TIER" = 2 ] && CHATSETS=ifbench,gpqad   # runner 69: the code scope's chat (switch-off) sets
if [ "$TIER" = 2 ]; then RUNF=card_${TAG}_t2${SCOPE/all/}_fwd; RUNR=card_${TAG}_t2${SCOPE/all/}_rev; RUNC=$RUNR; fi
if [ -n "${CARD_RUNSUF:-}" ]; then   # runner 69: a tier-2 run that is not the gate's (the stack's smoke) gets its own names
  [ "$TIER" = 2 ] || { echo "CARD_RUNSUF is for CARD_TIER=2 only"; exit 2; }
  echo "$CARD_RUNSUF" | grep -qE '^_[a-z0-9]+$' || { echo "CARD_RUNSUF is _ and lowercase letters or digits"; exit 2; }
  TG=${TAG}${CARD_RUNSUF}; LOG=results/day70_${TG}_runner.log; DONE=results/CARD_${TG}_DONE
  RUNF=card_${TG}_t2${SCOPE/all/}_fwd; RUNR=card_${TG}_t2${SCOPE/all/}_rev; RUNC=$RUNR
fi
TIMING=${CARD_TIMING:-0}   # runner 69: a timing run for a headline (tier 1's code keys, one order, a quiet lock; no verdict)
if [ "$TIMING" = 1 ]; then
  [ "$TIER" = 1 ] && [ "$SCOPE" = code ] && [ "$ORDER" = fwd ] \
    || { echo "CARD_TIMING=1 needs CARD_TIER=1, CARD_SCOPE=code and the forward order"; exit 2; }
  TG=${TAG}_timing; LOG=results/day70_${TG}_runner.log; DONE=results/CARD_${TG}_DONE; RUNF=card_${TAG}_timing; RUNC=$RUNF
fi
AR=${CARD_AR:-0}   # runner 70: a re-certification's rows (the header)
if [ "$AR" = 1 ]; then
  [ "$TIER" = 1 ] && [ "$SCOPE" = all ] && [ "$ORDER" = fwd ] && [ "$TIMING" != 1 ] && [ -s "${CARD_KEYSFILE:-}" ] \
    || { echo "CARD_AR=1 needs CARD_TIER=1, CARD_SCOPE=all, the forward order, no CARD_TIMING, and CARD_KEYSFILE"; exit 2; }
  TG=${TAG}_ar; LOG=results/day70_${TG}_runner.log; DONE=results/CARD_${TG}_DONE; RUNF=card_${TAG}_ar; RUNC=$RUNF
fi
TLED=results/$RUNF/timing_procs.txt   # the timing run's process windows: tag, start, end (epoch s)
if [ "$TIER" = 2 ]; then
  case "$REFS" in *,*) echo "CARD_TIER=2 takes one reference run (NAX6-N: ref2tnaxn.r6), not a comma list"; exit 2;; esac
  [ -z "${CARD_IDENTSETS:-}" ] || echo "${CARD_IDENT_REF:-}" | grep -qE '^[^+,]+\+[^+,]+$' \
    || { echo "CARD_IDENTSETS needs CARD_IDENT_REF=<base fwd run>+<base rev run>"; exit 2; }
fi
STAGE=${CARD_STAGE:-arm}; PERPROC=20; CAP=2048; NK=364; [ "$TIER" = 0 ] && NK=40; [ "$TIER" = 2 ] && NK=0   # tier 2: set below
KF=$AF/lossy_keys_${TG}_t${TIER}${SCOPE}.txt   # tools/lossy_panel.json: the tier's keys in the panel's round-robin order
EXT=""; [ "$TIER" = 2 ] && EXT=$(echo "${CARD_EXTEND:-}" | tr ',' ' ')   # runner 69: amendment 1's extension, tier 2 only
IDS=""; [ "$TIER" = 2 ] && IDS=$(echo "${CARD_IDENTSETS:-}" | tr ',' ' ')   # runner 69: sets judged by identity with the base
XF=$AF/card_ext_keys_${TG}.txt; RKF=$AF/lossy_ref_keys_${TG}_t${TIER}${SCOPE}.txt; RNK=$NK   # the reference's keys: its own file (01:3x: card_ref_keys.txt is the card gate's; runner 64 had overwritten it)
EARLIEST=0   # runner 64: env files may predate the registration; CARD_FBLOB pins them instead
CUTOFF=$(date -j -f "%Y-%m-%d %H:%M:%S" "${CARD_CUTOFF:?set CARD_CUTOFF to the end of the GPU window, YYYY-mm-dd HH:MM}:00" +%s) \
  || { echo "CARD_CUTOFF does not parse as YYYY-mm-dd HH:MM"; exit 2; }
export EVALSUITE_DEADLINE=$((CUTOFF + 1800))
export PEER_QUEUE_TIMEOUT=21600
R3IDX=$HOME/BigMoE/models/bare/records3/index.json
BM=$HOME/BigMoE
FENV=${CARD_FENV:-}   # the arm stage needs it (checked at the start); never an apostrophe inside ${} text
DRY=${CARD_DRY:-0}
[ "$DRY" = 1 ] && LOG=results/day70_dry_${TG}.log
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
  BMPIN=1 gate "CHAT $3" "$((GATE - 8))" "${STEP:-900}"     # needs GATE GB NO-SWAP free (runner 65: MEM + 4)
  rm -f "$BM/data/r57/${tag}_"*
  [ "$TIMING" = 1 ] || share_lock "evalsuite: MEM $MEM GB: $PHASE, $3" "$(( ${STEP:-900} / 30 ))"   # the lease: 2x the step (30 min at HARD 600); runner 69's timing run holds its quiet lock instead
  t0=$(date +%s)
  env "${UNSETS[@]}" "${STACKENV[@]}" "${EXTRAENV[@]}" R57_PRESET=45 R57_REPS=1 R57_WARM="${WARM:-1}" R57_TAG="$tag" R57_KEYS="$keys" \
    "$BM/.venv/bin/python" tools/r57_wrap.py "$SCREEN" "$AF/${tag}_modules.json" "$C" "$CAP" > "$AF/${tag}.log" 2>&1; rc=$?
  [ "$TIMING" = 1 ] || peer unlock > /dev/null 2>&1
  [ "$TIMING" = 1 ] && mkdir -p "results/$RUNF" && echo "$tag $t0 $(date +%s)" >> "$TLED"   # runner 69: the process's window
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
  QSETS="$QS" TSETS="$TS" python3 - "$AF/${tag}_runs.json" <<'ENG' >> "$LOG" 2>&1 || void "CHAT $3: QOM not engaged on its sets, or engaged where 1c replaces it (amendment 4b)"
import json, os, sys
runs = json.load(open(sys.argv[1]))
S = {s for s in os.environ.get("QSETS", "").split(",") if s}     # the env's QOM sets (chat ones included for chat candidates)
T = {s for s in os.environ.get("TSETS", "").split(",") if s}     # amendment 4b: 1c's sets (R57_TIERS set); there 1c replaces QOM
cut = [r["key"] for r in runs if r.get("cut")]      # cut at the screen's time budget: not rows (r57rows leaves them out; Mon 11-09)
def of(Z):
    return [r for r in runs if str(r.get("key", "")).split("/")[0] in Z and not r.get("cut")]
def qo(r):                                          # runs.json "pf": the expert store's stats over the prompt (not the prefill)
    return (r.get("pf") or {}).get("qom_opt", 0) or 0
def lc(r, k):                                       # res1c's lossy_check counters (absent with res1c unset: 0)
    return (r.get("lossy_check") or {}).get(k, 0) or 0
code, rep = of(S - T), of(S & T)
qom = [r["key"] for r in code if qo(r) > 0]
stk = [r["key"] for r in rep if qo(r) != lc(r, "opt_post")]   # runner 68: QOM's optional uses where 1c replaced it must all be
att = [r["key"] for r in rep if lc(r, "res_opt") + lc(r, "com_opt") != lc(r, "opt_post")]   # res1c's (opt_post); att: reported
note = ((f"; 1c in place of QOM on {sorted(S & T)}: pf.qom_opt == opt_post (res1c's; 0 without res1c) on "
         f"{len(rep) - len(stk)} of {len(rep)} runs (must be all)" + (f", NOT on {' '.join(stk[:5])}" if stk else "")
         + f"; reported: opt_post {sum(lc(r, 'opt_post') for r in rep)}, res_opt + com_opt "
         f"{sum(lc(r, 'res_opt') + lc(r, 'com_opt') for r in rep)} ({sum(lc(r, 'res_opt') for r in rep)} + "
         f"{sum(lc(r, 'com_opt') for r in rep)}), equal on {len(rep) - len(att)} of {len(rep)} runs"
         + (f" (differs on {' '.join(att[:5])})" if att else "")) if S & T else "")
print(f"engaged: QOM optional uses on {len(qom)} of {len(code)} runs in the QOM sets {sorted(S - T)}; types "
      f"{sorted({str(r.get('type')) for r in runs})}; inert {sum(1 for r in runs if r.get('inert'))}, relaxed "
      f"{sum(1 for r in runs if r.get('rlx'))}, span {sum(1 for r in runs if r.get('spn'))} runs" + note
      + (f"; cut at the time budget (not rows): {' '.join(cut)}" if cut else ""))
sys.exit(0 if len(qom) == len(code) and not stk else 1)
ENG
  ENGAGE="$ENGAGE" ENGMODE="${CARD_ENGAGE_MODE:-each}" PANEL="$KF" SCOPE="$SCOPE" CHATSETS="$CHATSETS" python3 - "$AF/${tag}_runs.json" <<'ENG2' >> "$LOG" 2>&1 || void "LOSSY $3: the variant is not engaged as registered (runs.json ${ENGAGE:-}, ${CARD_ENGAGE_MODE:-each})"
import json, os, sys
runs = json.load(open(sys.argv[1]))
panel, eng = set(open(os.environ["PANEL"]).read().split()), os.environ.get("ENGAGE", "")
terms = []                                              # runner 67: comma-separated field[:key+key] terms; EACH must hold
for t_ in [x for x in eng.split(",") if x]:
    w_ = t_.endswith("@run")                            # runner 68: a whole-run term (checked over the run, not per process)
    f_, _, s_ = (t_[:-4] if w_ else t_).partition(":")  # field, or field:key+key (a dict field's counts that prove the lossy path)
    terms.append((t_, f_, [k for k in s_.split("+") if k], w_))
kept = [r for r in runs if not r.get("cut")]
outside = [r["key"] for r in kept if r["key"] not in panel]
def val(r, field, sub):                                 # a number, or a dict of counts: the listed keys summed (all keys if none listed)
    v = r.get(field) or 0
    if isinstance(v, dict):
        return sum(v.get(k) or 0 for k in sub) if sub else sum(x for x in v.values() if isinstance(x, (int, float)))
    return v
CS = set(os.environ.get("CHATSETS", "ifeval").split(","))   # runner 69: tier 2's chat sets are IFBench and GPQA
chat = lambda k: os.environ.get("SCOPE") == "code" and k.split("/")[0] in CS   # CODE-ONLY mode: chat runs the base path
off, lit = [], []
for t_, field, sub, w_ in terms:
    o_ = [r["key"] for r in kept if not chat(r["key"]) and not val(r, field, sub) > 0]
    if w_:                                              # runner 68: the code side is the run's (the EARLY / FULL checks)
        print(f"  term {t_}: this process {sum(val(r, field, sub) for r in kept if not chat(r['key']))} over "
              f"{sum(1 for r in kept if not chat(r['key']))} code runs (at 0 on {len(o_)}); whole-run sum checked by the runner")
        o_ = []
    elif os.environ.get("ENGMODE") == "any":            # accept rules (LOSSY_GATE section 11): engaged where a draft qualifies,
        codes = [r for r in kept if not chat(r["key"])] # so > 0 summed over this process's code runs, not on every one
        tot = sum(val(r, field, sub) for r in codes)
        print(f"engagement mode any, {t_}: {tot} over {len(codes)} code runs; runs at 0: {len(o_)}")
        o_ = [] if not codes or tot > 0 else [r["key"] for r in codes]
    l_ = [r["key"] for r in kept if chat(r["key"]) and val(r, field, sub) > 0]
    if len(terms) > 1:
        print(f"  term {t_}: not engaged {o_}; engaged on chat (must be off) {l_}")
    off += [k for k in o_ if k not in off]; lit += [k for k in l_ if k not in lit]
field = eng                                             # the summary line names the whole check
print(f"engagement ({field or 'none: a base arm'}, scope {os.environ.get('SCOPE')}): {len(kept) - len(off) - len(lit)} of {len(kept)} kept runs as expected; "
      f"not engaged {off}; engaged on chat (must be off) {lit}; outside the panel {outside}")
sys.exit(1 if off or lit or outside else 0)
ENG2
  return $rc
}
runterms() {    # run, early|full: runner 68's whole-run terms (CARD_ENGAGE's "@run" ones) over the run's kept code answers
  [ -n "$RUNTERMS" ] || return 0   # so far: each key's latest record in the runs.json files of results/<run>/engage_runs.txt,
  ENGAGE="$RUNTERMS" SCOPE="$SCOPE" WHEN="$2" CHATSETS="$CHATSETS" python3 - "$1" <<'RT' >> "$LOG" 2>&1 \
    || void "LOSSY: a whole-run term is not engaged ($2 check: $(grep '^whole-run' "$LOG" | tail -1 | cut -c1-160))"
import json, os, sys                                     # keys in the run's decode.jsonl only; 0 on chat is each process's check
run, when = sys.argv[1], os.environ["WHEN"]
CS = set(os.environ.get("CHATSETS", "ifeval").split(","))   # runner 69: tier 2's chat sets are IFBench and GPQA
chat = lambda k: os.environ.get("SCOPE") == "code" and k.split("/")[0] in CS
rows = {json.loads(l)["key"] for l in open(f"results/{run}/decode.jsonl")} if os.path.exists(f"results/{run}/decode.jsonl") else set()
led = f"results/{run}/engage_runs.txt"
files = [p for p in open(led).read().split() if p] if os.path.exists(led) else []
rec = {}
for p in files:                                         # later processes win: a key cut in one process and kept in a later one
    for r in json.load(open(p)):
        if not r.get("cut"):
            rec[r["key"]] = r
codes = [k for k in rec if k in rows and not chat(k)]
def val(r, field, sub):
    v = r.get(field) or 0
    if isinstance(v, dict):
        return sum(v.get(k) or 0 for k in sub) if sub else sum(x for x in v.values() if isinstance(x, (int, float)))
    return v
sums = {}
for t_ in [x for x in os.environ["ENGAGE"].split(",") if x]:
    f_, _, s_ = t_[:-4].partition(":")
    sums[t_] = sum(val(rec[k], f_, [x for x in s_.split("+") if x]) for k in codes)
zero = [t_ for t_, s in sums.items() if not s > 0]
decide = when == "full" or len(codes) >= 32             # EARLY: from 32 kept code answers on; FULL: before every verdict
print(f"whole-run terms ({when}, {len(files)} processes, {len(codes)} kept code answers): "
      + "; ".join(f"{t_} {s}" for t_, s in sums.items())
      + (f"; at 0: {zero}" if zero else "") + ("" if decide else " (not decided before 32 kept code answers)"))
sys.exit(1 if decide and zero else 0)
RT
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
    if { [ -n "$RUNTERMS" ] || [ "$AR" = 1 ]; } && { [ "$run" = "$RUNF" ] || [ "$run" = "$RUNR" ]; } && [ -s "$AF/${tag}_runs.json" ] && [ -d "results/$run" ]; then   # runner 68: the ledger (69: both tier-2 runs)
      echo "$AF/${tag}_runs.json" >> "results/$run/engage_runs.txt"; runterms "$run" early
    fi
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
refcount() {    # keys of the file (default: the panel) present in the union of the reference runs (CARD_REF, comma list)
  python3 - "$REFS" "${1:-$RKF}" <<'RC'
import json, os, sys
have = set()
for run in sys.argv[1].split(","):
    p = f"results/{run}/decode.jsonl"
    if os.path.exists(p):
        have |= {json.loads(l)["key"] for l in open(p)}
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
if [ "$TIER" = 2 ]; then   # runner 69: the card gate's keys (tools/card_keys.json); the extension keys also apart, in XF
  python3 - "$KF" "$XF" "${CARD_SETS:-ifbench,gpqad,lcb6}" "${CARD_KEYSFILE:-}" "$EXT" "$IDS" <<'PK2' >> "$LOG" 2>&1 \
    || { say "the card keys do not build (CARD_SETS / CARD_KEYSFILE / CARD_EXTEND / CARD_IDENTSETS): runner stops"; exit 2; }
import json, sys
kf, xf, sets, kfile, ext, ids = sys.argv[1], sys.argv[2], [s for s in sys.argv[3].split(",") if s], sys.argv[4], sys.argv[5].split(), sys.argv[6].split()
d = json.load(open("tools/card_keys.json"))
assert sets and all(s in d["keys"] for s in sets), f"CARD_SETS {sets}: the card sets are {sorted(d['keys'])}"
assert all(s in sets for s in ext + ids), f"CARD_EXTEND {ext} / CARD_IDENTSETS {ids} name a set outside CARD_SETS {sets}"
assert not set(ext) & set(ids), "a set judged by identity is never extended"
if kfile:                                               # an explicit, registered key list (e.g. a spot check's subset)
    ks = [k for k in open(kfile).read().split() if k]
    allk = {k for s in d["keys"] for k in d["keys"][s]}
    assert set(ks) <= allk, f"CARD_KEYSFILE: {len(set(ks) - allk)} keys are not gate keys"
    assert {k.split("/")[0] for k in ks} == set(sets), "CARD_KEYSFILE's sets are not CARD_SETS"
else:
    ks = [k for s in sets for k in d["keys"][s]]
xs = [k for s in ext for k in d["extend"][s]]
assert len(ks) == len(set(ks)) and not set(ks) & set(xs), "duplicate keys"
open(kf, "w").write("\n".join(ks + xs) + "\n")          # the panel (the engagement check's PANEL) = the reference's keys
open(xf, "w").write("\n".join(xs) + ("\n" if xs else ""))
print(f"card keys: {len(ks)} gate keys of {sets}" + (f" ({kfile})" if kfile else "") + (f"; + {len(xs)} extension keys of {ext}" if xs else "")
      + (f"; judged by identity: {ids}" if ids else ""))
PK2
  NK=$(sort -u "$KF" | grep -c .); RNK=$NK
elif [ "$AR" = 1 ]; then   # runner 70: CARD_KEYSFILE as given (decode items of data/items), in its own order
python3 - "$KF" "$CARD_KEYSFILE" <<'PKA' >> "$LOG" 2>&1 || { say "CARD_KEYSFILE does not build (keys outside data/items): runner stops"; exit 2; }
import glob, json, sys
ks = [k for k in open(sys.argv[2]).read().split() if k]
have = {json.loads(l)["key"] for f in glob.glob("data/items/dec_*.jsonl") for l in open(f)}
assert ks and len(ks) == len(set(ks)), "empty, or duplicate keys"
assert set(ks) <= have, f"{len(set(ks) - have)} keys are not decode items, e.g. {sorted(set(ks) - have)[:3]}"
open(sys.argv[1], "w").write("\n".join(ks) + "\n")
print(f"AR keys: {len(ks)} from {sys.argv[2]}")
PKA
  NK=$(grep -c . "$KF"); RNK=$NK
else
python3 - "$KF" "$TIER" "$SCOPE" <<'PK' || { say "tools/lossy_panel.json does not build: runner stops"; exit 2; }
import json, sys
d = json.load(open("tools/lossy_panel.json"))
t, sc = sys.argv[2], sys.argv[3]
ks = (d["code"] + d["ifeval"] if sc == "code" else d["tier1"]) if t == "1" else (d["tier0_code"] if sc == "code" else d["tier0"])
assert len(ks) == len(set(ks)) == (364 if t == "1" else 40), len(ks)
open(sys.argv[1], "w").write("\n".join(ks) + "\n")
PK
fi
cp "$KF" "$RKF"; RNK=$(sort -u "$RKF" | grep -c .)
[ "$RNK" = "$NK" ] || { say "the panel has $RNK keys, not $NK: runner stops"; exit 2; }
[ "$(refcount)" = "$RNK" ] || [ "${CARD_REF_LATER:-0}" = 1 ] || { say "the reference ($REFS) holds $(refcount) of $RNK keys: runner stops"; exit 2; }
strata_done() { # amendment 4: a finished Strata only (DRIVER_DONE, or a last waiter line "WAITER END: DONE"), or the release file
  [ -e "$HOME/StrataEval/h2h/results/DRIVER_DONE" ] || [ -e "$AF/STRATA_RELEASE" ] \
    || tail -1 "$HOME/StrataEval/h2h/logs/waiter.log" 2>/dev/null | grep -q "WAITER END: DONE"
}
pids_gone() { local p_; for p_ in ${CARD_AFTER_PIDS:-}; do kill -0 "$p_" 2>/dev/null && return 1; done; return 0; }
[ "$STAGE" = arm ] || { say "runner 69 has the arm stage only: runner stops"; exit 2; }
[ -n "$FENV" ] || { say "the arm stage needs CARD_FENV: runner stops"; exit 2; }
[ -z "$EXT" ] || [ "${CARD_RESUME:-0}" = 1 ] || { say "CARD_EXTEND continues an arm: it needs CARD_RESUME=1: runner stops"; exit 2; }
if [ "$DRY" != 1 ]; then
if [ "${CARD_RESUME:-0}" != 1 ]; then
  for r in "$RUNF" "$RUNC"; do [ -e "results/$r" ] && { echo "results/$r exists: move it aside, or CARD_RESUME=1 to continue it"; exit 2; }; done
fi
echo $$ > results/q70_${TG}.pid; trap 'rm -f results/q70_${TG}.pid' EXIT
rm -f "$DONE"
say "runner 69 (the lossy gate, $TAG, tier $TIER$([ "$TIER" = 2 ] && echo ': the card gate, both orders'), scope $SCOPE) start (pid $$)$([ "${CARD_RESUME:-0}" = 1 ] && echo ', a resume'): waits for $FENV (written after Wed 14:33); cutoff $(date -r "$CUTOFF" '+%m-%d %H:%M')"
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
[ -n "${CARD_C:-}" ] && C=$CARD_C   # runner 70: CARD_C (Part A: 192)
case "$C" in 100|160|176|192|208|224|240|256) ;; *) say "C '$C' from the comment is not a release C: runner stops"; exit 2;; esac   # 100: W24 (24 GB)
MEM=$(echo "$COMMENT" | grep -oE 'MEM [0-9]+' | head -1 | awk '{print $2}'); MEM=${MEM:-47}; MEM=${CARD_MEM:-$MEM}   # PREREG_p2gate45: 45 (49 GB NO-SWAP)
GATE=${CARD_GATE:-$((MEM + 4))}       # runner 66 (amendment 4e): the CHAT step's NO-SWAP gate, whole GB 41-49
case $GATE in ''|*[!0-9]*) say "CARD_GATE=$CARD_GATE is not a whole number of GB: runner stops"; exit 2;; esac
[ "$GATE" -ge 41 ] && [ "$GATE" -le 49 ] || { say "CARD_GATE=$GATE is outside 41-49 GB: runner stops"; exit 2; }
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
if [ -n "${CARD_REPLAY:-}" ]; then   # tier 0a needs a screen with R57_REPLAY and BigMoE's two start-up fixes (d1fd8c5: the
  grep -q 'RPL is not None and RPL\["key"\] is not None' "$SCREEN" && grep -q 'drafter = "dfl"' "$SCREEN" \
    || { say "CARD_REPLAY: the screen ($SCREEN) lacks R57_REPLAY with BigMoE's d1fd8c5 + e5d17c6 fixes (name a newer snapshot, or none for a freeze): runner stops"; exit 2; }
fi                                    # warm-up has no key; e5d17c6: DFlash2 carries the reference drafts on every set)
SETS_="humaneval humanevalx mbpp ifeval"; [ "$TIER" = 2 ] && SETS_="ifbench gpqad lcb6"   # runner 69: the card sets
for s_ in $SETS_; do   # the panel's sets (keys carry their set name before "/")
  grep -q "\"$s_\": *\"$s_\"" "$SCREEN" || { say "the screen's SETOF does not map $s_ ($SCREEN): runner stops"; exit 2; }
done
say "config env (${#STACKENV[@]} keys): $ENVLINE"
say "C $C (MEM $MEM; CHAT gate $GATE GB NO-SWAP$([ -n "${CARD_GATE:-}" ] && echo ", CARD_GATE")); BigMoE $BMC; ext $EB .so $QOMSO; code: $MODE${SNAP:+ $SNAP}${FZ:+ $FZ} (${SNAPP[*]:1}); screen $(shasum -a 256 "$SCREEN" | cut -c1-12)"
PIN0=$(pin)
say "code pin $PIN0"
QS=$(printf '%s\n' "${STACKENV[@]}" | grep '^R57_QOM_SETS=' | tail -1); QS=${QS#R57_QOM_SETS=}
TI=$(printf '%s\n' "${STACKENV[@]}" | grep '^R57_TIERS=' | tail -1); TI=${TI#R57_TIERS=}; TS=""   # amendment 4b: 1c's sets
if [ -n "$TI" ]; then
  TS=$(printf '%s\n' "${STACKENV[@]}" | grep '^R57_TIERS_SETS=' | tail -1); TS=${TS#R57_TIERS_SETS=}
  [ -n "$TS" ] || { say "R57_TIERS=$TI without R57_TIERS_SETS: runner stops"; exit 2; }
fi
say "QOM's validity rule: pf.qom_opt > 0 on every run in ${QS:-no set}$([ -n "$TS" ] && echo "; R57_TIERS=$TI on $TS: pf.qom_opt == lossy_check.opt_post (res1c's uses; 0 without res1c) on the sets in both (1c replaced QOM's keep; runner 68)")"
say "engagement: ${ENGAGE:-none (a base arm)}$([ -n "$RUNTERMS" ] && echo "; whole-run terms $RUNTERMS: > 0 over the run's kept code answers (EARLY from 32, FULL before each verdict), 0 on chat")"
# amendment 12a: the screen stops a process at R57_HARD s of running time (default 600; the w24 env sets 1200). A process
# takes HARD + its load, the cut row and teardown (762 s at most at HARD 600, Mon 11-09), so each is priced HARD + 300 s
# before the cutoff (900 s at 600, as before) and its shared lock lease is 2x that
HARDS=$(printf '%s\n' "${STACKENV[@]}" | grep '^R57_HARD=' | tail -1); HARDS=${HARDS#R57_HARD=}; HARDS=${HARDS:-600}
case "$HARDS" in *[!0-9]*) say "R57_HARD '$HARDS' is not whole seconds: runner stops"; exit 2;; esac
STEP=$((HARDS + 300))
say "R57_HARD $HARDS s: each process is priced $STEP s before the cutoff; shared lock lease $((STEP / 30)) min"
case "${CARD_YIELD_GB:-5}" in ''|*[!0-9]*) say "CARD_YIELD_GB '${CARD_YIELD_GB}' is not whole GB: runner stops"; exit 2;; esac
say "each process start yields to another session's shared MEM run above ${CARD_YIELD_GB:-5} GB, or to two above 5 GB"
KEYS=$(grep . "$KF" | paste -sd, -)     # the panel's round-robin order (tools/lossy_panel.json)
if [ "$ORDER" = rev ]; then   # amendment 1c: reversed; in the code scope the code keys and the IFEval block reverse apart,
  if [ "$SCOPE" = code ]; then  # so IFEval still runs last, in its own processes, and the interim falls on code keys
    KEYS=$( (grep -v '^ifeval/' "$KF" | grep . | tail -r; grep '^ifeval/' "$KF" | tail -r) | paste -sd, -)
  else
    KEYS=$(grep . "$KF" | tail -r | paste -sd, -)
  fi
fi
RKEYS=""; XKEYS=""; XRKEYS=""
[ "$TIMING" = 1 ] && KEYS=$(echo "$KEYS" | tr ',' '\n' | grep -v '^ifeval/' | paste -sd, -)   # runner 69: the timed code keys
if [ "$TIER" = 2 ]; then   # runner 69 (runner 61's orders): forward = the gate keys sorted, reverse = that reversed; the
  KEYS=$(grep . "$KF" | grep -v -x -F -f "$XF" | LC_ALL=C sort | paste -sd, -)   # extension keys likewise, appended after
  RKEYS=$(echo "$KEYS" | tr ',' '\n' | tail -r | paste -sd, -)
  if [ -s "$XF" ]; then XKEYS=$(LC_ALL=C sort "$XF" | grep . | paste -sd, -); XRKEYS=$(echo "$XKEYS" | tr ',' '\n' | tail -r | paste -sd, -); fi
fi
pinned() {      # the registered code (CARD_TREES path=tree,...; CARD_SCREEN_SHA; CARD_SO_SHA), not just whatever HEAD holds
  local t_ p_ h_
  for t_ in $(echo "${CARD_TREES:-}" | tr ',' ' '); do
    p_=${t_%%=*}; h_=${t_#*=}
    [ "$(git -C "$BM" rev-parse "HEAD:$p_" 2>/dev/null)" = "$h_" ] || { say "pin: $p_ is not tree $h_"; return 1; }
  done
  [ -z "${CARD_SCREEN_SHA:-}" ] || [ "$(shasum -a 256 "$SCREEN" | cut -c1-12)" = "$CARD_SCREEN_SHA" ] || { say "pin: the screen is not $CARD_SCREEN_SHA"; return 1; }
  [ -z "${CARD_SO_SHA:-}" ] || [ "$QOMSO" = "$CARD_SO_SHA" ] || { say "pin: the .so is not $CARD_SO_SHA"; return 1; }
}
pinned || { say "the registered code is not what BigMoE's HEAD holds: runner stops"; exit 7; }
say "pins: trees [${CARD_TREES:-none}], screen ${CARD_SCREEN_SHA:-unpinned}, .so ${CARD_SO_SHA:-unpinned}: OK"
[ "$DRY" = 1 ] && [ "$TIMING" = 1 ] && { say "DRY: a timing run (no verdict): run $RUNF; $(echo "$KEYS" | tr ',' '\n' | grep -c .) code keys in the panel's forward order ($(echo "$KEYS" | tr ',' '\n' | cut -d/ -f1 | sort | uniq -c | awk '{printf "%s %s ", $2, $1}')), first $(echo "$KEYS" | cut -d, -f1-2); one quiet lock of ${CARD_QUIET_MIN:-90} min, no share locks; heat pauses read from ${CARD_FANLOG:-$HOME/Library/Logs/fanwatch.log}; engagement ${ENGAGE:-none} (whole-run terms ${RUNTERMS:-none}); QOM sets ${QS:-unset}; 1c sets ${TS:-none}; pin $PIN0; bm_pin $(bm_pin && echo OK || echo FAILS); stops before the first GPU step"; exit 0; }
[ "$DRY" = 1 ] && [ "$TIER" = 2 ] && { say "DRY: tier 2 (the card gate), scope $SCOPE, chat sets $CHATSETS; runs $RUNF / $RUNR; forward first $(echo "$KEYS" | cut -d, -f1-2), last $(echo "$KEYS" | tr ',' '\n' | tail -1); reverse first $(echo "$RKEYS" | cut -d, -f1); $(echo "$KEYS" | tr ',' '\n' | grep -c .) gate keys per order ($(echo "$KEYS" | tr ',' '\n' | cut -d/ -f1 | sort | uniq -c | awk '{printf "%s %s ", $2, $1}'))$([ -n "$XKEYS" ] && echo "+ $(echo "$XKEYS" | tr ',' '\n' | grep -c .) extension keys ($EXT)"); identity sets ${IDS:-none}${IDS:+ vs ${CARD_IDENT_REF:-UNSET}}; engagement ${ENGAGE:-none} (whole-run terms ${RUNTERMS:-none}); QOM sets ${QS:-unset}; 1c sets ${TS:-none}; reference $REFS holds $(refcount) of $RNK; pin $PIN0; bm_pin $(bm_pin && echo OK || echo FAILS); stops before the first GPU step"; exit 0; }
[ "$DRY" = 1 ] && { say "DRY: tier $TIER, scope $SCOPE, order $ORDER (first $(echo "$KEYS" | cut -d, -f1-2), last $(echo "$KEYS" | tr ',' '\n' | tail -1))${IDENT:+ (IFEval identity)}, $(echo "$KEYS" | tr ',' '\n' | grep -c .) keys; engagement field ${ENGAGE:-none} (whole-run terms ${RUNTERMS:-none}); QOM sets ${QS:-unset}; 1c sets ${TS:-none}; reference $REFS holds $(refcount) of $RNK; pin $PIN0; bm_pin $(bm_pin && echo OK || echo FAILS); stops before the first GPU step"; exit 0; }
# ------------------------------------------------ the arm: the panel in its round-robin order
PHASE="lossy gate $TAG t$TIER"; EXTRAENV=(); TAGP0="es_cd${TG}_"; TAGP="${TAGP0}$(date +%d%H%M)_"; IFECOPY=0; REFMAN=""
[ "${CARD_RESUME:-0}" = 1 ] && REFMAN=$(ls -tr "$AF"/${TAGP0}*_modules.json 2>/dev/null | head -1)
[ -n "$REFMAN" ] && say "resume: reference manifest $REFMAN"
verdict() {     # mode: lossy_d4.py -> results/card_<tag>_<mode>.txt, its last line -> DONE
  share_lock "evalsuite: lossy gate scoring (CPU only: graders)" 60 "--cpu 150 --gpu 0"
  $PY tools/lossy_d4.py "$1" "$RUNF" "$REFS" "$SCOPE" $IDENT > "results/card_${TG}_$1${SCOPE/all/}.txt" 2>&1; local rc_=$?
  peer unlock > /dev/null 2>&1
  grep -E "^(HumanEval|MBPP|IFEval|pooled|interim|tier 0)" "results/card_${TG}_$1${SCOPE/all/}.txt" | while read -r l_; do say "LOSSY $TAG: $l_"; done
  return $rc_
}
replay_pass() {   # tier 0a (PREREG_lossygate; LOSSY_GATE section 9): BigMoE's R57_REPLAY on the tier-0 keys, before the mini-panel
  # CARD_REPLAY=exact: the exact config's own pass writes the margins (one file per process, merged here) and must accept
  # 100 %, its self-check; CARD_REPLAY=<merged margins.json>: a variant's pass, its flips binned by the exact margins.
  local rows="$AF/replay_rows_${TG}.jsonl" rc_
  local rsrc=${CARD_REPLAY_ROWS:-$REFS}     # BigMoE 03:0x: replay the EXACT config's OWN rows (same flags, same C); rows from
  [ -n "${CARD_REPLAY_ROWS:-}" ] || say "REPLAY: rows from CARD_REF ($REFS); another exact path's rows flip at its ties (REF2-T: 525 / 527)"
  python3 - "$rsrc" "$KF" "$rows" <<'RR' >> "$LOG" 2>&1 || { say "REPLAY: reference rows incomplete: runner stops"; exit 4; }
import json, sys
refs, keys, out = sys.argv[1].split(","), [k for k in open(sys.argv[2]).read().split() if k], sys.argv[3]
want, have = set(keys), {}
for run in refs:                                  # earlier references win, as cert_report.load merges them
    for l in open(f"results/{run}/decode.jsonl"):
        r = json.loads(l)
        if r["key"] in want and r["key"] not in have:
            have[r["key"]] = r
miss = [k for k in keys if k not in have or not have[k].get("completion_ids")]
if miss:
    print(f"REPLAY rows: {len(miss)} keys without a reference row in {refs}: {miss[:5]}"); sys.exit(1)
with open(out, "w") as f:
    for k in keys:
        f.write(json.dumps({x: have[k][x] for x in ("key", "completion_ids", "n_prompt", "prompt_ids") if x in have[k]}) + "\n")
print(f"REPLAY rows: {len(keys)} reference rows from {refs} -> {out}")
RR
  case "$CARD_REPLAY" in
    exact) EXTRAENV=("R57_REPLAY=$rows") ;;
    *) [ -s "$CARD_REPLAY" ] || { say "REPLAY: the margins file $CARD_REPLAY is missing: runner stops"; exit 2; }
       EXTRAENV=("R57_REPLAY=$rows:$CARD_REPLAY") ;;
  esac
  order "${RUNF}_rpl" r0 "$KEYS"
  EXTRAENV=()
  python3 - "$AF" "${TAGP}r0" "$CARD_REPLAY" "$AF/replay_margins_${TG}.json" "$KF" <<'RS' > "results/card_${TG}_replay.txt" 2>&1; rc_=$?
import glob, json, sys
af, pre, mode, mout, kf = sys.argv[1:6]
keys = [k for k in open(kf).read().split() if k]
rs = {r["key"]: r["replay"] for f in sorted(glob.glob(f"{af}/{pre}*_runs.json")) for r in json.load(open(f)) if r.get("replay")}
miss = [k for k in keys if k not in rs]
n, acc, na, aa = (sum(rs[k][x] for k in rs) for x in ("n", "acc", "n_all", "acc_all"))
fb = [sum(rs[k]["flips_by_margin"][b] for k in rs) for b in ("lt0.5", "0.5-1", "1-2", "ge2")]
pf = all(rs[k]["pf_ok"] for k in rs); src = sorted({rs[k]["margins_from"] for k in rs})
for k in keys:
    if k in rs:
        r = rs[k]; print(f"  {k}: drafted rows accepted {r['acc']} / {r['n']} (A {r['A']:.4f}), all rows {r['acc_all']} / {r['n_all']}, "
                         f"flips by exact margin {r['flips_by_margin']}, prefill pick {'ok' if r['pf_ok'] else 'DIFFERS'}")
A, F = acc / max(n, 1), 1e3 * fb[3] / max(n, 1)
print(f"REPLAY ({len(rs)} of {len(keys)} keys; margins from {src}): A {A:.4f} ({acc} / {n} drafted rows), F_hi {F:.2f} per 1,000 "
      f"(flips by exact margin <0.5 / 0.5-1 / 1-2 / >=2: {' / '.join(map(str, fb))}); all rows {aa} / {na}; prefill picks "
      f"{'all equal' if pf else 'NOT all equal'}")
if mode == "exact":                               # the self-check: the exact config accepts every reference token
    mg = {}
    for f in sorted(glob.glob(f"{af}/{pre}*_margins.json")):
        mg.update(json.load(open(f)))
    json.dump(mg, open(mout, "w"))
    ok = not miss and acc == n and aa == na and pf and set(mg) >= set(keys) and src == ["own"]   # "own": every process wrote
    print(f"REPLAY exact self-check: {'PASS' if ok else 'FAIL'} (needs every drafted and check row accepted, prefill picks equal, "   # fresh margins (BigMoE 02:0x:
          f"margins written fresh by every process [{', '.join(src)}] and for every key: {len(set(mg) & set(keys))} of {len(keys)} "   # an existing file means READ mode)
          f"keys -> {mout})")
    sys.exit(0 if ok else 1)
if miss or src != ["exact"]:
    print(f"REPLAY: INCOMPLETE ({len(miss)} keys without a replay summary; margins from {src}, not the exact run's)")
    sys.exit(1)
print("REPLAY: information until the calibration amendment registers the A / F_hi lines")
RS
  grep -E "^REPLAY" "results/card_${TG}_replay.txt" | while read -r l_; do say "LOSSY $TG: $l_"; done
  [ "$rc_" = 0 ] || void "REPLAY: $(tail -1 "results/card_${TG}_replay.txt" | cut -c1-120)"
}
card_verdict() {   # runner 69, tier 2: per benchmark and order vs $REFS (runner 61's assembly; PREREG_card_gates' lines,
  # amendment 1); the CARD_IDENTSETS sets by token identity with the base's same-order run instead
  local s_ o l n_ ext_ nf nd ni p_ ir_ out_ rr_ F="" I="" X="" S=""
  share_lock "evalsuite: card verdict scoring (CPU only: graders; LCB tests one process at a time)" 60 "--cpu 150 --gpu 0"
  for s_ in $(echo "${CARD_SETS:-ifbench,gpqad,lcb6}" | tr ',' ' '); do
    nf=0; nd=0; ni=0; ext_=0; case " $EXT " in *" $s_ "*) ext_=1 ;; esac
    n_=$(grep -c "^$s_/" "$KF")   # the set's keys in this panel (its gate keys, or the full set once extended): set_d4's N
    for o in fwd rev; do
      rr_=$RUNF; [ "$o" = rev ] && rr_=$RUNR   # this call's run of that order
      if case " $IDS " in *" $s_ "*) true ;; *) false ;; esac; then
        ir_=$(echo "$CARD_IDENT_REF" | cut -d+ -f"$([ "$o" = fwd ] && echo 1 || echo 2)"); out_="results/card_${TG}_t2_${s_}_${o}_ident.txt"
        python3 - "$s_" "$rr_" "$ir_" "$KF" > "$out_" 2>&1 <<'ID'
import json, os, sys
s, arm, base, kf = sys.argv[1:5]
keys = [k for k in open(kf).read().split() if k.split("/")[0] == s]
def rows(run):
    p = f"results/{run}/decode.jsonl"
    return {r["key"]: r["completion_ids"] for r in map(json.loads, open(p))} if os.path.exists(p) else {}
A, B = rows(arm), rows(base)
miss = [k for k in keys if k not in A or k not in B]
same = sum(1 for k in keys if k in A and k in B and A[k] == B[k])
print(f"{s} identity vs {base}: " + (f"INCOMPLETE ({len(miss)} of {len(keys)} keys without a row in both)" if miss
      else f"tokens identical on {same} of {len(keys)}: " + ("PASS" if same == len(keys) else "FAIL")))
ID
        l=$(tail -1 "$out_")
        say "CARD $TAG $s_ $o (identity, the switch-off check): ${l:-missing}"
        case "$l" in *": PASS") ;; *INCOMPLETE*|"") ni=$((ni + 1)) ;; *) nf=$((nf + 1)) ;; esac
      else
        out_="results/card_${TG}_t2_${s_}_${o}_d4.txt"
        SET_D4_N="$n_" $PY tools/set_d4.py "$s_" "$rr_" "$REFS" > "$out_" 2>&1
        l=$(grep "^$s_ vs $REFS " "$out_" | tail -1)
        say "CARD $TAG $s_ $o (decision 4, n $n_$([ "$ext_" = 1 ] && echo ', the full set')): ${l:-missing}"
        case "$l" in
          "") ni=$((ni + 1)) ;;
          *": PASS"|*": DIFFERS (HIGHER)") ;;
          *) nf=$((nf + 1)); p_=$(echo "$l" | sed -E 's/.*McNemar p ([0-9.]+).*/\1/')
             awk -v p="$p_" 'BEGIN {exit !(p >= 0.05)}' && nd=$((nd + 1)) ;;
        esac
      fi
      S="$S; $s_ $o$(echo "$l" | sed -E 's/^[^:]*://' | cut -c1-60)"
    done
    if [ "$ni" -gt 0 ]; then I="$I $s_"
    elif [ "$nf" = 0 ]; then :
    elif [ "$ext_" = 0 ] && [ "$nf" = 1 ] && [ "$nd" = 1 ] && ! case " $IDS " in *" $s_ "*) true ;; *) false ;; esac; then
      X="$X $s_"   # amendment 1: one order failing on the delta alone
    else F="$F $s_"; fi
  done
  peer unlock > /dev/null 2>&1
  V=PASS
  [ -n "$X" ] && V="PENDING: extend$X to the full set in both orders (amendment 1: runner 61's ref stage, then CARD_RESUME=1 CARD_EXTEND=\"$(echo $X)\")"
  [ -n "$I" ] && V="INCOMPLETE ($I )"; [ -n "$F" ] && V="FAIL ($F )"
  echo "CARD $TG (tier 2, scope $SCOPE): $V vs $REFS$S" > "$DONE"
}
[ "$(refcount)" = "$RNK" ] || { say "the reference ($REFS) holds $(refcount) of $RNK keys: no verdict: runner stops"; echo "LOSSY $TAG: INCOMPLETE (the reference holds $(refcount) of $RNK keys)" > "$DONE"; exit 6; }
quiet_take() {   # runner 69, timing: one queued quiet lock for the whole run (peer lock quiet --queue <min>), no share locks
  local r
  say "timing: queueing for a quiet lock of ${CARD_QUIET_MIN:-90} min"
  r=$(peer lock quiet --queue "${CARD_QUIET_MIN:-90}m" "evalsuite: MEM $MEM GB: lossy headline timing $TAG (HE+ 164 + MBPP 100, one order)" 2>&1) \
    || { say "timing: no quiet lock ($(echo "$r" | tail -1 | cut -c1-160)): runner stops"; exit 3; }
  [ -e "$QPAUSE" ] && { peer unlock > /dev/null 2>&1; say "timing: PAUSE right after the quiet lock: released it: runner stops"; exit 3; }
  say "timing: quiet lock held from $(date +%H:%M:%S) ($(echo "$r" | tail -1 | cut -c1-140))"
}
timing_py() {    # runner 69, timing: tainted | summary, over the process windows (TLED), their runs.json and fanwatch's pauses
  TLED="$TLED" AF="$AF" FANLOG="${CARD_FANLOG:-$HOME/Library/Logs/fanwatch.log}" ENVLINE="$ENVLINE" FB="$FB" TREES="${CARD_TREES:-}" \
    SCR="$(shasum -a 256 "$SCREEN" | cut -c1-12)" C_="$C" MEM_="$MEM" KEYS="$KEYS" TAG="$TAG" python3 - "$1" <<'TM'
import datetime as dt, json, os, re, sys
mode = sys.argv[1]
keys = [k for k in os.environ["KEYS"].split(",") if k]
procs = [l.split() for l in open(os.environ["TLED"]).read().splitlines() if l.strip()] if os.path.exists(os.environ["TLED"]) else []
pauses, start = [], None                                # fanwatch: "<date> <time> heat pause ...: PAUSE set" .. "<date> <time> cool ... PAUSE cleared"
if os.path.exists(os.environ["FANLOG"]):
    for l in open(os.environ["FANLOG"], errors="replace"):
        m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (.*)", l)
        if not m:
            continue
        t = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
        if "heat pause" in m.group(2) and "PAUSE set" in m.group(2):
            start = t if start is None else start
        elif "PAUSE cleared" in m.group(2) and start is not None:
            pauses.append((start, t)); start = None
if start is not None:                                   # a pause still on
    pauses.append((start, dt.datetime.now().timestamp()))
def hit(a, b):
    return [(p, q) for p, q in pauses if p < b and q > a]
rec, bad = {}, {}                                       # per key: the latest record from an untainted window; tainted ones apart
for tag, a, b in procs:
    a, b = int(a), int(b)
    p = os.path.join(os.environ["AF"], f"{tag}_runs.json")
    rs = json.load(open(p)) if os.path.exists(p) else []
    h = hit(a, b)
    for r in rs:
        if r.get("cut") or not r.get("tps"):
            continue
        (bad if h else rec)[r["key"]] = dict(r, _tag=tag, _pause=h)
if mode == "tainted":                                   # the keys to re-run: a timed key with no clean record
    print(",".join(k for k in keys if k not in rec))
    sys.exit(0)
env = dict(t.split("=", 1) for t in re.split(r",(?=[A-Z0-9_]+=)", os.environ["ENVLINE"]) if "=" in t)
rows = env.get("R57_ROWS") or "16 (R57_ROWS unset)"
tn = env.get("R57_TREE_NODES", "")                      # BigMoE's tree flags (Tue 10:56): pops + the current token = rows
shape = (f"a tree, {tn} best-first pops + the current token = {int(tn) + 1} rows per check, top-{env.get('R57_TREE_K', '?')} "
         f"children" if tn.isdigit() else f"{rows} rows per check, a chain")
drv = "; ".join(f"{k}={env[k]}" for k in ("R57_DRAFT", "R57_T30B_FMT", "R57_TREE_NODES", "R57_TREE_K", "R57_TREE_SF",
                 "R57_TREE_CC_MAX", "R57_XRW") if k in env)
acc = "; ".join(f"{k}={env[k]}" for k in ("R57_FAST", "R57_QOM_SETS", "R57_TIERS", "R57_TIERS_S", "R57_TIERS_SETS",
                 "R57_TIERS_LAM", "R57_RELAX_DELTA", "R57_RELAX_DELTA_SETS", "R57_COMQOM_SETS", "R57_COMQOM_DOC_SETS",
                 "R57_RES_SETS", "R57_RELAX_SPAN", "R57_RELAX_SPAN_RULE", "R57_COPY") if k in env)
miss = [k for k in keys if k not in rec]
def agg(ks):
    rs = [rec[k] for k in ks if k in rec]
    n = sum(r["n"] for r in rs); dw = sum(r["n"] / r["tps"] for r in rs); pf = sum(r.get("pfs") or 0 for r in rs)
    ck = sum(r.get("checks") or 0 for r in rs)
    return len(rs), n, dw, pf, ck
out = []
for lab, ks in (("humaneval", [k for k in keys if k.startswith("humaneval/")]),
                ("humanevalx", [k for k in keys if k.startswith("humanevalx/")]),
                ("HE+ (humaneval + humanevalx)", [k for k in keys if k.split("/")[0] in ("humaneval", "humanevalx")]),
                ("mbpp", [k for k in keys if k.startswith("mbpp/")]), ("pooled code", keys)):
    na, n, dw, pf, ck = agg(ks)
    out.append(f"{lab} ({na} of {len(ks)} answers): whole decode {n / dw if dw else 0:.2f} tok/s = n {n} / decode wall "
               f"{dw:.1f} s; with prefill {n / (dw + pf) if dw + pf else 0:.2f} (information); kept per check {n / ck if ck else 0:.2f}")
drf = {}
for k, r in rec.items():
    drf.setdefault(k.split("/")[0], {}).setdefault(str(r.get("drafter")), 0)
    drf[k.split("/")[0]][str(r.get("drafter"))] += 1
win = [(int(a), int(b)) for _, a, b in procs]
print(f"TIMING {os.environ['TAG']} (runner 69; one quiet lock; the panel's forward order; {len(procs)} processes, "
      f"{(max(b for _, b in win) - min(a for a, _ in win)) / 60 if win else 0:.1f} min with loads)")
for l in out:
    print("  " + l)
print(f"  config: {shape}; draft flags: {drv or 'none (R57_DRAFT unset)'}; drafters {drf}; acceptance: {acc}; "
      f"env blob {os.environ['FB']}; trees [{os.environ['TREES']}]; screen {os.environ['SCR']}; C {os.environ['C_']}; MEM {os.environ['MEM_']}")
print(f"  heat pauses (fanwatch) inside a timed window: " + (", ".join(f"{t} {[(dt.datetime.fromtimestamp(p).strftime('%H:%M:%S'), dt.datetime.fromtimestamp(q).strftime('%H:%M:%S')) for p, q in hit(int(a), int(b))]}" for t, a, b in procs if hit(int(a), int(b))) or "none")
      + (f"; tainted answers replaced by clean re-runs: {len(bad)}" if bad else ""))
pooled = agg(keys)
print(f"TIMING {os.environ['TAG']}: whole decode {pooled[1] / pooled[2] if pooled[2] else 0:.2f} tok/s pooled code "
      f"(n {pooled[1]} / {pooled[2]:.1f} s, {pooled[0]} of {len(keys)} answers)" + (f"; INCOMPLETE: {len(miss)} answers without a clean record" if miss else ""))
sys.exit(1 if miss else 0)
TM
}
timing_rerun() { # runner 69, timing: re-run the keys whose only records come from a window with a heat pause (twice at most)
  local round ks i
  for round in 1 2; do
    ks=$(timing_py tainted 2> /dev/null); [ -z "$ks" ] && return 0
    say "timing: $(echo "$ks" | tr ',' '\n' | grep -c .) answers lack a clean record (a heat pause inside their window, or no record): re-run $round"
    i=0
    while [ -n "$ks" ]; do
      i=$((i + 1))
      fast "${TAGP}tr${round}_${i}" "$(echo "$ks" | tr ',' '\n' | head -"$PERPROC" | paste -sd, -)" "timing re-run $round process $i"
      ks=$(echo "$ks" | tr ',' '\n' | tail -n +"$((PERPROC + 1))" | paste -sd, -)
    done
  done
}
if [ "$TIMING" = 1 ]; then   # runner 69: the timing run (one quiet lock, the code keys in one order, no verdict)
  quiet_take
  order "$RUNF" tm "$KEYS"
  timing_rerun
  peer unlock > /dev/null 2>&1; say "timing: quiet lock released"
  runterms "$RUNF" full
  timing_py summary > "results/card_${TG}.txt" 2>&1
  grep -E '^ |^TIMING' "results/card_${TG}.txt" | while read -r l_; do say "$l_"; done
  tail -1 "results/card_${TG}.txt" > "$DONE"
elif [ "$AR" = 1 ]; then   # runner 70: the rows, one order, no verdict
  order "$RUNF" ar "$KEYS"
  runterms "$RUNF" full
  echo "AR ROWS $TAG: $(grep -c . "results/$RUNF/decode.jsonl") rows of $NK keys in results/$RUNF (no verdict: the PREREG grades them)" > "$DONE"
elif [ "$TIER" = 2 ]; then   # runner 69: the card gate, both orders in one call (runner 61's arm flow on runner 68's checks)
  order "$RUNF" fwd "$KEYS"
  order "$RUNR" rev "$RKEYS"
  if [ -n "$XKEYS" ]; then   # amendment 1: the extension keys, in both orders, appended to the same runs
    say "extension ($EXT): $(echo "$XKEYS" | tr ',' '\n' | grep -c .) keys per order"
    order "$RUNF" xfwd "$XKEYS"
    order "$RUNR" xrev "$XRKEYS"
  fi
  runterms "$RUNF" full; runterms "$RUNR" full
  card_verdict
elif [ "$TIER" = 0 ]; then
  [ -n "${CARD_REPLAY:-}" ] && replay_pass
  order "$RUNF" t0 "$KEYS"
  runterms "$RUNF" full
  verdict tier0
  echo "$(tail -1 "results/card_${TG}_tier0${SCOPE/all/}.txt") (vs $REFS)$([ -n "${CARD_REPLAY:-}" ] && echo "; $(grep '^REPLAY (' "results/card_${TG}_replay.txt" | cut -c1-150)")" > "$DONE"
else
  N2=$(python3 -c 'import json, sys; d = json.load(open("tools/lossy_panel.json")); print(d["code_interim_n"] if sys.argv[1] == "code" else d["interim_n"])' "$SCOPE")
  FIRST=$(echo "$KEYS" | tr ',' '\n' | head -"$N2" | paste -sd, -); SECOND=$(echo "$KEYS" | tr ',' '\n' | tail -n +"$((N2 + 1))" | paste -sd, -)
  IFE=""
  if [ "$SCOPE" = code ]; then   # the code keys first (interim at 132), then IFEval in its own processes (the switch-off check)
    IFE=$(echo "$KEYS" | tr ',' '\n' | grep '^ifeval/' | paste -sd, -); SECOND=$(echo "$SECOND" | tr ',' '\n' | grep -v '^ifeval/' | paste -sd, -)
  fi
  order "$RUNF" h1 "$FIRST"
  if ! verdict interim; then
    runterms "$RUNF" full
    echo "LOSSY $TAG tier1: FAIL (the interim look stopped it:$(grep '^interim' "results/card_${TG}_interim${SCOPE/all/}.txt")) (vs $REFS)" > "$DONE"
  else
    order "$RUNF" h2 "$SECOND"
    [ -n "$IFE" ] && order "$RUNF" ife "$IFE"
    runterms "$RUNF" full
    verdict tier1
    echo "$(tail -1 "results/card_${TG}_tier1${SCOPE/all/}.txt"); $(grep '^pooled' "results/card_${TG}_tier1${SCOPE/all/}.txt" | cut -c1-140) (vs $REFS)" > "$DONE"
    if [ "$ORDER" = rev ] && [ -n "${CARD_FWD_RUN:-}" ]; then   # amendment 1c: the verdict that licenses tier 2 = both orders pooled
      PF="results/card_${TG}_tier1pooled${SCOPE/all/}.txt"
      $PY tools/lossy_d4.py tier1 "$CARD_FWD_RUN+$RUNF" "${CARD_FWD_REF:?set CARD_FWD_REF to the reference of the forward order}+$REFS" "$SCOPE" $IDENT > "$PF" 2>&1
      grep -E "^(HumanEval|MBPP|IFEval|pooled)" "$PF" | while read -r l_; do say "LOSSY $TG pooled: $l_"; done
      echo "$(tail -1 "$PF"); $(grep '^pooled' "$PF" | cut -c1-140) (both orders: $CARD_FWD_RUN + $RUNF vs $CARD_FWD_REF + $REFS)" > "$DONE"
    fi
  fi
fi
say "the lossy verdict: $(cat "$DONE")"
peer say --to "${CARD_NOTIFY:-AI/LLM project recommendations}" "EvalSuite lossy gate $(date +%H:%M): $(cut -c1-230 "$DONE")" > /dev/null 2>&1
say "runner 69 end"
