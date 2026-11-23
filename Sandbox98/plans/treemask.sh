# Tree-aware mask (NewIdeas 20:3x, the follow-up to WALK-MIX; results/LEVERS_A.md section TREEMASK). Smoke (k1 must equal
# ctl, k4 must differ), two training processes under the 5 GB cap (A: ctl + tm2; B: tm4; deterministic, same slices),
# strict read-out, tree-32 kept (MTP Head's tree8.py, read-only), verdict.
NAME=treemask
# Lock sizes = measured LIFETIME phys_footprint (Tue 11-17 correction; /usr/bin/time -l in the step logs), not the MLX
# peak: 3-arm smoke ~6-7 GB, 2-arm training 5.1 GB, 1-arm 4.5 GB, evalx 6.4-7.3 GB, tree8 7.1-7.5 GB. mem N sets the
# share lock (MEMGB) and the NO-SWAP start gate (N + 8) for the steps after it.
mem() { MEMGB=$1; GUARD_NOSWAP=$(( $1 + 8 )); }
plan() {
  mem 7
  step smoke logs/treemask_smoke.log 10 $T -l $PY sb/train_multi.py --tag tmasksm --arms ctl:-,k1:treemask1,k4:treemask4 --depth 30 --steps 3 --mem 5
  if [ "$DRY" != 1 ]; then
    $PY -I -c 'import json, sys
a = json.load(open("ckpt/tmasksm.json"))["arms"]
c, o, f = a["ctl"]["loss"], a["k1"]["loss"], a["k4"]["loss"]
same = len(c) == len(o) == 3 and all(abs(x - y) <= 1e-4 * abs(x) for x, y in zip(c, o))
diff = any(abs(x - y) > 1e-3 * abs(x) for x, y in zip(c, f))
print("SMOKECHECK", "OK" if same and diff else "FAIL", "ctl", [round(x, 6) for x in c], "k1", [round(x, 6) for x in o], "k4", [round(x, 6) for x in f])
sys.exit(0 if same and diff else 1)' > logs/treemask_smokecheck.log 2>&1 || { STATUS="smoke check failed (VOID): $(tail -1 logs/treemask_smokecheck.log | cut -c1-250)"; exit 5; }
    say "$(tail -1 logs/treemask_smokecheck.log)"
  fi
  mem 6
  step trainA logs/treemask_trainA.log 30 $T -l $PY sb/train_multi.py --tag tmask --arms ctl:-,tm2:treemask2 --depth 30 --steps 400 --mem 5
  mem 5
  step trainB logs/treemask_trainB.log 20 $T -l $PY sb/train_multi.py --tag tmaskb --arms tm4:treemask4 --depth 30 --steps 400 --mem 5
  mem 8
  step eval logs/treemask_eval.log 15 $T -l $PY sb/evalx.py --runs s30,tmask_ctl,tmask_tm2,tmaskb_tm4 --sets mbpp,he --nm 50 --nh 30 --depth 31 --mode chain --out results/treemask_eval.json
  step tree logs/treemask_tree.log 25 $T -l env PAIRS=tmask_tm2:tmask_ctl,tmaskb_tm4:tmask_ctl $PY $HOME/TrainLab/sb98/tree8.py results/treemask_tree.json tmask_ctl tmask_tm2 tmaskb_tm4
  [ "$DRY" = 1 ] && { VERDICT="(dry)"; return; }
  $PY -I sb/treemask_verdict.py > logs/treemask_verdict.log 2>&1
  VERDICT=$(grep -E "^VERDICT" logs/treemask_verdict.log | awk '{print $2, $3}' | tr '\n' ' ' | cut -c1-250)
}
