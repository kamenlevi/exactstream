# Drift screen plan template (THINK_1116c). Copy to plans/NAME.sh, pre-register the LINE in results/LEVERS_A.md, then
#   scripts/detach.sh logs/drift_NAME_runner.log scripts/drift_run.sh plans/NAME.sh
# Steps run under the runner's gates and share lock; CPU lines run in the runner itself. $PY, $T and $DRY are set by
# the runner. This template = an eval-only lever (levers/LEVER.py's standin hook) on the frozen s30, scored as the share
# of the own -> true walk-strict gap it closes at each band (sb/driftg.py).
NAME=template
# Size each step's lock to its measured LIFETIME phys_footprint (/usr/bin/time -l in the step log), not the MLX peak:
# evalx / tree8 read-outs ~6.4-7.5 GB, 1-arm training ~4.5 GB, 2-arm ~5.1 GB (Tue 11-17). Before a step: MEMGB=8 GUARD_NOSWAP=16.
LEVER=noisehead
plan() {
  MEMGB=8 GUARD_NOSWAP=16
  # the base's own + true read-out with per-window counts, once (~5 min, <= 4.5 GB): reused by every eval-only arm
  step base logs/drift_base_s30.log 20 $T -l $PY sb/evalx.py --runs s30 --sets he,mbpp --mode chain,true --out results/drift_base_s30.json
  step arm logs/drift_${NAME}_arm.log 20 $T -l $PY sb/evalx.py --runs s30 --lever $LEVER --sets he,mbpp --mode chain --out results/drift_${NAME}.json
  [ "$DRY" = 1 ] && { VERDICT="(dry)"; return; }
  $PY -I sb/driftg.py --own "results/drift_base_s30.json:s30|{set}|chain" --true "results/drift_base_s30.json:s30|{set}|true" \
    --arm "$LEVER=results/drift_${NAME}.json:s30|{set}|chain" --out results/drift_${NAME}_g.json > logs/drift_${NAME}_g.log 2>&1
  VERDICT=$(grep VERDICT logs/drift_${NAME}_g.log | tr '\n' ' ' | cut -c1-250)
}
