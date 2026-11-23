# Drift screen 2 (NewIdeas 18:5x): FAR's WALK-MIX TTT (results/LEVERS_A.md, section WALKMIX). Smoke (w0 must match ctl),
# two training processes under the 5 GB cap (A: ctl + mix3; B: mix5; deterministic, same slices), strict read-out, tree-32 kept.
NAME=walkmix
plan() {
  step smoke logs/walkmix_smoke.log 10 $T -l $PY sb/train_multi.py --tag wmixsm --arms ctl:-,w0:walkmix0,w5:walkmix5 --depth 30 --steps 3 --mem 5
  if [ "$DRY" != 1 ]; then
    $PY -I -c 'import json, sys
a = json.load(open("ckpt/wmixsm.json"))["arms"]
c, z, f = a["ctl"]["loss"], a["w0"]["loss"], a["w5"]["loss"]
same = all(abs(x - y) <= 1e-3 * abs(x) for x, y in zip(c, z)) and len(c) == len(z) == 3
diff = any(abs(x - y) > 1e-3 * abs(x) for x, y in zip(c, f))
print("SMOKECHECK", "OK" if same and diff else "FAIL", "ctl", [round(x, 5) for x in c], "w0", [round(x, 5) for x in z], "w0.5", [round(x, 5) for x in f])
sys.exit(0 if same and diff else 1)' > logs/walkmix_smokecheck.log 2>&1 || { STATUS="smoke check failed: $(tail -1 logs/walkmix_smokecheck.log | cut -c1-250)"; exit 5; }
    say "$(tail -1 logs/walkmix_smokecheck.log)"
  fi
  step trainA logs/walkmix_trainA.log 25 $T -l $PY sb/train_multi.py --tag wmix --arms ctl:-,mix3:walkmix3 --depth 30 --steps 400 --mem 5
  step trainB logs/walkmix_trainB.log 20 $T -l $PY sb/train_multi.py --tag wmixb --arms mix5:walkmix5 --depth 30 --steps 400 --mem 5
  step eval logs/walkmix_eval.log 15 $T -l $PY sb/evalx.py --runs s30,wmix_ctl,wmix_mix3,wmixb_mix5 --sets mbpp,he --nm 50 --nh 30 --depth 31 --mode chain --out results/walkmix_eval.json
  step tree logs/walkmix_tree.log 15 $T -l env PAIRS=wmix_mix3:wmix_ctl,wmixb_mix5:wmix_ctl $PY $HOME/TrainLab/sb98/tree8.py results/walkmix_tree.json wmix_ctl wmix_mix3 wmixb_mix5
  [ "$DRY" = 1 ] && { VERDICT="(dry)"; return; }
  $PY -I sb/walkmix_verdict.py > logs/walkmix_verdict.log 2>&1
  VERDICT=$(grep -E "^VERDICT" logs/walkmix_verdict.log | awk '{print $2, $3}' | tr '\n' ' ' | cut -c1-250)
}
