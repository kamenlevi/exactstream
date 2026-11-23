# BREAK98 Screen 1 (Fable FAR, ~/FableGod/far/IDEAS_1117_futurestates.md; results/LEVERS_A.md section B98S1): FUTURE-STATE
# ROWS on the 2B. One variant at a time (disk: each variant's states, ~1.75 GB, are deleted after its read-out):
# generate the placeholder states (train slices + MBPP-50 / HE-30 anchors), train the one-arm head (s30o + its Adam, 400
# steps at depth 30, the WALK-MIX setup), read out the chain and tree-32, then the zero-shot print. Control = WALK-MIX's
# wmix_ctl (same args, deterministic), read out here in chain + true modes; its tree comes from WALK-MIX's read-out when
# this chain read-out equals WALK-MIX's exactly, else tree8 re-reads it.
NAME=b98s1
plan() {
  step ctleval logs/b98s1_ctleval.log 10 $T -l $PY sb/evalx.py --runs wmix_ctl --sets mbpp,he --nm 50 --nh 30 --depth 31 --mode chain,true --out results/b98s1_ctl_eval.json
  if [ "$DRY" != 1 ]; then
    $PY -I sb/b98s1_verdict.py --detcheck > logs/b98s1_detcheck.log 2>&1 || CTLTREE=1
    say "$(tail -1 logs/b98s1_detcheck.log | cut -c1-200)"
  fi
  [ "${CTLTREE:-0}" = 1 ] && step treectl logs/b98s1_treectl.log 15 $T -l $PY $HOME/TrainLab/sb98/tree8.py results/b98s1_tree_ctl.json wmix_ctl
  for V in A B C; do
    step gen$V logs/b98s1_gen$V.log 10 $T -l $PY sb/futgen.py --variant $V --sets train,mbpp,he --batch 16
    if [ $V = A ]; then
      step smoke logs/b98s1_smoke.log 5 $T -l $PY sb/train_multi.py --tag b98sm --arms ctl:-,fA:futA --depth 30 --steps 3 --mem 5
      if [ "$DRY" != 1 ] && [ ! -f logs/.done/$RUN/smokecheck ]; then
        $PY -I -c 'import json, sys
import mlx.core as mx
a = json.load(open("ckpt/b98sm.json"))["arms"]
c, f = a["ctl"]["loss"], a["fA"]["loss"]
w = mx.load("ckpt/b98sm_fA.safetensors")
pn = float(mx.linalg.norm(w["fut_fc.weight"])); has = "fut_norm.weight" in w
same = len(c) == len(f) == 3 and abs(c[0] - f[0]) <= 1e-6 * abs(c[0])
ok = same and pn > 0 and has
print("SMOKECHECK", "OK" if ok else "FAIL", "step-1 loss ctl", round(c[0], 8), "fA", round(f[0], 8), "| losses ctl", [round(x, 6) for x in c],
      "fA", [round(x, 6) for x in f], "| |P| after 3 steps", f"{pn:.3e}", "norm_f saved", has)
sys.exit(0 if ok else 1)' > logs/b98s1_smokecheck.log 2>&1 || { STATUS="smoke check failed (VOID): $(tail -1 logs/b98s1_smokecheck.log | cut -c1-250)"; exit 5; }
        say "$(tail -1 logs/b98s1_smokecheck.log | cut -c1-250)"
        rm -f ckpt/b98sm_ctl.safetensors ckpt/b98sm_fA.safetensors; touch logs/.done/$RUN/smokecheck
      fi
    fi
    step train$V logs/b98s1_train$V.log 15 $T -l $PY sb/train_multi.py --tag b98s1$V --arms f$V:fut$V --depth 30 --steps 400 --mem 5
    [ "$DRY" != 1 ] && rm -rf data/fut_$V/mc data/fut_$V/mc2
    step eval$V logs/b98s1_eval$V.log 10 $T -l $PY sb/evalx.py --runs b98s1${V}_f$V@fut$V --sets mbpp,he --nm 50 --nh 30 --depth 31 --mode chain --out results/b98s1_eval_$V.json
    step tree$V logs/b98s1_tree$V.log 15 $T -l $PY $HOME/TrainLab/sb98/tree8.py results/b98s1_tree_$V.json b98s1${V}_f$V@fut$V
    if [ "$DRY" != 1 ]; then
      [ -f results/b98s1_zeroshot_$V.json ] || $PY -I sb/futzs.py --variant $V > logs/b98s1_zeroshot_$V.log 2>&1
      [ -f results/b98s1_zeroshot_$V.json ] && rm -rf data/fut_$V
      $PY -I sb/b98s1_verdict.py --variants $V --out results/b98s1_verdict_$V.json > logs/b98s1_verdict_$V.log 2>&1
      say "$(grep -E '^VERDICT' logs/b98s1_verdict_$V.log | head -1 | cut -c1-250)"
    fi
  done
  [ "$DRY" = 1 ] && { VERDICT="(dry)"; return; }
  $PY -I sb/b98s1_verdict.py > logs/b98s1_verdict.log 2>&1
  VERDICT=$(grep -E "^VERDICT" logs/b98s1_verdict.log | awk '{print $2, $3}' | tr '\n' ' ' | cut -c1-250)
}
