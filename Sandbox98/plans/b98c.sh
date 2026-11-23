# BREAK98c PREFIX EYES (Fable FAR, ~/FableGod/far/IDEAS_1117c_prefix_eyes.md section 1; results/LEVERS_A.md section B98C):
# the frozen trunk's own K / V cache at full-attention layers 15 / 19 for the 297 training files + MBPP-50 / HE-30 (one
# capture-shaped forward per file, ~1 GB), a 3-step smoke (ctl + kvx3), both arms in one process (kvx3: the block at 3e-5,
# kvx10: 1e-4; WALK-MIX's setup), the chain and tree read-outs, the cache deleted (this run's own scratch), the verdict
# vs wmix_ctl (results/b98s1_ctl_eval.json chain + true; results/walkmix_tree.json).
NAME=b98c
# Lock sizes = measured LIFETIME phys_footprint (/usr/bin/time -l in the step logs): capture ~4 GB (futgen 3.9), 2-arm
# training ~5.1 GB + the blocks' weights and Adam (~0.6 GB), evalx 6.4-7.3, tree8 7.1-7.5. mem N sets the share lock
# (MEMGB) and the NO-SWAP start gate (N + 8) for the steps after it.
mem() { MEMGB=$1; GUARD_NOSWAP=$(( $1 + 8 )); }
plan() {
  mem 5
  step kvgen logs/b98c_kvgen.log 10 $T -l $PY sb/kvxgen.py --sets train,mbpp,he --check 3
  [ "$DRY" != 1 ] && say "$(grep -E '^\[kv\] DONE' logs/b98c_kvgen.log | cut -c1-250)"
  mem 7
  step smoke logs/b98c_smoke.log 5 $T -l $PY sb/train_multi.py --tag b98csm --arms ctl:-,kvx3:kvx3 --depth 30 --steps 3 --mem 6
  if [ "$DRY" != 1 ] && [ ! -f logs/.done/$RUN/smokecheck ]; then
    $PY -I -c 'import json, sys
import mlx.core as mx
a = json.load(open("ckpt/b98csm.json"))["arms"]
wm = json.load(open("ckpt/wmix.json"))["arms"]["ctl"]["loss"][:3]
c, k = a["ctl"]["loss"], a["kvx3"]["loss"]
w = mx.load("ckpt/b98csm_kvx3.safetensors")
on = [float(mx.linalg.norm(w[f"kvx_o{L}.weight"])) for L in (15, 19)]
s1 = len(c) == len(k) == 3 and c[0] == k[0]
cw = c[:2] == wm[:2]                 # steps 1-2 only: from step 2 on, a 3-step cosine schedule differs from a 400-step one
ok = s1 and cw and all(x > 0 for x in on) and all(f"kvx_q{L}.weight" in w for L in (15, 19))
print("SMOKECHECK", "OK" if ok else "FAIL", "| step-1 loss ctl", repr(c[0]), "kvx3", repr(k[0]), "(bitwise)" if s1 else "(DIFFER)",
      "| ctl steps 1-2 == wmix_ctl logged:", cw, "| losses ctl", [round(x, 7) for x in c], "kvx3", [round(x, 7) for x in k],
      "| |o_x| after 3 steps", [f"{x:.3e}" for x in on], "| grad norms (head, block)", a["kvx3"].get("gnorm"))
sys.exit(0 if ok else 1)' > logs/b98c_smokecheck.log 2>&1 || { STATUS="smoke check failed (VOID): $(tail -1 logs/b98c_smokecheck.log | cut -c1-250)"; exit 5; }
    say "$(tail -1 logs/b98c_smokecheck.log | cut -c1-250)"
    rm -f ckpt/b98csm_ctl.safetensors ckpt/b98csm_kvx3.safetensors; touch logs/.done/$RUN/smokecheck
  fi
  mem 7
  step train logs/b98c_train.log 20 $T -l $PY sb/train_multi.py --tag b98c --arms kvx3:kvx3,kvx10:kvx10 --depth 30 --steps 400 --mem 6
  mem 8
  step eval logs/b98c_eval.log 10 $T -l $PY sb/evalx.py --runs b98c_kvx3@kvx3,b98c_kvx10@kvx10 --sets mbpp,he --nm 50 --nh 30 --depth 31 --mode chain --out results/b98c_eval.json
  step tree logs/b98c_tree.log 15 $T -l $PY $HOME/TrainLab/sb98/tree8.py results/b98c_tree.json b98c_kvx3@kvx3 b98c_kvx10@kvx10
  [ "$DRY" = 1 ] && { VERDICT="(dry)"; return; }
  [ -f results/b98c_tree.json ] && [ -f results/b98c_eval.json ] && rm -rf data/kvx      # this run's own scratch (~1 GB)
  $PY -I sb/b98c_verdict.py > logs/b98c_verdict.log 2>&1
  VERDICT=$(grep -E "^VERDICT" logs/b98c_verdict.log | awk '{print $2, $3}' | tr '\n' ' ' | cut -c1-250)
}
