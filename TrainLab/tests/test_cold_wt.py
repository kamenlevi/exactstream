import sys, os
sys.path.insert(0, os.path.expanduser("~/TrainLab/sticky"))
import numpy as np, mlx.core as mx
mx.set_default_device(mx.cpu)
import budget125_tf as B
C = B.COLD
acc = dict(v3_bytes=0.0, b2_bytes=0.0, cold_reads=0, two_reads=0, rows=0)
C.update(on=True, mode="single", C=4, r2=99, wt=0.05, lru={}, cur={}, cold={}, two={}, acc=acc)
prompt = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]] * 9)
B.cold_observe(0, prompt, np.full((9, 10), 0.1, np.float32))
p1 = np.array([[7, 8, 11, 12, 13, 14, 15, 16, 17, 18],
               [8, 7, 9, 12, 19, 20, 21, 22, 23, 24]])
w1 = np.array([[.30, .20, .10, .08, .06, .04, .04, .03, .03, .02],   # 13 at rank 4 still >= 0.05 -> stays 3-bit
               [.40, .20, .10, .04, .07, .05, .04, .04, .03, .03]])  # 12: max(.08, .04) -> 3-bit; 19 .07 -> 3-bit
B.cold_observe(0, p1, w1)
two = C["two"][0]
exp = {14, 15, 16, 17, 18, 21, 22, 23, 24}                           # 20: weight .05 is not < .05 -> 3-bit
print("cold:", sorted(C["cold"][0]), "| 2-bit:", sorted(two), "->", two == exp)
print("bytes: cold reads", acc["cold_reads"], "2-bit", acc["two_reads"], "saved", round(1 - acc["b2_bytes"] / acc["v3_bytes"], 3))
# the gate wrapper passes the top-k weights in rank order (softmax over the top-k logits)
lg = mx.array(np.log(np.array([[.5, .3, .2, 1e-9, 1e-9]], np.float32)))
g = B.BudgetGate(lambda x: lg, 3, 1)
seen = {}
B.cold_observe, orig = (lambda L, ids, w=None: seen.update(L=L, ids=ids, w=w)), B.cold_observe
B.BUD.update(U=None, k0=None)
g(mx.zeros((1, 4)))
B.cold_observe = orig
print("gate: ids", seen["ids"].tolist(), "w", np.round(seen["w"], 3).tolist(), "->",
      seen["ids"].tolist() == [[0, 1, 2]] and np.allclose(seen["w"], [[.5, .3, .2]], atol=1e-3))
C["wt"] = None
g(mx.zeros((1, 4)))
