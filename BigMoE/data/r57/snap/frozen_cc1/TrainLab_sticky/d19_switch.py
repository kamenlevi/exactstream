"""D-19 SWITCH (MTP Head, Fri 10-30; NewIdeas' GO; REPORT 'D-19'): CPU replay on D-17's per-pass data of switch rules
that use only the past, between the native MTP chain (K15, from the true prefix) and DFlash2 L1-20 (block 16, 1 step).
Same data and walk as sticky/d17_route.py: both drafters at every position of the same answers, each cut at its first
end token; tokens per pass = accepted + 1. Rules (decided before drafting each pass):
- type: by text type known up front (MTP on ifeval / multi / agent, DFlash2 on code / gsm);
- bandit W (partial feedback): only the drafter that ran is scored. Each answer starts on DFlash2; after every pass the
  arm's accepted count enters its window of the last W; pick the arm with the higher window mean; an arm with an empty
  window, or not tried in the last 4W passes, is tried once;
- bandit W (full feedback, capped): DFlash2 is cheap, so it is drafted every pass and scored on the verified tokens
  too; the other arm's count is observable only up to the verified length: min(its accepted, chosen accepted + 1).
  When DFlash2 is chosen MTP is not drafted (sequential steps cost time), so MTP is scored only when it ran;
- both: start from the type rule's arm, then the full-feedback bandit.
Lines (NewIdeas): PASS >= +0.4 kept on chat (multi) or agent with code / math >= -0.1; past-only rules only (the type
rule counts as past-only if the text type is known from the prompt).
Usage: taskpolicy -b nice -n 10 [BigMoE venv] python sticky/d19_switch.py -> data/d19_switch.txt"""
import collections, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d18_hybrid import texts, TD

CHAT = {"ifeval", "multi", "agent"}

def kM(out, mtp, p, K=15):
    n = 0
    while n < K and p + n < len(out) and mtp[p][n] >= 0 and mtp[p][n] == out[p + n]:
        n += 1
    return n

class Bandit:
    def __init__(self, W, full, start):
        self.W, self.full, self.arm, self.t, self.order = W, full, start, 0, (start, 1 - start)
        self.win = [collections.deque(maxlen=W), collections.deque(maxlen=W)]
        self.last = [0, 0]
    def pick(self):
        self.t += 1
        for a in self.order:
            if not self.win[a] or self.t - self.last[a] > 4 * self.W:
                return a
        m = [sum(w) / len(w) for w in self.win]
        return int(m[1] > m[0]) if m[1] != m[0] else self.arm
    def update(self, arm, k_arm, k_other_capped):
        self.win[arm].append(k_arm); self.last[arm] = self.t; self.arm = arm
        if self.full and arm == 1 and k_other_capped is not None:   # DFlash2 scored on the verified tokens too
            self.win[0].append(k_other_capped); self.last[0] = self.t

def walk(T, make):
    tok = passes = used = 0
    for out, mtp, kD in T:
        r = make()
        p, L = 1, len(out)
        while p < L:
            arm = r.pick()
            kd, km = min(kD.get(p, 0), L - p), min(kM(out, mtp, p), L - p)
            k = km if arm else kd
            r.update(arm, k, min(kd, k + 1) if arm else None)
            tok += k + 1; passes += 1; used += arm; p += k + 1
    return tok / passes, used / passes

class Fixed:
    def __init__(self, arm): self.arm = arm
    def pick(self): return self.arm
    def update(self, *a): pass

lines = []
def say(s):
    print(s, flush=True); lines.append(s)
say("D-19 SWITCH: past-only rules between the native MTP chain (K15) and DFlash2 L1-20; tokens per pass (MTP share)")
res = {}
for dom in ("code", "gsm", "ifeval", "multi", "agent"):
    T = texts(dom)
    d0, _ = walk(T, lambda: Fixed(0))
    m0, _ = walk(T, lambda: Fixed(1))
    row = {"DFlash2": (d0, 0.0), "MTP": (m0, 1.0), "type": walk(T, lambda: Fixed(int(dom in CHAT)))}
    for W in (2, 4, 8):
        row[f"bandit-p W{W}"] = walk(T, lambda W=W: Bandit(W, False, 0))
        row[f"bandit-f W{W}"] = walk(T, lambda W=W: Bandit(W, True, 0))
        row[f"both W{W}"] = walk(T, lambda W=W: Bandit(W, True, int(dom in CHAT)))
    res[dom] = row
    say(f"  {dom} ({len(T)}): " + "; ".join(f"{k} {v:.2f} ({v - d0:+.2f}, {u:.0%})" for k, (v, u) in row.items()))
say("Lines: PASS >= +0.4 on multi or agent with code / gsm >= -0.1. Per rule:")
for rule in res["code"]:
    if rule in ("DFlash2", "MTP"):
        continue
    g = {d: res[d][rule][0] - res[d]["DFlash2"][0] for d in res}
    ok = max(g["multi"], g["agent"]) >= 0.4 and min(g["code"], g["gsm"]) >= -0.1
    say(f"  {rule}: " + " ".join(f"{d} {g[d]:+.2f}" for d in g) + (" -> PASS" if ok else ""))
open(os.path.join(TD, "d19_switch.txt"), "w").write("\n".join(lines) + "\n")
