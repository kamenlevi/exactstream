"""M-33 COPYROUTE (MoonShots, Tue 2026-11-03 23:0x; pre-registered in ~/NewIdeas/QUEUE.md M-33 before any run): can the
routing recorded at an earlier position forecast a copied row's experts? For an answer row t whose last 8 tokens match an
earlier span of the window ending at row s (prompt or earlier answer), forecast row t's top-10 experts at every layer with
row s's. The forecast exists before the check starts and covers all 48 layers at once (lever 2 needs it one layer ahead).
Data: Breakthrough's 36 map windows (ids [T-1, 48, 10]: rows = positions, prompt and answer; gen = the answer's start).
Per domain: coverage = matched answer rows / answer rows; recall@10 per layer (|ids_t & ids_s| / 10) averaged over layers
4-47, split by match into the prompt vs the earlier answer; baselines on the same rows: the previous row's routing and a
random row of the window.
Line: code recall >= 0.80 over layers 4-47 AND >= 15 % of code answer rows matched: PASS; recall < 0.65 or < 5 % matched:
KILL; else BETWEEN.
usage: [BigMoE venv] python copyroute.py -> data/copyroute.txt"""
import glob, os, time
import numpy as np

CAP = os.path.expanduser("~/FableGod/decision_map/out/moonshots")
N = 8
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def recall(a, b):
    """per-layer recall@10 of b's experts against a's: [48]"""
    return np.array([len(set(a[l].tolist()) & set(b[l].tolist())) / 10 for l in range(a.shape[0])])


def main():
    t0 = time.time()
    rng = np.random.default_rng(20261103)
    say(f"== M-33 COPYROUTE ({time.strftime('%a %H:%M')}): Breakthrough's 36 map windows, {N}-gram matches, recall@10 per layer")
    res = {}
    for dom in ("code", "ifeval", "multi"):
        R = {k: [] for k in ("prompt", "answer", "prev", "rand")}
        rows = matched = 0
        for f in sorted(glob.glob(os.path.join(CAP, f"tf_{dom}_*.npz"))):
            z = np.load(f)
            tok, ids, g = z["tokens"], z["ids"], int(z["gen"])
            last = {}
            for t in range(N - 1, ids.shape[0]):
                key = tuple(tok[t - N + 1:t + 1].tolist())
                if t >= g - 1:                                   # answer rows (the anchor row predicts the first answer token)
                    rows += 1
                    if key in last:
                        s = last[key]
                        matched += 1
                        R["prompt" if s < g - 1 else "answer"].append(recall(ids[t], ids[s]))
                        R["prev"].append(recall(ids[t], ids[t - 1]))
                        r = int(rng.integers(0, ids.shape[0] - 1))
                        R["rand"].append(recall(ids[t], ids[r if r != t else t - 1]))
                last[key] = t
        res[dom] = R
        mp, ma = len(R["prompt"]), len(R["answer"])
        avg = lambda k: float(np.mean(np.array(R[k])[:, 4:])) if R[k] else float("nan")
        both = R["prompt"] + R["answer"]
        rb = float(np.mean(np.array(both)[:, 4:])) if both else float("nan")
        say(f"  {dom:6s}: {matched} of {rows} answer rows matched ({100 * matched / max(1, rows):.1f} %: {mp} into the prompt, "
            f"{ma} into the earlier answer); recall@10, layers 4-47: copy forecast {rb:.3f} (prompt {avg('prompt'):.3f}, "
            f"answer {avg('answer'):.3f}); baselines on the same rows: previous row {avg('prev'):.3f}, random row {avg('rand'):.3f}")
        if both:
            pl = np.mean(np.array(both), 0)
            say(f"          per layer (copy forecast): " + " ".join(f"L{l} {pl[l]:.2f}" for l in range(0, 48, 6)))
        res[dom] = (matched / max(1, rows), rb)
    cov, rc = res["code"]
    say(f"  -> code: {100 * cov:.1f} % of answer rows matched, recall {rc:.3f} -> "
        + ("PASS" if rc >= 0.80 and cov >= 0.15 else "KILL" if rc < 0.65 or cov < 0.05 else "BETWEEN")
        + f" (line 0.80 / 15 %; KILL < 0.65 or < 5 %)  ({time.time() - t0:.0f} s)")
    open(os.path.expanduser("~/MoonShots/data/copyroute.txt"), "a").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
