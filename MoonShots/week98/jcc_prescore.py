"""J(CC): score one or more finished arms into jcc/scores.json (cache for jcc_score.py); prints pass@1 per set.
usage: ~/EvalSuite/.venv/bin/python -I -B jcc_prescore.py arm0[,armB,...]"""
import json, os, sys
sys.path.insert(0, os.path.expanduser("~/MoonShots/week98"))
import jcc_score as S

if __name__ == "__main__":
    p = os.path.join(S.J, "scores.json")
    for name in sys.argv[1].split(","):
        rows = S.load(name)
        sc = json.load(open(p)) if os.path.exists(p) else {}
        todo = {k: r for k, r in rows.items() if k not in sc.get(name, {})}
        if todo:
            sc.setdefault(name, {}).update(S.score_all(todo))
            json.dump(sc, open(p, "w"), indent=0)
        a = sc[name]
        print(name, {g: f"{100 * sum(a[k] for k in rows if S.grp(rows[k]) == g) / max(1, sum(1 for k in rows if S.grp(rows[k]) == g)):.1f} % of {sum(1 for k in rows if S.grp(rows[k]) == g)}" for g in ("humaneval", "mbpp")}, flush=True)
