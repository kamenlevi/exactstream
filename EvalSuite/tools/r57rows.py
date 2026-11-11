"""BigMoE's r57_screen_fast.py tokens -> EvalSuite decode rows, for any decode set (tonight's same-scores run on the FAST
stack; tools/agentfast.py's rows step does the same for dec_agentlong alone).
  .venv/bin/python tools/r57rows.py <tokens.json> <run> [arm] [--append]
Per key: the arm's first run's ids, cut after the first EOS (kept, as decode rows end) or at the item's max_tokens; finish
eos / length; set and n_prompt from the item. --append adds to results/<run>/decode.jsonl (keys already there are kept).
A run the screen cut at its time budget (HARD; `cut` in the <tag>_runs.json beside the tokens) is not a row: it is left
out, so its key re-runs in the next process (Mon 11-09: 10 such runs had been kept as "length" rows in card_lcoh_fwd)."""
import glob, json, os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from evalsuite import data as D

EOS = {248044, 248046}


def main(tok_path, run, arm=None, append=False):
    its = {}
    for p in glob.glob(os.path.join(D.ITEMS, "dec_*.jsonl")):
        for it in D.load(os.path.basename(p)[:-len(".jsonl")]):
            its.setdefault(it["key"], it)
    tok = json.load(open(os.path.expanduser(tok_path)))
    path = os.path.join(ROOT, "results", run, "decode.jsonl")
    have = {json.loads(l)["key"] for l in open(path)} if append and os.path.exists(path) else set()
    runs_path = tok_path[:-len("_tokens.json")] + "_runs.json" if tok_path.endswith("_tokens.json") else ""
    cutrun = {}                                           # (key, arm) -> the first run's cut flag
    if os.path.exists(os.path.expanduser(runs_path)):
        for r in json.load(open(os.path.expanduser(runs_path))):
            cutrun.setdefault((r["key"], r["arm"]), bool(r.get("cut")))
    else:
        print(f"WARNING: no runs file beside {os.path.basename(tok_path)}: runs cut at the time budget cannot be told apart")
    out, dropped = [], []
    for key, by_arm in tok.items():
        if key in have:
            continue
        assert key in its, f"no decode item {key}"
        arms = sorted(by_arm)
        a = arm or (arms[0] if len(arms) == 1 else None)
        assert a in by_arm, f"{key}: arms {arms}; name one"
        if cutrun.get((key, a)):
            dropped.append(key)
            continue
        ids = list(by_arm[a][0])
        cap = its[key]["max_tokens"]
        cut = next((i + 1 for i, t in enumerate(ids) if t in EOS), None)
        ids = ids[:cut] if cut is not None and cut <= cap else ids[:cap]
        out.append(dict(key=key, set=its[key]["set"], completion_ids=ids,
                        finish="eos" if ids and ids[-1] in EOS else "length", n_prompt=len(its[key]["prompt_ids"]),
                        source=f"BigMoE r57_screen_fast {os.path.basename(tok_path)} arm {a}"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a" if append else "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"results/{run}/decode.jsonl: {len(out)} rows from {os.path.basename(tok_path)}"
          f" ({sum(r['finish'] == 'eos' for r in out)} end in EOS){' (appended)' if append else ''}"
          + (f"; {len(dropped)} cut at the time budget, not kept: {' '.join(dropped)}" if dropped else ""))


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--append"]
    main(a[0], a[1], a[2] if len(a) > 2 else None, "--append" in sys.argv)
