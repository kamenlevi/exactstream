"""The agent check (docs/PREREG_agentfast_2026-11-06.md): plain FAST on dec_agentlong's 17 turns, two prompt orders.
  .venv/bin/python tools/agentfast.py rows <R57_TAG> <run> [arm]   # BigMoE's ~/BigMoE/data/r57/<TAG>_tokens.json -> results/<run>/decode.jsonl
  .venv/bin/python tools/agentfast.py report <run>,<run> [--out FILE]
rows: per key the arm's first run's ids, cut after the first EOS (kept, as decode rows end) or at the item's max_tokens;
finish eos / length. report: the line on the runs' tfwin scores (results/<run>/tfwin/, the exact reference along FAST's
own tokens): PASS = 0 blunders (the reference's p of a committed token < 0.01) over every row of both orders; FAIL = 1 or
more; INCOMPLETE = a row or its score missing. Information: deviations (committed != the reference's argmax) per order
and row, where each blunder sits in the tool call (the first parameter = the command / code, or later text), and token
agreement between the two orders. Last line `AGENTFAST: PASS | FAIL | INCOMPLETE`; exit 0 PASS, 1 FAIL, 2 INCOMPLETE."""
import gzip, json, math, os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from evalsuite import data as D

RES = os.path.join(ROOT, "results")
EOS = {248044, 248046}
LN_P01 = math.log(0.01)


def items():
    return {it["key"]: it for it in D.load("dec_agentlong")}


def rows(tag, run, arm=None):
    tok = json.load(open(os.path.expanduser(f"~/BigMoE/data/r57/{tag}_tokens.json")))
    its = items()
    out = []
    for key, by_arm in tok.items():
        if key not in its:
            continue
        arms = sorted(by_arm)
        a = arm or (arms[0] if len(arms) == 1 else None)
        assert a in by_arm, f"{key}: arms {arms}; name one"
        ids = list(by_arm[a][0])
        cap = its[key]["max_tokens"]
        cut = next((i + 1 for i, t in enumerate(ids) if t in EOS), None)
        ids = ids[:cut] if cut is not None and cut <= cap else ids[:cap]
        out.append(dict(key=key, set="agentlong", completion_ids=ids, finish="eos" if ids and ids[-1] in EOS else "length",
                        n_prompt=len(its[key]["prompt_ids"]), source=f"BigMoE r57_screen_fast {tag} arm {a}"))
    d = os.path.join(RES, run)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "decode.jsonl"), "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"results/{run}/decode.jsonl: {len(out)} rows from {tag} ({sum(r['finish'] == 'eos' for r in out)} end in EOS)")
    return len(out)


def tokenizer():
    try:
        from tokenizers import Tokenizer
        t_ = Tokenizer.from_file(os.path.expanduser("~/BigMoE/models/bare/tokenizer.json"))
        return lambda i: t_.decode([int(i)])
    except Exception:
        return None


def report(runs, outfile=None):
    its, tok, L = items(), tokenizer(), []
    say = L.append
    complete, blunders, per_run = True, [], {}
    for run in runs:
        rows_ = {r["key"]: r for r in map(json.loads, open(os.path.join(RES, run, "decode.jsonl")))}
        missing = sorted(set(its) - set(rows_))
        sc = {}
        for k in rows_:
            p = os.path.join(RES, run, "tfwin", k.replace("/", "__") + ".json.gz")
            if os.path.exists(p):
                sc[k] = json.load(gzip.open(p, "rt"))
        unscored = sorted(set(rows_) - set(sc))
        if missing or unscored:
            complete = False
            say(f"{run}: rows missing {missing}, scores missing {unscored}")
        n_tok = dev = 0
        for k in sorted(sc):
            out, s = rows_[k]["completion_ids"], sc[k]
            d_ = sum(x != y for x, y in zip(s["am"], out))
            b_ = [(i, out[i], math.exp(lp)) for i, lp in enumerate(s["lp"]) if lp < LN_P01]
            n_tok += len(out)
            dev += d_
            for i, t, p in b_:
                where = ""
                if tok:
                    where = (" (after the first parameter)" if "</parameter>" in "".join(tok(x) for x in out[:i])
                             else " (in the first parameter: the command / code)")
                blunders.append((run, k, i, t, p, where))
            say(f"{run} {k}: {len(out)} tokens ({rows_[k]['finish']}), {d_} not the reference's argmax "
                f"({100 * d_ / max(len(out), 1):.2f} %), {len(b_)} blunder(s)")
        per_run[run] = (rows_, n_tok, dev)
        say(f"{run}: {len(sc)} scored rows, {n_tok} tokens, deviations {dev} ({100 * dev / max(n_tok, 1):.2f} %)")
    say("\n== blunders (the reference's p of the committed token < 0.01)")
    for run, k, i, t, p, where in blunders:
        say(f"{run} {k} @{i}: token {t}{(' ' + repr(tok(t))) if tok else ''}, p {p:.5f}{where}")
    if not blunders:
        say("none")
    if len(runs) == 2 and all(r in per_run for r in runs):
        a_, b_ = per_run[runs[0]][0], per_run[runs[1]][0]
        same = [k for k in a_ if k in b_ and a_[k]["completion_ids"] == b_[k]["completion_ids"]]
        say(f"\n== the two orders (information): {len(same)} of {len(set(a_) & set(b_))} rows token-identical")
        for k in sorted(set(a_) & set(b_)):
            x, y = a_[k]["completion_ids"], b_[k]["completion_ids"]
            if x != y:
                fd = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))
                say(f"  {k}: first difference at token {fd} ({len(x)} vs {len(y)} tokens)")
    v = "INCOMPLETE" if not complete else ("FAIL" if blunders else "PASS")
    say(f"\nAGENTFAST: {v} ({len(blunders)} blunder(s) over {len(runs)} order(s))")
    text = "\n".join(L)
    if outfile:
        open(outfile, "w").write(text + "\n")
    print(text)
    return {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2}[v]


if __name__ == "__main__":
    if sys.argv[1] == "rows":
        rows(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
    else:
        args = sys.argv[2:]
        out = args[args.index("--out") + 1] if "--out" in args else None
        sys.exit(report(args[0].split(","), out))
