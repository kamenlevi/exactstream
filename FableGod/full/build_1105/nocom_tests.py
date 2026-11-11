"""nocom_tests.py (Fable FULL, LOSSY350 T-B1): the HumanEval+ tests of the qall and comment-free arms' answers, with
EvalSuite's own scorer (evalsuite.decode_report.score_humaneval: evalplus base + plus tests, the code block of the answer).
CPU. It runs the model's generated code inside evalplus's checker, exactly as EvalSuite's reports do.
usage (EvalSuite's venv, which has evalplus): ~/EvalSuite/.venv/bin/python nocom_tests.py <C> [--dir ~/BigMoE/data/r57]
                                              [--out ~/BigMoE/data/fri/nocom_tests.json] [--tags fqall,fqallb,fqallp,fqallpb]
Writes {tag: {key: true / false}}; nocom_report.py reads it (only tasks that ended before the cap are compared there)."""
import json, os, sys

H = os.path.expanduser("~")
sys.path.insert(0, f"{H}/EvalSuite")


def arg(name, default):
    return next((sys.argv[i + 1] for i, x in enumerate(sys.argv) if x == name), default)


def main():
    C = sys.argv[1]
    d = os.path.expanduser(arg("--dir", f"{H}/BigMoE/data/r57"))
    out = os.path.expanduser(arg("--out", f"{H}/BigMoE/data/fri/nocom_tests.json"))
    tags = arg("--tags", "fqall,fqallb,fqallp,fqallpb").split(",")
    from tokenizers import Tokenizer
    from evalsuite import decode_report as DR
    tk = Tokenizer.from_file(f"{H}/BigMoE/models/bare/tokenizer.json")
    items = {r["key"]: r for r in map(json.loads, open(f"{H}/EvalSuite/data/items/dec_humaneval.jsonl"))}
    res = {}
    for t in tags:
        p = os.path.join(d, f"{t}_{C}_tokens.json")
        if not os.path.exists(p):
            print(f"{t}: no {p}"); continue
        toks = json.load(open(p))
        rows = {k: dict(text=tk.decode([int(x) for x in next(iter(v.values()))[-1]], skip_special_tokens=True))
                for k, v in toks.items() if k in items}
        sc = DR.score_humaneval(rows, items)
        res[t] = {k: bool(v[0]) for k, v in sc.items()}
        print(f"{t}: {sum(res[t].values())} of {len(res[t])} pass")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
