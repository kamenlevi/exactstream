"""Give decode rows their text from their completion_ids, as EvalSuite's decode stage writes it (evalsuite/decode.py and
relax.py: the tokenizer's decode of the ids without EOS, special tokens kept). tools/r57rows.py writes ids only, and the
scorers (evalsuite/decode_report.py, tools/d4x.py) read row["text"] (found Sat 2026-11-07 01:2x, before any scoring).
  .venv/bin/python tools/rows_text.py <run> [<run> ...]     # rows without text get it (results/<run>/decode.jsonl, atomic)
  .venv/bin/python tools/rows_text.py --check <run> ...     # rows WITH text: recompute; every one must be equal (exit 1 if not)"""
import json, os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
EOS = {248044, 248046}
_TOK = {}


def text(ids):
    if "t" not in _TOK:
        from tokenizers import Tokenizer
        _TOK["t"] = Tokenizer.from_file(os.path.expanduser("~/BigMoE/models/bare/tokenizer.json"))
    return _TOK["t"].decode([int(i) for i in ids if int(i) not in EOS], skip_special_tokens=False)


def main():
    check = sys.argv[1] == "--check"
    bad = 0
    for run in sys.argv[2:] if check else sys.argv[1:]:
        p = os.path.join(ROOT, "results", run, "decode.jsonl")
        rows = [json.loads(l) for l in open(p)]
        if check:
            have = [r for r in rows if r.get("text") is not None]
            diff = [r["key"] for r in have if text(r["completion_ids"]) != r["text"]]
            bad += len(diff)
            print(f"{run}: {len(have)} rows with text; recomputed equal: {len(have) - len(diff)}" + (f"; DIFFERENT: {diff[:8]}" if diff else ""))
            continue
        n = 0
        for r in rows:
            if r.get("text") is None:
                r["text"] = text(r["completion_ids"])
                r["text_from"] = "tools/rows_text.py (the ids decoded as the decode stage does)"
                n += 1
        if n:
            with open(p + ".tmp", "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)
            os.replace(p + ".tmp", p)
        print(f"{run}: {len(rows)} rows, text filled on {n}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
