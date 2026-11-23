"""CHAN helper: the coupled arm's per-token classes (MoonShots' J(CC) exact arm0, read-only) -> the class of each
check-ending commit and the class counts, with the same tracker logic as sb/chan.py (MoonShots' jcc_decode.where).
Check i commits checks[i] tokens after the prefill's first token; its last commit (a miss, or the bonus at 32) sits at
answer position sum(checks[:i + 1]). Needs only the tokenizer. -> results/chan/arm0_cls.json
usage: ~/BigMoE/.venv/bin/python -I -B sb/chan_cls.py"""
import json, os, sys
H = os.path.expanduser
sys.path.insert(0, H("~/Sandbox98")); sys.path.insert(0, H("~/MoonShots/week98"))
from span_tracker import SpanTracker                      # noqa: E402
from sb.chan import where                                 # noqa: E402


def main():
    from transformers import AutoTokenizer
    from sb import core as C
    tok = AutoTokenizer.from_pretrained(C.MODEL)

    class Lazy(dict):
        def __missing__(self, k):
            v = tok.decode([int(k)]); self[k] = v; return v
    table = Lazy()
    out = {}
    for l in open(H("~/MoonShots/week98/jcc/arm0.jsonl")):
        r = json.loads(l)
        tr = SpanTracker(table); cls = []
        for x in r["ids"]:
            cls.append(where(tr, table[x])); tr.update(x)
        ends, pos = [], 0
        for c in r["checks"]:
            pos += c
            ends.append(cls[pos] if c < 32 and pos < len(cls) else "bonus")
        cnt = {c: cls.count(c) for c in set(cls)}
        assert cnt == {k: v for k, v in r["cls"].items() if v}, (r["key"], cnt, r["cls"])
        out[r["key"]] = {"end_cls": ends, "cls": cnt}
    os.makedirs(H("~/Sandbox98/results/chan"), exist_ok=True)
    json.dump(out, open(H("~/Sandbox98/results/chan/arm0_cls.json"), "w"))
    print(f"[cls] {len(out)} keys; class counts equal arm0's own for every key")


if __name__ == "__main__":
    main()
