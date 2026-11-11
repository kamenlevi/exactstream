"""NewIdeas D-20 pre-check (Fri 10-30, CPU only): can a simple rule on the prompt pick the drafter up front?
D-19: the native MTP chain wins on chat / instructions / agent turns, DFlash2 on code and math. The engine has only the
prompt when it starts. This labels every Eval-suite decode prompt with a keyword rule and scores it against the set's own
type. Sets: humaneval -> code, math -> math, mgsm -> math (non-English, reported apart: D-19 has no data there), ifeval -> chat,
multi -> chat (non-English), agent -> agent. Drafter: code / math -> DFlash2, chat / agent -> MTP.
Usage: python -I type_rule.py <tokenizer.json> <items dir>"""
import json, re, sys
from tokenizers import Tokenizer

tok = Tokenizer.from_file(sys.argv[1])
ITEMS = sys.argv[2]
SETS = {"humaneval": "code", "math": "math", "mgsm": "math", "ifeval": "chat", "multi": "chat", "agent": "agent"}
DRAFTER = {"code": "dflash2", "math": "mtp", "chat": "mtp", "agent": "mtp"}  # D-20b (Sun 08:0x): math moved to MTP K7

CODE = re.compile(r"(^|\n)\s*(def |class |import |from \w+ import|#include|fn |func |public |const |let |var )|```|\breturn\b|\w+\(.*\)\s*:|>>>")
MATH = re.compile(r"\\boxed|\\frac|\\sqrt|\$[^$]+\$|\b(solve|compute|calculate|how many|how much|what is the (sum|product|value|area|probability))\b"
                  r"|\d+\s*[-+*/^=]\s*\d+", re.I)
TOOLS = re.compile(r"<tools>|\"type\":\s*\"function\"|tool_call|<function|\bfunctions?\b.*\bparameters\b", re.I | re.S)

def text_of(item):
    if item.get("user"):
        return item.get("prompt") or item["user"], item["user"]
    full = tok.decode(item["prompt_ids"], skip_special_tokens=False)
    return full, full

def non_ascii_share(s):
    letters = [c for c in s if c.isalpha()]
    return sum(1 for c in letters if ord(c) > 127) / max(1, len(letters))

def rule(full, user):
    if TOOLS.search(full):
        return "agent"
    if CODE.search(user) or CODE.search(full[-4000:]):
        return "code"
    if MATH.search(user):
        return "math"
    return "chat"

rows = []
for s, want in SETS.items():
    for line in open(f"{ITEMS}/dec_{s}.jsonl"):
        it = json.loads(line)
        full, user = text_of(it)
        got = rule(full, user)
        rows.append((s, want, got, round(non_ascii_share(user), 2), it["key"]))

print("set        n   type-acc  drafter-acc  confusions")
for s, want in SETS.items():
    r = [x for x in rows if x[0] == s]
    ta = sum(x[2] == want for x in r) / len(r)
    da = sum(DRAFTER[x[2]] == DRAFTER[want] for x in r) / len(r)
    conf = {}
    for x in r:
        if x[2] != want:
            conf[x[2]] = conf.get(x[2], 0) + 1
    print(f"{s:10s} {len(r):3d}   {ta:6.3f}    {da:6.3f}      {conf}")
core = [x for x in rows if x[0] != "mgsm"]
print(f"all but mgsm: drafter-acc {sum(DRAFTER[x[2]] == DRAFTER[x[1]] for x in core) / len(core):.3f} over {len(core)} prompts")
bad = [x for x in core if DRAFTER[x[2]] != DRAFTER[x[1]]]
for x in bad[:12]:
    print("  wrong:", x[4], "want", x[1], "got", x[2])
