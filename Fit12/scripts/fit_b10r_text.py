"""Fit12 B10-r texts (2026-10-25; NewIdeas' B10-r in ~/NewIdeas/QUEUE.md: is B10a's held-out failure domain shift?).
Calibration parts R0..R9, 4,096 tokens each: R0..R7 = FIT (32,768 tokens), R8..R9 = VALIDATION (for the H-damping
choice only). Each part is a round-robin over 7 domains. Nothing in them is an Eval-suite pick, and nothing is in judge B
(the suite's tf_gen prompts + the reference model's completions). The suite's picks are reproduced from
~/EvalSuite/evalsuite/data.py and asserted against its item files.
  math   GSM8K test 0-599 (tf uses 600-659, dec 700-724) + MATH-500 minus dec_math's 25: question, reference solution
  code   HumanEval+ minus the suite's 84 picks (tf 24 + dec 60): prompt, canonical solution
  instr  IFEval minus the suite's 130 picks: the prompt
  multi  MGSM (de es fr ja ru sw zh) minus the 80 problem ids dec_mgsm uses in any language: question, answer
  wiki   Bulgarian Wikipedia (data/raw) after the suite's first 65,536 tokens, 300-token chunks (raw text)
  know   MMLU-Pro test minus the suite's picks: question, options, answer letter
  agent  Kamen's own Claude Code sessions (~/.claude/projects/-Users-kamen-OjoX), only messages after route4 was built
         (2026-10-17 14:42 UTC; dec_agent = all 29 route4 turns); system reminders stripped; 400-token chunks.
         FIT and VALIDATION use different sessions.
Q&A items use the model's chat format, like judge B. U = the FRAGILITY judge, a domain absent from the calibration:
English literary prose (Pride and Prejudice, tokens 70,000-74,095; the suite's tf_doc uses 0-65,535), raw text.
Writes /Users/kamen/Fit12/data/b10r/texts/{R0..R9,U}.json + prints the composition and a 32-gram overlap check vs B.
Usage: ~/BigMoE/.venv/bin/python -B fit_b10r_text.py   (CPU, 1 core, ~1 min)"""
import glob, json, os, random, re
from transformers import AutoTokenizer

H = os.path.expanduser
RAW, ITEMS, OUT = H("~/EvalSuite/data/raw"), H("~/EvalSuite/data/items"), H("~/Fit12/data/b10r/texts")
SESS = H("~/.claude/projects/-Users-kamen-OjoX")
AGENT_FIT = ("45a09590", "d1c40623", "2d8d31fe", "cce44250")      # Wine / other / OjoX repo study / image session
AGENT_VAL = ("5d99dc09", "7fd9135b", "e1d93b2d")                  # network / TV / SSD totals
CUT = "2026-10-17T15:00"
NPART, NFIT, T = 10, 8, 4096
tok = AutoTokenizer.from_pretrained(H("~/BigMoE/models/bare"))
enc = lambda s: tok.encode(s, add_special_tokens=False)
jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]


def chat(user, answer=None):
    s = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    return enc(s + (f"{answer}<|im_end|>\n" if answer is not None else ""))


def keys(name):
    return {r["key"] for r in jl(f"{ITEMS}/{name}.jsonl")}


# ---- the suite's picks, reproduced and checked
tfk = keys("tf_prompts")
rng = random.Random(77)
ife = jl(f"{RAW}/ifeval_input_data.jsonl")
ife_pick = rng.sample(range(len(ife)), 130)
he = sorted((r["task_id"], r) for r in jl(H("~/Library/Caches/evalplus/HumanEvalPlus-v0.1.10.jsonl")))
assert len(he) == 164
he_pick = rng.sample(range(len(he)), 84)
assert {f"tf_ifeval/{ife[j]['key']}" for j in ife_pick[:30]} == {k for k in tfk if k.startswith("tf_ifeval/")}
assert {f"ifeval/{ife[j]['key']}" for j in ife_pick[30:]} == keys("dec_ifeval")
assert {f"tf_code/{he[j][0]}" for j in he_pick[:24]} == {k for k in tfk if k.startswith("tf_code/")}
assert {f"humaneval/{he[j][0]}" for j in he_pick[24:84]} == keys("dec_humaneval")
m5 = jl(f"{RAW}/math500_test.jsonl")
m5_pick = {p["unique_id"] for p in random.Random(5).sample([p for p in m5 if int(p.get("level", 0)) >= 3], 25)}
dm = keys("dec_math")
assert {f"math500/{u}" for u in m5_pick} == {k for k in dm if k.startswith("math500/")}
assert {f"gsm8k/{i}" for i in range(700, 725)} == {k for k in dm if k.startswith("gsm8k/")}
assert {f"tf_gsm/{i}" for i in range(600, 660)} == {k for k in tfk if k.startswith("tf_gsm/")}
mg_used = set(random.Random(23).sample(range(250), 250)[:80])
assert {int(re.sub(r"^mgsm/[a-z]+", "", k)) for k in keys("dec_mgsm")} == mg_used
mm_pick = {int(k.split("/")[1]) for k in keys("mmlu_pro")}
print(f"[text] suite picks reproduced: ifeval 130, humaneval 84, math500 25, gsm8k 600-659 + 700-724, mgsm 80 ids, "
      f"mmlu_pro {len(mm_pick)}", flush=True)

# ---- domain pools (token lists), seeded shuffles
R = random.Random(2510)
gsm = jl(f"{RAW}/gsm8k_test.jsonl")[:600]
math = [chat(p["question"], p["answer"]) for p in gsm] + [chat(p["problem"], p["solution"]) for p in m5 if p["unique_id"] not in m5_pick]
code = [chat("Complete the following Python function.\n\n```python\n" + r["prompt"] + "```",
             "```python\n" + r["prompt"] + r["canonical_solution"] + "```") for j, (_, r) in enumerate(he) if j not in set(he_pick)]
instr = [chat(ife[j]["prompt"]) for j in range(len(ife)) if j not in set(ife_pick)]
multi = []
for lang in ("de", "es", "fr", "ja", "ru", "sw", "zh"):
    for i, line in enumerate(open(f"{RAW}/mgsm/mgsm_{lang}.tsv", encoding="utf-8")):
        q, a = line.rstrip("\n").split("\t")
        if i not in mg_used:
            multi.append(chat(q, a.strip()))
know = []
for r in jl(f"{RAW}/mmlu_pro_test.jsonl"):
    if r["question_id"] not in mm_pick:
        opts = "\n".join(f"{'ABCDEFGHIJ'[i]}. {o}" for i, o in enumerate(r["options"]))
        know.append(chat(f"{r['question']}\n\n{opts}", f"The answer is ({r['answer']})."))
for p in (math, code, instr, multi, know):
    R.shuffle(p)
wid = enc(open(f"{RAW}/doc_bg_wikipedia.txt", encoding="utf-8").read())
wiki = [wid[i:i + 300] for i in range(70000, len(wid) - 300, 300)]
wiki_fit, wiki_val = wiki[:len(wiki) // 2], wiki[len(wiki) // 2:]


def session(sid):
    f, = glob.glob(f"{SESS}/{sid}*.jsonl")
    msgs = []
    for line in open(f, errors="replace"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("type") not in ("user", "assistant") or r.get("timestamp", "") < CUT or r.get("isSidechain"):
            continue
        m = r.get("message") or {}
        c, parts = m.get("content"), []
        for b in ([{"type": "text", "text": c}] if isinstance(c, str) else c if isinstance(c, list) else []):
            t = b.get("type")
            if t == "text":
                parts.append(re.sub(r"<system-reminder>.*?</system-reminder>", "", b.get("text", ""), flags=re.S))
            elif t == "tool_use":
                parts.append("<tool_call>\n" + json.dumps({"name": b.get("name"), "arguments": b.get("input")}, ensure_ascii=False)[:1500] + "\n</tool_call>")
            elif t == "tool_result":
                cc = b.get("content")
                s = cc if isinstance(cc, str) else "\n".join(x.get("text", "") for x in cc or [] if isinstance(x, dict) and x.get("type") == "text")
                parts.append("<tool_response>\n" + re.sub(r"<system-reminder>.*?</system-reminder>", "", s, flags=re.S)[:1500] + "\n</tool_response>")
        text = "\n".join(p.strip() for p in parts if p.strip())
        if not text:
            continue
        role = m.get("role", r["type"])
        if msgs and msgs[-1][0] == role:
            msgs[-1][1] += "\n" + text
        else:
            msgs.append([role, text])
    ids = enc("".join(f"<|im_start|>{ro}\n{tx}<|im_end|>\n" for ro, tx in msgs))
    return [ids[i:i + 400] for i in range(0, len(ids) - 400, 400)]


agent_fit = [c for s in AGENT_FIT for c in session(s)]
agent_val = [c for s in AGENT_VAL for c in session(s)]
R.shuffle(agent_fit)
R.shuffle(agent_val)
print("[text] pool items: " + ", ".join(f"{n} {len(p)}" for n, p in (("math", math), ("code", code), ("instr", instr),
      ("multi", multi), ("wiki fit/val", wiki_fit), ("", wiki_val), ("know", know), ("agent fit", agent_fit), ("agent val", agent_val))), flush=True)

os.makedirs(OUT, exist_ok=True)
DOM = ("math", "code", "instr", "multi", "wiki", "know", "agent")
for p in range(NPART):
    val = p >= NFIT
    pools = {"math": math, "code": code, "instr": instr, "multi": multi, "know": know,
             "wiki": wiki_val if val else wiki_fit, "agent": agent_val if val else agent_fit}
    ids, comp = [], dict.fromkeys(DOM, 0)
    while len(ids) < T:
        for d in DOM:
            if len(ids) >= T:
                break
            x = pools[d].pop()[:T - len(ids)]
            ids += x
            comp[d] += len(x)
    json.dump({"name": f"R{p}", "role": "validation" if val else "fit", "ids": ids, "tokens_by_domain": comp},
              open(f"{OUT}/R{p}.json", "w"))
    print(f"[text] R{p} ({'val' if val else 'fit'}): {len(ids)} tokens | " + " ".join(f"{d} {comp[d]}" for d in DOM), flush=True)
pp = open(f"{RAW}/doc_en_pride_prejudice.txt", encoding="utf-8").read()
a = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n", pp)
b = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK", pp)
uid = enc(pp[a.end() if a else 0:b.start() if b else len(pp)].strip())
assert len(uid) >= 74096, len(uid)
json.dump({"name": "U", "role": "fragility judge", "ids": uid[70000:74096]}, open(f"{OUT}/U.json", "w"))
print(f"[text] U: 4096 tokens of English literary prose (of {len(uid)})", flush=True)

# ---- overlap check vs judge B (32-grams of the suite's tf_gen prompts + completions)
bg = set()
for r in jl(H("~/EvalSuite/cache/ref/tf_gen.jsonl")):
    s = r["prompt_ids"] + r["completion_ids"]
    bg.update(tuple(s[i:i + 32]) for i in range(len(s) - 31))
for nm in [f"R{p}" for p in range(NPART)] + ["U"]:
    s = json.load(open(f"{OUT}/{nm}.json"))["ids"]
    hit = sum(tuple(s[i:i + 32]) in bg for i in range(len(s) - 31))
    print(f"[text] {nm}: {hit} of {len(s) - 31} 32-grams also in judge B", flush=True)
print("[text] done", flush=True)
