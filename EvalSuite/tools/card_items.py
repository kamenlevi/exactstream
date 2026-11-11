> (a quote from the project owner, kept private)
known benchmarks of the model"). CPU; run with BigMoE's venv for the reference tokenizer, as evalsuite/data.py:
  ~/BigMoE/.venv/bin/python tools/card_items.py
Non-thinking chat template, greedy, as every dec_ set; the sources sit in data/raw (never pushed):
  dec_ifbench  IFBench (allenai/IFBench cd189a9c2889, data/IFBench_test.jsonl): 300 prompts, the prompt as the user turn;
               max 1,536 tokens; the vendored official grader (evalsuite/ifbench), prompt-level loose (the paper's metric)
  dec_gpqad    GPQA Diamond, generative (data/raw/gpqa_diamond.csv; openai/simple-evals 652c89d0ca9d): its
               QUERY_TEMPLATE_MULTICHOICE with the options in simple-evals' first-repeat permutation (random.Random(0),
               one rng.sample(range(4), 4) per row in file order); max 2,048 tokens; ANSWER_PATTERN_MULTICHOICE (re.search)
  dec_lcb6     LiveCodeBench v6 (livecodebench/code_generation_lite 0fe84c3912ea, test6.jsonl): all 175 problems of the
               file, "window" = 2025-02-01 <= contest_date < 2025-05-01 (the Qwen cards' "v6 (25.02-25.05)", 131); LCB's
               OpenAIChat messages (SYSTEM_MESSAGE_GENERIC + get_generic_question_template_answer, lcb_runner
               28fef95ea8c9); max 2,048 tokens; graded by LCB's run_test on the public + private tests (evalsuite/lcb)
tools/card_keys.json: "keys" = the gate's keys (seeded subsets of IFBench and GPQA Diamond, the LCB window); "extend" =
the rest of each full set (IFBench 300, GPQA Diamond 198, LCB test6 175), for amendment 1's extension."""
import csv, json, os, random, sys, unicodedata
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from evalsuite import data as D

RAW = D.RAW
CAP = {"ifbench": 1536, "gpqad": 2048, "lcb6": 2048}
SUBSET = {"ifbench": 150, "gpqad": 100}       # seeded samples (SEED) of the sorted keys; lcb6: the whole window
SEED = 1108
LCB_FROM, LCB_TO = "2025-02-01", "2025-05-01"

# openai/simple-evals common.py, verbatim
QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

# LiveCodeBench lcb_runner/prompts/code_generation.py, verbatim
SYSTEM_MESSAGE_GENERIC = "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."
FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."


def lcb_user(q):    # get_generic_question_template_answer
    prompt = f"### Question:\n{q['question_content']}\n\n"
    if q["starter_code"]:
        prompt += f"### Format: {FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{q['starter_code']}\n```\n\n"
    else:
        prompt += f"### Format: {FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def chat(tok, user, system=None):   # data.chat's form, with an optional system turn
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False)
    return tok.encode(text, add_special_tokens=False)


def marks(text):    # combining marks: where MTPLX's pre-tokenizer regex and the HF tokenizer's could split differently
    return sum(unicodedata.category(c) == "Mn" for c in text)


def main():
    tok = D.tokenizer()
    out, nm = {}, {}
    rows = []
    for r in map(json.loads, open(os.path.join(RAW, "ifbench_src", "data", "IFBench_test.jsonl"))):
        rows.append(dict(key=f"ifbench/{r['key']}", set="ifbench", prompt=r["prompt"], instruction_id_list=r["instruction_id_list"],
                         kwargs=r["kwargs"], user=r["prompt"], prompt_ids=chat(tok, r["prompt"]), max_tokens=CAP["ifbench"]))
    out["ifbench"] = rows
    rows, rng = [], random.Random(0)
    with open(os.path.join(RAW, "gpqa_diamond.csv")) as f:
        for r in csv.DictReader(f):
            perm = rng.sample(range(4), 4)
            ch = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
            ch = [ch[i] for i in perm]
            user = QUERY_TEMPLATE_MULTICHOICE.format(A=ch[0], B=ch[1], C=ch[2], D=ch[3], Question=r["Question"])
            rows.append(dict(key=f"gpqad/{r['Record ID']}", set="gpqad", subject=r.get("Subdomain", ""),
                             gold="ABCD"[ch.index(r["Correct Answer"])], user=user, prompt_ids=chat(tok, user),
                             max_tokens=CAP["gpqad"]))
    out["gpqad"] = rows
    rows = []
    for r in map(json.loads, open(os.path.join(RAW, "lcb6", "test6.jsonl"))):
        user = lcb_user(r)
        rows.append(dict(key=f"lcb6/{r['question_id']}", set="lcb6", question_id=r["question_id"], platform=r["platform"],
                         difficulty=r["difficulty"], contest_date=r["contest_date"][:10],
                         window=LCB_FROM <= r["contest_date"][:10] < LCB_TO, starter_code=r["starter_code"],
                         fn_name=json.loads(r["metadata"]).get("func_name"), system=SYSTEM_MESSAGE_GENERIC, user=user,
                         prompt_ids=chat(tok, user, SYSTEM_MESSAGE_GENERIC), max_tokens=CAP["lcb6"]))
    out["lcb6"] = rows
    keys, ext = {}, {}
    for s, rs in out.items():
        assert len({r["key"] for r in rs}) == len(rs), s
        D.write("dec_" + s, rs)
        nm[s] = sum(marks(r["user"]) for r in rs)
        ks = sorted(r["key"] for r in rs)
        keys[s] = (sorted(random.Random(SEED).sample(ks, SUBSET[s])) if s in SUBSET
                   else sorted(r["key"] for r in rs if r.get("window", True)))
        ext[s] = sorted(set(ks) - set(keys[s]))
    json.dump(dict(seed=SEED, subset=SUBSET, caps=CAP, keys=keys, extend=ext),
              open(os.path.join(os.path.dirname(__file__), "card_keys.json"), "w"), indent=0)
    print("sets:", {s: len(r) for s, r in out.items()}, "gate keys:", {s: len(k) for s, k in keys.items()},
          "extension keys:", {s: len(k) for s, k in ext.items()}, "combining marks in the user turns:", nm)


if __name__ == "__main__":
    main()
