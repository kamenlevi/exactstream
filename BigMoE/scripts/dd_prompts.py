#!/usr/bin/env python3
"""DEEPDRAFT capture prompts (Thu 12 Nov; the orchestrator's OK 00:1x, under Kamen's 11-07 standing download OK).
Source: Magicoder-OSS-Instruct-75K (HF ise-uiuc, revision 5f839b1f368a76b161028bb9edff055db34022b2, MIT licence; file
data-oss_instruct-decontaminated.jsonl, 203,201,169 bytes, sha256 f988f14d3516a9f958eeea310f63903727c6d3463bc9cb435288b6ca4bb09cf9),
kept in ~/datasets/deepdraft/magicoder/ (outside every git repo and the public sync).
Selection, deterministic:
  1. Python only (lang == python), problem 200-3,000 chars, identical problems kept once.
  2. Eval hygiene against EvalSuite's local items, the sets we measure on:
     - the sets: HumanEval (dec_humaneval / dec_humanevalt / dec_humanevalx), MBPP (dec_mbpp), LCB v6 (dec_lcb6 and
       data/raw/lcb6/test6.jsonl);
     - drop a problem that shares any word 10-gram (lowercase alphanumeric words) with any eval text (prompt, user turn,
       MBPP text / code / tests, LCB question / starter code);
     - drop a problem that names an eval function as a whole word (entry points, fn_names, defs; names of >= 7 chars,
       minus a short generic stoplist).
  3. Shuffled with seed 1112: the capture takes them in that order.
Writes ~/datasets/deepdraft/magicoder/prompts_py_v1.jsonl ({key, problem}) and the counts to
~/BigMoE/data/deepdraft/prompts_py_v1_summary.json.
usage: ~/BigMoE/.venv/bin/python -I scripts/dd_prompts.py"""
import hashlib, json, os, random, re

H = os.path.expanduser
SRC = H("~/datasets/deepdraft/magicoder/data-oss_instruct-decontaminated.jsonl")
OUT = H("~/datasets/deepdraft/magicoder/prompts_py_v1.jsonl")
SUM = H("~/BigMoE/data/deepdraft/prompts_py_v1_summary.json")
ITEMS = H("~/EvalSuite/data/items")
N = 10
STOP = {"solution", "function", "process", "calculate", "compute", "helper", "result", "answer", "count_pairs",
        "maximum", "minimum", "is_valid", "is_prime", "reverse", "fibonacci", "factorial"}
WORD = re.compile(r"[a-z0-9]+")


def grams(text):
    w = WORD.findall(text.lower())
    return {" ".join(w[i:i + N]) for i in range(len(w) - N + 1)}


def main():
    ev_txt, ev_ids, n_ev = [], set(), {}
    for nm in ("dec_humaneval", "dec_humanevalt", "dec_humanevalx", "dec_mbpp", "dec_lcb6"):
        rows = [json.loads(l) for l in open(os.path.join(ITEMS, nm + ".jsonl"))]
        n_ev[nm] = len(rows)
        for r in rows:
            for k in ("prompt", "user", "text", "ref_code", "starter_code"):
                if isinstance(r.get(k), str):
                    ev_txt.append(r[k])
            if isinstance(r.get("test_list"), list):
                ev_txt.extend(t for t in r["test_list"] if isinstance(t, str))
            for k in ("entry_point", "fn_name"):
                if isinstance(r.get(k), str) and r[k]:
                    ev_ids.add(r[k])
    raw = [json.loads(l) for l in open(H("~/EvalSuite/data/raw/lcb6/test6.jsonl"))]
    n_ev["lcb6_raw"] = len(raw)
    for r in raw:
        for k in ("question_title", "question_content", "starter_code"):
            if isinstance(r.get(k), str):
                ev_txt.append(r[k])
    for t in ev_txt:
        ev_ids.update(re.findall(r"def\s+([A-Za-z_][A-Za-z0-9_]*)", t))
    ev_ids = {i for i in ev_ids if len(i) >= 7 and i.lower() not in STOP}
    G = set()
    for t in ev_txt:
        G |= grams(t)
    id_re = re.compile(r"\b(" + "|".join(sorted(map(re.escape, ev_ids))) + r")\b") if ev_ids else None
    rows = [json.loads(l) for l in open(SRC)]
    c = dict(source_rows=len(rows))
    py = [r for r in rows if r.get("lang") == "python"]
    c["python"] = len(py)
    py = [r for r in py if 200 <= len(r["problem"]) <= 3000]
    c["length_200_3000"] = len(py)
    seen, uniq = set(), []
    for r in py:
        h = hashlib.sha1(r["problem"].encode()).hexdigest()
        if h not in seen:
            seen.add(h); uniq.append(r)
    c["unique"] = len(uniq)
    keep, drop_g, drop_i = [], 0, 0
    for r in uniq:
        if grams(r["problem"]) & G:
            drop_g += 1; continue
        if id_re is not None and id_re.search(r["problem"]):
            drop_i += 1; continue
        keep.append(r)
    c.update(dropped_10gram=drop_g, dropped_eval_identifier=drop_i, kept=len(keep))
    random.Random(1112).shuffle(keep)
    with open(OUT, "w") as f:
        for r in keep:
            f.write(json.dumps(dict(key=f"mc_{r['index']}", problem=r["problem"])) + "\n")
    c.update(eval_items=n_ev, eval_texts=len(ev_txt), eval_10grams=len(G), eval_identifiers=len(ev_ids), seed=1112, ngram=N,
             out_sha256=hashlib.sha256(open(OUT, "rb").read()).hexdigest())
    os.makedirs(os.path.dirname(SUM), exist_ok=True)
    json.dump(c, open(SUM, "w"), indent=1)
    print(json.dumps(c))


if __name__ == "__main__":
    main()
