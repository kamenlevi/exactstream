"""Item builders (CPU; run with BigMoE's venv for the reference tokenizer). Everything is seeded and written once to
data/items/*.jsonl, so every variant sees identical token ids.

Multiple choice (letter probability, zero-shot, non-thinking chat template, assistant prefix "The answer is ("):
  mmlu_pro  100 per subject x 14 subjects (seed 1234)        gpqa  GPQA-Diamond, 198, options shuffled (seed 0)
Teacher-forced corpus (the reference's own greedy outputs are generated once, then cached):
  tf_gsm 60 GSM8K CoTs, tf_ifeval 30, tf_code 24 HumanEval+ completions, tf_multi 24 prompts in 8 languages,
  tf_doc 4 long documents (English novel, French novel, Bulgarian Wikipedia, Python source), DOC_TOKENS each.
Decode sets (behind --decode): ifeval 100, humaneval 60, math 50 (MATH-500 levels 3-5 x 25 + GSM8K x 25), ruler 24."""
from __future__ import annotations
import csv, glob, json, os, random, re
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
ITEMS = os.path.join(ROOT, "data", "items")
REF_MODEL = os.path.expanduser("~/BigMoE/models/bare")
DOC_TOKENS = int(os.environ.get("EVALSUITE_DOC_TOKENS", "65536"))
GSM_INSTR = "Solve the problem step by step, briefly. End with a line of the form 'Answer: <number>'.\n\n"  # MoonShots'
LETTERS = "ABCDEFGHIJ"
MC_PREFIX = "The answer is ("

MULTI = [
    ("bg", "Обясни накратко как работи фотосинтезата и защо е важна за живота на Земята."),
    ("bg", "Напиши кратък разказ (около 150 думи) за котка, която се загубва в София и намира пътя към дома си."),
    ("bg", "Какви са най-добрите съвети за ученик, който иска да научи програмиране сам? Дай пет конкретни стъпки."),
    ("de", "Erkläre in einfachen Worten, wie ein Kühlschrank funktioniert."),
    ("de", "Schreibe eine kurze E-Mail an einen Vermieter, in der du eine defekte Heizung meldest."),
    ("de", "Was sind die Vor- und Nachteile von Elektroautos? Antworte in einer kurzen Liste."),
    ("fr", "Explique pourquoi le ciel est bleu, comme si tu parlais à un enfant de dix ans."),
    ("fr", "Rédige une courte recette de crêpes avec les quantités pour quatre personnes."),
    ("fr", "Quels sont les principaux événements de la Révolution française ? Résume-les en quelques phrases."),
    ("es", "¿Cómo puedo mejorar mi concentración cuando estudio? Dame consejos prácticos."),
    ("es", "Escribe un poema corto sobre el mar al atardecer."),
    ("es", "Explica la diferencia entre el clima y el tiempo meteorológico."),
    ("ru", "Объясни, что такое машинное обучение, простыми словами."),
    ("ru", "Напиши короткое письмо другу с приглашением на день рождения."),
    ("ru", "Почему зимой холоднее, чем летом? Объясни кратко."),
    ("zh", "请简单解释一下黑洞是如何形成的。"),
    ("zh", "写一段关于春天的短文，大约一百字。"),
    ("zh", "学习一门新语言有哪些有效的方法？请列出五条建议。"),
    ("ja", "日本の四季について短く説明してください。"),
    ("ja", "初心者向けに、お茶の入れ方を手順で説明してください。"),
    ("ja", "インターネットの仕組みを簡単に説明してください。"),
    ("ar", "اشرح بإيجاز أهمية الماء لجسم الإنسان."),
    ("ar", "اكتب قصة قصيرة عن صديقين يسافران إلى الصحراء."),
    ("ar", "ما هي فوائد القراءة اليومية؟ اذكر خمس فوائد."),
]


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(REF_MODEL)


def chat(tok, user, prefix="", thinking=False):
    text = tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True,
                                   enable_thinking=thinking, tokenize=False)
    return tok.encode(text + prefix, add_special_tokens=False)


def letter_ids(tok):
    """token ids of the bare option letters after MC_PREFIX, checked to be prefix-stable"""
    base = tok.encode("Q" + MC_PREFIX, add_special_tokens=False)
    out = []
    for L in LETTERS:
        ids = tok.encode("Q" + MC_PREFIX + L, add_special_tokens=False)
        assert ids[:-1] == base and tok.decode(ids[-1:]) == L, (L, ids, base)
        out.append(ids[-1])
    return out


def write(name, rows):
    os.makedirs(ITEMS, exist_ok=True)
    with open(os.path.join(ITEMS, name + ".jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(rows)} items, {sum(len(r.get('ids') or r.get('prompt_ids')) for r in rows)} prompt tokens",
          flush=True)


def load(name):
    with open(os.path.join(ITEMS, name + ".jsonl")) as f:
        return [json.loads(l) for l in f]


def mc_prompt(subject, question, options):
    body = "\n".join(f"({LETTERS[j]}) {o}" for j, o in enumerate(options))
    return (f"The following is a multiple-choice question about {subject}. Choose the correct option.\n\n"
            f"Question: {question}\n\nOptions:\n{body}\n\nAnswer with the letter of the correct option.")


def build_mc(tok):
    lids = letter_ids(tok)
    rows = [json.loads(l) for l in open(os.path.join(RAW, "mmlu_pro_test.jsonl"))]
    rng = random.Random(1234)
    out = []
    for cat in sorted({r["category"] for r in rows}):
        pool = [r for r in rows if r["category"] == cat]
        for r in sorted(rng.sample(pool, 100), key=lambda r: r["question_id"]):
            ids = chat(tok, mc_prompt(cat, r["question"], r["options"]), MC_PREFIX)
            assert tok.encode(tok.decode(ids) + "A", add_special_tokens=False)[-1] == lids[0]
            out.append(dict(key=f"mmlu_pro/{r['question_id']}", set="mmlu_pro", subject=cat, ids=ids,
                            pos=[len(ids) - 1], gold=int(r["answer_index"]), n_opts=len(r["options"])))
    write("mmlu_pro", out)
    out = []
    with open(os.path.join(RAW, "gpqa_diamond.csv")) as f:
        for j, r in enumerate(csv.DictReader(f)):
            opts = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
            perm = random.Random(j).sample(range(4), 4)
            opts = [opts[p].strip() for p in perm]
            ids = chat(tok, mc_prompt(r.get("High-level domain", "science").lower(), r["Question"].strip(), opts), MC_PREFIX)
            out.append(dict(key=f"gpqa/{r['Record ID']}", set="gpqa", subject=r.get("Subdomain", ""), ids=ids,
                            pos=[len(ids) - 1], gold=perm.index(0), n_opts=4))
    write("gpqa", out)
    return lids


def strip_gutenberg(text):
    a = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n", text)
    b = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK", text)
    return text[a.end() if a else 0:b.start() if b else len(text)].strip()


def code_doc():
    import mlx_lm
    base = os.path.dirname(mlx_lm.__file__)
    parts = []
    for p in sorted(glob.glob(os.path.join(base, "**", "*.py"), recursive=True)):
        src = open(p).read()
        if len(src) > 2000:
            parts.append(f"# ===== file: mlx_lm/{os.path.relpath(p, base)} =====\n{src}")
    return "\n\n".join(parts)


def build_tf(tok):
    rng = random.Random(77)
    gsm = [json.loads(l) for l in open(os.path.join(RAW, "gsm8k_test.jsonl"))]
    rows = [dict(key=f"tf_gsm/{600 + j}", set="tf_gsm", prompt_ids=chat(tok, GSM_INSTR + p["question"]), max_tokens=400)
            for j, p in enumerate(gsm[600:660])]
    ife = [json.loads(l) for l in open(os.path.join(RAW, "ifeval_input_data.jsonl"))]
    ife_pick = rng.sample(range(len(ife)), 130)
    rows += [dict(key=f"tf_ifeval/{ife[j]['key']}", set="tf_ifeval", prompt_ids=chat(tok, ife[j]["prompt"]), max_tokens=512)
             for j in ife_pick[:30]]
    from evalplus.data import get_human_eval_plus
    he = sorted(get_human_eval_plus().items())
    he_pick = rng.sample(range(len(he)), 84)
    rows += [dict(key=f"tf_code/{he[j][0]}", set="tf_code", max_tokens=400,
                  prompt_ids=chat(tok, "Complete the following Python function. Reply with the whole function in one "
                                       f"```python code block.\n\n```python\n{he[j][1]['prompt']}```"))
             for j in he_pick[:24]]
    rows += [dict(key=f"tf_multi/{lang}{j}", set="tf_multi", prompt_ids=chat(tok, text), max_tokens=384)
             for j, (lang, text) in enumerate(MULTI)]
    write("tf_prompts", rows)
    docs = []
    for name, text in (("en_novel", strip_gutenberg(open(os.path.join(RAW, "doc_en_pride_prejudice.txt")).read())),
                       ("fr_novel", strip_gutenberg(open(os.path.join(RAW, "doc_fr_monte_cristo.txt")).read())),
                       ("bg_wiki", open(os.path.join(RAW, "doc_bg_wikipedia.txt")).read()),
                       ("py_code", code_doc())):
        ids = tok.encode(text, add_special_tokens=False)
        assert len(ids) >= DOC_TOKENS, (name, len(ids))
        ids = ids[:DOC_TOKENS]
        docs.append(dict(key=f"tf_doc/{name}", set="tf_doc", ids=ids, pos=list(range(len(ids) - 1))))
    write("tf_docs", docs)
    return dict(ifeval=ife_pick[30:], humaneval=[he[j][0] for j in he_pick[24:]])


def build_decode(tok, picks):
    ife = [json.loads(l) for l in open(os.path.join(RAW, "ifeval_input_data.jsonl"))]
    write("dec_ifeval", [dict(key=f"ifeval/{ife[j]['key']}", set="ifeval", prompt=ife[j]["prompt"],
                              instruction_id_list=ife[j]["instruction_id_list"], kwargs=ife[j]["kwargs"],
                              user=ife[j]["prompt"], prompt_ids=chat(tok, ife[j]["prompt"]), max_tokens=1024)
                             for j in picks["ifeval"]])
    from evalplus.data import get_human_eval_plus
    he = get_human_eval_plus()
    write("dec_humaneval", [dict(key=f"humaneval/{t}", set="humaneval", task_id=t, entry_point=he[t]["entry_point"],
                                 prompt=he[t]["prompt"], max_tokens=768, user=_he_user(he[t]["prompt"]),
                                 prompt_ids=chat(tok, _he_user(he[t]["prompt"])))
                            for t in picks["humaneval"][:60]])
    rng = random.Random(5)
    m5 = [json.loads(l) for l in open(os.path.join(RAW, "math500_test.jsonl"))]
    m5 = rng.sample([p for p in m5 if int(p.get("level", 0)) >= 3], 25)
    gsm = [json.loads(l) for l in open(os.path.join(RAW, "gsm8k_test.jsonl"))][700:725]
    instr = "Solve the problem. Put the final answer in \\boxed{}.\n\n"
    write("dec_math", [dict(key=f"math500/{p['unique_id']}", set="math", gold=p["answer"], max_tokens=1536,
                            user=instr + p["problem"], prompt_ids=chat(tok, instr + p["problem"])) for p in m5] +
                      [dict(key=f"gsm8k/{700 + j}", set="math", gold=p["answer"].split("####")[-1].strip().replace(",", ""),
                            max_tokens=768, user=instr + p["question"], prompt_ids=chat(tok, instr + p["question"]))
                       for j, p in enumerate(gsm)])
    build_ruler(tok)


def _he_user(prompt):
    return f"Complete the following Python function. Reply with the whole function in one ```python code block.\n\n```python\n{prompt}```"


RULER_TOKENS = int(os.environ.get("EVALSUITE_RULER_TOKENS", "16384"))
NOISE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "


def _fill(tok, filler_ids, inserts, budget, rng):
    """filler token stream cut to `budget` tokens with the insert sentences placed at random sentence boundaries"""
    text = tok.decode(filler_ids[:budget])
    sents = re.split(r"(?<=[.!?])\s+", text)
    pos = sorted(rng.sample(range(1, max(2, len(sents))), len(inserts)))
    for k, (p, ins) in enumerate(zip(pos, inserts)):
        sents.insert(p + k, ins)
    return " ".join(sents)


def build_ruler(tok):
    """RULER-style synthetic long-context items (Hsieh et al. 2024 templates): 12 multi-key needle-in-a-haystack
    (4 needles, essay haystack) and 12 variable tracking (5-hop chain + 1 distractor chain, noise haystack), sized to
    RULER_TOKENS prompt tokens."""
    rng = random.Random(11)
    essay = strip_gutenberg(open(os.path.join(RAW, "doc_en_pride_prejudice.txt")).read())
    essay_ids = tok.encode(essay, add_special_tokens=False)
    noise_ids = tok.encode(NOISE * (RULER_TOKENS // 15 + 50), add_special_tokens=False)
    words = ["apple", "river", "candle", "violin", "garden", "falcon", "marble", "lantern", "harbor", "meadow", "copper",
             "thunder", "saffron", "glacier", "orchid", "compass", "velvet", "ember", "pepper", "summit"]
    rows = []
    for j in range(12):
        keys = rng.sample(words, 4)
        vals = [str(rng.randint(1000000, 9999999)) for _ in keys]
        q = rng.randrange(4)
        head = ("Some special magic numbers are hidden within the following text. Make sure to memorize it. I will "
                "quiz you about the numbers afterwards.\n")
        tail = f"\nWhat is the special magic number for {keys[q]} mentioned in the provided text?"
        budget = RULER_TOKENS - len(tok.encode(head + tail, add_special_tokens=False)) - 120
        start = rng.randrange(0, len(essay_ids) - budget)
        ctx = _fill(tok, essay_ids[start:], [f"One of the special magic numbers for {k} is: {v}." for k, v in zip(keys, vals)],
                    budget, rng)
        user = head + ctx + tail
        rows.append(dict(key=f"ruler/niah_mk{j}", set="ruler", task="niah", gold=[vals[q]], user=user,
                         prompt_ids=chat(tok, user), max_tokens=64))
    for j in range(12):
        val, dval = str(rng.randint(10000, 99999)), str(rng.randint(10000, 99999))
        names = ["".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(5)) for _ in range(10)]
        chain, dist = names[:5], names[5:]
        ins = [f"VAR {chain[0]} = {val}."] + [f"VAR {chain[k]} = VAR {chain[k - 1]}." for k in range(1, 5)]
        ins += [f"VAR {dist[0]} = {dval}."] + [f"VAR {dist[k]} = VAR {dist[k - 1]}." for k in range(1, 5)]
        head = "Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n"
        tail = (f"\nQuestion: Find all variables that are assigned the value {val} in the text above. Answer with the "
                f"variable names only.")
        budget = RULER_TOKENS - len(tok.encode(head + tail, add_special_tokens=False)) - 200
        order = list(range(10))                       # keep each chain in order, interleave the two chains
        rng.shuffle(order)
        merged, ia, ib = [], 0, 0
        for o in order:
            if o < 5:
                merged.append(ins[ia]); ia += 1
            else:
                merged.append(ins[5 + ib]); ib += 1
        ctx = _fill(tok, noise_ids, merged, budget, rng)
        user = head + ctx + tail
        rows.append(dict(key=f"ruler/vt{j}", set="ruler", task="vt", gold=chain, user=user,
                         prompt_ids=chat(tok, user), max_tokens=96))
    write("dec_ruler", rows)


def picks():
    """the decode-set picks build_tf makes (same seeded draws, in the same order), without rewriting the TF files"""
    rng = random.Random(77)
    ife = [json.loads(l) for l in open(os.path.join(RAW, "ifeval_input_data.jsonl"))]
    ife_pick = rng.sample(range(len(ife)), 130)
    from evalplus.data import get_human_eval_plus
    he = sorted(get_human_eval_plus().items())
    he_pick = rng.sample(range(len(he)), 84)
    return dict(ifeval=ife_pick[30:], humaneval=[he[j][0] for j in he_pick[24:]])


def decode_items():
    build_decode(tokenizer(), picks())


AGENT_TURNS = os.path.expanduser("~/SpecDecode/data/route4")
MGSM_LANGS = ("bn", "de", "es", "fr", "ja", "ru", "sw", "te", "th", "zh")


def build_mgsm(tok, per_lang=8):
    """MGSM (data/raw/mgsm, SOURCE.txt): per_lang problems per language, different problems in each language (one
    seeded order of the 250, language l takes positions l*per_lang ...), the question in its language after dec_math's
    English instruction (\\boxed{}), scored like dec_math"""
    order = random.Random(23).sample(range(250), 250)
    instr = "Solve the problem. Put the final answer in \\boxed{}.\n\n"
    rows = []
    for l, lang in enumerate(MGSM_LANGS):
        probs = [line.rstrip("\n").split("\t") for line in open(os.path.join(RAW, "mgsm", f"mgsm_{lang}.tsv"),
                                                                 encoding="utf-8")]
        assert len(probs) == 250 and all(len(p) == 2 for p in probs), (lang, len(probs))
        for i in order[l * per_lang:(l + 1) * per_lang]:
            q, a = probs[i]
            rows.append(dict(key=f"mgsm/{lang}{i}", set="mgsm", lang=lang, gold=a.strip(), max_tokens=1024,
                             prompt_ids=chat(tok, instr + q)))
    write("dec_mgsm", rows)


def decode_items_extra(tok=None):
    """the behaviour rows without gold, scored as agreement with the reference next to the floor (decode_report.AGREE):
    dec_multi = the 24 multilingual teacher-forced prompts (8 languages, 384 tokens, as cache/ref/tf_gen.jsonl);
    dec_agent = 29 real Claude Code turns (SpecDecode's route4: ~9K-token contexts, 125B token ids), the next 128 tokens;
    and the scored multilingual set dec_mgsm (build_mgsm). tok: the runtime's (MTPLX restores the pre-tokenizer regex
    that keeps combining marks on their letters: Thai, Telugu, Bengali), so the stage passes rt.tokenizer"""
    if not os.path.exists(os.path.join(ITEMS, "dec_mgsm.jsonl")):
        assert tok is not None, "build dec_mgsm with the runtime's tokenizer (rt.tokenizer)"
        build_mgsm(tok)
    if not os.path.exists(os.path.join(ITEMS, "dec_multi.jsonl")):
        rows = []
        for r in load("tf_prompts"):
            if r["set"] == "tf_multi":
                name = r["key"].split("/", 1)[1]
                rows.append(dict(key="multi/" + name, set="multi", lang=re.match(r"[a-z]+", name).group(0),
                                 prompt_ids=r["prompt_ids"], max_tokens=r["max_tokens"]))
        write("dec_multi", rows)
    if not os.path.exists(os.path.join(ITEMS, "dec_agent.jsonl")):
        rows = []
        for p in sorted(glob.glob(os.path.join(AGENT_TURNS, "*.npz"))):
            ids = [int(x) for x in np.load(p)["ids"]]
            rows.append(dict(key="agent/" + os.path.basename(p)[:-4], set="agent", prompt_ids=ids, max_tokens=128))
        write("dec_agent", rows)


def main():
    tok = tokenizer()
    lids = build_mc(tok)
    picks = build_tf(tok)
    build_decode(tok, picks)
    json.dump({"letter_ids": lids, "mc_prefix": MC_PREFIX, "doc_tokens": DOC_TOKENS},
              open(os.path.join(ITEMS, "meta.json"), "w"))


if __name__ == "__main__":
    main()
