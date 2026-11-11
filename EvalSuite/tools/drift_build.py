"""Items for RELAX's long-code drift re-check (docs/PREREG_relax_drift_2026-11-05.md). Two steps, because evalplus is
only in EvalSuite's venv and the chat template (jinja2) only in BigMoE's:
  .venv/bin/python tools/drift_build.py dump                 # HumanEval+'s problems outside the core 60 -> results/drift/he_rest.json
  ~/BigMoE/.venv/bin/python tools/drift_build.py build       # -> data/items/dec_humanevalx.jsonl, dec_agentlong.jsonl
  ~/BigMoE/.venv/bin/python tools/drift_build.py buildt      # -> data/items/dec_humanevalt.jsonl (amendment 2's top-up)
dec_humanevalx: the 104 HumanEval+ problems not in dec_humaneval, with its template (data._he_user) and max_tokens 768;
build first checks that the template still gives the core 60's saved prompt ids. dec_agentlong: dec_agent's 17 write /
edit turns (real code into files) with max_tokens 1024 instead of 128. dec_humanevalt (PREREG_relax_drift amendment 2): 41
HumanEval+ problems (task ids 0, 4, ... 160) asking for the function plus a pytest module of >= 15 tests, max_tokens 1536."""
import hashlib, json, os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from evalsuite import data as D

OUT = os.path.join(ROOT, "results", "drift")
REST = os.path.join(OUT, "he_rest.json")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:12]


def dump():
    from evalplus.data import get_human_eval_plus
    he = get_human_eval_plus()
    core = {it["task_id"] for it in D.load("dec_humaneval")}
    rest = [dict(task_id=t, prompt=he[t]["prompt"], entry_point=he[t]["entry_point"]) for t in sorted(he) if t not in core]
    assert len(he) == 164 and len(core) == 60 and len(rest) == 104, (len(he), len(core), len(rest))
    os.makedirs(OUT, exist_ok=True)
    json.dump(rest, open(REST, "w"))
    print(f"{os.path.relpath(REST, ROOT)}: {len(rest)} problems, sha256 {sha(REST)}")


def build():
    tok = D.tokenizer()
    core = D.load("dec_humaneval")
    bad = [it["key"] for it in core if D.chat(tok, D._he_user(it["prompt"])) != it["prompt_ids"]]
    assert not bad, f"the chat template no longer gives dec_humaneval's prompt ids: {bad[:3]}"
    rest = json.load(open(REST))
    rows = [dict(key=f"humanevalx/{r['task_id']}", set="humanevalx", task_id=r["task_id"], entry_point=r["entry_point"],
                 prompt=r["prompt"], max_tokens=768, user=D._he_user(r["prompt"]),
                 prompt_ids=D.chat(tok, D._he_user(r["prompt"]))) for r in rest]
    assert len(rows) == 104
    D.write("dec_humanevalx", rows)
    ag = [dict(it, key="agentlong/" + it["key"].split("/", 1)[1], set="agentlong", max_tokens=1024)
          for it in D.load("dec_agent") if it["key"].startswith(("agent/write", "agent/edit"))]
    assert len(ag) == 17, len(ag)
    D.write("dec_agentlong", ag)
    for n in ("dec_humanevalx", "dec_agentlong"):
        p = os.path.join(D.ITEMS, n + ".jsonl")
        print(f"{os.path.relpath(p, ROOT)}: sha256 {sha(p)}")
    keys = os.path.join(OUT, "keys_B.txt")             # the re-check's keys for arm B: all 121 rows
    open(keys, "w").write("\n".join([r["key"] for r in rows] + [r["key"] for r in ag]) + "\n")
    print(f"{os.path.relpath(keys, ROOT)}: {len(rows) + len(ag)} keys, sha256 {sha(keys)}")


T_USER = ("Complete the following Python function. Then write a pytest test module for it with at least 15 test "
          "functions that cover normal cases and edge cases. Reply with the function and the tests in one ```python code "
          "block.\n\n```python\n{prompt}```")


def buildt():
    tok = D.tokenizer()
    probs = {it["task_id"]: it for it in D.load("dec_humaneval")}
    probs.update({r["task_id"]: r for r in json.load(open(REST))})
    assert len(probs) == 164, len(probs)
    pick = [f"HumanEval/{i}" for i in range(0, 164, 4)]
    rows = [dict(key=f"humanevalt/{t}", set="humanevalt", task_id=t, entry_point=probs[t]["entry_point"],
                 prompt=probs[t]["prompt"], max_tokens=1536, user=T_USER.format(prompt=probs[t]["prompt"]),
                 prompt_ids=D.chat(tok, T_USER.format(prompt=probs[t]["prompt"]))) for t in pick]
    assert len(rows) == 41
    D.write("dec_humanevalt", rows)
    p = os.path.join(D.ITEMS, "dec_humanevalt.jsonl")
    print(f"{os.path.relpath(p, ROOT)}: sha256 {sha(p)}")
    keys = os.path.join(OUT, "keys_T.txt")
    open(keys, "w").write("\n".join(r["key"] for r in rows) + "\n")
    print(f"{os.path.relpath(keys, ROOT)}: {len(rows)} keys, sha256 {sha(keys)}")


if __name__ == "__main__":
    {"dump": dump, "build": build, "buildt": buildt}[sys.argv[1]]()
