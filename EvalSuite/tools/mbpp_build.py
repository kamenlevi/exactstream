"""MBPP decode items (the orchestrator Fri 09:5x: decision 4 on MBPP for CD16; local data, no download).
  ~/BigMoE/.venv/bin/python tools/mbpp_build.py      # -> data/items/dec_mbpp.jsonl (the chat template needs BigMoE's venv)
The 100 tasks of MTP Head's exact walk (~/TrainLab/data/d02b_walk_g_gall16_mbpp.npz names: task ids 2-321), from
~/TrainLab/data/mbpp/mbpp.jsonl (Austin et al. 2021; Kamen's download OK 11-02; SHA256.txt beside it), with MTP Head's
template (tools: ~/TrainLab/sticky/dlog_capture.py MB_USER: the task text + its tests, which fix the function name) and
max_tokens 1024 (6 of the walk's 100 think inside a docstring until the cap, Fable GOAL)."""
import hashlib, json, os, re, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from evalsuite import data as D
import numpy as np

H = os.path.expanduser
MB_USER = ("Write the following Python function. Reply with the whole code in one ```python code block.\n\n"
           "{q}\nYour code should pass these tests:\n\n{t}")


def main():
    names = np.load(H("~/TrainLab/data/d02b_walk_g_gall16_mbpp.npz"), allow_pickle=True)["names"]
    ids = sorted(int(re.search(r"mbpp(\d+)", str(n)).group(1)) for n in names)
    assert len(ids) == 100 == len(set(ids)), len(ids)
    mb = {x["task_id"]: x for x in map(json.loads, open(H("~/TrainLab/data/mbpp/mbpp.jsonl")))}
    assert len(mb) == 974, len(mb)
    tok = D.tokenizer()
    rows = []
    for i in ids:
        x = mb[i]
        user = MB_USER.replace("{q}", x["text"].strip()).replace("{t}", "\n".join(x["test_list"]))
        rows.append(dict(key=f"mbpp/{i}", set="mbpp", task_id=i, text=x["text"], test_list=x["test_list"],
                         test_setup_code=x["test_setup_code"], ref_code=x["code"], user=user, max_tokens=1024,
                         prompt_ids=D.chat(tok, user)))
    D.write("dec_mbpp", rows)
    p = os.path.join(D.ITEMS, "dec_mbpp.jsonl")
    print(f"{os.path.relpath(p, ROOT)}: {len(rows)} items, sha256 {hashlib.sha256(open(p, 'rb').read()).hexdigest()[:12]}")


if __name__ == "__main__":
    main()
