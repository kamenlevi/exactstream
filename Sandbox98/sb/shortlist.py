"""Token counts for the CHAIN SIZE screen's lm_head shortlists: the 2B's tokenizer over the Python part of Magicoder
OSS-Instruct (decontaminated; problem + solution, 38,284 samples) plus my training captures' answer tokens (the 2B's own
greedy answers: base + pool + mc2). No MBPP / HumanEval text. -> data/shortlist_counts.npy (int64 [V], 2 MB)
usage: shortlist.py"""
import glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

H = os.path.expanduser
V = 248320


def main():
    from transformers import AutoTokenizer
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = sorted(glob.glob(H("~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-2B-MLX-4bit/snapshots/*")))[0]
    tok = AutoTokenizer.from_pretrained(model)
    t0 = time.time()
    texts = []
    for l in open(H("~/datasets/deepdraft/magicoder/data-oss_instruct-decontaminated.jsonl")):
        d = json.loads(l)
        if d.get("lang") == "python":
            texts += [d["problem"], d["solution"]]
    cnt = np.zeros(V, np.int64)
    ntok = 0
    for i in range(0, len(texts), 2000):
        for ids in tok(texts[i:i + 2000], add_special_tokens=False)["input_ids"]:
            a = np.asarray(ids, np.int64); a = a[a < V]
            np.add.at(cnt, a, 1); ntok += len(a)
    s = json.load(open(os.path.join(root, "data", "split.json")))
    fs = [os.path.join(root, "data", "mc", k + ".npz") for k in s["base"] + s["pool"]] + \
         [os.path.join(root, "data", "mc2", k + ".npz") for k in s["mc2"]]
    ncap = 0
    for f in fs:
        z = np.load(f); ids = np.asarray(z["ids"]); gen = int(z["gen"])
        np.add.at(cnt, ids[gen:], 1); ncap += len(ids) - gen
    np.save(os.path.join(root, "data", "shortlist_counts.npy"), cnt)
    print(f"[sl] {len(texts)} Magicoder python texts = {ntok} tokens + {ncap} capture answer tokens; {int((cnt > 0).sum())} "
          f"distinct tokens; {time.time() - t0:.0f}s -> data/shortlist_counts.npy", flush=True)


if __name__ == "__main__":
    main()
