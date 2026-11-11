"""D3 first experiment: the must-fix gap F (Breakthrough, 2026-10-20). How many tokens can a small in-RAM drafter write
before the 125B would have to correct it, under exact and lossy acceptance rules? (research/sparse_consult.md: chunked
verification only beats eager decoding on the SSD if F >= ~100-250, and it is the key number for one-stream speed.)
  gen125 C  : the 125B (BigMoE's streamed runtime, exact eager greedy, C = fraction of experts resident) answers 40 prompts
              (20 GSM8K, 8 code, 8 knowledge, 4 Bulgarian; <= 320 tokens), then is teacher-forced over each prompt +
              answer to log its top-20 log-probs at every answer position -> data/d3/gen125.npz
  draft NAME: a same-vocabulary drafter (q2b / q4b = Qwen3.5-2B / 4B MLX 4-bit) teacher-forced over the same tokens:
              its top-1, top-5, entropy and its log-prob of the 125B's token -> data/d3/draft_NAME.npz
  analyze   : (CPU) per drafter and acceptance rule, must-fix rate and F = answer tokens / (must-fix tokens + 1) per
              answer, pooled and by prompt type; AUROC of the drafter's entropy as a stop predictor. Teacher-forced: the
              drafter always continues from the 125B's own text (the usual approximation; real drafts diverge after a
              lossy accept, so end-to-end answer identity is the later check).
"""
import glob, json, os, sys, time
import numpy as np

OUT = os.path.expanduser(os.environ.get("BT_D3_OUT", "~/Breakthrough/data/d3"))
os.makedirs(OUT, exist_ok=True)
MD = os.path.expanduser("~/BigMoE/models/bare")
DRAFT = {"q2b": "models--mlx-community--Qwen3.5-2B-MLX-4bit", "q4b": "models--mlx-community--Qwen3.5-4B-MLX-4bit"}
CODE = ["Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring and two tests.",
        "Write a Python function that parses an ISO 8601 date string (YYYY-MM-DD) and returns the day of the week.",
        "Write a shell one-liner that counts the lines of every .py file under the current directory, largest first.",
        "Write a Python class LRUCache with get and put in O(1), using collections.OrderedDict.",
        "Explain and fix the bug: def avg(xs): return sum(xs) / len(xs) - it crashes on an empty list.",
        "Write a SQL query that returns the top 3 customers by total order amount from tables customers and orders.",
        "Write a Python function that merges two sorted lists into one sorted list without using sort().",
        "Write a JavaScript function debounce(fn, ms) and show how to use it on a window resize handler."]
KNOW = ["Why is the sky blue? Explain briefly.", "What causes the seasons on Earth?",
        "Explain how a hash table works and what a collision is.", "What is the difference between TCP and UDP?",
        "Summarize the causes of the First World War in a short paragraph.", "How does a vaccine train the immune system?",
        "What is the time complexity of binary search, and why?", "Explain what inflation is and one way central banks fight it."]
BG = ["Разкажи ми накратко историята на Рилския манастир.", "Обясни какво е фотосинтеза с прости думи.",
      "Какви са предимствата и недостатъците на електрическите коли?", "Напиши кратко писмо до приятел, в което го каниш на рожден ден."]


def prompts():
    gsm = [json.loads(l) for l in open(os.path.expanduser("~/MoEPrefetch/data/gsm8k/test.jsonl"))][300:320]
    ps = [("gsm", "Solve the problem step by step, briefly. End with a line of the form 'Answer: <number>'.\n\n" + p["question"])
          for p in gsm]
    return ps + [("code", t) for t in CODE] + [("know", t) for t in KNOW] + [("bg", t) for t in BG]


def gen125(frac):
    import mlx.core as mx
    sys.path.insert(0, os.path.expanduser("~/BigMoE"))
    os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
    from bigmoe.loader import load_streamed
    from bigmoe import decode as D
    from bigmoe.fuse import reset_generation_state
    cfg = json.load(open(os.path.join(MD, "config.json")))
    E = cfg.get("text_config", cfg)["num_experts"]
    rt, st = load_streamed(MD, max(16, round(frac * E)), mtp=False)
    tok, model = rt.tokenizer, rt.model
    rec = {"kind": [], "plen": [], "seq": [], "top": [], "tlp": [], "match": []}
    t0 = time.time()
    for kind, text in prompts():
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
        ids = list(ids if isinstance(ids[0], int) else ids["input_ids"])
        out = D.greedy(model, ids, 320)
        eos = getattr(tok, "eos_token_id", None)
        if eos is not None and eos in out:
            out = out[:out.index(eos) + 1]
        seq = ids + out
        reset_generation_state(model)
        cache = model.make_cache()
        lg = model(mx.array([seq[:-1]]), cache=cache)
        lg = (lg[0] if isinstance(lg, tuple) else lg)[0, len(ids) - 1:].astype(mx.float32)
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        top = mx.argsort(-lp, axis=-1)[:, :20]
        tlp = mx.take_along_axis(lp, top, axis=-1)
        mx.eval(top, tlp)
        top, tlp = np.array(top), np.array(tlp)
        rec["kind"].append(kind); rec["plen"].append(len(ids)); rec["seq"].append(np.array(seq, np.int32))
        rec["top"].append(top.astype(np.int32)); rec["tlp"].append(tlp.astype(np.float16))
        rec["match"].append(float((top[:, 0] == np.array(out)).mean()))
        reset_generation_state(model)
        print(f"{kind:4s} answer {len(out):3d} tok, teacher-forced top-1 = decoded token {rec['match'][-1]:.3f}, "
              f"{time.time() - t0:.0f}s", flush=True)
    np.savez(os.path.join(OUT, "gen125.npz"), kind=np.array(rec["kind"]), plen=np.array(rec["plen"]),
             **{f"seq{i}": s for i, s in enumerate(rec["seq"])}, **{f"top{i}": s for i, s in enumerate(rec["top"])},
             **{f"tlp{i}": s for i, s in enumerate(rec["tlp"])}, match=np.array(rec["match"]))
    st.gate.check()
    print(f"gen125: {len(rec['seq'])} answers, mean decode/teacher-forced agreement {np.mean(rec['match']):.4f}, "
          f"{time.time() - t0:.0f}s, peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)


def draft(name):
    import mlx.core as mx
    from mlx_lm import load
    g = np.load(os.path.join(OUT, "gen125.npz"))
    model, _ = load(glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/{DRAFT[name]}/snapshots/*/"))[0])
    rec = {}
    for i in range(len(g["kind"])):
        seq, plen = g[f"seq{i}"], int(g["plen"][i])
        lg = model(mx.array(seq[None, :-1]))
        lg = (lg[0] if isinstance(lg, tuple) else lg)[0, plen - 1:].astype(mx.float32)
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        top5 = mx.argsort(-lp, axis=-1)[:, :5]
        ent = -(mx.exp(lp) * lp).sum(-1)
        own = mx.take_along_axis(lp, mx.array(seq[plen:, None].astype(np.int32)), axis=-1)[:, 0]
        mx.eval(top5, ent, own)
        rec[f"top5_{i}"], rec[f"ent{i}"], rec[f"own{i}"] = np.array(top5), np.array(ent), np.array(own)
    np.savez(os.path.join(OUT, f"draft_{name}.npz"), **rec)
    print(f"draft {name}: {len(g['kind'])} answers, peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)


def auroc(score, label):
    order = np.argsort(score); r = np.empty(len(score)); r[order] = np.arange(1, len(score) + 1)
    n1 = label.sum(); n0 = len(label) - n1
    return (r[label].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)


def analyze():
    g = np.load(os.path.join(OUT, "gen125.npz"))
    kinds = g["kind"]
    print(f"125B decode vs teacher-forced top-1 agreement {g['match'].mean():.4f} over {len(kinds)} answers")
    rules = {"exact top-1": lambda d, top, tlp: d != top[:, 0],
             "in 125B top-3": lambda d, top, tlp: ~(top[:, :3] == d[:, None]).any(1),
             "margin <= 1 nat": lambda d, top, tlp: margin(d, top, tlp) > 1.0,
             "margin <= 2 nats": lambda d, top, tlp: margin(d, top, tlp) > 2.0,
             "margin <= 3 nats": lambda d, top, tlp: margin(d, top, tlp) > 3.0}
    for name in DRAFT:
        f = os.path.join(OUT, f"draft_{name}.npz")
        if not os.path.exists(f):
            continue
        dr = np.load(f)
        print(f"\ndrafter {name}: must-fix share / F (answer tokens per must-fix) pooled and gsm/code/know/bg; "
              f"AUROC of drafter entropy for must-fix")
        for rname, rule in rules.items():
            mf, n, per = [], [], {}
            labels, ents = [], []
            for i in range(len(kinds)):
                top, tlp = g[f"top{i}"].astype(np.int64), g[f"tlp{i}"].astype(np.float32)
                d = dr[f"top5_{i}"][:, 0].astype(np.int64)
                m = rule(d, top, tlp)
                mf.append(m.sum()); n.append(len(m))
                per.setdefault(str(kinds[i]), [0, 0]); per[str(kinds[i])][0] += m.sum(); per[str(kinds[i])][1] += len(m)
                labels.append(m); ents.append(dr[f"ent{i}"])
            F = sum(n) / (sum(mf) + len(n))
            lab, en = np.concatenate(labels), np.concatenate(ents)
            pk = "  ".join(f"{k} {v[1] / (v[0] + 1):6.1f}" for k, v in per.items())
            print(f"  {rname:18s} must-fix {sum(mf) / sum(n):6.3f}  F {F:6.1f}   [{pk}]  AUROC {auroc(en, lab):.3f}")


def margin(d, top, tlp):
    hit = top == d[:, None]
    m = np.full(len(d), 99.0)
    rows = hit.any(1)
    m[rows] = tlp[rows, 0] - tlp[rows][hit[rows]]
    return m


if __name__ == "__main__":
    {"gen125": lambda: gen125(float(sys.argv[2])), "draft": lambda: draft(sys.argv[2]), "analyze": analyze}[sys.argv[1]]()
