"""TrainLab training text (2026-10-21): one shuffled token stream from local sources, disjoint from every held-out text.

Held out (never trained on): ~/BigMoE/FINDINGS.md, ~/MoonShots/scripts/*.py, GSM8K test items 200-399 (defer35's three
segments), ~/EvalSuite/data (the gate's corpora), and the account-1 transcripts (TrainLab's chat held-out segment).
Sources and token budgets: Claude Code transcripts of the main account (user and assistant text, tool calls, the head
of tool results), Python from site-packages, Swift from ~/OjoX, markdown notes of the research repos, GSM8K test items
outside 200-399, multilingual text if data/multi/*.txt exists.
Writes data/train_tokens.npy (uint32 stream, documents separated by <|endoftext|>) and data/train_sources.json.
"""
import glob, json, os, random, sys
import numpy as np

sys.path.insert(0, os.path.expanduser("~/NewIdeas/scripts"))
H = os.path.expanduser("~")
OUT = os.path.join(H, "TrainLab/data")
BUDGET = {"chat": 2_000_000, "python": 1_500_000, "swift": 300_000, "notes": 400_000, "math": 200_000, "multi": 600_000}
EXCLUDE = {os.path.join(H, "BigMoE/FINDINGS.md")}


def chat_docs():
    for f in sorted(glob.glob(os.path.join(H, ".claude/projects/*/*.jsonl"))):
        parts = []
        for line in open(f, errors="replace"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            m = d.get("message")
            if not isinstance(m, dict) or d.get("type") not in ("user", "assistant"):
                continue
            role, c = d["type"], m.get("content")
            if isinstance(c, str):
                parts.append(f"<{role}>\n{c}")
                continue
            for b in c if isinstance(c, list) else []:
                if not isinstance(b, dict):
                    continue
                t = b.get("type")
                if t == "text":
                    parts.append(f"<{role}>\n{b.get('text', '')}")
                elif t == "tool_use":
                    parts.append(f"<tool_call {b.get('name')}>\n{json.dumps(b.get('input'), ensure_ascii=False)[:3000]}")
                elif t == "tool_result":
                    r = b.get("content")
                    if isinstance(r, list):
                        r = "\n".join(x.get("text", "") for x in r if isinstance(x, dict))
                    if isinstance(r, str) and r:
                        parts.append(f"<tool_result>\n{r[:2500]}")
            if len(parts) >= 40:                       # one document per ~40 turns
                yield "\n\n".join(parts); parts = []
        if parts:
            yield "\n\n".join(parts)


def file_docs(pattern_list, min_size=1500, max_chars=60000):
    files = []
    for p in pattern_list:
        files += glob.glob(p, recursive=True)
    for f in sorted(set(files)):
        if f in EXCLUDE or "/MoonShots/scripts/" in f or "/.build/" in f or "/EvalSuite/data" in f:
            continue
        try:
            if os.path.getsize(f) < min_size:
                continue
            yield open(f, errors="replace").read()[:max_chars]
        except Exception:
            continue


def math_docs():
    gsm = [json.loads(l) for l in open(os.path.join(H, "MoEPrefetch/data/gsm8k/test.jsonl"))]
    items = gsm[:200] + gsm[400:]
    for i in range(0, len(items), 8):
        yield "\n\n".join("Question: " + p["question"] + "\nAnswer: " + p["answer"] for p in items[i:i + 8])


def main():
    from mlx_lm.utils import load_tokenizer
    import defer35 as D
    from pathlib import Path
    tok = load_tokenizer(Path(D.MODEL))
    eos = tok.convert_tokens_to_ids("<|endoftext|>")
    sp = os.path.join(H, "BigMoE/.venv/lib/python3.12/site-packages")
    gens = {
        "chat": chat_docs(),
        "python": file_docs([sp + "/**/*.py", "/opt/homebrew/lib/python3.14/site-packages/**/*.py"]),
        "swift": file_docs([H + "/OjoX/**/*.swift"]),
        "notes": file_docs([H + f"/{r}/**/*.md" for r in ("Breakthrough", "MoEPrefetch", "SpecDecode", "NewIdeas",
                                                              "MoonShots", "BigMoE", "PrefillKernels")], min_size=800),
        "math": math_docs(),
        "multi": file_docs([OUT + "/multi/*.txt"], min_size=200, max_chars=200000),
    }
    rng = random.Random(0)
    docs, stats = [], {}
    for name, g in gens.items():
        n, cand = 0, list(g)
        rng.shuffle(cand)
        for text in cand:
            ids = tok.encode(text)
            if name == "multi":                         # long multilingual files: cut into 2K-token documents
                chunks = [ids[i:i + 2048] for i in range(0, len(ids), 2048)]
            else:
                chunks = [ids[:8192]]
            for c in chunks:
                if n >= BUDGET[name]:
                    break
                docs.append(c); n += len(c)
            if n >= BUDGET[name]:
                break
        stats[name] = n
        print(name, n, flush=True)
    rng.shuffle(docs)
    stream = np.concatenate([np.array(d + [eos], np.uint32) for d in docs])
    np.save(os.path.join(OUT, "train_tokens.npy"), stream)
    json.dump({"tokens": int(stream.size), "docs": len(docs), "per_source": stats, "eos": int(eos)},
              open(os.path.join(OUT, "train_sources.json"), "w"), indent=1)
    print("total", stream.size)


if __name__ == "__main__":
    main()
