"""WEEK98 J(CC) amendment 1 (MoonShots, Sun 15 Nov 2026; LEVERS_A_MOONSHOTS.md): the NUMERICS CONTROL arm. Exact greedy
of the same Qwen3.5-2B through mlx_lm's BatchGenerator (Sandbox98's sb/gen.greedy_batch: batch 8 per set, left-padded
prefill), with the same prompts, chat template and caps as jcc_decode.py. Its discordance with the batch-1 exact arm
(jcc/arm0.jsonl) is the 2B's paired noise floor under a numerics-only change.
Writes ~/MoonShots/week98/jcc/armB.jsonl (the fields jcc_score.py reads; no checks, no R1 commits).
usage: ~/BigMoE/.venv/bin/python -I -B jcc_batch.py [--batch 8] [--mem 4]"""
import argparse, json, os, sys, time

H = os.path.expanduser
sys.path.insert(0, H("~/Sandbox98")); sys.path.insert(0, H("~/MoonShots/week98"))
from span_tracker import SpanTracker                      # noqa: E402
import jcc_decode as JD                                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--mem", type=float, default=4.0)
    a = ap.parse_args()
    from sb import core as C
    from sb.gen import greedy_batch, chat_ids
    import mlx.core as mx
    C.limit_memory(a.mem)
    model, tok = C.load_target()
    eos = set(tok.eos_token_ids)

    class Lazy(dict):
        def __missing__(self, k):
            v = tok.decode([int(k)]); self[k] = v; return v
    table = Lazy()
    its = JD.panel_items()
    path = os.path.join(JD.OUT, "armB.jsonl")
    done = {json.loads(l)["key"] for l in open(path)} if os.path.exists(path) else set()
    for grp in ("humaneval", "mbpp"):
        part = [it for it in its if ("mbpp" if it["set"] == "mbpp" else "humaneval") == grp and it["key"] not in done]
        if not part:
            continue
        cap = int(part[0]["max_tokens"])
        assert all(int(it["max_tokens"]) == cap for it in part)
        t0 = time.perf_counter()
        prompts = [chat_ids(tok, it["user"]) for it in part]

        def on_done(i, ans, part=part, prompts=prompts):
            it = part[i]
            tr = SpanTracker(table); cls_n = {c: 0 for c in JD.CLASSES}
            for x in ans:
                cls_n[JD.where(tr, table[x])] += 1; tr.update(x)
            row = dict(key=it["key"], set=it["set"], delta="B", n=len(ans), stop="eos" if ans and ans[-1] in eos else "cap",
                       ids=[int(x) for x in ans], text=tok.decode([x for x in ans if x not in eos]), checks=[],
                       cut_check=None, relaxed=[], cls=cls_n, secs=None, prompt_len=len(prompts[i]))
            with open(path, "a") as f:
                f.write(json.dumps(row) + "\n")
        greedy_batch(prompts, cap, batch=a.batch, on_done=on_done)
        mx.clear_cache()
        print(f"[jcc B] {grp}: {len(part)} answers in {time.perf_counter() - t0:.0f} s; peak "
              f"{mx.get_peak_memory() / 2 ** 30:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
