"""Near-tie check (Kamen's acceptance condition for the dup-row + compiled-HC reference): regenerate GSM8K problems
12/87/119 (MoEPrefetch's prompts, greedy, 600 tokens) and record, at every step, the top-2 tokens and logits (fp32).
Run once per arm (separate processes: dup-row patches classes globally), then `near_tie.py compare`:
first divergence per problem, and each arm's logit/probability gap between the two candidate tokens there.
Usage: near_tie.py eager|ref [frac=0.50]   (ref: BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1)  ;  near_tie.py compare"""
import json, os, sys
OUT = os.path.expanduser(os.environ.get("NT_OUT", "~/BigMoE/data/neartie_{}.json"))
PROBS = tuple(int(x) for x in os.environ.get("NT_PROBS", "12,87,119").split(","))   # NT_PROBS=..., NT_OUT=.../nt_{}.json
if sys.argv[1] == "compare":
    a, b = (json.load(open(OUT.format(t))) for t in ("eager", os.environ.get("NT_ARM", "ref")))
    for p in map(str, PROBS):
        ta, tb = a[p]["tokens"], b[p]["tokens"]
        d = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
        if d is None:
            print(f"problem {p}: no divergence in {min(len(ta), len(tb))} tokens"); continue
        ea, eb = a[p]["top2"][d], b[p]["top2"][d]          # [tok1, logit1, tok2, logit2, logsumexp]
        def gap(r, x, y):
            lx = r[1] if r[0] == x else (r[3] if r[2] == x else None)
            ly = r[1] if r[0] == y else (r[3] if r[2] == y else None)
            if lx is None or ly is None: return "the other arm's token is not in this arm's top-2"
            import math
            return f"logit gap {lx - ly:+.4f}, p {math.exp(lx - r[4]):.4f} vs {math.exp(ly - r[4]):.4f}"
        print(f"problem {p}: first divergence at generated token {d}: eager picks {ta[d]}, ref picks {tb[d]} | "
              f"eager arm: {gap(ea, ta[d], tb[d])} | ref arm: {gap(eb, tb[d], ta[d])}", flush=True)
    sys.exit(0)
arm = sys.argv[1]
frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.50
for k, v in (("MTPLX_NGRAM_PREWARM", "off"), ("MTPLX_COMPILED_VERIFY", "0"), ("MTPLX_QWEN4EXP_COMPILE", "0"),
             ("MTPLX_COMPILE_AR_FORWARD", "0"), ("MTPLX_COMPILED_TARGET_PREFIX", "0")):
    os.environ.setdefault(k, v)
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import mlx.core as mx
from bigmoe.loader import load_streamed
rt, st = load_streamed(os.path.expanduser("~/BigMoE/models/bare"), round(frac * 512))
model, tok = rt.model, rt.tokenizer
probs = [json.loads(l) for l in open(os.path.expanduser("~/MoEPrefetch/data/gsm8k/test.jsonl"))]
INSTR = "Solve the problem step by step, briefly. End with a line of the form 'Answer: <number>'.\n\n"
L0 = lambda o: o[0] if isinstance(o, tuple) else o
res = {}
if os.environ.get("NT_PROBE"):            # NT_PROBE=prob:pos:tokA,tokB,... -> replay this arm's own tokens, read logits there
    pp, pos, want = os.environ["NT_PROBE"].split(":"); pos = int(pos); want = [int(t) for t in want.split(",")]
    own = json.load(open(OUT.format(arm)))[pp]["tokens"]
    ids = tok.apply_chat_template([{"role": "user", "content": INSTR + probs[int(pp)]["question"]}], add_generation_prompt=True,
                                  enable_thinking=False)
    ids = ids if isinstance(ids[0], int) else ids["input_ids"]
    cache = model.make_cache(); mx.eval(L0(model(mx.array([ids[:-1]]), cache=cache)))
    y = int(ids[-1])
    for t in range(pos + 1):
        lg = L0(model(mx.array([[y]]), cache=cache))[0, -1].astype(mx.float32)
        if t == pos:
            lse = float(mx.logsumexp(lg).item())
            print(f"{arm} probe problem {pp} token {pos}: " + ", ".join(
                f"{w}: logit {float(lg[w].item()):.4f} p {float(mx.exp(lg[w] - lse).item()):.4f}" for w in want), flush=True)
            break
        assert int(mx.argmax(lg).item()) == own[t], "replay diverged from this arm's own tokens"
        y = own[t]
    sys.exit(0)
for p in PROBS:
    ids = tok.apply_chat_template([{"role": "user", "content": INSTR + probs[p]["question"]}], add_generation_prompt=True,
                                  enable_thinking=False)
    ids = ids if isinstance(ids[0], int) else ids["input_ids"]
    cache = model.make_cache()                 # same as bigmoe.decode.greedy / MTPLX generate_ar: prefill all but last
    mx.eval(L0(model(mx.array([ids[:-1]]), cache=cache)))
    y, toks, top2 = int(ids[-1]), [], []
    for _ in range(600):
        lg = L0(model(mx.array([[y]]), cache=cache))[0, -1].astype(mx.float32)
        i2 = mx.argpartition(-lg, kth=1)[:2]
        v2 = lg[i2]; lse = mx.logsumexp(lg)
        y = int(mx.argmax(lg).item())
        i2, v2 = i2.tolist(), v2.tolist()
        if v2[1] > v2[0]: i2, v2 = i2[::-1], v2[::-1]
        toks.append(y); top2.append([i2[0], v2[0], i2[1], v2[1], float(lse.item())])
        if y in (tok.eos_token_id,) or (getattr(tok, "eos_token_ids", None) and y in tok.eos_token_ids):
            break
    res[str(p)] = {"tokens": toks, "top2": top2, "text_tail": tok.decode(toks)[-160:]}
    print(f"{arm} problem {p}: {len(toks)} tokens; tail {res[str(p)]['text_tail']!r}", flush=True)
json.dump(res, open(OUT.format(arm), "w"))
st.gate.check()
