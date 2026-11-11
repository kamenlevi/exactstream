"""Breakthrough: B12 / B8 through MTPLX's generate_ar, the way the Eval suite's decode stage runs a variant
(~/EvalSuite/evalsuite/decode.py _gen: greedy, the whole prompt prefilled at MTPLX_PREFILL_CHUNK_SIZE, the first token
from the prefill's logits, then one token per call). Env as evalsuite/env.py sets it (MTPLX defaults, BIGMOE_SPARE 0,
PF_NOPUBLISH 1); BT_AR_PF=1 adds the PF1 staging flags env.py forces for non-frugal variants (the tier layers bypass
them; this checks that). None of the CLI's speed flags (COMPILE_HC, HC_VREAD, ASYNC_EVERY).
Usage: bt_b12_ar.py FRAC [N]   (B12_* switches as for bt_b12.py, e.g. B12_LATE3=2,4 for B8)"""
import hashlib, os, sys, time

for _k, _v in {"MTPLX_NGRAM_PREWARM": "off", "MTPLX_COMPILED_VERIFY": "0", "MTPLX_QWEN4EXP_COMPILE": "0",
               "MTPLX_COMPILE_AR_FORWARD": "0", "MTPLX_COMPILED_TARGET_PREFIX": "0"}.items():
    os.environ.setdefault(_k, _v)
os.environ.update({"BIGMOE_SPARE": "0", "BIGMOE_PF_NOPUBLISH": "1"})
if os.environ.get("BT_AR_PF") == "1":
    os.environ.update({"BIGMOE_PF_PREFILL": "1", "BIGMOE_PF_FULLSCR": "1"})
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_b12 import load_b12, CFG

frac, N = float(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 96
rt, st = load_b12(frac)
import ctypes
import mlx.core as mx
from mtplx import generation as G
from mtplx.sampling import SamplerConfig


class _RU(ctypes.Structure):                                          # rusage_info_v4 (as bt_b12's CLI)
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"u{i}", ctypes.c_uint64) for i in range(40)]


def footprint_peak():
    ru = _RU()
    return ru.u28 / 1e9 if ctypes.CDLL(None).proc_pid_rusage(os.getpid(), 4, ctypes.byref(ru)) == 0 else float("nan")


tok = rt.tokenizer
EOS = {248044, 248046}                                                # the suite's EOS set
greedy = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)
PROMPTS = {"code": open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read(),
           "prose": ("Explain, in plain prose for a curious 14-year-old, how a CPU cache works and why it makes "
                     "programs faster. Use one everyday analogy and keep it to about 300 words."),
           "ifeval": ("Write a short note to a neighbour asking them to water your plants while you are away. "
                      "Your entire response must be in lowercase letters, contain exactly 3 paragraphs separated by "
                      "the markdown divider ***, and end with the phrase 'thanks again'.")}
tag = (f"b12 C_m={CFG['cm']} C_l={CFG['cl']} late3={CFG['l3']} w={CFG['w']} chunk "
       f"{os.environ.get('MTPLX_PREFILL_CHUNK_SIZE', '2048')} pf1={os.environ.get('BT_AR_PF', '0')}")
for name, text in PROMPTS.items():
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
    ids = list(ids if isinstance(ids[0], int) else ids["input_ids"])
    first = {}
    t0 = time.perf_counter()
    res = G.generate_ar(rt, ids, max_tokens=N, sampler=greedy, seed=0,
                        token_callback=lambda _t: first.setdefault("t", time.perf_counter()))
    t1 = time.perf_counter()
    toks = [int(x) for x in (getattr(res, "tokens", None) or [])]
    cut = next((j + 1 for j, x in enumerate(toks) if x in EOS), None)
    toks, finish = (toks[:cut], "eos") if cut else (toks, "length")
    tf = first.get("t", t1)
    print(f"[{name}] generate_ar {tag}: {len(toks)} tokens {finish}; prompt {len(ids)} tokens, first token after "
          f"{tf - t0:.2f} s, decode {(len(toks) - 1) / max(t1 - tf, 1e-9):.1f} tok/s; PEAK FOOTPRINT "
          f"{footprint_peak():.1f} GB; sha {hashlib.sha1(str(toks).encode()).hexdigest()[:10]} | "
          f"{tok.decode([x for x in toks if x not in EOS])[:150]!r}", flush=True)
