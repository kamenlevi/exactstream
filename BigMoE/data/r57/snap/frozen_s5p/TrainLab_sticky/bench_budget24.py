"""TrainLab (10-22, NewIdeas 13:22): BigMoE's v3 24 GB main-track MTP decode with the per-pass expert budget (an opt-in,
NON-EXACT mode; Kamen's call) vs natural routing. SpecDecode's test_spec125.py (BigMoE's driver for the main track)
runs unchanged in this process, importing ~/BigMoE read-only; nothing in BigMoE is edited. Two wrappers:
- spec125.generate: from its first call (the 16-token warm-up included) the router hook of budget_hook.py on the target
  layers (TL_BUDGET = b18 / b16 / o5; unset = natural, no hook). For timed calls (N > 16) the loader counters around
  the call (ext.reset_stats / ext.stats, MoEPrefetch's spec_pf pattern), printed per verify pass: misses, stalled
  layers, loader-busy io ms, JIT prefetch issued / used / late; and the hook's distinct experts per layer-pass
  (natural vs used) and the process's lifetime peak footprint.
- D.greedy (ar2, one row per call) is never budgeted: it is the reference the spec run's sha is compared with.
PROMPT: test_spec125's own (code / edit / long / prose / loop / long32) or two patched in at run time: math (GSM8K test
0) and agent (a real Claude Code turn's pre-tokenized prompt, TL_AGENT = Chat1000's data/agent125/<turn>.npz).
usage: [BigMoE v3 24 GB env] TL_BUDGET=b18 PROMPT=code bench_budget24.py 0.25 256 3:3:0"""
import ctypes, os, sys, time

for _k, _v in (("MTPLX_NGRAM_PREWARM", "off"), ("MTPLX_FAMILY_CAPTURE_COMMIT", "1"), ("MTPLX_DROP_EVENTS", "1"),
               ("MTPLX_LAZY_MTP_HISTORY_APPEND", "1"), ("MTPLX_DEFER_REPAIR_EVAL", "1"), ("MTPLX_COMPILED_VERIFY", "0"),
               ("MTPLX_QWEN4EXP_COMPILE", "0"), ("MTPLX_COMPILE_AR_FORWARD", "0"),
               ("MTPLX_COMPILED_TARGET_PREFIX", "0")):                 # = test_spec125's defaults, before any import
    os.environ.setdefault(_k, _v)
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
sys.path.insert(2, os.path.dirname(os.path.abspath(__file__)))
import bigmoe.gated as G  # noqa: E402
import spec125  # noqa: E402
import budget_hook as H  # noqa: E402

BUD = os.environ.get("TL_BUDGET", "")
KEYS = ("misses", "stalled", "io_s", "pf_issued", "pf_used", "pf_late")
STATE = {"installed": False}
_gen = spec125.generate


class _RU(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"u{i}", ctypes.c_uint64) for i in range(40)]


def footprint():
    ru = _RU()
    return ru.u28 / 1e9 if ctypes.CDLL(None).proc_pid_rusage(os.getpid(), 4, ctypes.byref(ru)) == 0 else float("nan")


def generate(model, ids, n, *a, **kw):
    if not STATE["installed"]:
        k = H.install(model, BUD, stats=bool(BUD)) if BUD else 0
        j = H.install_jit(model, BUD) if BUD and os.environ.get("TL_BUDGET_JIT") == "1" else 0
        print(f"[budget] {BUD or 'natural'}: router hook on {k} target layers"
              + (f", budget-aware JIT on {j} look-ahead routers" if j else ""), flush=True)
        STATE["installed"] = True
    if n <= 16:
        return _gen(model, ids, n, *a, **kw)
    H.STATS.update(calls=0, nat=[], used=[])                         # this call's passes only
    G.ext.reset_stats()
    t = time.perf_counter()
    out, s = _gen(model, ids, n, *a, **kw)
    dt = time.perf_counter() - t
    e = G.ext.stats()
    p = max(1, s.get("passes", 1))
    per = " ".join(f"{k} {e[k] / p * (1e3 if k == 'io_s' else 1):.2f}" + (" ms" if k == "io_s" else "")
                   for k in KEYS if k in e)
    u = H.stats_summary()
    print(f"[budget] {BUD or 'natural'}{'+jit' if os.environ.get('TL_BUDGET_JIT') == '1' and BUD else ''} N={n}: {s.get('tok_s', 0):.2f} tok/s, {s.get('tok_per_pass', 0):.2f} tok/pass, "
          f"{p} passes, wall {dt:.1f}s | per verify pass: {per}"
          + (f" | distinct experts per layer-pass {u['used']:.1f} of natural {u['natural']:.1f}" if u else "")
          + f" | footprint {footprint():.1f} GB", flush=True)
    return out, s


spec125.generate = generate
spec = os.path.expanduser("~/SpecDecode/scripts/test_spec125.py")
sys.argv = [spec] + sys.argv[1:]
src = open(spec).read()                                   # read at run time, never edited: two more prompts patched in
MATH = ("Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends "
        "every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much "
        "in dollars does she make every day at the farmers' market? Solve it step by step.")   # GSM8K test 0
AGENT = os.path.expanduser(os.environ.get("TL_AGENT", "~/Chat1000/data/agent125/bash17.npz"))  # a real Claude Code turn
src = src.replace('PROMPTS = {', 'PROMPTS = {\n    "math": ' + repr(MATH) + ',\n    "agent": "",', 1)
src = src.replace('ids = rt.tokenizer.apply_chat_template(',
                  'ids = ([int(t) for t in __import__("numpy").load(' + repr(AGENT) + ')["prompt"]] if PROMPT == "agent" else None) or '
                  'rt.tokenizer.apply_chat_template(', 1)
assert '"math":' in src and '"agent"' in src and 'PROMPT == "agent"' in src, "test_spec125.py changed: re-check the patch"
exec(compile(src, spec, "exec"), {"__name__": "__main__", "__file__": spec})
