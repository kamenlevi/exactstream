"""S10b trace capture (NewIdeas QUEUE.md row S10b, GO 10-25 15:27; slim spec 10-26 00:xx: ~8 spread predictor layers +
all layers' routing ids, <= ~1 GB) for TrainLab's long-lead expert predictor. Runs the shipped 24 GB v3 decode (JOINT,
spec125 adaptive width, MTP) over 4 domains x 5 prompts with BIGMOE_S10B=1 (bigmoe/s10b.py) and BIGMOE_S10B_MISS (ext),
and writes OUT_DIR/:
  pNN_<domain><i>.npz  per prompt, bigmoe/s10b.py flush(): one entry per decode/verify call of a streamed MoE layer:
                       seq (pass), L (layer: 0-47 trunk, 48 = MTP head), rows, row_off -> ids [rows, 10] int16 (routed
                       experts, score descending); x_off (-1 unless L is a predictor layer) -> x [rows, 2560] uint16 = bf16
                       bits of that layer's MoE input (the router's input). A verify pass = one seq with L 0..47, the
                       same rows in the same order at every layer (take each layer's FIRST call: after a partly accepted
                       verify, MTPLX's commit re-runs layer 1 over the kept rows under the same seq; the ext loader
                       never sees it); MTP draft calls are L = 48, each under its own seq.
  miss.bin             ext, written at exit: int32 records [seq, L, n, (e | miss << 16) x n], one per (pass, ext layer)
                       the loader served: its unique experts, miss = not resident at service (prefetched-not-promoted
                       counts as a miss, as in b30's FC1 log). Ext L < side0 (meta) = base pools (= model layer),
                       L >= side0 = LSB sidecars of layer L - side0 (4-bit ranks only).
  res.bin              (b32, with BIGMOE_S10B_RES) ext, written at exit: int32 records [seq, L, nT, (T, w_0..w_15) x nT],
                       one per JIT prefetch decision of base layer L (right before the ext's schedule(), which skips the
                       resident experts): the residency of pools T = L+1..L+3 at that moment, bit e % 32 of w_{e / 32} =
                       expert e resident. The deployable score: recall of (seq, T)'s misses by the top-16 predictions
                       among experts NOT resident in (seq, L -> T).
  routers.npz          wT = layer T's router weight [512, 2560] f16 (dequantized) for every target T = L+1..L+3.
  meta.json            layout, prompts, per-prompt counts and shas, env.
S10B_X_IDX=4 (optional) keeps x only for those prompt indices per domain (b32: the held-out prompts), x_off = -1 elsewhere.
usage: s10b_trace.py FRAC N OUT_DIR   (the stack's env from the runner, b31_runner.sh / b32_runner.sh)"""
import ctypes, hashlib, json, os, sys, time
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for _k, _v in (("MTPLX_FAMILY_CAPTURE_COMMIT", "1"), ("MTPLX_DROP_EVENTS", "1"), ("MTPLX_LAZY_MTP_HISTORY_APPEND", "1"),
               ("MTPLX_DEFER_REPAIR_EVAL", "1")):
    os.environ.setdefault(_k, _v)
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
assert os.environ.get("BIGMOE_S10B") and os.environ.get("BIGMOE_S10B_MISS"), "set BIGMOE_S10B=1 and BIGMOE_S10B_MISS"
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
import numpy as np
import mlx.core as mx
from bigmoe.loader import load_streamed
from bigmoe.gated import ext, PAUSED
from bigmoe import s10b
import spec125

FRAC, N, OUT = float(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
os.makedirs(OUT, exist_ok=True)
MD = os.path.expanduser("~/BigMoE/models/bare")
cfg = json.load(open(os.path.join(MD, "config.json"))); cfg = cfg.get("text_config", cfg)
NL = cfg["num_hidden_layers"]
rt, st = load_streamed(MD, max(16, round(FRAC * cfg["num_experts"])), mtp=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mtphead                                          # R-60: BIGMOE_MTP_HEAD=<safetensors> overlays a trained MTP head
MTP_HEAD = mtphead.from_env(rt)
tok = rt.tokenizer

SNIP_LOG = '''class Uploader:
    def __init__(self, client, bucket):
        self.client = client
        self.bucket = bucket
        print("uploader ready for", bucket)

    def upload(self, path):
        print("uploading", path)
        try:
            with open(path, "rb") as f:
                self.client.put(self.bucket, path, f.read())
        except OSError as e:
            print("failed to read", path, e)
            return False
        print("done", path)
        return True

    def upload_all(self, paths):
        ok = 0
        for p in paths:
            if self.upload(p):
                ok += 1
        print(ok, "of", len(paths), "uploaded")
        return ok
'''
SNIP_JS = '''const fs = require("fs");

function loadConfig(path, cb) {
  fs.readFile(path, "utf8", (err, text) => {
    if (err) return cb(err);
    let cfg;
    try { cfg = JSON.parse(text); } catch (e) { return cb(e); }
    fs.stat(cfg.dataDir, (err2, st) => {
      if (err2) return cb(err2);
      if (!st.isDirectory()) return cb(new Error("dataDir is not a directory"));
      fs.readdir(cfg.dataDir, (err3, files) => {
        if (err3) return cb(err3);
        cb(null, { cfg, files: files.filter(f => f.endsWith(".csv")) });
      });
    });
  });
}

module.exports = { loadConfig };
'''
SNIP_HINTS = '''import math

def mean(xs):
    return sum(xs) / len(xs)

def stdev(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

def zscores(xs):
    m, s = mean(xs), stdev(xs)
    return [(x - m) / s for x in xs]

def outliers(xs, limit=3.0):
    return [x for x, z in zip(xs, zscores(xs)) if abs(z) > limit]

def summary(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return {"n": len(vals), "mean": mean(vals), "stdev": stdev(vals), "outliers": outliers(vals)}
'''
SNIP_SHAPES = '''import math

class Shape:
    def area(self):
        raise NotImplementedError

class Circle(Shape):
    def __init__(self, r):
        self.r = r
    def area(self):
        return math.pi * self.r ** 2

class Rect(Shape):
    def __init__(self, w, h):
        self.w, self.h = w, h
    def area(self):
        return self.w * self.h

def total_area(shapes: list[Shape]) -> float:
    return sum(s.area() for s in shapes)
'''
GSM = [json.loads(l)["question"] for l in open(os.path.expanduser("~/EvalSuite/data/raw/gsm8k_test.jsonl"))][:5]
MATH_ASK = "Solve this step by step, then give the final line as 'Answer: <number>'.\n\n"
PROMPTS = {
    "code": [open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read(),
             "Write a Python class LRUCache with get(key) and put(key, value) in O(1) time using collections.OrderedDict, "
             "with type hints and docstrings, then three pytest tests for it.",
             "Write a Python function that reads a CSV file of transactions (date, account, amount) and returns the monthly "
             "total per account as a nested dict. Use only the standard library, add type hints, and handle malformed rows.",
             "Write a Rust function that returns the k most frequent words in a text (case-insensitive, ties broken "
             "alphabetically), plus unit tests.",
             "Write a Node.js script that reads users.json (an array of objects with name, email and an optional birthDate), "
             "prints the five oldest users, and skips entries without a valid date. Use async/await."],
    "prose": ["Explain, in plain prose for a curious 14-year-old, how a CPU cache works and why it makes programs faster. "
              "Use one everyday analogy and keep it to about 300 words.",
              "Write a short story of about 300 words about a lighthouse keeper who finds a message in a bottle.",
              "Explain the main causes of the French Revolution to a high-school student in about 300 words.",
              "Write a friendly email to a neighbour asking them to water your plants while you travel for two weeks, "
              "with the practical details they will need.",
              "Describe in plain language, in about 250 words, how vaccines train the immune system."],
    "edit": ["Here is a file:\n\n```python\n" + open(os.path.expanduser("~/SpecDecode/scripts/inventory_sample.py")).read()
             + "```\n\nRewrite the whole file with total_value renamed to total everywhere. Output only the full file in one code block.",
             "Here is a file:\n\n```python\n" + SNIP_LOG + "```\n\nRewrite it to use the logging module (a module-level logger) "
             "instead of print. Output only the full file in one code block.",
             "Here is a file:\n\n```javascript\n" + SNIP_JS + "```\n\nRewrite loadConfig to use fs.promises and async/await "
             "instead of callbacks, keeping the behaviour. Output only the full file in one code block.",
             "Here is a file:\n\n```python\n" + SNIP_HINTS + "```\n\nAdd type hints and a one-line docstring to every function. "
             "Output only the full file in one code block.",
             "Here is a file:\n\n```python\n" + SNIP_SHAPES + "```\n\nRename the class Shape to Figure everywhere and add a "
             "perimeter() method to every class. Output only the full file in one code block."],
    "math": [MATH_ASK + q for q in GSM],
}
V0 = {1: 42, 2: 59, 3: 74, 4: 87, 5: 94, 6: 107, 7: 117, 8: 123}
if os.environ.get("S10B_KEYS"):                            # W1B (docs/WALL1_BENCH.md): EvalSuite's prompts by key, e.g. S45's
    _SET = {"humaneval": "humaneval", "math500": "math", "gsm8k": "math", "ifeval": "ifeval", "multi": "multi", "agent": "agent",
            "mbpp": "mbpp"}                                    # mbpp: E24 (Sat 17:4x), EvalSuite's dec_mbpp.jsonl
    _items = {}
    PROMPTS = {}
    for _k in os.environ["S10B_KEYS"].split(","):            # six; their prompt ids are used as they are (EvalSuite's template)
        _s = _SET[_k.split("/")[0]]
        if _s not in _items:
            _items[_s] = [json.loads(l) for l in open(os.path.expanduser(f"~/EvalSuite/data/items/dec_{_s}.jsonl"))]
        PROMPTS.setdefault(_s, []).append(next(x for x in _items[_s] if x["key"] == _k)["prompt_ids"])


def ids_of(t):
    if isinstance(t, list):                                    # S10B_KEYS: EvalSuite's prompt ids as they are
        return list(t)
    i = tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True, enable_thinking=False)
    return list(i if isinstance(i[0], int) else i["input_ids"])


if os.environ.get("BIGMOE_WIRED") == "1":                     # Sat 17:xx (E24; Leap FCP 038bfa1): MLX's residency set for the run
    _wl = int(mx.device_info()["max_recommended_working_set_size"])
    print(f"BIGMOE_WIRED: mx.set_wired_limit({_wl / 1e9:.2f} GB), was {mx.set_wired_limit(_wl) / 1e9:.2f} GB", flush=True)
spec125.generate(rt.model, ids_of("Hi"), 8, k_mtp=1, max_draft=7, lookup=1)   # warm-up
st.gate.check()
s10b.drop()
meta = {"spec": "S10b (NewIdeas QUEUE.md); layout in this file's docstring (scripts/s10b_trace.py)", "frac": FRAC, "N": N,
        "layers_pred": list(s10b.LAYERS), "n_layers": NL, "side0": getattr(st, "_side0", None), "K": cfg.get("num_experts_per_tok"),
        "E": cfg["num_experts"], "hidden": cfg["hidden_size"], "x_dtype": "bf16 bits as uint16",
        "env": {k: v for k, v in os.environ.items() if k.startswith(("BIGMOE_", "MTPLX_", "S10B_"))}, "prompts": []}
if os.environ.get("BIGMOE_S10B_RES"):
    meta["res"] = {"file": os.path.basename(os.environ["BIGMOE_S10B_RES"]), "W": (cfg["num_experts"] + 31) // 32,
                   "record": "[seq, L, nT, (T, w_0..w_{W-1}) x nT], bit e % 32 of w_{e // 32} = expert e resident"}
X_IDX = {int(v) for v in os.environ["S10B_X_IDX"].split(",")} if os.environ.get("S10B_X_IDX") else None
IDX = {int(v) for v in os.environ["S10B_IDX"].split(",")} if os.environ.get("S10B_IDX") else None   # b34: prompt subset


class _RU(ctypes.Structure):                                   # rusage_info_v2: this process's SSD bytes (b34)
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f, ctypes.c_uint64) for f in (
        "user", "system", "idle_wk", "int_wk", "pageins", "wired", "resident", "footprint", "start", "exit", "c_user",
        "c_system", "c_idle_wk", "c_int_wk", "c_pageins", "c_elapsed", "diskio_read", "diskio_written")]


def disk_read():
    ru = _RU()
    return ru.diskio_read if ctypes.CDLL(None).proc_pid_rusage(os.getpid(), 2, ctypes.byref(ru)) == 0 else -1


class _RU4(ctypes.Structure):                                  # rusage_info_v4: u28 = lifetime peak phys_footprint
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"u{i}", ctypes.c_uint64) for i in range(35)]


def lifetime_fp_gb():                                          # the 24 GB line (NewIdeas 10-28)
    ru = _RU4()
    return ru.u28 / 1e9 if ctypes.CDLL(None).proc_pid_rusage(os.getpid(), 4, ctypes.byref(ru)) == 0 else float("nan")


def fp_now_gb():                                               # b46: phys_footprint now (u7)
    ru = _RU4()
    return ru.u7 / 1e9 if ctypes.CDLL(None).proc_pid_rusage(os.getpid(), 4, ctypes.byref(ru)) == 0 else float("nan")


EXT_KEYS = ("misses", "uses", "pf_issued", "pf_used", "pf_ontime", "pf_late", "pf_qdemand", "pf_cancel", "joint_reads",
            "miss_pred_unissued", "miss_unpred", "iss_d1", "iss_d2", "iss_used_d1", "iss_used_d2")
TRACE = bool(os.environ.get("BIGMOE_TRACE"))                  # b34: gate wait, SSD busy and reads per layer per prompt


def union_len(iv):                                            # as in s10_anatomy.py
    tot, cur = 0.0, None
    for a, b in sorted(iv):
        if cur is None or a > cur[1]:
            if cur:
                tot += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    return tot + (cur[1] - cur[0] if cur else 0.0)


def drain_trace():
    """ext.trace(): [L, seq, kind, t0, t1] (kind 1 = the loader saw L's routing row at t0 and released the layer at t1,
    so [t0, t1] is the GPU's gate wait); ext.trace_reads(): [L, e, tag 0 demand / 1 prefetch / 2 joint, issue, done]"""
    gev, rds = np.array(ext.trace()), np.array(ext.trace_reads())
    ext.trace_cov()
    if not len(gev) or not len(rds):
        return {}
    g1 = gev[gev[:, 2] == 1]
    Lt = rds[:, 0].astype(int) * 3 + rds[:, 2].astype(int)
    u, n = np.unique(Lt, return_counts=True)
    return dict(gate_wait_s=union_len(g1[:, 3:5].tolist()), ssd_busy_s=union_len(rds[:, 3:5].tolist()),
                reads=[[int(v // 3), int(v % 3), int(c)] for v, c in zip(u, n)])


t_all = time.perf_counter()
for dom, texts in PROMPTS.items():
    for i, text in enumerate(texts):
        if IDX is not None and i not in IDX:
            continue
        name = f"p{len(meta['prompts']):02d}_{dom}{i}"
        if TRACE:
            drain_trace()
        e0, b0 = {k: ext.stats().get(k, 0) for k in EXT_KEYS}, disk_read()
        pm0 = ext.pom_stats() if hasattr(ext, "pom_stats") else None   # R-44 b: POM / LP1 read and use counts
        aw = spec125.AdaptiveWidth(V0, step_ms=2.2, alpha=2.0)
        p0 = dict(PAUSED); PAUSED["t_last"] = 0.0
        mx.reset_peak_memory()                                  # R-38: this prompt's MLX peak (prefill + decode)
        t0 = time.perf_counter()
        out, s = spec125.generate(rt.model, ids_of(text), N, k_mtp=1, max_draft=7, lookup=1, adapt=aw,
                                  stop_ids=({248044, 248046} if os.environ.get("S10B_EOS") == "1" else None))   # S10B_EOS=1
                                  # (Sat 18:5x, E24 MBPP comparable to r57): stop at the screen's EOS set (r57 EOS = {248044, 248046})
        peak_gb = mx.get_peak_memory() / 1e9
        wall = time.perf_counter() - t0
        st.gate.check()
        tr = drain_trace() if TRACE else {}
        d = s10b.flush()
        if X_IDX is not None and i not in X_IDX:                   # keep ids + seqs, drop the MoE inputs
            d["x"] = d["x"][:0]; d["x_off"][:] = -1
        np.savez(os.path.join(OUT, name + ".npz"), **d)
        sha = hashlib.sha1(str(list(out)).encode()).hexdigest()[:10]
        rows_v = int(d["rows"][d["L"] == 0].sum())                 # verify rows = layer-0 rows (every trunk layer sees them)
        meta["prompts"].append(dict(name=name, domain=dom, idx=i, text_sha=hashlib.sha1((text if isinstance(text, str) else str(text)).encode()).hexdigest()[:10],
                                    n_tok=len(out), tok_s=s["tok_s"], passes=s["passes"], sha=sha, wall=wall,
                                    calls=int(len(d["L"])), verify_rows=rows_v, x_rows=int(d["x"].shape[0]),
                                    paused=PAUSED["pause_s"] - p0["pause_s"], n_gap=PAUSED["n_gap"] - p0["n_gap"],
                                    disk_read=disk_read() - b0,
                                    ext={k: ext.stats().get(k, 0) - e0[k] for k in EXT_KEYS}, trace=tr,
                                    pom=[a - b for a, b in zip(ext.pom_stats(), pm0)] if pm0 is not None else None,
                                    peak_gb=round(peak_gb, 3), lifetime_fp_gb=round(lifetime_fp_gb(), 3),
                                    fp_now_gb=round(fp_now_gb(), 4),
                                    ids=[int(x) for x in out] if os.environ.get("S10B_SAVE_IDS") == "1" else None))   # E24m (Sat 18:5x)
        print(f"[{name}] {len(out)} tok, {s['tok_s']:.1f} tok/s, {s['passes']} passes, verify rows {rows_v}, "
              f"x rows {d['x'].shape[0]}, sha {sha}", flush=True)
        json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
meta["wall_s"] = time.perf_counter() - t_all
# b46 (bounce pool fix): S10B_GAPS="n:s" -> after the prompts, a warm 16-token generate of the first prompt, then n idle
# gaps of s seconds (longer than GCD's idle-worker reap time), each followed by the same generate; phys_footprint now
# around each step (thread-local read buffers leaked about one per retired worker per gap)
if os.environ.get("S10B_GAPS"):
    n_gap, gap_s = os.environ["S10B_GAPS"].split(":")
    text0 = next(iter(PROMPTS.values()))[0]
    meta["gaps"] = []
    for g in range(-1, int(n_gap)):
        f0 = fp_now_gb()
        if g >= 0:
            time.sleep(float(gap_s))
        f1 = fp_now_gb()
        out, s = spec125.generate(rt.model, ids_of(text0), 16, k_mtp=1, max_draft=7, lookup=1,
                                  adapt=spec125.AdaptiveWidth(V0, step_ms=2.2, alpha=2.0))
        s10b.flush()                                           # this generate's trace is not kept
        if TRACE:
            drain_trace()
        meta["gaps"].append(dict(gap=g, fp_before_gap=round(f0, 4), fp_after_gap=round(f1, 4),
                                 fp_after_gen=round(fp_now_gb(), 4), lifetime_fp_gb=round(lifetime_fp_gb(), 4),
                                 sha=hashlib.sha1(str(list(out)).encode()).hexdigest()[:10],
                                 bounce=ext.bounce_stats() if hasattr(ext, "bounce_stats") else None))
        print(f"[gap {g}] {json.dumps(meta['gaps'][-1])}", flush=True)
# routers of the target layers (qwen4_exp: the forecaster applies router_{L+d} to layer L's MoE input as is)
try:
    layers = rt.model.language_model.model.layers
    W = {}
    for T in sorted({L + dd for L in s10b.LAYERS for dd in (1, 2, 3) if L + dd < NL}):
        g = layers[T].mlp.gate
        if hasattr(g, "scales"):
            w = mx.dequantize(g.weight, g.scales, getattr(g, "biases", None), g.group_size, g.bits,
                              **({"mode": g.mode} if hasattr(g, "mode") else {}))
        else:
            w = g.weight
        W[f"w{T}"] = np.array(w.astype(mx.float16))
    np.savez(os.path.join(OUT, "routers.npz"), **W)
    meta["routers"] = sorted(W)
except Exception as e:                                         # the trace itself is already on disk
    meta["routers_error"] = repr(e)
    print("routers:", repr(e), flush=True)
try:                                                           # b33: the ext's own miss split, to cross-check the labels
    meta["ext_stats"] = {k: v for k, v in ext.stats().items() if isinstance(v, (int, float))}
except Exception as e:
    meta["ext_stats_error"] = repr(e)
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
tot = sum(p["verify_rows"] for p in meta["prompts"])
print(f"saved {OUT}: {len(meta['prompts'])} prompts, {tot} verify rows, {meta['wall_s']:.0f} s "
      f"(miss.bin follows at exit)", flush=True)
