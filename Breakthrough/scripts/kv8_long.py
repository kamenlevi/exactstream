"""Queue L2-06 task check (Breakthrough, 2026-10-25): does 8-bit QSA KV (b12_kv8) change long-context retrieval?
One long document per length (the first 32K / 60K tokens of data/prompts/ctx64k.txt: real Python source) with planted
facts, then three questions as successive user turns on the same cache (one long prefill per length and arm):
  mq   : 8 needles "The special magic number for {word} is: {7 digits}." at depths 10..95 %; 4 asked, the other 4 are
         distractors of the same form (RULER multi-key style); one point per asked number in the answer
  vt   : a 5-step variable chain (VAR A = 48213. VAR B = VAR A. ...) over depths 5..93 % plus a 3-step distractor chain
         with another value (RULER vt); one point per chain name in the answer
  uuid : 40 UUID key -> value pairs as one JSON block at depth 30 %; one key asked; one point for its value
Model path: B12LL (B12_LATE3=2,10, mass), decoupled prefill (512 / 4096) of the first turn; later turns are fed one row
at a time (decode semantics, so B12LL's late writes stay in order); greedy one-row decode with bt_b12's repetition guard
as b12_generate runs it, EOS stop. Arms bf16 then kv:64 per length in one process, on the same token ids (memory is
measured separately, one arm per process, by kv8_tf.py).
usage: kv8_long.py build | run    (env KV8L_LENS=32000,60000 KV8L_ARMS=bf16,kv:64)
       build -> data/kv8/long_tasks.json (CPU, tokenizer only); run -> data/kv8/long_<L>.json + a line per arm in
       data/kv8/results.txt
KV8L_SET=_dx (10-26, NewIdeas: more distractor items before any 8-bit KV adoption): per length one document with 6
interleaved 4-step variable chains (distinct values, names unique in the document), each chain asked as its own turn:
6 vt items per prefill, and every other chain's names are an item's distractors. Files long_tasks_dx.json /
long_<L>_dx<TAG>.json.
Re-run (NewIdeas 12:30, after the INVALID 09:44 run): every item, the first included, starts from fresh copies of the
post-document buffers (snapshot / restore below; the first run handed back the snapshot's own arrays, which later writes
changed); a 2K self-test per arm first (prefill, answer, restore, other item, restore, answer again: same ids, same
first-step logits and the same cache fingerprint after every restore, else the run stops); cap KV8L_CAP (160) tokens;
a run whose re-asked item 0 differs is INVALID. The 14:00 self-test failed with the cache restored bit-exactly (the
single-row passes' bits depend on the decode pools), so since NewIdeas' 14:08 GO the pools are also emptied before every
item, as PFR leaves them after a long prefill (run 3, tag _g64_lane_r3)."""
import hashlib, json, os, random, sys, time

HOME = os.path.expanduser("~/Breakthrough")
CTX = os.path.join(HOME, "data/prompts/ctx64k.txt")
SET = os.environ.get("KV8L_SET", "")                   # "" (the 3-question set) or "_dx"
TASKS = os.environ.get("KV8L_TASKS", os.path.join(HOME, f"data/kv8/long_tasks{SET}.json"))
LENS = [int(x) for x in os.environ.get("KV8L_LENS", "32000,60000").split(",")]
ARMS = os.environ.get("KV8L_ARMS", "bf16,kv:64").split(",")
LANE = os.environ.get("KV8L_LANE", "0") == "1"         # L2-10: the first-turn prefill inside mtplx attention_phase("prefill")
TAG = os.environ.get("KV8L_TAG", "")                   # suffix of the record files / result lines (e.g. "_g64_lane")
CAP = int(os.environ.get("KV8L_CAP", "160"))           # _dx answer cap (48 cut every 60K answer in the 09:44 run)
ST_N = int(os.environ.get("KV8L_ST_N", "2048"))        # _dx self-test: prefix tokens (0 = no self-test)
WORDS = ["harbor", "violet", "glacier", "lantern", "meadow", "copper", "falcon", "orchid", "summit", "willow", "ember",
         "quartz"]
TURN = "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def build():
    import uuid
    from tokenizers import Tokenizer
    TK = Tokenizer.from_file(os.path.expanduser("~/BigMoE/models/bare/tokenizer.json"))
    enc = lambda s: TK.encode(s, add_special_tokens=False).ids
    text = open(CTX).read()
    e = TK.encode(text, add_special_tokens=False)
    rng = random.Random(2510)
    out = {"eos": [TK.token_to_id("<|im_end|>"), TK.token_to_id("<|endoftext|>")], "tasks": {}}
    if SET == "_dx":
        return build_dx(TK, enc, text, e, out)
    for L in LENS:
        hay = text[:text.rfind("\n", 0, e.offsets[L][0])]
        words = rng.sample(WORDS, 8)
        nums = [str(rng.randint(1_000_000, 9_999_999)) for _ in words]
        depths = [0.10, 0.22, 0.35, 0.47, 0.60, 0.72, 0.85, 0.95]
        asked = [0, 2, 4, 6]                                  # depths 10 / 35 / 60 / 85 %; 22 / 47 / 72 / 95 % distract
        facts = [(d, f"The special magic number for {w} is: {n}.") for d, w, n in zip(depths, words, nums)]
        names = rng.sample(["".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(5)) for _ in range(20)], 8)
        val, val2 = str(rng.randint(10000, 99999)), str(rng.randint(10000, 99999))
        chain = [f"VAR {names[0]} = {val}."] + [f"VAR {names[i]} = VAR {names[i - 1]}." for i in range(1, 5)]
        dchain = [f"VAR {names[5]} = {val2}."] + [f"VAR {names[i]} = VAR {names[i - 1]}." for i in range(6, 8)]
        facts += list(zip([0.05, 0.27, 0.50, 0.73, 0.93], chain)) + list(zip([0.15, 0.55, 0.80], dchain))
        pairs = [(str(uuid.UUID(int=rng.getrandbits(128))), str(uuid.UUID(int=rng.getrandbits(128)))) for _ in range(40)]
        kq = rng.randrange(len(pairs))
        facts.append((0.30, "UUID pairs (key -> value):\n" + json.dumps(dict(pairs), indent=0)))
        doc = hay
        for d, f in sorted(facts, key=lambda x: -x[0]):       # deepest first: earlier positions do not move
            p = doc.rfind("\n", 0, int(len(hay) * d)) + 1
            doc = doc[:p] + f + "\n" + doc[p:]
        qs = [f"What are the special magic numbers for {', '.join(words[i] for i in asked[:3])} and {words[asked[3]]} "
              f"mentioned in the text above? Answer with the four numbers only, in that order.",
              f"Find all variables that are assigned the value {val} in the text above, directly or through other "
              f"variables. Answer with the variable names only.",
              f"In the JSON object of UUID pairs in the text above, what is the value for the key \"{pairs[kq][0]}\"? "
              f"Answer with the value only."]
        first = ("<|im_start|>user\nBelow is the source of several files from one repository, with some facts mixed in. "
                 "Read it, then answer my questions about it.\n\n" + doc + "\n\n" + qs[0] +
                 "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        out["tasks"][str(L)] = {"first": enc(first), "turns": [enc(TURN.format(q)) for q in qs[1:]], "questions": qs,
                                "gold": [[nums[i] for i in asked], names[:5], [pairs[kq][1]]], "max_tokens": [48, 48, 64]}
        print(f"kv8_long build L={L}: first turn {len(out['tasks'][str(L)]['first'])} tokens, asked {[words[i] for i in asked]}, "
              f"chain {names[:5]} = {val}, uuid key #{kq}", flush=True)
    json.dump(out, open(TASKS, "w"))


def build_dx(TK, enc, text, e, out, nc=6, steps=4):
    rng = random.Random(2610)
    for L in LENS:
        hay = text[:text.rfind("\n", 0, e.offsets[L][0])]
        pool = set()
        while len(pool) < nc * steps:
            pool.add("".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(5)))
        names = rng.sample(sorted(pool), nc * steps)
        vals = [str(v) for v in rng.sample(range(10000, 100000), nc)]
        chains = [names[c * steps:(c + 1) * steps] for c in range(nc)]
        facts = []
        for nm, val in zip(chains, vals):                     # root first, links deeper: each chain reads in order
            ds = sorted(rng.uniform(0.03, 0.97) for _ in range(steps))
            facts += list(zip(ds, [f"VAR {nm[0]} = {val}."] + [f"VAR {nm[i]} = VAR {nm[i - 1]}." for i in range(1, steps)]))
        doc = hay
        for d, f in sorted(facts, key=lambda x: -x[0]):       # deepest first: earlier positions do not move
            p = doc.rfind("\n", 0, int(len(hay) * d)) + 1
            doc = doc[:p] + f + "\n" + doc[p:]
        qs = [f"Find all variables that are assigned the value {v} in the text above, directly or through other "
              f"variables. Answer with the variable names only." for v in vals]
        prefix = ("<|im_start|>user\nBelow is the source of several files from one repository, with some facts mixed in. "
                  "Read it, then answer my question about it.\n\n" + doc)
        # NewIdeas 03:10: every item starts from the same state right after the document (run() rolls the cache back
        # to the prefix before each question), so the 6 items are independent; each question ends the same user turn
        out["tasks"][str(L)] = {"prefix": enc(prefix), "suffixes": [enc("\n\n" + q + "<|im_end|>\n<|im_start|>assistant\n"
                                                                        "<think>\n\n</think>\n\n") for q in qs],
                                "questions": qs, "gold": chains, "kinds": ["vt"] * nc, "max_tokens": [CAP] * nc}
        print(f"kv8_long build_dx L={L}: prefix {len(out['tasks'][str(L)]['prefix'])} tokens, {nc} chains x {steps}: "
              + " ".join(f"{v}={'/'.join(c)}" for v, c in zip(vals, chains)), flush=True)
    json.dump(out, open(TASKS, "w"))


def run():
    os.environ.setdefault("B12_LATE3", "2,10")
    os.environ.setdefault("B12_NORM", "mass")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import bt_b12 as B
    rt, st = B.load_b12(0.10)
    import mlx.core as mx
    import numpy as np
    import b12_kv8 as K
    K.install()
    from bigmoe_v3f.fuse import reset_generation_state
    from bigmoe_v3f import repguard as RG
    from contextlib import nullcontext
    from mtplx.attention_context import attention_phase
    model, tok = rt.model, rt.tokenizer
    T = json.load(open(TASKS))
    EOS, IM_END = set(T["eos"]), T["eos"][0]
    import ctypes, threading

    class _RU(ctypes.Structure):                             # rusage_info_v4 (as kv8_tf): u7 = physical footprint now
        _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"u{i}", ctypes.c_uint64) for i in range(40)]
    libc = ctypes.CDLL(None)

    def fp_now():
        ru = _RU()
        return ru.u7 / 1e9 if libc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(ru)) == 0 else 0.0

    PK = {}                                                  # (footprint, MLX active, b12_prefill PHASE) at the max
                                                             # footprint and at the max footprint - active, per phase
    def pk_reset():
        PK.update(fp=(0.0, 0.0, None), gap=(0.0, 0.0, None))

    def sampler():                                           # where the long prefill's footprint peak sits (20 Hz)
        while True:
            try:
                f, a, p = fp_now(), mx.get_active_memory() / 1e9, B.PHASE.get("p")
                if f > PK["fp"][0]:
                    PK["fp"] = (f, a, p)
                if f - a > PK["gap"][0] - PK["gap"][1]:
                    PK["gap"] = (f, a, p)
            except Exception:
                pass
            time.sleep(0.05)
    pk_reset()
    threading.Thread(target=sampler, daemon=True).start()
    fmt = lambda r: f"{r[0]:.2f} GB (MLX active {r[1]:.2f}) at {r[2]}"

    def step(y, cache):
        lg = model(mx.array([[y]]), cache=cache)
        return (lg[0] if isinstance(lg, tuple) else lg)[0, -1]

    def answer(y, cache, n, lgh=None):                       # b12_generate's loop (lgh: gets the first-step logits sha)
        guard = RG.make()
        out = []
        while len(out) < n:
            lg = step(y, cache)
            if lgh is not None and not out:
                lgh.append(hashlib.sha1(np.array(lg.astype(mx.float32)).tobytes()).hexdigest()[:10])
            y = int(mx.argmax(RG.steer(guard, out, lg) if guard is not None and guard.armed else lg).item())
            if out:
                RG.observe(guard, out)
            out.append(y)
            if y in EOS:
                break
        return out

    def _leaves(v):
        if isinstance(v, mx.array):
            return [v]
        if isinstance(v, (tuple, list)):
            return [a for x in v for a in _leaves(x)]
        if isinstance(v, dict):
            return [a for x in v.values() for a in _leaves(x)]
        return []

    def _clone(v):
        """a fresh buffer with the same bits. Never the snapshot's own array objects: a later setitem or a donated write
        would change them (MTPLX cache_state._clone_tree's rule, its issue #247; the 09:44 run kept them and its fork
        check failed). Floats go through an integer view, so -0.0 stays -0.0 (x + 0 would not)."""
        if isinstance(v, mx.array):
            u = {2: mx.uint16, 4: mx.uint32, 8: mx.uint64}.get(v.dtype.size) if mx.issubdtype(v.dtype, mx.floating) else None
            return mx.view(mx.view(v, u) + 0, v.dtype) if u is not None else v + mx.zeros((), v.dtype)
        if isinstance(v, (tuple, list)):
            return type(v)(_clone(x) for x in v)
        if isinstance(v, dict):
            return {k: _clone(x) for k, x in v.items()}
        return v

    QF = ("raw_keys", "pooled", "pooled_f32_t", "pooled_len", "_reserved_raw_capacity", "_reserved_pooled_capacity")
    KF = ("keys", "values", "_k", "_v", "offset")            # stock QSAKVCache keys / values, b12_kv8's 8-bit _k / _v

    def _fields(c):                                          # (kind, the cache's own fields, its KV's fields)
        if hasattr(c, "kv"):
            d, kv = vars(c), vars(c.kv)
            extra = [k for k, v in list(d.items()) + list(kv.items()) if _leaves(v) and k not in QF + KF]
            assert not extra, f"QSA cache arrays the snapshot does not cover: {extra}"
            return "qsa", {k: d[k] for k in QF if k in d}, {k: kv[k] for k in KF if k in kv}
        assert isinstance(getattr(c, "cache", None), list), type(c)
        assert getattr(c, "left_padding", None) is None and getattr(c, "lengths", None) is None, "batched ArraysCache"
        return "arr", list(c.cache), None

    def snapshot(cache):
        """every layer's buffers right after the prefix, copied whole (full capacity; the 8-bit KV as stored, no
        dequantize / requantize): QSA = KV + offset, the indexer's raw and pooled keys, the fp32 mirror and lengths;
        linear layers = their ArraysCache list (conv / recurrent state, the PLE n-gram history)"""
        assert not B.STASH and not B.PENDH, "late writes pending after the prefill"
        snap = [_clone(_fields(c)) for c in cache]
        mx.eval(_leaves(snap))
        return snap

    def empty_pools():
        """NewIdeas 14:08 (dx fix GO): every item gets the post-document pool state. The feed and the answer are
        single-row passes, which add cold MSB-3 ranks 2-9 late (B12_LATE3), so their bits depend on what the decode
        pools hold (the 14:00 self-test: cache restored bit-exactly, item 0's answer still changed). A long prefill
        leaves the pools empty (PFR re-creates them empty; without PFR the prefill reads into its own scratch), so they
        are emptied before every item: bt_b12's PFR release + re-create (ext.reset_layer, every table invalidated)"""
        B.PFR_LOG.append(("reset",))
        with B._PFR():
            pass

    def restore(cache, snap):
        """fresh copies of the snapshot into every layer and empty pools, before every item (the first included)"""
        for c, (kind, a, b) in zip(cache, snap):
            if kind == "qsa":
                for k, v in a.items():
                    object.__setattr__(c, k, _clone(v))
                for k, v in b.items():
                    object.__setattr__(c.kv, k, _clone(v))
            else:
                c.cache[:] = _clone(a)
        mx.eval(_leaves([_fields(c) for c in cache]))
        B.STASH.clear(); B.PENDH.clear()
        reset_generation_state(model)                        # LC1: no indexer-core state carried between items
        empty_pools()

    def fingerprint(cache):                                  # sha of every snapshotted buffer, bit for bit
        h = hashlib.sha1()
        for a in _leaves([_fields(c) for c in cache]):
            u = {2: mx.uint16, 4: mx.uint32, 8: mx.uint64}.get(a.dtype.size) if mx.issubdtype(a.dtype, mx.floating) else None
            h.update(np.array(mx.view(a, u) if u is not None else a).tobytes())
        for c in cache:                                      # and the plain ints (offsets, lengths)
            _, a, b = _fields(c)
            h.update(repr(sorted((k, v) for k, v in (a.items() if isinstance(a, dict) else ()) if isinstance(v, int))).encode())
            h.update(repr(sorted((k, v) for k, v in (b or {}).items() if isinstance(v, int))).encode())
        return h.hexdigest()[:10]

    def set_arm(arm):
        K.ON[0] = arm != "bf16"
        if K.ON[0]:
            part, gs = arm.split(":")
            K.CONF.update(part=part, gs=int(gs))
        reset_generation_state(model)
        B.STASH.clear(); B.PENDH.clear()
        mx.clear_cache()

    def selftest(t, arm):
        """NewIdeas 12:30 (2): prefill ST_N tokens of the document, item 0, restore, item 1, restore, item 0 again: the
        two item-0 answers must match (ids + first-step logits) and every restore must give the post-prefill fingerprint"""
        set_arm(arm)
        cache = model.make_cache()
        with (attention_phase("prefill") if LANE else nullcontext()):
            B.b12_prefill(model, mx.array([t["prefix"][:ST_N]]), cache, 512, 4096)
        snap = snapshot(cache)
        h0, got = fingerprint(cache), []
        for q in (0, 1, 0):
            restore(cache, snap)
            h, lgh = fingerprint(cache), []
            for f in t["suffixes"][q][:-1]:
                mx.eval(step(f, cache))
            got.append((h, answer(t["suffixes"][q][-1], cache, CAP, lgh), lgh[0]))
        ok = all(g[0] == h0 for g in got) and got[0][1:] == got[2][1:]
        print(f"kv8_long selftest {arm} ({ST_N} tokens, pools emptied before every item, last reset "
              f"{B.PFR_LOG[-1][1:]} GB / release s / re-create s): {'PASS' if ok else 'FAIL'} | fingerprint after prefill {h0}, after "
              f"each restore {' '.join(g[0] for g in got)} | item 0 logits {got[0][2]} / again {got[2][2]}, ids equal "
              f"{got[0][1] == got[2][1]} ({len(got[0][1])} / {len(got[2][1])} tokens) | item 1 logits {got[1][2]} | "
              f"{tok.decode([v for v in got[0][1] if v not in EOS])[:60]!r}", flush=True)
        del cache, snap
        return ok

    t0_ = T["tasks"][str(LENS[0])]
    if "prefix" in t0_ and ST_N:                             # NewIdeas 12:30: the restore is proven first, per arm
        if not all([selftest(t0_, arm) for arm in ARMS]):
            print("kv8_long selftest FAILED: no long runs", flush=True)
            sys.exit(3)
    for L in LENS:
        t = T["tasks"][str(L)]
        fork = "prefix" in t                                 # the _dx set: one prefix, each question from its end
        first = t["prefix"] if fork else t["first"]
        rec = {"L": L, "n_first": len(first), "questions": t["questions"], "gold": t["gold"], "fork": fork, "arms": {}}
        for arm in ARMS:
            set_arm(arm)
            cache = model.make_cache()
            B.PHASE.clear(); pk_reset(); mx.reset_peak_memory()
            t0 = time.perf_counter()
            with (attention_phase("prefill") if LANE else nullcontext()):
                B.b12_prefill(model, mx.array([first if fork else first[:-1]]), cache, 512, 4096)
            tp = time.perf_counter() - t0
            mem = {"prefill_fp_peak": PK["fp"], "prefill_gap_peak": PK["gap"], "prefill_mlx_peak": mx.get_peak_memory() / 1e9}
            B.PHASE["p"] = "qa"; pk_reset()
            snap = snapshot(cache) if fork else None
            h0 = fingerprint(cache) if fork else None
            y, prev, res = first[-1], None, []
            for qi in range(len(t["questions"]) + (1 if fork else 0)):   # fork: item 0 again at the end (fork check)
                q = qi % len(t["questions"])
                if fork:
                    restore(cache, snap)
                    hn = fingerprint(cache) if qi == len(t["questions"]) else None   # outside the timings
                t1 = time.perf_counter()
                if fork:
                    feed = t["suffixes"][q]
                elif qi:                                     # close the last answer, then the next user turn
                    feed = ([prev[-1]] if prev and prev[-1] != IM_END else []) + t["turns"][qi - 1]
                else:
                    feed = [y]
                for f in feed[:-1]:
                    mx.eval(step(f, cache))
                t2 = time.perf_counter()
                prev = answer(feed[-1], cache, t["max_tokens"][q])
                text = tok.decode([v for v in prev if v not in EOS])
                if qi == len(t["questions"]):                # a differing re-ask makes the run INVALID (NewIdeas 12:30)
                    rec.setdefault("fork_check", {})[arm] = {"ids_equal": prev == res[0]["ids"], "text": text,
                                                             "fingerprint": [h0, hn]}
                    break
                hits = [str(g).lower() in text.lower() for g in t["gold"][q]]
                res.append({"text": text, "hits": hits, "s": round(time.perf_counter() - t1, 1), "ids": prev,
                            "feed": [len(feed) - 1, round(t2 - t1, 2)], "gen": [len(prev), round(time.perf_counter() - t2, 2)]})
            st.gate.check()
            mem["qa_fp_peak"] = PK["fp"]
            gen_n, gen_s = sum(r["gen"][0] for r in res), sum(r["gen"][1] for r in res)
            feed_n, feed_s = sum(r["feed"][0] for r in res), sum(r["feed"][1] for r in res)
            rec["arms"][arm] = {"prefill_s": round(tp, 1), "res": res, "mem": mem,
                                "decode_tok_s": round(gen_n / max(gen_s, 1e-9), 2), "feed_rows_s": round(feed_n / max(feed_s, 1e-9), 2)}
            line = (f"long L={L}{SET}{TAG} ({len(first)} tok) {arm}: " + " ".join(
                f"{q} {sum(r['hits'])}/{len(r['hits'])}" for q, r in zip(t.get("kinds", ("mq", "vt", "uuid")), res)) +
                f" | prefill {tp:.0f} s | decode {rec['arms'][arm]['decode_tok_s']} tok/s ({gen_n} tokens), feed "
                f"{rec['arms'][arm]['feed_rows_s']} rows/s" +
                (f" | fork check item 0 again: {'identical' if rec['fork_check'][arm]['ids_equal'] else 'DIFFERS -> INVALID'}"
                 f" (fingerprint {'/'.join(rec['fork_check'][arm]['fingerprint'])})" if fork else "") +
                " | " + " || ".join(repr(r["text"][:70]) for r in res))
            print(line, flush=True)
            print(f"long L={L} {arm} memory: prefill footprint peak {fmt(mem['prefill_fp_peak'])}; widest footprint - "
                  f"active {fmt(mem['prefill_gap_peak'])}; prefill MLX peak {mem['prefill_mlx_peak']:.2f} GB; Q&A footprint "
                  f"peak {fmt(mem['qa_fp_peak'])}", flush=True)
            if "KV8L_TASKS" not in os.environ:                # smoke runs on other task files leave no record
                with open(os.path.join(HOME, "data/kv8/results.txt"), "a") as f:
                    f.write(time.strftime("%m-%d %H:%M ") + line + "\n")
                json.dump(rec, open(os.path.join(HOME, f"data/kv8/long_{L}{SET}{TAG}.json"), "w"), indent=1)
            del cache


def score():
    """precision beside the hits: distractors named in each answer (vt: the names of every VAR line in the document that
    is not in the gold chain, i.e. the distractor chain; mq: the 7-digit numbers of the unasked needles). CPU, tokenizer."""
    import re
    from tokenizers import Tokenizer
    TK = Tokenizer.from_file(os.path.expanduser("~/BigMoE/models/bare/tokenizer.json"))
    T = json.load(open(TASKS))
    for L in LENS:
        p = os.path.join(HOME, f"data/kv8/long_{L}{SET}{TAG}.json")
        if not os.path.exists(p):
            continue
        rec, doc = json.load(open(p)), TK.decode(T["tasks"][str(L)].get("first") or T["tasks"][str(L)]["prefix"])
        gold = rec["gold"]
        if SET == "_dx":                                      # each item: its chain's names found, other chains' named
            for arm, a in rec["arms"].items():
                per = []
                for qi, r in enumerate(a["res"]):
                    others = [(c, [v for v in g if v in r["text"]]) for c, g in enumerate(gold) if c != qi]
                    per.append((sum(r["hits"]), sum(len(n) for _, n in others),
                                [c for c, n in others if len(n) >= len(gold[c]) - 1]))
                print(f"long L={L}{SET} {arm}: gold names {sum(x[0] for x in per)}/{sum(len(g) for g in gold)}, "
                      f"distractor names {sum(x[1] for x in per)}, items with a (near-)whole other chain "
                      f"{sum(1 for x in per if x[2])} {[i for i, x in enumerate(per) if x[2]]} | per item hits/distractors " +
                      " ".join(f"{x[0]}/{x[1]}" for x in per) + f" | decode {a.get('decode_tok_s')} tok/s | fork check " +
                      ("same" if rec.get("fork_check", {}).get(arm, {}).get("ids_equal") else "DIFFERS -> INVALID RUN"),
                      flush=True)
            continue
        dvars = sorted(set(re.findall(r"VAR ([A-Z]{5}) =", doc)) - set(gold[1]))
        dnums = sorted(set(re.findall(r"special magic number for \w+ is: (\d{7})", doc)) - set(gold[0]))
        for arm, a in rec["arms"].items():
            mq, vt = a["res"][0]["text"], a["res"][1]["text"]
            print(f"long L={L} {arm}: mq {sum(a['res'][0]['hits'])}/4 + {sum(n in mq for n in dnums)} distractor numbers | "
                  f"vt {sum(a['res'][1]['hits'])}/5 + {sum(v in vt for v in dvars)} distractor names {[v for v in dvars if v in vt]}"
                  f" | uuid {sum(a['res'][2]['hits'])}/1", flush=True)


if __name__ == "__main__":
    {"build": build, "run": run, "score": score}[sys.argv[1]]()
