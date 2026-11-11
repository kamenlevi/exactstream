"""B12 (NewIdeas' PLAN_B candidate for one Eval suite decode verdict, Breakthrough 2026-10-21): frugal routing at ~10 % on
BigMoE's v3 tier engine, in a pinned copy (~/Breakthrough/v3f = BigMoE 7b8c7b4 bigmoe/ + ext/ as bigmoe_v3f, plus a
flag-mode loader: bit 20 = read and wait, bit 21 = late-eligible, unflagged = used only if resident; tierglu.py and hc8.py
pinned from MoonShots). Levers, each an env switch so the suite can run B12 minus one without a rebuild:
  B12_CM=51 B12_CL=20   MSB-3 base slots / LSB-1 sidecar slots per layer (plan 4433333333: ranks 0-1 at 4 bits, the rest
                        MSB-3; RAM ~ 48 x (CM x 2.15 + CL x 0.77) MB). Default CM = round(FRAC x 512).
  frugal (always)       router top-20 by score; ranks 0-1 read and waited for; the first 10 resident in rank order are
                        used (missing ranks 2-9 -> resident ranks 10-19), weights renormalised over those 10 (w2f).
  B12_HC=4 | 8 | 0      hyper-connection mixers (+ PLE K/V) quantised to 4 / 8 bits (MoonShots' patches/hc8); default 8
                        (8-bit: KLD +0.0000 hard / +0.0001 easy, -0.62 GB, not slower; 4-bit: +0.024 hard. Quantised mixers turn
                        BigMoE's HC fusions off; B12_HC=0 = bf16 mixers with the compiled HC).
  B12_LATE1=1 | 0       (default 0: no speed gain on v3) wait for rank 0 only; a rank 1 with either record part cold is read after phase A and its output
                        (on layer L's MoE input, rank 1's renormalised weight, at 4 bits) is written into the HC streams
                        after layer L+1's attention, before its MLP read (tf_frugal_var.py l1t2); default 1.
  B12_PF=1 [B12_SPARE=4] prefetch arm 1 (NewIdeas I8): layer L+1's corrected d=1 top-2 per row posted on L's base and
                        sidecar rows (both record parts), prefetched after L's demand reads (non-JIT), spares from C.
  B12_PRE=1 | 0         TrainLab's pre-routed frugal (replaces late rank 1; default 0 until validated): in layers 8-39
                        (B12_PRE_BAND) the two waited (4-bit) experts are layer L's router + MoEPrefetch's correction
                        on layer L-1's MoE input, top-2; the rest = the true router's best outside them, resident only;
                        weights = the TRUE router's probabilities over the chosen 10, renormalised. Here the pre-picks
                        ride in layer L's own row (quality-faithful; the one-layer slack needs an early post).
  B12_LATE3=2,4         "B8" (NewIdeas): MSB-3 ranks 2-3, when cold, are read after phase A and added half a layer
                        late (base pool only, BIGMOE_DEFER = 2 pairs); with FRAC 0.05 = B8 (frugal 5 % on v3).
  B12_ENGINE=v3o | v3p | v3n | v3m | v3l | v3i | v3g | v3f   engine copy. Default v3o again since 11-01 (Fit12 M-5b: v3p,
                        the default 10-31..11-01 = v3o + Fit12's R-53 prefill scratch reuse, adds +0.69 GB while a short
                        prompt (no PFR below B12_PFR_MIN) is read with the pools resident: math lifetime 12.35 GB > the 11.95
                        line; v3o 11.64 / 11.61 GB on the same rows, tokens equal). v3o (R-45): v3n + read_expert's
                        bounce buffer held in a thread_local holder that frees it at thread exit (exact; GCD retires idle
                        workers, ~7 per request, and v3n leaked 2.8 MiB with each). v3n since R-48 (10-29): v3m + BigMoE's M-01
                        unpack; load_b12 sets BIGMOE_M01=1 (B12_M01 below), so the pools read BigMoE's records3m (records3
                        with the per-group fields re-packed: exact; decode SSD bytes/token -5.3 %, tok/s +3.5 %).
                        Earlier default v3m since R-41 (10-28): v3l + drain() (no async read
                        lands in a released pool) + reset_layer with a new slot count; bitwise = v3l (AR 8.5K refs, MTP
                        p2k / p8k / p16k / p32k, write16). Earlier default v3l (10-26, below). v3i (10-25) = v3g + MoonShots' gate seq stamps: every routing-row,
                        release-region and table word carries seq & 0x3FF, so a torn read under another process's GPU load
                        is re-read instead of serving the wrong experts (the load-only B8/B12L/B12LL divergence); bitwise =
                        v3g when quiet, cost <= 0.2 %. v3g = v3f + Fit12's B1 v2: base + LSB sidecar rows of a layer in ONE
                        read burst, BIGMOE_JOINT=1; plus Fit12 SB1 BIGMOE_SPIN_G2=1.05 + BIGMOE_SPIN_PAIRS=256; both exact,
                        +5.4 % and +1.8 % on B12LL. v3f = the pinned engine of the 10-22..24 verdicts, run exactly as then.
                        v3l (10-26) = v3i + Fit12's PFR engine calls (reset_layer, invalidate_tables): = v3i unless called
                        (same KV / logits / sha32 as Fit12's v3p at 8.5K and 64K).
  B12_PFR=1 | 0         (default 1 since 10-26 14:0x; grid keep since R-41, see B12_PFR_GRID) Fit12's PFR-a: an AR-path
                        prefill of n > B12_PFR_MIN tokens releases the 48 trunk tier pools (6.0 GB), runs k = ceil(n / 16384)
                        balanced MoE chunks on the 512-token query grid, then re-creates the pools empty (decode starts
                        cold). A/B (data/pfr): KV, first-step logits, sha32 identical at 8,474 and 65,000 tokens; prefill
                        27.5 -> 18.9 s and 265 -> 208 s; prefill-phase peak 11.05 -> 9.27 and 13.54 -> 11.08 GB. Needs
                        v3l; skipped (noted once) on paths not A/B'd: MTP feeds, kept streams, the lane, frugal, D1I ring.
  B12_PFR_MIN=4096      PFR only on prefills above this many tokens (or on a fold C_m change). 512 (R-41E, 10-30, exact:
                        below 4096 the MoE still runs once over the prompt, grid keep) removes the short-prompt spike
                        (agent/write14, 1,945 tokens: +1.2 GB in its row without PFR, the R-41D fold-fast overshoot;
                        lifetime -0.7 GB) but each 513-4096-token row then decodes from empty pools (+9.5 / +34 % row
                        time on write14 / edit10); so 512 is for a process whose memory line binds (fold-fast).
  B12_PFR_GRID=keep | balanced   (default keep since R-41, 10-28) keep: PFR releases the pools but keeps the caller's
                        MoE grid (s_moe 4096), so its bits = PFR off. balanced: Fit12's k = ceil(n / 16384) chunks (the
                        L2-11 speed: 64K prefill -21 %), NOT exact: when the 4096 grid's last chunk is small its rows
                        take another gather kernel (write16: 8,276 tokens = 4096 + 4096 + 84 rows split at token 32);
                        L2-11's A/B lengths had 282- and 3,560-row tails. Non-exact option: Kamen's call.
  B12_ALLWAIT=0 | 1     (default 0; Fit12 CMP3 / AW 10-26; fold pending Fit12's R-01 and the AR screen) single-row trunk
                        passes take the verify rows' flags (FL_MR: every forced rank read and waited for) instead of
                        B12_LATE3's late cold ranks, so decode bits stop depending on what the pools hold. MTP path:
                        bitwise across C_m 51 / 59 / 65, cost -0.3 %. B.AW[0] switches it at run time; B.AW_N counts it.
                        On the AR path it is -3.6 % at 8.5K (AW-AR, 10-27): AR keeps the late rule.
  B12_FOLD=1 | 0        (default 1 since R-41, 10-28, NewIdeas: "fold-exact") Fit12's MTP-path stack for b12_prefill_state
                        + spec125.generate (the Eval suite's MTP rows). ON by default, all bitwise to the old path
                        (R-41: p2k / p8k / p16k / p32k in one process and write16, every hash; lifetime peak 11.09-11.46
                        GB where the old path reached 13.03 at 32K):
                          PFR on MTP prefills above B12_PFR_MIN tokens (pools released, the 4096 grid kept: B12_PFR_GRID);
                          the QSA buffers sized once to n + max_new + 16 rows around prefill and decode (R-02; max_new
                          = B12_FOLD_ROWS, 2048); the prefilled state MOVED into the decode instead of deep-copied
                          (MOV: such a state feeds ONE generate).
                        OPT-IN, not bitwise to the old path (Kamen's call, with an Eval-suite verdict): B12_FOLD_AW=1
                        (ALLWAIT inside spec125.generate) and B12_FOLD_CM=1 or "n:C_m" points (C_m per prompt: default
                        points 2700:64,8600:60,16500:56,32800:47, linear between, rounded down; needs v3m; a short
                        prompt then takes PFR only when its C_m changes). Off: B12_FOLD_PFR / _ROWS / _MOV = 0.
> (a quote from the project owner, kept private)
                        Eval suite's b12llff3 ran it (same scores, lifetime 11.51 GB on the 29-row agent process, ~6 %
                        faster answers, ~27 % faster long prompts; NOT bit-exact): ALLWAIT inside the decode (B12_FOLD_AW 1),
                        C_m per prompt at the FFX points 2700:60,8600:60,16500:56,32800:47 (B12_FOLD_CM), PFR on prompts
                        above 512 tokens (B12_PFR_MIN) with balanced MoE chunks (B12_PFR_GRID). exact = fold-exact (the
                        bit-exact mode: AW 0, CM 0, PFR_MIN 4096, grid keep). Each variable still overrides its piece.
  B12_TRIMS=1 | 0       exact memory trims (Fit12 L2-02/03, default 1): hc8 applied to the still-lazy bf16 mixers (no bf16
                        copy at load: peak -1.3 GB) and embed_tokens streamed from the safetensors rows (-0.36 GB always).
                        B12_CM=59 + trims: +3.5-5.4 % tok/s, peak 11.76 GB at a 2.6K prompt but 12.43 GB at 8K (opt-in).
  B12_FRS_NOCYCLE=1 | 0 exact memory fix (default 1; Fit12 R-45): mtplx's _full_vocab_head builds every FR-Spec draft head
                        in a local class that super()'s __class__ cell keeps in a reference cycle, and its closure held
                        each replaced pruned head (0.095 GB) until a gen-2 GC: the ~0.1 GB per request creep of R-41C.
                        The wrapper empties the closure's head / ids cells once built (only __init__ reads them).
  B12_M01=1 | 0         exact (default 1 on engine v3n; NewIdeas R-48): BIGMOE_M01=1, the expert pools and the prefill
                        read ~/BigMoE/models/bare/records3m (MSB part 2,048,000 / LSB 720,896 B per expert vs records3's
                        2,162,688 / 770,048) and v3n's ext unpacks the fields (BigMoE's m01_* block, verbatim). R-48 A S S A:
                        every hash equal (p2k, p8k, write16), decode rusage bytes/token -5.26 %, prefill -5.66 %, tok/s
                        +3.45 %, peak 10.98 vs 11.00 GB. 0 (or any other engine) = records3, v3m's reads.
Agreed configs (NewIdeas 22:05, HC8 default from 01:00): B12 = load_b12(0.10) with the defaults (8-bit mixers, no late);
  B8 = load_b12(0.05) with B12_LATE3=2,4; B8-fast = load_b12(0.05) with B12_W=3 (wait for ranks 0-2, no phase B).
Decode 1-row calls take this path; prefill and multi-row calls stay v3's exact paths. Note: the compiled module is named
bigmoe_ext like BigMoE's: import nothing from ~/BigMoE/bigmoe in the same process.
API (Eval suite): load_b12(frac=0.10) -> (rt, st).   CLI: bt_b12.py FRAC speed (PROMPT code|prose, BT_N) | check | gsmN"""
import glob, hashlib, json, os, re, sys, time

ENGINE = os.environ.get("B12_ENGINE", "v3o")   # v3o (default again since 11-01, Fit12 M-5b) = v3n + the bounce-buffer leak fix; v3p (default 10-31..11-01) = v3o + Fit12's R-53 prefill scratch reuse (exact; +0.69 GB in short prompts' prefill); v3n (default since R-48) = v3m + M-01 unpack; v3m = v3l + drain() + resizable pools; v3l = v3i + PFR calls; v3i = v3g + gate stamps; v3g = v3f + B1 v2; v3f = pinned
V3F = os.path.expanduser(f"~/Breakthrough/{ENGINE}")
_EXT_ENV = None                             # the engine env the ext was first imported with (read once, at import)
_EXT_KEYS = ("BIGMOE_DEFER", "BIGMOE_SPARE", "BIGMOE_PF_NOPUBLISH", "BIGMOE_JOINT", "BIGMOE_SPIN_PAIRS", "BIGMOE_SPIN_G2",
             "BIGMOE_M01")
K, KC = 10, 20
STASH, PENDH = {}, {}
CFG = {}
PHASE = {}                                  # b12_prefill's current (MoE chunk, layer, part), for B12_FP_SAMPLE
_ST = None                                  # load_b12's streamer: B12_PFR releases and re-creates its trunk tier pools
PFR_LOG = []                                # per PFR prefill: (tokens, s_moe, GB released, release s, restore s)
AW = [os.environ.get("B12_ALLWAIT", "0") == "1"]   # B12_ALLWAIT, switchable at run time (B.AW[0] = True / False)
AW_N = [0]                                  # trunk layer calls that took the ALLWAIT flags (proves the switch engaged)
FOLD = [os.environ.get("B12_FOLD", "1") == "1"]   # B12_FOLD (default on since R-41: its exact pieces; see the docs)


> (a quote from the project owner, kept private)
assert MODE in ("fast", "exact"), f"B12_MODE={MODE}: fast or exact"
_FAST = MODE == "fast"
_FOLD_ON = {"PFR": "1", "ROWS": "1", "MOV": "1", "AW": "1" if _FAST else "0",   # exact pieces always on; fast adds
            "CM": "2700:60,8600:60,16500:56,32800:47" if _FAST else "0"}      # ALLWAIT + C_m per prompt (FFX points)
_PFR_GRID0, _PFR_MIN0 = ("balanced", "512") if _FAST else ("keep", "4096")
os.environ.setdefault("MTPLX_NGRAM_HOT_MB", "16")   # R-46 (Fit12, exact): the n-gram hot-row cache capped for 12 GB


def _fs(part):                              # one B12_FOLD piece: B12_FOLD_<PART>=0 turns it off, =1 (or C_m points) on
    return FOLD[0] and os.environ.get("B12_FOLD_" + part, _FOLD_ON[part]) != "0"


def _env():
    os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
    for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
        os.environ.setdefault(k, "0")
    os.environ["BIGMOE_TIER"] = "4444444444" if CFG.get("plan4") else "4433333333"
    os.environ.setdefault("BIGMOE_TIER_GLU", "3c")
    os.environ["BIGMOE_SPARE"] = str(CFG["spare"]) if CFG["pf"] else "0"   # frugal: no background reads, unless B12_PF
    os.environ.pop("BIGMOE_PF_JIT", None)            # the loader turns JIT on if the variable exists at all
    os.environ["BIGMOE_PF_NOPUBLISH"] = "1"
    os.environ["BIGMOE_DEFER"] = str(CFG["l3"][1] - CFG["l3"][0]) if CFG["l3"] else ("1" if CFG["late1"] else "0")   # pairs
    if ENGINE in ("v3n", "v3o", "v3p") and os.environ.get("B12_M01", "1") != "0":   # R-48 (exact): records3m, unpacked by v3n-v3p's ext
        os.environ["BIGMOE_M01"] = "1"                               # (read at import: before bigmoe_v3f loads the ext)
    else:                                            # every other engine reads records3 (its ext has no M-01 unpack)
        os.environ.pop("BIGMOE_M01", None)
    if ENGINE != "v3f":                              # Fit12 B1 v2 + SB1 (exact): read at import / loader start
        os.environ.setdefault("BIGMOE_JOINT", "1")
        os.environ.setdefault("BIGMOE_SPIN_G2", "1.05")
        os.environ.setdefault("BIGMOE_SPIN_PAIRS", "256")


_SE_CLS = None


def _streamed_embedding(model_dir, group_size=64, bits=4, mode="affine"):
    """Fit12 L2-03 (fit_embed.py, copied 2026-10-24): the untied 4-bit embed_tokens (248,320 x 2560, g64: 358 MB) only ever
    gathers the rows of a call's tokens. Gather them from numpy memmaps over the safetensors shard (clean file pages, not in
    the physical footprint) and run QuantizedEmbedding's own mx.dequantize: bitwise the resident table's rows. An nn.Module
    (built lazily, so importing this file does not import mlx), so assigning it replaces the table in the module tree."""
    global _SE_CLS
    if _SE_CLS is None:
        import mmap, struct
        import numpy as np
        import mlx.core as mx
        import mlx.nn as nn
        prefix = "language_model.model.embed_tokens"

        def view(x, dt):
            return x if dt is None else (x.view(dt) if hasattr(x, "view") else mx.view(x, dt))

        class StreamedEmbedding(nn.Module):
            def __init__(self, model_dir, group_size, bits, mode):
                super().__init__()
                wm = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
                mm = {}
                for f in ("weight", "scales", "biases"):
                    path = os.path.join(model_dir, wm[f"{prefix}.{f}"])
                    with open(path, "rb") as fh:
                        n = struct.unpack("<Q", fh.read(8))[0]
                        hdr, start = json.loads(fh.read(n)), 8 + n
                    info = hdr[f"{prefix}.{f}"]
                    dt = {"U32": np.uint32, "BF16": np.uint16, "F16": np.uint16}[info["dtype"]]
                    a = np.memmap(path, mode="r", dtype=dt, offset=start + info["data_offsets"][0], shape=tuple(info["shape"]))
                    try:
                        a._mmap.madvise(mmap.MADV_WILLNEED)    # page-cache read-ahead (not in the footprint)
                    except (AttributeError, OSError, ValueError):
                        pass
                    mm[f] = (a, {"BF16": mx.bfloat16, "F16": mx.float16}.get(info["dtype"]))
                object.__setattr__(self, "_mm", mm)            # outside the module tree: no parameters
                object.__setattr__(self, "_q", (group_size, bits, mode))

            def _rows(self, f, flat):
                a, dt = self._mm[f]
                return view(mx.array(np.ascontiguousarray(a[flat])), dt)

            def __call__(self, x):
                ids = np.asarray(x).astype(np.int64)            # decode / prefill ids are host data already
                flat = ids.reshape(-1)
                gs, bits, mode = self._q
                out = mx.dequantize(self._rows("weight", flat), scales=self._rows("scales", flat),
                                    biases=self._rows("biases", flat), group_size=gs, bits=bits, mode=mode)
                return out.reshape(*ids.shape, out.shape[-1])
        _SE_CLS = StreamedEmbedding
    return _SE_CLS(model_dir, group_size, bits, mode)


def _install_repguard():
    """BigMoE's repetition guard (bigmoe/repguard.py = MTPLX's LoopGuard: exact while disarmed; arms when a 12-gram recurs
    4x in the last 2,048 tokens past 256, then steers 1-token steps) is what SpecDecode's spec125.generate imports as
    `from bigmoe import repguard`. On the B12 path ~/BigMoE is not importable, so the guard was silently OFF (Eval suite
    B12 verdict 10-22: truncation rule failed by one row, a looping poem). Register the identical copy pinned in v3f under
    that name, without importing BigMoE's package; a real importable bigmoe is left alone. BIGMOE_LOOP_GUARD=0 = off."""
    import importlib, types
    try:
        importlib.import_module("bigmoe.repguard")
        return
    except ImportError:
        pass
    rg = importlib.import_module("bigmoe_v3f.repguard")
    pkg = sys.modules.get("bigmoe")
    if pkg is None:
        pkg = types.ModuleType("bigmoe")
        pkg.__path__ = []                                            # a namespace-like stub: only repguard lives here
        sys.modules["bigmoe"] = pkg
    pkg.repguard = rg
    sys.modules["bigmoe.repguard"] = rg


def load_b12(frac=0.10, mtp=False):
    cm = int(os.environ.get("B12_CM", "0")) or round(frac * 512)
    plan4 = os.environ.get("B12_PLAN4") == "1"                      # every used rank at 4 bits (D1n a; opt-in)
    CFG.update(frac=frac, hc=int(os.environ.get("B12_HC", "8")), late1=os.environ.get("B12_LATE1", "0") == "1",
               cm=cm, cl=int(os.environ.get("B12_CL", "0")) or (cm if plan4 else round(0.4 * cm)), plan4=plan4,
               pre=os.environ.get("B12_PRE", "0") == "1",
               band=tuple(int(v) for v in os.environ.get("B12_PRE_BAND", "8,40").split(",")),
               l3=tuple(int(v) for v in os.environ["B12_LATE3"].split(",")) if os.environ.get("B12_LATE3") else None,
               w=int(os.environ.get("B12_W", "2")),
               pf=os.environ.get("B12_PF") == "1", spare=int(os.environ.get("B12_SPARE", "4")),
               norm=os.environ.get("B12_NORM", "used"))
    assert CFG["w"] >= 2 and (not CFG["l3"] or CFG["l3"][0] >= CFG["w"]), (CFG["w"], CFG["l3"])
    assert not (CFG["plan4"] and (CFG["l3"] or CFG["late1"])), "B12_PLAN4 has no late paths yet"
    if CFG["pre"]:
        CFG["late1"] = False                         # pre-routing replaces late rank 1 (both decide the waited pair)
    if CFG["l3"]:                                    # B8: MSB-3 ranks A..B-1 late (base pool only); not with late rank 1
        assert CFG["l3"][0] >= 2 and CFG["l3"][1] > CFG["l3"][0] and not CFG["late1"], CFG["l3"]
    global _EXT_ENV, _ST
    if _EXT_ENV is None and "bigmoe_ext" in sys.modules and os.environ.get("B12_ALLOW_EARLY_EXT") != "1":
        raise RuntimeError("bigmoe_ext was imported before load_b12 set the engine env: the ext reads BIGMOE_DEFER / SPARE / "
                           "PF_NOPUBLISH / JOINT once, at import (Fit12 04:40: 47 spin timeouts). Call load_b12 before "
                           "importing bigmoe_v3f, or set B12_ALLOW_EARLY_EXT=1 if you set that env yourself.")
    _env()
    os.environ["BIGMOE_TIER_CM"], os.environ["BIGMOE_TIER_CL"] = str(CFG["cm"]), str(CFG["cl"])
    if CFG["hc"]:     # hc8 swaps the mixers for quantised ones: MTPLX's eager HC path only (hc8.py: not with BigMoE's HC
        for k in ("BIGMOE_COMPILE_HC", "BIGMOE_HC_VREAD", "BIGMOE_HC_PACK"):   # fusions; COMPILE_HC's kernels expect
            os.environ[k] = "0"                                                  # bf16 mixers: Metal build error)
    if V3F not in sys.path:
        sys.path.insert(0, V3F)
    import mlx.core as mx
    from bigmoe_v3f.loader import load_streamed
    from bigmoe_v3f import tier as T
    _install_repguard()                                               # spec125's `from bigmoe import repguard`
    md = os.path.expanduser("~/BigMoE/models/bare")
    cfg = json.load(open(os.path.join(md, "config.json"))); cfg = cfg.get("text_config", cfg)
    NL = cfg["num_hidden_layers"]
    now = {k: os.environ.get(k) for k in _EXT_KEYS}
    if _EXT_ENV is None:
        _EXT_ENV = now
    elif now != _EXT_ENV:
        raise RuntimeError(f"load_b12 again with a different engine env {now} != {_EXT_ENV}: the ext keeps its first values; "
                           "use a new process")
    trims = os.environ.get("B12_TRIMS", "1") == "1"
    import bigmoe_v3f.stream as BS
    inst0 = BS.ExpertStreamer.install

    def inst(self, model, layers=None):              # Fit12 L2-02/03: before the loader's mx.eval(model.parameters())
        r = inst0(self, model, layers)
        tm = model.language_model.model
        qe = tm.embed_tokens
        if hasattr(qe, "group_size"):                # first: hc8.install evaluates TM.parameters() at its end
            tm.embed_tokens = _streamed_embedding(self.model_dir, qe.group_size, qe.bits, getattr(qe, "mode", "affine"))
            CFG["trim_embed"] = f"{qe.bits}-bit g{qe.group_size}"
        if CFG["hc"]:                                # the bf16 mixers are quantised while still lazy: never resident
            import hc8
            CFG["trim_hc"] = hc8.install(model, bits=CFG["hc"])
        return r
    if trims:
        BS.ExpertStreamer.install = inst
    try:
        rt, st = load_streamed(md, CFG["cm"], mtp=mtp)
    finally:
        BS.ExpertStreamer.install = inst0
    model = rt.model if hasattr(rt, "model") else rt
    TM = model.language_model.model if hasattr(model, "language_model") else model.model
    for i, l in enumerate(TM.layers):
        l._b12_L = i                                  # layer index for the late writes (layer_call)
    CFG["routers"] = [l.mlp.gate for l in TM.layers]   # B12_PRE: layer L+1's router, applied at layer L
    if CFG["hc"]:
        import hc8                                   # v3f/hc8.py (MoonShots 3055bf7)
        print("b12: hc", hc8.install(model, bits=CFG["hc"]), flush=True)
        mx.clear_cache()
    _install(mx, T, NL)
    if os.environ.get("B12_PF_LA") == "1":            # D1k: exact prefill look-ahead reads (tier.LA; outputs unchanged)
        import numpy as np
        z = np.load(os.path.expanduser(os.environ.get("BIGMOE_PF_CORR", "~/MoEPrefetch/data/corr125_r128.npz")))
        corr = [tuple(mx.array(z[k_][i].astype(np.float32)) for k_ in ("A", "B", "mu", "delta")) for i in range(z["A"].shape[0])]
        bms = [l.mlp.switch_mlp._bm for l in TM.layers]
        T.LA = dict(L0={bm["L"]: i for i, bm in enumerate(bms)}, bms=bms, routers=[l.mlp.gate for l in TM.layers],
                    corr=corr if os.environ.get("B12_PF_LA_CORR", "1") == "1" else None,
                    k=int(os.environ.get("B12_PF_LA_K", "10")))
    if os.environ.get("B12_CACHE_MB"):                # cap MLX's buffer cache AFTER loading (MTPLX's memory plan sets
        mx.set_cache_limit(int(os.environ["B12_CACHE_MB"]) << 20)   # its own at load); freed buffers count as RAM
    print(f"b12: engine {ENGINE}, trims {'embed ' + CFG.get('trim_embed', '-') + ' hc ' + str(CFG.get('trim_hc', '-')) if trims else 'off'}", flush=True)
    print(f"b12: C_m={CFG['cm']} C_l={CFG['cl']} ({100 * CFG['cm'] / 512:.0f} %), HC {CFG['hc'] or 'bf16'}, late rank 1 "
          f"{'on' if CFG['late1'] else 'off'}, late MSB-3 ranks {CFG['l3'] or 'none'}, pools {st.resident_gb():.2f} GB, "
          f"active {mx.get_active_memory() / 1e9:.2f} GB",
          flush=True)
    _ST = st
    if mtp:
        _frs_nocycle()
    return rt, st


def _install(mx, T, NL):
    from mtplx.models import qwen4_exp as Q
    from mtplx.models.qwen4_exp import _hyper_residual_write
    from mtplx.attention_context import current_attention_phase
    from mtplx.attention_math import attention_gate
    TG = T._tier_glu()
    orig = Q.SparseMoeBlock.__call__                 # v3's tier block (prefill, multi-row): installed by load_streamed
    R = mx.arange(KC)
    W20, L21 = 1 << 20, 1 << 21
    PREP, CORR, ROUT = {}, [], []
    R22 = mx.arange(KC + 2)
    FL22 = mx.where(R22 < 2, W20, 0).astype(mx.int32)[None]          # pre-routed rows: the two pre-picks waited
    if CFG["pre"] or CFG["pf"]:                      # layer L+1's routers + MoEPrefetch's d=1 correction (source-indexed)
        import numpy as np
        z = np.load(os.path.expanduser(os.environ.get("BIGMOE_PF_CORR", "~/MoEPrefetch/data/corr125_r128.npz")))
        CORR = [(mx.array(z["A"][i]), mx.array(z["B"][i]), mx.array(z["mu"][i]), mx.array(z["delta"][i]))
                for i in range(z["A"].shape[0])]
        ROUT = CFG["routers"]
    A3, B3 = CFG["l3"] or (0, 0)
    W = CFG["w"]                                                      # B12_W: ranks < W read and waited for (default 2)
    PLAN4 = CFG.get("plan4", False)                                   # B12_PLAN4: LSB sidecar for every used rank
    NORM_MASS = CFG.get("norm") == "mass"                             # B12_NORM: used (w2f, default) | mass
    FL = mx.where(R == 0, W20, mx.where(R == 1, L21 if CFG["late1"] else W20,
                                        mx.where(R < W, W20, mx.where((R >= A3) & (R < B3), L21, 0)))).astype(mx.int32)[None]
    NF = max(2, B3, W)                                                # ranks 0..NF-1 always among the 10 used
    FL_MR = mx.where(R < NF, W20, 0).astype(mx.int32)[None]          # verify rows: every forced rank waited
    from bigmoe_v3f.gated import MAXK
    SMAX = MAXK // KC                                                 # rows per call on the frugal path
    FL_ALL = mx.where(R < max(2, B3, CFG["w"]), W20, 0).astype(mx.int32)[None]   # last layer: nothing late, all forced waited
    WK_M, WK_L = 255 | (K << 8), 255 | (2 << 8)                       # flag mode; LRU-touch ranks < 10 / < 2

    def late3_fn(x2, wl, sml, dep_m, n_m, t, gate, L):
        """B8: the late MSB-3 ranks A3..B3-1 of layer L (base pool only), after layer L+1's attention."""
        Mp = B3 - A3

        def f(after):
            o = mx.isnan(after.reshape(-1)[:1]).astype(mx.int32) * 0
            pr = gate.spin(mx.concatenate([dep_m[:1] + o, mx.array([n_m, 0] * Mp, mx.int32)]), L, 2 * Mp, phase=1)
            p, s = pr[0::2], pr[1::2]                                 # (position, slot) pairs; pads (n, 0)
            match = p[None, :] == mx.arange(A3, B3)[:, None]          # [late ranks, pairs]
            sl_late = mx.where(match, s[None, :], 0).sum(axis=-1)     # each late rank's slot (0 if it was resident)
            ms = mx.where(sml[0] >= 0, sml[0], sl_late)[None]         # resident at phase A: weight 0 here anyway
            return TG(x2, t["base"], t["sideP"], ms, None, plan="3" * Mp, scores=wl).reshape(1, 1, -1)
        return f

    def late_fn(x2, wl, sm1, sl1, dep_m, dep_l, n_m, n_l, t, gate, L):
        def f(after):                                                 # spins ordered after layer L+1's attention
            o = mx.isnan(after.reshape(-1)[:1]).astype(mx.int32) * 0
            pm = gate.spin(mx.concatenate([dep_m[:1] + o, mx.array([n_m, 0], mx.int32)]), L, 2, phase=1)
            pl = gate.spin(mx.concatenate([dep_l[:1] + o, mx.array([n_l, 0], mx.int32)]), t["side"], 2, phase=1)
            ms = mx.where(sm1 >= 0, sm1, pm[1:2].reshape(1, 1))       # each record part: phase A's slot if it was in,
            ls = mx.where(sl1 >= 0, sl1, pl[1:2].reshape(1, 1))       # else the late read's
            return TG(x2, t["base"], t["sideP"], ms, ls, plan="4", scores=wl).reshape(1, 1, -1)   # weight 0 unless late
        return f

    def pre_routed(x, x2, gates, ids, pre, L, t, gate):
        """B12_PRE: the pre-picked pair (read and waited, 4-bit), then the true router's best resident outside it."""
        row = mx.concatenate([pre, ids], axis=-1)                     # [1, 22]: pre-picks, then the true top-20
        dep_m, n_m = gate.post(L, row | FL22, wk=WK_M)
        dep_l, n_l = gate.post(t["side"], pre | FL22[:, :2], wk=WK_L)
        sm = gate.spin(dep_m, L, n_m).reshape(1, KC + 2)
        sl = gate.spin(dep_l, t["side"], n_l).reshape(1, 2)
        dup = (ids == pre[:, 0:1]) | (ids == pre[:, 1:2])             # a pre-pick also in the true top-20: used once
        pool = (sm[:, 2:] >= 0) & ~dup
        selr = pool & (mx.cumsum(pool.astype(mx.int32), axis=-1) <= K - 2)
        sel = mx.concatenate([mx.ones((1, 2), mx.bool_), selr], axis=-1)
        pos = mx.argsort(mx.where(sel, R22, R22 + 1000), axis=-1)[:, :K]
        used = mx.take_along_axis(sel, pos, axis=-1)
        s10 = mx.where(used, mx.take_along_axis(sm, pos, axis=-1), sm[:, :1])
        ids10 = mx.take_along_axis(row, pos, axis=-1)
        w10 = mx.where(used, mx.take_along_axis(gates, ids10, axis=-1), 0.0)   # the TRUE router's probabilities
        w10 = w10 / w10.sum(axis=-1, keepdims=True)
        y = TG(x2, t["base"], t["sideP"], s10, sl, plan="".join(T.PLAN), scores=w10)
        return y.reshape(*x.shape[:-1], y.shape[-1])

    def block(self, x):
        bm = getattr(self.switch_mlp, "_bm", None)
        S = x.shape[-2] if x.ndim == 3 else 0
        if bm is None or "tier" not in bm or x.ndim != 3 or x.shape[0] != 1 or not 1 <= S <= SMAX:
            return orig(self, x)                                      # prefill / long verify: v3's exact tier paths
        L, t, gate = bm["L"], bm["tier"], bm["gate"]
        if L == bm["first"]:
            gate.new_pass()
        ae = bm.get("async_every", 0)
        if ae and L % ae == ae - 1:
            mx.async_eval(x)
        if current_attention_phase() == "decode_verify":             # v3's shared-expert form per phase (numerics)
            shared = attention_gate(self.shared_expert(x), self.shared_expert_gate(x))
        else:
            shared = mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
        x2 = x.reshape(S, x.shape[-1])
        gates = mx.softmax(self.gate(x2), axis=-1, precise=True)
        inds = mx.argpartition(gates, kth=-KC, axis=-1)[..., -KC:]
        sc = mx.take_along_axis(gates, inds, axis=-1)
        o = mx.argsort(-sc, axis=-1)
        ids = mx.take_along_axis(inds, o, axis=-1).astype(mx.int32)   # [S, 20] rank order per row
        p20 = mx.take_along_axis(sc, o, axis=-1)
        if CFG["pre"] and S == 1:
            b0, b1 = CFG["band"]
            if b0 <= L + 1 < b1 and L < len(CORR):                    # pick layer L+1's waited pair from this input
                A, Bm, mu, dl = CORR[L]
                x32 = x2.astype(mx.float32)
                lg = ROUT[L + 1](x2).astype(mx.float32) + ((x32 - mu) @ A.astype(mx.float32)) @ Bm.astype(mx.float32) + dl
                PREP[L + 1] = mx.argpartition(-lg, kth=1, axis=-1)[:, :2].astype(mx.int32)   # [1, 2]
            pre = PREP.pop(L, None)
            if pre is not None:
                return pre_routed(x, x2, gates, ids, pre, L, t, gate) + shared
        last = L >= NL - 1                                            # the last trunk layer, and the MTP head's layer
                                                                      # (L 48): no late part there (it was stashed and
                                                                      # never written: MoonShots 20:19)
        one = S == 1
        if one and not last and AW[0]:                                # B12_ALLWAIT (Fit12 CMP3 / AW, 10-26): a single
            fl = FL_MR                                                # row waits for every forced rank like the verify
            AW_N[0] += 1                                              # rows, so its bits do not depend on the pools
        else:
            fl = (FL_ALL if last else FL) if one else FL_MR           # verify rows: late-eligible ranks are waited for
        rows = (ids | fl).T.reshape(-1)                               # rank-major: every row's rank 0, then rank 1, ...
        pk = {}
        if CFG["pf"] and L + 1 < NL and L < len(CORR):               # B12_PF arm 1: layer L+1's corrected d=1 top-2 per
            A, Bm, mu, dl = CORR[L]                                   # row, prefetched (both record parts) after L's
            x32 = x2.astype(mx.float32)                               # demand reads (non-JIT loader, spares from C)
            lg = ROUT[L + 1](x2).astype(mx.float32) + ((x32 - mu) @ A.astype(mx.float32)) @ Bm.astype(mx.float32) + dl
            pr = mx.argpartition(-lg, kth=1, axis=-1)[:, :2].reshape(-1).astype(mx.int32)
            pk = dict(preds=pr, counts=(2 * S,))
        dep_m, n_m = gate.post(L, rows, wk=255 | ((K * S) << 8), **pk)   # flag mode; LRU-touch every row's ranks < 10
        if PLAN4:                                                     # the same top-20 rows and flags as the base:
            dep_l, n_l = gate.post(t["side"], rows, wk=255 | ((K * S) << 8), **pk)   # every rank's LSB part, if resident
        else:
            dep_l, n_l = gate.post(t["side"], (ids[:, :2] | fl[:, :2]).T.reshape(-1), wk=255 | ((2 * S) << 8), **pk)
        sm = gate.spin(dep_m, L, n_m).reshape(KC, S).T                # [S, 20] base (MSB-3) slots, -1 = absent / late
        sl = gate.spin(dep_l, t["side"], n_l).reshape(KC if PLAN4 else 2, S).T   # [S, 20] (PLAN4) / [S, 2] LSB slots
        forced = (R < NF)[None]                                       # ranks 0-1 (+ waited / late ranks) always counted
        pool = ((sm >= 0) & (sl >= 0) if PLAN4 else (sm >= 0)) & ~forced   # PLAN4: resident = BOTH record parts
        sel = (pool & (mx.cumsum(pool.astype(mx.int32), axis=-1) <= K - NF)) | forced   # + first resident of the rest
        pos = mx.argsort(mx.where(sel, R, R + 1000), axis=-1)[:, :K]   # the 10 used positions, rank order (0, 1, ...)
        used = mx.take_along_axis(sel, pos, axis=-1)                  # fewer than 10 resident: the rest are padding
        s10 = mx.take_along_axis(sm, pos, axis=-1)
        s10 = mx.where(used, s10, s10[:, :1])                         # padding reads rank 0's slots at weight 0
        w10 = mx.where(used, mx.take_along_axis(p20, pos, axis=-1), 0.0)
        if NORM_MASS:                                                 # B12_NORM=mass: each used expert at its exact top-10
            w10 = w10 / p20[:, :K].sum(axis=-1, keepdims=True)        # weight (probability / the true top-10 mass); Wed TF
        else:                                                         # probe: hard KLD 0.1066 -> 0.0961 (subnr)
            w10 = w10 / w10.sum(axis=-1, keepdims=True)               # renormalised over the 10 used (w2f)
        l2 = mx.where(used, mx.take_along_axis(sl, pos, axis=-1), sl[:, :1]) if PLAN4 else sl   # [S, 10] (PLAN4)
        if CFG["late1"] and not last and one:
            late = (sm[:, 1:2] < 0) | (sl[:, 1:2] < 0)                # rank 1 not in at phase A (either record part)
            s10 = mx.concatenate([s10[:, :1], mx.where(late, s10[:, :1], s10[:, 1:2]), s10[:, 2:]], axis=-1)
            l2 = mx.concatenate([sl[:, :1], mx.where(late, sl[:, :1], sl[:, 1:2])], axis=-1)
            w_now = mx.concatenate([w10[:, :1], mx.where(late, 0.0, w10[:, 1:2]), w10[:, 2:]], axis=-1)
            STASH[L] = late_fn(x2, mx.where(late, w10[:, 1:2], 0.0), sm[:, 1:2], sl[:, 1:2], dep_m, dep_l, n_m, n_l,
                               t, gate, L)
        elif CFG["l3"] and not last and one:                          # B8: cold ranks A3..B3-1 land half a layer late
            late = sm[:, A3:B3] < 0                                    # [1, B3-A3] (positions = ranks for r < NF)
            s10 = mx.concatenate([s10[:, :A3], mx.where(late, s10[:, :1], s10[:, A3:B3]), s10[:, B3:]], axis=-1)
            w_now = mx.concatenate([w10[:, :A3], mx.where(late, 0.0, w10[:, A3:B3]), w10[:, B3:]], axis=-1)
            STASH[L] = late3_fn(x2, mx.where(late, w10[:, A3:B3], 0.0), sm[:, A3:B3], dep_m, n_m, t, gate, L)
        else:
            w_now = w10
        y = TG(x2, t["base"], t["sideP"], s10, l2, plan="".join(T.PLAN), scores=w_now)   # [S, D] combined in-kernel
        return y.reshape(*x.shape[:-1], y.shape[-1]) + shared

    if not hasattr(Q.DecoderLayer, "_b12_orig_call"):                 # MTPLX's own call, kept once per process
        Q.DecoderLayer._b12_orig_call = Q.DecoderLayer.__call__

    def layer_call(self, hidden, *, input_ids, ssm_mask, cache):     # MTPLX DecoderLayer.__call__ + the late writes
        if getattr(self, "_b12_L", None) is None:                     # not a trunk layer (the MTP head's layer): plain
            return Q.DecoderLayer._b12_orig_call(self, hidden, input_ids=input_ids, ssm_mask=ssm_mask, cache=cache)
        if "ple" in self:
            hidden = hidden + self.ple(hidden, input_ids, cache)
        mixed, hyper, inject = self.attn_hyper_connection(hidden)
        if self.is_linear:
            block_out = self.linear_attn(mixed, ssm_mask, cache)
        else:
            block_out = self.self_attn(mixed, cache)
        L = self._b12_L
        if L == 0:                                                    # a new trunk forward: drop leftovers. MTPLX's
            STASH.clear(); PENDH.clear()                              # commit_verified_window replays layer 1 ALONE on
                                                                      # the kept rows; with keep == 1 its late part went
                                                                      # to PENDH[2] and the NEXT verify pass applied it
                                                                      # (crash on row mismatch, else a stale write)
        for f, inj in PENDH.pop(L, []):                               # layer L-1's late rank 1, before this MLP read
            hyper = _hyper_residual_write(hyper, f(block_out), inj)
        mixed, hyper, inject = self.mlp_hyper_connection(hyper, pending=(block_out, inject))
        block_out = self.mlp(mixed)
        hidden = _hyper_residual_write(hyper, block_out, inject)
        f = STASH.pop(L, None)
        if f is not None:
            PENDH.setdefault(L + 1, []).append((f, inject))
        return hidden

    Q.SparseMoeBlock.__call__ = block
    if CFG["late1"] or CFG["l3"]:
        Q.DecoderLayer.__call__ = layer_call


# ---- D1i (TrainLab, adopted 04:35 Thu): the decoupled prefill as an API for the Eval suite and other harnesses.
# Prefill ids (all prompt tokens but the last) into a fresh model.make_cache(); attention in s_attn-token query
# chunks, the MoE once per s_moe tokens. Not bitwise vs plain chunked prefill (gather kernel shapes per group).
def pfr_chunk(n, s_attn=512, cap=16384):
    """Fit12's PFR-a chunk rule (QUEUE L2-11): k = ceil(n / cap) balanced MoE chunks on the s_attn query grid"""
    k = -(-n // cap)
    sm = s_attn * -(-n // (k * s_attn))
    assert n - (k - 1) * sm > 8, (n, k, sm)            # the last chunk keeps > 8 rows: no tier layer takes the decode path
    return sm


def _pfr_no_decode(*a, **kw):
    raise RuntimeError("B12_PFR: a tier layer took the decode path while its pools were released")


class _PFR:
    """Fit12's PFR (~/Fit12/PLAN.md 'PFR-a RESULT', release / restore of scripts/fit_pfr64.py): the 48 trunk tier pools
    (decode slots, ~6 GB at C_m 51) are released for a long prefill, whose branches all read records3 into their own
    scratch (Fit12's T0 audit), and re-created empty after it; the engine's slots move to the new buffers (v3l
    ext.reset_layer) and every published table is invalidated. tier.tier_call raises while the pools are gone.
    cm (B12_FOLD's C_m per prompt): the base pools come back with cm slots; that needs v3m, whose reset_layer takes a
    new slot count and whose drain() first waits out every in-flight read, so none lands in a released buffer."""

    def __init__(self, cm=None):
        self.cm = cm

    def __enter__(self):
        import gc
        import mlx.core as mx
        from bigmoe_v3f import tier as T
        from bigmoe_v3f.stream import PIECES
        from bigmoe_v3f.gated import ext
        if hasattr(ext, "drain"):                      # v3m: no async read is still writing into the pools let go here
            ext.drain()
        t0, self.info, self.nb, self.call = time.perf_counter(), [], 0, T.tier_call
        for sw in _ST._tier_sws:
            bm = sw._bm
            t = bm["tier"]
            L, Ls = bm["L"], t["side"]
            sb = [(tuple(p.shape), p.dtype) for p in bm["pools"]]
            ss = [(tuple(p.shape), p.dtype) for p in _ST.pools[Ls]]
            self.nb += sum(p.nbytes for p in bm["pools"]) + sum(p.nbytes for p in _ST.pools[Ls])
            ph = [mx.zeros((1, *s[1:]), d) for s, d in sb]
            for (proj, field), arr in zip(PIECES, ph):
                setattr(getattr(sw, proj), field, arr)
            bm["pools"] = _ST.pools[L] = ph
            _ST.pools[Ls] = [mx.zeros((1, *s[1:]), d) for s, d in ss]
            t["base"] = t["sideP"] = None
            self.info.append((sw, L, Ls, sb, ss))
        T.tier_call = _pfr_no_decode
        gc.collect()
        mx.clear_cache()
        self.t_rel = time.perf_counter() - t0
        return self

    def __exit__(self, *exc):
        import gc
        import mlx.core as mx
        from bigmoe_v3f import tier as T
        from bigmoe_v3f.stream import PIECES
        from bigmoe_v3f.gated import ext
        t0 = time.perf_counter()
        T.tier_call = self.call
        gc.collect()
        mx.clear_cache()                               # the prefill's cached buffers go first: no stacked peak
        nb = 0
        for sw, L, Ls, sb, ss in self.info:
            bm = sw._bm
            t = bm["tier"]
            if self.cm is not None and self.cm != sb[0][0][0]:   # B12_FOLD: C_m per prompt (base pools only; C_l stays)
                assert hasattr(ext, "drain"), f"C_m per prompt needs engine v3m (B12_ENGINE=v3m), not {ENGINE}"
                sb = [((self.cm, *s[1:]), d) for s, d in sb]
                bm["C"] = self.cm
            P = [mx.zeros(s, d) for s, d in sb]
            S = [mx.zeros(s, d) for s, d in ss]
            mx.eval(*P, *S)
            for (proj, field), arr in zip(PIECES, P):
                setattr(getattr(sw, proj), field, arr)
            bm["pools"] = P
            _ST.pools[L], _ST.pools[Ls] = P, S
            t["base"] = {p: {f: P[i] for i, (pp, f) in enumerate(T._MSB) if pp == p} for p in ("gate_proj", "up_proj", "down_proj")}
            t["sideP"] = {p: {f: S[i] for i, (pp, f) in enumerate(T._LSB) if pp == p} for p in ("gate_proj", "up_proj", "down_proj")}
            ext.reset_layer(L, P)
            ext.reset_layer(Ls, S)
            nb += sum(p.nbytes for p in P) + sum(p.nbytes for p in S)
        ext.invalidate_tables()
        if self.cm is None:
            assert nb == self.nb, (nb, self.nb)
        else:
            CFG["cm"] = self.cm
        self.info = None
        PFR_LOG[-1] += (round(self.nb / 1e9, 2), round(self.t_rel, 2), round(time.perf_counter() - t0, 2)) + (
            (self.cm, round(nb / 1e9, 2)) if self.cm is not None else ())
        return False


def fold_cm(n):
    """B12_FOLD's C_m for an n-token prompt from the B12_FOLD_CM points "n_i:C_m": the first point's C_m up to its n,
    linear between points, rounded down (never above the points), and the last two points' line extended below the
    last one (floor 16). Default (NewIdeas 10-27 22:2x, after Fit12's R-01): the multi-prompt points, i.e. what a
    long-lived process holds within 11.95 GB while its footprint grows over prompts: 64 up to 2.6K, 60 at 8.5K, 56 at
    16K, 47 at 32K (32K and beyond to be revisited after Fit12's GROW screen)."""
    import math
    v = os.environ.get("B12_FOLD_CM", _FOLD_ON["CM"])  # "n:C_m,..." points (fast: FFX), or 1 for the older points below
    pts = sorted((int(a), int(b)) for a, b in (x.split(":") for x in
                 (v if ":" in v else "2700:64,8600:60,16500:56,32800:47").split(",") if x))
    if n <= pts[0][0] or len(pts) == 1:
        return pts[0][1]
    for (a0, c0), (a1, c1) in zip(pts, pts[1:]):
        if n <= a1:
            return math.floor(c0 + (c1 - c0) * (n - a0) / (a1 - a0))
    (a0, c0), (a1, c1) = pts[-2], pts[-1]
    return max(16, math.floor(c1 + (c1 - c0) * (n - a1) / (a1 - a0)))


def b12_prefill(model, ids, cache, s_attn=512, s_moe=4096, keep=False, on_chunk=None, frugal=None):
    """D1i (3), TrainLab: per layer, the attention part (PLE, HC read, attention, HC write-into-read) in query chunks of
    s_attn tokens, then the MoE ONCE over the whole s_moe chunk (its experts read once per layer per s_moe tokens),
    then the HC residual write per query chunk. Same ops on the same shapes as a chunk-s_attn prefill except the MoE
    block, whose per-token outputs do not depend on the batch (grouped / sorted gather per pair, per-token combine),
    so it should equal B12_PREFILL_CHUNK=s_attn bit for bit while reading experts like chunk s_moe, when s_moe is a
    multiple of s_attn (asserted: otherwise later chunks shift the query grid; Fit12 10-26)."""
    import mlx.core as mx
    import mtplx.models.qwen4_exp as Q
    from contextlib import nullcontext
    assert not STASH and not PENDH, "decoupled prefill: no pending late writes (B8 writes them from decode rows only)"
    tm = model.language_model.model
    P = ids.shape[1]
    scope = Q._rope_table_scope() if Q.qwen4_opdiet_enabled("rope") else nullcontext()
    st = lambda c: [a for a in (c.state if hasattr(c, "state") else []) if isinstance(a, mx.array)]
    SPLIT = os.environ.get("D1I_SPLIT", "1") == "1"
    # NewIdeas B5c (3): FRUGAL PREFILL of short prompts (opt-in, non-exact): B12's decode routing on prefill rows, ranks
    # 0..W-1 always, then the first experts already resident in the decode pool (MSB-3 part) among ranks W..19, up to 10;
    # weights = the router's probabilities renormalised over those used. Emulation for the quality check: the experts are
    # computed at 4 bits from records3 (decode runs ranks >= 2 at 3 bits from the pool) and residents are re-read, so the
    # speed here is not the speed of the real path. W = frugal (or B12_FRUGAL_PF), only for prompts <= B12_FRUGAL_PF_MAX.
    FRU = int(os.environ.get("B12_FRUGAL_PF", "0") or 0) if frugal is None else int(frugal)
    if FRU and P > int(os.environ.get("B12_FRUGAL_PF_MAX", "1024")):
        FRU = 0
    if FRU:
        from bigmoe_v3f.gated import ext as _ext
        import numpy as _np

    def frugal_pick(gates, L, k):
        KC_ = 20
        top = mx.argpartition(-gates, kth=KC_ - 1, axis=-1)[..., :KC_]
        p20 = mx.take_along_axis(gates, top, axis=-1)
        o = mx.argsort(-p20, axis=-1)                                # rank order 0..19
        top, p20 = mx.take_along_axis(top, o, axis=-1), mx.take_along_axis(p20, o, axis=-1)
        u = _np.unique(_np.array(top))
        tab = _np.zeros(gates.shape[-1], bool)
        tab[u[_np.array(_ext.lookup(L, u.astype(_np.int32).tolist())) >= 0]] = True
        res = mx.array(tab)[top]                                     # MSB-3 part resident in layer L's decode pool
        R_ = mx.arange(KC_)
        forced = R_ < FRU
        pool = res & ~forced
        sel = (pool & (mx.cumsum(pool.astype(mx.int32), axis=-1) <= k - FRU)) | forced
        pos = mx.argsort(mx.where(sel, R_, R_ + 1000), axis=-1)[..., :k]
        used = mx.take_along_axis(sel, pos, axis=-1)
        inds = mx.take_along_axis(top, pos, axis=-1)
        inds = mx.where(used, inds, inds[..., :1])                  # padding: rank 0's expert at weight 0
        w = mx.where(used, mx.take_along_axis(p20, pos, axis=-1), 0.0)
        if os.environ.get("B12_FRUGAL_PF_NORM", "mass") == "mass":   # the exact top-10 weights of the experts used (divided
            w = w / p20[..., :k].sum(axis=-1, keepdims=True)         # by the true top-10 mass; dropped experts add 0)
        else:                                                        # B12's decode rule (w2f): renormalised over the used
            w = w / w.sum(axis=-1, keepdims=True)
        _T.PF["fr_rows"] = _T.PF.get("fr_rows", 0) + int(gates.shape[-2])
        _T.PF["fr_used"] = _T.PF.get("fr_used", 0) + int(used.sum().item())
        _T.PF["fr_union"] = _T.PF.get("fr_union", 0) + len(_np.unique(_np.array(inds)))
        return inds, w

    def moe_split(blk, xs_, L=None):
        """mlx_lm Qwen3NextSparseMoeBlock.__call__ (the prefill path of the tier layers: bt_b12's block -> v3's tier
        block -> MTPLX SparseMoeBlock -> super) with router, shared expert and combine per query chunk (the
        matmul shapes of a chunk-s_attn prefill) and the routed experts ONCE over all chunks (grouped, per-pair)."""
        k, parts = blk.top_k, []
        for x in xs_:
            gates = mx.softmax(blk.gate(x), axis=-1, precise=True)
            if FRU:
                parts.append(frugal_pick(gates, L, k))
                continue
            inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            scores = mx.take_along_axis(gates, inds, axis=-1)
            if blk.norm_topk_prob:
                scores = scores / scores.sum(axis=-1, keepdims=True)
            parts.append((inds, scores))
        y = blk.switch_mlp(mx.concatenate(xs_, axis=1) if len(xs_) > 1 else xs_[0],
                           mx.concatenate([p_[0] for p_ in parts], axis=1) if len(xs_) > 1 else parts[0][0])
        outs, off = [], 0
        for x, (inds, scores) in zip(xs_, parts):
            n_ = x.shape[1]
            yi = (y[:, off:off + n_] * scores[..., None]).sum(axis=-2)
            shared_y = blk.shared_expert(x)
            shared_y = mx.sigmoid(blk.shared_expert_gate(x)) * shared_y
            outs.append(yi + shared_y); off += n_
        return outs

    from bigmoe_v3f import tier as _T
    SM = os.environ.get("D1I_STREAM", "0") == "3"   # TrainLab v3 streaming (ring of D1I_RING groups, D1I_RB2=1; exact; +0.9 GB at
    kept = []                                        # ring 6): opt-in, off at 32K (TrainLab: 13.7 vs 12.5 GB)
    mtpf = _fs("PFR") and on_chunk is not None           # B12_FOLD: an MTP feed takes PFR too (Fit12's PFM): above 4096
    want = fold_cm(P) if mtpf and _fs("CM") else None    # tokens as on the AR path, shorter only to resize the pools (the
    resize = want is not None and want != CFG.get("cm")  # re-create costs ~0.2 s: EvalSuite's +7 % on short prefills)
    pfr = os.environ.get("B12_PFR", "1") == "1" and (P > int(os.environ.get("B12_PFR_MIN", _PFR_MIN0)) or (resize and P > 8))
    if pfr:                                          # AR path, records3 branches only (Fit12's audit + its exact runs)
        from mtplx.attention_context import current_attention_phase as _cap
        why = ("an MTP feed" if on_chunk is not None and not mtpf else "kept streams" if keep else "frugal prefill" if FRU
               else "the L2-10 lane (lane + PFR not A/B'd)" if _cap() == "prefill" else "the D1I_STREAM ring" if SM
               else "tier.LA" if _T.LA is not None else "TIER_WARM / PF_RESCOUNT" if os.environ.get("BIGMOE_TIER_WARM")
               or os.environ.get("BIGMOE_TIER_PF_RESCOUNT") else "no load_b12 streamer" if _ST is None else None)
        if why is None:
            from bigmoe_v3f.gated import ext as _x
            why = None if hasattr(_x, "reset_layer") else f"engine {ENGINE} has no reset_layer (B12_ENGINE=v3l)"
        if why:
            pfr = False
            if CFG.get("pfr_skip") != why:
                CFG["pfr_skip"] = why
                print(f"b12: B12_PFR skipped: {why}", flush=True)
    cm = None
    if pfr:
        if os.environ.get("B12_PFR_GRID", _PFR_GRID0) == "balanced":   # keep (default since R-41): the caller's MoE grid;
            s_moe = pfr_chunk(P, s_attn)                             # balanced moves bits when that grid's last chunk is small
        PFR_LOG.append((P, s_moe))
        cm = want if mtpf else None
    assert s_moe % s_attn == 0 or s_moe >= P, (
        f"b12_prefill: s_moe {s_moe} is off the {s_attn}-token query grid, so every later MoE chunk would shift its query "
        f"chunks and change the bits (Fit12's PFR A/B); use a multiple of {s_attn}")
    with scope, (_PFR(cm) if pfr else nullcontext()):
        for m0 in range(0, P, s_moe):
            m1 = min(m0 + s_moe, P)
            stream = None
            if SM and m1 - m0 >= int(os.environ.get("D1I_STREAM_MIN", "1024")):   # crossover < 1.2K (05:08)
                stream = _T.start_stream3([l.mlp.switch_mlp._bm for l in tm.layers], R=int(os.environ.get("D1I_RING", "4")),
                                          threads=int(os.environ.get("D1I_THREADS", "16")))
            xs = [ids[:, a:min(a + s_attn, m1)] for a in range(m0, m1, s_attn)]
            hs = [mx.tile(tm.embed_tokens(x), (1, 1, tm.args.hc_count)) for x in xs]
            masks = [Q.create_ssm_mask(h, cache[tm.ssm_idx]) for h in hs]
            for L, (layer, c) in enumerate(zip(tm.layers, cache)):
                PHASE["p"] = (m0, L, "attn")
                mixed, hyper, inject = [], [], []
                for i, x in enumerate(xs):
                    if L == tm._ple_stage_idx and Q._COMPILED_VERIFY_PLE.get() is None:
                        layer.ple.ple_embedding.stage(x, c, layer.ple.NGRAM_IDX)
                    h = hs[i]
                    if "ple" in layer:
                        h = h + layer.ple(h, x, c)
                    mx_, hy, inj = layer.attn_hyper_connection(h)
                    bo = layer.linear_attn(mx_, masks[i], c) if layer.is_linear else layer.self_attn(mx_, c)
                    mx_, hy, inj = layer.mlp_hyper_connection(hy, pending=(bo, inj))
                    if stream is not None:                 # the stream's threads read while the GPU runs attention
                        mx.async_eval(mx_, hy, inj, *st(c))
                    else:
                        mx.eval(mx_, hy, inj, *st(c))
                    mixed.append(mx_); hyper.append(hy); inject.append(inj)
                PHASE["p"] = (m0, L, "moe")
                if SPLIT or FRU:                           # dense MoE parts per query chunk: exact vs chunk s_attn
                    bos = moe_split(layer.mlp, mixed, L)
                else:
                    bo = layer.mlp(mx.concatenate(mixed, axis=1) if len(mixed) > 1 else mixed[0])
                    bos, off = [], 0
                    for x in xs:
                        bos.append(bo[:, off:off + x.shape[1]]); off += x.shape[1]
                PHASE["p"] = (m0, L, "post")
                for i in range(len(xs)):
                    hs[i] = Q._hyper_residual_write(hyper[i], bos[i], inject[i])
                mx.eval(*hs)
                del mixed, hyper, inject, bos
            if stream is not None:
                for k_, v_ in _T.stop_stream().items():
                    _T.PF["s_" + k_] = round(_T.PF.get("s_" + k_, 0) + v_, 2)
            tm._last_widened = hs[-1]
            if keep:                                               # (span, final widened stream) per query chunk
                kept += [((a, a + x.shape[1]), h) for a, x, h in zip(range(m0, m1, s_attn), xs, hs)]
            if on_chunk is not None:                               # the same pairs, handed over per MoE chunk
                on_chunk([((a, a + x.shape[1]), h) for a, x, h in zip(range(m0, m1, s_attn), xs, hs)])
            del hs
            mx.clear_cache()
    return kept


def _frs_nocycle():
    """B12_FRS_NOCYCLE (Fit12 R-45), once per process: wrap mtplx.frspec_draft._full_vocab_head so the local class it
    builds no longer holds the pruned head and its ids in the closure cells only its __init__ reads (the methods use
    self.head / self._ids). The class stays in its super() cycle (a few KB until a gen-2 GC); a replaced head's arrays
    now go when the head does. Memory only: the head computes the same thing."""
    if os.environ.get("B12_FRS_NOCYCLE", "1") == "0":
        return
    from mtplx import frspec_draft as fd
    if getattr(fd._full_vocab_head, "_b12", False):
        return
    f0 = fd._full_vocab_head

    def _full_vocab_head(head, ids, vocab_rows):
        full = f0(head, ids, vocab_rows)
        for c in type(full).__init__.__closure__ or ():
            if c.cell_contents is head or c.cell_contents is ids:
                c.cell_contents = None
        return full
    _full_vocab_head._b12 = True
    fd._full_vocab_head = _full_vocab_head


def _fold_install(spec125):
    """B12_FOLD, once per process: spec125.restore_state MOVES a state b12_prefill_state marked (Fit12's MOV: the
    decode takes the prefill's arrays, no deep copy; a second restore of that state finds no cache), and
    spec125.generate runs with ALLWAIT on and closes the state's QSA row target when it returns. Unmarked states and
    every other spec125 path behave as before, apart from ALLWAIT inside generate."""
    if getattr(spec125.generate, "_b12_fold", False):
        return
    rs0, gen0 = spec125.restore_state, spec125.generate

    def restore_state(model, state, with_head=True):
        if not state.get("_b12_move"):
            return rs0(model, state, with_head=with_head)
        state["_b12_move"] = False
        cache, head = state.pop("cache"), None
        if with_head:
            head = spec125._MTPHead(model)
            head.cache = state.pop("mtp_cache")
            head.logits, head.h = state["logits"], state["h"]
        return cache, head

    def generate(*a, **kw):
        pre = kw.get("prefilled")
        aw0 = AW[0]
        if _fs("AW"):
            AW[0] = True
        try:
            return gen0(*a, **kw)
        finally:
            AW[0] = aw0
            tcm = pre.pop("_b12_qsa", None) if isinstance(pre, dict) else None
            if tcm is not None:
                tcm.__exit__(None, None, None)
    generate._b12_fold = True
    spec125.restore_state, spec125.generate = restore_state, generate


def b12_prefill_state(model, prompt_ids, s_attn=512, s_moe=4096, max_new=None):
    """spec125-compatible prefilled state (spec125.prefill_state's keys: cache, mtp_cache, logits, h) from the
    decoupled trunk prefill; the MTP head's history is appended per query chunk as spec125._prefill does per
    prefill chunk (draft logits for the last row only). Pass as spec125.generate(..., prefilled=state).
    BIGMOE_FRSPEC=1: the FR-Spec draft head is rebuilt with this prompt's ids first, as spec125.generate does.
    B12_FOLD=1: the prefill takes PFR (and C_m per prompt), its QSA buffers are sized once to n + max_new + 16 rows
    (mtplx qsa_rows_target, open until spec125.generate returns; max_new defaults to B12_FOLD_ROWS = 2048), and the
    state is marked to be MOVED into the decode, so it feeds exactly ONE spec125.generate."""
    import mlx.core as mx
    if os.path.expanduser("~/SpecDecode") not in sys.path:
        sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
    import spec125
    if os.environ.get("BIGMOE_FRSPEC") == "1":       # Fit12 10-27 pos 2: spec125.generate rebuilds the FR-Spec draft head
        if os.path.expanduser("~/Scout") not in sys.path:   # with the prompt's ids only when it starts, so without this
            sys.path.append(os.path.expanduser("~/Scout"))  # the prefill's draft (state['logits']) used the previous
        from patches.frspec4 import FRSpec                  # request's subset (draft-only; tokens cannot change)
        _frs_nocycle()
        FRSpec(model, extra_ids=[int(t) for t in prompt_ids]).on()
    tcm = None
    if FOLD[0]:
        _fold_install(spec125)
    if _fs("ROWS"):
        from mtplx.models import qwen4_exp as Q
        tcm = Q.qsa_rows_target(len(prompt_ids) - 1 + int(max_new or os.environ.get("B12_FOLD_ROWS", "2048")) + 16)
        tcm.__enter__()
    try:
        state = _prefill_state(model, prompt_ids, s_attn, s_moe, spec125)
    except BaseException:
        if tcm is not None:
            tcm.__exit__(None, None, None)
        raise
    if _fs("MOV"):
        state["_b12_move"] = True
    if tcm is not None:
        state["_b12_qsa"] = tcm
    return state


def _prefill_state(model, prompt_ids, s_attn, s_moe, spec125):
    import mlx.core as mx
    cache = model.make_cache()
    ids = mx.array([list(prompt_ids)[:-1]])
    head = spec125._MTPHead(model)

    def feed(pairs):                                   # the head's history, in token order, one query chunk per call
        for (a, b), h in pairs:
            head.append(h, [int(v) for v in prompt_ids[a + 1:b + 1]])
            mx.eval(head.logits)
    if os.environ.get("D1I_HEAD_EACH", "1") == "1":    # per MoE chunk (default; exact: same sha at 2.6K + 32K, 05:48) (32K:
        b12_prefill(model, ids, cache, s_attn, s_moe, on_chunk=feed)   # the trunk's kept streams were the MTP peak)
    else:
        kept = b12_prefill(model, ids, cache, s_attn, s_moe, keep=True)
        feed(kept)
        del kept
    mx.clear_cache()
    return {"cache": cache, "mtp_cache": head.cache, "logits": head.logits, "h": head.h}


def b12_generate(model, prompt_ids, max_tokens, prefill="decoupled", s_attn=512, s_moe=4096, chunk=512, stop_ids=None,
                 frugal=None, frugal_from=None):
    """Greedy eager decode as this CLI's generate() runs it, for other harnesses' paired checks (e.g. decoupled vs
    chunked prefill on long prompts): the prompt minus its last token is prefilled either decoupled (prefill=
    "decoupled": attention in s_attn query chunks, the MoE once per s_moe tokens) or in cache-only chunks of `chunk`
    tokens (prefill="chunked"); then the last prompt token and each new one go through B12's one-row frugal decode.
    Returns the new token ids, through the first id in stop_ids. frugal=W (decoupled only): frugal prefill (B12's decode
    routing, ranks 0..W-1 always, the rest only if resident) for prompts <= B12_FRUGAL_PF_MAX; with frugal_from=i, the
    first i tokens are prefilled exactly and only prompt[i:] (e.g. a turn's new tool output) frugally."""
    import mlx.core as mx
    from bigmoe_v3f.fuse import reset_generation_state
    assert prefill in ("decoupled", "chunked"), prefill
    reset_generation_state(model); STASH.clear(); PENDH.clear()
    cache = model.make_cache()
    ids = mx.array([[int(t) for t in prompt_ids][:-1]])
    if prefill == "decoupled":
        a_ = min(int(frugal_from or 0), ids.shape[1]) if frugal else 0
        if a_ > 0:                                              # the context exactly, then the new tokens frugally
            b12_prefill(model, ids[:, :a_], cache, s_attn, s_moe, frugal=0)
        if ids.shape[1] > a_:
            b12_prefill(model, ids[:, a_:], cache, s_attn, s_moe, frugal=frugal)
    else:
        for i in range(0, ids.shape[1], chunk):
            model(ids[:, i:i + chunk], cache=cache, emit_logits=False)
            mx.eval([c.state for c in cache if hasattr(c, "state")])
    from bigmoe_v3f import repguard as _RG                             # the main track's repetition guard, timed as its
    guard = _RG.make()                                                # pipelined greedy loop: output t steered iff armed
    out, y = [], int(prompt_ids[-1])                                  # after observing outputs 0..t-2 (disarmed = unchanged)
    while len(out) < max_tokens:
        lg = model(mx.array([[y]]), cache=cache)
        lg = (lg[0] if isinstance(lg, tuple) else lg)[0, -1]
        y = int(mx.argmax(_RG.steer(guard, out, lg) if guard is not None and guard.armed else lg).item())
        if out:
            _RG.observe(guard, out)
        out.append(y)
        if stop_ids and y in stop_ids:
            break
    return out


if __name__ == "__main__":
    for _k, _v in (("BIGMOE_COMPILE_HC", "1"), ("BIGMOE_HC_VREAD", "1"), ("BIGMOE_ASYNC_EVERY", "1")):
        os.environ.setdefault(_k, _v)                                 # speed flags for the CLI only (the suite refuses them)
    # D1i default since 05:41 Thu (Chat1000 REPORT 8: decoupled vs chunk 512 on 14 long tasks = 13/14 identical, 11/13 both,
    # 3.0x faster prefill; chunk 2048 vs 512 differ more): attention chunks of 512, the MoE once per D1I_MOE (4096) tokens.
    # D1I_ATTN=0 = the plain chunked prefill at B12_PREFILL_CHUNK.
    if os.environ.setdefault("D1I_ATTN", "512") in ("", "0"):
        os.environ.pop("D1I_ATTN")
    frac, TASK = float(sys.argv[1]), sys.argv[2]
    SPEC = TASK.startswith("spec")                                    # spec3:3:0 = MTP k:max_draft:lookup (spec125)
    MTPGSM = TASK.startswith("gsm") and bool(os.environ.get("B12_GSM_MTP"))   # GSM8K through spec125 (B12_GSM_MTP=3:3:0)
    if SPEC or MTPGSM:
        for _k, _v in (("BIGMOE_MTP_STREAM", "1"), ("BIGMOE_MTP_C", "64"), ("BIGMOE_MAXK", "80"), ("BIGMOE_FRSPEC", "1"),
                       ("MTPLX_FAMILY_CAPTURE_COMMIT", "1"), ("MTPLX_DROP_EVENTS", "1"), ("MTPLX_LAZY_MTP_HISTORY_APPEND", "1"),
                       ("MTPLX_DEFER_REPAIR_EVAL", "1")):
            os.environ.setdefault(_k, _v)
    rt, st = load_b12(frac, mtp=SPEC or MTPGSM)
    import mlx.core as mx
    from bigmoe_v3f.gated import ext
    model, tok = rt.model, rt.tokenizer
    EOS = set()
    for _a in ("eos_token_ids", "eos_token_id"):
        _v = getattr(tok, _a, None)
        if _v is not None:
            EOS |= set(_v) if isinstance(_v, (list, set, tuple)) else {_v}
    for _t in ("<|im_end|>", "<|endoftext|>"):
        try:
            _i = tok.convert_tokens_to_ids(_t)
            if isinstance(_i, int) and _i >= 0: EOS.add(_i)
        except Exception:
            pass

    def chat_ids(text):
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
        return list(ids if isinstance(ids[0], int) else ids["input_ids"])

    def prefill_decoupled(ids, cache, s_attn, s_moe, keep=False):        # module-level b12_prefill on this model
        return b12_prefill(model, ids, cache, s_attn, s_moe, keep)

    def d1i_prefill_state(prompt_ids, s_attn, s_moe):
        tp = time.perf_counter()
        state = b12_prefill_state(model, prompt_ids, s_attn, s_moe)
        PREF.update(n=len(prompt_ids) - 1, t=time.perf_counter() - tp)
        return state

    def generate(prompt_ids, n, stop_eos=False, per_tok=None):
        from bigmoe_v3f.fuse import reset_generation_state
        from bigmoe_v3f import tier as _T
        reset_generation_state(model)
        STASH.clear(); PENDH.clear()
        out, cache = [], model.make_cache()
        ids = mx.array([list(prompt_ids)[:-1]])
        PFC = int(os.environ.get("B12_PREFILL_CHUNK", "2048"))      # smaller chunks: smaller exact-prefill scratch
        NOLOG = os.environ.get("D1I_NOLOGITS", "1") == "1"          # D1i: prefill chunks skip the vocab head (unused; default on)
        tp = time.perf_counter()
        S_ATTN = int(os.environ.get("D1I_ATTN", "0") or 0)            # D1i (3): decoupled attention / MoE chunks
        for i in (range(0, ids.shape[1], PFC) if not S_ATTN else []):
            _T.PF["last"] = i + PFC >= ids.shape[1]                  # D1i warming: the last chunk's experts -> pools
            if NOLOG:
                model(ids[:, i:i + PFC], cache=cache, emit_logits=False)
                mx.eval([c.state for c in cache if hasattr(c, "state")])
            else:
                lg = model(ids[:, i:i + PFC], cache=cache)
                mx.eval(lg[0] if isinstance(lg, tuple) else lg)
        if S_ATTN:
            prefill_decoupled(ids, cache, S_ATTN, int(os.environ.get("D1I_MOE", "4096")))
        _T.PF["last"] = False
        PREF.update(n=ids.shape[1], t=time.perf_counter() - tp)
        y = int(prompt_ids[-1])
        ext.reset_stats(); ext.defer_reset()
        m0, t0, dk0 = ext.stats()["misses"], time.perf_counter(), diskread()
        from bigmoe_v3f import repguard as _RG                         # repetition guard as b12_generate (exact disarmed)
        guard = _RG.make()
        while len(out) < n:
            lg = model(mx.array([[y]]), cache=cache)
            lg = (lg[0] if isinstance(lg, tuple) else lg)[0, -1]
            y = int(mx.argmax(_RG.steer(guard, out, lg) if guard is not None and guard.armed else lg).item())
            if out:
                _RG.observe(guard, out)
            if stop_eos and y in EOS:
                break
            out.append(y)
            if per_tok is not None:
                per_tok.append(ext.defer_stats()[1])                 # cumulative waited misses since the decode began
        t = time.perf_counter() - t0
        st.gate.check()
        rows, waited, late, over = ext.defer_stats()
        return out, dict(t=t, n=len(out), reads=ext.stats()["misses"] - m0, waited=waited, late=late, over=over, rows=rows,
                         disk=diskread() - dk0)

    sha = lambda t: hashlib.sha1(str(list(t)).encode()).hexdigest()[:10]
    import resource, ctypes

    class _RU(ctypes.Structure):                                      # rusage_info_v4: uuid, then 64-bit counters
        _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"u{i}", ctypes.c_uint64) for i in range(40)]

    def footprint_peak():
        """lifetime max physical footprint (ri_lifetime_max_phys_footprint; Activity Monitor's Memory), GB"""
        ru = _RU()
        if ctypes.CDLL(None).proc_pid_rusage(os.getpid(), 4, ctypes.byref(ru)) != 0:
            return float("nan")
        return ru.u28 / 1e9

    def footprint_now():
        ru = _RU()
        return ru.u7 / 1e9 if ctypes.CDLL(None).proc_pid_rusage(os.getpid(), 4, ctypes.byref(ru)) == 0 else float("nan")

    def diskread():
        """bytes this process has read from disk (ri_diskio_bytesread: F_NOCACHE expert reads included)"""
        ru = _RU()
        return ru.u16 if ctypes.CDLL(None).proc_pid_rusage(os.getpid(), 4, ctypes.byref(ru)) == 0 else 0

    LOADED = mx.get_active_memory() / 1e9

    mem = lambda: (f"peak active {mx.get_peak_memory() / 1e9:.1f} GB, cache {mx.get_cache_memory() / 1e9:.1f} GB, "
                   f"max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.1f} GB, "
                   f"PEAK FOOTPRINT {footprint_peak():.1f} GB")
    PREF = {}
    # long-context prompts: fixed snapshots (04:40 Thu) of the old "scripts/*.py[:115000]" recipe, so the text no longer
    # moves with the scripts; ctx64k = the same recipe at 200,000 characters
    PROMPTS = {"ctx32k": lambda: open(os.path.expanduser("~/Breakthrough/data/prompts/ctx32k.txt")).read(),
               "ctx64k": lambda: open(os.path.expanduser("~/Breakthrough/data/prompts/ctx64k.txt")).read(),
               "long": lambda: ("Here is a Python script. Summarise what it does, section by section.\n\n```python\n"
                               + open(os.path.expanduser("~/Breakthrough/scripts/tf_frugal_var.py")).read()[:int(os.environ.get("B12_LONG_CHARS", "7600"))] + "\n```"),
               "code": lambda: open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read(),
               "prose": lambda: ("Explain, in plain prose for a curious 14-year-old, how a CPU cache works and why it makes "
                                 "programs faster. Use one everyday analogy and keep it to about 300 words.")}
    tag = (f"b12 C_m={CFG['cm']} C_l={CFG['cl']} hc{CFG['hc']} late1={int(CFG['late1'])} late3={CFG['l3']} w={CFG['w']} pre={int(CFG['pre'])} pf={int(CFG['pf'])}"
           + (f" norm={CFG['norm']}" if CFG["norm"] != "used" else "") + (" plan4" if CFG.get("plan4") else ""))
    if TASK == "check":
        out, s = generate(chat_ids(PROMPTS["code"]()), 48)
        print(f"check {tag}: sha {sha(out)} reads {s['reads']} waited {s['waited']} late {s['late']} over {s['over']} | "
              f"{tok.decode(out)[:160]!r}", flush=True)
    elif TASK == "d1i":                                               # D1i (TrainLab): one cold call after load, then a repeat
        import numpy as np
        from bigmoe_v3f import tier as _T
        P = os.environ.get("PROMPT", "long")
        ids = chat_ids(PROMPTS[P]())
        N = int(os.environ.get("BT_N", "64"))
        tag2 = (f"chunk {os.environ.get('B12_PREFILL_CHUNK', '2048')} attn/moe {os.environ.get('D1I_ATTN', '-')}/{os.environ.get('D1I_MOE', '-')} split {os.environ.get('D1I_SPLIT', '1')} "
                f"gpu {os.environ.get('BIGMOE_TIER_PF_GPU', '0')} nologits {os.environ.get('D1I_NOLOGITS', '1')} group"
                f"{os.environ.get('BIGMOE_TIER_PF_GROUP', '0')} warm {os.environ.get('BIGMOE_TIER_WARM', '0')}")
        for run in ("cold", "repeat")[:int(os.environ.get("D1I_RUNS", "2"))]:
            for k_ in [k_ for k_ in _T.PF if k_ != "last"]:          # every counter and timer (t_sync, t_moe, ...) per run
                _T.PF[k_] = 0.0 if isinstance(_T.PF[k_], float) else 0
            waits = []
            if os.environ.get("B12_FP_SAMPLE") == "1":                # where the prefill's footprint peak sits
                import threading
                samp, stop_ = [], threading.Event()

                def _sampler():
                    t00 = time.perf_counter()
                    while not stop_.is_set():
                        samp.append((time.perf_counter() - t00, footprint_now(), mx.get_active_memory() / 1e9, PHASE.get("p")))
                        stop_.wait(0.05)
                th_ = threading.Thread(target=_sampler, daemon=True); th_.start()
            out, s = generate(ids, N, per_tok=waits)
            if os.environ.get("B12_FP_SAMPLE") == "1":
                stop_.set(); th_.join()
                top = sorted(samp, key=lambda r: -r[1])[:4]
                gap = sorted(samp, key=lambda r: -(r[1] - r[2]))[:4]
                fmt = lambda r: f"t {r[0]:.1f} s fp {r[1]:.2f} act {r[2]:.2f} gap {r[1] - r[2]:.2f} at {r[3]}"
                print(f"[{P}] fp sampler {run}: {len(samp)} samples; max footprint: " + " | ".join(map(fmt, top)), flush=True)
                print(f"[{P}] fp sampler {run}: max footprint - MLX active: " + " | ".join(map(fmt, gap)), flush=True)
            w = np.diff([0] + waits)
            seg = lambda a, b: float(w[a:b].mean()) if len(w) > a else float("nan")
            pf = {k_: (round(v, 2) if isinstance(v, float) else v) for k_, v in _T.PF.items() if k_ != "last"}
            if _T.LA is not None:                                     # D1k: experts advised, time spent advising
                pf.update(la_n=_T.LA.get("n", 0), la_t=round(_T.LA.get("t", 0.0), 2), la_err=_T.LA.get("err"))
                _T.LA.update(n=0, t=0.0)
            print(f"[{P}] d1i {run} {tag2}: prefill {PREF['n']} tokens in {PREF['t']:.2f} s ({PREF['n'] / PREF['t']:.0f} tok/s); "
                  f"waited misses/token 0-16 {seg(0, 16):.1f} 16-32 {seg(16, 32):.1f} 32-64 {seg(32, 64):.1f}; decode "
                  f"{s['n'] / s['t']:.1f} tok/s, reads/token {s['reads'] / max(1, s['n']):.1f}; {mem()}; sha {sha(out)} | "
                  f"pf {pf} | {tok.decode(out)[:60]!r}", flush=True)
    elif TASK == "speed":
        P = os.environ.get("PROMPT", "code")
        ids = chat_ids(PROMPTS[P]())
        N = int(os.environ.get("BT_N", "128"))
        generate(ids, 16)
        out, s = generate(ids, N)
        nt = max(1, s["n"])
        print(f"[{P}] {tag}: {s['n'] / s['t']:.1f} tok/s, SSD reads/token {s['reads'] / nt:.1f} (waited misses "
              f"{s['waited'] / nt:.1f}, late {s['late'] / nt:.1f}, overflow {s['over'] / nt:.2f}), peak "
              f"{mx.get_peak_memory() / 1e9:.1f} GB, active {mx.get_active_memory() / 1e9:.1f} GB, {mem()}, sha {sha(out)} | "
              f"{tok.decode(out)[:70]!r}", flush=True)
        print(f"[{P}] SSD during decode: {s['disk'] / 1e9 / s['t']:.1f} GB/s, {s['disk'] / nt / 1e6:.0f} MB/token, late reads "
              f"{s['late'] / nt:.1f}/token", flush=True)
        print(f"[{P}] prefill {PREF.get('n', 0)} tokens in {PREF.get('t', 0):.2f} s ({PREF.get('n', 0) / max(PREF.get('t', 1e-9), 1e-9):.0f} tok/s, "
              f"{('decoupled ' + os.environ['D1I_ATTN'] + '/' + os.environ.get('D1I_MOE', '4096')) if os.environ.get('D1I_ATTN') else 'chunk ' + os.environ.get('B12_PREFILL_CHUNK', '2048')})", flush=True)
    elif SPEC:                                                      # MTP speed: verify passes take the multi-row path
        sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
        import spec125
        from bigmoe_v3f.fuse import reset_generation_state
        k, md_, lk = (int(v) for v in TASK[4:].split(":")[:3])
        P = os.environ.get("PROMPT", "code")
        ids = chat_ids(PROMPTS[P]())
        N = int(os.environ.get("BT_N", "128"))

        def run(n):
            reset_generation_state(model); STASH.clear(); PENDH.clear()
            out, first, s = [], {}, {}

            def cb(tk):
                if not first:
                    first.update(t=time.perf_counter(), m=ext.stats()["misses"], st=ext.stats(), d=ext.defer_stats(),
                                 act=mx.get_active_memory() / 1e9, pk=mx.get_peak_memory() / 1e9, fp=footprint_now(),
                                 dk=diskread())
                    mx.reset_peak_memory()                    # from here: the decode (draft + verify) peak alone
                out.append(int(tk))
            pre = PRESTATE.get("s")                                   # D1i: one decoupled prefill shared by the runs
            spec125.generate(model, ids, n, k_mtp=k, max_draft=md_, lookup=lk, stats=s, on_token=cb,
                             prefill_chunk=int(os.environ.get("B12_PREFILL_CHUNK", "2048")), prefilled=pre)
            st.gate.check()
            e1, d1 = ext.stats(), ext.defer_stats()
            cnt = {k: e1[k] - first["st"][k] for k in ("pf_issued", "pf_used", "pf_ontime", "pf_late")}
            cnt["waited"] = d1[1] - first["d"][1]
            cnt["late"] = d1[2] - first["d"][2]
            cnt["disk"] = diskread() - first["dk"]
            s["cnt"] = cnt
            s["memck"] = (f"memory: loaded {LOADED:.1f} GB active; at the first token active {first['act']:.1f}, prefill "
                          f"peak {first['pk']:.1f}, footprint {first['fp']:.1f}; decode-phase peak active "
                          f"{mx.get_peak_memory() / 1e9:.1f} GB, footprint now {footprint_now():.1f}")
            return out, s, time.perf_counter() - first["t"], ext.stats()["misses"] - first["m"]
        PRESTATE = {}
        if os.environ.get("D1I_ATTN") and os.environ.get("D1I_FRESH") == "1":
            # the server's footprint: each generate runs the decoupled prefill itself (spec125's _prefill swapped for
            # b12_prefill_state), so no shared state is held and deep-copied beside the live cache
            S_A, S_M = int(os.environ["D1I_ATTN"]), int(os.environ.get("D1I_MOE", "4096"))

            def _fresh_prefill(model_, prompt_ids, with_head, prefill_chunk=2048):
                tp = time.perf_counter()
                s_ = b12_prefill_state(model_, prompt_ids, S_A, S_M)
                PREF.update(n=len(prompt_ids) - 1, t=time.perf_counter() - tp)
                head = spec125._MTPHead(model_) if with_head else None
                if head is not None:
                    head.cache, head.logits, head.h = s_["mtp_cache"], s_["logits"], s_["h"]
                return s_["cache"], head
            spec125._prefill = _fresh_prefill
        elif os.environ.get("D1I_ATTN"):
            PRESTATE["s"] = d1i_prefill_state(ids, int(os.environ["D1I_ATTN"]), int(os.environ.get("D1I_MOE", "4096")))
            print(f"[{P}] d1i prefilled state: {PREF['n']} tokens in {PREF['t']:.2f} s ({PREF['n'] / PREF['t']:.0f} tok/s), "
                  f"{mem()}", flush=True)
        if os.environ.get("B12_SPEC_WARM", "1") == "1":
            run(16)                                                   # warm-up (B12_SPEC_WARM=0: skip, e.g. at 32K+)
        out, s, tsec, rd = run(N)
        if os.environ.get("D1I_FRESH") == "1" and PREF:
            print(f"[{P}] d1i fresh prefill (no state copy): {PREF['n']} tokens in {PREF['t']:.2f} s "
                  f"({PREF['n'] / PREF['t']:.0f} tok/s)", flush=True)
        nt = max(1, len(out) - 1)                                     # decode from the first token on
        c, npass = s["cnt"], max(1, s.get("passes", 1))
        print(f"[{P}] {tag} spec {TASK[4:]}: {nt / tsec:.1f} tok/s, {s.get('tok_per_pass', float('nan')):.2f} tok/pass, "
              f"SSD reads/token {rd / nt:.1f}; per pass: waited misses {c['waited'] / npass:.1f}, reads {rd / npass:.1f}, "
              f"prefetch issued {c['pf_issued'] / npass:.1f} used {c['pf_used'] / npass:.1f} (on time {c['pf_ontime'] / npass:.1f}, "
              f"late {c['pf_late'] / npass:.1f}); {mem()}, sha {sha(out)} | "
              f"{tok.decode(out)[:70]!r}", flush=True)
        print(f"[{P}] {s['memck']}", flush=True)
        print(f"[{P}] SSD during decode: {c['disk'] / 1e9 / tsec:.1f} GB/s, {c['disk'] / npass / 1e6:.0f} MB/pass, late reads "
              f"{c['late'] / npass:.1f}/pass, waited {c['waited'] / npass:.1f}/pass", flush=True)
    elif TASK.startswith("gsm"):
        N = int(TASK[3:]); OFF = int(os.environ.get("OFFSET", "0"))
        G8 = [json.loads(l) for l in open(os.path.expanduser("~/MoEPrefetch/data/gsm8k/test.jsonl"))][OFF:OFF + N]
        INSTR = "Solve the problem step by step, briefly. End with a line of the form 'Answer: <number>'.\n\n"
        okn, same = 0, 0
        ref = json.load(open(os.path.expanduser("~/MoEPrefetch/data/w125_gsm_eager.json")))["answers"]
        nt = tt = 0
        if MTPGSM:                                                    # the MTP decode the release would ship (spec125)
            sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
            import spec125
            from bigmoe_v3f.fuse import reset_generation_state
            kk, mdd, lkk = (int(v) for v in os.environ["B12_GSM_MTP"].split(":")[:3])

            def spec_gen(prompt_ids, n):
                reset_generation_state(model); STASH.clear(); PENDH.clear()
                pre = (b12_prefill_state(model, prompt_ids, int(os.environ["D1I_ATTN"]), int(os.environ.get("D1I_MOE", "4096")))
                       if os.environ.get("D1I_ATTN") else None)
                o, s_ = [], {}
                t_ = time.perf_counter()
                spec125.generate(model, prompt_ids, n, k_mtp=kk, max_draft=mdd, lookup=lkk, stats=s_,
                                 on_token=lambda tk: o.append(int(tk)), prefilled=pre, stop_ids=EOS,
                                 prefill_chunk=int(os.environ.get("B12_PREFILL_CHUNK", "2048")))
                st.gate.check()
                e = next((i for i, x in enumerate(o) if x in EOS), None)
                o = o[:e] if e is not None else o
                return o, dict(t=time.perf_counter() - t_ - s_.get("t_prefill", 0.0), n=len(o))
        for j, p in enumerate(G8):
            q = chat_ids(INSTR + p["question"])
            out, s = spec_gen(q, 600) if MTPGSM else generate(q, 600, stop_eos=True)
            text = tok.decode(out)
            m = re.findall(r"Answer:\s*\**\s*\$?\s*(-?[\d,]*\.?\d+)", text) or re.findall(r"(-?[\d,]*\.?\d+)", text)
            gold = p["answer"].split("####")[-1].strip().replace(",", "")
            a = m[-1].replace(",", "").rstrip(".") if m else None
            try:
                okn += int(a is not None and abs(float(a) - float(gold)) < 1e-6)
            except ValueError:
                pass
            same += int(OFF + j < len(ref) and ref[OFF + j] == a)
            nt += s["n"]; tt += s["t"]
        print(f"125B {tag}{' MTP ' + os.environ['B12_GSM_MTP'] if MTPGSM else ''}: GSM8K {okn}/{len(G8)} (offset {OFF}), answers equal to exact eager {same}, "
              f"{nt / max(tt, 1e-9):.1f} tok/s decode", flush=True)
        from bigmoe_v3f import tier as _T
        if _T.PF.get("rc_calls"):                                     # BIGMOE_TIER_PF_RESCOUNT=1: union already resident
            u_ = max(1, _T.PF["rc_union"])
            print(f"125B residency over {_T.PF['rc_calls']} prefill MoE calls: union {_T.PF['rc_union']} experts, MSB-3 part "
                  f"resident {_T.PF['rc_base']} ({100 * _T.PF['rc_base'] / u_:.1f} %), both parts {_T.PF['rc_both']} "
                  f"({100 * _T.PF['rc_both'] / u_:.1f} %)", flush=True)
        if _T.PF.get("fr_rows"):                                      # B12_FRUGAL_PF: experts per prefill row, union per call
            print(f"125B frugal prefill W={os.environ.get('B12_FRUGAL_PF')}: {_T.PF['fr_used'] / _T.PF['fr_rows']:.2f} experts "
                  f"used per row (of 10), union {_T.PF['fr_union']} over {_T.PF['fr_rows']} rows", flush=True)
