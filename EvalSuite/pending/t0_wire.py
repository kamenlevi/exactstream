"""T0 wiring (docs/PREREG_t0_2026-11-03.md): the 12 GB variants b12llff4t0c (theta 0.8) and b12llff4t0h (theta 0.95) =
Fit12's top-2 two-pass decode (~/Fit12/scripts/t0_twopass.py via t0_hook) on the pinned live bt_b12 (8db71cc537), MTP
off, Fit12's gate env exactly. Edits evalsuite/model.py, cli.py and decode.py; run only once runner 38 has ended (its
pin hashes evalsuite/*.py). --dry: apply in memory, check that every edit lands once and the files parse; write nothing.
  python3 pending/t0_wire.py [--dry]"""
import ast, os, sys

ROOT = os.path.expanduser("~/EvalSuite")
T0_ENV = {"MTPLX_PREFILL_CHUNK_SIZE": "512", "BIGMOE_TIER_PF_GROUP": "64", "BIGMOE_RESET_GEN": "0", "EVALSUITE_SHIP": "1",
          "B12_LATE3": "2,10", "B12_NORM": "mass", "BIGMOE_PF_GROUP": "64", "BIGMOE_ASYNC_EVERY": "1", "B12_TRIMS": "1",
          "B12_PFR": "1", "B12_ALLWAIT": "0", "B12_CACHE_MB": "64", "BIGMOE_LOOP_GUARD": "0", "MTPLX_NGRAM_PREWARM": "off",
          "MTPLX_COMPILED_VERIFY": "0", "MTPLX_QWEN4EXP_COMPILE": "0", "MTPLX_COMPILE_AR_FORWARD": "0",
          "MTPLX_COMPILED_TARGET_PREFIX": "0", "BIGMOE_SPARE": "0", "BIGMOE_PF_NOPUBLISH": "1"}   # Fit12 21:43 (fit_t0.py)
T0_POP = ("B12_ENGINE", "B12_FOLD", "B12_PFR_GRID", "B12_CM", "B12_CL", "B12_FOLD_CM", "B12_FOLD_AW", "B12_W", "B12_PMIN")

MODEL = [
    ('''       "b12llff4": 0.10, "b12llff4p": 0.10}       # PMIN12: fold-fast on bt_b12 8db71cc537 (v3p), without / with B12_PMIN 0.5''',
     '''       "b12llff4": 0.10, "b12llff4p": 0.10,       # PMIN12: fold-fast on bt_b12 8db71cc537 (v3p), without / with B12_PMIN 0.5
       "b12llff4t0c": 0.10, "b12llff4t0h": 0.10}  # T0 (PREREG_t0): Fit12's top-2 two-pass at theta 0.8 / 0.95, MTP off'''),
    ('''              "b12llff4": "8db71cc537", "b12llff4p": "8db71cc537"}   # PMIN12: Breakthrough 3676db3 (v3p default + B12_PMIN)''',
     '''              "b12llff4": "8db71cc537", "b12llff4p": "8db71cc537",   # PMIN12: Breakthrough 3676db3 (v3p default + B12_PMIN)
              "b12llff4t0c": "8db71cc537", "b12llff4t0h": "8db71cc537"}   # T0: Fit12's gate file (= the live sha)'''),
    ('''           "b12llfx2": True, "b12llff2": True, "b12llff3": True, "b12llff4": True, "b12llff4p": True}
FRS_NOCYCLE''',
     '''           "b12llfx2": True, "b12llff2": True, "b12llff3": True, "b12llff4": True, "b12llff4p": True,
           "b12llff4t0c": False, "b12llff4t0h": False}   # T0 has no drafter
FRS_NOCYCLE'''),
    ('''R53A = {"b12llff4": "0", "b12llff4p": "0"}''',
     '''R53A = {"b12llff4": "0", "b12llff4p": "0"}
T0 = {"b12llff4t0c": 0.8, "b12llff4t0h": 0.95}   # T0's theta per arm (PREREG_t0: Breakthrough's code / chat settings)
T0_ENV = %r   # Fit12's T0 gate env (21:43), asserted at load
T0_POP = %r   # unset: the live file's fast defaults''' % (T0_ENV, T0_POP)),
    ('''B12_RE = re.compile(r"(b12llff4p|b12llff4|''', '''B12_RE = re.compile(r"(b12llff4t0c|b12llff4t0h|b12llff4p|b12llff4|'''),
    ('''        return dict(model="b12", sorted=False, plan=None, frac=B12[base], mtp=B12_MTP[base], guard=bool(m.group(2)),
                    base=base, name=name)''',
     '''        return dict(model="b12", sorted=False, plan=None, frac=B12[base], mtp=B12_MTP[base], guard=bool(m.group(2)),
                    base=base, name=name, t0=T0.get(base))'''),
    ('''        if v["base"] in FOLD_ENGINE:              # the fold verdict: both arms in bt_b12's shipped engine + trims''',
     '''        if v.get("t0") is not None:               # T0 (PREREG_t0): Fit12's gate env exactly, on the pinned live file
            bad = {k: os.environ.get(k) for k, x in T0_ENV.items() if os.environ.get(k) != x}
            bad.update({k: os.environ[k] for k in T0_POP if k in os.environ})
            assert not bad, f"{v['name']}: env differs from Fit12's T0 gate: {bad}"
            import hashlib
            sha = hashlib.sha256(open(os.path.expanduser("~/Breakthrough/scripts/bt_b12.py"), "rb").read()).hexdigest()[:10]
            assert sha == BT_B12_SHA[v["base"]], f"{v['name']}: bt_b12.py is {sha}; T0 is gated on {BT_B12_SHA[v['base']]}"
        elif v["base"] in FOLD_ENGINE:            # the fold verdict: both arms in bt_b12's shipped engine + trims'''),
    ('''        else:
            rt, st = load_b12(v["frac"], mtp=v["mtp"])''',
     '''        else:
            if v.get("t0") is not None:           # T0: Fit12's t0_twopass patches bt_b12's decode block before load_b12
                sys.path.insert(1, os.path.expanduser("~/Fit12/scripts"))   # (t0_hook.load, keeping the streamer)
                import bt_b12, t0_twopass
                t0_twopass.apply(bt_b12)
            rt, st = load_b12(v["frac"], mtp=v["mtp"])'''),
]
CLI = [
    ('''"b12llfx2": 14, "b12llff2": 14, "b12llff3": 14, "b12llff4": 14, "b12llff4p": 14}''',
     '''"b12llfx2": 14, "b12llff2": 14, "b12llff3": 14, "b12llff4": 14, "b12llff4p": 14,
          "b12llff4t0c": 14, "b12llff4t0h": 14}'''),
    ('''B12_ENV["b12llff4p"] = dict(B12_ENV["b12llff4"], B12_PMIN="0.5")''',
     '''B12_ENV["b12llff4p"] = dict(B12_ENV["b12llff4"], B12_PMIN="0.5")
B12_ENV["b12llff4t0c"] = B12_ENV["b12llff4t0h"] = %r   # T0: Fit12's gate env (model.T0_ENV) less EVALSUITE_SHIP (set beside)
T0_POP_BASES = ("b12llff4t0c", "b12llff4t0h")   # their env drops T0_POP (model.T0_POP): the live file's defaults
T0_POP = %r'''
     % ({k: x for k, x in T0_ENV.items() if k != "EVALSUITE_SHIP"}, T0_POP)),
    ('''            env.update(SHIP_AR, EVALSUITE_SHIP="1", **B12_ENV[base], **(SHIP_MTP if B12_MTP[base] else {}))
            if not guard:''',
     '''            env.update(SHIP_AR, EVALSUITE_SHIP="1", **B12_ENV[base], **(SHIP_MTP if B12_MTP[base] else {}))
            if base in T0_POP_BASES:              # T0 (PREREG_t0): Fit12's gate leaves these to the live file's defaults
                for k in T0_POP:
                    env.pop(k, None)
            if not guard:'''),
    ('''           "b12llfx2": True, "b12llff2": True, "b12llff3": True, "b12llff4": True, "b12llff4p": True}
B12_RE''',
     '''           "b12llfx2": True, "b12llff2": True, "b12llff3": True, "b12llff4": True, "b12llff4p": True,
           "b12llff4t0c": False, "b12llff4t0h": False}
B12_RE'''),
    ('''B12_RE = re.compile(r"(b12llff4p|b12llff4|''', '''B12_RE = re.compile(r"(b12llff4t0c|b12llff4t0h|b12llff4p|b12llff4|'''),
]
DECODE = [
    ('''def fast_decode_gate(rt, v, out, sampler, log):''',
     '''T0 = {}                                           # T0 (PREREG_t0): the arm's theta, and the last row's counts


def _gen_t0(rt, ids, cap, sampler, seed):
    """T0: Fit12's top-2 two-pass greedy decode at the arm's theta (t0_hook.gen); its counts go to the row"""
    import t0_hook
    toks, fin = t0_hook.gen(rt, ids, cap, sampler, seed, T0["theta"])
    T0["stats"] = dict(t0_hook.STATS)
    return toks, fin


def t0_gate(rt, sampler, log):
    """T0's VOID check: theta 1.01 on humaneval/HumanEval/156, 96 tokens, gives Fit12's gate logits-chain sha"""
    import t0_hook
    it = next(x for x in D.load("dec_humaneval") if x["key"] == "humaneval/HumanEval/156")
    t0_hook.gen(rt, it["prompt_ids"], 96, sampler, 0, 1.01)
    got = t0_hook.STATS.get("lsha")
    log(f"T0 gate: theta 1.01 on HumanEval/156 x96: lsha {got} (Fit12's G1: 37ccd09ba2)")
    if got != "37ccd09ba2":
        log("T0: VOID: the gate does not reproduce Fit12's exact AR decode")
        raise SystemExit(5)


def fast_decode_gate(rt, v, out, sampler, log):'''),
    ('''    if not thinking:
        fast_decode_gate(rt, v, out, sampler, log)''',
     '''    if not thinking:
        fast_decode_gate(rt, v, out, sampler, log)
    if v.get("t0") is not None:
        T0["theta"] = v["t0"]
        t0_gate(rt, sampler, log)'''),
    ('''                gen = _gen_fast24 if v.get("fast24") else _gen_mtp if v.get("mtp") else _gen_ref2 if v.get("ref2") else _gen''',
     '''                gen = (_gen_t0 if v.get("t0") is not None else _gen_fast24 if v.get("fast24") else _gen_mtp if v.get("mtp")
                       else _gen_ref2 if v.get("ref2") else _gen)'''),
    ('''                if SSD:                                  # MTP rows: decode-phase SSD expert reads (misses), waited, late''',
     '''                if T0.get("stats"):                      # T0 rows: steps, pass-2 rows, rate, logits-chain sha, times
                    row["t0"] = T0.pop("stats")
                if SSD:                                  # MTP rows: decode-phase SSD expert reads (misses), waited, late'''),
]


def apply(path, edits, dry):
    p = os.path.join(ROOT, path)
    s = open(p).read()
    if "b12llff4t0c" in s:
        print(f"{path}: T0 already in")
        return
    for a, b in edits:
        assert s.count(a) == 1, (path, a[:70], s.count(a))
        s = s.replace(a, b)
    ast.parse(s)
    if not dry:
        open(p, "w").write(s)
    print(f"{path}: {len(edits)} edits {'checked (dry)' if dry else 'written'}")


if __name__ == "__main__":
    dry = "--dry" in sys.argv
    if not dry and os.system("pgrep -f tools/day_runner38.sh >/dev/null") == 0:
        sys.exit("runner 38 is alive: not wiring")
    apply("evalsuite/model.py", MODEL, dry)
    apply("evalsuite/cli.py", CLI, dry)
    apply("evalsuite/decode.py", DECODE, dry)
