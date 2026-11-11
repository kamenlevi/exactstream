"""R-57 PREFETCH16 (NewIdeas Fri 2026-10-30 00:55, from Leap's S45; with R-55): the exact live REF2 loop at C 192 misses
413-779 experts per 16-row check (111-194 ms of SSD reads per check, serial with the ~185 ms warm compute). With R-55's
wide gate the 16-row check takes the sync-free gate, so BigMoE's look-ahead predictor can ride on it: at layer L the GPU
applies layer L+1's router to layer L's MoE input (qwen4_exp: as is), ranks the experts by their best row log-prob
(BIGMOE_PF_RANKED, one unique list for the 16 rows) and posts the top-B with L's ids; the loader reserves and reads the
non-resident ones (PF_JIT, PF_QD 4, up to PF_VICTIM in flight per layer) while L computes. Reads only: bits unchanged.
EvalSuite's REF env refuses BIGMOE_PREFETCH, so the predictors are attached here after the load (stream.py's own
wiring, ratio None), and switched per arm.
Harness = Leap's S45 (~/Leap/scripts/r40s3k_screen.py speed mode): EvalSuite's REF harness, C 192, PF1 for prefills only,
REF2 + SB, DFlash2 L1-20 drafting live (greedy, keep = accepted + 1), up to CAP tokens per run.
Arms (one process, BIGMOE_MAXK=160, BIGMOE_PRED_LINES=4): slot = the shipped exact path (16-row calls on _sync_call, Leap's
slot call); gate = R-55 (stream.GATE_N 160); gate+pf = R-55 + the look-ahead prefetch (B = PF_B ranked ids for L+1).
HumanEval/68 slot gate gate+pf gate+pf gate slot, prealgebra/1640 the reverse.
Per run: tok/s, median cycle, check + commit, misses and demand SSD ms (the loader's io_s) per check, tokens vs
EvalSuite's REF2 row and vs the other arms; since C45 (NewIdeas 11-01 11:5x) also the gate service of calls with misses
(slow_s: post seen -> release, the GPU's spin bound; OOO overlaps part of it) and the drafter's ms (the SSD idles); since
NewIdeas 11-01 15:3x (M-17 LAYSZ45) the whole loop's SSD MB (ri_diskio_bytesread) and misses per kept token.
Lines (NewIdeas 00:55): R-57 PASS = demand SSD ms per check -50 % or better AND check + commit -25 % or better (gate+pf vs
gate, medians over the prompts), every token == REF2; KILL = the cycle under -10 %. R-55 in the live loop (gate vs slot):
information.
Usage: [BigMoE venv] BIGMOE_MAXK=160 BIGMOE_PRED_LINES=4 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PF_VICTIM=24
       BIGMOE_EXT_BUILD=build_next r57_screen.py [C=192] [CAP=256] [PF_B=96] -> data/r57/r57_screen.txt"""
import ctypes, json, os, sys, threading, time
os.environ.pop("MTPLX_FUSED_GATE_UP", None)            # the streamed engine has no fused gate | up
os.environ.pop("BIGMOE_TIER", None)
os.environ.pop("BIGMOE_VERIFY_MIDLOOP", None)          # SB comes from the lambda below, as in S45
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# THE COMBINED EXACT DEFAULT (WED_TESTS item 5, Wed 4 Nov 08:3x; R57_D45=0 = the 11-02 default): the chat switch is part of
# R57_PRESET=45 now, so it is set before CHATSW is read; the rest of the stack is set in the preset block below
_D45 = os.environ.get("R57_PRESET") == "45" and os.environ.get("R57_D45", "1") == "1"
if _D45:
    os.environ.setdefault("R57_CHATSW", "1")
> (a quote from the project owner, kept private)
if CHATSW:                                               # ~/TrainLab/INTEGRATION_CHATSWITCH.md): prompts the type rule calls chat
    os.environ["TL_INFER"] = "1"                         # draft on the 125B's MTP head (d22 b5, K15, cut t, early exit); code /
    os.environ.pop("BIGMOE_MTP_STREAM", None)            # math / agent stay on DFlash2. Head resident (D-20 / D-23d); TL_INFER:
                                                         # mtp125 keeps MTPLX's fused HC read (EvalSuite's env pins the rest)
if os.environ.get("R57_PRESET") == "45":                 # THE 45 GB DEFAULT (NewIdeas 11-01 06:1x): REF2-T tiers 4433333333 / 3c
    os.environ.setdefault("R57_TIER", "1")               # + the R-55 gate (MAXK default 160) + L1-14 truncation t 0.3 + ref2's
    os.environ.setdefault("R57_ARMS", "gate+t")          # defaults (REF2b key buckets, GDNX); REF2b 4-bit exact = R57_PRESET unset
    os.environ.setdefault("R57_TRUNC", "0.3")
    os.environ.setdefault("R57_COMBINE", "1")            # + OOO + TIER_COMBINE (NewIdeas 11-01 11:4x on EvalSuite's R-56d PASS,
    os.environ.setdefault("BIGMOE_TIER_COMBINE", "1")    # exact); the no-OOO control: R57_COMBINE=0 BIGMOE_OOO=0
    os.environ.setdefault("BIGMOE_OOO", "1")
    # NewIdeas 11-02 01:3x on C45 (all exact): ADOPT MoonShots' M-17 per-layer pool sizes (L: -17.2 % SSD MB per kept token,
    # +8.3 % steady at C 224) from the md5-pinned data/seed45/laysz45_<C>.json (uniform pools at other C; R57_LAYSZ=0 = the
    # uniform control) and ADOPT Leap's K-73 exact-safe stack (+2.4 %; off with R57_COMBINE=0, which it needs).
    _LS = {"224": "3098a50550e27a1ebb2075e3caf86960", "256": "ee76e2a7390c18f8d53b539a3ba9700a",
           "300": "cd0df64a980d217d75a53a10d600ddce", "320": "f0a3997c88379d01cdc2b39175c39307"}
    _C = sys.argv[1] if len(sys.argv) > 1 else "256"     # the 45 GB default's C since 11-02 (41.7 GB lifetime, C50)
    if os.environ.get("R57_LAYSZ", "1") == "1" and "BIGMOE_TIER_SIZES" not in os.environ and _C in _LS:
        import hashlib
        _p = os.path.expanduser(f"~/BigMoE/data/seed45/laysz45_{_C}.json")
        assert hashlib.md5(open(_p, "rb").read()).hexdigest() == _LS[_C], f"{_p}: md5 differs from the pinned one"
        os.environ["BIGMOE_TIER_SIZES"] = _p
    if os.environ.get("R57_COMBINE") == "1":
        os.environ.setdefault("K73_EXACT", "1")
    # Wed 4 Nov 08:3x, WED_TESTS item 5 (data/d45/d45_report.txt; A-B-C-C-B-A at C 160, all 6 arms 14 / 14 == REF2-T): ADOPT
    # the night's exact levers, steady +12.1 % geo over 14 prompts (chat +17.3, code +12.4, math +12.8, agent +0.5; SSD MB per
    # kept token -30 %; lifetime footprint +1.8 GB): the chat switch (above), the SSD keepalive (50 us), the volatile tier
    # 64/32 (7.8 GB purgeable, outside phys_footprint; needs an ext with set_vtier: ext/build since item 0) and K-83; and ADD
    # lever 1's full stack on top, +6.4 % (code +11.7): JOINT, the copy split, the coherent spin, the system-scope post (these
    # need OOO: R57_COMBINE=1). ~+19 % together. Each stays off with its own =0 (R57_CHATSW, BIGMOE_SSD_KEEPALIVE_US,
    # K83_EXACT, BIGMOE_JOINT, BIGMOE_COPY_PAR, BIGMOE_SPIN_COHERENT, BIGMOE_POST_SYS) or R57_VTIER= (empty); R57_D45=0 = all.
    if _D45:
        os.environ.setdefault("BIGMOE_SSD_KEEPALIVE_US", "50")
        os.environ.setdefault("R57_VTIER", "64/32")
        if os.environ.get("R57_COMBINE") == "1":
            for _k in ("K83_EXACT", "BIGMOE_JOINT", "BIGMOE_COPY_PAR", "BIGMOE_SPIN_COHERENT", "BIGMOE_POST_SYS"):
                os.environ.setdefault(_k, "1")
if os.environ.get("BIGMOE_COPY_PAR") == "0":             # the ext turns the copy split on for ANY value: =0 means off here
    os.environ.pop("BIGMOE_COPY_PAR")
TIER = os.environ.get("R57_TIER") == "1"                 # S45-T (NewIdeas 18:2x): REF2-T on the gate at C's RAM (M-08 sizing)
if TIER:
    os.environ["BIGMOE_TIER"] = "4433333333"; os.environ["BIGMOE_TIER_GLU"] = "3c"
    os.environ.setdefault("BIGMOE_TIER_CL", "112")       # C_m follows from C's 4-bit bytes (stream.py's budget split)
    if os.environ.get("R57_COMBINE") != "1":             # R-69 (Leap): TIER_COMBINE (+ BIGMOE_OOO) is bitwise == this path;
        os.environ.pop("BIGMOE_TIER_COMBINE", None)      # R57_COMBINE=1 keeps it for the OOO arm
_PF_ARM = bool({"gate+pf", "gate+t+pf"} & set(os.environ.get("R57_ARMS", "slot,gate,gate+pf").split(",")))   # look-ahead arms
os.environ.setdefault("BIGMOE_MAXK", "160")             # the default since R-55's adoption (gated.py); explicit for the asserts
assert os.environ.get("BIGMOE_MAXK") == "160", "see Usage"
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s4/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
sys.path.insert(1, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s4/Leap_third_party"))
import evalsuite.env                                     # EvalSuite's forced / refused env (read-only use), before mlx
if os.environ.get("R57_MODEL_DIR"):                      # a scratch model dir (scratch/model_mix: records3 with converted layers,
    evalsuite.env.REF_MODEL = os.path.expanduser(os.environ["R57_MODEL_DIR"])   # the rest symlinks; PLAN_records3_convert.md)
if os.environ.get("R57_SPARE"):                          # R-57b: reserved prefetch slots per layer (EvalSuite forces 0); the
    os.environ["BIGMOE_SPARE"] = os.environ["R57_SPARE"]  # ext reads it when bigmoe loads (EM.load), so set it here
from evalsuite import model as EM, data as ED
from evalsuite.decode import footprint_now_gb, footprint_peak_gb
import numpy as np
import mlx.core as mx
import dflash_model_mlx as DF

OUT = os.path.expanduser("~/BigMoE/data/r57"); os.makedirs(OUT, exist_ok=True)
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
C = int(sys.argv[1]) if len(sys.argv) > 1 else (256 if os.environ.get("R57_PRESET") == "45" else 192)
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 256
PF_B = int(sys.argv[3]) if len(sys.argv) > 3 else 96
R, K = int(os.environ.get("R57_ROWS", "16")), 16     # R57_ROWS=32 (THE COMBINATION: the copy pays most at 32 rows): a
# 32-row block (31 drafts) on the prompts the copy runs on, with MATCH 2.5's rule (32 rows only when >= 24 drafted positions carry a used copy,
# else the first 15). Over 16 rows the check leaves the sync-free gate (MAXK 160) for the slot path and EvalSuite's block
# (prefill's): outside REF2-T's certified 1-16-row verify, so the token check decides whether it is exact; slow either way
assert R in (16, 32) or (R == 64 and os.environ.get("R57_COPYRUN") == "64"), R   # 64: R57_COPYRUN only (inert_span.py)
EOS = {248044, 248046}
TAPS = (5, 15, 25, 35, 45)
PICK = (("humaneval", "humaneval/HumanEval/68"), ("math", "math500/test/prealgebra/1640.json"),
        ("humaneval", "humaneval/HumanEval/91"), ("math", "math500/test/algebra/694.json"),        # S45's other prompts
        ("math", "math500/test/algebra/1282.json"), ("math", "gsm8k/708"))
# R-55L (the orchestrator 17:0x: headline the steady-state cycle; a clean slot re-run settles code): R57_ARMS=slot,gate
# R57_PROMPTS=6 R57_TAG=r55l runs only those arms on S45's six prompts and writes data/r57/r55l_screen.txt
ARMS_ = tuple(os.environ.get("R57_ARMS", "slot,gate,gate+pf").split(","))
PICK = PICK[:int(os.environ.get("R57_PROMPTS", "2"))]
SETOF = {"humaneval": "humaneval", "math500": "math", "gsm8k": "math", "ifeval": "ifeval", "multi": "multi", "agent": "agent",
         "agentlong": "agentlong", "mbpp": "mbpp", "humanevalt": "humanevalt", "humanevalx": "humanevalx", "mathx": "mathx"}   # Fri 08:2x-21:5x: EvalSuite dec_agentlong / mbpp / humanevalt / humanevalx / mathx (FAST there needs R57_QOM_SETS)
if os.environ.get("R57_KEYS"):                           # an explicit prompt list ("set/key,..." or '+'-joined inside c45_runner's
    PICK = tuple((SETOF[k.split("/")[0]], k) for k in os.environ["R57_KEYS"].replace("+", ",").split(","))   # arm lists)
TAG = os.environ.get("R57_TAG", "r57")
REPS = int(os.environ.get("R57_REPS", "2"))
TRUNC = float(os.environ.get("R57_TRUNC", "0.3"))
# R57_COPY (THE COMBINATION, the orchestrator 11-02 21:1x; FableGod copygate/GATE.md, MTP Head's walk 9d9ae1e: G2 kept per
# check HumanEval+ x1.041 / x1.211 at 16 / 32 rows, MBPP x1.000, agent x1.032): the gated session copy over DFlash2's block on
# the host (scripts/copygate.py, vendored verbatim): the longest k-gram 8-16 of the committed window replaces a draft when the
# gate allows it; the check still decides every token (exact). g2 = COPYGATE, copies count as sure in the t cut (the walk
# had no cut; the copies are ~99 % right on code); g2t = COPYGATE, the cut on the drafter's own cf; g1 / g0 / all = the
# walk's other arms. Off on the type rule's math (GATE.md 3: a harm domain). The prompt's k-grams build in a thread during
# the prefill (~14 ms for 9K tokens); per check ~0.15 ms on the host, logged.
COPY = os.environ.get("R57_COPY", "")
COPYMODE = {"g2": "copygate", "g2t": "copygate", "g1": "pduel", "g0": "prompt", "all": "all"}
assert COPY in ("", *COPYMODE), COPY
if COPY:
    import importlib.util as _ilu
    _sp = _ilu.spec_from_file_location("copygate", os.path.expanduser("~/BigMoE/scripts/copygate.py"))
    CGM = _ilu.module_from_spec(_sp); _sp.loader.exec_module(CGM)
    CGM.nl_ids()                                         # the vocabulary's newline ids, once (~70 ms), outside every clock
# R57_D7B (Leap 11-02 21:2x, ~/Leap/scripts/d7b_live.py; R-85b arm E): the drafter's 5 layers in 4-bit (RTN, group 64) at
# load, fc / selector / head bf16: propose 6.30 vs 10.39 ms on code, kept code +0.00 %, math -3.39 %. code = a second,
# 4-bit drafter for the type rule's code prompts (the rest keep the bf16 one); all = every DFlash2 prompt. Drafts change,
# the target's tokens don't (every token == REF2-T). With K-83 its drafter forms stay on the bf16 drafter only.
D7B = os.environ.get("R57_D7B", "")
assert D7B in ("", "code", "all"), D7B
# R57_CHATROWS=8 (the orchestrator 11-02 21:2x, cloud task 14 ~/FableGod/CLOUD_ANSWER_FASTMODE.md: chat's kept per check is
# flat in block length, 2.50 / 2.47 at 16 / 8 rows, while every row costs): the type rule's chat prompts check at most 8
# rows (MTP chain K 7 with the chat switch; DFlash2's block cut to 7 drafts without it). Exact: block size only.
CHATROWS = int(os.environ.get("R57_CHATROWS", "0"))
# R57_CHATTRUNC=t (NewIdeas 11-02 23:31: Fable B's B5 per-row cut and MoonShots' M-21b both point at t ~0.4 on chat): the
# type rule's chat prompts cut their drafts at running confidence t (the MTP chain's early exit, or DFlash2's L1-14 cut);
# the rest keep R57_TRUNC. Exact (the check decides): speed only.
CHATTRUNC = float(os.environ.get("R57_CHATTRUNC", "0"))
# R57_FAST=<arm> (Kamen 11-02 21:31, two first-class tracks: EXACT + FAST; THE COMBINATION's FAST preset): EvalSuite's RELAX
# rules, vendored verbatim (scripts/relax_core.py from ~/EvalSuite/pending; PREREG_relax_2026-11-02), per prompt set as its
# arm has them: the cf gate (a row whose draft the drafter is >= thr sure of keeps its top-2 experts, the dropped ranks routed
# to its rank 0, so they are never read), the copy gate (a row whose last 8 input tokens occur in the prompt: top-2 too),
# the relaxed rules R1 / R2 and the comment loophole (R2 inside code comments). Quality-defined: tokens differ from REF2-T by
# design (EvalSuite's same-scores test judges them; run without R57_REFTOK). Here the gate rides on the t 0.3 cut (the
# rows present); EvalSuite's test checks all 16. The gate needs DFlash2's cf (no gate on MTP-drafted chat: rules only).
# HC8 (arms B / C there) is not wired. 16-row blocks only.
FAST = os.environ.get("R57_FAST", "")
RLX_D = float(os.environ.get("R57_RELAX_DELTA", "0") or 0)   # R57_RELAX_DELTA=<nats> (MoonShots' relax_delta.py): relaxed
RLX_THR = RLX_COPY = None                                    # acceptance at any delta (EvalSuite's R1); R57_RELAX_GATE
if RLX_D > 0:                                                # (cf:<thr> / copy / both) picks the rows that may relax;
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # R57_RELAX_LOG logs each relaxed accept. 0 / unset =
    import relax_delta as RLD                                # exact: none of the RLX blocks below runs
    RLX_THR, RLX_COPY = RLD.parse_gate(os.environ.get("R57_RELAX_GATE", ""))
    if not FAST:                                             # the gate's relax_core (R57_FAST loads it below otherwise)
        import importlib.util as _ilu3
        _sp3 = _ilu3.spec_from_file_location("relax_core", os.path.expanduser("~/BigMoE/scripts/relax_core.py"))
        RXC = _ilu3.module_from_spec(_sp3); _sp3.loader.exec_module(RXC)
RLXV2 = os.environ.get("R57_RELAX_V2", "1") != "0"     # R57_RELAX_V2 (MoonShots' apply_v2.py, default on): one readback for the
                                                     # relaxed rules' dlm / rank, cached token texts, the tagger only on blocks
                                                     # that can touch a comment, rule C + R1 together; 0 = v1 exactly
SPAN = os.environ.get("R57_RELAX_SPAN", "0") not in ("", "0")   # R57_RELAX_SPAN=1 (MoonShots' relax_span.py): FULL's rule C,
if SPAN:                                                 # relaxed acceptance on code-comment rows only (the causal tagger,
    assert not RLX_D > 0 or os.environ.get("R57_RELAX_V2", "1") != "0", "v1: R57_RELAX_SPAN and R57_RELAX_DELTA are separate arms"
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # unset / 0 = exact: no SPAN block below runs
    import relax_span as RSP
    SPAN_RULE = RSP.parse_rule(os.environ.get("R57_RELAX_SPAN_RULE", ""))
    if not FAST:                                         # the tagger's relax_core (R57_FAST loads it below otherwise)
        import importlib.util as _ilu4
        _sp4 = _ilu4.spec_from_file_location("relax_core", os.path.expanduser("~/BigMoE/scripts/relax_core.py"))
        RXC = _ilu4.module_from_spec(_sp4); _sp4.loader.exec_module(RXC)
INERT, CRUN = None, int(os.environ.get("R57_COPYRUN", "0") or 0)   # R57_INERT=<arm> / R57_COPYRUN=64 (GOAL's inert_span.py,
if os.environ.get("R57_INERT", "0") not in ("", "0") or CRUN:     # ~/FableGod/goal/LOSSY350_GOAL.md). INERT: a drafted row
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # inside a '#' comment or a docstring of a python fence
    import inert_span as INM                             # is kept whatever the 125B's argmax (span ends exact, code tokens
    INERT = INM.parse_arm(os.environ.get("R57_INERT", ""), os.environ.get("R57_INERT_DOCTEST", "1"))   # exact). Unset / 0 =
    if INERT is not None:                                # exact: no INERT block below runs
        assert not SPAN and not RLX_D > 0, "R57_INERT, R57_RELAX_SPAN and R57_RELAX_DELTA are separate arms"
        assert R >= INERT["rows"], f"R57_INERT={INERT['name']} needs R57_ROWS={INERT['rows']}"
        if not FAST:                                     # the tagger's relax_core (R57_FAST loads it below otherwise)
            import importlib.util as _ilu5
            _sp5 = _ilu5.spec_from_file_location("relax_core", os.path.expanduser("~/BigMoE/scripts/relax_core.py"))
            RXC = _ilu5.module_from_spec(_sp5); _sp5.loader.exec_module(RXC)
    assert CRUN in (0, 64) and (not CRUN or (COPY and R == 64)), "R57_COPYRUN=64 needs R57_COPY and R57_ROWS=64"
if os.environ.get("R57_GDNB") == "1":                    # GDNB (Leap 70cd998): its NSG must be set before gdnbody is imported
    os.environ.setdefault("GDNB_NSG", "16")
D17 = os.environ.get("R57_D17") == "1"                  # DRAFT17 (MTP Head ~/TrainLab/sticky/draft17.py 7cc44f8, read-only): the
if D17:                                                  # drafter proposes a 16th draft for QOM's last-row gate; needs Leap's
    sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s4/TrainLab_sticky"))   # draft_fused FORM_ROWS 16..64 (05fafe7). Module level: the
    import draft17 as D17M                               # drafting code reads D17 on every run (FAST or not)
if FAST:
    import importlib.util as _ilu2
    _sp2 = _ilu2.spec_from_file_location("relax_core", os.path.expanduser("~/BigMoE/scripts/relax_core.py"))
    RXC = _ilu2.module_from_spec(_sp2); _sp2.loader.exec_module(RXC)
    _S = lambda thr=None, copy=False, rules=(), loop=None, q1u=False, t3=False, cold=False, qom=False: dict(thr=thr, copy=copy,
                                                                    rules=tuple(rules), loop=loop, q1u=q1u, t3=t3, cold=cold, qom=qom)
    _G99 = dict(thr=0.99, copy=True)
    FASTARMS = {                                         # ~/EvalSuite/pending/relax.py ARMS (PLAN_relax sec 9), sets only
        "A": dict(humaneval=_S(**_G99), math=_S(**_G99), agent=_S(**_G99)),
        "B": dict(humaneval=_S(**_G99, loop=RXC.R2), math=_S(**_G99), agent=_S(**_G99)),
        "E": dict(ifeval=_S(rules=[RXC.R1]), multi=_S(rules=[RXC.R1])),
        "F": dict(ifeval=_S(rules=[RXC.R1, RXC.R2]), multi=_S(rules=[RXC.R1, RXC.R2])),
        "C": dict(humaneval=_S(**_G99, rules=[RXC.R1], loop=RXC.R2), math=_S(**_G99, rules=[RXC.R1])),
        "G": dict(ifeval=_S(0.999, True, [RXC.R1]), multi=_S(0.999, True, [RXC.R1])),
    }
    # U3 / U3A (the orchestrator 11-03 11:01; the map's q1u_cf09_t3 PASS on code / IFEval / multi, Breakthrough SCOREBOARD
    # Tue 10:1x): rows the drafter is >= 0.9 sure of keep ranks 0-1 at 3 bits (MSB only: no LSB read) and ranks 2-9 only
    # where the expert is already in the check's union of unsure rows (no read), the rest dropped; unsure rows exact
    _U3T = float(os.environ.get("R57_U3_THR", "0.9"))  # R57_U3_THR=2 + R57_U3_FORCE=1 = the null control: no sure rows,
    U3FORCE = os.environ.get("R57_U3_FORCE") == "1"       # yet the copied OOO path runs: tokens must equal REF2-T's
    FASTARMS["U3"] = dict(humaneval=_S(thr=_U3T, q1u=True, t3=True))
    FASTARMS["U3A"] = dict(humaneval=_S(thr=_U3T, q1u=True, t3=True), math=_S(thr=_U3T, q1u=True, t3=True),
                           agent=_S(thr=_U3T, q1u=True, t3=True))
    # U3C (Breakthrough's P1, ~/Breakthrough/WED_PLAN.md b800a4b; ~/FableGod/decision_map/dm_cold_masks.py hyb_cold4_pmtc_t3,
    # R57_U3C_LEN=1: hyb_cold4_len_t3): U3 with the per-request COLD table. A row's base set: unsure = ranks 0-3 + the ranks
    # whose expert is warm for this request (its prompt routed to it at any rank, OR pop_cap45 >= 0.05; LEN: the footprint
    # alone when the prompt has >= 128 tokens); sure = ranks 0-1 (3 bits, as U3). Every row keeps its base set + any rank
    # whose expert is in the union of ALL rows' base sets. No domain switch: every prompt set. FAST (task results decide).
    FASTARMS["U3C"] = {s_: _S(thr=_U3T, q1u=True, t3=True, cold=True) for s_ in ("humaneval", "math", "ifeval", "multi", "agent")}
    # QOM (Wed 11-04 build day; Breakthrough's keep-set rule ~/Breakthrough/keepset 6f1ca44, the map's C: qom_0_live_bq_c147_t3):
    # U3 at thr 0 (every drafted row sure, the bonus row exact) + a sure row's rank 2-9 expert outside the check's loads is
    # KEPT WHEN RESIDENT in the pool: posted with bit 21 (optional), which the loader never reads on a miss (ext/build_qom: slot
    # -2), then dropped after the spin (weight 0, no renormalisation). Tail 3: every kept expert of a sure row at MSB-3 (t3);
    # R57_QOM_TAIL=4 (LSB reads for kept ranks 2-9, the map's tested C) is not built. Not with REF2_OOO_1C (-2 base slots).
    _QT = float(os.environ.get("R57_QOM_THR", "0.0"))
    assert os.environ.get("R57_QOM_TAIL", "3") == "3", "R57_QOM_TAIL=4 (LSB reads for kept ranks 2-9) is not built yet"
    FASTARMS["QOM"] = {s_: _S(thr=_QT, q1u=True, t3=True, qom=True) for s_ in   # R57_QOM_SETS (Fri 08:1x, EvalSuite's agent
                       os.environ.get("R57_QOM_SETS", "humaneval,ifeval,multi").split(",")}   # check): agent sets run exact unless listed
    if FAST == "QOM":
        sys.path.insert(0, "/Users/kamen/BigMoE/data/r57/snap/frozen_s4/Breakthrough_keepset")
        import qom_keep as QK                            # read-only: Breakthrough's module
    FASTARMS["BE"] = dict(FASTARMS["B"], **FASTARMS["E"])   # the code mode + chat R1: one process for every domain
    FASTARMS["CE"] = dict(FASTARMS["C"], **FASTARMS["E"])
    assert FAST in FASTARMS and R == 16, (FAST, R)
# R57_PREROUTE=1 (the orchestrator 11-02 22:2x; Fable A's ~/FableGod/NEWIDEAS_SCREEN.md 1, cloud's pick #1 "pre-route on the
# partial sum", exact): a capture. At every MoE layer L of a check, OOO's combine runs again with the scores of ranks >= 3
# (and >= 6) zeroed = L's partial output (+ the shared expert); L+1's MLP-side HC read (the read its router sees, without
# its attention write) and router on the residual written with it -> top-10 ("p3", "p6"); also on L+1's own pre-attention
# stream ("full": L's full output, PLE included, only L+1's attention write missing; read at L+1's start) and R-57's
# forecast ("r57": L+1's router on L's MoE input); each vs L+1's true top-10. "fullA" = L+1's ATTENTION-side read of L's
# full output: the pr_160 night's "full" / p3 / p6 read that (MTP Head's catch Wed 05:3x: qwen4_exp has one hc_norm + read
# mix per side, the router sees the MLP side), kept as the control. p3 / p6 on pair 0 -> 1 miss layer 1's PLE term (PLE
# writes the cache: it runs once, at layer 1's start). Side computations only: tokens stay REF2-T's. Per run: recall@10 per
# row and of the union (the reads) per variant -> {TAG}_preroute.json. Needs K-73's block and OOO (R57_PRESET=45).
# R57_PR_DUMP=1 (MTP Head's SAMELAYER-T data): per row and layer pair, "full"'s input X (L+1's MLP-side read of its
# pre-attention stream) and L+1's true MoE input Y, float32 x 2560 -> data/r57/pr_dump/{TAG}_{run:02d}.npz (key, typ, L =
# L+1's layer, chk, row, X, Y; ~140 MB per run).
PREROUTE = os.environ.get("R57_PREROUTE") == "1"
PRDUMP = PREROUTE and os.environ.get("R57_PR_DUMP") == "1"
PR = dict(on=False, cur=None, y={}, blk={}, cap={}, dump=None)   # this check's partial combines / block outputs / pre-routes
PRS = {}                                                 # this run's samples: variant -> [layer, rows, row recall, union recall]
PRD = dict(n=0, L=[], chk=[], row=[], X=[], Y=[])        # R57_PR_DUMP: this run's rows
# A6 (DAY9 condition 7, start-up; Leap's K-80 read-out 11-02: the first cycle after a ~9K agent prefill pays ~130 ms = a miss
# burst when the pools hold another prompt's experts + a one-time non-SSD cost in the first check's graph build + the
# drafter's window prefill). R57_A6 = comma flags, one process per arm, every token still checked:
#   rep    the first check twice, rolled back in between (CS.rollback_after_verify): copy 1 - copy 2 = the one-time cost
#   keep   no mx.clear_cache() at the end of a run (MLX's buffer cache stays warm across requests: Leap's candidate a)
#   dummy  one 16-row check of [tok] x 16 right after the prefill, rolled back (pre-pays first-use costs: candidates a-c)
#   dwin   the drafter's window prefill right after the prefill (all but the last position), outside the decode clock
#   pools  after the prefill, ensure each layer's A6_POOLS (24) most frequent experts of the prompt's last 64 rows (MSB base
#          pools; the miss burst's upper-bound fix; the read time is logged, outside the decode clock)
#   warm   Fable's PREFILL_WARM (mission 20, ~/FableGod/CACHE.md 5): the whole prompt's routing (every prefill chunk) ->
#          each layer's experts into its base pool in frequency order, the most used last (= most recent; ensure() pushes
#          each id to the LRU front in order), up to the pool size C_m; the 4-bit ranks' ones into the LSB sidecar up to C_l
A6 = set(filter(None, os.environ.get("R57_A6", "").replace("+", ",").split(",")))   # '+' too: c45_runner's arm lists split on ','
A6_POOLS = int(os.environ.get("R57_A6_POOLS", "24"))
# WARMSPLIT (NewIdeas 11-02 07:3x, Monday night; lead ~/.local/share/pacer/leads/2026-11-01_per_row_cost_opus.md step 1):
# R57_WARMSPLIT=N: every Nth check from the 3rd drafts all 15 (a 16-row check) and, before the real one, runs the identical
# input cold at 16 rows (rolled back), then warm (all its experts now resident) at 1 / 4 / 8 / 16 rows, each rolled back.
# Warm slope = the GPU-only cost per row; cold 16 - warm 16 = the SSD's exposed share. The real check then commits as
# always (tokens unchanged); sampled checks carry ws=1 in the per-check log. -> {TAG}_warmsplit.json
WARMSPLIT = int(os.environ.get("R57_WARMSPLIT", "0"))
# R57_STALL=1 (measurement 2 of cloud task 12, the orchestrator 11-02 21:0x; with BIGMOE_TRACE=1, which the ext reads at
# load): per run the decode's ext events (every read: layer, expert, tag, issue -> bytes in place; every loader service:
# routing seen -> release, 4 = had misses; every ensure's coverage) + each check's UPTIME_RAW boundaries (the ext's clock:
# cycle start, check start, graph built, eval done, commit submitted, rows, accepted) -> {TAG}_stall.json
STALL = os.environ.get("R57_STALL") == "1"
assert not STALL or os.environ.get("BIGMOE_TRACE"), "R57_STALL needs BIGMOE_TRACE=1"
S10B_DIR = os.environ.get("R57_S10B_DIR")               # with BIGMOE_S10B=1 (R-57b data for Fit12): per run the gated calls'
if S10B_DIR:                                            # ids + MoE inputs (bigmoe/s10b.py) -> <dir>/<prompt>_<arm>_<n>.npz
    os.makedirs(S10B_DIR, exist_ok=True)        # arm gate+t: drafts cut where the running confidence < TRUNC              # runs per arm per prompt: 2 = ABBA order, 1 = one each
HARD = 600                                               # s of running time: no new check after this (job stop at 660)
ACT = [0.0]                                              # running time without SIGSTOP freezes (Leap's ticker)
def _tick():
    last = time.time()
    while True:
        time.sleep(1)
        now = time.time()
        ACT[0] += min(now - last, 3.0)
        last = now
threading.Thread(target=_tick, daemon=True).start()
el = lambda: ACT[0]
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
med = lambda v: float(np.median(v)) if len(v) else float("nan")

t0 = time.time()
if CHATSW:                                               # EvalSuite's load asks for no MTP head: load it resident (d23c_mtptrunc)
    import bigmoe.loader as BL
    _ls = BL.load_streamed
    BL.load_streamed = lambda md, C_, mtp=False: _ls(md, C_, mtp=True)
rt, st = EM.load(EM.variant("ref"), C=C)
if CHATSW:
    BL.load_streamed = _ls
model = rt.model
_CONV = []                                               # records3 layers converted in place (native / native+p11; the
if TIER and getattr(st, "_tier_idx", None):              # engine's layout patch + Leap's ref2 router run them per layer)
    _CONV = sorted((int(L), r.get("layout")) for L, r in st._tier_idx["layers"].items() if r.get("layout", "row-major") != "row-major")
if _CONV or os.environ.get("R57_MODEL_DIR"):
    import bigmoe.tier as _BTL
    assert not _CONV or hasattr(_BTL, "LAYOUTS"), "converted records3 layers on an engine without the layout patch"
    print(f"[r57] model dir {evalsuite.env.REF_MODEL}: converted layers {_CONV if len(_CONV) < 48 else 'all 48 ' + _CONV[0][1]}",
          flush=True)
import ref2
if os.environ.get("R57_NAX") == "1":                     # C1 (DAY9 #4; Leap 101573d): NAX6 on every tier call, OOO's two phases
    assert "tier_nax_ooo" in ref2.CFG, "Leap's ref2 has no tier_nax_ooo (needs ~/Leap 101573d)"   # included; a NEW reference:
    ref2.CFG["tier_nax_ooo"] = True                      # tokens != REF2-T, so determinism across NAX processes only
    if _CONV:                                            # converted layers: NAX6-N (== NAX6 on row-major records, bitwise;
        assert len(_CONV) == 48, "R57_NAX on a part-converted store (nax_native is per store, not per layer)"
        ref2.CFG["nax_native"] = True                    # p11 found from the pools' dtype; Leap N7-13 / P11 90456ce)
if os.environ.get("R57_H1C") == "1":                     # H-1c / the in-kernel hybrid (Leap 11-02 23:1x; FAST only: a pair's bits
    assert os.environ.get("R57_NAX") == "1" and "nax_hybrid" in ref2.CFG, "R57_H1C needs R57_NAX=1 and Leap's ref2 nax_hybrid"
    ref2.CFG["nax_hybrid"] = True                        # depend on whether it is alone in its group): NAX6's one-pair groups
import bigmoe.stream as BS
import bigmoe.gated as BG
from mtplx import cache_state as CS
Q = ref2.Q
assert BG.MAXK == 160 and BG.XLINES == 4 and (not _PF_ARM or PF_B <= 30 + 31 * BS._PRED_LINES), (PF_B, BS._PRED_LINES)
if TIER and os.environ.get("R57_COMBINE") == "1":       # R-69 (Leap R-69L/R-69D): EM.load's block does the sorted top-k
    import bigmoe.tier as _BT                            # and MLX's combine itself, so tier_call never got the scores and
    from mtplx.models import qwen4_exp as _QX            # TIER_COMBINE / OOO never engaged. This copy of EvalSuite's
    _ev_block = _QX.SparseMoeBlock.__call__              # sorted block (evalsuite/model.py block) hands the combine to
                                                         # tier_call at 1-16 rows (R-69: bitwise == MLX's combine there)
    def _fused_block(self, x):
        bm = getattr(self.switch_mlp, "_bm", None)
        if (bm is None or "tier" not in bm or not EM.STATE.sorted or not (_BT._COMBINE and _BT._GLU == "3c")
                or x.ndim < 2 or int(np.prod(x.shape[:-1])) > 16):
            return _ev_block(self, x)                    # prefill and every other call: EvalSuite's block as is
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        order = mx.argsort(-scores, axis=-1)
        inds, scores = mx.take_along_axis(inds, order, axis=-1), mx.take_along_axis(scores, order, axis=-1)
        if EM.STATE.capture is not None:
            EM.STATE.capture.setdefault(bm["L"], []).append(inds)
        shared = mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)   # built first: OOO's spins wait behind it
        bm["tier"]["scores"], bm["tier"]["shared"] = scores, shared
        try:
            y = self.switch_mlp(x, inds)
        finally:
            bm["tier"].pop("scores", None); bm["tier"].pop("shared", None)
        if y.ndim == inds.ndim + 1:                      # a path that left the scores unused: MLX's combine, as EvalSuite
            y = (y * scores[..., None]).sum(axis=-2)
        return y + shared
    _QX.SparseMoeBlock.__call__ = _fused_block
if os.environ.get("K73_EXACT") == "1":                  # K-73 (Leap 11-01, ~/Leap/scripts/k73_live.py, verbatim): the exact-safe
    assert TIER and os.environ.get("R57_COMBINE") == "1", "K73_EXACT runs on R57_PRESET=45"   # stack on the 45 GB default:
    from router_fused import router_fused                # ref2 att_rope + idx_prep (R-72), hc_write (R-73); the MoE block with
    from glue_fused import moe_tail, shared_cat          # R-70's fused router and R-73's fused tail (y + sigmoid(g) * s);
                                                         # + R-74 (Leap 12:1x): the shared expert's gate / up + its gate as one call
    for _k in ("att_rope", "idx_prep", "hc_write"):
        ref2.CFG[_k] = True
    import draft_fused                                   # R-76 (Leap 12:3x): DFlash2's grouped conv as one kernel + the mask
    draft_fused.install()                                # memoized (drafts bitwise identical; 649 -> 364 kernels per propose)
    _fb = _QX.SparseMoeBlock.__call__
    K73N = [0]
    def _exact_block(self, x):
        bm = getattr(self.switch_mlp, "_bm", None)
        if (bm is None or "tier" not in bm or not EM.STATE.sorted or not (_BT._COMBINE and _BT._GLU == "3c")
                or x.ndim < 2 or int(np.prod(x.shape[:-1])) > 16 or not self.norm_topk_prob):
            return _fb(self, x)
        K73N[0] += 1
        inds, scores = router_fused(self.gate(x), self.top_k)   # R-70: bitwise == softmax .. argsort .. take chain
        if EM.STATE.capture is not None:
            EM.STATE.capture.setdefault(bm["L"], []).append(inds)
        g, s = shared_cat(self, x)                              # R-74: == shared_expert_gate(x), shared_expert(x), bitwise
        bm["tier"]["scores"], bm["tier"]["shared"] = scores, s   # OOO's spins wait behind the shared expert, as before
        PR["cur"] = bm["L"] if PR["on"] else None              # R57_PREROUTE: OOO's combine stashes L's partial sums
        try:
            y = self.switch_mlp(x, inds)
        finally:
            bm["tier"].pop("scores", None); bm["tier"].pop("shared", None)
            PR["cur"] = None
        if y.ndim == inds.ndim + 1:
            y = (y * scores[..., None]).sum(axis=-2)
        if PR["on"] and bm["L"] in PR["y"]:                     # R57_PREROUTE: L's partial block outputs
            PR["blk"][bm["L"]] = [moe_tail(yp.reshape(y.shape), g, s, exp="precise") for yp in PR["y"].pop(bm["L"])]
        return moe_tail(y, g, s, exp="precise")                  # R-73: == y + sigmoid(g) * s, bitwise
    _QX.SparseMoeBlock.__call__ = _exact_block
if PREROUTE:                                             # OOO's combine, once more per partial (same y1 / y2 / phase)
    assert os.environ.get("K73_EXACT") == "1" and _BT._OOO, "R57_PREROUTE hooks K-73's block and OOO's combine"
    _TGm = _BT._tier_glu_mod(); _cp0 = _TGm.combine_phase
    def _cp_pr(y1, y2, phase, scores, *a, **kw):
        out = _cp0(y1, y2, phase, scores, *a, **kw)
        if PR["on"] and PR["cur"] is not None:           # scores [M, R] in gate-rank order: keep ranks < 3 / < 6
            rk = mx.arange(scores.shape[-1])
            PR["y"][PR["cur"]] = [_cp0(y1, y2, phase, scores * (rk < c_).astype(scores.dtype), *a, **kw) for c_ in (3, 6)]
        return out
    _TGm.combine_phase = _cp_pr
MTP_HEAD, MTP_K = os.environ.get("R57_MTP_HEAD", "b5"), int(os.environ.get("R57_MTP_K", "15"))
_MTP_MD5 = {"b5": "1f8de68a468b6ae16a9f42c62cbca39f"}   # D-23d's head, pinned like the M-17 sizes
if CHATSW:                                               # the head's trained parts over the shipped MTP layer (D-22 b5: attention,
    import hashlib                                       # router, shared expert; experts, norms, lm_head as shipped)
    assert getattr(model.language_model, "mtp", None) is not None, "no MTP head loaded"
    _hp = os.path.expanduser(f"~/TrainLab/ckpt/d22_{MTP_HEAD}.safetensors")
    assert MTP_HEAD not in _MTP_MD5 or hashlib.md5(open(_hp, "rb").read()).hexdigest() == _MTP_MD5[MTP_HEAD], f"{_hp}: md5"
    sys.path.insert(0, os.path.expanduser("~/TrainLab"))
    import mtp125 as _M                                  # its import-time env defaults = EvalSuite's forced ones (no-ops here)
    _TH = _M.Trainable(model.language_model.mtp)
    _TH.load_weights(_hp)
    _M.install(model.language_model.mtp, _TH, dtype=mx.bfloat16)   # touches the head's own modules only
    del _TH
import re
from tokenizers import Tokenizer                         # NewIdeas' type rule (TrainLab scripts/type_rule.py, verbatim from
_TKZ = Tokenizer.from_file(os.path.expanduser("~/BigMoE/models/bare/tokenizer.json"))   # d23c_mtptrunc.py rule()): the chat
_TDC = {}                                                # R57_RELAX_V2: token-text cache for the span tagger (same strings)


def _tdec(x):
    x = int(x)
    t = _TDC.get(x)
    if t is None:
        t = _TDC[x] = _TKZ.decode([x])
    return t
_CODE = re.compile(r"(^|\n)\s*(def |class |import |from \w+ import|#include|fn |func |public |const |let |var )|```|\breturn\b|\w+\(.*\)\s*:|>>>")
_MATH = re.compile(r"\\boxed|\\frac|\\sqrt|\$[^$]+\$|\b(solve|compute|calculate|how many|how much|what is the (sum|product|value|area|probability))\b"
                   r"|\d+\s*[-+*/^=]\s*\d+", re.I)   # switch, R57_COPY, R57_D7B=code, R57_CHATROWS and R57_FAST all ask it
_TOOLS = re.compile(r"<tools>|\"type\":\s*\"function\"|tool_call|<function|\bfunctions?\b.*\bparameters\b", re.I | re.S)
NOCOM = os.environ.get("R57_NOCOM") == "1"              # R57_NOCOM (Fable FULL's LOSSY350 T-B1; ~/FableGod/full/build_1105/
NOCOM_END = _TKZ.token_to_id("<|im_end|>")              # INSTALL.md): the comment-free code mode's PROMPT form. On the humaneval
NOCOM_IDS = _TKZ.encode("\n\nDo not write comments in the code.", add_special_tokens=False).ids if NOCOM else []   # set one
assert not NOCOM or (NOCOM_END is not None and NOCOM_IDS), "R57_NOCOM: the tokenizer"   # sentence joins the end of the user turn
def rule(item):
    full = item.get("prompt") or _TKZ.decode(item["prompt_ids"], skip_special_tokens=False)
    user = item.get("user") or full
    if _TOOLS.search(full):
        return "agent"
    if _CODE.search(user) or _CODE.search(full[-4000:]):
        return "code"
    if _MATH.search(user):
        return "math"
    return "chat"
TXT = model.language_model.model
lay = TXT.layers
NL = len(lay)
SW_ = [l.mlp.switch_mlp for l in lay]
VTIER = os.environ.get("R57_VTIER", "")                  # "base,sidecar" slots per tier layer (BIGMOE_EXT_BUILD=build_vt;
if VTIER:                                                # NewIdeas' RANKED 33, the orchestrator's Wed #1): a purgeable
    assert hasattr(BG.ext, "set_vtier"), "R57_VTIER needs BIGMOE_EXT_BUILD=build_vt"   # victim tier behind each LRU pool
    _vb, _vs = (int(x) for x in VTIER.replace("/", ",").split(","))   # "64/32" too: c45_runner turns arm commas into spaces
    VT_BYTES = sum(BG.ext.set_vtier(sw._bm["L"], _vb) + (BG.ext.set_vtier(sw._bm["tier"]["side"], _vs)
                   if (sw._bm.get("tier") or {}).get("side") is not None else 0) for sw in SW_ if not sw._bm["full"])
    VT_MAP = [[sw._bm["L"], (sw._bm.get("tier") or {}).get("side")] for sw in SW_]
    say(f"R57_VTIER: {_vb} / {_vs} slots per base / sidecar pool = {VT_BYTES / 1e9:.2f} GB volatile (purgeable: NOT in "
        f"phys_footprint; macOS reclaims it before compression or swap)")
NW = os.environ.get("R57_NEVERWAIT") == "1"             # NEVER-WAIT (docs/NEVERWAIT_IFACE.md, Leap's ~/Leap/data/nw/NW_DESIGN.md):
if NW:                                                   # a cold demand expert runs one layer late in ref2's one-call instead of
    assert FAST and os.environ.get("REF2_OOO_1C") == "1", "R57_NEVERWAIT needs a FAST arm on the one-call (REF2_OOO_1C=1)"
    assert os.environ.get("REF2_NEVERWAIT") == "1", "R57_NEVERWAIT arms ref2 with REF2_NEVERWAIT=1: set both"
    assert all(hasattr(ref2, a_) for a_ in ("NW_RESET", "NW_ON", "NWCNT")), "R57_NEVERWAIT needs Leap's never-wait ref2"
QSPLIT = float(os.environ["R57_QOM_SPLIT"]) if os.environ.get("R57_QOM_SPLIT") else None   # ported from r57_screen.py (Fri 10:0x,
assert QSPLIT is None or FAST == "QOM", "R57_QOM_SPLIT is a QOM-KEEP option (R57_FAST=QOM)"   # NewIdeas' SPLIT-NAX arm): the rows'
QC1C = os.environ.get("R57_QCNT1C") == "1"               # draft cf reach ref2's one-call as t["qom"][2]; R57_QCNT1C sums ref2.QCNT_1C's
assert not QC1C or hasattr(ref2, "QCNT_1C"), "R57_QCNT1C needs Leap's ref2 eb43fcc2 or later"   # [kept, dropped, groups, split drops]
FG = dict(on=False, keep=None, sure=None, sure_np=None, t3=False,   # R57_FAST: this check's keep mask [rows, top-k] / U3's
          cold=False, warm=None, cap=None, dump=None, chk=0, qom=False, qcf=None)   # qom: QOM's optional uses (bit 21)   # U3C: the request's warm table [48, 512], prefill capture, dump
# R57_CBB=1 (the stale-slot check, the orchestrator 11-03 22:5x on Leap's S7B-PROBE 50de28c / fb999be: a slot the GPU read
# earlier in the SAME command buffer can be re-read stale after the CPU rewrites it; across buffers it is fresh): every MoE
# layer call starts a new command buffer (its input and routing evaluated first), so each earlier read of any slot sits in
# a finished buffer. Same kernels, same bytes: with no stale reads, tokens and logits equal the sync-free engine's bit for
# bit. R57_LOGHASH=1: each check's verify logits hashed (run record "lh"), for that comparison.
CBB = os.environ.get("R57_CBB") == "1"
LOGHASH = os.environ.get("R57_LOGHASH") == "1"
if CBB:
    _SWB = type(SW_[0]); _cbb_prev = _SWB.__call__
    def _cbb_call(self, x, indices, *a, **kw):
        mx.eval(x, indices)
        return _cbb_prev(self, x, indices, *a, **kw)
    _SWB.__call__ = _cbb_call
U3C_LEN = os.environ.get("R57_U3C_LEN") == "1"
U3C_DUMP = os.environ.get("R57_U3C_DUMP", "")           # a directory: one .npz per prompt for Breakthrough's mask check
POP45 = None
if FAST and any(v.get("cold") for v in FASTARMS[FAST].values()):
    POP45 = np.load(os.path.expanduser("~/FableGod/fablec/masks/pop_cap45.npy"))   # [48, 512] share of cap45's checks
    assert POP45.shape == (48, 512), POP45.shape
                                                         # sure rows [rows] (mx + numpy) while it verifies
if FAST:                                                 # EvalSuite's install_gate (route mode), for any row count <= 16
    _SWC = type(SW_[0])
    assert all(type(sw) is _SWC for sw in SW_), {type(sw) for sw in SW_}
    _sw_call = _SWC.__call__
    def _fast_call(self, x, indices, *a, **kw):
        km = FG["keep"]
        sr = FG["sure"]
        if FG["cap"] is not None and self._bm["L"] < 48:  # U3C: the prompt's footprint, every prefill call of a trunk layer
            FG["cap"][self._bm["L"], np.unique(np.array(indices).reshape(-1))] = True
        if FG["on"] and sr is not None and indices.ndim >= 2 and indices.shape[-2] == sr.shape[0]:   # U3: Q1-UNION
            K_ = indices.shape[-1]; ids2 = indices.reshape(-1, K_); uns = ~sr   # a sure row keeps ranks 0-1 and the ranks
            top2 = mx.arange(K_)[None, :] < 2                # whose expert is in the check's union (dm_q1u_masks.py: every
            if FG["cold"]:                                   # U3C: the base sets (unsure: ranks 0-3 + warm; sure: 0-1)
                rk = mx.arange(K_)[None, :]
                cand = mx.where(uns[:, None], (rk < 4) | FG["warm"][self._bm["L"]][ids2], top2)
                inU = ((ids2[:, :, None, None] == ids2[None, None, :, :]) & cand[None, None, :, :]).any(axis=(2, 3))
                km = (cand | inU).reshape(indices.shape)
                if FG["dump"] is not None:                   # kept as graph arrays, read out after the check evaluates
                    FG["dump"].append((FG["chk"], self._bm["L"], ids2, sr, cand | inU))
            else:
                cand = uns[:, None] | top2                   # rank of the unsure rows, ranks 0-1 of the sure ones)
                inU = ((ids2[:, :, None, None] == ids2[None, None, :, :]) & cand[None, None, :, :]).any(axis=(2, 3))
                km = (uns[:, None] | top2 | inU).reshape(indices.shape)
                if FG["qom"]:                                # QOM: U3's drops become OPTIONAL uses (bit 21): the loader
                    opt = sr[:, None] & ~(top2 | inU)        # serves them only when resident (else -2, never read);
                    self._bm["tier"]["qom"] = (opt, sr, FG.get("qcf"))   # _ooo_call_t3 flags ONLY its base post's copy (bit 21) and
                    km = None                                # applies Breakthrough's rule after the spin; the graph's ids
                                                             # stay clean (15:29: flagged ids in the graph -> a bad id at L5)
            if FG["t3"]:
                self._bm["tier"]["nolsb"] = (sr, FG["sure_np"])   # _ooo_call_t3: those rows' ranks 0-1 at 3 bits
        if FG["on"] and km is not None and indices.ndim >= 2 and indices.shape[-2] == km.shape[-2]:
            t = self._bm["tier"]
            assert t.get("scores") is not None, "the gate needs the fused combine's scores (BIGMOE_TIER_COMBINE)"
            t["scores"] = mx.where(km, t["scores"], mx.zeros_like(t["scores"]))
            indices = mx.where(km, indices, indices[..., :1])   # a dropped rank reads the row's rank-0 expert: no new read
        try:
            return _sw_call(self, x, indices, *a, **kw)
        finally:
            self._bm["tier"].pop("nolsb", None)
            self._bm["tier"].pop("qom", None)
    _SWC.__call__ = _fast_call
    if any(v.get("t3") for v in FASTARMS[FAST].values()):   # U3's 3-bit sure rows: LP1's kernel test (a pair at a 4-bit
        import bigmoe.tier as _BT3                        # rank whose LSB slot is < 0 runs at tier 3; unchanged for slots
        _BT3._tier_glu()                                  # >= 0) and tier._ooo_call with the sure rows' sidecar reads off;
        _BT3._lp1_patch()                                 # _tier_glu first puts tierglu on sys.path (U3 crashed Wed 01:02)
        _osrc = open(_BT3.__file__).read(); _osrc = _osrc[_osrc.index("def _ooo_call("):_osrc.index("def tier_call(")]
        import hashlib as _hl3
        assert _hl3.md5(_osrc.encode()).hexdigest() == "d5bf5acd2a3f5db315852ee86a59f84a", "tier._ooo_call changed: re-copy"
        _ooo_call0 = _BT3._ooo_call
        def _ooo_call_t3(x, ids, bm, t, post, sc, shared):   # tier._ooo_call verbatim + the sure rows of t["nolsb"]
            nlz = t.pop("nolsb", None)
            qm = t.pop("qom", None)                      # QOM: (optional uses [M, K], sure rows [M]) or None
            if nlz is None:
                assert qm is None, "QOM needs U3's t3 path (nolsb)"
                return _ooo_call0(x, ids, bm, t, post, sc, shared)
            nl, nl_np = nlz
            TG = _BT3._tier_glu_mod()
            _BT3.OOO_STATS["calls"] += 1
            K = ids.shape[-1]
            x2 = x.reshape(-1, x.shape[-1])
            M = x2.shape[0]
            lo4 = [s_ for s_ in t["segs"] if s_[2] == 4]
            R4 = lo4[-1][1] if lo4 else 0
            assert not lo4 or lo4[0][0] == 0, t["segs"]
            gate, L = bm["gate"], bm["L"]
            ids_p = ids if qm is None else mx.where(qm[0].reshape(ids.shape), ids.astype(mx.int32) | (1 << 21),
                                                    ids.astype(mx.int32))   # QOM: optional uses marked for the loader only
            dep_m, n_m = post(L, ids_p)
            pre_m = dep_m[1:1 + n_m].reshape(M, K)
            ok = pre_m >= 0
            nl2 = nl.reshape(M, 1)
            if R4:                                       # sure rows ask the sidecar for an unsure row's rank-0 expert
                un = np.where(~np.asarray(nl_np))[0]     # (needed anyway, so nothing new is read), then use slot -2
                dup = ids.reshape(M, K)[int(un[0]) if len(un) else 0, 0]
                ids_s = mx.where(nl2, dup, ids.reshape(M, K)[:, :R4]).reshape(*ids.shape[:-1], R4)
                dep_l, n_l = post(t["side"], ids_s, after=dep_m)
                pre_l = dep_l[1:1 + n_l].reshape(M, R4)
                ok = ok & mx.concatenate([(pre_l >= 0) | nl2, mx.ones((M, K - R4), mx.bool_)], axis=-1)
            phase = mx.where(ok, 1, 2).astype(mx.int32)
            l1 = mx.where(nl2, -2, mx.maximum(pre_l, 0)) if R4 else None
            y1 = TG.tier_glu_phase(x2, t["base"], t["sideP"], pre_m, l1, phase, 1, plan=_BT3.PLAN)
            after = [y1] + ([shared] if shared is not None else [])
            sm = gate.spin(mx.depends(dep_m, after), L, n_m).reshape(M, K)
            sl = mx.where(nl2, -2, gate.spin(mx.depends(dep_l, after), t["side"], n_l).reshape(M, R4)) if R4 else None
            if qm is not None:                           # QOM: Breakthrough's keep-set rule on the loader's answer (resident
                ids_r = ids.reshape(M, K).astype(mx.int32) & 0xFFFFF   # = a slot; an optional miss came back -2, unread)
                cf_ = mx.where(qm[1].reshape(M), 0.0, float("nan"))    # sure rows verify a draft; the bonus row is exact
                kq, _, _ = QK.qom_keep_mx(ids_r, sc.reshape(M, K), cf_, sm >= 0)
                sm = mx.where(kq, sm, sm[:, :1])         # a dropped use reads the row's rank-0 slot (always loaded), weight 0
                sc = mx.where(kq, sc.reshape(M, K), mx.zeros_like(sc.reshape(M, K)))
            y2 = TG.tier_glu_phase(x2, t["base"], t["sideP"], sm, sl, phase, 2, plan=_BT3.PLAN)
            return TG.combine_phase(y1, y2, phase, sc.reshape(-1, K)).reshape(*x.shape[:-1], -1)
        _BT3._ooo_call = _ooo_call_t3
class _Corr:                                             # R-57b: Fit12's logit correction on the L+1 router (raw @ W + b),
    def __init__(self, r):                               # a callable in the router's place, so stream.py stays unchanged
        self.r, self.W, self.b = r, None, None
    def __call__(self, x):
        lg = self.r(x)
        return lg if self.W is None else lg.astype(mx.float32) @ self.W + self.b
CORR = {L: _Corr(lay[L + 1].mlp.gate) for L in range(NL - 1)}
NEXT = {L: [(CORR[L] if os.environ.get("R57_CORR") else lay[L + 1].mlp.gate, None, PF_B)] for L in range(NL - 1)
        if not SW_[L]._bm["full"] and not SW_[L + 1]._bm["full"]}   # stream.py's wiring for qwen4_exp (ratio None)
def corr_for(key):
    """Fit12's held-out 2-fold maps: each prompt gets the map fit on the other three (R57_CORR = the maps' directory)"""
    if not os.environ.get("R57_CORR"):
        return None
    d = os.environ["R57_CORR"]
    f = "corr_fit68_91_1282.npz" if any(k in key for k in ("1640", "694", "708")) else "corr_fit1640_694_708.npz"
    z = np.load(os.path.join(d, f))
    for L, c in CORR.items():
        T = L + 1
        c.W = mx.array(z[f"W{T}"].astype(np.float32)); c.b = mx.array(z[f"b{T}"].astype(np.float32))
    mx.eval([c.W for c in CORR.values()] + [c.b for c in CORR.values()])
    return f
say(f"R-57 harness{' (REF2-T tiers ' + os.environ['BIGMOE_TIER'] + ', C_l ' + os.environ['BIGMOE_TIER_CL'] + ')' if TIER else ''}: full 125B via load_streamed (C = {C}), loaded in {time.time() - t0:.0f} s, {mx.get_active_memory() / 1e9:.1f} GB "
    f"active; REF2 + SB; gate rows MAXK {BG.MAXK} (ext {BG._BUILD}), PRED_LINES {BS._PRED_LINES} (ranked list <= "
    f"{30 + 31 * BS._PRED_LINES}), PF_B {PF_B}, PF_JIT {os.environ.get('BIGMOE_PF_JIT')}, PF_QD {os.environ.get('BIGMOE_PF_QD')}, "
    f"PF_VICTIM {os.environ.get('BIGMOE_PF_VICTIM')}; {len(NEXT)} look-ahead layers; CAP {CAP}")

PRIOR = os.environ.get("R57_PRIOR")                      # SEED45: BIGMOE_EVICT_PRIOR's popularity (data/seed45/prior45.npz),
def prior_for(key):                                      # each prompt held out (its "loo_" map: built on the other five)
    if not PRIOR:
        return None
    z = np.load(os.path.expanduser(PRIOR))
    k = "loo_" + key.replace("/", "_").replace(".json", "")
    k = k if k in z.files else "all"
    for L, sw in enumerate(SW_):
        if not sw._bm["full"]:
            BG.ext.set_prior(sw._bm["L"], z[k][L].astype(np.float32).tolist())
    return k

CPROBE = None                                            # R57_COPYPROBE (Fable FULL's LOSSY350 T-A1; ~/FableGod/full/build_1105/
if os.environ.get("R57_COPYPROBE"):                      # INSTALL.md): rolled-back timing passes over copied rows (copyprobe.py,
    import importlib.util as _ilu_cp                     # loaded by path; it wraps the MoE block for calls > 16 rows while a probe
    _sp_cp = _ilu_cp.spec_from_file_location("copyprobe", os.path.expanduser(   # pass is being built, and only then)
        os.environ.get("R57_COPYPROBE_PY", "~/BigMoE/scripts/copyprobe.py")))
    CPROBE = _ilu_cp.module_from_spec(_sp_cp); _sp_cp.loader.exec_module(CPROBE)
    CPROBE.install(globals())
def set_arm(arm):                                        # gate+t+pf (R-57 at 45 GB, the orchestrator 11-02 21:4x): the t cut
    BS.GATE_N[0] = 40 if arm == "slot" else 160          # + the ranked L+1 look-ahead (the tier path posts L+1's preds)
    BS._PF_RANKED = arm in ("gate+pf", "gate+t+pf")
    for L, sw in enumerate(SW_):
        if arm in ("gate+pf", "gate+t+pf") and L in NEXT:
            sw._bm["next"] = NEXT[L]
        else:
            sw._bm.pop("next", None)

def pf1(on):
    if not on and BS._STAGE["bufs"] is not None:
        BS._stage_drop()
    BS._PF_PREFILL = on

def stats():                                             # misses, demand SSD busy (load batches' wall time), gate service of
    s = st.stats()                                       # calls with misses (post seen -> release: the GPU's spin bound;
    return np.array([s["misses"], s["io_s"], s["slow_s"], s["slow_calls"],      # OOO overlaps part of it); then (Cloud Fable 3's
                     s.get("uses", 0), s.get("stalled", 0)], dtype=np.float64)  # C3-R8, Thu) the ext's expert uses = distinct
                                                         # experts computed per layer call, resident included, and the layer calls
                                                         # that waited on loads (ext :881); appended, so indices 0-3 keep their meaning

class _RU4(ctypes.Structure):                            # rusage_info_v4: a 16-byte uuid, then uint64 fields
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"u{i}", ctypes.c_uint64) for i in range(40)]
_LIBC = ctypes.CDLL(None)

def disk_read_bytes():                                   # ri_diskio_bytesread (u16): this process's SSD reads. The ext reads
    ru = _RU4()                                          # F_NOCACHE, so every load counts (checked 11-01: 210 MB -> +209.7 MB)
    return ru.u16 if _LIBC.proc_pid_rusage(os.getpid(), 4, ctypes.byref(ru)) == 0 else 0

# ---------------------------------------------------------------- taps for the drafter (Leap's S45 harness)
TAP = {"on": False, "x": [[] for _ in TAPS]}
TAP_OF = {id(lay[L]): i for i, L in enumerate(TAPS)}
_layer_call = Q.DecoderLayer.__call__
PR_IDX = {id(l_): i_ for i_, l_ in enumerate(lay)}
def _pr_top(lay_, z):                                    # lay_'s router on z: per row top-10, and the ranked unique list of 96
    lg = lay_.mlp.gate(z).astype(mx.float32)             # (each expert by its best row log-prob: the look-ahead's PF_RANKED
    best = (lg - mx.logsumexp(lg, axis=-1, keepdims=True)).reshape(-1, lg.shape[-1]).max(axis=0)   # list at PF_B 96)
    return mx.argpartition(-lg, kth=9, axis=-1)[..., :10], mx.argsort(-best)[:96]
def _layer_pr(self, hidden, *, input_ids, ssm_mask, cache):   # R57_PREROUTE: qwen4_exp DecoderLayer.__call__ verbatim
    if "ple" in self:                                         # (REF2's patched HC read / write looked up at call time)
        hidden = hidden + self.ple(hidden, input_ids, cache)
    i = PR_IDX.get(id(self))
    pre = None
    if i is not None and (i - 1) in PR["cap"]:           # "full": this layer's router on its MLP-side read of the stream
        pre = self.mlp_hyper_connection(hidden)[0]       # before its attention (pending None: no attention write)
        PR["cap"][i - 1]["full"] = _pr_top(self, pre)
    mixed, hyper, inject = self.attn_hyper_connection(hidden)
    if self.is_linear:
        block_out = self.linear_attn(mixed, ssm_mask, cache)
    else:
        block_out = self.self_attn(mixed, cache)
    mixed, hyper, inject = self.mlp_hyper_connection(hyper, pending=(block_out, inject))
    if pre is not None and PR["dump"] is not None:       # R57_PR_DUMP: "full"'s input and this layer's true MoE input
        PR["dump"].append((i, pre, mixed))
    block_out = self.mlp(mixed)
    hidden = Q._hyper_residual_write(hyper, block_out, inject)
    if i is not None and i + 1 < NL and not SW_[i]._bm["full"]:   # the capture: L+1's router on L's MoE input (r57), on
        nxt = lay[i + 1]                                 # its attention-side read of L's output (fullA, the control) and
        caps = dict(r57=_pr_top(nxt, mixed), fullA=_pr_top(nxt, nxt.attn_hyper_connection(hidden)[0]))   # on its MLP-side
        for nm, bo in zip(("p3", "p6"), PR["blk"].pop(SW_[i]._bm["L"], [])):   # read of L's partial outputs; "full"
            caps[nm] = _pr_top(nxt, nxt.mlp_hyper_connection(Q._hyper_residual_write(hyper, bo, inject))[0])   # joins
        PR["cap"][i] = caps                              # at L+1's start
    return hidden
def pr_eval(capd):                                       # R57_PREROUTE: recall@10 of each variant vs L+1's true top-10
    got = [(i, capd[SW_[i + 1]._bm["L"]][-1], c) for i, c in PR["cap"].items() if capd.get(SW_[i + 1]._bm["L"])]
    mx.eval([t for _, t, _ in got] + [z for _, _, c in got for zz in c.values() for z in zz])
    for i, t, caps in got:                               # sample: layer, rows, recall@10 per row, union recall of the
        T = np.array(t).reshape(-1, 10).tolist(); Tu = {e for r_ in T for e in r_}   # top-10s, precision@96, recall@96
        for nm, (z, r96) in caps.items():
            P_ = np.array(z).reshape(-1, 10).tolist(); h96 = len(set(np.array(r96).tolist()) & Tu)
            PRS.setdefault(nm, []).append([i, len(T), float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(P_, T)])),
                                           len({e for r_ in P_ for e in r_} & Tu) / len(Tu), h96 / 96, h96 / len(Tu)])
    PR["cap"].clear()
    if PR["dump"]:                                       # R57_PR_DUMP: this check's rows (float32: exact for bf16 / f16)
        mx.eval([a for _, a, _ in PR["dump"]] + [b for _, _, b in PR["dump"]])
        for j, a, b in PR["dump"]:
            a = np.array(a.astype(mx.float32)).reshape(-1, a.shape[-1]); b = np.array(b.astype(mx.float32)).reshape(-1, b.shape[-1])
            PRD["L"].append(np.full(len(a), j, np.int16)); PRD["chk"].append(np.full(len(a), PRD["n"], np.int32))
            PRD["row"].append(np.arange(len(a), dtype=np.int16)); PRD["X"].append(a); PRD["Y"].append(b)
        PRD["n"] += 1; PR["dump"].clear()
def _layer(self, *a, **kw):
    out = _layer_pr(self, *a, **kw) if PR["on"] else _layer_call(self, *a, **kw)
    i = TAP_OF.get(id(self))
    if TAP["on"] and i is not None:
        o = out if isinstance(out, mx.array) else out[0]
        B, L, W = o.shape
        TAP["x"][i].append(o.reshape(B, L, 4, W // 4).astype(mx.float32).mean(axis=2)[0].astype(mx.float16))
    return out
Q.DecoderLayer.__call__ = _layer
def tap_reset():
    TAP["x"] = [[] for _ in TAPS]
def taps():
    f = [mx.concatenate(x) if len(x) > 1 else x[0] for x in TAP["x"]]
    return mx.stack(f, axis=1).reshape(f[0].shape[0], -1)

D8NAX = os.environ.get("R57_D8NAX") == "1"
def _load_drafter(q4):
    d_ = DF.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
    d_.load_weights(os.path.expanduser("~/Leap/data/l120_ft.safetensors"), strict=False)
    if q4:                                               # D7b: quantized as it loads, before bind and any draft_fused install
        sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s4/Leap_scripts"))
        import d7b_live
        d7b_live.apply(d_)
    elif D8NAX:                                          # R57_D8NAX (Leap 2945bf7, BP-6): the drafter's layer Linears + fc at
        sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s4/Leap_scripts"))   # 8-bit RTN g64 on qmm16c8 (needs no REF2): propose ~10.4 ->
        import d7b_nax                                   # 7.5 ms, kept x1.003 code / x1.001 math; exact (drafts are verified)
        d7b_nax.apply8(d_)
    d_.bind(model)
    mx.eval(d_.parameters())
    return d_
draft = _load_drafter(D7B == "all")
DRAFT16, DRAFT4 = draft, (_load_drafter(True) if D7B == "code" else None)   # D7B=code: the code prompts' drafter
if os.environ.get("R57_D7BNAX") == "1" and D7B:        # Leap def104e: the D7b drafter's 4-bit layers run qmm16c at <= 16
    sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s4/Leap_scripts"))   # rows whatever REF2's state (r57 removes REF2 around each
    import d7b_nax                                       # propose, so MLX's slow 16-row qmm ran: 9.35 -> 7.3 ms, kept x0.996)
    d7b_nax.install(DRAFT4 if DRAFT4 is not None else draft)
MASK, SWIN = int(draft.config.mask_token_id), int(draft.config.sliding_window)
K83 = os.environ.get("K83_EXACT") == "1"                # K-83 (Leap 11-02, runner 62 at C 160: every token == REF2-T, draft 11.4
if K83:                                                  # -> 10.2 ms, steady +0.7 %): the second exact stack on K-73, verbatim from
    assert os.environ.get("K73_EXACT") == "1", "K83_EXACT stacks on K73_EXACT"   # ~/Leap/scripts/k73_live.py: R-78 the
    import draft_fused as _DFU                           # drafter's 16-row spellings (forms; R-76 is K-73's), R-77 the fused
    from chain_fused import chain_fused as _chain_fused  # selector walk, R-82 hc_gate, R-83 gdn_inproj_x
    _DFU.install(forms=D7B != "all" and not D8NAX, draft=draft)   # R-78's forms are bf16 spellings: never on D7b / D8
    for _k in ("hc_gate", "gdn_inproj_x"):
        ref2.CFG[_k] = True
# BUILD DAY (Wed 11-04) switches, each off unless set; the all-in FAST code preset sets all three (+ U3C + R57_D7B=code):
if FAST == "QOM" and os.environ.get("REF2_OOO_1C") == "1":   # one-call OOO knows QOM's -2 base slots since Leap 00961fc
    assert "ooo_1c_qom" in ref2.COUNTS, "REF2_OOO_1C with QOM needs Leap's ref2 00961fc (else QOM's drops are misread)"
if os.environ.get("R57_GDNB") == "1":                    # GDNB (Leap 70cd998, EXACT, bitwise == off): the one-dispatch GDN body
    assert "gdn_body" in ref2.CFG, "Leap's ref2 has no gdn_body"   # at 1-16 rows (-0.71 ms per check); prefill falls back
    ref2.CFG["gdn_body"] = True
if os.environ.get("R57_HCB") == "1":                     # hc_b (Leap D1-B kernel B, EXACT): the HC read's partition sum + gate,
    assert "hc_b" in ref2.CFG, "Leap's ref2 has no hc_b"   # MLX's K-320 up GEMM and the tail as one kernel, bitwise == the S4
    ref2.CFG["hc_b"] = True                              # chain (-0.94..-1.13 ms per 96-read chain, quiet); over hc_gate
if os.environ.get("R57_DNR") == "1":                     # DENSE-N-RAM (Leap 2190e10, EXACT): every 4-bit dense weight with
    assert hasattr(ref2, "dense_native_convert"), "Leap's ref2 has no dense_native_convert (needs 2190e10)"   # N >= 1024
    _dn_n, _dn_mb = ref2.dense_native_convert(model)     # in place -> its native copy (one copy, no RAM): <= 16-row calls run
    print(f"[r57] DENSE-N-RAM: {_dn_n} weights native ({_dn_mb:.0f} MB, in place)", flush=True)   # qmm16n == qmm16c bitwise
if os.environ.get("R57_HC8") == "1":                     # HC8-1P (Leap c805fd0, v4 34206f9; FAST: tokens differ from REF2-T):
    sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s4/Leap_scripts"))   # the trunk mixers -> 8-bit g64 (-0.6 GB) + the one-pass 8-bit
    import hc8r                                          # HC read at <= 16 rows; the install and the switch go together
    assert "hc8_fast" in ref2.CFG, "Leap's ref2 has no hc8_fast"
    _h8 = hc8r.hc8_install(model)
    ref2.CFG["hc8_fast"] = True
    print(f"[r57] HC8: {_h8[0]} mixers 8-bit ({_h8[1]:.0f} -> {_h8[2]:.0f} MB), hc8_fast v{ref2.CFG.get('hc8_v')}", flush=True)
# FAST-TAX (Fable B Thu 11-05; ~/FableGod/fableb/LOSSY350_B.md section 4, build_1105/INSTALL.md; FAST ONLY, off unless set):
# the exact track's dense machinery that a FAST arm does not owe. R57_FASTTAX=1 (or a list: att, inproj, att+inproj) turns
# on ref2's two REF2c pieces, which no FAST arm carried:
#   att     ref2 att_single (R-62): each 16-row attention call as ONE fused SDPA instead of MTPLX's 12 head chunks. Bitwise
#           == REF2-T below 512 keys, other bits past it; the key buckets and the class re-runs stay. Leap GDNB-2: -0.2 ms
#           per call, 12 calls per check. With REF2_FATT_VERIFY=1 fs4 keeps the bool-mask calls and att covers the rest.
#   inproj  ref2 gdn_inproj (R-68): a GDN layer's qkv | z | b | a as one padded 4-bit matmul (b's bits change). NOT with
#           R57_DNR=1: ref2's _InProjPad concatenates the four modules' packs as stored, and DENSE-N-RAM has replaced
#           qkv / z by 4-D native copies. "1" then skips it (K-83's exact gdn_inproj_x keeps the layer); "inproj" asserts.
# Unset or empty: nothing below runs (build_1105/check_fasttax.py: the AST without these two blocks == the file before).
if os.environ.get("R57_FASTTAX"):
    _ft = os.environ["R57_FASTTAX"].replace("+", ",").split(",")
    _ft = ["att", "inproj"] if _ft == ["1"] else _ft
    assert set(_ft) <= {"att", "inproj"}, f"R57_FASTTAX={os.environ['R57_FASTTAX']}: the pieces are att, inproj (or 1)"
    assert FAST and not os.environ.get("R57_REFTOK"), \
        "R57_FASTTAX is FAST-only (tokens differ from REF2-T past 512 keys): set R57_FAST=<arm> and R57_REFTOK= (empty)"
    assert "att_single" in ref2.CFG and "gdn_inproj" in ref2.CFG, "Leap's ref2 has no att_single / gdn_inproj"
    if "inproj" in _ft and os.environ.get("R57_DNR") == "1":
        assert os.environ["R57_FASTTAX"] == "1", \
            "R57_FASTTAX=inproj cannot run with R57_DNR=1 (ref2's _InProjPad is not DENSE-N-RAM-aware)"
        _ft = [_p_ for _p_ in _ft if _p_ != "inproj"]
        print("[r57] FAST-TAX: gdn_inproj skipped under R57_DNR=1 (K-83's gdn_inproj_x stays)", flush=True)
    if "att" in _ft:
        ref2.CFG["att_single"] = True
    if "inproj" in _ft:
        ref2.CFG["gdn_inproj"] = True
    print(f"[r57] FAST-TAX: {' + '.join(_ft)} on (ref2 att_single {ref2.CFG['att_single']}, gdn_inproj "
          f"{ref2.CFG['gdn_inproj']}, fatt_verify {ref2.CFG.get('fatt_verify')})", flush=True)

HEADC = os.environ.get("R57_HEADC") == "1"              # Leap 1e0676d (BP-7, exact-safe): the drafter's head logits on qmm16c
if HEADC:                                                # (-0.56 ms per propose; falls back for a non-4-bit / native head or
    sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s4/Leap_scripts"))   # > 16 rows). Not yet smoked: next slot.
    import d7b_nax as _D7N
SV = None                                                # SESSION-VOCAB (Leap ~/Leap/scripts/session_vocab.py): one session-id
if os.environ.get("R57_SVH") == "1" or os.environ.get("R57_DSUB") == "1":   # feed for SVH (R57_SVH=1: the screened VERIFIER
    sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s4/Leap_scripts"))                 # head, ref2 CFG screened_head, FAST) and BP-9
    import session_vocab as _SVM                                              # (R57_DSUB=1: the drafter's screened head on
    SV = _SVM.SessionVocab(model.language_model.lm_head, svh=os.environ.get("R57_SVH") == "1",   # qmm16c, exact-safe; it
                           dsub=os.environ.get("R57_DSUB") == "1",                              # supersedes R57_HEADC);
                           check=os.environ.get("R57_SV_CHECK") == "1")                         # R57_SV_CHECK=1: smoke counters
def head2s(hid):
    if SV is not None and SV.dsub:                       # BP-9: the drafter's head over FR-Spec 98,304 + the session's ids
        return SV.head2s(draft, hid)
    lg = _D7N.head_logits(draft, hid) if HEADC else draft.compute_logits(hid)
    B, s = 128, lg.shape
    blk = lg.reshape(*s[:-1], s[-1] // B, B)
    tb = mx.argpartition(blk.max(axis=-1), -K, axis=-1)[..., -K:]
    sel = mx.take_along_axis(blk, mx.broadcast_to(tb[..., None], (*s[:-1], K, B)), axis=-2).reshape(*s[:-1], K * B)
    j = mx.argpartition(sel, -K, axis=-1)[..., -K:]
    c = mx.take_along_axis(tb, j // B, axis=-1) * B + j % B
    return c, mx.take_along_axis(lg, c, axis=-1)

def chain(blk, unary, cand, hp):
    sel = draft.candidate_selector
    pred, path, conf = blk[:, 0], [], []
    for p in range(hp.shape[1]):
        sc = unary[:, p] + mx.sum(sel.predecessor_codebook(pred)[:, None] * hp[:, p, None]
                                  * sel.successor_codebook(cand[:, p]), axis=-1)
        s_ = mx.argmax(sc, axis=-1)
        pred = mx.take_along_axis(cand[:, p], s_[:, None], axis=-1)[:, 0]
        path.append(pred)
        conf.append(mx.take_along_axis(mx.softmax(sc.astype(mx.float32), axis=-1), s_[:, None], axis=-1)[:, 0])
    return mx.stack(path, axis=1), mx.stack(conf, axis=1)

def propose(blk, ctx, dcache):
    hid = draft.hidden_states(blk, ctx, dcache, 1)
    cand, unary = head2s(hid)
    hp = draft.candidate_selector.hidden_projection(hid)
    return _chain_fused(blk, unary, cand, hp, draft.candidate_selector) if K83 else chain(blk, unary, cand, hp)   # R-77

class MTPHead:                                           # the chat switch's drafter: D-20's native MTP chain + D-23c's early exit
    def __init__(self):                                  # (~/TrainLab/sticky/d23c_mtptrunc.py, arm mtp15e, verbatim)
        self.cache = model.make_mtp_cache()
        self.logits = self.h = None

    def append(self, hidden, next_ids):
        """commit pairs (hidden row i, next_ids[i]); the last one runs through mtp_forward for the next draft"""
        if hidden.shape[1] > 1:
            model.mtp_update_cache(hidden[:, :-1], mx.array([next_ids[:-1]]), mtp_cache=self.cache)
        self.logits, self.h = model.mtp_forward(hidden[:, -1:], mx.array([next_ids[-1:]]), mtp_cache=self.cache,
                                                return_hidden=True)

    def draft_cut(self, k, t):
        """evaluate each step and stop at the first one whose running product of top-1 p is < t; returns the leading
        drafts at >= t (the DFlash2 path's cut rule) and every evaluated step's p"""
        l0 = self.logits[0, -1].astype(mx.float32)
        tk, p = mx.argmax(l0), mx.softmax(l0).max()
        mx.eval(tk, p)
        toks, ps, h, run_, n_fw = [int(tk.item())], [float(p.item())], self.h, float(p.item()), 0
        while run_ >= t and len(toks) < k:
            lgx, h = model.mtp_forward(h, mx.array([[toks[-1]]]), mtp_cache=self.cache, return_hidden=True)
            n_fw += 1
            lx = lgx[0, -1].astype(mx.float32)
            tk, p = mx.argmax(lx), mx.softmax(lx).max()
            mx.eval(tk, p)
            toks.append(int(tk.item())); ps.append(float(p.item())); run_ *= ps[-1]
        if n_fw:
            self.cache[0].trim(n_fw)                     # the chain's own entries are not committed history
        return toks[:int((np.cumprod(np.array(ps)) >= t).sum())], ps

_midloop = Q._prefill_midloop_eval_layers
SB_LO = 1 if os.environ.get("R57_SB1") == "1" else 2     # Leap's C0 timeline (11-02): 1-row checks build their graph without the
                                                         # midloop async_eval (warm 48 vs 44 ms at 2 rows); R57_SB1=1 = from 1 row
SBN = int(os.environ.get("R57_SBN", "4") or "4")         # Leap S2 (Sat): submit every SBN layers on SB's 2-16-row checks (4 = today)
def ref_on(on):
    if on:
        ref2.apply()
        Q._prefill_midloop_eval_layers = lambda rows: _midloop(rows) or (SBN if SB_LO <= int(rows) <= 16 else 0)   # SB
    else:
        ref2.remove()
        Q._prefill_midloop_eval_layers = _midloop

def states(cache):
    return [c.state for c in cache if c is not None and hasattr(c, "state")]

A6CAP = {}                                               # A6 pools: the last prefill chunk's routing, layer -> [1, rows, top-k]
def prefill(ids, want_w=False, chunk=1024):
    pf1(True)
    cache, lg, ws = model.make_cache(), None, []
    tap_reset(); TAP["on"] = True
    for a in range(0, len(ids), chunk):
        if ("pools" in A6 and a + chunk >= len(ids)) or ("warm" in A6 and a == 0):   # EvalSuite's block records each
            EM.STATE.capture = {}                        # layer's rank-ordered top-k (warm: from the first chunk on)
        lg = model(mx.array([ids[a:a + chunk]]), cache=cache)
        mx.eval(lg, states(cache), [x[-1] for x in TAP["x"]])
        if want_w:                                       # the chat switch: each chunk's widened stream for the MTP head
            w = TXT._last_widened
            assert w.shape[1] == min(chunk, len(ids) - a), w.shape
            mx.eval(w)
            ws.append((a, w))
    TAP["on"] = False
    pf1(False)
    if EM.STATE.capture is not None and A6 & {"pools", "warm"}:
        A6CAP.clear(); A6CAP.update(EM.STATE.capture)
        EM.STATE.capture = None
    return cache, lg[0, -1], taps(), ws

def run(ids, arm, name=None, drafter="dfl", cpy=False, maxd=0, fset=None, tcut=None):
    tcut = tcut or TRUNC                                 # R57_CHATTRUNC: this prompt's confidence cut
    while os.path.exists(PAUSE):
        time.sleep(5)
    mtp = drafter == "mtp"
    set_arm("slot")                                      # the prefill as shipped in every arm
    ref_on(True)
    p0, g0 = BG.PAUSED["pause_s"], BG.PAUSED["n_gap"]
    mx.reset_peak_memory()                               # C45 memory anatomy: MLX peak in prefill vs decode, cache, footprint
    cg = None
    if cpy and not mtp:                                  # R57_COPY: the prompt's k-grams build beside the prefill
        cg = CGM.CopyGate(list(ids), len(ids), COPYMODE[COPY])
        cg_t = threading.Thread(target=cg.commit, args=(len(ids) - 1,), daemon=True)
        cg_t.start()
    cold_ = bool(FAST and (FASTARMS[FAST].get(fset) or {}).get("cold")) and not mtp
    if cold_:                                            # U3C: the prompt's own footprint per layer, during its prefill
        FG["cap"] = np.zeros((48, 512), bool)
    if SV is not None:                                   # SESSION-VOCAB: this prompt's ids join both subsets (before the prefill:
        SV.new_prompt(ids)                               # a short last prefill chunk takes SVH's <= 16-row path too)
    cache, last, T, ws = prefill(ids, mtp)
    if cold_:
        cap_, FG["cap"] = FG["cap"], None
        warm_ = cap_ if (U3C_LEN and len(ids) >= 128) else (cap_ | (POP45 >= 0.05))
        FG.update(warm=mx.array(warm_), dump=[] if U3C_DUMP else None, chk=0)
    mem = [mx.get_peak_memory() / 1e9, footprint_peak_gb()]
    mx.reset_peak_memory()
    P = len(ids)
    dcache = head = pend = None
    if not mtp:
        dcache = draft.make_cache()
        start = max(0, P - (SWIN - 1))
        for c in dcache:
            c.offset = start
        ctx = T[start:P]
    a = P
    tok = int(mx.argmax(last).item())
    if SV is not None:
        SV.committed([tok])
    set_arm(arm)
    a6 = {}                                              # A6's pre-loop steps (outside the decode clock), in this order
    R_ = R if (cg is not None or (INERT is not None and INERT["rows"] > 16)) else 16                     # R57_ROWS=32 only where the copy runs (MATCH 2.5), 16 elsewhere
    fz = FASTARMS[FAST].get(fset) if FAST else None      # R57_FAST: this prompt set's setting (None = exact acceptance)
    nwp_ = NW and fz is not None                         # R57_NEVERWAIT per prompt set (NewIdeas Fri 15:2x): only sets with a FAST rule
    grams = RXC.prompt_grams(ids) if fz and fz["copy"] else None
    scan = RXC.CodeScan() if fz and fz["loop"] else None
    rgrams = RXC.prompt_grams(ids) if RLX_D > 0 and RLX_COPY else None   # R57_RELAX_GATE copy: its own grams (FAST's stay)
    rst = dict(rows=0, kx=0, k=0, relaxed=0)              # R57_RELAX_DELTA per run: rows checked, kept exact / in all, relaxed
    rlog = RLD.Log(os.environ.get("R57_RELAX_LOG", "")) if RLX_D > 0 else None
    assert not (SPAN and fz and fz["loop"]), "R57_RELAX_SPAN replaces the FAST arm's comment loop: use an arm without it"
    sscan = RXC.CodeScan() if SPAN else None             # R57_RELAX_SPAN: the causal comment tagger, fed the answer's text
    sst = dict(rows=0, kx=0, k=0, relaxed=0, span_rows=0)   # per run: rows checked, kept exact / in all, relaxed, rows open
    slog = RSP.Log(os.environ.get("R57_RELAX_LOG", "")) if SPAN else None
    if sscan is not None:
        sscan.feed(_TKZ.decode([tok]))
    assert not (INERT is not None and fz and fz["loop"]), "R57_INERT replaces the FAST arm's comment loop: use an arm without it"
    iscan = INM.InertScan(RXC) if INERT is not None and not mtp else None   # R57_INERT: the engine's tagger + the docstring
    ist = dict(rows=0, kx=0, k=0, whole=0, ruled=0, com=0, doc=0, open=0)   # state; per run: rows checked, kept exact / in all,
    ilog = INM.Log(os.environ.get("R57_INERT_LOG", "")) if INERT is not None else None   # kept whole / by a rule, per region,
    if iscan is not None:                                # rows open to INERT
        iscan.feed(_TKZ.decode([tok]))
    if scan is not None:
        scan.feed(_TKZ.decode([tok]))
    fst = dict(rows=0, gated=0, kx=0, k=0)               # R57_FAST per run: rows checked / gated, kept exact / relaxed
    PRS.clear(); PRD.update(n=0, L=[], chk=[], row=[], X=[], Y=[])
    cst = dict(pos=0, used=0, chg=0, ok=0, ms=0.0)       # R57_COPY per run: drafted positions checked, copies used / that
    if cg is not None:                                   # changed the draft / accepted, host ms (join + propose + learn)
        t_ = time.perf_counter()
        cg_t.join()
        cg.ids.append(tok)                               # index P: the anchor of the first check
        cst["ms"] += time.perf_counter() - t_; a6["copy_join_ms"] = round(1e3 * cst["ms"], 2)
    if A6 and not mtp:
        if "pools" in A6 and A6CAP:                      # the miss burst's upper-bound fix: this prompt's likely experts in place
            t_, s_, b_ = time.perf_counter(), stats(), disk_read_bytes()
            for sw in SW_:
                c_ = A6CAP.get(sw._bm["L"])
                if sw._bm["full"] or not c_:
                    continue
                u_, n_ = np.unique(np.array(c_[-1][0, -64:]).reshape(-1), return_counts=True)
                BG.ext.ensure(sw._bm["L"], mx.array(u_[np.argsort(-n_, kind="stable")[:A6_POOLS]].astype(np.int32)))
            a6.update(pools_ms=round(1e3 * (time.perf_counter() - t_), 1), pools_miss=float((stats() - s_)[0]),
                      pools_mb=round((disk_read_bytes() - b_) / 1e6, 1))
            A6CAP.clear()
        if "warm" in A6 and A6CAP:                       # PREFILL_WARM: one ensure() per pool, ascending frequency (ties by id)
            t_, s_, b_ = time.perf_counter(), stats(), disk_read_bytes()
            r4 = [i for i, b in enumerate(os.environ.get("BIGMOE_TIER", "")) if b == "4"]   # the ranks the sidecar serves
            for sw in SW_:
                c_, t4 = A6CAP.get(sw._bm["L"]), sw._bm.get("tier") or {}
                if sw._bm["full"] or not c_:
                    continue
                r_ = np.concatenate([np.array(x).reshape(-1, x.shape[-1]) for x in c_])   # [prompt rows, top-k], rank order
                for L_, u_, C_ in ((sw._bm["L"], r_, sw._bm["C"]), (t4.get("side"), r_[:, r4], t4.get("C_l", 0))):
                    if L_ is None or C_ <= 0 or not u_.size:
                        continue
                    e_, n_ = np.unique(u_.reshape(-1), return_counts=True)
                    k_ = C_ - min(int(os.environ.get("BIGMOE_SPARE", "0")), C_ // 4)   # the ext's C_main (ensure aborts past it)
                    BG.ext.ensure(L_, mx.array(e_[np.argsort(n_, kind="stable")][-k_:].astype(np.int32)))
            a6.update(warm_ms=round(1e3 * (time.perf_counter() - t_), 1), warm_miss=float((stats() - s_)[0]),
                      warm_mb=round((disk_read_bytes() - b_) / 1e6, 1))
            A6CAP.clear()
        if "dwin" in A6 and P - 1 > (s0 := max(0, P - SWIN)):   # the drafter's window prefill now (all but the last position):
            t_ = time.perf_counter()                     # the first propose then takes 1 context position + the block. From
            for c in dcache:                             # s0 = start - 1 (if > 0), not start: the rotating buffer (SWIN - 1) must be
                c.offset = s0                            # full, else the 1-position update grows it by SWIN - 1 - offset < 0
            ref2.remove()                                # (a6w_160 crash 11-02 15:34); that update rotates s0 out -> start..P-1
            pth_, _ = propose(mx.array([[tok] + [MASK] * (min(R, 32) - 1)]), T[s0:P - 1][None].astype(mx.bfloat16), dcache)
            mx.eval(pth_)
            if (trim := dcache[0].offset - (P - 1)) > 0:
                DF._trim_recent_cache(dcache, trim)
            ref2.apply()
            ctx = T[P - 1:P]
            a6["dwin_ms"] = round(1e3 * (time.perf_counter() - t_), 1)
        if "dummy" in A6:                                # pre-pay first-use costs: one 16-row check of [tok] x 16, rolled back
            t_, s_ = time.perf_counter(), stats()
            snap_ = CS.snapshot_untrimmable_cache_lazy(cache)
            tap_reset(); TAP["on"] = True
            with ref2.verify_scope(model):
                lg_ = model(mx.array([[tok] * R]), cache=cache)
            TAP["on"] = False
            mx.eval(mx.argmax(lg_[0], axis=-1), taps())
            CS.rollback_after_verify(cache, snap_, R)
            mx.eval(states(cache))
            tap_reset()
            a6.update(dummy_ms=round(1e3 * (time.perf_counter() - t_), 1), dummy_miss=float((stats() - s_)[0]))
            del lg_, snap_
    res0 = None
    if S10B_DIR:
        BS._S10B.drop()                                  # only the live loop's checks + commit replays
        res0 = np.array([BS.ext.lookup(L, list(range(512))) for L in range(NL)], np.int32)   # pools at loop start
    out, cyc, ck_t, kept, sm, wl, cut, drf = [tok], [], [], [], [], [], False, []
    lh_ = []                                             # R57_LOGHASH: per check, the verify logits' hash
    q1c_ = []                                            # per check, R57_QCNT1C: [kept, dropped, groups computed, split drops]
    nw_ = []                                             # per check, R57_NEVERWAIT: ref2.NWCNT summed over the layer calls
    rows_, cb_, bld_, wt_, wsf_ = [], [], [], [], []     # per check: verified rows, SSD bytes (the per-row cost split, 11-01
                                                         # lead), graph-build s, eval-wait s (A6 / K-80's split), warm-split flag
    WSPL = []                                            # this run's warm-split samples
    rb0, sl0 = disk_read_bytes(), stats()                # the whole loop's SSD bytes + misses (M-17's MB per kept token)
    px0 = st.stats()                                     # the look-ahead's / JOINT's counters over the loop
    if STALL:                                            # the prefill's events out; perf_counter -> UPTIME_RAW offset
        BG.ext.trace(); BG.ext.trace_reads(); BG.ext.trace_cov()
        stl_, off_ = [], time.clock_gettime(time.CLOCK_UPTIME_RAW) - time.perf_counter()
    td = c_prev = time.perf_counter()
    hist = 0.0
    if mtp:                                              # the prompt's pairs (W[u], token u + 1), the last one with tok; in the
        ref2.remove()                                    # timed window (D-23d kept its 0.03-0.34 s out), as DFlash2's window
        head, full = MTPHead(), list(ids) + [tok]        # prefill sits in its first propose
        for a_, w in ws:
            head.append(w, full[a_ + 1:a_ + 1 + w.shape[1]])
            mx.eval(head.logits)
        ref2.apply()
        hist = time.perf_counter() - td
    del ws
    while len(out) < CAP and tok not in EOS:
        if el() > HARD:
            cut = True
            break
        c0 = time.perf_counter()
        if out[1:]:
            cyc.append(c0 - c_prev)
        c_prev = c0
        ref2.remove()
        fk_ = sq_ = qcf_ = q1l_ = None                   # R57_FAST: this check's keep mask / U3's sure rows (DFlash2 only)
        rg_ = None                                       # R57_RELAX_DELTA: the rows that may relax (DFlash2 only)
        wsp = WARMSPLIT and not mtp and len(kept) >= 2 and len(kept) % WARMSPLIT == 0   # a warm-split sample (16 rows)
        if mtp:
            if pend is not None:                         # the last check's kept rows -> the head's history
                head.append(*pend)
            dr, _ = head.draft_cut(min(MTP_K, maxd) if maxd else MTP_K, tcut)   # R57_CHATROWS: maxd = rows - 1
        else:
            _d17 = D17 and R_ == 16                      # R57_D17: 17 positions, the 16th draft only gates the last row
            pth, cf = propose(mx.array([[tok] + [MASK] * (R_ if _d17 else min(R_, 32) - 1)]), ctx[None].astype(mx.bfloat16), dcache)
            dr = np.array(pth[0]).tolist()
            cfl = np.array(cf[0]).tolist()
            x16 = c16 = None
            if _d17:                                     # MTP Head's draft17.split17: today's 15-draft block + the extra
                dr, cfl, x16, c16 = D17M.split17(dr, cfl, R_)
            cf_raw = list(cfl)
            if cg is not None:                           # R57_COPY: the gated copy over the whole block, before the cut
                t_ = time.perf_counter()
                d0_ = dr
                dr, cu_, ci_ = cg.propose(d0_, len(cg.ids) - 1)
                if COPY != "g2t":                        # a used copy counts as sure in the running confidence
                    cfl = [1.0 if u_ else c_ for u_, c_ in zip(cu_, cfl)]
                cst["ms"] += time.perf_counter() - t_
                if R_ >= 32 and sum(cu_[:31]) < 24 and not (INERT is not None and INERT["rows"] > 16):           # MATCH 2.5: a 16-row check outside long copies
                    dr, cfl = dr[:15], cfl[:15]
                elif CRUN == 64 and R_ == 64:            # R57_COPYRUN=64 (exact): a long verbatim run goes on past the
                    d0_, dr, cu_, ci_, cfl = INM.copyrun_extend(cg, d0_, dr, cu_, ci_, cfl, R_)   # drafter's 31 picks
            if iscan is not None:                        # R57_INERT: rows in a span are sure in the cut; a longer block
                idx_ = [_TKZ.decode([int(x)]) for x in dr]   # keeps its drafts past the 15th only inside a span (or a copy)
                in_, cfl, _ = INM.pre_block(RXC, iscan, dr, cfl, idx_, INERT, cu_ if cg is not None else None)
                dr, cfl = dr[:in_], cfl[:in_]
            if fz is not None and (fz["thr"] is not None or fz["copy"]):   # R57_FAST: EvalSuite's row gate on the block
                tl_ = out[-RXC.COPY_N:] if len(out) >= RXC.COPY_N else ids[len(out) - RXC.COPY_N:] + out
                cg_ = (cf_raw if cg is None else            # a row whose draft the copy replaced: the drafter's cf is
                       [c_ if d_ == x_ else 0.0 for c_, d_, x_ in zip(cf_raw, d0_, dr)])   # about another token: no trust
                if fz.get("q1u"):                        # U3: the sure rows (cf gate, row 15 never); the wrapper
                    sq_ = np.array([g_ == 1 for g_ in RXC.gate_rows(cg_, tl_, dr, fz["thr"], None)])   # builds the union
                else:                                    # mask per layer
                    fk_ = RXC.keep_mask(RXC.gate_rows(cg_, tl_, dr, fz["thr"], grams))
            if RLX_D > 0 and (RLX_THR is not None or RLX_COPY):   # R57_RELAX_DELTA: the row gate, on the block before the cut
                tlr = out[-RXC.COPY_N:] if len(out) >= RXC.COPY_N else ids[len(out) - RXC.COPY_N:] + out
                cgr = (cf_raw if cg is None else [c_ if d_ == x_ else 0.0 for c_, d_, x_ in zip(cf_raw, d0_, dr)])
                rg_ = RLD.gate_flags(RXC, RLX_THR, RLX_COPY, cgr, tlr, dr, rgrams)
            if arm in ("gate+t", "gate+t+pf") and not wsp:   # Leap's S45c: L1-14 running-confidence truncation
                dr = dr[:int((np.cumprod(np.array(cfl)) >= tcut).sum())]
            if maxd:                                     # R57_CHATROWS on DFlash2: at most maxd drafts
                dr = dr[:maxd]
            if fk_ is not None:                          # the rows this check verifies; the bonus row stays exact (a cut
                fk_ = fk_[:len(dr) + 1].copy(); fk_[len(dr)] = True   # check's last row verifies no draft)
                fst["gated"] += int((~fk_.all(axis=1)).sum())
            if sq_ is not None:
                sq_ = sq_[:len(dr) + 1].copy()
                if D17 and fz.get("qom"):                # R57_D17 + QOM (the map's screen 6): the last row is gated too:
                    if len(dr) == R_ - 1:                # a full block by the 16th draft's cf; a cut block keeps the gate
                        sq_[len(dr)] = c16 is not None and c16 >= fz["thr"]   # of the draft past the cut (gate_rows')
                else:
                    sq_[len(dr)] = False                 # today: the last row verifies no draft -> exact
                if QSPLIT is not None:                   # R57_QOM_SPLIT (ported Fri 10:0x): each row's draft cf, aligned with sq_
                    qcf_ = np.array((list(cg_) + [c16 if c16 is not None else 1.0])[:len(dr) + 1], dtype=np.float32)
                fst["gated"] += int(sq_.sum())
                if not sq_.any() and not U3FORCE:
                    sq_ = None
            if (trim := dcache[0].offset - a) > 0:
                DF._trim_recent_cache(dcache, trim)
        ref2.apply()
        s0, b0_ = stats(), disk_read_bytes()
        c1 = time.perf_counter()
        drf.append(c1 - c0)                              # the drafter's share of the cycle: the SSD idles (R-70D's window)
        if "rep" in A6 and not kept and not mtp:         # A6 rep: copy 1 of the first check, rolled back; the real check
            snap_ = CS.snapshot_untrimmable_cache_lazy(cache)   # below is copy 2 (same rows, its experts now resident)
            tap_reset(); TAP["on"] = True
            with ref2.verify_scope(model):
                lg_ = model(mx.array([[tok] + dr]), cache=cache)
            tb_ = time.perf_counter()
            TAP["on"] = False
            mx.eval(mx.argmax(lg_[0], axis=-1), taps())
            te_ = time.perf_counter()
            CS.rollback_after_verify(cache, snap_, len(dr) + 1)
            mx.eval(states(cache))
            sd_ = stats() - s0
            a6["rep1"] = [round(1e3 * (tb_ - c1), 2), round(1e3 * (te_ - tb_), 2), int(sd_[0]), round(1e3 * sd_[1], 2),
                          round(1e3 * sd_[2], 2)]           # build ms, eval wait ms, misses, demand SSD ms, gate service ms
            del lg_, snap_
            s0, b0_ = stats(), disk_read_bytes()
            c1 = time.perf_counter()
        if wsp:                                          # the warm split: the identical 16-row input cold, then warm at 1 / 4 /
            inp_, smp = mx.array([[tok] + dr]), dict(n=len(kept), rows=len(dr) + 1, warm=[])   # 8 / 16 rows, each rolled back
            for n_ in (0, 1, 4, 8, 16):                  # n_ 0 = the cold 16-row pass
                w_ = n_ or len(dr) + 1
                snap_ = CS.snapshot_untrimmable_cache_lazy(cache)
                s_, t_ = stats(), time.perf_counter()
                FG.update(on=fk_ is not None or sq_ is not None, keep=mx.array(fk_[:w_]) if fk_ is not None else None,
                          sure=mx.array(sq_[:w_]) if sq_ is not None else None, sure_np=sq_[:w_] if sq_ is not None else None,
                          t3=bool(fz and fz.get("t3")), qom=bool(fz and fz.get("qom")) and sq_ is not None,   # R57
                          qcf=mx.array(qcf_[:w_]) if qcf_ is not None else None)   # R57_FAST's gate in
                try:                                     # the warm split too: a gated arm's warm per-row cost (Fable C's
                    if nwp_:                             # R57_NEVERWAIT: on for this verify pass only (prefill stays as today)
                        ref2.NW_RESET(); ref2.NW_ON = True
                    with ref2.verify_scope(model):       # gated expert term, the orchestrator 23:2x)
                        lg_ = model(inp_[:, :w_], cache=cache)
                finally:
                    if nwp_:
                        ref2.NW_ON = False
                    FG.update(on=False, keep=None, sure=None, sure_np=None, t3=False, qom=False, qcf=None)
                mx.eval(mx.argmax(lg_[0], axis=-1))
                d_ = stats() - s_
                rec_ = [w_, round(1e3 * (time.perf_counter() - t_), 2), int(d_[0]), round(1e3 * d_[1], 2), round(1e3 * d_[2], 2)]
                CS.rollback_after_verify(cache, snap_, w_)
                mx.eval(states(cache))
                if n_:
                    smp["warm"].append(rec_)             # [rows, ms, misses, demand SSD ms, gate service ms]
                else:
                    smp["cold"] = rec_
                del lg_, snap_
            WSPL.append(smp)
            s0, b0_ = stats(), disk_read_bytes()         # the real check below (warm now) commits as always
            c1 = time.perf_counter()
        if CPROBE is not None and not mtp:               # R57_COPYPROBE: nothing is committed (every pass is rolled back). The
            _cpd = CPROBE.maybe(globals(), locals())     # probe's time leaves the cycle / decode clocks; the real check's
            c0 += _cpd; c_prev += _cpd; td += _cpd       # counters start after it
            s0, b0_ = stats(), disk_read_bytes()
            c1 = time.perf_counter()
        wsf_.append(int(bool(wsp)))
        snap = CS.snapshot_untrimmable_cache_lazy(cache)
        if not mtp:
            tap_reset(); TAP["on"] = True
        FG.update(on=fk_ is not None or sq_ is not None, keep=mx.array(fk_) if fk_ is not None else None,   # R57_FAST
                  sure=mx.array(sq_) if sq_ is not None else None, sure_np=sq_, t3=bool(fz and fz.get("t3")),   # (in the graph)
                  cold=bool(fz and fz.get("cold")) and sq_ is not None, qom=bool(fz and fz.get("qom")) and sq_ is not None,
                  qcf=mx.array(qcf_) if qcf_ is not None else None)
        if PREROUTE and not mtp:                         # R57_PREROUTE: this check's capture; EvalSuite's block records
            PR.update(on=True, cur=None, y={}, blk={}, cap={}, dump=[] if PRDUMP else None)   # each layer's true top-10
            EM.STATE.capture = {}
        nwl_ = None
        if nwp_:                                         # R57_NEVERWAIT: this real check's late groups, counted by ref2
            ref2.NW_RESET(); ref2.NWCNT = []; ref2.NW_ON = True
        if QC1C:
            ref2.QCNT_1C = []                            # R57_QCNT1C (ported Fri 10:0x): this real check's one-call counters
        try:
            with ref2.verify_scope(model):
                lg = model(mx.array([[tok] + dr]), cache=cache)
        finally:
            FG.update(on=False, keep=None, sure=None, sure_np=None, t3=False, cold=False, qom=False, qcf=None,
                      chk=FG["chk"] + 1)
            PR["on"] = False
            if QC1C:
                q1l_, ref2.QCNT_1C = ref2.QCNT_1C, None
            if nwp_:
                ref2.NW_ON = False
                nwl_, ref2.NWCNT = ref2.NWCNT, None
        cb = time.perf_counter()                         # graph built: the sync-free gate waits on each layer's router ids
        TAP["on"] = False
        am = mx.argmax(lg[0], axis=-1)
        rx2 = bool(RLXV2 and (RLX_D > 0 or SPAN) and dr)    # R57_RELAX_V2: dlm / rank ride in the argmax readback
        if rx2:
            l0v = lg[0, :len(dr)].astype(mx.float32)
            ldv = mx.take_along_axis(l0v, mx.array(dr, dtype=mx.int32)[:, None], axis=-1)
            dlv_, rkv_ = ldv[:, 0] - l0v.max(axis=-1), (l0v > ldv).sum(axis=-1)
        if mtp:                                          # the MTP head's history needs the verified rows' widened stream
            wv = TXT._last_widened
            mx.eval(am, wv[:, :len(dr) + 1], *((dlv_, rkv_) if rx2 else ()))
        else:
            tp = taps()
            mx.eval(am, tp, *((dlv_, rkv_) if rx2 else ()))
        if LOGHASH:                                      # R57_LOGHASH: this check's verify logits, bit for bit
            import hashlib as _hlg
            lh_.append(_hlg.sha1(np.array(lg[0].astype(mx.float32)).tobytes()).hexdigest()[:16])
        ce = time.perf_counter()
        bld_.append(cb - c1); wt_.append(ce - cb)        # K-80's split: graph build vs the eval wait (+ the commit after)
        am = am.tolist()
        if rx2:
            dlm2, drk2 = dlv_.tolist(), rkv_.tolist()
        if PREROUTE and EM.STATE.capture is not None:   # R57_PREROUTE: score this check's pre-routes
            pr_eval(EM.STATE.capture); EM.STATE.capture = None
        k = 0
        while k < len(dr) and dr[k] == am[k]:
            k += 1
        kx0 = k                                          # R57_RELAX_DELTA: the exact rule's kept drafts
        if fz is not None:
            fst["rows"] += len(dr) + 1; fst["kx"] += k + 1
        if fz is not None and (fz["rules"] or fz["loop"]) and dr:   # R57_FAST: EvalSuite's relaxed rules (kept_rows)
            l0 = lg[0, :len(dr)].astype(mx.float32)
            ld = mx.take_along_axis(l0, mx.array(dr, dtype=mx.int32)[:, None], axis=-1)
            dl_, rk_ = ld[:, 0] - l0.max(axis=-1), (l0 > ld).sum(axis=-1)   # logit(d) - max logit; the draft's rank
            mx.eval(dl_, rk_)
            dlm, drk = dl_.tolist(), rk_.tolist()
            cm_ = set()
            if scan is not None:                         # the comment rule: the scan state before each draft
                dtx, pre, s_ = [_TKZ.decode([int(x)]) for x in dr], [], scan.copy()
                for x in dtx:
                    pre.append((s_.in_comment(), s_.ctext)); s_.feed(x)
                cm_ = set(RXC.comment_rows(pre, dtx, [_TKZ.decode([int(x)]) for x in am[:len(dr)]]))
            k = 0
            while k < len(dr) and (dr[k] == am[k] or any(RXC.rule_ok(r_, dlm[k], drk[k]) for r_ in
                                                          fz["rules"] + ((fz["loop"],) if k in cm_ else ()))):
                k += 1
        if RLX_D > 0 and dr and (rg_ is not None or (RLX_THR is None and not RLX_COPY)):   # R57_RELAX_DELTA
            if rx2 and (fz is None or not (fz["rules"] or fz["loop"])):   # R57_RELAX_V2: from the one readback
                dlm, drk, cm_ = dlm2, drk2, set()
            elif fz is None or not (fz["rules"] or fz["loop"]):     # dlm / rank (the R57_FAST block computed them otherwise)
                l0 = lg[0, :len(dr)].astype(mx.float32)
                ld = mx.take_along_axis(l0, mx.array(dr, dtype=mx.int32)[:, None], axis=-1)
                dl_, rk_ = ld[:, 0] - l0.max(axis=-1), (l0 > ld).sum(axis=-1)
                mx.eval(dl_, rk_)
                dlm, drk = dl_.tolist(), rk_.tolist()
                cm_ = set()
            fok = None if fz is None or not (fz["rules"] or fz["loop"]) else (
                lambda j: any(RXC.rule_ok(r_, dlm[j], drk[j]) for r_ in fz["rules"] + ((fz["loop"],) if j in cm_ else ())))
            k, rl_ = RLD.relaxed_k(dr, am, dlm, RLX_D, rg_, fok)  # the exact rule, the arm's rules, or R1(delta) where gated
            rst["rows"] += len(dr) + 1; rst["kx"] += kx0 + 1; rst["k"] += k + 1; rst["relaxed"] += len(rl_)
            for j in rl_:
                rlog.write(prompt=name, check=FG["chk"], row=j, draft=int(dr[j]), argmax=int(am[j]), dlm=round(dlm[j], 4),
                           rank=int(drk[j]), cf=None if mtp else round(float(cf_raw[j]), 4),
                           gate="all" if rg_ is None else "+".join(x for x, on in (("cf", RLX_THR is not None), ("copy", RLX_COPY)) if on))
        if SPAN and dr:                                  # R57_RELAX_SPAN: rule C on the comment rows (FULL R2.1)
            if rx2:                                      # R57_RELAX_V2: dlm / rank from the one readback, cached texts,
                sdlm, sdrk = dlm2, drk2                  # the tagger only where a comment is possible
                dtx = [_tdec(x) for x in dr]
                if RSP.may_touch(sscan, dtx):
                    atx = [_tdec(x) for x in am[:len(dr)]]
                    srows, spre = RSP.span_rows(RXC, sscan, dtx, atx)
                else:
                    srows, spre, atx = set(), None, None
            else:
                l0s = lg[0, :len(dr)].astype(mx.float32)
                lds = mx.take_along_axis(l0s, mx.array(dr, dtype=mx.int32)[:, None], axis=-1)
                sd_, sr_ = lds[:, 0] - l0s.max(axis=-1), (l0s > lds).sum(axis=-1)   # logit(d) - max logit; the draft's rank
                mx.eval(sd_, sr_)
                sdlm, sdrk = sd_.tolist(), sr_.tolist()
                dtx = [_TKZ.decode([int(x)]) for x in dr]
                atx = [_TKZ.decode([int(x)]) for x in am[:len(dr)]]
                srows, spre = RSP.span_rows(RXC, sscan, dtx, atx)   # conditions 1-4 (a copy of the tagger is fed the drafts)
            rlx_on = RLX_D > 0 and (rg_ is not None or (RLX_THR is None and not RLX_COPY))   # the combined arm: v1's R1 gate
            rlx_ok = (lambda j: (rg_ is None or bool(rg_[j])) and sdlm[j] >= -RLX_D) if rlx_on else None
            fok_s = None if (fz is None or not fz["rules"]) and rlx_ok is None else (
                lambda j: (fz is not None and any(RXC.rule_ok(r_, sdlm[j], sdrk[j]) for r_ in fz["rules"]))
                or (rlx_ok is not None and rlx_ok(j)))
            k, sl_ = RSP.span_k(dr, am, sdlm, sdrk, srows, SPAN_RULE, fok_s)   # exact, the arm's rules, or rule C
            sst["rows"] += len(dr) + 1; sst["kx"] += kx0 + 1; sst["k"] += k + 1; sst["relaxed"] += len(sl_)
            sst["span_rows"] += len(srows)
            for j in sl_:
                slog.write(prompt=name, check=FG["chk"], row=j, region="comment", ctext=spre[j][1][-80:], draft=int(dr[j]),
                           argmax=int(am[j]), dtext=dtx[j], atext=atx[j], rank=int(sdrk[j]), dlm=round(sdlm[j], 4),
                           cf=None if mtp else round(float(cf_raw[j]), 4))
        if iscan is not None and dr:                     # R57_INERT: the rows in comments / docstrings (inert_span.py)
            l0i = lg[0, :len(dr)].astype(mx.float32)
            ldi = mx.take_along_axis(l0i, mx.array(dr, dtype=mx.int32)[:, None], axis=-1)
            id_, ir_ = ldi[:, 0] - l0i.max(axis=-1), (l0i > ldi).sum(axis=-1)   # logit(d) - max logit; the draft's rank
            mx.eval(id_, ir_)
            idlm, idrk = id_.tolist(), ir_.tolist()
            dtx = [_TKZ.decode([int(x)]) for x in dr]
            atx = [_TKZ.decode([int(x)]) for x in am[:len(dr)]]
            icls = INM.classify(RXC, iscan, dtx, atx, INERT)   # the scan before each draft (a copy is fed the drafts)
            fok_i = None if fz is None or not fz["rules"] else (
                lambda j: any(RXC.rule_ok(r_, idlm[j], idrk[j]) for r_ in fz["rules"]))
            k, il_ = INM.inert_k(dr, am, idlm, idrk, icls, fok_i)   # exact, the arm's rules, or INERT
            ist["rows"] += len(dr) + 1; ist["kx"] += kx0 + 1; ist["k"] += k + 1
            ist["open"] += sum(1 for c_ in icls if c_[1] is not None)
            for j, kind_, how_ in il_:
                ist["whole" if how_ == "whole" else "ruled"] += 1; ist["com" if kind_ == "comment" else "doc"] += 1
                ilog.write(prompt=name, check=FG["chk"], row=j, region=kind_, mode=how_, ctext=icls[j][2][-80:],
                           draft=int(dr[j]), argmax=int(am[j]), dtext=dtx[j], atext=atx[j], rank=int(idrk[j]),
                           dlm=round(idlm[j], 4), cf=round(float(cf_raw[j]), 4) if j < len(cf_raw) else None)
        if fz is not None:
            fst["k"] += k + 1
        model.commit_verified_window(cache, snap.states, keep_tokens=k + 1, verified_tokens=len(dr) + 1)
        mx.async_eval(states(cache))                     # SB: the commit replay submitted alone
        c2 = time.perf_counter()
        sm.append(stats() - s0)
        if QC1C:
            q1c_.append([int(x_) for x_ in np.array(mx.stack(q1l_).sum(axis=0))] if q1l_ else [0, 0, 0, 0])
        if nwp_:                                         # [late groups, late uses, late mass, L47 drops, L47 rank-0 stalls, forced]
            nw_.append([float(x_) for x_ in np.array(mx.stack(nwl_).sum(axis=0))] if nwl_ else [0.0] * 8)   # NW v3: 8 fields (Leap dcd5f4f8)
        rows_.append(len(dr) + 1); cb_.append(disk_read_bytes() - b0_)
        ck_t.append(c2 - c1); kept.append(k + 1); wl.append(c2 - c0)
        if STALL:
            stl_.append([round(x + off_, 6) for x in (c0, c1, cb, ce, c2)] + [len(dr) + 1, k])
        new = dr[:k] + [am[k]]
        for j, x in enumerate(new):
            if x in EOS:
                new = new[:j + 1]
                break
        out += new
        if SV is not None:                               # SESSION-VOCAB: the committed ids (kept drafts + the target's token)
            SV.committed(new)
        if scan is not None:
            scan.feed("".join(_TKZ.decode([int(x)]) for x in new))
        if sscan is not None:                            # R57_RELAX_SPAN: the committed tokens join the tagger's text
            sscan.feed("".join((_tdec(x) if RLXV2 else _TKZ.decode([int(x)])) for x in new))
        if iscan is not None:                            # R57_INERT: the committed tokens join the tagger's text
            iscan.feed("".join(_TKZ.decode([int(x)]) for x in new))
        if cg is not None:                               # R57_COPY: the check labels the duel; the kept tokens join the window
            t_ = time.perf_counter()
            cg.learn(d0_, ci_, dr[:k] + [am[k]], k)         # the committed truth (== am[:k + 1] unless R57_FAST relaxed it)
            cg.ids.extend(new)
            n_ = len(dr)
            cst["pos"] += n_; cst["used"] += sum(cu_[:n_]); cst["ok"] += sum(cu_[:k])
            cst["chg"] += sum(1 for p_ in range(n_) if cu_[p_] and d0_[p_] != dr[p_])
            cst["ms"] += time.perf_counter() - t_
        if mtp:
            pend = (wv[:, :len(new)], new)               # rows 0..len(new)-1 hold the pairs (W[a + i], new[i])
        else:
            ctx = tp[:k + 1]
        tok, a = out[-1], a + k + 1
    mx.eval(states(cache))
    dec = time.perf_counter() - td
    stall = dict(chk=stl_, tr=BG.ext.trace(), rd=BG.ext.trace_reads(), cov=BG.ext.trace_cov()) if STALL else None
    rb, ml, kt = disk_read_bytes() - rb0, float((stats() - sl0)[0]), max(sum(kept), 1)
    px1 = st.stats()
    pfx = {k_: px1.get(k_, 0) - px0.get(k_, 0) for k_ in ("pf_issued", "pf_used", "pf_victims", "joint_reads", "joint_paired",
                                                       "ka_reads", "bursts", "burst_gap025", "burst_gap1",
                                                       "vt_hits", "vt_ins", "vt_purged", "vt_drops",
                                                       "fc_issued", "fc_landed", "fc_hits", "fc_late", "fc_drops", "fc_skip",
                                                       "fc_busy_bursts", "fc_busy_n", "fc_wait_s",   # fc_*: lever 2 (build_l2)
                                                       "qom_skip", "qom_opt")}   # QOM (build_qom)
    pfx["vt_used_gb"] = px1.get("vt_used_bytes", 0) / 1e9   # the tier's filled part at the run's end
    if VTIER and "vt_hits_layer" in px1:                 # per ext layer (NewIdeas 15:46: its replay of the victim policy),
        pfx["vt_hits_layer"] = [a_ - b_ for a_, b_ in zip(px1["vt_hits_layer"], px0["vt_hits_layer"])]   # this run's
        pfx["vt_ins_layer"] = [a_ - b_ for a_, b_ in zip(px1["vt_ins_layer"], px0["vt_ins_layer"])]
        pfx["vt_laymap"] = VT_MAP                         # model layer -> [base ext id, sidecar ext id]
    mem += [mx.get_peak_memory() / 1e9, mx.get_active_memory() / 1e9, mx.get_cache_memory() / 1e9, footprint_now_gb(),
            footprint_peak_gb()]
    if S10B_DIR and name:
        np.savez(os.path.join(S10B_DIR, name + ".npz"), **BS._S10B.flush(), tokens=np.array(out, np.int32), res0=res0)
    bad = ""
    try:
        st.gate.check()
    except RuntimeError as e:
        bad = f" INVALID ({e})"
    if BG.PAUSED["pause_s"] - p0 > 0.5 or BG.PAUSED["n_gap"] > g0:
        bad += f" PAUSED ({BG.PAUSED['pause_s'] - p0:.1f} s waited, {BG.PAUSED['n_gap'] - g0} gaps)"
    set_arm("slot")
    ref_on(False)
    del cache, dcache, head, pend
    if "keep" not in A6:                                 # A6 keep: MLX's buffer cache stays warm for the next request
        mx.clear_cache()
    sm = np.array(sm)
    if sm.ndim != 2:                                     # Fri 09:0x (EvalSuite): a run cut before its first check (HARD) has no
        sm = np.zeros((0, 6))                            # per-check stats; sm[:, 0] on a 1-D array raised IndexError
    if cold_ and U3C_DUMP and FG["dump"]:               # U3C: per (check, layer) ids / sure / keep, -1 / False padded to 17 rows
        os.makedirs(U3C_DUMP, exist_ok=True)
        recs = [(c_, L_, np.array(i_), np.array(s_), np.array(k_)) for c_, L_, i_, s_, k_ in FG["dump"]]
        n_ = len(recs); RW = max(r_[2].shape[0] for r_ in recs)
        I_ = np.full((n_, RW, 10), -1, np.int16); S_ = np.zeros((n_, RW), bool); K2 = np.zeros((n_, RW, 10), bool)
        for j_, (c_, L_, i_, s_, k_) in enumerate(recs):
            I_[j_, :i_.shape[0]] = i_; S_[j_, :s_.shape[0]] = s_; K2[j_, :k_.shape[0]] = k_
        np.savez(os.path.join(U3C_DUMP, (name or "p").replace("/", "_") + ".npz"), check=np.array([r_[0] for r_ in recs]),
                 layer=np.array([r_[1] for r_ in recs]), rows=np.array([r_[2].shape[0] for r_ in recs]), ids=I_, sure=S_,
                 keep=K2, warm=np.array(FG["warm"]), prompt_len=len(ids), len_rule=U3C_LEN)
        FG["dump"] = None
    return dict(lh=lh_, out=out[:CAP], n=min(len(out), CAP), tps=min(len(out), CAP) / dec, drafter=drafter, hist=hist, a6=a6,
                rlx=rst if RLX_D > 0 else None,      # R57_RELAX_DELTA: rows checked, kept exact / in all, relaxed accepts
                spn=sst if SPAN else None,           # R57_RELAX_SPAN: rows checked, kept exact / in all, relaxed, span rows
                inert=dict(ist, arm=INERT["name"]) if INERT is not None else None,   # R57_INERT: rows checked, kept exact / in
                                                     # all, kept whole / by a rule, per region, rows open
                cyc=1e3 * med(cyc), ck=1e3 * med(ck_t),
                kept=float(np.mean(kept)), checks=len(kept), miss=float(np.median(sm[:, 0])), io=1e3 * float(np.median(sm[:, 1])),
                svc=1e3 * float(np.median(sm[:, 2])), slow=float(np.median(sm[:, 3])), drf=1e3 * med(drf), mem=mem,
                rb=rb, ml=ml, kt=kt, mb=rb / 1e6 / kt, mpk=ml / kt,   # per kept token: SSD MB and misses, the whole loop
                pc=dict(rows=rows_, kept=kept, ck=[1e3 * x for x in ck_t], drf=[1e3 * x for x in drf[:len(ck_t)]],
                        miss=sm[:, 0].tolist(), io=(1e3 * sm[:, 1]).tolist(), svc=(1e3 * sm[:, 2]).tolist(), mb=[x / 1e6 for x in cb_],
                        nw=nw_, q1c=q1c_,        # R57_NEVERWAIT / R57_QCNT1C: per-check counters (empty when off)
                        uses=sm[:, 4].tolist(), stl=sm[:, 5].tolist(),   # C3-R8: expert groups and stalled layer calls per check
                        bld=[1e3 * x for x in bld_], wt=[1e3 * x for x in wt_], ws=wsf_), wspl=WSPL,
                tpsw=sum(kept[2:]) / max(sum(wl[2:]), 1e-9), cut=cut, bad=bad, stall=stall,   # tpsw: steady state, checks >= 2
                copy=dict(cst, ms=round(1e3 * cst["ms"], 2)) if cg is not None else None,
                fast=dict(fst, arm=FAST) if fz is not None else None, pf=pfx,
                preroute={k_: list(v_) for k_, v_ in PRS.items()} if PREROUTE else None)

REFTOK, REFARM = {}, None                                # R57_REFTOK=<tokens.json>[:arm]: every run vs those tokens
if os.environ.get("R57_REFTOK"):
    _f, _, REFARM = os.environ["R57_REFTOK"].partition(":")
    REFTOK, REFARM = json.load(open(os.path.expanduser(_f))), REFARM or "gate+t"
REFROWS = {d["key"]: d for d in map(json.loads, open(os.path.expanduser("~/EvalSuite/results/ref2/decode.jsonl")))}
same = lambda a, b: next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
RES, ok_all, valid, TOK = {}, True, True, {}
PC = []                                                  # per-check rows / ms / misses / MB of every run -> {TAG}_checks.json
if S10B_DIR:                                             # every layer's router, dequantized (Fit12's lead-1 forecast at every L)
    W = {}
    for T_ in range(1, NL):
        g = lay[T_].mlp.gate
        w = (mx.dequantize(g.weight, g.scales, getattr(g, "biases", None), g.group_size, g.bits,
                           **({"mode": g.mode} if hasattr(g, "mode") else {})) if hasattr(g, "scales") else g.weight)
        W[f"w{T_}"] = np.array(w.astype(mx.float16))
    np.savez(os.path.join(S10B_DIR, "routers.npz"), **W)
    say(f"  routers.npz: {len(W)} layers, {sum(v.nbytes for v in W.values()) / 1e6:.0f} MB")
if os.environ.get("R57_WARM") == "1":                    # warm-up, discarded (NewIdeas: S45-T's method)
    it = next(x for x in ED.load("dec_math") if x["key"] == "gsm8k/708")
    CAP, cap0 = 64, CAP
    prior_for(it["key"])
    for dr_ in (("dfl", "mtp") if CHATSW else ("dfl",)) + (("d7b",) if DRAFT4 is not None else ()):   # the chat switch warms
        draft = DRAFT4 if dr_ == "d7b" else DRAFT16      # its MTP chain too (D-23d: one per arm); D7B=code its 4-bit drafter
        r = run(list(it["prompt_ids"]), ARMS_[0], drafter="dfl" if dr_ == "d7b" else dr_)
        say(f"  warm-up (discarded): gsm8k/708 [{ARMS_[0]}, {dr_}] {r['n']} tokens, {r['tps']:.1f} tok/s{r['bad']}; MLX peak "
            f"prefill {r['mem'][0]:.1f} / decode {r['mem'][2]:.1f} GB, lifetime peak {r['mem'][1]:.1f} -> {r['mem'][6]:.1f} GB")
    draft = DRAFT16
    CAP = cap0
RUNS = []                                                # per run: type, drafter, tok/s, kept, ms, misses, MB -> {TAG}_runs.json
WSALL = []                                               # warm-split samples of every run -> {TAG}_warmsplit.json
STL = []                                                 # R57_STALL: every run's decode events -> {TAG}_stall.json
PRALL = []                                               # R57_PREROUTE: every run's samples -> {TAG}_preroute.json
LAYMAP = [[sw._bm["L"], (sw._bm.get("tier") or {}).get("side")] for sw in SW_] if STALL else []   # ext ids: base, sidecar
for n_p, (s, key) in enumerate(PICK):
    it = next(x for x in ED.load("dec_" + s) if x["key"] == key)
    ids, want = list(it["prompt_ids"]), REFROWS.get(key, {}).get("completion_ids", [])
    if NOCOM and s == "humaneval":                       # R57_NOCOM: before the user turn's <|im_end|> (the prompt's last one)
        _j = len(ids) - 1 - ids[::-1].index(NOCOM_END)
        ids, want = ids[:_j] + NOCOM_IDS + ids[_j:], []  # the stored REF2 row belongs to the unchanged prompt
    typ = rule(it) if (CHATSW or COPY or DRAFT4 is not None or CHATROWS or CHATTRUNC or FAST) else ""
    drafter = "mtp" if CHATSW and typ == "chat" else "dfl"   # the chat switch; without R57_CHATSW every prompt drafts on DFlash2
    draft = DRAFT4 if DRAFT4 is not None and typ == "code" else DRAFT16   # D7B=code: code prompts draft on the 4-bit drafter
    cpy = bool(COPY) and typ != "math"                   # R57_COPY: off on math (GATE.md 3)
    cf = corr_for(key)
    pk = prior_for(key)
    if pk:
        say(f"  {key}: eviction prior {pk} x {os.environ.get('BIGMOE_EVICT_PRIOR')} (held out)")
    if cf:
        say(f"  {key}: forecast correction {cf} (held out)")
    arms = (ARMS_ + ARMS_[::-1]) if REPS == 2 else ARMS_
    arms = arms if n_p % 2 == 0 else arms[::-1]
    for arm in arms:
        if el() > HARD:
            say(f"  time budget: stopped before {key} [{arm}]")
            break
        r = run(ids, arm, f"{key.replace('/', '_').replace('.json', '')}_{arm}_{len(TOK.get(key, {}).get(arm, []))}", drafter,
                cpy, CHATROWS - 1 if CHATROWS and typ == "chat" else 0, s, CHATTRUNC if CHATTRUNC and typ == "chat" else None)
        PC.append(dict(key=key, arm=arm, paused="PAUSED" in r["bad"], **r["pc"]))
        if r["stall"]:
            STL.append(dict(key=key, arm=arm, paused="PAUSED" in r["bad"], **r["stall"]))
        if r["preroute"]:
            PRALL.append(dict(key=key, arm=arm, typ=typ, samples=r["preroute"]))
            say("    pre-route recall@10 (per row / of the union): " + ", ".join(
                f"{nm} {np.mean([x[2] for x in v]):.3f} / {np.mean([x[3] for x in v]):.3f}" for nm, v in r["preroute"].items()))
        if PRDUMP and PRD["X"]:                          # R57_PR_DUMP: this run's rows -> data/r57/pr_dump/{TAG}_{run}.npz
            os.makedirs(os.path.join(OUT, "pr_dump"), exist_ok=True)
            fn_ = os.path.join(OUT, "pr_dump", f"{TAG}_{len(PRALL) - 1:02d}.npz")
            np.savez(fn_, key=key, arm=arm, typ=typ, **{k_: np.concatenate(PRD[k_]) for k_ in ("L", "chk", "row", "X", "Y")})
            say(f"    pr dump: {sum(len(x) for x in PRD['X'])} rows x layers, {PRD['n']} checks -> {fn_}")
            PRD.update(n=0, L=[], chk=[], row=[], X=[], Y=[])
        if r["wspl"]:
            WSALL.append(dict(key=key, arm=arm, samples=r["wspl"]))
        RUNS.append(dict(key=key, arm=arm, type=typ, bad=r["bad"], cut=r["cut"], P=len(ids),
                         **{k_: r[k_] for k_ in ("drafter", "hist", "a6", "n", "tps", "tpsw", "kept", "checks", "cyc", "ck",
                                                 "drf", "miss", "io", "mb", "mpk", "mem", "copy", "fast", "pf", "lh", "inert")},
                         d7b=draft is DRAFT4, rlx=r.get("rlx"), spn=r.get("spn")))   # Thu 18:5x / Fri 10:0x: relaxed / span / INERT
                                                         # stats were computed but never written (the pre-flight's MISSING)
        m = min(len(want), r["n"])
        agree = same(r["out"], want)
        ok = (agree >= m or TIER) and not r["cut"]          # REF2-T is its own reference (tokens compared across runs)
        TOK.setdefault(key, {}).setdefault(arm, []).append(r["out"])
        try:                                             # Fri 09:0x (EvalSuite): written after every run too, so a later crash
            json.dump(TOK, open(os.path.join(OUT, f"{TAG}_tokens.json"), "w"))   # keeps the finished keys' tokens
        except OSError:
            pass
        note = ""
        if REFTOK:                                       # REF2-T's own reference (R57_REFTOK, e.g. s45t_B gate+t)
            rt_ = REFTOK.get(key, {}).get(REFARM, [None])[0]
            a_t, m_t = (same(r["out"], rt_), min(len(rt_), r["n"])) if rt_ is not None else (0, 1)
            ok = ok and a_t >= m_t
            note = f"; tokens == REFTOK {a_t} / {m_t}"
        ok_all &= ok
        valid &= not r["bad"]
        if "PAUSED" not in r["bad"]:                     # a paused run's clock is wrong: out of the aggregates
            RES.setdefault(key, {}).setdefault(arm, []).append(r)
        sw_ = (f" [{typ}: {drafter}" + (f", history {r['hist']:.2f} s" if drafter == "mtp" else "") + "]") if CHATSW else ""
        say(f"  {key} [{arm}]{sw_}: {r['n']} tokens in {r['checks']} checks (kept {r['kept']:.2f}), {r['tps']:.1f} tok/s "
            f"(steady state, checks >= 2: {r['tpsw']:.1f}), median "
            f"cycle {r['cyc']:.1f} ms, check + commit {r['ck']:.1f} ms; per check (medians) {r['miss']:.0f} misses, demand "
            f"SSD {r['io']:.1f} ms, gate service {r['svc']:.1f} ms ({r['slow']:.0f} calls with misses), draft {r['drf']:.1f} "
            f"ms; per kept token SSD {r['mb']:.1f} MB, {r['mpk']:.1f} misses; tokens == REF2 row {agree} / {m}{note}"
            f"{'; cut' if r['cut'] else ''}{r['bad']}"
            + (f"; copy {COPY}: used {r['copy']['used']} of {r['copy']['pos']} drafted positions ({r['copy']['chg']} changed "
               f"the draft, {r['copy']['ok']} accepted), host {r['copy']['ms']:.1f} ms" if r["copy"] else "")
            + ("; drafter D7b (4-bit)" if draft is DRAFT4 else "")
            + (f"; FAST {FAST}: {r['fast']['gated']} of {r['fast']['rows']} rows gated, kept {r['fast']['k']} vs exact "
               f"{r['fast']['kx']}" if r["fast"] else "")
            + (f"; look-ahead: {r['pf']['pf_issued']} reads issued, {r['pf']['pf_used']} used ({r['pf']['pf_victims']} victims)"
               if r["pf"]["pf_issued"] else "") + (f"; JOINT reads {r['pf']['joint_reads']}" if r["pf"]["joint_reads"] else "")
            + (f"; demand bursts {r['pf']['bursts']}: after >= 0.25 ms idle {r['pf']['burst_gap025']}, >= 1 ms "
               f"{r['pf']['burst_gap1']}; keepalive reads {r['pf']['ka_reads']}" if r["pf"]["bursts"] else "")
            + (f"; tier: {r['pf']['vt_hits']} hits (copied back, no SSD read), {r['pf']['vt_ins']} copied in, "
               f"{r['pf']['vt_drops']} dropped, {r['pf']['vt_purged']} slabs found purged; filled {r['pf']['vt_used_gb']:.2f} of "
               f"{VT_BYTES / 1e9:.2f} GB (volatile, beside the footprint)" if VTIER else "")
            + (f"; forecast reads (lever 2): {r['pf']['fc_issued']} into the tier, {r['pf']['fc_hits']} used "
               f"({r['pf']['fc_late']} still landing, {r['pf']['fc_wait_s'] * 1e3:.1f} ms waited), {r['pf']['fc_drops']} "
               f"dropped unused, {r['pf']['fc_skip']} skipped (resident / tiered); {r['pf']['fc_busy_bursts']} demand bursts "
               f"met forecasts in flight ({r['pf']['fc_busy_n']} reads)" if r["pf"]["fc_issued"] else "")
            + (f"; QOM: {r['pf']['qom_opt']} optional uses, {r['pf']['qom_skip']} optional-only misses not read (dropped)"
               if r["pf"]["qom_opt"] else ""))
        mm = r["mem"]
        say(f"    mem: MLX peak prefill {mm[0]:.1f} / decode {mm[2]:.1f} GB; at the end active {mm[3]:.1f}, cache {mm[4]:.1f}, "
            f"footprint {mm[5]:.1f} GB; lifetime peak after prefill {mm[1]:.1f} -> after decode {mm[6]:.1f} GB")
        if A6:                                           # R57_A6=ctl: the control, split logged, nothing changed
            p_ = r["pc"]
            fc = [(round(p_["drf"][i], 1), round(p_["bld"][i], 1), round(p_["wt"][i], 1), round(p_["ck"][i], 1),
                   int(p_["miss"][i]), round(p_["io"][i], 1), round(p_["svc"][i], 1), p_["rows"][i]) for i in range(min(8, len(p_["ck"])))]
            say(f"    A6 {','.join(sorted(A6))}: pre-loop {r['a6']}; first checks (draft, build, wait, check + commit ms, misses, "
                f"demand SSD ms, gate service ms, rows): {fc}; prompt {len(ids)} tokens")
eq_arms = all(len({tuple(r["out"]) for runs in v.values() for r in runs}) == 1 for v in RES.values())
def agg(arm, f):
    return med([med([f(r) for r in v[arm]]) for v in RES.values() if arm in v])
for arm in ARMS_:
    say(f"  {arm}: tok/s {agg(arm, lambda r: r['tps']):.1f} (steady state {agg(arm, lambda r: r['tpsw']):.1f}), cycle "
        f"{agg(arm, lambda r: r['cyc']):.1f} ms, check + commit {agg(arm, lambda r: r['ck']):.1f} ms, misses "
        f"{agg(arm, lambda r: r['miss']):.0f}, demand SSD {agg(arm, lambda r: r['io']):.1f} ms per check (medians over valid "
        f"runs, then prompts)")
    say(f"  {arm} where the time goes (medians per check): check + commit {agg(arm, lambda r: r['ck']):.1f} ms, demand SSD "
        f"busy {agg(arm, lambda r: r['io']):.1f} ms ({agg(arm, lambda r: r['io'] / r['ck']):.0%} of the check), gate service on "
        f"misses {agg(arm, lambda r: r['svc']):.1f} ms (the GPU's spin bound; OOO overlaps part of it), draft "
        f"{agg(arm, lambda r: r['drf']):.1f} ms (SSD idle)")
    rs_ = [r for v in RES.values() for r in v.get(arm, [])]
    if rs_:                                              # M-17 / C45 sizing: bytes and misses over all kept tokens of the arm
        kt_ = sum(r["kt"] for r in rs_)
        say(f"  {arm} per kept token (pooled over valid runs): SSD {sum(r['rb'] for r in rs_) / 1e6 / kt_:.1f} MB, misses "
            f"{sum(r['ml'] for r in rs_) / kt_:.1f} ({kt_} kept tokens; median per prompt {agg(arm, lambda r: r['mb']):.1f} MB, "
            f"{agg(arm, lambda r: r['mpk']):.1f} misses)")
if "slot" in ARMS_ and "gate" in ARMS_:
    for key, v in RES.items():
        if "slot" in v and "gate" in v:
            f = lambda arm, k: med([r[k] for r in v[arm]])
            say(f"    {key}: steady-state tok/s slot {f('slot', 'tpsw'):.1f} -> gate {f('gate', 'tpsw'):.1f} "
                f"({f('gate', 'tpsw') / f('slot', 'tpsw') - 1:+.0%}); whole run {f('slot', 'tps'):.1f} -> {f('gate', 'tps'):.1f}; "
                f"median cycle {f('slot', 'cyc'):.0f} -> {f('gate', 'cyc'):.0f} ms")
    cy_g, cy_s = agg("gate", lambda r: r["cyc"]), agg("slot", lambda r: r["cyc"])
    w_g, w_s = agg("gate", lambda r: r["tpsw"]), agg("slot", lambda r: r["tpsw"])
    say(f"R-55 live (gate vs slot): median cycle {cy_g:.1f} vs {cy_s:.1f} ms ({cy_g / cy_s - 1:+.1%}); steady-state tok/s "
        f"{w_g:.1f} vs {w_s:.1f} ({w_g / w_s - 1:+.1%}); whole-run tok/s {agg('gate', lambda r: r['tps']):.1f} vs "
        f"{agg('slot', lambda r: r['tps']):.1f}; tokens equal across every run of a prompt: {eq_arms}; every token == REF2: "
        f"{ok_all}; all runs valid: {valid}")
if "gate+pf" in ARMS_:
    io_g, io_p = agg("gate", lambda r: r["io"]), agg("gate+pf", lambda r: r["io"])
    ck_g, ck_p = agg("gate", lambda r: r["ck"]), agg("gate+pf", lambda r: r["ck"])
    cy_g, cy_p = agg("gate", lambda r: r["cyc"]), agg("gate+pf", lambda r: r["cyc"])
    d_io, d_ck, d_cy = io_p / io_g - 1, ck_p / ck_g - 1, cy_p / cy_g - 1
    v = ("FAIL (a token left REF2's row)" if not ok_all
         else "PASS" if d_io <= -0.5 and d_ck <= -0.25 else "KILL" if d_cy > -0.10 else "BETWEEN")
    say(f"R-57 verdict (PASS: demand SSD per check -50 % AND check + commit -25 %, gate+pf vs gate, every token == REF2; "
        f"KILL: cycle under -10 %): {v} (SSD {d_io:+.1%}, check + commit {d_ck:+.1%}, cycle {d_cy:+.1%})")
if TIER:
    import bigmoe.tier as _BT
    say(f"  tier flags: TIER_COMBINE {_BT._COMBINE}, OOO {_BT._OOO} ({_BT.OOO_STATS['calls']} OOO calls); EVICT_PRIOR "
        f"{os.environ.get('BIGMOE_EVICT_PRIOR')}; REFTOK {os.environ.get('R57_REFTOK')}: every token equal {ok_all}")
if os.environ.get("K73_EXACT") == "1":
    _c = ref2.counts()
    say(f"  K-73 engagement: exact MoE block calls {K73N[0]}; ref2 att_rope {_c['att_rope']}, idx_prep {_c['idx_prep']}, "
        f"hc_write {_c['hc_write']} (counted at graph build)"
        + (f"; K-83: hc_gate / hc_read {_c.get('hc_read')}, gdn_inproj_x {_c.get('gdn_inproj_x')}, drafter forms + chain_fused"
           if K83 else "") + (f"; SB from {SB_LO} row(s)" if SB_LO == 1 else ""))
if any(os.environ.get(k_) == "1" for k_ in ("REF2_OOO_1C", "R57_HC8", "R57_DNR", "R57_HCB")):   # build day (Wed 11-04)
    _c = ref2.counts()
    say("  build-day engagement (ref2 counts at graph build): " + ", ".join(
        f"{k_} {v_}" for k_, v_ in sorted(_c.items()) if any(s_ in k_ for s_ in ("ooo_1c", "hc8", "dense_native", "hc_b"))))
if os.environ.get("R57_FASTTAX"):                       # FAST-TAX engagement (ref2 counts at graph build)
    _c = ref2.counts()
    say("  FAST-TAX engagement: " + ", ".join(f"{k_} {_c.get(k_)}" for k_ in (
        "att_single", "gdn_inproj", "gdn_inproj_x", "fatt_verify", "verify_old", "verify_old_eligible", "sdpa_split", "kb_pad")))
if WSALL:                                               # per sample: warm ms vs rows (1 / 4 / 8 / 16) -> slope + intercept;
    sl_, ic_, ex_, cm_, wm_ = [], [], [], [], []         # cold 16 - warm 16 = the SSD's exposed share (+ cold-only effects)
    for d in WSALL:
        for s_ in d["samples"]:
            x_, y_ = np.array([w[0] for w in s_["warm"]], float), np.array([w[1] for w in s_["warm"]], float)
            b_, a_ = np.polyfit(x_, y_, 1)
            sl_.append(b_); ic_.append(a_)
            ex_.append(s_["cold"][1] - next(w[1] for w in s_["warm"] if w[0] == s_["cold"][0]))
            cm_.append(s_["cold"][2]); wm_.append(sum(w[2] for w in s_["warm"]))
    q_ = lambda v: f"{np.percentile(v, 25):.2f}-{np.percentile(v, 75):.2f}"
    say(f"  warm split ({len(sl_)} samples, every {WARMSPLIT}th check from the 3rd): warm slope {med(sl_):.2f} ms per row (IQR "
        f"{q_(sl_)}), warm intercept {med(ic_):.1f} ms; cold 16 rows - warm 16 rows {med(ex_):.1f} ms (IQR {q_(ex_)}); misses: "
        f"cold {med(cm_):.0f}, all warm passes {med(wm_):.0f} (0 = fully warm)")
    try:
        json.dump(WSALL, open(os.path.join(OUT, f"{TAG}_warmsplit.json"), "w"))
    except Exception as e:
        say(f"  warmsplit.json not written: {e}")
if os.environ.get("R57_NAX") == "1":
    _c = ref2.counts()
    say(f"  C1 NAX6 (ref2 tier_nax_ooo): engagement nax6_ooo {_c.get('nax6_ooo')} OOO calls, nax6 {_c.get('nax6')} other tier "
        f"calls (counted at graph build); tokens are NAX6's own reference: compare across NAX processes only"
        + (f"; H-1c nax_hybrid {_c.get('nax_hybrid')} calls" if os.environ.get("R57_H1C") == "1" else ""))
if CHATSW:
    _nt = {t: sum(1 for x in RUNS if x["type"] == t) for t in ("chat", "code", "math", "agent")}
    say(f"  chat switch: MTP head d22_{MTP_HEAD} (md5 pinned) resident, K{MTP_K}, cut t {TRUNC} with early exit; type rule "
        f"runs: {_nt} (chat -> MTP, the rest -> DFlash2)")
say(f"  footprint now {footprint_now_gb():.1f} GB, lifetime peak {footprint_peak_gb():.1f} GB; {time.time() - t0:.0f} s")
json.dump(TOK, open(os.path.join(OUT, f"{TAG}_tokens.json"), "w"))
try:                                                     # never let the side log cost a run
    json.dump(PC, open(os.path.join(OUT, f"{TAG}_checks.json"), "w"))
    json.dump(RUNS, open(os.path.join(OUT, f"{TAG}_runs.json"), "w"))
    if STALL:
        json.dump(dict(laymap=LAYMAP, runs=STL, anchor=[time.time(), time.clock_gettime(time.CLOCK_UPTIME_RAW)]),   # wall
                  open(os.path.join(OUT, f"{TAG}_stall.json"), "w"))   # clock <-> the trace's UPTIME_RAW (fanwatch's SSD temps)
    if PREROUTE:
        json.dump(PRALL, open(os.path.join(OUT, f"{TAG}_preroute.json"), "w"))
except Exception as e:
    say(f"  checks.json / runs.json not written: {e}")
if SV is not None:                                       # SESSION-VOCAB: the smoke's counters (svh_calls ~ checks; head2s ==
    say(f"SESSION-VOCAB {SV.report()}")                  # proposes; chk_out ~0 with R57_SV_CHECK=1; dsub_extra == svh_extra)
open(os.path.join(OUT, f"{TAG}_screen.txt"), "w").write("\n".join(lines) + "\n")
