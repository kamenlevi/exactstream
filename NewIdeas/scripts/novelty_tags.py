"""Write ~/TRIED/novelty.tsv: a published / adapted / new / engineering tag for every ADOPTED / PASS lever in TRIED.md
> (a quote from the project owner, kept private)
published = a published method used as is (ported to MLX / this engine); adapted = a published idea changed in a way that
matters (made bit-exact, a new signal, a new setting); new = no prior art found in our checks (web searches Mon 2 Nov
22:xx, listed in ~/TRIED/NOVELTY.md); engineering = an implementation step, a fix, a measurement or a diagnostic (no
method claim). Usage: python3 -I novelty_tags.py [~/TRIED]"""
import os, sys

D = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/TRIED")
TM = "Thinking Machines batch-invariant kernels"
HOB = "HOBBIT / MC-MoE mixed-precision experts"
ANY = "Any-Precision LLM bit-plane layout"
GATE = "anemll-flash-mlx / SlotStream sync-free slot gate"
OVL = "MoE-Infinity / ProMoE / Fiddler load-compute overlap"
TAGS = {
    "published": {
        "DECAYED-FREQUENCY": "frequency-based expert caching (MoE-Infinity), LFU",
        "S45B": "a bigger cache", "S45C": "EVICT / DraftExpert / SpecDec++ confidence truncation",
        "T": "EVICT / DraftExpert confidence truncation", "L1-14": "EVICT / DraftExpert confidence truncation",
        "V1": "pre-gating (Pre-gated MoE, ProMoE)", "L1-17A": "top-k reduction (MoE-Spec, dynamic routing)",
        "L1-17": "top-k reduction", "TK1": "top-k reduction", "B10(B)-3INST": "QTIP",
        "RND19#2": "SliceMoE gate-sharpness rule", "M-03": "HOBBIT (low precision on a cache miss)",
        "15E": "MTPLX hc_verify_read", "HC_VERIFY_READ.READ_ROWS": "MTPLX", "P1": "mlx-lm pipelined generate",
        "B5C": "MTPLX compiled GDN", "HC8": "8-bit weights (MTPLX's 8-bit HC)", "MTPLX_FUSED_GDN_INPROJ": "MTPLX",
        "R-72": "MTPLX fused RoPE / indexer prep", "AW1": "SpecDec++ / adaptive draft length",
        "D-10": "DistillSpec soft KD", "L2": "DFlash (z-lab)", "MTP": "DeepSeek-V3 MTP drafting",
        "S1": "the standard greedy speculative loop", "SC1": "FR-Spec", "D-22": "EAGLE-3 training-time test",
        "E2": "drafter data scaling", "REPETITION": "MTPLX loop_guard", "L2-10": "MTPLX sparse prefill lane",
        "SMALL35": "DFlash on Qwen3.6-35B (z-lab)",
        # Tue 3 Nov: sub-rows the merge fix surfaced (a session's 2nd+ row under one ID)
        "L1-10#3": "SAM decoding (suffix-automaton longest-match lookup)",
        "DFLASH2": "token trees over drafter candidates (SpecInfer / Sequoia)",
        "R-16": "entropy-coded weights decoded on the GPU (EntroLLM, DFloat11, tile-level ANS)", "TRL-3": "QTIP",
        "E2#2": "drafter data scaling", "PF1": "double-buffered next-layer prefetch (FlexGen / MoE-Lightning)",
    },
    "adapted": {
        "F24B16": "expert budgets in verify (MoE-Spec, AcceptMoE)", "FM1": "expert budgets + JIT prefetch",
        "R-22": "expert read budgets", "M-17": "AdapMoE per-layer cache allocation",
        "PER-LAYER": "AdapMoE per-layer cache allocation", "RND17": "AdapMoE per-layer cache allocation",
        "ONE-RECORD": "LLM in a flash row-column bundling", "OOO": OVL + "; the bitwise fixed order is ours",
        "OUT-OF-ORDER": OVL, "RND20": OVL, "V2-3/M6C": OVL + " + fixed-order reduce",
        "V2": "PACMan / RRIP insertion", "VICTIM-SLOT": "PACMan / RRIP insertion",
        "VERIFY-PASS": "JIT prefetch (ProMoE)", "CORR": "learned prefetch predictors (ProMoE)",
        "S10C-1": "learned prefetch predictors",
        "B12LL": "KTransformers Expert Deferral (SOSP '25); found independently", "DH_W4": "KTransformers Expert Deferral",
        "3-BIT": HOB + " + " + ANY, "PLAN": HOB, "PLAN4433333333": HOB, "RND18-RND20": HOB,
        "RECORDS3": ANY, "MIX4-MSB2M": ANY + " / HOBBIT", "MIX6-MSB2M": ANY + " / HOBBIT",
        "RES1": "bit-plane residency (HOBBIT, Any-Precision LLM)", "V3": "bundle of adapted pieces",
        "REF2-T": TM + " + " + HOB, "R-58": TM + " + " + HOB, "REF2B": TM + " (fixed KV split sizes)",
        "R1": TM, "L1-23": TM, "L1-01B": TM, "BATCH-INVARIANT": TM, "TK2": "top-k + an exactness replay",
        "GATE16": GATE, "R-55": GATE, "SPIN": GATE,
        "L1-25": "tensor-unit grouped expert GEMM (MonoMoE-style)", "K-62F/G": "tensor-unit expert kernel (MonoMoE)",
        "L1-26A": "Marlin-style pre-layout", "QMM16C": "low-bit tensor-core GEMM (Marlin, QServe)",
        "R-09B": "MTPLX's library-template GEMM replica", "R-61A": "MTPLX fused GDN, made exact",
        "L1-20": "self-distillation on target outputs (EAGLE / Medusa)", "D-17": "deep MTP chaining (FastMTP)",
        "D-20": "drafter by task (WhiFlash)", "D-20B": "drafter by task (WhiFlash)",
        "D-23D": "drafter by task (WhiFlash) + our keyword rule", "TYPERULE": "drafter by task (WhiFlash)",
        "B12_NORM=MASS": "top-k cut without renormalising",
        "D1I": "chunked + layer-major prefill (FlexGen, MoE-Lightning)", "LMP4/PK4": "FlexGen layer-major prefill",
        "PF2": "FlexGen layer-major prefill", "#18": "FlexGen layer-major prefill",
        "M-29#3": "anticipatory device wake: an early PCIe wake before the doorbell (US 11789658), HotOS '15 "
                  "anticipatory CPU wake-up (Software Engagement with Sleeping CPUs); ours: a software 4 KB read timed "
                  "by the MoE router's launch, on Apple's internal SSD",
        "EXACT": "expert offloading with LRU pools (MoE-Infinity, Mixtral-offloading), made exact by the gate",
        "S45-T": TM + " + " + HOB, "PASS-LEVEL": "expert budgets in verify (MoE-Spec)", "D2#2": HOB + " + " + ANY,
        "RND20/RND25#2": ANY + " + " + HOB, "RND18#3": ANY + " / HOBBIT",
        "T3R": "QTIP base + a residual refinement for the top ranks",
        "R-62P": "expert skipping by residency (SwapMoE / AdapMoE)",
        "T0": "self-speculative decoding with a cheap pass of the same model (Draft & Verify, LayerSkip); here a top-2-expert pass certified by max softmax",
        "T2": "self-speculative decoding with a cheap pass of the same model (Draft & Verify, LayerSkip); MTP 4-row checks", "T3": "self-speculative decoding with a cheap pass of the same model (Draft & Verify, LayerSkip); 16-row checks (cf. the staged check)",
        "P1-OEA": "OEA union members (2511.02237)", "P2-PREFETCH": "JIT prefetch (ProMoE)",
        "S2": "runahead route prediction (SP-MoE, ProMoE)", "R-40#2": TM, "R-40#3": TM, "REF2#2": TM,
        "R-40#18": TM + " (fixed KV split sizes)", "PF3": "FlexGen layer-major prefill",
        "Q1U-T3": "OEA union (2511.02237) + low precision for drafter-sure rows (HOBBIT-style); the certainty keying is ours (M-18)",
        "REACH-CUT": "reach probability as a product of draft confidences (EAGLE-2 / SpecDec++), used here to cut the verifier's experts",
        "UNION-STACK": "OEA (2511.02237) stacked with our certainty gate",
        "DM-Q1": "PSR §7 (2610.34721): the verify block's expert demand weighted by the drafter's confidence, one set per block, rerouted; ours: a per-row expert count, no reroute, no renorm",
        "M-18": "PSR §7 (2610.34721): the verify block's expert demand weighted by the drafter's confidence, one set per block, rerouted; ours: a per-row expert count, no reroute, no renorm",
        "M-18D": "PSR §7 (2610.34721): the verify block's expert demand weighted by the drafter's confidence, one set per block, rerouted; ours per row, at the 24 GB default",
        "QOM": "cache-conditional experts / cache-aware routing (2412.00099, Cache-Prior, BuddyMoE); ours: no rerouting, only drafter-sure rows, only the experts that would miss",
    },
    "new": {
        "DM": "reuse of the previous position's MoE output; prior art caches / prefetches experts across tokens "
              "(ST-MoE) or skips them (MoDES), none found reusing outputs",
        "DM#10": "reuse of the previous position's MoE output (as DM: none found reusing outputs)",
    },
}
ENGINEERING = """CONCURRENT JT1 PS R-69B REC2 RECORDS=3 S45-STACK W1 ITEM R-24 R-69 S10 S45 S45B-2 3C PLANE R-44A R-48 EXPERT
W24-SP B5B F1 PASS-AHEAD R-11C R-39 R-50 R-52B R-56D SB1 SCOUT CA D32 FUSED GATES GP1 L1-23F L1-29 LAZY MERGED PACK4 R-13
R-13D R-36 R-59 R-61 R-65 R-70 R-73 R-74 R-81 R-82 R-83 REBUILD.PY REF2C R-06A RND21/RND22C D-01 D-02B L1-04 R-15 R-76 R-77
R-78 R-79 RND22B 24 BP1 CACHE-ONLY FFX GROUPED HEAD_EACH LIFETIME M24B MEM1 MOV MTP_C N-GRAM NO-COPY R-01 R-41E R-45 R-45T
R-45V R-46 T1 V3P D1D FOLD-EXACT FRS1 GROW M-5 R-34 RES3 L2-11 PF4 PFG1 PFM PFR PF_WARM R-52 R-53 STREAMING B6D B8 CMP3
FR-SPEC G1/G2/#12 GATE GATE-STAMPS GDN_EXACT GS1 LC1 LONG1 ORDER-PRESERVING R-41 R1B REF2 B8-NONDET B8MR-HAZARD L1-24S
MMA-TILE POS-2 R-40 R-56B R-56C R-58G WV1-WV3 M-29 M-29#2 M-29#4
L1-15#2 GDN#2 B5A R-09#2 R-15#2 R-40#4 R-40#5 R-40#6 R-40#10 R-40#11 R-40#12 R-40#14 R-40#15 R-40#16 R-40#17 V3#2
R-38#3 FOLD-EXACT#2 FOLD-FAST#3 FOLD-FAST R-45#2 M-26 W24-SP#2 M-01 3C#2 3C#3 RECORDS4#2 EXACT#2 FUSED#3 M-28#3 F1#2 X1#2
B1#2 R-45U SGB HT1 T2S T3#2 UNION-THETA UNION-STACK-24 CALLS-SHARE D-22#3 D-24#2 KWALL#2""".split()
rows = []
for tag, d in TAGS.items():
    rows += [(k, tag, v) for k, v in d.items()]
rows += [(k, "engineering", "") for k in ENGINEERING]
keys = [r[0] for r in rows]
assert len(keys) == len(set(keys)), [k for k in keys if keys.count(k) > 1]
with open(os.path.join(D, "novelty.tsv"), "w", encoding="utf-8") as f:
    f.write("# key\ttag\tprior art or note (NewIdeas, Mon 2 Nov 2026; see NOVELTY.md)\n")
    for k, t, v in rows:
        f.write(f"{k}\t{t}\t{v}\n")
from collections import Counter
print(len(rows), "tags:", dict(Counter(t for _, t, _ in rows)))
