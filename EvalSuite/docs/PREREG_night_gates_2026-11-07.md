# PREREG: tonight's two code gates — t95ck1u (12 GB) and C30 (~32 GB) (Sat 2026-11-07 19:5x; before any row)

Written before any row of either gate.
- **Request:** the orchestrator, 19:5x, and NewIdeas' QUEUE 223bf55. Kamen wants the 45 / 24 / 12 GB x exact / lossy
  table by Sun ~08:00. The order is the 12 GB gate, then the C30 gate, then CERT at 01:30 (unchanged, never delayed).
- **F24 (24 GB lossy) is not gated tonight:** BigMoE 18:0x, r57 cannot run its first exact check below C ~160.

## Gate 1: t95ck1u, the 12 GB θ 0.95 arm
- **A NEW arm after two FAILs on the same 60 prompts:** t80ck1u (θ 0.8) and t90ck1u (θ 0.9), RESULT_t80ck1u. This is
  the third θ of one lever on the same rows. The RESULT reports it with that history: a PASS here is one of three
  tries, not a fresh test.
- **The arm:** Fit12's t95ck1u (P1-K1, KU 4, cf 0.9, θ 0.95), the form Breakthrough's map passes (1.10 % code flips).
  - Fit12's measurement: 40.59 tok/s whole decode on HumanEval, 18.74 on MBPP, 11.28 GB.
  - Entry point: fit_t3.py with F12_IDS_OUT (sha256 11cadf9ec149). The arms' code paths are unchanged since t80ck1u; it
    now also reads items missing from b12llff4's rows from data/items.
- **The lines, exactly as PREREG_t80ck1u + amendment 1:**
  - HumanEval+ 60 in two orders (forward sorted, then reverse), each order ONE process, F12_CAP=item (768);
  - decision 4 vs REF2-T (its exact base) per order: delta > -5 pt AND McNemar p >= 0.05. PASS = both orders;
  - NAX6-N (results/ref2tnax) and REF as information;
  - VOID: ex's first 32 tokens vs REF2-T on the first 4 forward keys.
  - No fallback arm.
- **The runner:** tools/day_runner57.sh, runner 56 with the arm t95ck1u and no fallback -> results/t95_{fwd,rev},
  results/T95_DONE.

## Gate 2: C30, the ~32 GB lossy context point
- **The config:** BigMoE's data/stack/c30_env.txt (b861a17, written 18:50): S5''s config (v1' + wired + the Sunday
  bundle) at C 160 with R57_VTIER off. Its comment gives MEM 35 GB and the screen data/r57/snap/s5p/r57_screen_s5p.py
  (snapshot 6db4c24, 0 live modules). BigMoE measured 153.5 / 114.8 tok/s whole decode (HumanEval / MBPP), 31.8 GB.
- **Why a gate:** FAST is cache-dependent, so S4 at C 256 does not cover C 160.
- **The lines, CODE-ONLY**, S4's form (PREREG_s4 + amendments 1-2) restricted to code:
  - Same scores: HumanEval+ 60 in two orders, 10 prompts per process; decision 4 vs NAX6-N (results/ref2tnax) per
    order. PASS = both orders. REF2-T as information.
  - Long code: every second key of keys_T (21 rows), one order, QOM on humanevalt, tfwin vs NAX6-N. Blunders <= 1 per
    10K tokens.
  - C30 PASS = both lines.
- **Math / IFEval are out:** they run EXACT in this config, and r57's exact path is limited at low C.
- **Aborts:** a process that aborts with "no free slot" makes the gate INCOMPLETE (no retry). The table says so.
- **VOID:** S4's code checks (every BigMoE module = BigMoE's HEAD; manifests equal across processes), or QOM not
  engaged on code rows.
- **The runner:** tools/day_runner58.sh, runner 54's gates and stages restricted to code -> results/c30_{fwd,rev,ht},
  results/C30_DONE.
