# MISSION B1 (Fable B): THE GDN PASS. 12.4 ms of the dense pass against a ~5 ms floor.

(Draft by the orchestrator for Kamen, Mon 2 Nov 2026 ~21:0x. Fable B's first mission; Fable A works the expert kernel in parallel.)

**Why.**
- 350 needs the dense pass at <= 17 ms (cloud task 9). The dense pass is the part of each check that costs the same at 1 row as at 16.
- Today it is ~32 ms of GPU (36-38 ms unsynced).
- Fable A's HC_READ.md plan (D1 + D5, exact) takes ~3.9 ms off the hyper-connection read. With the other known levers, the stack reaches ~18-23 ms exact.
- The biggest piece left is GDN: 12.4 ms for the 36 Gated DeltaNet layers, against a floor of ~5 ms.
  - Leap's C0 measured 12.3-13.4 ms at 1 row and 12.0-13.1 at 16, with ~0 per added row.

**What is known** (read the sources; don't trust this summary over them)
- **The GDN projections are already near their floors** with REF2's qmm16c kernel (~/Leap/TABLE_350.md ~line 531, and R-13 OCC ~line 1774). So most of GDN's 12.4 ms is not the matrix multiplies.

  | projection (count per check) | REF2 (qmm16c) | floor |
  |---|---|---|
  | in_proj_qkv (36) | 2.53 ms | 1.90 |
  | in_proj_z (36) | 1.79 ms | 1.14 |
  | o_proj + GDN out_proj (48) | 2.39 ms | 1.52 |
- **Fusing the body didn't help much.** Leap's GDNB (TABLE_350 ~line 4651) runs the GDN body in one dispatch. It is exact but saved only 0.1-0.6 ms. R-21 GDNB step 1 (~line 1837) priced the floor of a batched replay.
- **C0's synced numbers include CPU setup time** (~8.5 ms per check across all pieces). It overlaps the GPU when unsynced. Fable A found ~1.5 ms of it inside the HC number.
- **Where to read:**
  - the code: ~/BigMoE/ref/qwen4_exp.py (the GDN layer: projections, conv, gates, the delta-rule recurrence and its state, norm, output gate) and ~/BigMoE/models/bare/config.json
  - Leap's scripts: gdnbody.py, gdnbody_test.py, c0_timeline.py, r73_glue_anat.py
- **The method that worked:** ~/FableGod/HC_READ.md. Read the code, take Leap's data, run a numeric check, and write an accounting that reconciles to the measured number.

**Deliver**
1. **What one GDN layer computes at 16 rows,** from the code: for each step, its expression, shapes, weight bytes, FLOPs and activations. Also the recurrent state's size, and how it is read and written.
2. **Where the 12.4 ms goes.** Split it into:
   - the projections (measured standalone)
   - the body
   - conv, gates and norms
   - kernel boundaries and fills
   - the CPU setup inside C0's synced record

   Reconcile with Leap's numbers, mark each part measured / arithmetic / judgement, and name the one timeline Leap should run to settle what you can't.
3. **At least three exact restructurings** (bitwise = REF2-T) that remove dependent steps, bytes or CPU setup. First explain why GDNB's one-dispatch body gained so little, so your designs don't repeat it. For each design give:
   - ms per check (upper bound, then halved)
   - why it stays bitwise
   - the risk
   - what Leap builds
   - the cheapest check
4. **The bet, and the dense pass after it.** With Fable A's D1 + D5 and the known levers, does the stack reach <= 17 ms exact? If not, what's left and where is it?

**Rules**
- CPU only: read code and data, run numpy checks at nice 15. Writes only in ~/FableGod/fableb/. No downloads, installs or messages to other sessions. GPU timing is Leap's.
- Before writing, read Leap's GDN entries and ~/TRIED/TRIED.md. Say why each design differs from what was tried. The GDN entries are:
  - R-12, R-21, GDNB
  - R-13 OCC, C0
  - the 4-bit dense set (~line 520)
- Exact mode stays exact; reference changes are Kamen's call. Mark every number as measured / arithmetic / judgement.
- Read in targeted pieces (grep, line ranges).
