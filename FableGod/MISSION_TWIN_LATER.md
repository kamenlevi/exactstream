# MISSION 24: THE ENGINE TWIN. Build a simulator of the real engine that predicts its measured speed, then use it to find the road to 350 at 45 GB and 120 at 12 GB.

> (a quote from the project owner, kept private)

**Why this is the hard problem now.** The measured exact default is 63.6 tok/s on code at 45 GB, while every plan talks about 200-350. Today's estimates kept coming out about 2x high against real runs:
- MATCH x1.071 modelled vs x1.038 real
- the dense-pass levers: -3 ms modelled vs -0.1-0.6 real for the GDN body
- the copy override's coverage

Every decision left (this week's build order, the 32-row runahead at 45 GB, the two-pass overlap at 12 GB, the Neural Engine draft) depends on predicting time correctly, and nobody can yet. A twin that reproduces the engine's measured numbers would turn every idea into a minutes-long CPU question with an honest answer. Nobody in the project has built one.

**What you have to build it from**
- **Per-piece GPU timings of a check:** ~/Leap/data/c0 (c0_c160_screen.txt, c0_c160_checks.json, c0_c160_probes.json). Warm check 36-38 ms + 2.9-3.2 ms per row; GDN 12.4, HC 11.4, attention 7.4, MoE glue ~5, head ~2.5.
- **Per-check timings in the live loop** (draft, build, wait, check + commit, misses, SSD ms, gate service): ~/BigMoE/data/r57/a6*_160_checks.json and a6*_160.txt (agent, C 160); K-80 / S45 / r57 logs.
- **The SSD model:** 0.088 ms per MB / 0.155 ms per miss in the loop (A6's fit, R^2 0.95); bandwidth 258-284 GB/s (bw_probe.txt).
- **Expert kernel cost:** NAX6 1.5-1.7x above its byte floor (k62p_floor.txt); ~2.0 ms per added row at 32 rows (B3).
- **Routing:** decision_map/out/routing (36 windows, prompt + answer, top-10 per layer). Your floor_bytes.py and cache_replay.py LRU replays (absolute misses ~1.3x the engine's).
- **Acceptance:** the real walks (kept per check, per-depth rates, check boundaries) for 16 / 32 rows, with and without the copy override.
- **The measured endpoint:** 63.6 tok/s steady at C 256 on code (MoonShots' six-prompt median; cycle ~166 ms by day).
- **New pieces:** the Neural Engine drafter (12.4 ms per batched call, GPU x1.02 at the floor; ~/Leap/data/ane_v2.txt); the GDN body (-0.1-0.6 ms); Leap's census (~2,800 kernels per check at 1.5-6 us).

**Deliver**
1. **The twin.** A script that simulates a decode check by check: draft (GPU or Neural Engine), the dense pieces, experts per row from the routing union with the gate's per-row k, per-layer LRU caches at C slots, SSD reads with the measured service rate and the OOO overlap of resident experts, syncs and commit, and acceptance from the real walks (16 / 32 rows, copy override on or off).
2. **Calibration first, honestly.** Reproduce within ~10 %:
   - the default's 63.6 tok/s (C 256, code)
   - C0's per-piece times and per-row slope
   - A6's per-check timings (C 160, agent)
   - the SSD ms per check

   List which knobs you fitted and which numbers it predicts without fitting. If it cannot fit something, say so; don't force it.
3. **Use it at 45 GB.** Predicted tok/s after each lever of this week's build order:
   - the copy gate
   - the 4-bit drafter
   - the start-up fix
   - the night caches C 300 / 320
   - NAX6 live
   - the cheap-experts gate
   - attention -5 ms
   - a 16-row 4-bit GEMM near its floor
   - drafting on the Neural Engine

   Give each by day and at night, and say which single number moves 350 most.
4. **Use it at 12 GB.** Cloud Fable's two-pass check with overlap (~/FableGod/CLOUD_ANSWER_12GB_V2.md): predicted tok/s at kept 10.5 / 12 / 13, and the schedule that gets closest to 120.
5. **Its own validation:** 2-3 real runs for tomorrow, with the twin's predicted numbers written down first and PASS / KILL lines on the twin itself (e.g. within 10 % of the measured result).

**Rules**
- CPU only on existing files; writes only in ~/FableGod; no downloads.
- Build in stages and save each: 45 GB calibration first. A calibrated 45 GB twin alone is already a big result.
- Mark every number as measured / fitted / predicted.
