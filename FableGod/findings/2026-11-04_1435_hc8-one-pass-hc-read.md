# An 8-bit one-pass hyper-connection read runs the 125B's 16-row HC reads in 0.54x the exact read's time, as accurate as plain 8-bit
- Date: 2026-11-04 14:35 EET. Session: Leap (kernels).
- Claim: the HC read (pending write, grouped RMS norm, low-rank down / up mixers, gates, 4-stream mix) for a 16-row
  check, with 8-bit g64 mixers, one pass over the streams (the norm weight applied before the down GEMM and 1 / rms after
  it), and 3 dispatches on the M5's matrix units, takes 0.54x the time of the exact bf16 read, and deviates from it
  exactly as much as MLX's own 8-bit path does.
- Number: 5.22 vs 9.67 ms per 96-read chain (real weights of all 96 mixers; quiet lock; median of 40) [K]. That's ~-4.3 ms
  per 16-row check, plus -0.6 GB of resident RAM from the 8-bit mixers at the 45 GB line [P]. Vs the exact read, mixed
  rel rms 6.69e-3 against 6.70e-3 for HC8-eager [K]; the written state is bitwise; row-invariant (SD == AR). FAST track.
- How: ~/Leap/scripts/hc8r.py (v2), hc8r_screen.py, hc8_wire_check.py; commits ec7140b, 4aa8de8, c805fd0 (ref2
  REF2_HC8_FAST, default off); data/hc8r_screen.txt, data/hc8r_runner.log; TABLE_350 'HC8-1P'.
- Prior art: adapted. 8-bit HC mixers are the project's own HC8 recipe. Folding the RMSNorm weight into the next linear
  and scaling after it is standard (QuaRot / SpinQuant style), and split-K with a fixed-order reduction is standard.
  The fused one-pass 8-bit HC read at 16 rows on Apple's matrix units: no prior found (search not exhaustive).
- Status: first measurement. The pre-registered v1 was BETWEEN (6.13 vs 9.44); the 0.54x is the post-hoc v2 (input tile
  built once per K partition, codes prefetched).
- Next: BigMoE's 45 GB arm (HC8 vs plain; keep at >= +2 % geo steady, no domain < -3 %) and EvalSuite same-scores.
