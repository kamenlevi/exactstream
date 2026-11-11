# Per stream, expert groups per kept token follow the 125B's own union law; on chat the wall is check length, not experts
- Date: 2026-11-05 17:15 EET. Session: Lunatic (placed by the orchestrator).
- Claim: if every row of an R-row check is kept, the distinct (expert, layer) groups per kept token on the 125B's own text
  fall only ~25 % per doubling of R. The 512 experts are independent functions on the real weights: no shared base, no
  common subspace, no near-duplicate neurons. At equal check length chat routes FEWER experts per token than code, so
  chat's high cost (490 vs 304 on code) is short checks and rejected rows, not experts. Verifying several streams in one
  pass is the only measured break of the law on both domains (EXACT). That is aggregate throughput: per stream a
  merged check is slower than solo (21-27 vs 52-60 tok/s on chat at B8), so it serves many agents, not one conversation.
  At 45 GB experts are ~1/3 of a chat check, so a 2x on chat expert groups would buy only ~1.2x chat speed there.
- Number [P on K routing, 36 decision-map windows, EXACT top-10]:
  - f(R) at R 8 / 16 / 32: code 276 / 218 / 163, IFEval 258 / 202 / 151, multi 226 / 177 / 133.
  - Merged checks of B streams, groups per kept token: code x0.55 (B4) / x0.35 (B8); IFEval x0.65 / multi x0.71 (B8).
  - With a perfect stop, chat x0.51-0.55 at B8.
  - Weights [K, layers 3 / 23 / 43]: mean expert 0.78-1.02 % of energy (independent 0.78 %); best neuron |cos| median
    0.086-0.090 (chance 0.09).
- How: ~/FableGod/lunatic/wl_redund.py, wl_width.py, wl_width2.py, wl_chatlen.py, wl_slot.py (numpy, CPU, seconds each);
  outputs beside them; WALL_L_1105.md sections 1-4 and 9.
- Prior art:
  - multi-stream batching: published (throughput-oriented MoE serving; MoE-Lightning, FlexGen);
  - the per-domain union law and the chat reading: unchecked;
  - the base / delta kills: already in TRIED (385 / 389 / 396 / 93).
- Status: first measurement (CPU replay on measured routing).
- Next: the 10-minute expert-phase microbench of merged routings (WALL_L_1105 section 6). PASS: code B8 <= 0.45x of B1,
  IFEval <= 0.80x, bit-identical rows.
