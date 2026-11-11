# K wall: where the walk data lives (MTP Head for Fable GOAL, Fri 2026-11-06; the orchestrator 10:1x)

GOAL runs no GPU. Every array below is already on disk, so CPU replays need nothing new. New walks and any GPU probe go
through MTP Head, under pre-registered lines and the SEED noise caveat: one reseed moves code kept by 0.16 and math by
0.01, so single-run code deltas below ~0.3 can't be read.

## Stored per-check arrays (np.load; read with `python -I`; ~/TrainLab/data, local only, gitignored)
- **`d02b_walk_kw_code.npz`**: L1-20 on the 24 HumanEval+ windows, with candidates. Also `_kw_ifeval`, `_kw_multi`.
  - `names`: one file name per sample (e.g. tf_code_HumanEval_10.npz).
  - Per sample i:
    - `i_ids` [T]: the 125B's token ids. `i_off`: the prompt length. `i_gen`: the generated length.
    - `i_checks` [n, 3]: per check (drafts accepted k, depths checkable m, anchor position); kept = k + 1.
    - `i_cand` [n, 15, 16]: the drafter's top-16 candidates at each depth.
    - `i_tsm` [n, 15, 16]: the selector's softmax over those candidates, teacher-forced.
    - `i_dtok` / `i_dcf` [n, 15]: the official drafts and the drafter's confidence in them.
    - `i_ttok` / `i_tcf` [n, 15]: the 125B's tokens and the 125B's confidence.
- **`d02b_walk_code.npz` / `d02b_walk_math.npz`**: L1-20's baseline walks (ids / off / gen / checks only). Their
  reproduced kept: code 9.57 (666 checks), math 7.00 (2,046).
- **`d02b_walk_seed1_{code,math}.npz`**: the seed-1 retrain (9.73 / 6.99), for noise.
- **Text summaries** with per-depth rates: `d02b_walk{TAG}.txt` (L1-20: `d02b_walk.txt`).
- **MBPP's own walks** so far are other drafters' (`d02b_walk_d23h_dfl_mbpp.npz`, `d02b_walk_et_h*_mbpp.npz`).
  L1-20's MBPP result (5.94 at 16 rows) is in REPORT 'MATCH RESULT' and data/match_read.txt.

## Inputs (the 125B's answers + the context taps the drafter reads)
- Code: ~/Leap/data/dfl2d_code/*.npz (24 HumanEval+). Math: ~/Leap/data/dfl2d_math/*.npz (60 GSM8K).
- MBPP: ~/TrainLab/data/taps_mbpp_walk/*.npz (100, held out from L1-20's training).
- Agent: ~/TrainLab/data/d02a/agent_*. IFEval / multi: ~/TrainLab/data/d02a/tf_*.

## CPU replays to start from (~/TrainLab/sticky)
- `rerank0.py`: the ceiling of a better pick among the 16 candidates (top-3 x1.21, top-16 x1.29 on code).
- `da_replay.py`: draft-ahead hit rates (DA-0).
- `kwall_mtp.py`: the MTP chain against DFlash2's ranks (KWALL b).
- The walk harness itself is `d02b_walk.py` (MLX GPU, ~6 GB; env TL_DRAFT, TL_WALK_TAG, TL_WALK_SETS="tag=glob,...",
  TL_WALK_CF=1 to store candidates). Ask MTP Head to run it.

## Coming Saturday (MTP Head's weekend pick; GOAL will be told)
- L1-20 with candidates on MBPP (100) and math, i.e. `d02b_walk_kw_mbpp.npz` / `_kw_math.npz`.
- HumanEval+ checks split by a lexer into code vs docstring / comment, giving K on docstring-free code.
