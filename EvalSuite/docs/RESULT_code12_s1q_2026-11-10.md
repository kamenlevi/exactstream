# RESULT: the 12 GB code gate on Fit12's S1q (Tue 2026-11-10 11:02) — STOPPED at the 12.0 GB line; NO VERDICT

Pre-registered in docs/PREREG_code12_s1q_2026-11-10.md (89d6938, before any row; dated start line 4f61d3c). The design
was HumanEval+ 60 vs NAX6 (results/ref2tnax) and MBPP 100 vs NAX6-N (results/ref2tnaxn.r2), both orders, with decision 4
per set and order. Fit12 ran it in its own session (route (i)).

**Outcome [K]:** Fit12's runner stopped the run at 11:02:14 on the pre-registered 12.0 GB guard (~/Fit12/logs/s1qg_run.log:
"qgf06: STOP: footprint peak 12.027 GB (exact bytes) > 12.0").
- It stopped during forward process qgf06, while mbpp/228 was decoding (a 126-token prompt, cap 1,024). The guard's
  reading 10 s earlier was 11.673 GB.
- The forward order has 118 of 160 keys: all 60 HumanEval+ and 58 of 100 MBPP. qgf06 completed 18 of its 20 keys before
  the stop; its mbpp/228 run was cut, and mbpp/230 never started. The reverse order never started.
- With fewer than 160 rows per order there is no verdict (PREREG, "The decision").

## Memory per process [K]
| Process | Keys | Harness lifetime footprint (GB) | Runner's exact-bytes probe (GB) |
|---|---|---|---|
| qgf01 | HumanEval+ 20 | 11.773 | 11.773 |
| qgf02 | HumanEval+ 20 | 11.856 | 11.856 |
| qgf03 | HumanEval+ 20 | 11.833 | 11.833 |
| qgf04 | MBPP 20 | 11.685 | 11.685 |
| qgf05 | MBPP 20 | 11.715 | 11.715 |
| qgf06 | MBPP 18 of 20 | 11.673 at its last completed key | 11.673, then **12.027** at the stop |

Each process starts fresh, so its lifetime figure covers only its own keys.
- The harness's figures on completed prompts stayed at or below 11.856 GB (Fit12: HumanEval/4, /144 and /12 in qgf02).
- The excess came during mbpp/228's run. This time the probe read exact bytes (footprint -f bytes); ifb12's probe read
  whole GiB.
- S1q's own measurement (11.765 GB, ~/Fit12/data/scr/s1q) used 16 HumanEval prompts at a cap of 256. This gate used the
  full caps (768 / 1,024).

## Code stayed constant [K]
- After the run, every registered hash is unchanged:
  - fit_onecall.py 6fd6d2951273, t3_onecall.py 24c21d9902a7, t2_twopass.py 36dd222c77a9, fit_pr1.py abe5e79915c8;
  - s1qg.sh 9cc5e78ed4fc, s1qg_merge.py e0abf895b0b7, env blob 524dec45a1e2.
- The outside-code manifest (tools/code12_outside_manifest.py) is identical before and after: 507 entries, digest
  9435e36075a8704c (results/card/code12_outside_manifest_pre.txt and _post.txt).

## A second problem (Fit12's merge)
s1qg_merge.py crashed with KeyError 'ex'. The chunk files store each run under the arm name "q0r", and the merge looked
for "ex", so no merged tokens or runs files were written. A resume would need the merge fixed. That changes a registered
hash, so it needs an amendment before any new row.

## Information-only score of the 118 forward rows (NOT A VERDICT)
The PREREG gives no partial verdict. NewIdeas (the queue owner, 11:05) asked for an information-only score, to decide
whether a lower-memory arm is worth a new PREREG. The rows were rebuilt from Fit12's chunk files (arm q0r) with
tools/r57rows.py into results/card_s1q_info_fwd. There is one order only, and MBPP is partial.
- **HumanEval+ (60 of 60, forward):** 91.7 % vs NAX6's 91.7 %, delta +0.00 pt, McNemar p 1.0000 (1 / 1). 24 of 60 rows
  are token-identical to the reference (results/card_s1q_info_humaneval_fwd_d4.txt).
- **MBPP (58 of 100, forward, paired on the same 58 keys):** 75.9 % vs NAX6-N's 75.9 %, delta +0.00 pt, McNemar p 1.0000
  (2 / 2). 13 of 58 rows are token-identical, and 4 hit the cap (reference 4) (results/card_s1q_info_mbpp_fwd58.txt).
  - The 58 are the first 58 in the key order, not a random sample. The reference scores 83.3 % on the 42 missing keys
    against 79.0 % on all 100.
- So, as information only: in this one partial order, S1q's code scores match the references. Memory, not quality,
  stopped this arm. Any retry is a new arm with its own PREREG (NewIdeas: in a later slot, after the release).

## What it means
- As built (C_m 51 / C_l 32), S1q does not stay under 12.0 GB on this gate's MBPP prompts at the full cap.
- Fit12's read: a retry needs less memory, and a new map line if the pool shrinks. It is not retried today; the decision
  is NewIdeas' and the orchestrator's.
- A retry with another config is a new arm, and it needs its own PREREG (or an amendment before any of its rows).

## Data
results/card_s1q_src/ holds copies of Fit12's files, since its folder is not under version control:
- the chunk files qgf01-qgf06: ids, meta and logs;
- the runner log s1qg_run.log, the merge report and the DONE file;
- s1qg.sh, s1qg_merge.py, the env record and the key file;
- Fit12's summary.txt lines for this run.

Also: results/card/code12_keys.txt, and results/card/code12_outside_manifest_pre.txt and _post.txt.
