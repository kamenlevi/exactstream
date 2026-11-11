# RESULT: the CHAT150 chat gate on M1-B + COH (Sun 2026-11-08 09:1x) — IFEval same scores PASS; the multilingual verdict waits for MGSM

Pre-registered in docs/PREREG_chat_gate_2026-11-08.md (02:4x, before any row).

> (a quote from the project owner, kept private)
should compare against known benchmarks of the model. Not something new".
- Quality gates are scored only on known benchmarks: lossy vs its exact reference on the same harness.
- Our own blunder lines (multi and long code) are information only.
- So the multi FAIL below is **information under Kamen's 11-08 rule**. The IFEval PASS stands.
- The multilingual verdict comes from MGSM vs NAX6-N (PREREG amendment 2).

**The config:** BigMoE's data/stack/chat_env.txt as rewritten at 07:48 (97e81b8), 35 keys.
- M1-B = S5' + QOM on the chat sets (R57_QOM_SETS=humaneval,mbpp,ifeval,multi) + QOM gating on MTP drafts
  (R57_QOM_MTP=2) + REF2_NAX_COH=1.
- Sat 256, MEM 47 GB. Snapshot data/r57/snap/chs4 (342f914, ref2 11fa14ff), screen 41b146a77504. BigMoE 5abc51a, ext
  build_qom .so ff69fa24a745, code pin 6b6a4fb64229.
- In every process, all 40 outside modules equal BigMoE's HEAD, and the manifests are equal. QOM engaged on every run
  in the QOM sets.

**Process notes:**
- **First pin, voided:** runner 59 first pinned the 07:16 env (CHS2 snapshot 09ef078, no COH). BigMoE and the
  orchestrator asked for a re-pin (Leap: without COH the one-call reads stale pool slots). I stopped it at 07:51, before
  any row.
- **Restart:** the peers hook held the restart through BigMoE's PFX and CHS5 quiet locks. Runner 59 restarted at
  08:13:02 on the 07:48 env. BigMoE gated its FU on CHAT_m1b_DONE, so chat still ran first.
- **Timing:** forward 08:13-08:43 (13 processes), reverse 08:45-09:10 (13), tfwin 09:10-09:14.

## The lines (vs NAX6-N)
| Line | Result | Verdict |
|---|---|---|
| IFEval forward (n 100) | 86.0 vs 87.0 %, -1.00 pt, McNemar p 1.0 (4 / 5) | PASS |
| IFEval reverse | 83.0 vs 87.0 %, -4.00 pt, p 0.388 (4 / 8) | PASS |
| multi blunders (forward, 24 rows, tfwin vs NAX6-N) | 7,033 tokens, 235 blunders = 334 per 10K (bar <= 1), deviations 15.29 % | FAIL: information under Kamen's 11-08 rule |
| **Chat gate** | | **IFEval PASS; the multilingual line (MGSM, amendment 2) is pending** |

Written 09:1x as "FAIL (multi)". Changed 09:4x under Kamen's ruling; the numbers are unchanged.

## Is the multi FAIL real? (information)
Checks (tools/tfwin_rows.py chat_m1b_multi chat_m1b_rev):
- **Not a prompt or template mismatch.**
  - The multi items carry fixed prompt_ids, which the screen and tfwin both use.
  - The deviations and blunders are spread through all 24 rows and all 8 languages. Each row's first blunder sits at
    token 2-261, and the first rows' starting tokens are ordinary text. A context mismatch would hit the first tokens of
    every row.
- **Unlike code:** 22 % of the deviations are blunders (235 of 1,075). On code, the same tfwin against the same
  reference found 1.70 % deviations and 0 blunders (CERT's long-code line: v1' + wired, QOM on humanevalt).
- **Every language fails by more than 100x:**

| Language | Tokens | Deviations | Blunders per 10K |
|---|---|---|---|
| ar | 885 | 24.97 % | 474.6 |
| bg | 1,076 | 16.36 % | 399.6 |
| de | 906 | 8.61 % | 143.5 |
| es | 887 | 8.23 % | 169.1 |
| fr | 947 | 5.39 % | 116.2 |
| ja | 908 | 16.08 % | 330.4 |
| ru | 681 | 19.24 % | 440.5 |
| zh | 743 | 26.78 % | 686.4 |
| all | 7,033 | 15.29 % | 334.1 |

- **Not tested here: the cause.**
  - M1-B's non-exact parts on chat are QOM on the chat sets (with the MTP-draft gating) and REF2_SMALL_NATIVE, per
    BigMoE's C4 bisect. COH made the one-call exact.
  - A bisect of those on multi would name the lever.

## Information
- **vs REF2-T (90.0 %):** forward -4.00 (p 0.125, PASS as information), reverse -7.00 (p 0.0156, FAIL as information).
  REF (90.0 %) gives the same.
- **Token-identical to NAX6-N:** IFEval 8 / 8 of 100 (forward / reverse). NAX6-N has no multi rows.
- **The orders agree** on 9 / 100 IFEval rows and 0 / 24 multi rows. M1-B + COH repeats exactly on one history
  (BigMoE CHS5: 12 / 12), but its tokens depend on the cache history.
- **Cap hits (forward / reverse):** IFEval 10 / 7 (NAX6-N 5); multi 11 / 13 of 24 (at 384).
- **Rough speed** (screen's own row stats, under a share lock, not an A/B):
  - IFEval whole decode 45.4 / 46.5 tok/s, kept 2.77 / 2.68 per check;
  - multi 35.9 / 52.6 tok/s (24 rows, noisy);
  - BigMoE's quiet A/B: 49.8 (CHS5).

## What this means
- **The verdict waits for MGSM.** M1-B gives the same IFEval scores. On multilingual prompts it commits a token the
  exact model gives < 1 % on about 1 in 30 tokens (1 in 15 for zh, 1 in 86 for fr). That is information; MGSM vs
  NAX6-N decides (amendment 2).
- **m1bs** (chat2_env, BigMoE's CHS5 stack, DRY OK 08:15) carries M1-B's QOM plus more lossy parts. It needs the same
  MGSM line.

Runs: results/chat_m1b_{fwd,rev}, results/chat_m1b_multi (tfwin), results/chat_m1b_{fwd,rev}_d4.txt,
results/chat_m1b_multi_report.txt, results/CHAT_m1b_DONE; log results/day59_m1b_runner.log.
