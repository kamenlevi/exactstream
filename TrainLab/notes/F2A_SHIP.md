# Shipping the F2a chat head in the exact mode: what it takes (MTP Head; template written Sun 11-08 19:3x, filled after stage 1)

Status: NOT APPLICABLE. F2a stage 1 was a KILL (REPORT 'F2A night RESULT', Sun 23:48): nothing ships. Kept as the recipe for a future head.

- **Why exactness holds.** The head only proposes drafts. The engine checks every drafted row against the model, so the
  output tokens can't change. With COH the one-call check is deterministic, with 12 / 12 equal to one-token decoding
  (BM CHS4 / CHS5).
  - A new head changes how many rows each check verifies. On these kernels the row count doesn't change the bits.
  - That still needs its own proof: token equality plus the logits hash, the hash clause BM CHS5 used for SB1.
- **What the engine needs.** No code change. The engine loads the chat head from
  ~/TrainLab/ckpt/d22_${R57_MTP_HEAD}.safetensors (BM scripts/r57_screen_fast.py line 433, default b5).
  - Shipping means R57_MTP_HEAD=f2a in the exact chat env.
  - Pin the md5 [[md5]]: add it to the engine's _MTP_MD5 table (line 439 asserts it when the name is listed), or put it
    in the published env file.
- **The one test before it ships** (~15-20 min quiet, 12 IFEval, the exact config E with COH, arms in separate passes):
  - (a) == one-token decoding 12 / 12 and the logits hash equal to b5's run;
  - (b) whole decode, whole request, first token and the lifetime peak, f2a vs b5.
  - It ships if (a) holds and (b) whole decode >= +2 %. Kept per check should rise by the held-out ratio ([[x]]). The
    draft cost per step is unchanged: same architecture, same bf16 parts.
- **Lossy configs (M1-B and up).** The head changes which drafted tokens reach the relaxed rules. So the lossy chat gates
  (EvalSuite's IFEval card) would need a re-run. Tonight's plan ships it on the exact mode only.
- **Release files.** ckpt/d22_f2a.safetensors (277 MB as trained, fp32 parts) or a bf16 copy (~138 MB), its md5, and
  the data note.
  - Text: UltraChat-200k (MIT) + Aya (Apache-2.0).
  - Labels: the model's own top-32.
  - Recipe: REPORT 'F2A night'.
- **Time.** One engine A/B in the next test slot (Mon 07:05, or a chunk boundary before 12:00), then a one-line env change
  in the 45 GB / 32 GB exact chat env files.
