# Cloud Fable 5.1, task 6 (CLOUD_TASK_DENSE.md): its summary as pasted by Kamen, Mon 2 Nov 2026 ~18:5x
(The full ANSWER.md stays in the cloud session; 1,153 words, 3 of 8 Python runs, no web.)

- Diagnosis: the dense pass is not bandwidth-bound. The bytes floor is 12.7 ms and today is ~38, and the 26 ms gap is almost all launches (~1,500 kernels at 10-15 us) and the 48 per-layer CPU readbacks of routing ids. The practical floor is 15-17 ms, because a 16-row 4-bit GEMM reaches 75-85 % of bandwidth, not 100 %. So ~18-20 is the honest target and 13 is not.
- Six designs, halved:
  - compile and command-buffer batching: -4 ms
  - HC fused read plus grouped GEMM: -2
  - attention N-fused QKV plus a GQA tail that reads each KV head once: -2.5
  - router fusion with no per-layer sync: -2
  - the one-kernel GDN body: -3
  - a fused head epilogue: -0.5

  Cumulative halved: 37 -> 23. Un-halved: 15-16. Getting to 18-20 needs two of the six to land at their upper bound; the fused HC precedent (24.4 -> 14.7, bitwise) says fusion levers can.
- Exactness pushback: chasing bitwise equality with REF2-T kernel by kernel is what killed bmm16 and will kill the GDN body too. The exactness that matters is SD = AR under the same arithmetic, which needs row-invariant kernels, not a frozen reduction order. Collect every bit-changing lever into one new reference REF3-T and re-certify once at the end: the ~15,000-row bit-exact run against its own autoregressive decode, plus the 210-answer scores. One certification, not six.
- Tomorrow's kernel: the GDN body for one layer at 16 rows, recurrent form, the state column-sliced into 16 KB tiles in threadgroup memory. Checks: 1-row and 16-row bits identical, bits vs REF2-T, <= 0.2 ms per layer. Kill above 0.25 ms or any row dependence.

## Orchestrator's notes (18:5x)
- To check against Leap's data before acting: R-55 GATE16 made the gate sync-free (160 ids per layer), so the "48 per-layer CPU readbacks" may already be gone. C0's piece times are GPU-side. Leap should count kernels per check and look for remaining host syncs before taking the -4 / -2 ms launch and sync items.
- REF3-T (a new exact reference re-certified once) is a decision for Kamen, with EvalSuite's full-length test. The same path made REF2-T the reference on Nov 1.
