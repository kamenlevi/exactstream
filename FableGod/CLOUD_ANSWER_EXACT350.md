# Cloud Fable 5.1, task 9 (CLOUD_TASK_EXACT350.md): its summary as pasted by Kamen, Mon 2 Nov 2026 ~20:0x
- **Verdict: no exact 350.** A 16-row check must move 10.8 GB through RAM (the expert union + the dense trunk) whatever the kernels do: 38-42 ms at 258-284 GB/s. With the drafter at its ceiling (12.2-12.5 kept), 350 tok/s allows a 35 ms cycle, so the bytes alone don't fit, before the draft, the SSD or any kernel inefficiency. At 32 rows it is worse (15.2 GB against a 43-49 ms budget). On MBPP-like code the budget is below the dense pass alone.
- **The exact ceilings:**
  - every kernel at its byte floor, SSD and draft hidden: ~290-320 on HumanEval+-like code, ~255 MBPP-like
  - realistic exact at night with the ceiling drafter: ~147-159 (2.3-2.5x today's 63.6)
- **The smallest addition that closes it is the gate,** the only lever that removes bytes rather than overhead.
  - At cf >= 0.99 (1.34 % code-token deviation, inside the ~1-1.5 % budget) the byte ceiling moves to ~428.
  - 350 then needs dense <= 17 ms, the expert kernel <= 1.15x its floor, and SSD and draft hidden.
  - Add the comment loophole for margin: code tokens stay exact, and it buys kept without spending the budget.
  - R1 stays out (it would push deviation to 2.4 %).
  - Realistic for that set: ~230, ~200 after the haircut.
- **Pushback:** even with the quality set, 350 is a kernel-floor question. Four things must all land: the HC fused read (not found yet), the expert kernel within 15 % of its floor, the forecast that hides the SSD, and the early-layer drafter that hides the draft.
- **The three measurements that decide it soonest:**
  1. Wednesday's gate test, with live bytes per check
  2. the comment loophole, split by code and comment tokens
  3. one night logging dense + experts at 16 and 32 rows with the achieved GB/s

  If dense + experts come in above 60 ms, exact stays <= 200 this quarter, and 350 is quality-defined only.

## Orchestrator's check (20:0x)
- 7.38 GB of experts per 16-row check (k62p_floor, S 16) + 3.44 GB dense = 10.82 GB; at 270 GB/s ~40 ms. 350 x cycle <= kept means cycle <= 12.35 / 350 = 35.3 ms. The verdict's arithmetic holds.
