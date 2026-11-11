# Cloud Fable 5.1, task 7 (CLOUD_TASK_12GB_V2.md): its summary as pasted by Kamen, Mon 2 Nov 2026 ~19:3x
(The full ANSWER.md stays in the cloud session; 1,027 words, 4 of 8 Python runs, no web.)

- What changed since v1: the replay puts the top-2 two-pass at 0.96 GB per check on the engine (1.25 GB / the replay's ~1.3x pessimism), below the 125B-sure oracle, because rows after the first miss stay cheap. So the SSD is no longer the wall at 12 GB: at 13 GB/s it costs 74 ms per check, and the two dense passes plus expert compute cost 61-71 ms. The check is now a two-resource problem, and the whole game is overlap.
- The budget: serial sum 147-157 ms, full-overlap bound 80-84 ms, honest midpoint 113-120 ms. That is 100-106 tok/s at kept 12 and 108-115 at kept 13, with 143-163 at the bound. Chat goes from 6 to ~27-39, because it becomes compute-bound instead of SSD-bound.
- The gap to 120 at kept 12-13 is only 10-20 ms. Three levers close it:
  1. hide compute under the stream: issue pass 2's reads all at once from pass 1's routing, and use a one-layer lookahead in pass 1
  2. land the HC fused read (it counts twice, once per pass)
  3. hide the draft on the ANE during pass 2

  150 needs all of that at its best plus a lower certification threshold, which is quality-defined. Still the stretch.
- Pushbacks:
  - Pass 2 cannot be made cheap by skipping rows: it re-reads 3.44 GB of dense weights whatever its row count. Its frequency is the lever: at p1 >= 0.9 it runs in 71 % of blocks instead of 97 %, at 0.7 in 37 %. That makes the threshold sweep a decisive test, not a tuning detail.
  - A resident-only pass 1 is a dead end. With 51 LRU slots against a prompt that touches ~253 experts per layer, a row's two experts are both resident at only a minority of layers, the pass degrades, and certification collapses the way the layer-35 probe did. One replay count settles it.
- Tests:
  - overlap on the engine: PASS at wall <= SSD + half the compute
  - the ANE draft in pass 2's shadow: PASS at <= 2 ms exposed and >= 0.97x kept
  - the threshold sweep, riding on tonight's decision-map run
- Hold to ~105-115 tok/s on code at kept 12-13; 120 if the overlap test lands near its bound.
