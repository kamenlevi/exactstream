# MACSTATE (written by the orchestrator's watcher; refreshed every ~10 min)
updated: 2026-11-05T00:09Z / 02:09 EET
free RAM (no-swap): 25 GB; free disk: 130 GiB

## Live locks (mode, minutes left, what)
- quiet, 84 min left: evalsuite: relax ref --arm T3ALL --keys results/relax/keys_T3ALL.txt --chunk-minutes 60 --C 160

GPU schedule on the Mac (EET). Maintained by the orchestrator; the "live locks" section above is refreshed every 10 min.

## Thursday 5 Nov
- **Now to ~07:10:** EvalSuite RELAX (quiet lock). The Mac is busy: no other GPU work.
- **07:05-08:45:** BigMoE ticket: C 256 headline, C 224 FAST trio, DKV token check, agent LOO.
- **~08:45-10:55:** EvalSuite runner 42 (NAX6 certification, part 1).
- **~11:10:** expert-store conversion (disk-heavy).
- **~11:25-14:25:** EvalSuite runner 43; ~14:10-15:40 runner 44.
- **~15:50 to Fri 00:15:** EvalSuite T0 (share lock, 14 GB, no timing).

## Friday 5-6 Nov night, from 00:30
1. MTP Head P-OFF training (~90 min).
2. BigMoE: the FAR-G1 capture + a C 224 relaxed / Rule C run.
3. Leap NATIVE-FRI arms.

## Slots for new short tests
- Kinds: decision-map arms (~5-10 min, 18 GB) and CPU replays.
- Thursday by day: share slots beside the runners when memory allows (one > 5 GB job at a time by day).
- Otherwise the Fri 00:30 night, after the listed items.
- The orchestrator decides placement. Ask through cloud3/REQUESTS.md.
