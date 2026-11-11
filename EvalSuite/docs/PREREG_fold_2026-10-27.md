# B12_FOLD on vs off: pre-registration (2026-10-27 22:25 EET, before any run)

Request: Breakthrough, Tue 22:3x (NewIdeas' call). It asks for B12_FOLD on vs off on the standard 12 GB set (the MTP rows, `decode._gen_mtp`). The default flip waits for this verdict.

The fold is Fit12's exact MTP-path stack, behind `B12_FOLD=1` in `~/Breakthrough/scripts/bt_b12.py`:
- PFR on every MTP prefill;
- C_m per prompt;
- QSA rows sized once to n + 2048 + 16;
- the prefilled state moved into the decode (MOV);
- ALLWAIT inside `spec125.generate` only.

Breakthrough's claim: exact tokens, +2.7 to +8 % warm decode, −30 % MTP prefill at 8.5K, and peaks ≤ 11.95 GB at its points.

## Arms (one config per process; both B12LL = `load_b12(0.10, mtp=True)` + `B12_LATE3=2,10`, `B12_NORM=mass`)

| variant | engine | B12_FOLD | other pinned switches |
|---|---|---|---|
| `b12llfoff` | v3l | 0 | B12_TRIMS 1, B12_PFR 1, B12_ALLWAIT 0, B12_CACHE_MB 64 |
| `b12llfon` | v3m | 1 | the same, plus B12_FOLD_CM `2700:64,8600:60,16500:56,32800:47` (0228898's default) and B12_FOLD_ROWS 2048 (every cap here is ≤ 1536) |

Settings common to both arms:
- **Environment:** B12LL's suite environment (BIGMOE_PF_GROUP 64, ASYNC_EVERY 1, TIER_PF_GROUP 64, MTPLX_PREFILL_CHUNK_SIZE 512, RESET_GEN 0, the MTP stream, MTP_C 64, MAXK 80, FR-Spec, the MTPLX commit flags), with the repetition guard off. This is the environment of Breakthrough's `fold_check.py`, which made its fold hashes and peaks.
- **Pinned defaults:** every pinned value equals today's bt_b12 default. They are pinned so that a default flip during the run cannot change an arm.
- **Breakthrough code:** HEAD 5962fef (bt_b12 at 0228898). sha256 of `bt_b12.py` 664b010b0d, v3l build d50006dffc, v3m build 5d7c3d06a7. SpecDecode is at 2d4fa7b.
- **Provenance:** each chunk records these hashes in `results/<v>/chunks.jsonl`. A chunk run on other code is reported as such.
- **Not the verdict pin:** these arms do not use the suite's earlier pin (v3f, trims off). Both arms run the shipped engine, as the request says.

## Rows

239 rows, run in this order:
1. agent 29 (prompts of 1.6K to 12K tokens, 128 new tokens);
2. math 50;
3. ifeval 100;
4. humaneval 60.

The last three are the core 210 of every 12 GB verdict. Agent comes first because its long prompts are where C_m per prompt (60 to 64) and the MTP PFR actually act.

Decoding is greedy and non-thinking, through `decode._gen_mtp`: `b12_prefill_state` (512 / 4096), then one `spec125.generate` 3:3:0 per prefilled state, stopping at EOS.

## Procedure

- **Chunks:** 10 per arm, with the same row ranges in both arms: 17, 16, 12, 13, 34, 24, 22, 34, 33, 34 rows. Each is about 12 minutes at B12LL's earlier speed.
- **One process per chunk:** each chunk runs in its own process under its own queued quiet lock, released between chunks (RULES: runs over 10 minutes lock per chunk of at most 15 minutes).
- **Alternation:** the arms alternate in ABBA order (chunk 1 off then on, chunk 2 on then off, ...), so drift falls evenly on both arms.
- **Restarts:** a chunk cut short is finished by the next call of that arm, which runs up to the same cumulative row count.
- **Guards:**
  - memory gate: 18 GB free;
  - a stage stops (its whole process group) if swap grows more than 2 GB during it;
  - no chunk starts while `soft_end.sh` runs or PAUSE exists.
- **Detached runner:** `tools/day_runner24.sh`, log `results/day24_runner.log`, deadline Wed 07:00.

**Logged per row** (new in `decode.py`, logging only):
- **Timing:** prefill seconds (the `b12_prefill_state` call) and decode seconds (the `spec125.generate` call).
- **Decode-phase SSD:** reads and MB.
- **Fold counters:** C_m after the prefill, ALLWAIT layer calls, and PFR releases.
- **Footprint:** at row start and end, the max sampled every 25 ms, and the lifetime peak.
- **Pauses:** seconds with PAUSE present, and seconds the process was frozen.

## Answers and rules

1. **Tokens.** Per row, `completion_ids` on vs off.
   - **EXACT:** all 239 rows are identical.
   - **Otherwise:** report the rows and their first differing token. Then re-run the differing rows once in each arm (a fresh process per arm, same order), to tell a fold effect (off repeats itself, on differs) from run-to-run noise in either arm.
2. **Task scores.** The suite's standard table for each arm against the 4-bit reference (math, IFEval, HumanEval+, answer agreement against the floor, truncations; the 5 rules), and each arm against the other.
   - Information: tokens against B12LL's saved run (`results/b12ll`, v3f, trims off), i.e. whether today's shipped code still gives the judged B12LL tokens.
3. **Speed.** Per row: decode tok/s (tokens / decode seconds) and prefill seconds.
   - Per set and overall: the geometric mean of the per-row on/off ratios, and the ratio of total seconds.
   - Prefill matters mainly on agent rows; core prompts are ≤ 520 tokens.
   - Rows with PAUSE or more than 1 s frozen in either arm are left out of the timing.
   - No pass line: Breakthrough asked for the numbers, and the flip is NewIdeas' call.
4. **Memory.**
   - **PASS (≤ 11.95 GB):** no fold-on row's peak (sampled max or lifetime peak) passes 11.95 GB. Otherwise, report those rows (set, prompt length, C_m, footprint).
   - Fold-off is reported the same way, for comparison.
   - Caveat: processes here are chunks of 12 to 34 prompts (about 12 minutes), not one process for all 239 rows.
5. **SSD.** Decode-phase MB per generated token, per set and arm.
