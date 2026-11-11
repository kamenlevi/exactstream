# RUNBOOK: first Qwen3.8-Flash-Next (125B, 4-bit) end-to-end run

The overnight download (`scripts/night_download.sh`, log `models/night.log`) ends with a line
`done; ... on disk; free ...`. Then, in this order:

## 0. Preconditions (stop if any fails)
```
grep "done;" ~/BigMoE/models/night.log && df -h /        # free must be >= 200 GB
```
- Sizes: each file in `models/bare/` must match `models/files_sizes.txt`
  (`while read f s; do [ $(stat -f %z models/bare/$f) = $s ] || echo BAD $f; done < models/files_sizes.txt`).
- Memory: `memory_pressure | tail -1` must show ≥ 60 % free, and `peer top` must show no other large
  MLX model loaded. If SpecDecode's server (claude-spec) holds a model, ask it to unload first.
  This run needs ~48 GB of GPU memory (see budget).
- Lock: `peer lock quiet --queue 30m "BigMoE first 125B run"` in the background; start only once granted.

## 1. Memory budget at 60 % resident (C = 307 of 512 experts per layer)
| part | GB |
|---|---|
| expert slot pools: 307 × 2.765 MB × 48 layers | 40.7 |
| dense trunk (attention, GDN, hyper-connections bf16, shared experts, routers, embed, lm_head) | ~3.7 |
| MTP head (own 512 experts, resident) | 1.7 |
| KV (short prompts) + GDN state + scratch; prefill scratch pool ≤ 1.4 GB transient | ~1.5 |
| **total** | **~47.6** (MLX max recommended working set: 51.8 GiB = 55.6 GB) |
The n-gram table (32 GB) is read from SSD by MTPLX (16 rows per token); it is not resident.
First smoke run at **55 %** (C = 282: pools 37.4 GB) to leave headroom.

## 2. Order of experiments (each a separate process; evict the page cache before timing runs)
```
cd ~/BigMoE
# a. smoke: loads, decodes 16 tokens, gate works on qwen4_exp (no timeouts), text is sane
.venv/bin/python scripts/bench.py 0.55 ar 16
# b. exactness, layer level: streamed SwitchGLU == full-tensor SwitchGLU, bit for bit (layers 0, 24, 47)
.venv/bin/python scripts/check_layer.py ~/BigMoE/models/bare 0.55 0,24,47
# c. exactness, model level: two budgets must give identical tokens (different slot layouts, misses)
.venv/bin/python scripts/evict.py models/bare
.venv/bin/python scripts/bench.py 0.55 ar 128       # note sha
.venv/bin/python scripts/bench.py 0.60 ar 128       # same sha required
# d. headline: 60 %, plain + MTP depth 2/3, cold SSD (evict before each)
.venv/bin/python scripts/bench.py 0.60 ar,mtp2,mtp3 128
# e. look-ahead recall on qwen4 (hyper-connections: no exact norm swap) -> decides whether to try prefetch
.venv/bin/python scripts/recall_target.py 0.55 64
# f. if recall@12 >= ~0.6: prefetch run with BIGMOE_SPARE=20 BIGMOE_PREFETCH=10 BIGMOE_PREFETCH2=10
```
MTPLX trunk compile is switched off in the scripts (`MTPLX_COMPILED_VERIFY=0`, `MTPLX_QWEN4EXP_COMPILE=0`,
`MTPLX_COMPILE_AR_FORWARD=0`, `MTPLX_COMPILED_TARGET_PREFIX=0`): a traced forward would run the per-layer
gate's Python only once.

## 3. What to record (scripts write `data/runs/*.json`; copy headline lines to FINDINGS.md)
decode tok/s, time to first token, peak GB, misses / stalled layers / SSD ms per token, fast-path hits,
MTP tokens per pass, token sha, first 60 characters of text. Then send the orchestrator one line:
`RESULT: Qwen3.8-Flash-Next 4-bit on 64 GB, exact: <plain> tok/s plain, <mtp> tok/s MTP k=<k>`.

## 4. Abort conditions (stop, `peer unlock`, write down why)
- `memory_pressure` reports critical, or swap grows by > 2 GB during a run.
- `RuntimeError: bigmoe: ... spin gate(s) timed out` (the pass is invalid; never report its numbers).
- `check_layer.py` reports anything other than 6/6 bit-identical, or the two budgets' shas differ.
- MTPLX refuses to load (version / missing qwen4 support): record the error; do not patch MTPLX in place.
- Repetitive or garbage text in the smoke run.

## 5. Known open points
- MTPLX's own MTP path (`generate_mtpk`) may compile or fuse the verify forward despite the env switches; if
  timeouts appear only with mtpN, run `ar` only and note it.
- Look-ahead on qwen4_exp uses layer L's mixed MoE input with layer L+1's router (no shared norm);
  recall is unmeasured until step e.

## Status at session stop (2026-10-18 ~01:30)
- Download running detached (`night_download.sh`, pid in `models/night.pid`): 17 / 105 GB at 01:25,
  ~6 MB/s → expected done ~06:00; 200 GB floor checked per file. No Claude watcher is attached.
- All scripts above dry-run on the 35B (check_layer 6/6 bit-identical; bench.py via MTPLX OK).
- Prefetch default for step f: `BIGMOE_SPARE=12 BIGMOE_PREFETCH=12` (depth 2 measured no better).
