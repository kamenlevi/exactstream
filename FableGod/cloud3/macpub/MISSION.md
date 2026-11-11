# MISSION CLOUD3 (consolidated, v3): fewer distinct experts per kept token, at the same quality
(Thu 5 Nov 2026 ~00:05 EET = Wed 22:05 UTC. Written by the orchestrator, approved by Kamen, the project owner. This file
replaces every earlier message; any "not delivered" messages in your window can be ignored.)

## 0. Your situation (read this section twice)
- **Who you are.** You are Fable at max effort, a cloud session working for Kamen's local LLM speed project. The orchestrator, a Claude session on Kamen's Mac, wrote this.
- **Budget.** Kamen's remaining cloud credits. They end on Thu 5 Nov at ~11:00-12:00 EET.
  - Use them fully and well: work continuously until you are stopped.
  - There is deliberately more work than one night can finish. Never end your turn while work remains.
- **Channels.** Messages from other sessions may not reach you. Your two channels are files on GitHub:
  - OUT: your branch `cloud3-reports` (section 2), which the orchestrator reads every 5 minutes.
  - IN: `ORDERS.md` on branch `cloud3-mac` of kamenlevi/350-tps. Read it at least every 20-30 minutes: `git -C ~/c3mac pull -q && cat ~/c3mac/ORDERS.md`. Orders are numbered; follow every new one, since they override this file.
- **The repo you started in** (kamenlevi/OjoX) is Kamen's unrelated image-viewer app.
  - Never edit, commit to or push any of its existing branches.
  - Never open PRs or issues anywhere. Never post anything anywhere.

## 1. The exact goal
Find, price and design ways to cut the number of DISTINCT routed experts per kept token while keeping benchmark quality.

That one number drives two walls at once:
- **(a) Compute at the 45 GB line.** The expert kernel is at its ceiling, so expert time is proportional to distinct (expert, layer) tiles per check.
- **(b) SSD bytes at the 24 / 12 GB lines.** Every distinct expert not in RAM is an SSD read.

It also matters to every MoE that verifies drafts or serves batches, so general methods are a plus.

**Target:** >= 30 % fewer distinct experts per kept token at the FAST quality lines (section 3.5).

**Stretch:** >= 50 %, or anything that changes the scaling (e.g. union growth per row).

## 2. Reporting: automatic, zero token cost
1. **Where.**
   - First choice: a new branch `cloud3-reports` on kamenlevi/350-tps, containing only a folder `cloud3/`.
   - If pushing there fails 3 times: the same branch pushed to kamenlevi/OjoX as `research/cloud3-reports`.
   - Never touch any other branch. Never merge, rebase or force-push.
2. **What you write** (in `~/c3/cloud3/`):
   - `STEP.md`: one line, what you are doing now. Update it when you switch tasks.
   - `REPORT_P0.md` ... `REPORT_P3.md`: the deliverables of section 5, written as you go.
   - `notes/<short-name>.md`: one note per paper or source.
   - `REQUESTS.md`: test requests for the Mac (section 8).
   - Never let findings exist only in your head or your transcript.
3. **The heartbeat is a background loop (section 7).** Every 10 minutes it writes HEARTBEAT.md, commits and pushes `cloud3/`. You spend no tokens on it. If it dies (no new commits), restart it.
4. **How the orchestrator judges you.** A watcher reads your branch every 5 minutes. It alarms if:
   - commits stop for 25 minutes;
   - reports + notes stop growing for 60 minutes;
   - a phase misses its time.
5. **Phase ends.** When a phase is complete, write `PHASE <n> COMPLETE <time>` as the last line of its report.

## 3. The project: everything you need
### 3.1 Model and machine
- **Model:** Qwen3.8-Flash-Next, a 125B MoE.
  - 48 layers: 36 gated-delta-net linear-attention layers + 12 full attention (every 4th; 24 query heads, 2 KV heads). Hidden 2,560.
  - 512 routed experts per layer, top-10, SwiGLU width 640, plus 1 shared expert (width 640). Vocabulary 248,320. One MTP layer.
  - One expert ~4.92 M weights. Stored as "3-bit tiers": ranks 0-1 at 4 bits, ranks 2-9 at 3 bits (the REF2-T reference). One expert record is ~1.98 MB.
- **Machine:** one MacBook Pro M5 Pro, 64 GB unified memory. GPU streaming 284-286 GB/s measured; internal SSD ~12-14 GB/s. Experts not in RAM stream from the SSD.

### 3.2 Lines and modes
- **Lines:** 45 GB (main), 24 GB and 12 GB of RAM, measured as the lifetime peak physical footprint.
- **EXACT:** bit-identical to token-by-token decoding of its reference.
- **FAST:** keeps benchmark scores (section 3.5).
- Routing changes (fewer or different experts) are FAST-only.

### 3.3 Decoding and the check
- **Drafter:** DFlash2, a ~0.5B block drafter (5 layers). It proposes 16 tokens; a 17th draft gates the last "bonus" row (DRAFT17).
- **The check:** the 125B verifies all 16 rows in one pass, keeping the agreeing prefix plus one bonus token.
- **Kept per check:** code ~10.5 (per-token agreement ~93 %), math ~7, chat ~3.
- **One FAST check on code at 45 GB, ~90 ms:**
  - experts ~38-43 ms: ~3,700-3,840 distinct (expert, layer) tiles = ~78-80 distinct experts per layer, 7.38 GB read per check, measured;
  - SSD wait ~24 ms;
  - dense ~12 ms;
  - drafter 5.4 ms;
  - commit ~5 ms.
- **One EXACT check:** ~126 ms, of which ~55-60 ms is SSD wait [P].
- **The SSD wait law** (FAR-WAIT, measured on 106 runs): wait per check ~ 10-16 ms floor + 0.155 ms per missed expert (~14 GB/s).
- **The expert kernel is at its ceiling** (`FableGod/KERNEL90.md`):
  - 77 % of streaming EXACT, 82-83 % FAST; twelve bitwise redesigns KILLed.
  - It is bound by compute per byte: the matrix unit computes 16 rows per tile, but only ~2 rows are real per expert.
  - So expert time scales with tiles, not bits.

### 3.4 Today's numbers and the goals
- **Today:**
  - FAST ~110-117 tok/s on code at 45 GB; EXACT ~83.
  - 12 GB EXACT: 40.8 tok/s in 11.35 GB.
  - 24 GB FAST: ~19.5 MB of SSD per kept token on code; ~88 MB on chat (the only chat route that passes).
- **Goals:**
  - 300 tok/s FAST and 200 EXACT on code at 45 GB (300 = ~35 ms per check at 10.5 kept);
  - <= 10 MB SSD per kept token at 24 GB;
  - ~120-150 tok/s at <= 12 GB.

### 3.5 The quality rule for FAST
- **First screen:** Breakthrough's decision map, 36 windows. It reports flip rates vs the exact reference, against the lines code 1.34 %, IFEval 5.42 %, multilingual 3.69 %:
  - PASS = under all three;
  - BETWEEN = near a line;
  - KILL = clearly over.
- **Then:** the EvalSuite same-scores test.
- **The 16-row trap:** union arms must be judged at LIVE length. Rows past a running draft confidence product of 0.3 are not live. 16-row unions overstated chat quality before.

### 3.6 Facts about expert unions [K]
- **Union of a 16-row check on code:** ~78-80 distinct experts per layer.
- **Rejected rows:** in R-24's 20 real checks, the per-layer union was 89.3 experts, and 63.8 without the rejected rows. So 29 % is pulled only by rows that get rejected. Rejected rows also hold 32 % of the cache reuse (FAR).
- **Width scaling:** the union grows ~k^0.67 with rows k. A sibling row adds 2.35-2.55 new experts (M-10). Three prompts in one pass share experts only at chance level (R-63S).
- **The router is flat:** top-8 holds ~88 % of the top-10 weight.
  - top-6 = 2.98x and top-4 = 9.70x the error of the 3-bit tier (L1-17c, KILL);
  - adaptive-k p 0.8 KILLed (L1-17b);
  - dynamic top-k at tau 0.06 KILLed (R-26).
- **Sure rows (the drafter is confident):**
  - top-2 or even top-1 (+ the check's union members + residents) PASSes: QOM, TOP1-QOM, Q1, QOM-CF0 0.42 / 2.53 / 1.77 %;
  - unsure / middle rows are where quality breaks (DM topk5: sure 0.00 %, middle 8.6-14.5 %).
- **Union-member pricing:** UNION-THETA: code x0.720 of the union at cf 0.99.
- **Tried and dead or weak:** per-pass expert budget b16 (BETWEEN), REACH-CUT, STOPPRED, PRUNE-COLD (KILL), pre-routers (top-2 precision <= 0.68, KILL), cheap runahead slices (KILL), RUNAHEAD (KILL), sibling-union trees (KILL), FRUGAL10 (KILL), wide verify (weak).
- **Already passed:** DM P-frame MoE (reuse the previous position's routed output; PASS for layers 12-35 on sure rows, one 12-layer group at a time).

### 3.7 Neighbouring results from this week
- A 2.4-bit trellis-coded store FAILED chat quality. Data-aware codes cost 2-3.5x more KL per unit of output error than truncation here.
- Relaxed acceptance: kept 10.5 -> 11.7 at 1.2 % changed tokens.
- Rule C (relax only inside '#' comments): MBPP x1.125 kept, HumanEval+ x1.074, 0 code changes.
- FAR: a hybrid near + far draft forecaster gives -20 % cache misses on CPU.
- Kept per check is capped at ~12-13 by every known drafter means.

## 4. Reading: what and how (reading copy `~/350-tps`)
- **Must read:**
  - `FableGod/TRIED/TRIED.md`: ~1,280 tried ideas. Grep it for every idea you consider: union, top-k, top-1, top-2, skip, budget, capacity, sure, QOM, Q1, OEA, reach, sibling, tree, P-frame, reuse, prune, merge, shared expert, router, pre-route, runahead.
  - `FableGod/TRIED/NOVELTY.md`, `FableGod/KERNEL90.md`.
  - `FableGod/goal/GOAL.md` (its 8.8 GB and dense ~6 ms are estimates; use 7.38 GB and ~12 ms).
  - `FableGod/full/FULL.md`, `FableGod/far/FAR.md`, `FableGod/fablec/BYTES10.md`, `FableGod/findings/`.
- **Real data** (inspect the fields first; never assume them):
  - `Leap/data/r24_routes.npz`: per-row expert routes of 20 real drafted 16-row checks. Reader: `Leap/scripts/gmr24.py`; result: `Leap/data/gmr24.txt`.
  - `MoonShots/data/stall/*_trace.json`.
  - `FableGod/floor_rows.json`, `FableGod/drafter_anatomy.json`, `FableGod/relax_positions.json`.
  - More: `find . -name '*.npz' -o -name '*.json' | xargs ls -la`.
- **Code** (read-only): `BigMoE/` (r57 screen, ext), `Leap/scripts/`, `Breakthrough/` (the decision map), `NewIdeas/research/`.
> (a quote from the project owner, kept private)
  - Triage first: abstract, figures, tables, conclusion. Decide relevant / maybe / no.
  - Go deep on relevant papers whatever their length. Pull the PDF, convert it to text, and read it by section (method, experiments, appendix tables, limitations); grep to jump.
  - One note per paper in `cloud3/notes/`: link, what they did, numbers, how it maps to us, open questions. Never re-read a paper; reuse your notes.
  - Never paste large raw text or data into your context. Compute with scripts and read summaries; read big files in parts.
  - Prefer primary sources: arXiv, venues, official code.

## 5. Phases and times (EET)
### P0, setup and proof of understanding (as soon as possible, <= 30 min)
Do section 7. Then write REPORT_P0.md:
- your sandbox (CPU, RAM, disk);
- 15 facts from the must-reads that matter to your task (file:line);
- the full list of TRIED lines in your field (ID, verdict, one number);
- what the real data files contain (fields, shapes).

### P1, literature, until 03:00
- **Search** 2023-2026 work that reduces distinct experts per token or per batch / verify pass at the same quality. Cover at least:
  - batch- / union-aware expert selection, i.e. reassigning a token to an expert other rows already load (e.g. "Lynx");
  - verification-time expert caps (MoE-Spec);
  - speculative expert prefetch / eviction: SpecMD 2602.03921, MoE-SpeQ 2511.14102, SP-MoE 2510.10302, SPICE 2608.21240;
  - dynamic / adaptive top-k: "Harder tasks need more experts" (top-p routing), Ada-K, DynMoE;
  - expert skipping and pruning (NAEE);
  - expert merging (MC-SMoE, HC-SMoE);
  - inter-layer expert affinity (ExFlow);
  - shared-expert substitution;
  - router bias toward loaded experts;
  - MoE speculative-decoding analyses;
  - anything else you find.

  Verify that every paper exists, with a link.
- **One table row per method:**
  1. name + link;
  2. the mechanism (<= 3 lines);
  3. its TRIED status (line ID + verdict, or "untried");
  4. how it maps to our 16-row check (sure / unsure / rejected / bonus rows);
  5. the expected cut in distinct experts per kept token, [P] on section 3 numbers;
  6. the quality risk [G], with the reason;
  7. what it buys: tok/s at 45 GB FAST and MB per kept token at 24 GB;
  8. training needed (yes / no);
  9. the cheapest test (a <= 10-min decision-map arm or a CPU replay) with PASS / KILL lines.
- Price the top 10 precisely, with test specs.
- REPORT_P1.md. End with `PHASE 1 COMPLETE <time>`.

### P2, the anatomy of our own unions on the real data, 03:00-04:45
- Where a 16-row check's distinct experts come from: accepted vs rejected rows, row position, ranks 0-1 vs 2-9, gate weight, overlap between rows, layers.
- Upper bounds: how much a perfect policy of each kind could remove.
  - Kinds: reassignment within a gate-weight tolerance, sure-row cuts, rejected-row cuts, reuse across layers or positions.
- Say exactly which data is missing.
- REPORT_P2.md. End with `PHASE 2 COMPLETE <time>`.

### P3, THE BREAKTHROUGH ATTEMPT, 04:45 until stopped
- Mechanisms aimed at the biggest sources P2 measured. Both kinds count: new mechanisms nobody has published, and published ideas whose potential was overlooked for this setting.
- For each:
  - the mechanism;
  - [P] numbers on P2's data, with simulations on the real data where possible;
  - a prior-art check (published / adapted / new, with links);
  - a <= 10-min test with PASS / KILL lines.
- Aim for 2-3 serious candidates.
- At 08:00, add the line `P3 CHECKPOINT 08:00` with your best 2 so far.
- From 10:30, keep a paper-style summary of the best result at the top of REPORT_P3.md.

## 6. Rules
- **Marks:** [K] known, with a source (file:line or a URL); [P] arithmetic, with its inputs; [G] judgement. No number without its mark.
- **No repeats:** never re-propose an idea TRIED killed without a concrete new reason.
- **No inventions:** never invent engine facts. If unsure, say which file would settle it.
- **Use section 3's measured numbers** over older estimates in the files.
- **Downloads:** allowed in your sandbox (public models / datasets, e.g. a tiny public MoE for proxy routing experiments on CPU). No storage limit. Log each download in a note (name, URL, size). No paid services, no accounts.
- **Experiments:** run each with a clear question and a time box. No unattended endless loops.
- **Style:** plain, short sentences. No personal data (none exists in your inputs).

## 7. Start checklist (do it literally)
1. **Orders channel:** `git clone -q --depth 5 -b cloud3-mac https://github.com/kamenlevi/350-tps ~/c3mac && cat ~/c3mac/ORDERS.md`
2. **Reading copy:** `git clone -q https://github.com/kamenlevi/350-tps ~/350-tps`. Read-only. Skip if you already have it.
3. **Reporting repo:**
   ```
   mkdir -p ~/c3/cloud3/notes && cd ~/c3 && git init -q && git checkout -q -b cloud3-reports
   git config user.name cloud3 && git config user.email cloud3@localhost
   echo "P0: setup" > cloud3/STEP.md && git add cloud3 && git commit -qm "start"
   git remote add origin https://github.com/kamenlevi/350-tps && git push -u origin cloud3-reports
   ```
   If that push fails 3 times:
   ```
   git remote set-url origin https://github.com/kamenlevi/OjoX && echo research/cloud3-reports > .pushref && git push -u origin cloud3-reports:research/cloud3-reports
   ```
4. **Heartbeat loop:**
   ```
   cat > ~/c3/beat.sh <<'EOF'
   #!/bin/bash
   cd ~/c3; PUSHREF=$(cat .pushref 2>/dev/null || echo cloud3-reports)
   N=$(cat cloud3/.beat 2>/dev/null || echo 0); RX0=$(awk '/:/ && $1 !~ /lo:/ {s+=$2} END {print s}' /proc/net/dev)
   while true; do
     N=$((N+1)); echo $N > cloud3/.beat; RX=$(awk '/:/ && $1 !~ /lo:/ {s+=$2} END {print s}' /proc/net/dev)
     { echo "beat: $N"; echo "time: $(date -u +%Y-%m-%dT%H:%MZ) / $(TZ=Europe/Sofia date +%H:%M) EET"
       echo "step: $(head -1 cloud3/STEP.md 2>/dev/null)"
       echo "reports_bytes: $(cat cloud3/REPORT_*.md 2>/dev/null | wc -c)"
       echo "notes_bytes: $(cat cloud3/notes/*.md 2>/dev/null | wc -c)"
       echo "net_MB: $(( (RX - RX0) / 1000000 ))"; } > cloud3/HEARTBEAT.md
     echo "$(date -u +%H:%MZ) beat $N" >> cloud3/LOG.md
     git add -A cloud3 && git commit -qm "beat $N (auto)"; git push -q origin cloud3-reports:$PUSHREF 2>>cloud3/push_errors.log
     sleep 600
   done
   EOF
   chmod +x ~/c3/beat.sh && (nohup ~/c3/beat.sh > /tmp/beat.out 2>&1 &)
   ```
5. Check that the first auto beat reached GitHub: `git ls-remote origin` shows the branch. Then start P0.

## 8. The Mac's orchestration channel
- **MACSTATE.md** on branch `cloud3-mac`, refreshed every ~10 min: the Mac's live locks, the GPU schedule and free slots. `git -C ~/c3mac pull -q && cat ~/c3mac/MACSTATE.md`.
- **Requests.** To get something run on the Mac (a decision-map arm, a replay that needs data not in the repo, a data export), append a request to `cloud3/REQUESTS.md`:
  - ID;
  - exactly what to run;
  - why;
  - minutes and MEM GB;
  - PASS / KILL lines;
  - what data comes back.
- The watcher wakes the orchestrator, who places the request in the Mac's queue. The GPU is busy until ~07:10; short slots open Thursday by day.
- Results come back as `RESULTS/<ID>.md` on branch `cloud3-mac`, or as an order in ORDERS.md.
