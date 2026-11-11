# ORDERS for Cloud Fable 3, from the orchestrator
Numbered. Read this file every 20-30 minutes and follow every new order.

0. 23:58 EET: no orders yet. Start with MISSION.md section 7, then P0 -> P1 -> P2 -> P3.

1. 00:06 EET: NEW FILES on this branch, read them before P1 deepens:
   - STATE.md: what the project has achieved, what is killed, what is in flight.
   - REPO_MAP.md: which of our documents to read for your wall, in order (FLOOR_125B.md and BREAK_THE_WALL.md first).
   - DATA.md + data/: the routing of all 36 decision-map windows (13 MB), MoonShots exports (14 MB) and the cap45 replay data (3 MB). Use them for P1 simulations and as the main dataset in P2.
   Fetch with: git -C ~/c3mac pull -q (the branch now holds ~30 MB of data).

2. 00:28 EET, PAPERS + P1 REOPENED. Your sandbox can't reach arXiv; the Mac can, and it now delivers papers to you automatically.
   - First, try your WebFetch tool (it fetches outside your sandbox) on https://arxiv.org/abs/<id> and https://arxiv.org/html/<id>. If it works, use it.
   - Otherwise, request full texts by adding lines to cloud3/REQUESTS.md, one per line, at the start of the line:
     `PAPER 2511.14102` (arXiv id), or `PAPERURL https://openreview.net/pdf?id=...` (openreview, aclanthology, proceedings.mlr.press, neurips, usenix, dl.acm).
   - Within ~10 minutes (your next auto-beat + the watcher's poll) the PDFs appear on the PRIVATE branch research/cloud3-papers of kamenlevi/OjoX:
     `git clone -q --depth 1 -b research/cloud3-papers https://github.com/kamenlevi/OjoX ~/papers`, later `git -C ~/papers pull -q`. LIST.md lists them.
     MoE-SpeQ (2511.14102.pdf) is already there as a test. Use pdftotext / pypdf.
   - P1 is REOPENED as P1b, a full-text pass, until 03:00 (Kamen: literature until 03:00):
     - request the full text of the top ~15 papers in your table now;
     - read method, experiments, appendix tables and limitations;
     - re-check every [S] number against the full text;
     - deepen the top 10;
     - add a "P1b: full-text pass" section to REPORT_P1.md, ending with "PHASE 1b COMPLETE <time>".
   - While papers are on their way, work on P2's data preparation in parallel; don't idle. P2 proper starts at 03:00, P3 at 04:45, as planned.

3. MEASURE YOUR OWN CREDIT SPEND (Kamen asked; zero token cost). Replace your heartbeat loop with this version.
   It adds a `tokens:` line: your exact token counts, summed from your own session transcript (.jsonl), and an estimate in USD at Fable 5.1 list prices
   ($10 / MTok input, $50 output, $20 cache write (1 h, upper bound), $0.25 cache read).
```
pkill -f 'c3/beat.sh'
cat > ~/c3/beat.sh <<'EOS'
#!/bin/bash
cd ~/c3; PUSHREF=$(cat .pushref 2>/dev/null || echo cloud3-reports); TD=${TD:-$HOME/.claude}
N=$(cat cloud3/.beat 2>/dev/null || echo 0); RX0=$(awk '/:/ && $1 !~ /lo:/ {s+=$2} END {print s}' /proc/net/dev)
tok() { find "$TD" -name '*.jsonl' -type f -print0 2>/dev/null | xargs -0 cat 2>/dev/null | jq -c 'select(.message.usage? != null) | {id: (.message.id // .uuid), i: (.message.usage.input_tokens // 0), o: (.message.usage.output_tokens // 0), w: (.message.usage.cache_creation_input_tokens // 0), r: (.message.usage.cache_read_input_tokens // 0)}' 2>/dev/null | jq -s -r 'group_by(.id) | map({i: (map(.i)|max), o: (map(.o)|max), w: (map(.w)|max), r: (map(.r)|max)}) | {i: (map(.i)|add // 0), o: (map(.o)|add // 0), w: (map(.w)|add // 0), r: (map(.r)|add // 0)} | "in \(.i) out \(.o) cache_write \(.w) cache_read \(.r) est_usd \(((.i*10 + .o*50 + .w*20 + .r*0.25)/1e6*100|floor)/100)"' 2>/dev/null; }
while true; do
  N=$((N+1)); echo $N > cloud3/.beat; RX=$(awk '/:/ && $1 !~ /lo:/ {s+=$2} END {print s}' /proc/net/dev)
  { echo "beat: $N"; echo "time: $(date -u +%Y-%m-%dT%H:%MZ) / $(TZ=Europe/Sofia date +%H:%M) EET"
    echo "step: $(head -1 cloud3/STEP.md 2>/dev/null)"
    echo "reports_bytes: $(cat cloud3/REPORT_*.md 2>/dev/null | wc -c)"
    echo "notes_bytes: $(cat cloud3/notes/*.md 2>/dev/null | wc -c)"
    echo "net_MB: $(( (RX - RX0) / 1000000 ))"
    echo "tokens: $(tok)"; } > cloud3/HEARTBEAT.md
  echo "$(date -u +%H:%MZ) beat $N $(grep '^tokens:' cloud3/HEARTBEAT.md)" >> cloud3/LOG.md
  git add -A cloud3 && git commit -qm "beat $N (auto)"; git push -q origin cloud3-reports:$PUSHREF 2>>cloud3/push_errors.log
  sleep 600
done
EOS
chmod +x ~/c3/beat.sh && (nohup ~/c3/beat.sh > /tmp/beat.out 2>&1 &)
```
   - If the first new beat shows `tokens: in 0 ...`, your transcript lives elsewhere. Find it (`find / -name '*.jsonl' -size +10k 2>/dev/null | grep -v proc | head`), then restart the loop with `TD=<that folder>` in front of the nohup.
   - Count only your own session's files if the folder holds others: point TD at the session's folder.

4. 00:35 EET, YOUR REQUESTS C3-R1..R8 ARE RECEIVED AND PLACED. Good requests.
   - R1-R6 (map arms): Breakthrough, in the Fri 00:30 night (the GPU is held by EvalSuite until ~07:10, then by BigMoE, the NAX6 certification and the store conversion through Thursday). Results are published after your credits end; the team acts on them.
   - R7 (router scores export): Breakthrough, at the first CPU gap Thursday. It appears in cloud3-mac/data/decision_map_scores/ if it lands before you stop.
   - R8 (group-count log): BigMoE, in its Fri night run.
   - Don't wait for any of them. Price and rank with what you have; state the assumption each pending result would settle.
   - Order 2 (paper delivery, P1b until 03:00) still stands.

5. 00:37 EET, BREAKTHROUGH'S ANSWER ON R1-R7: all feasible, and none was measured before. Nearest measured: q1u_0.9_live, qom_0_live_bq (R4's resident-keep form), C / screen 6 at 147 slots. Two choices are yours. Answer as new lines in REQUESTS.md starting with `- C3-R6 choice:` and `- C3-R7 choice:`.
   - R6 massB_0.5:
     - (a) masks precomputed from the baseline's router scores: cheap, but the selection ignores the perturbation's drift;
     - (b) the live rule on the perturbed scores, a new harness gate, ~1-2 h build.
     - Default (a), flagged.
   - R7 scores: the router's top-10 scores for the ANSWER rows (10,316 rows, 36 windows, 48 layers; softmax, rank-sorted, renormalised over the top 10, float16) already exist. They are exported Thursday at the first CPU gap. Prompt-row scores need a ~5 min GPU capture. Do you need them?
   - All arms run after Thursday's store conversion (the native layout); results stay comparable.

6. 00:42 EET, P2 IS ACCEPTED: good work. The biggest source is the kept rows' rank 2-9 tail (about two thirds of the union), and singletons are 59-67 %.
   - Timetable adjustment: finish P1b (the full texts of the papers you requested) by 03:00, then go STRAIGHT to P3, the breakthrough attempt. Don't redo P2.
   - Revisit a P2 number only where a paper changes it.
   - P3 now has ~03:00 to the end of your credits. Keep the 08:00 checkpoint and the 10:30 summary.
   - The best P3 targets, from your own P2: the rank 2-9 tail of kept rows without quality loss; making rejected rows nearly free; singleton-heavy layers.

7. 01:03 EET, ON REQUESTS:
   - R9 withdrawal noted.
   - R10 (REACH-CUT-X): BigMoE says it isn't cheap. REACH-CUT exists only as a map arm, so the live engine needs an optional-use predicate, per-row reach, the stop rule and tests (~2 h). Thursday is full, so it gets built Friday for a later night, after FAR-G1.
   - From now on, requests are BATCHED: the orchestrator routes new ones at most every 2 h, and all of them at ~06:50.
   - Nothing can run on the Mac's GPU before ~11:00 Thursday (EvalSuite, then BigMoE, the NAX6 certification and the store conversion).
   - So keep requests for the decisive tests of your best P3 candidates, with exact PASS / KILL lines, and don't wait on them.

8. 01:30 EET, SPEND AND PACE. Your own meter says ~$118 of ~$140 credits by 01:25, about $21 per hour. After the credits you continue on Kamen's weekly plan, and a guard stops you at 92 %. So from now on every hour must produce new results.
   - No waiting or polling loops. Never idle until a checkpoint time: 08:00 and 10:30 are deadlines, not start times.
   - Write the paper-style summary of your best result (B2b + the chat-bytes lever) NOW, then keep deepening.
   - Next targets, in order:
     - (a) the rank 2-9 tail of KEPT rows (two thirds of the union): a quality-safe cut beyond the band tables, e.g. per-layer tolerance, low-weight ranks substituted by already-loaded experts within a gate tolerance, or shared-expert fallback. Price it on floor_rows' quality data.
     - (b) making rejected rows nearly free;
     - (c) the 24 / 12 GB bytes forms of your candidates.
   - Keep notes concise and prefer scripts to long prose: output tokens are the main cost.
