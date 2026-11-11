"""check_sources.py: does every number MAIN_PAPER.md quotes still stand in the file it names?
CPU, read-only, seconds. Run before every revision:  python3 -I check_sources.py          (add --keys for Appendix D's table)
Each check = (label, file, key, substrings).
  - key None: every substring must be somewhere in the file.
  - key "Tn" (a ledger entry of ~/TRIED/TRIED.md): exactly ONE line must hold every substring. The entry is found by its
    content, so a re-merge of the ledger (line numbers move) does not break the check. "Tn" is a stable LABEL: n was the
    entry's line number in the merge of Fri 6 Nov ~21:00. Appendix D of the paper maps each label to the entry's ID.
  - key int: that line of the file must hold every substring (files that are not re-merged).
It also lists WITHDRAWN / CORRECTED / VOID on the cited entries and in the findings' titles, the ledger's tallies, the
paper's open placeholders, and it fails on the owner's name or a home path in the paper."""
import os, re, sys

H = os.path.expanduser("~")
T = f"{H}/TRIED/TRIED.md"
EV = f"{H}/EvalSuite/docs/"
FN = f"{H}/FableGod/findings/"
LL = f"{H}/TRIED/leap.md"
LT = f"{H}/Leap/TABLE_350.md"
INV = f"{H}/NewIdeas/PUBLISH_INVENTORY.md"

CHECKS = [
    # --- section 2: exactness
    ("REF2 adopted, x0.75 AR", T, "T1210", ["R-40 REF2", "ADOPTED", "-51 / -60 ms", "math +8.0 p 0.12", "11.65 vs 15.50"]),
    ("INV1 padding", T, "T1209", ["R-40 INV1", "all 16 logits rows and 37 records"]),
    ("key buckets", T, "T1208", ["key buckets", "160 / 160", "220 / 220", "buckets off 2 / 160"]),
    ("sdpa classes", T, "T1218", ["sdpa_classes", "1477 / 1477", "441 / 441", "216 / 256"]),
    ("REF2b same scores", T, "T1219", ["- REF2b:", "math +6.0 p 0.25", "IFEval -3.0 p 0.38", "HumanEval+ 0.0 p 1.00"]),
    ("REF2-T default", T, "T1214", ["R-58 REF2-T", "16 / 16 rows", "383 and 536 keys"]),
    ("REF2-T vs REF2b", T, "T307", ["- REF2-T (R-58)", "math 0.0 p 1.00", "IFEval +3.0 p 0.38", "HumanEval+ +3.3 p 0.50"]),
    ("R-56 KILL", T, "T1259", ["R-56 REF2 live loop", "28/36", "math 4/12"]),
    ("R-56d OOO audit", T, "T490", ["- R-56d:", "36/36 prompts bitwise", "15,489/15,489"]),
    ("kernel switch at 6 rows", T, "T1225", ["MMA-tile exactness check", "MLX switches kernels at 6 rows"]),
    ("dense 56.8 -> 27.4", T, "T478", ["- L1-23 batch-invariant", "56.8 -> 27.4", "padding-independent for every shape"]),
    ("batch-invariant reframing", T, "T1195", ["- Batch-invariant reframing"]),
    ("experts_nax", T, "T479", ["- L1-25 experts_nax"]),
    ("chunked GDN KILL", T, "T1244", ["Chunked (WY / UT) GDN verify", "KILL"]),
    ("0 OOO calls void run", T, "T18", ["R-69b OOO-LIVE", "0 OOO calls"]),
    ("r56b audit", EV + "RESULT_r56b_2026-10-31.md", None, ["36 / 36 | 16,341 | 16,305 / 16,305", "All 15 prompts that run past 512 keys", "2,056 keys"]),
    ("nax6 part 2", EV + "RESULT_nax6_2026-11-05.md", None, ["15,726 tokens; 15,690/15,690 kept rows", "15/15 past 512 keys and 6/6 past 1,024", "753,120 / 753,120", "147,036 / 147,036", "66.8 tok/s, math 42.6, IFEval 21.0", "7.20 rows kept per check"]),
    ("nax6 part 1", EV + "RESULT_nax6_2026-11-05.md", None, ["| math (n 50) | 88.0 % | 88.0 %", "| IFEval (n 100, strict, prompt level) | 87.0 % | 90.0 %", "| HumanEval+ (n 60, pass@1) | 91.7 % | 95.0 %", "right on 2 items and REF2-T alone on 7", "p 0.18"]),
    ("ref2 baseline + loops", EV + "RESULT_ref2_2026-10-28.md", None, ["(ref.r2)", "REF2 has 4 loop rows"]),
    ("REF2c-T failed", f"{H}/BigMoE/HANDOVER.md", None, ["REF2c-T failed decision 4", "IFEval -5.0"]),
    ("INV0 / INV1 / R-40 screen", LT, None, ["only attention breaks batch invariance under REF2", "REF2 IS BATCH-INVARIANT", "| 1K | 39.60 | 26.78 | 12.82 |", "| 8K | 41.88 | 27.36 | 14.91 |"]),
    # --- 12 GB
    ("12 GB exact loop", f"{H}/Fit12/PLAN_12GB_90_fit12.md", None, ["| ex (exact loop) | 18.58 | - | 9.71 | 11.2 / - / 510.7 / - / 522.5", "lifetime 11.39 GB in every arm", "runs 18.4-18.9 tok/s on HumanEval (T3t 18.9 and T3R 18.42", "exd = ex on 14 / 16", "ex ~4.2 GB", "(= ex on 5 / 16 here)"]),
    ("B12LL", T, "T29", ["- B12LL:", "5/5 in both runs", "210/210 identical"]),
    ("12 GB default adopted", T, "T1057", ["Fold-fast with the memory fix", "b12llff3", "ADOPTED"]),
    ("12 GB note corrected", FN + "2026-11-04_1439_12gb-exact-40tps.md", None, ["# CORRECTED", "18.4-18.9 tok/s", "(1.48 tok/s)", "matched the exact loop on 5 of 16 prompts", "checked on 4 of the 16 prompts"]),
    ("T0 KILL", EV + "RESULT_t0_2026-11-05.md", None, ["| HumanEval+ (n 60, t0c at 0.8) | 80.0 % | 91.7 %", "0.02 (0/7)", "| IFEval (n 100, t0h) | 84.0 % | 90.0 %", "at most 10.84 GB", "23 / 7 vs REF's 13 / 2"]),
    ("map predicted T0", T, "T54", ["PLAN-12GB-90", "the map predicted T0's KILL", "1.63 % code"]),
    ("certificate token-exact", T, "T1265", ["- 12 GB lever 1:", "0-1 divergences in ~3,400 tokens", "pass 1's KV"]),
    # --- section 3: system
    ("model table", f"{H}/BigMoE/FINDINGS.md", None, ["| routed experts | 120.8 B |", "36 of 48 layers", "all routed experts =\n67.9 GB", "2.46 GB of dense weights + 1.33 GB of routed", "| 2.77 MB expert | 5.8 GB/s (p50 0.45 ms) | 12.3 | **14.0**"]),
    ("record sizes", f"{H}/BigMoE/DESIGN_3bit.md", None, ["MSB_SZ 2,162,688", "LSB_SZ 770,048", "72.1 GB"]),
    ("native layout", T, "T527", ["- N7-13 NAX6-N", "+18.7 % quiet at S 16"]),
    ("p11", T, "T334", ["- P11:", "+1.2 % (3c-N) / +2.2 % (NAX6-N)"]),
    ("SSD wait law", T, "T288", ["- FAR-WAIT", "10-16 ms + 0.15-0.17 ms per miss", "12.7-14.4 GB/s"]),
    ("per-layer pools", T, "T12", ["- M-17 LAYSZ45", "-17.2 %", "+8.3 %"]),
    ("concurrent loads", T, "T7", ["- Concurrent miss loads"]),
    ("volatile tier", T, "T1088", ["- PURGE-1:", "toggle 0.7-0.8 / 0.2 us"]),
    ("drafter switch", T, "T782", ["- D-23d CHATSWITCH", "+24.5 %", "19 of 19"]),
    ("live rows per check", T, "T993", ["- LIVE-CUT AUDIT", "code 9.7 / IFEval 3.7 / multi 3.0", "q1u_0.9_live code 1.25 % PASS"]),
    ("t 0.3 cut", T, "T309", ["- t 0.3 truncation", "42.8 -> 50.9", "+19 %"]),
    ("sync-free gate", T, "T489", ["- R-55 GATE16", "232.6 -> 117.3 ms", "23.7 -> 32.0 tok/s (+35 %)"]),
    ("syncs share", T, "T476", ["- GATE16 sync-free 16-row gate", "~120-140 ms of a 200 ms check were syncs"]),
    ("OOO", T, "T14", ["- OOO + TIER_COMBINE", "+13 %, rough", "36/36"]),
    ("item 5", T, "T10", ["- Item 5:", "+12.1 % geo steady", "lever 1 +6.4 % more", "172.5 -> 120.2"]),
    ("lever 1 parts", T, "T565", ["- Item 2:", "joint issue, SSD keepalive, copy split, coherent spin, system post"]),
    ("pickup", T, "T76", ["- W1-PICKUP", "pickup median 2.9 us", "~20-23 us after the next dispatch"]),
    ("keepalive in engine", f"{H}/BigMoE/docs/NIGHT_1104.md", 37, ["keepalive 50 us", "about -3 ms per check on clean prompts"]),
    ("NATIVE-FRI run 2", LL, None, ["NATIVE-FRI run 2 (C 192, quiet, margin 3): PASS", "1.0665", "Dc 46.75, Nc 49.86", "peak 38.1 GB"]),
    # --- FAST parts
    ("QOM switch", f"{H}/BigMoE/scripts/r57_screen_fast.py", None, ["U3 at thr 0 (every drafted row sure, the bonus row exact)", "KEPT WHEN RESIDENT in the pool", "dropped after the spin (weight 0, no renormalisation)", 'FASTARMS["QOM"] = {s_: _S(thr=_QT, q1u=True, t3=True, qom=True)']),
    ("QOM cf03 == cf0", T, "T1011", ["- CF03-EQ-CF0", "36 / 36 windows bit for bit"]),
    ("split chat control / twin", T, "T552", ["- SPLIT-ARMS-CHAT", "twin 4.03 / 3.26 % (PASS)", "split 0.99 6.01 / 5.15 %"]),
    ("preset on the map", FN + "2026-11-05_0720_cloud3-repriced-split-form.md", None, ["first map run at 45 GB): 0.51 %", "0.92 % (31 rows)", "1.31 % (44 rows)"]),
    ("draft17", FN + "2026-11-04_1434_draft17.md", None, ["35.7 -> 25.2 (x0.706)", "code 1.07 % flips"]),
    ("hc8 note", FN + "2026-11-04_1435_hc8-one-pass-hc-read.md", None, ["5.22 vs 9.67 ms per 96-read chain"]),
    ("hc8 same scores", EV + "RESULT_hc8_2026-11-02.md", None, ["## Verdict: PASS"]),
    ("FAST-NAX 113.9", f"{H}/NewIdeas/QUEUE.md", None, ["113.9 tok/s [K, BigMoE data/fnx/fnx_report.txt", "passes 112.7 / 115.0", "Against qall's 97.5", "+16.8 %", "Kept 10.36 vs 10.55 per check (-1.8 %)"]),
    ("Leap Friday lines", LL, None, ["FLAT 55.17 / 51.05 / 55.30 -> -4.19 ms", "att_single FLAT -4.06 ms", "propose 8.8 -> 6.7 ms (-2.1)", "draft -0.75 ms/cycle", "Running 2-row FLAT 55.2 -> 44.2 ms today", "5.9 of 8.8 ms", "~-0.2 net per check"]),
    ("gap hunt", LL, None, ["the dense pass in situ at 16 rows = 25 ms wall, ~22 ms GPU (87-89 % busy, GPU-bound)", "(32-39 ms for the same forward)", "NEVER-WAIT v3"]),
    ("gap hunt + real blocks (table)", LT, None, ["25.0 / 24.9 ms per forward", "synced per-piece split sums 32-39 ms", "kept 14.42 per check", "check + commit 67.2 ms", "check + commit 61.2 (85 misses)", "the engine sees about half the chained gain (overlap)"]),
    ("RELAX arms", EV + "PREREG_relax_2026-11-02.md", None, ["| B, the mode (B) | HumanEval+ 60, math 50, AT14 (124) | Q1-UNION, cf >= 0.9 + copy | exact, + the comment rule on HumanEval+ | on |", "its last 8 input tokens occur in the prompt", "ranks first or second on its row", "p(d_j) >= 0.25 p1", "never in its first two lines, and never in docstrings, strings or code", "Q1-UNION 0.9 against Q1 0.99: code 0.98 vs 1.34 %, IFEval 4.92\n    vs 5.42 %, multi 3.04 vs 3.69 %"]),
    ("RELAX result", EV + "RESULT_relax_2026-11-05.md", None, ["| **ship** | B code + E chat | 90.0 vs 88.0 (+2.0) | 87.0 vs 90.0 (-3.0, p 0.25) | 93.3 vs 95.0 (-1.7) | PASS", "| A_E | A + E | **82.0** vs 88.0", "**90.0** vs 95.0 (-5.0, p 0.25)", "**85.0** vs 90.0", "**84.0** vs 90.0", "**88.3** vs 95.0", "tok/s x1.49 HumanEval+, x1.32 math", "-41 % on code", "the comment rule adds 1.8 % on HumanEval+", "0.21 per 10K (bar 1.0)", "| E | 20 | 1.149x | 0.820x", "| G | 20 | **1.433x** | 0.834x"]),
    ("RELAX drift", EV + "RESULT_relax_drift_2026-11-06.md", None, ["q4 / q1 = 0.82", "IFEval 84.0 vs 90.0, −6.0 pt, 0/6, McNemar p 0.03", "| humanevalx (new, HumanEval+'s other 104) | 6 | 0.00298 | 0.01215 | 4.07 |", "| humanevalx | 104 | 25,656 | 1.20 % | 0 | 0.00 |", "IFEval scores: E 87, HC8 alone 89, E8 84"]),
    ("agent check", EV + "RESULT_agentfast_2026-11-06.md", None, ["| both | 34 | 19,749 | 1,604 (8.12 %) | **85** (~43 per 10K) |", "75 of the 85 are inside the written code"]),
    ("agent bisect", EV + "RESULT_agentbisect_2026-11-06.md", None, ["| base | none | 21 (reverse 18)", "| qom | QOM-KEEP (R57_QOM_SETS without agentlong) | 4 (reverse **0**)", "about 27 tokens before the committed tokens do"]),
    ("tie floor", T, "T1049", ["- WALL-TIEFLOOR", "1.5 / 8.5 / 7.0 % of rows", "0.92 / 4.53 / 3.43 %", "0.4 / 0.9 / 0.3 points above"]),
    # --- section 4
    ("U3 union form", T, "T103", ["- Q1U-T3", "code 1.19 / IFEval 5.06 / multi 3.24 %", "-46.3 %", "-34.9 %"]),
    ("U3 QOM form", T, "T106", ["- QOM-T3", "qom_0.5_live_t3 0.65 / 4.42 / 3.49 %", "qom_0_live_t3 0.71 / 4.45 / 4.14 %"]),
    ("wake penalty", T, "T93", ["- M-20 SSDWAKE", "+50-65 us per burst", "+100-120 after 20 ms", "a busy CPU core does nothing", "~3 ms per check"]),
    ("per-call split", T, "T97", ["- M-28 SSDMICRO per-call cost decomposition", "R0 fixed 187 of the engine's 283 us (first byte ~119, wake 44, dispatch 8)", "~96 us outside I/O"]),
    ("keepalive spacing", T, "T44", ["- M-28 SSDMICRO keepalive spacing", "real spacing <= 270 us", "+44 us wake", "400+ us leaves half"]),
    ("wake-ahead ping", T, "T47", ["- M-29 SSDMICRO2 wake-ahead ping", "removes 97.5 % of the +48 us wake"]),
    ("ping at engine gaps", T, "T98", ["- M-29 follow-up", "removes 87-94 %"]),
    ("head ping KILL", T, "T202", ["- M-31 HEADPING", "KILL", "+0.6..+3.2 us"]),
    ("split reads KILL", T, "T199", ["- M-28 SSDMICRO pieces", "each piece pays its own request latency"]),
    ("page cache KILL", T, "T201", ["- M-28 SSDMICRO page cache", "+49 us per call"]),
    ("comment loophole", T, "T1350", ["- COMMENT-LOOPHOLE", "x1.076", "x1.157", "13.5 % of positions and 44.7 % of the drafter's misses"]),
    ("CD16 KILL", T, "T855", ["- CD16-DROP", "qall 11/12 vs CD16 8/12", "3 flips vs a line of 2", "K x1.449, +23.7 %"]),
    ("prompt footprint as the cold table", T, "T312", ["- 5b U3C len", "x0.732", "+3.7 %"]),
    ("novelty counts", f"{H}/TRIED/NOVELTY.md", None, ["Refresh Fri 6 Nov 21:3x", "309 tags: 36 published, 91 adapted, 5 new, 177 engineering", "Still no lever in a release"]),
    # --- section 5
    ("t_sync + kernels", LT, None, ["1.24 (32 threads), 1.40 (1 x 1,024), 1.55 (16 x 1,024), 1.56 (20 x 1,024)", "4.13 (512 x 256, 1 MB moved)", "113 / 115; 23.6 MB; 210; 36", "112 / 112; 8.9 MB; 81; 48", "45 / 46; 0.22 MB; 5; 36", "75 / 74; 17.7 MB; 243-247; 12", "48 / 49; 0.8 MB; 18; 24", "48 / 47; 2.0 MB; 42; 48", "21.9 / 21.8 each, 20 GB/s", "So ~3,900 / ~3,000 per check", "| out_proj / o_proj | 114.8 us | 41.4 us | 6 | 214 | -3.5 ms |", "+0.98 ms per row", "+3.25 ms per row"]),
    ("DN-KS step 1", LL, None, ["-7.74 ms per 2-row check (out_proj ks6 115->41 us, a|b ks10 47->10, k/v ks10 47->9, gate|up ks8 48->15", "bits <= 2e-3"]),
    ("FLAT-2 note", FN + "2026-11-06_0140_flat2-fixed-cost-per-check.md", None, ["FLAT 53.7 ms", "PASS <= 30 ms and KILL > 36 ms", "GDN 13.2, experts 11.1 (~4.6 of it fixed), HC reads 8.6, attention 6.4, router + shared 5.7", "Pure CPU graph build: 13.4 ms", "12 / 54 ms = ~220 tok/s", "11.1 ms at 2 rows, 57.0 ms at 16", "98.8 -> 92.9 / 93.7 ms", "0.63x / 0.49x"]),
    ("K plan", f"{H}/TrainLab/K_PLAN_1106.md", None, ["**10.36**", "HumanEval+ (24) **9.57**, MBPP (100) **5.94**, GSM8K\n  (60) 7.00, agent (12) 3.01", "On-policy data: +0.13 per doubling", "2x capacity (P-CAP): +0.18", "One reseed alone moves code by 0.16"]),
    ("thinking hour", FN + "2026-11-05_1700_lossy350-thinking-hour.md", None, ["67.8 % of generated tokens are one", "MBPP, math and IFEval answers copy 1-3 %", "mean cos +0.02"]),
    ("union law", FN + "2026-11-05_1715_union-law-chat-check-length.md", None, ["code 276 / 218 / 163", "fall only ~25 % per doubling of R"]),
    ("visibility", f"{H}/BigMoE/FINDINGS.md", None, ["**0.5–1000 ms**", "Metal guarantees CPU/GPU visibility only at command-buffer boundaries"]),
    ("spin gate", T, "T495", ["- Spin gate", "~69 µs per gated layer vs ~230 µs"]),
    # --- section 7
    ("never-wait map", T, "T50", ["- NW-BUILD", "MAP PASS Fri 13:46", "nw1 at 208 slots 0.68 [0.27] vs its twin 0.74"]),
    ("never-wait engine", T, "T209", ["- NW-ENGINE: never-wait's engine arm", "KILL", "-2.8 % (153.5 vs 157.9 tok/s), kept +4.6 %", "68.8 -> 75.3 ms", "~6.5 ms per check"]),
    ("never-wait correction", T, "T209c", ["- NW-ENGINE: correction to its line", "counts calls that issued any read, not waits", "14.0 -> 15.6 ms per check with NW", "v3 (Leap, staged)"]),
    ("split map", T, "T553", ["- SPLIT-ARMS-CODE", "split_0.99 0.92 %", "split_0.9 1.31 %"]),
    ("split engine", T, "T610", ["- SPLIT-ENGINE-OFF", "groups x0.625 / x0.494", "kept -4.2 / -1.7 %", "~2-3x high"]),
    ("split-nax", T, "T744", ["- SPLIT-NAX: result", "KILL", "-1.3 % vs the stack", "kept -5.7 %"]),
    ("157.9 withdrawn", FN + "2026-11-06_1550_45gb-fast-stack-158tps.md", None, ["# WITHDRAWN", "157.9 tok/s", "passes 157.2 / 158.1", "8 of 12 with CD16 against 11 of 12 without it"]),
    ("trellis", FN + "2026-11-04_2228_trellis_mse_miss.md", None, ["1.45 / 9.54 / 10.44 %", "~1.7-3.5x more KL per unit"]),
    ("gdn-chunk / k2c KILL", LL, None, ["GDN-CHUNK GPU check", "time KILL", "HC-K2C GPU check"]),
    ("NAX6 killed on 1 Nov", T, "T675", ["- K-62j NAX6", "KILL at the 45 GB operating point"]),
    ("wide checks KILL", T, "T462", ["- U-32:", "KILL", "-1.3 %"]),
    ("inventory rows", INV, None, ["49.9 [K, quiet, C 192, NATIVE-FRI run 2]", "71.3 [K, qall, C 224]", "~128-130 [P]", "3-bit kept experts on drafted rows (U3's t3, inside QOM at thr 0)"]),
    ("published comparison", f"{H}/MoonShots/research/results_first/RESULTS_FIRST.md", None, ["| SlotStream | 4-bit MLX | exact | 10.7 GB process target (16 GB M2 mini) | Mac mini M2 | 1.48 |", "| Kapitsa | Q2_0, 2.4 b | routing exact, verified drafting | 16 GB expert store (48 GB machine; total unknown) | MacBook Pro M5 Pro 48 GB | **61.3** |", "| Splash 1.4.0 (stock) | Q2_0, 2.4 b | routing exact, verified drafting | 8.8 GB expert store; total unknown | MacBook Air M5 24 GB | 28.8 |", "| SlotStream | 4-bit | exact | 33 GB ceiling | M5 Pro 48 GB | 23.69 |"]),
]

STATUS_WORDS = re.compile(r"WITHDRAWN|CORRECTED|CORRECTION|VOID\b|RETRACT", re.I)


def entry_id(line):
    """the ledger entry's ID: the text between '- ' and the first ': ' (or ' | '), cut to 70 chars"""
    s = line[2:] if line.startswith("- ") else line
    cut = min([i for i in (s.find(": "), s.find(" | ")) if i > 0] or [70])
    return s[:min(cut, 70)].strip()


def keynum(k):
    return (int(re.findall(r"\d+", k)[0]), k)


def main():
    cache, bad, n, found = {}, [], 0, {}
    for label, path, key, subs in CHECKS:
        n += 1
        if path not in cache:
            try:
                cache[path] = open(path, encoding="utf-8", errors="replace").read()
            except OSError as e:
                cache[path] = None
        text = cache[path]
        if text is None:
            bad.append((label, f"cannot read {path}"))
            continue
        if isinstance(key, str):                                  # a ledger entry, found by content
            lines = text.split("\n")
            hits = [i + 1 for i, l in enumerate(lines) if all(s in l for s in subs)]
            if len(hits) != 1:
                bad.append((label, f"{key}: {len(hits)} ledger lines hold {subs[:2]}... (need exactly 1): {hits[:5]}"))
            else:
                found[key] = (hits[0], lines[hits[0] - 1])
            continue
        hay = text.split("\n")[key - 1] if key else text
        miss = [s for s in subs if s not in hay]
        if miss:
            bad.append((label, f"not found in {'line %d of ' % key if key else ''}{path}: {miss}"))
    print(f"{n} checks, {n - len(bad)} ok, {len(bad)} failed")
    for label, why in bad:
        print(f"  FAIL {label}: {why[:400]}")
    moved = sorted((k for k, (ln, _) in found.items() if re.fullmatch(r"T\d+", k) and ln != int(k[1:])), key=keynum)
    print(f"ledger entries found by content: {len(found)}; {len(moved)} sit on another line than their label's number "
          f"(a re-merge; the labels stay as they are)")
    if "--keys" in sys.argv:
        print("\n| label | ledger entry (search `~/TRIED/TRIED.md` for this ID) | line on " + "the last check |\n|---|---|---|")
        for k in sorted(found, key=keynum):
            print(f"| {k} | {entry_id(found[k][1]).replace('|', '/')} | {found[k][0]} |")
        print()
    print("status words on cited ledger entries (read each: is the quoted number the one that stands?):")
    flagged = [(k, sorted(set(w.upper() for w in STATUS_WORDS.findall(l)))) for k, (_, l) in found.items() if STATUS_WORDS.search(l)]
    for k, w in sorted(flagged, key=lambda x: keynum(x[0])):
        print(f"  {k}: {w}: {found[k][1][:110]}")
    if not flagged:
        print("  none")
    print("status words in the titles of the findings notes:")
    for f in sorted(os.listdir(FN)):
        if f.endswith(".md") and f != "README.md":
            title = open(FN + f, encoding="utf-8").readline().strip()
            if STATUS_WORDS.search(title):
                print(f"  {f}: {title[:120]}")
    tl = cache.get(T) or ""
    tot = {"ideas": 0}
    for m in re.finditer(r"^## \S.*?\((\d+):(.*?)\)$", tl, re.M):
        tot["ideas"] += int(m.group(1))
        for name, v in re.findall(r"(ADOPTED|PASS|BETWEEN|PARKED|KILL) (\d+)", m.group(2)):
            tot[name] = tot.get(name, 0) + int(v)
    print(f"ledger tallies now: {tot}")
    paper = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "MAIN_PAPER.md"), encoding="utf-8").read()
    used = set(re.findall(r"(?<![-\w])T\d+c?\b", paper))        # not the 'T3' inside an entry ID such as Q1U-T3
    undefined = sorted(used - set(found), key=keynum)
    print(f"ledger labels used in the paper: {len(used)}; without a check here: {undefined if undefined else 'none'}")
    tbd = sorted(set(re.findall(r"\[\[(TBD-\d+[a-d]?)", paper)), key=lambda s: (int(re.findall(r"\d+", s)[0]), s))
> (a quote from the project owner, kept private)
    print(f"placeholders still open in MAIN_PAPER.md: {len(tbd)}: {' '.join(tbd)}")
    print(f"personal-context check (owner's name, home path): {'CLEAN' if not leaks else 'FOUND ' + str(leaks)}")
    return 1 if bad or leaks or undefined else 0


if __name__ == "__main__":
    sys.exit(main())
