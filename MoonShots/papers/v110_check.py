"""v1.1.0: wording fixes from the orchestrator's full number check (Wed 11 Nov 04:4x), applied on top of v110_chat2.py.
Each item re-derived by the editor first (papers/rd_w24_gate_peaks.py for 1, 4, 5, 6, 8):
 1. W24e's six prompts (P 2,801 / 1,465 / 1,102 / 942 / 919 / 726, pc*g_100_runs.json) are not the gate's six longest
    (the gate also has 1,360, 1,144, 1,120, 1,062, 1,036, 1,017 and 1,016): "six long gate prompts (726-2,801 tokens)".
 2. The 3,003- and 6,980-token runs answered 256 tokens (chain.log simA_long; pfa24_default.jsonl), 10K / 30K 64; the
    3,003-token peak (25.03 GB) is case A of the simulated 36 GB Mac (wired limit 27 GB). SMALL_RAM 6.3, RELEASE 2.
 3. RELEASE 4: "OpenAI's chat completions format ... and the messages format" (no "Anthropic": the lint bans it).
 4. SMALL_RAM: "can go over 24 GB": abc397_f (592 tokens), first in a fresh process twice, 23.954 (rev3) / 24.112 (fwd18).
 5. "About 500 tokens" as measured: a 448-token GPQA prompt raised rev17's running peak to 23.954 GB (1-based row 10,
    0-based 9); the 520-token math prompt 24.014 / 24.013 GB (bg1a / bg2a row 9); 0.16 GB between two runs of one prompt.
    No crossing on a prompt under 561 tokens in the gate; 22 processes crossed.
 6. SMALL_RAM 6.3: the 256-token chunks were run only on the six gate prompts (pc256g 23.954 GB, pc512g 24.345).
 7. RELEASE 1: "is measured against the memory line".
 8. W24g: two warm-ups (gsm8k/708, discarded) ran first (bg1a_100_screen.txt); the 520-token prompt is row 9 of 12.
 9. 6,980 = 6 x 1,024 + 836; 29,988 = 29 x 1,024 + 292 (KERNELS R33); the 7K default first token 42.06 s.
10. RELEASE: "whole decode"; the 135.9 tok/s run (he12) ran under a share lock (check_summary.txt 09:59 "share lock").
11. PF-A runs only on prompts over one chunk (r57_screen_rel1.py:987 `if PFLM and len(ids) > chunk`; pflm.py:18).
usage: python3 -I v110_check.py [--write]   (backups: ~/Papers/.v110_check_bak_<time>/)"""
import pathlib
import re
import sys
import time

P = pathlib.Path.home() / "Papers"
R = {"RELEASE.md": [], "MAIN.md": [], "SMALL_RAM.md": [], "KERNELS.md": []}
EVR = "EvalSuite/results/card/es_cdw24_100136_"

R["RELEASE.md"] += [
    ("engine's lifetime peak memory footprint is the memory line it is named for (45, 32 or 24 GB): the engine's own footprint on a\n",
     "engine's lifetime peak memory footprint is measured against the memory line it is named for (45, 32 or 24 GB): the engine's own footprint on a\n"),
    ("Two passes pooled per set. Under 24 GB on prompts of up to about 500 tokens (below) |",
     "Two passes pooled per set. Under 24 GB in our runs on prompts of up to about 500 tokens, with little room (below) |"),
    ("- Long prompts: the default prefill re-reads each layer's experts per 1,024-token chunk. PF-A (layer-major; exact except\n"
     "  when the prompt's last 1,024-token chunk is under about 200 tokens [K, LP:data/pfa_tail_1111.md]) and PF-B (layer\n",
     "- Long prompts: the default prefill re-reads each layer's experts per 1,024-token chunk. PF-A (layer-major, on prompts\n"
     "  over 1,024 tokens; exact except when the last 1,024-token chunk is under about 200 tokens [K,\n"
     "  LP:data/pfa_tail_1111.md]) and PF-B (layer\n"),
    ("  are one env line, the gated one, with the gated runs' decode loop [K, LP:data/w24_pkg.md]. It stays under 24 GB on\n"
     "  prompts of up to about 500 tokens; a 520-token math prompt reached 24.01 GB [K, BM:data/r57/bg1a_100_runs.json]. Longer\n"
     "  prompts use more: 25.0 GB at 3K tokens, 25.7 at 7K, 25.8 at 10K and 28.2 at 30K (default prefill, as run.py runs it)\n"
     "  [K, LP:data/tue_afternoon_1110.md sections 1, 2 and 4]. The package's lossy-24 gave the research runs' tokens on all 12\n"
     "  HumanEval prompts and the gate's on six of its IFBench and GPQA Diamond prompts, at 135.9 tok/s whole decode on\n"
     "  HumanEval (one pass) and a 23.78 GB peak [K, LP:data/w24_pkg/check_summary.txt].",
     "  are one env line, the gated one, with the gated runs' decode loop [K, LP:data/w24_pkg.md]. In our runs it stayed\n"
     "  under 24 GB on prompts of up to about 500 tokens, with little room, so this is a measurement, not a bound: a 448-token\n"
     f"  gate prompt raised its process's peak to 23.954 GB [K, {EVR}rev17_runs.json], a\n"
     "  520-token math prompt reached 24.01 GB [K, BM:data/r57/bg1a_100_runs.json], and one 592-token prompt, first in a\n"
     "  fresh process both times, peaked 0.16 GB apart (23.954 and 24.112 GB) [K,\n"
     f"  {EVR}fwd18_runs.json; {EVR}rev3_runs.json].\n"
     "  Longer prompts use more: 25.0 GB at 3K tokens (on the simulated 36 GB Mac, section 3), 25.7 at 7K, 25.8 at 10K and\n"
     "  28.2 at 30K (default prefill, as run.py runs it) [K, LP:data/tue_afternoon_1110.md sections 1, 2 and 4]. The\n"
     "  package's lossy-24 gave the research runs' tokens on all 12 HumanEval prompts and the gate's on six of its IFBench\n"
     "  and GPQA Diamond prompts, at 135.9 tok/s whole decode on HumanEval (one pass, under a share lock) and a 23.78 GB peak\n"
     "  [K, LP:data/w24_pkg/check_summary.txt]."),
    ("(127.0.0.1) in two API formats, OpenAI's chat completions (`/v1/chat/completions`) and messages (`/v1/messages`), with\n"
     "streaming, one request at a time. In 1.1.0 it handles text only: tool definitions and tool calls are not mapped, so it\n"
     "cannot serve an agent that calls tools.\n",
     "(127.0.0.1) in two API formats, OpenAI's chat completions format (`/v1/chat/completions`) and the messages format\n"
     "(`/v1/messages`), with streaming, one request at a time. In 1.1.0 it handles text only: tool definitions and tool calls\n"
     "are not mapped, so it cannot serve an agent that calls tools.\n"),
    ("- At 32 and 24 GB chat.py prefills long prompts layer-major (PF-A, section 2; `R57_PFLM=0` in the shell turns it off);\n"
     "  run.py and serve.py do not. On the chat configuration PF-A brings the first token 3.5-3.7x sooner at 32 GB (10K- and\n"
     "  30K-token prompts) and 2.4-2.7x sooner at 24 GB (7K-30K tokens), for 0.3-3.0 GB more lifetime peak (section 5) [K,\n"
     "  LP:data/prefill_1109.md; LP:data/tue_afternoon_1110.md sections 2 and 4]. Its tokens are the default prefill's except\n"
     "  when a prompt's last 1,024-token chunk is under about 200 tokens [K, LP:data/pfa_tail_1111.md].\n",
     "- At 32 and 24 GB chat.py prefills prompts over 1,024 tokens layer-major (PF-A, section 2; `R57_PFLM=0` in the shell\n"
     "  turns it off); run.py and serve.py do not. On the chat configuration PF-A brings the first token 3.5-3.7x sooner at\n"
     "  32 GB (10K- and 30K-token prompts) and 2.4-2.7x sooner at 24 GB (7K-30K tokens), for 0.3-3.0 GB more lifetime peak\n"
     "  (section 5) [K, LP:data/prefill_1109.md; LP:data/tue_afternoon_1110.md sections 2 and 4]. Its tokens are the default\n"
     "  prefill's except when the last 1,024-token chunk is under about 200 tokens [K, LP:data/pfa_tail_1111.md].\n"),
    ("build, with its own defaults, idled at 1.6 % (serve.py) and 1.5 % (chat.py) at 24 GB [K, LP:data/w24_pkg.md]. Decode\n"
     "  speed did not measurably change:",
     "build, with its own defaults, idled at 1.6 % (serve.py) and 1.5 % (chat.py) at 24 GB [K, LP:data/w24_pkg.md]. Whole\n"
     "  decode did not measurably change:"),
    ("  PF-A [K, LP:data/prefill_1109.md]. The 24 GB line holds only up to about 500 tokens: 25.0-28.2 GB at 3K-30K tokens\n"
     "  [K, section 2], and 26.0-30.6 GB at 7K-30K tokens with chat.py's PF-A [K, LP:data/tue_afternoon_1110.md].\n",
     "  PF-A [K, LP:data/prefill_1109.md]. The 24 GB line stayed under 24 GB only on prompts of up to about 500 tokens in our\n"
     "  runs, with little room (section 2): 25.0-28.2 GB at 3K-30K tokens [K, section 2], and 26.0-30.6 GB at 7K-30K tokens\n"
     "  with chat.py's PF-A [K, LP:data/tue_afternoon_1110.md].\n"),
    ("  section 6.2). PF-A is exact only when a prompt's last 1,024-token chunk is about 200 tokens or longer (section 2).\n",
     "  section 6.2). PF-A, which runs only on prompts over 1,024 tokens, is exact only when their last 1,024-token chunk is\n"
     "  about 200 tokens or longer (section 2). Section 1 now says the lifetime peak is measured against the memory line,\n"
     "  which long prompts go over (section 5).\n"),
]

R["MAIN.md"] += [
    ("prompt's last 1,024-token chunk is under about 200 tokens (section 6.2) [K, F:2026-11-08_0800; LP:data/pfa_tail_1111.md].\n",
     "prompt's last 1,024-token chunk is under about 200 tokens; it runs only on prompts over 1,024 tokens (section 6.2) [K,\n"
     "F:2026-11-08_0800; LP:data/pfa_tail_1111.md].\n"),
    ("- It is exact only when the prompt's last 1,024-token chunk is long enough. On the 32 GB exact configuration, a\n",
     "- It runs only on prompts over 1,024 tokens (shorter ones take the default path), and it is exact only when the\n"
     "  prompt's last 1,024-token chunk is long enough. On the 32 GB exact configuration, a\n"),
    ("  fix in a research copy, not in the package (section 6.2). The layer-major prefill is exact only when a prompt's\n"
     "  last 1,024-token chunk is about 200 tokens or longer; a shorter one changed the tokens (sections 3.2, 4.5, 6.2, 7\n",
     "  fix in a research copy, not in the package (section 6.2). The layer-major prefill, which runs only on prompts over\n"
     "  1,024 tokens, is exact only when their last 1,024-token chunk is about 200 tokens or longer; a shorter one changed\n"
     "  the tokens (sections 3.2, 4.5, 6.2, 7\n"),
]

R["KERNELS.md"] += [
    ("on a 29,988-token prompt (32 GB exact configuration, 64 tokens, one run each)",
     "on a 29,988-token prompt (last chunk 292 tokens; 32 GB exact configuration, 64 tokens, one run each)"),
]

R["SMALL_RAM.md"] += [
    ("- It stays under 24 GB on prompts of up to about 500 tokens. Longer prompts use more: about 25.0 GB at 3K tokens,\n"
     "  25.8 GB at 10K and 28.2 GB at 30K [K].\n",
     "- In our runs it stayed under 24 GB on prompts of up to about 500 tokens, with little room: a 448-token prompt reached\n"
     "  23.954 GB and a 520-token one 24.01 GB. Longer prompts use more: about 25.0 GB at 3K tokens, 25.8 GB at 10K and\n"
     "  28.2 GB at 30K [K].\n"),
    ("- Our 24 GB configuration goes over 24 GB on prompts longer than about 500 tokens: 25.0 GB at 3K tokens, 28.2 GB at\n"
     "  30K. On the gate's six longest prompts (up to 2,801 tokens), 256-token prompt-pass chunks keep it under, but cost\n"
     "  2.0-2.9x the first token and change the tokens, so its gate does not cover them.\n",
     "- Our 24 GB configuration can go over 24 GB on prompts longer than about 500 tokens: 25.0 GB at 3K tokens, 28.2 GB at\n"
     "  30K. On six long gate prompts (726-2,801 tokens), 256-token prompt-pass chunks kept it under, but cost 2.0-2.9x the\n"
     "  first token and changed the tokens, so its gate does not cover them.\n"),
    ("- **24 GB (w13b): under 24 GB on prompts of up to about 500 tokens; longer prompts use more, 28.2 GB at 30K tokens**\n",
     "- **24 GB (w13b): under 24 GB in our runs on prompts of up to about 500 tokens, with little room; longer prompts use more, 28.2 GB at 30K tokens**\n"),
    ("    reached 24.01 GB, 14 MB over the line, in the prompt pass of the 520-token prompt, the ninth in its process, in both\n"
     "    passes [K, BM:w24g].\n",
     "    reached 24.01 GB, 14 MB over the line, in the prompt pass of the 520-token prompt, the ninth of the 12 measured\n"
     "    prompts (two warm-ups ran first), in both passes [K, BM:w24g].\n"),
    ("  - Longer prompts use more: 25.03 GB at 3,003 tokens, 25.67 GB at 6,980, 25.82 GB at 9,988 and 28.17 GB at 29,988\n"
     "    (default prefill, 64-token answers) [K, LP:data/tue_afternoon_1110.md].\n",
     "  - Longer prompts use more: 25.03 GB at 3,003 tokens, 25.67 GB at 6,980, 25.82 GB at 9,988 and 28.17 GB at 29,988\n"
     "    (default prefill; 256-token answers at 3,003 and 6,980 tokens, 64-token ones at 9,988 and 29,988). The 3,003-token\n"
     "    run had a 36 GB Mac simulated on this one: memory locked away left the engine about 25 GB, and its GPU working set\n"
     "    was capped at 27 GB [K, LP:data/tue_afternoon_1110.md sections 1, 2 and 4].\n"),
    ("    561 tokens), but the 520-token math prompt above crossed by 14 MB, and the gate's 481-token row reached 23.910 GB.\n"
     "    Hence \"up to about 500 tokens\". The limit is not exact: the same 592-token prompt, first in a fresh process both\n"
     "    times, peaked at 23.954 GB once and at 24.112 GB once [K, EV:RESULT_card_w24; EV:w24runs; BM:w24g].\n",
     "    561 tokens), but the 520-token math prompt above crossed by 14 MB, and a 448-token GPQA prompt raised its\n"
     "    process's peak to 23.954 GB, 46 MB under the line. Hence \"up to about 500 tokens\", as measured, not as a bound:\n"
     "    the same 592-token prompt, first in a fresh process both times, peaked at 23.954 GB once and at 24.112 GB once,\n"
     "    0.16 GB apart [K, EV:RESULT_card_w24; EV:w24runs; BM:w24g]. Near 450-520 tokens the label is no guarantee [G].\n"),
    ("  - Only 256-token chunks keep long prompts under the line, and neither smaller chunk is the gated configuration. On the\n"
     "    gate's six longest prompts (726-2,801 tokens), 512-token chunks still peak at 24.345 GB, at 1.3-1.6x the first\n"
     "    token, and 256-token chunks at 23.954 GB, at 2.0-2.9x. Their tokens differ from the gated configuration's on 6 / 6 and 5 / 6\n"
     "    prompts, and their logits hashes on all 6 [K, BM:w24e].\n",
     "  - On six long gate prompts (726-2,801 tokens), only 256-token chunks kept it under the line, and neither smaller\n"
     "    chunk is the gated configuration: 512-token chunks still peaked at 24.345 GB, at 1.3-1.6x the first token, and\n"
     "    256-token chunks at 23.954 GB, at 2.0-2.9x. Their tokens differ from the gated configuration's on 6 / 6 and 5 / 6\n"
     "    prompts, and their logits hashes on all 6 [K, BM:w24e]. Prompts of 10K-30K tokens were not run with smaller chunks.\n"),
    ("  prompts at 24 GB [K, SE:h2h/results_low24]. On the gate's long\n"
     "  prompts (726-2,801 tokens) it is 5.7-14.6 s [K, BM:w24e]. On longer prompts it is 18.0 s at 3,003 tokens, 60.4 s at\n"
     "  9,988 and 195.0 s at 29,988 with the default prefill.",
     "  prompts at 24 GB [K, SE:h2h/results_low24]. On six long gate\n"
     "  prompts (726-2,801 tokens) it is 5.7-14.6 s [K, BM:w24e]. On longer prompts it is 18.0 s at 3,003 tokens, 42.1 s at\n"
     "  6,980, 60.4 s at 9,988 and 195.0 s at 29,988 with the default prefill."),
    ("  30.60 GB at 30K) [K, LP:data/tue_afternoon_1110.md]. Its 256-token answer to the 6,980-token prompt is the default\n"
     "  prefill's, character for character [K, same file]. At 32 GB it is exact except when the prompt's last 1,024-token\n"
     "  chunk is under about 200 tokens [K, LP:data/pfa_tail_1111.md]; the 24 GB line was not checked on such a prompt.\n",
     "  30.60 GB at 30K) [K, LP:data/tue_afternoon_1110.md]. Its 256-token answer to the 6,980-token prompt (last chunk 836\n"
     "  tokens) is the default prefill's, character for character [K, same file]. It runs only on prompts over 1,024 tokens,\n"
     "  and at 32 GB it is exact except when the last 1,024-token chunk is under about 200 tokens [K,\n"
     "  LP:data/pfa_tail_1111.md]; the 24 GB line was not checked on such a prompt.\n"),
    ("  memory line holds on prompts of up to about 500 tokens (section 6.3). The exact E24 has no chat or math, and\n",
     "  memory line held in our runs on prompts of up to about 500 tokens, with little room (section 6.3). The exact E24 has\n"
     "  no chat or math, and\n"),
    ("(prompt-pass chunks of 1,024, 512 and 256 tokens on the gate's six longest prompts)",
     "(prompt-pass chunks of 1,024, 512 and 256 tokens on six long gate prompts, 726-2,801 tokens)"),
]

MARKER = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s")
bad, out = 0, {}
for f, reps in R.items():
    t = (P / f).read_text()
    for o, w in reps:
        c = t.count(o)
        if c != 1:
            bad += 1; print(f"BAD x{c} {f}: {o.splitlines()[0][:90]!r}"); continue
        for ln in w.split("\n")[1:]:
            if ln.strip() and MARKER.match(ln) and not re.match(r"^\s*- ", ln) and not ln.startswith("| "):
                bad += 1; print(f"BAD marker {f}: {ln[:90]!r}")
        t = t.replace(o, w); print(f"OK  {f}: {o.splitlines()[0][:90]}")
    out[f] = t
for f, t in out.items():                                   # what the build's lint would flag, and leftovers
    for i, ln in enumerate(t.split("\n"), 1):
        if re.search(r"Anthropic|BIGMOE|\bLeap [0-9a-f]|six longest|It stays under 24 GB|holds only up to", ln):
            print(f"CHECK {f}:{i}: {ln.strip()[:110]}")
if "--write" in sys.argv and not bad:
    bak = P / f".v110_check_bak_{time.strftime('%H%M')}"; bak.mkdir(exist_ok=True)
    for f, t in out.items():
        (bak / f).write_text((P / f).read_text()); (P / f).write_text(t)
    print("written; backups in", bak)
print("bad", bad)
