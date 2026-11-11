"""v1.1.0: RELEASE describes what package 1.1.0 ships (chat.py and serve.py), and MAIN stops calling the layer-major
prefill (PF-A) unshipped: chat.py runs it by default at 32 and 24 GB, not at 45 GB (orchestrator Wed 11 Nov 03:4x;
Leap release/src d37fb68 = the candidate build pkg_110g). Applies on top of v110_pfa.py; release_smallram.py still
applies after it (MAIN's "- **11 November 2026 (v1.1.0).** PF-B's difference" kept).
Every number re-derived by the editor from Leap's raw records with rd_release_chat.py:
- BIGMOE_IDLE_MS=50: serve.py's CPU time over 60 s idle after a request, without it 47.44 -> 107.48 s (24 GB) and
  68.02 -> 128.05 s (32 GB) = 100 % of one core (data/agent_cc_1111/idle_cpu_1111/idle_cpu.log); with it 26.02 -> 27.0 s
  and 36.93 -> 37.86 s = 1.63 and 1.55 % (data/idle_ms_1111/runner.log). Decode A/B (run.py chat --kind code, HumanEval
  68 / 16 / 91, A B B A): 24 GB A 129.2, 128.7 / B 129.0, 129.5 -> x1.0023; 32 GB A 130.0, 131.9 / B 128.6, 129.7 ->
  x0.9863; spread between two runs of one setting 0.39 % (24) and 1.46 % (32); tokens equal in all 4 runs per line.
- PF-A on the chat configuration (first token / lifetime peak, default -> PF-A): 32 GB 10K 78.41 -> 22.71 s (x3.45),
  34.01 -> 34.83 GB; 30K 252.24 -> 68.55 s (x3.68), 35.48 -> 38.49 GB (data/prefill_1109/cli_1110_000737 / 000918 /
  001004 / 001438_runs.json). 24 GB 7K (chat.py) 42.06 -> 17.70 s (x2.38), 25.67 -> 25.98 GB; 10K 60.44 -> 23.74 s
  (x2.55), 25.82 -> 26.54 GB; 30K 194.95 -> 71.73 s (x2.72), 28.17 -> 30.60 GB (data/tue_afternoon_1110/pfa24_*.jsonl,
  p24_{10k,30k}_{default,pfa}_runs.json). So 3.5-3.7x at 32 GB, 2.4-2.7x at 24 GB, +0.3-3.0 GB.
- Checks: pkg_110c's clean-clone test (10 Nov 20:58-20:59): chat.py --mem 24 rc 0, 72 tokens, peak 23.63 GB; serve.py
  --mem 24: OpenAI [DONE] 1 + finish_reason 1, Anthropic message_stop 1 + 4 text deltas, stopped on SIGTERM
  (data/w24_pkg/candidate_110c/report.txt). chat.py's later changes: pty test 22 / 22 (stand-in loop), real engine
  chat.py --mem 32 twice: 10 Nov 22:22 (rc 0, peak 31.23 GB) and 11 Nov 01:32 (/file, rc 0, PASS)
  (data/chatpy_fixes_1110/).
- pkg_w24 (the build tested 10 Nov 10:18) vs pkg_110g: docs (CITATION, CONFIGS, INSTALL, LIMITS, README, RESULTS),
  MANIFEST.json, run.py (the --env help string only), tools/clean_clone_test.sh (+ chat.py / serve.py at 24 GB), and
  chat.py, serve.py new.
Wording within the build's lint: the 04:03 text failed it (04:04 dry build) on "Anthropic" (no product names) and on
"BIGMOE_IDLE_MS" and "Leap" (session names outside a snapshot path). So serve.py's formats are named by endpoint
(/v1/chat/completions, /v1/messages), the idle setting is described without its variable name, and the commit is
"LP d37fb68". Re-applied from the 04:03 backups.
usage: python3 -I v110_chat.py [--write]   (backups: ~/Papers/.v110_chat_bak_<time>/)"""
import pathlib
import re
import sys
import time

P = pathlib.Path.home() / "Papers"
R = {"RELEASE.md": [], "MAIN.md": [], "SMALL_RAM.md": [], "PATH_DRAFT.md": []}

# ---- RELEASE
R["RELEASE.md"] += [
    # abstract
    ("24 GB line: one gated lossy configuration, under the names lossy and chat (section 2; [SMALL-RAM]). At 45 GB, the exact\n",
     "24 GB line: one gated lossy configuration, under the names lossy and chat (section 2; [SMALL-RAM]). It also adds a chat\n"
     "in the terminal and a local server for chat apps in two common API formats, text only (section 4). At 45 GB, the exact\n"),
    # section 2, the long-prompt bullet
    ("  7.9K-token agent prompt at 433 and 500 tok/s [K] (exact E; lifetime peak 45.09 / 46.23 GB [K], above the 45.0 line\n"
     "  on that prompt).\n",
     "  7.9K-token agent prompt at 433 and 500 tok/s [K] (exact E; lifetime peak 45.09 / 46.23 GB [K], above the 45.0 line\n"
     "  on that prompt). At 32 and 24 GB, chat.py runs PF-A by default (section 4).\n"),
    # section 4: the 1.1.0 test sentence, and the new part
    ("Package 1.1.0's code passed the same test on 10 Nov (10:18-10:23): all eight configs gave their references' tokens on two\n"
     "prompts each, the 24 GB pair on HumanEval against the research runs, with every module and opened file inside the\n"
     "allowed folders [K, LP:data/w24_pkg/candidate/report.txt]. The 1.1.0 package differs from that build only in its\n"
     "documents and one help string [K, LP:data/w24_pkg.md].\n",
     "Package 1.1.0's engine passed the same test on 10 Nov (10:18-10:23): all eight configs gave their references' tokens on\n"
     "two prompts each, the 24 GB pair on HumanEval against the research runs, with every module and opened file inside the\n"
     "allowed folders [K, LP:data/w24_pkg/candidate/report.txt]. The 1.1.0 package differs from that build in its documents,\n"
     "one help string and two new programs, chat.py and serve.py, which the test script now also runs (below) [K,\n"
     "LP:data/w24_pkg.md].\n"
     "\n"
     "**Chat and a local server (1.1.0).** Package 1.1.0 adds two programs that load the engine once and run every answer\n"
     "through run.py's decode loop, unchanged; both default to the chat configuration at 32 GB (code: LP:release/src/chat.py\n"
     "and LP:release/src/serve.py). `chat.py` is a chat in the terminal: answers print as they are written, with the live\n"
     "speed in the window title; a paste is one message at any length, and `/file PATH message` sends a file, then the\n"
     "message. A built-in system line tells the model that it has no tools. `serve.py` serves chat apps on the same Mac\n"
     "(127.0.0.1) in two API formats, OpenAI's chat completions (`/v1/chat/completions`) and messages (`/v1/messages`), with\n"
     "streaming, one request at a time. In 1.1.0 it handles text only: tool definitions and tool calls are not mapped, so it\n"
     "cannot serve an agent that calls tools.\n"
     "- At 32 and 24 GB chat.py prefills long prompts layer-major (PF-A, section 2; `R57_PFLM=0` in the shell turns it off);\n"
     "  run.py and serve.py do not. On the chat configuration PF-A brings the first token 3.5-3.7x sooner at 32 GB (10K- and\n"
     "  30K-token prompts) and 2.4-2.7x sooner at 24 GB (7K-30K tokens), for 0.3-3.0 GB more lifetime peak (section 5) [K,\n"
     "  LP:data/prefill_1109.md; LP:data/tue_afternoon_1110.md sections 2 and 4]. Its tokens are the default prefill's except\n"
     "  when a prompt's last 1,024-token chunk is under about 200 tokens [K, LP:data/pfa_tail_1111.md].\n"
     "- Both make the extension's loader thread nap after 50 ms without work instead of spinning, so an idle chat or server\n"
     "  no longer keeps a core busy. Idle after a request (CPU time over 60 s), serve.py used a full core without the nap\n"
     "  and 1.5-1.6 % of one with it, at 24 and 32 GB [K, LP:data/agent_cc_1111.md; LP:data/idle_ms_1111.md]. Decode speed\n"
     "  changed by x1.002 at 24 GB and x0.986 at 32 GB, within the spread between two runs of the same setting (up to 0.4\n"
     "  and 1.5 %), and the tokens stayed the same (three HumanEval prompts, four runs per line) [K,\n"
     "  LP:data/idle_ms_1111.md].\n"
     "- Checks: in the clean-clone test of the first build that held them (10 Nov 20:57-20:59), chat.py answered a message at\n"
     "  24 GB (lifetime peak 23.63 GB), and serve.py at 24 GB answered a streamed request in each format and stopped on\n"
     "  SIGTERM [K, LP:data/w24_pkg/candidate_110c/report.txt]. chat.py's later changes (the paste handling, `/file`, the\n"
     "  system line, the window title) passed a terminal test against a stand-in for the engine (22 of 22 checks) and two\n"
     "  runs on the real engine at 32 GB [K, LP:data/chatpy_fixes_1110.md].\n"),
    # section 5: peaks with chat.py's PF-A
    ("  BM:data/pfd_long/summary.txt], and 34.0 / 35.5 GB for the 32 GB chat configuration [K, LP:data/prefill_1109.md]. The\n"
     "  24 GB line holds only up to about 500 tokens: 25.0-28.2 GB at 3K-30K tokens [K, section 2].\n",
     "  BM:data/pfd_long/summary.txt], and 34.0 / 35.5 GB for the 32 GB chat configuration, 34.8 / 38.5 GB with chat.py's\n"
     "  PF-A [K, LP:data/prefill_1109.md]. The 24 GB line holds only up to about 500 tokens: 25.0-28.2 GB at 3K-30K tokens\n"
     "  [K, section 2], and 26.0-30.6 GB at 7K-30K tokens with chat.py's PF-A [K, LP:data/tue_afternoon_1110.md].\n"),
    # section 5: serve.py / chat.py limits
    ("  pair); other Macs and macOS versions are untested, and the 24 GB line was not run on a real 36 GB Mac (section 3).\n",
     "  pair); other Macs and macOS versions are untested, and the 24 GB line was not run on a real 36 GB Mac (section 3).\n"
     "- serve.py handles text only (no tool calls in 1.1.0) and runs one request at a time. Every request, and every chat.py\n"
     "  turn, prefills its whole prompt: nothing is cached across requests or turns.\n"),
    # section 8
    ("  36 GB Mac and the peaks by prompt length LP:data/tue_afternoon_1110.md and LP:data/tue_afternoon_1110/; Strata at 24 GB\n"
     "  SE:h2h/results_low24.\n",
     "  36 GB Mac and the peaks by prompt length LP:data/tue_afternoon_1110.md and LP:data/tue_afternoon_1110/; Strata at 24 GB\n"
     "  SE:h2h/results_low24.\n"
     "- chat.py and serve.py (1.1.0): the code LP:release/src/chat.py and LP:release/src/serve.py (LP d37fb68); the first\n"
     "  build's clean-clone check LP:data/w24_pkg/candidate_110c/report.txt; chat.py's later changes and their tests\n"
     "  LP:data/chatpy_fixes_1110.md and LP:data/chatpy_fixes_1110/. PF-A on the chat configuration: 32 GB\n"
     "  LP:data/prefill_1109.md, runs LP:data/prefill_1109/ (cli_1110_000737 and 000918 at 10K, 001004 and 001438 at 30K,\n"
     "  default and PF-A); 24 GB LP:data/tue_afternoon_1110.md sections 2 and 4, runs LP:data/tue_afternoon_1110/\n"
     "  (pfa24_default and pfa24_pfa.jsonl at 7K, p24_10k_* and p24_30k_*). The idle nap: LP:data/agent_cc_1111.md and\n"
     "  LP:data/agent_cc_1111/idle_cpu_1111/ (idle CPU without it), LP:data/idle_ms_1111.md and LP:data/idle_ms_1111/ (with\n"
     "  it, and the decode runs).\n"),
    # Corrections
    ("- **11 November 2026 (v1.1.0).** Package 1.1.0 adds a 24 GB line (sections 1-5; [SMALL-RAM]). PF-B's difference on a\n",
     "- **11 November 2026 (v1.1.0).** Package 1.1.0 adds a 24 GB line (sections 1-5; [SMALL-RAM]), a chat in the terminal\n"
     "  and a local server (sections 4 and 5). PF-B's difference on a\n"),
]

# ---- MAIN
ROW11_OLD = "| [K, F:2026-11-08_0800; LP:data/pfa_tail_1111.md] | built; needs a 0.1 GB memory trim |"
ROW11_NEW = ("| [K, F:2026-11-08_0800; LP:data/pfa_tail_1111.md] | built; at 45 GB it needs a 0.1 GB memory trim; chat.py's "
             "default at 32 and 24 GB (package 1.1.0) |")
R["MAIN.md"] += [
    ("prompt's last 1,024-token chunk is under about 200 tokens (section 6.2) [K, F:2026-11-08_0800; LP:data/pfa_tail_1111.md].\n"
     "On short prompts the first token is still 2.7-4.6 s (section 6.3), except on chat with PF-D.\n",
     "prompt's last 1,024-token chunk is under about 200 tokens (section 6.2) [K, F:2026-11-08_0800; LP:data/pfa_tail_1111.md].\n"
     "The package's terminal chat (chat.py, package 1.1.0) runs it by default at 32 and 24 GB; at 45 GB it is not shipped\n"
     "([RELEASE] section 4). On short prompts the first token is still 2.7-4.6 s (section 6.3), except on chat with PF-D.\n"),
    ("exact unless the last chunk is under about 200 tokens (built, not shipped) | BM:data/pfd/pfd_report_0009.txt; "
     "F:2026-11-08_0800 |",
     "exact unless the last chunk is under about 200 tokens (chat.py's default at 32 and 24 GB in package 1.1.0; not "
     "shipped at 45 GB) | BM:data/pfd/pfd_report_0009.txt; F:2026-11-08_0800; LP:release/src/chat.py |"),
    (ROW11_OLD, ROW11_NEW),
    ("  last 1,024-token chunk is about 200 tokens or longer; a shorter one changed the tokens (sections 3.2, 4.5, 6.2, 7\n"
     "  and 11.1).\n",
     "  last 1,024-token chunk is about 200 tokens or longer; a shorter one changed the tokens (sections 3.2, 4.5, 6.2, 7\n"
     "  and 11.1). It now ships in package 1.1.0, as chat.py's default at 32 and 24 GB (sections 4.5, 7 and 11.1).\n"),
]

# ---- SMALL_RAM 6.4
R["SMALL_RAM.md"] += [
    ("  chunk is under about 200 tokens [K, LP:data/pfa_tail_1111.md]; the 24 GB line was not checked on such a prompt.\n",
     "  chunk is under about 200 tokens [K, LP:data/pfa_tail_1111.md]; the 24 GB line was not checked on such a prompt.\n"
     "  Package 1.1.0's terminal chat, chat.py, runs it by default at 24 and 32 GB ([RELEASE] section 4).\n"),
]

# ---- PATH_DRAFT (MAIN 11.1's row)
R["PATH_DRAFT.md"] += [(ROW11_OLD, ROW11_NEW)]

MARKER = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s")      # a wrapped line must not start like a list item
bad, out = 0, {}
for f, reps in R.items():
    t = (P / f).read_text()
    for o, w in reps:
        c = t.count(o)
        if c != 1:
            bad += 1; print(f"BAD x{c} {f}: {o.splitlines()[0][:90]!r}"); continue
        for ln in w.split("\n")[1:]:                      # continuation lines of the new text
            if ln.strip() and MARKER.match(ln) and not ln.startswith("- ") and not ln.startswith("| "):
                bad += 1; print(f"BAD marker {f}: {ln[:90]!r}")
        t = t.replace(o, w); print(f"OK  {f}: {o.splitlines()[0][:90]}")
    out[f] = t
for f in ("MAIN.md", "RELEASE.md", "SMALL_RAM.md", "PATH_DRAFT.md"):   # anything left that calls PF-A unshipped
    for i, ln in enumerate(out[f].split("\n"), 1):
        if re.search(r"(?i)(layer-major|PF-A)[^.|]{0,80}not shipped|not shipped[^.|]{0,20}(layer-major|PF-A)", ln) \
                and "45 GB" not in ln:
            print(f"CHECK {f}:{i}: {ln.strip()[:120]}")
if "--write" in sys.argv and not bad:
    bak = P / f".v110_chat_bak_{time.strftime('%H%M')}"; bak.mkdir(exist_ok=True)
    for f, t in out.items():
        (bak / f).write_text((P / f).read_text()); (P / f).write_text(t)
    print("written; backups in", bak)
print("bad", bad)
