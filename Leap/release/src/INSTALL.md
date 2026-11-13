# Install and run

## Requirements
- An Apple-silicon Mac with 64 GB of unified memory and the model on its internal SSD. Tested: MacBook Pro, M5 Pro
  (18-core CPU, 20-core GPU), 64 GB, 2 TB SSD; macOS 26.5 (25F71).
- The 24 GB setting (`--mem 24`) is meant for a Mac with 36 GB or more, but we have only tested it on the 64 GB MacBook
  Pro above. On a 36 GB Mac it leaves about 11 GB for macOS and your apps, so close the browser and other large apps
  first. Long prompts take it further: a 30K-token prompt reaches 28.2 GB, which leaves about 8 GB for macOS and your
  apps. Prompts near 64K tokens need about 32 GB (38 GB with `chat.py`'s layer-major prefill), more than a 36 GB Mac can
  spare. On a Mac with a different GPU type the tokens may differ slightly from ours, and the speed depends on that Mac's
  GPU and SSD.
- Python 3.12, CMake >= 3.27 and the Xcode command-line tools (for the extension).
- Disk: about 206 GB free for a fresh install: the model's own files (105.4 GB without its vision tower;
  [WEIGHTS.md](WEIGHTS.md)), the converted expert store (48 layers x 512 experts x 2.79 MB = 68.5 GB;
  [CONVERSION.md](CONVERSION.md)), the drafters (2.6 GB), the venv, and 25 GB kept free for the conversion and for swap.

## Install in one command
    git clone --single-branch https://github.com/kamenlevi/exactstream && cd exactstream
    bash setup.sh

`setup.sh` does the manual steps below in order and can be run again at any time: each step checks what is done and
skips it, and a stopped download or conversion resumes where it stopped.
1. It checks the Mac: Apple silicon, macOS, memory, Python 3.12, CMake, the Xcode command-line tools, and the free disk
   that what is left needs. It stops with the reason if one is missing.
2. It creates `.venv` and builds the extension (`install.sh`), before any download.
3. It downloads the three public pieces [WEIGHTS.md](WEIGHTS.md) lists, with no account or login: the model (105.4 GB)
   into `~/exactstream-data/model`, the drafter base (1.2 GB) into your Hugging Face cache, and our two drafter files
   (1.4 GB) into `~/exactstream-data/drafters`. Their revisions, sizes and checksums are pinned in `setup/files.tsv`.
4. It converts the expert store into `~/exactstream-data/model/records3` (about 10 minutes on the test MacBook Pro).
5. It runs `install.sh --store ~/exactstream-data/model --weights ~/exactstream-data/drafters`.

It prints the model's licence before downloading: the weights and everything derived from them, the converted store too,
stay under the Qwen Community License 1.0. `--data DIR` puts everything elsewhere; `--model DIR` and `--drafters DIR` use
pieces you already have; `--dry-run` checks and reports without changing anything; `--verify` also checks every file's
checksum (it reads about 107 GB).

## Install step by step
    git clone --single-branch https://github.com/kamenlevi/exactstream && cd exactstream
    bash install.sh --store /path/to/model-dir --weights /path/to/drafter-dir

`install.sh` writes the clone's path into the code (where it says `@ROOT@`), creates `.venv` with the pinned packages
(`requirements.txt`; MLX stays at 0.32.2, see [LIMITS.md](LIMITS.md)), builds the extension into `engine/ext/build`, and
links the store and the drafter files ([WEIGHTS.md](WEIGHTS.md)). The store must hold the converted expert store
`records3/` ([CONVERSION.md](CONVERSION.md)). `--offline` installs from a local uv cache instead of the network.

## Run
    .venv/bin/python run.py --config exact --mem 45 --prompt "Explain how a refrigerator works."
    .venv/bin/python run.py --config chat --mem 32 --prompt-file question.txt --max-tokens 512
    .venv/bin/python run.py --config chat --mem 24 --prompt "Explain how a refrigerator works."
    .venv/bin/python run.py --config exact --mem 45 --keys ifeval/1220,ifeval/3280   # benchmark items

It prints the answer, then whole decode (tokens / decode wall), whole request (tokens / (first token + decode wall)),
first token (the prefill wall) and the process's lifetime peak memory (GB = 10^9 bytes). The first run of a process
loads the model and runs one discarded warm-up prompt (`--no-warmup` skips it; the first prompt then runs cold).
The decode loop is the research harness's own loop, unchanged; `run.py` only turns a text prompt into a prompt item and
prints the answer and the loop's timing. The configs are in [CONFIGS.md](CONFIGS.md).

At `--mem 24`, and since 1.2.1 with `--config chat --mem 45`, a prompt of 256 tokens or fewer skips the prefill and runs
through the decode's check path (P2, new in 1.2.0 at 24 GB): about a third of the first-token time at 24 GB and a fifth at
45 GB, and an answer that can differ from the prefill path's. P2 is lossy and passed its own same-scores gate on each
memory line (CONFIGS.md).

## Chat and a local server
    .venv/bin/python chat.py --mem 24
    .venv/bin/python serve.py --mem 24 --port 8080

`chat.py` keeps the engine loaded between messages and prints each answer's tokens, speed, first token and lifetime peak;
Ctrl-C stops an answer, `/clear` starts over and `/exit` quits. A paste is one message, however long and over any number
of lines; Enter sends it. `/file PATH message` sends the file in a fenced block, then the message. The live speed
shows in the window title. At `--mem 32` and `--mem 24` it prefills long prompts
layer-major (a 2.4-3.8x faster first token, 0.3-3.2 GB more memory). Its tokens are the default prefill's, except when a
prompt runs less than about 200 tokens past a multiple of 1,024: there the arithmetic rounds differently and an answer
can differ. On a 36 GB
Mac, for conversations near 30K tokens, start it with `R57_PFLM=0` (`R57_PFLM=0 .venv/bin/python chat.py --mem 24`): that
saves 2.4 GB of peak memory, at about 2.7x the first-token time. Every turn in `chat.py` prefills the whole conversation.

`serve.py` serves chat apps and coding agents on this Mac (127.0.0.1) in two formats, with streaming and tool calls:
OpenAI (base URL `http://127.0.0.1:8080/v1`, any API key, model `exactstream`; chat completions, and the Responses API
that Codex uses) and Anthropic (base URL `http://127.0.0.1:8080`, `POST /v1/messages`). One request runs at a time and
the others wait. Like `chat.py`, it prefills long prompts layer-major at `--mem 32` and `--mem 24`. Ctrl-C or `kill`
stops it. It answers only programs on this Mac that address it as `127.0.0.1` or `localhost`: a request from a web page of
another site (its `Origin`), or one sent to another host name (DNS rebinding), is refused, so a site you visit cannot use it.

**Conversation cache.** At `--mem 45` and `--mem 32` the server keeps each conversation's state (up to 2.5 GB in all), so
a request that continues an earlier one reads only its new tokens. We checked it bit for bit against a full prefill on
every cache hit at 32 and at 45 GB (12 of 12 hits each). At 24 GB it is off, because it is unchecked there; `CONV_CACHE=1`
turns it on at your own risk, and `CONV_CACHE=0` turns it off at 45 and 32 GB. `CONV_CACHE_GB` sets its memory. With the
cache on, every request drafts with the DFlash2 drafter.

## Coding agents: Claude Code and Codex
    tools/claude_code.sh                # the 32 GB setting, Claude Code in the current folder
    tools/claude_code.sh --mem 45       # or 24; anything after it goes to claude, e.g. -p "fix the failing test"

`tools/claude_code.sh` starts `serve.py` unless one already answers on the port (`EXACTSTREAM_PORT`, default 8080),
waits until the engine is loaded (about a minute), runs Claude Code with the settings in `tools/claude_local.sh`, and
stops the server it started when Claude Code exits. Its settings apply to that one process: your own Claude Code
configuration (`~/.claude`) is not changed. You need Claude Code installed (`claude`).
- **The context window** Claude Code is told is 64K tokens at 45 and 32 GB and 128K at 24 GB (`EXACTSTREAM_CONTEXT`
  overrides it). A 128K prompt would take more memory than a 64 GB Mac can spare at 45 and 32 GB; at 24 GB it peaked
  at 51.6 GB. Claude Code compacts the conversation before it reaches the window. On a 36 GB Mac at 24 GB, use
  `EXACTSTREAM_CONTEXT=65536 R57_PFLM=0 tools/claude_code.sh --mem 24`, which keeps prompts near 32K tokens.
- **The status line** under Claude Code's prompt shows the time spent reading the prompt (`⏳ thinking 12s`), then the
  live speed (`⚡ 38.2 tok/s · 210 tok`), then the last answer (`✓ last: ...`). It comes from the server's `GET
  /statusline`; `tools/statusline.sh` prints it.
- **Tested:** EvalSuite's Claude Code agent test (a 30-prompt tool-call probe and 12 coding tasks in a sandbox) passed
  with the conversation cache at 32 GB, twice, and at 45 GB; without the cache at 32 GB the probe scored 27 of 30 and
  missed the 28 the test asks for. At 24 GB, without the cache, only a three-task smoke test ran (it passed), and every
  turn reads its whole prompt again: about 31 seconds to the first token at 12.6K tokens.
- **Codex:** `tools/codex_code.sh [--mem N]` does the same with Codex (`tools/codex_local.sh` writes its own Codex
  configuration folder, `~/.exactstream/codex`, and leaves `~/.codex` alone). It was only smoke-tested: three tasks at
  24 GB (two passed; in the third it appended a line without the newline before it), and its requests replayed through
  the conversation cache at 32 and 45 GB.

## Check your install
    bash tools/clean_clone_test.sh <this repository> /tmp/cc prep
    bash tools/clean_clone_test.sh <this repository> /tmp/cc run

The `run` phase runs every config on two benchmark prompts (up to 256 tokens each) with `HOME` set to an empty directory
and compares the tokens with the references in `verify/`: exact configs must match token for token. The lossy and chat
configs' tokens depend on what the process decoded before, so a prompt of theirs is checked only when the earlier prompts
ran exactly as in the reference run (same prompts, same order, same lengths); otherwise it is reported as information.
On the test MacBook Pro the `run` phase of 1.1.0 (Tue 10 Nov 2026) passed in about 4.5 minutes: all eight configs matched
their references, and every module and every file opened came from inside the allowed folders. Since 1.2.0 the 24 GB
configs' reference is the P2 gate's arm on 12 IFEval prompts, and since 1.2.1 that of `chat` at 45 GB is its P2 arm on
the same prompts (verify/README.md).
