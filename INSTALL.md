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
- Disk: about 174 GB in all, for the model's own files (about 105 GB without its vision tower; [WEIGHTS.md](WEIGHTS.md))
  and the converted expert store (48 layers x 512 experts x 2.79 MB = 68.5 GB; [CONVERSION.md](CONVERSION.md)); keep
  about 20 GB free for swap.

## Install
    git clone https://github.com/kamenlevi/exactstream && cd exactstream
    bash install.sh --store /path/to/model-dir --weights /path/to/drafter-dir

`install.sh` writes the clone's path into the code (where it says `@ROOT@`), creates `.venv` with the pinned packages
(`requirements.txt`; MLX stays at 0.32.2, see [LIMITS.md](LIMITS.md)), builds the extension into `engine/ext/build`, and
links the store and the drafter files ([WEIGHTS.md](WEIGHTS.md)). `--offline` installs from a local uv cache instead of
the network.

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
saves 2.4 GB of peak memory, at about 2.7x the first-token time.

`serve.py` serves chat apps on this Mac (127.0.0.1) in two formats, with streaming: OpenAI (base URL
`http://127.0.0.1:8080/v1`, any API key, model `exactstream`) and Anthropic (base URL `http://127.0.0.1:8080`,
`POST /v1/messages`). Ctrl-C or `kill` stops it.

## Check your install
    bash tools/clean_clone_test.sh <this repository> /tmp/cc prep
    bash tools/clean_clone_test.sh <this repository> /tmp/cc run

The `run` phase runs every config on two benchmark prompts (up to 256 tokens each) with `HOME` set to an empty directory
and compares the tokens with the references in `verify/`: exact configs must match token for token. The lossy and chat
configs' tokens depend on what the process decoded before, so a prompt of theirs is checked only when the earlier prompts
ran exactly as in the reference run (same prompts, same order, same lengths); otherwise it is reported as information.
On the test MacBook Pro (Tue 10 Nov 2026) the `run` phase passed in about 4.5 minutes: all eight configs matched their
references, and every module and every file opened came from inside the allowed folders.
