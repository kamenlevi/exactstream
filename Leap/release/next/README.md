# release/next: the agent server (next release; not in 1.1.0)

Copies of the tool-call working copy (release/cleanclone/agent/clone): serve.py + agentio.py (Anthropic Messages with
tools for Claude Code, OpenAI Responses for Codex, chat completions with tools; streaming, stop sequences, pings, a
priority queue; cache hooks sys_end / side / lane / conv; canonical per-turn rendering) and the launchers
tools/claude_local.sh, tools/codex_local.sh, tools/claude_code.sh and tools/codex_code.sh (one command: the server +
Claude Code, or + Codex). The clone's files must equal these (sha256 in each run folder).
serve.py sets BIGMOE_IDLE_MS=50 (an idle engine naps instead of spinning a core; data/idle_ms_1111.md).
EvalSuite's cold 32 GB agent test (11-11 01:58) ran pin 39947b1, before codex_code.sh and BIGMOE_IDLE_MS came in.
Records: data/agent_tools_1110.md, data/agent_cc_1111.md (+ later data/agent_*.md).

**Conversation cache (11-11 05:2x, the cached agent runs' pin).** serve.py now carries the cache hook (CONV_CACHE=1 by
default, CONV_CACHE=0 turns it off; CONV_CACHE_GB 0.6 / 2.5 / 6 at --mem 24 / 32 / 45 unless set), and
engine/loop/convcache.py is BigMoE's module at 3c068d7 (sha256 55631cb8: bitwise clones for the branch copy and the GDN
checkpoints). A turn restores the deepest stored checkpoint of the same conversation and prefills only the new tokens.
Checked bitwise against a cold prefill on every hit at 32 GB (T1e 05:07-05:20: 12 of 12 hits PASS, branches on;
data/agent_cc_1111.md); 45 GB (chunk-major) gets its own check (T1-45) before any cached 45 GB run; 24 GB is unchecked.
The agent clone of the cold runs (39947b1) is unchanged; the cached runs use release/cleanclone/cache, whose files equal
these (SHA256SUMS).
