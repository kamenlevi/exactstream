#!/bin/bash
# serve.py real-engine test (orchestrator Tue 10:1x; it had only passed the mock test): --mem 24 (chat-24 = w13b as gated),
# one streamed OpenAI request and one streamed Anthropic request; checks each SSE stream's structure, the server's own
# per-request line (tokens, tok/s, first token, lifetime peak) and errors; then stops the server. Usage:
# serve_test.sh <clone with serve.py> <outdir> [port]. Run under the caller's share lock.
CL=$1; O=$2; PORT=${3:-8093}
mkdir -p $O; cd $CL
.venv/bin/python serve.py --mem 24 --port $PORT > $O/serve.log 2>&1 &
SP=$!; T0=$(date +%s)
until [ "$(curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -o '"ready"')" = '"ready"' ]; do
  kill -0 $SP 2>/dev/null || { echo "FAIL: the server exited while loading"; tail -20 $O/serve.log; exit 2; }
  [ $(( $(date +%s) - T0 )) -gt 300 ] && { echo "FAIL: not ready in 300 s"; kill -INT $SP; exit 2; }
  sleep 3
done
echo "ready after $(( $(date +%s) - T0 )) s"
curl -s -m 10 http://127.0.0.1:$PORT/v1/models > $O/models.json
curl -sN -m 300 http://127.0.0.1:$PORT/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"Write a short Python function that checks whether a string is a palindrome, with one example."}],"stream":true,"stream_options":{"include_usage":true},"max_tokens":300}' \
  > $O/openai.sse; echo "openai curl rc $?"
curl -sN -m 300 http://127.0.0.1:$PORT/v1/messages -H 'content-type: application/json' -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"x","max_tokens":300,"stream":true,"messages":[{"role":"user","content":"Explain in two sentences why the sky is blue."}]}' \
  > $O/anthropic.sse; echo "anthropic curl rc $?"
kill -INT $SP; for i in $(seq 1 30); do kill -0 $SP 2>/dev/null || break; sleep 1; done
kill -0 $SP 2>/dev/null && { echo "the server ignored SIGINT for 30 s: TERM"; kill -TERM $SP; sleep 3; }
echo "server stopped (wall $(( $(date +%s) - T0 )) s)"
/opt/homebrew/opt/python@3.12/bin/python3.12 -I - $O <<'EOF'
import json, re, sys
O = sys.argv[1]
ok = True
def lines(p): return open(f"{O}/{p}", errors="replace").read().splitlines()
# OpenAI: data: {...} chunks, a finish_reason, a usage chunk, then data: [DONE]
d = [l[6:] for l in lines("openai.sse") if l.startswith("data: ")]
done = bool(d) and d[-1] == "[DONE]"
ch = [json.loads(x) for x in d if x != "[DONE]"]
txt = "".join((c["choices"][0].get("delta") or {}).get("content") or "" for c in ch if c.get("choices"))
fin = [c["choices"][0].get("finish_reason") for c in ch if c.get("choices") and c["choices"][0].get("finish_reason")]
use = [c["usage"] for c in ch if c.get("usage")]
o_ok = done and txt.strip() != "" and len(fin) == 1 and len(use) == 1
ok &= o_ok
print(f"OpenAI stream: {'PASS' if o_ok else 'FAIL'}: {len(ch)} chunks, [DONE] {done}, finish_reason {fin}, usage {use}, text {len(txt)} chars: {txt[:80]!r}")
# Anthropic: event order message_start, content_block_start, deltas, content_block_stop, message_delta, message_stop
ev, data = [], []
for l in lines("anthropic.sse"):
    if l.startswith("event: "): ev.append(l[7:])
    elif l.startswith("data: "): data.append(json.loads(l[6:]))
txt = "".join(x["delta"].get("text", "") for x in data if x.get("type") == "content_block_delta")
core = [e for e in ev if e != "ping"]
order = core[:2] == ["message_start", "content_block_start"] and core[-3:] == ["content_block_stop", "message_delta", "message_stop"] \
    and all(e == "content_block_delta" for e in core[2:-3])
md = [x for x in data if x.get("type") == "message_delta"]
a_ok = order and txt.strip() != "" and len(md) == 1 and md[0].get("delta", {}).get("stop_reason") in ("end_turn", "max_tokens")
ok &= a_ok
print(f"Anthropic stream: {'PASS' if a_ok else 'FAIL'}: events {len(ev)} (order {'ok' if order else 'WRONG: ' + ' '.join(core[:4]) + ' ... ' + ' '.join(core[-4:])}), "
      f"stop_reason {md[0]['delta'].get('stop_reason') if md else None}, usage {md[0].get('usage') if md else None}, text {len(txt)} chars: {txt[:80]!r}")
log = open(f"{O}/serve.log", errors="replace").read()
dn = re.findall(r"done: .*", log)
for x in dn: print("server:", x)
err = [l for l in log.splitlines() if re.search(r"Traceback|Error|Exception", l)]
ok &= len(dn) == 2 and not err
print(f"server log: {len(dn)} 'done' lines (2 expected), {len(err)} error lines" + "".join(f"\n  {l}" for l in err[:8]))
print("SERVE TEST", "PASS" if ok else "FAIL")
EOF
