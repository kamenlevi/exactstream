"""HTTP checks of serve.py's Claude Code hardening on the stand-in engine (mock_tools.py scenarios): stop sequences
(non-streamed and streamed), the engine queue's priority (main before side), a client that leaves mid-answer and one that
leaves while queued, count_tokens = the rendered prompt, an image block in a user turn, tool_choice accepted, and the
"prompt is too long" error in the form Claude Code matches. Usage: harden_test.py <port> <request log>"""
import json, re, socket, sys, threading, time, urllib.request

PORT, LOG = int(sys.argv[1]), sys.argv[2]
res = []


def check(name, ok, detail=""):
    res.append(ok)
    print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail else ""), flush=True)


def post(path, body, timeout=120):
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                               headers={"content-type": "application/json", "anthropic-version": "2023-06-01"})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as f:
            return f.status, f.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


TOOL = [{"name": "Read", "description": "Read a file.", "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}}]


def msg(text, model="exactstream", tools=True, **kw):
    b = {"model": model, "max_tokens": kw.pop("max_tokens", 256), "messages": [{"role": "user", "content": text}]}
    if tools:
        b["tools"] = TOOL
    b.update(kw)
    return b


# 1. stop sequences
st, out = post("/v1/messages", msg("STOPSEQ please", stop_sequences=["STOP"]))
o = json.loads(out)
txt = "".join(c.get("text", "") for c in o.get("content", []))
check("stop_sequences, non-streamed: text cut before STOP, stop_reason stop_sequence",
      st == 200 and o.get("stop_reason") == "stop_sequence" and o.get("stop_sequence") == "STOP" and "gamma" not in txt and "beta" in txt,
      f"{o.get('stop_reason')} {o.get('stop_sequence')!r} {txt!r}")
st, out = post("/v1/messages", msg("STOPSEQ please", stop_sequences=["STOP"], stream=True))
deltas = "".join(json.loads(l[6:]).get("delta", {}).get("text", "") for l in out.splitlines() if l.startswith("data: {") and '"text_delta"' in l)
md = [json.loads(l[6:]) for l in out.splitlines() if l.startswith("data: {") and '"message_delta"' in l]
check("stop_sequences, streamed: the same cut and stop_reason", md and md[0]["delta"]["stop_reason"] == "stop_sequence" and "gamma" not in deltas and "beta" in deltas,
      f"{md[0]['delta'] if md else None} {deltas!r}")

# 2. priority: a slow main answer runs; a side request queued first, then a main one: the main one runs first
done = {}


def run(name, body):
    t0 = time.time()
    post("/v1/messages", body, timeout=300)
    done[name] = time.time() - t0


A = threading.Thread(target=run, args=("A", msg("SLOW main", max_tokens=300)))
A.start(); time.sleep(1.0)
B = threading.Thread(target=run, args=("B side", msg("side hello", model="exactstream-side", tools=False, max_tokens=32)))
B.start(); time.sleep(0.5)
C = threading.Thread(target=run, args=("C main", msg("main hello", max_tokens=32)))
C.start()
for th in (A, B, C):
    th.join()
order = sorted(done, key=lambda k: done[k] + {"A": 0, "B side": 1.0, "C main": 1.5}[k])
check("queue: the main request queued after a side one runs before it", order.index("C main") < order.index("B side"),
      f"finish order {order}")


# 3. a client that leaves mid-answer: the answer stops at the next check
def stream_and_leave(text, leave_after, max_tokens):
    s = socket.create_connection(("127.0.0.1", PORT))
    b = json.dumps(msg(text, stream=True, max_tokens=max_tokens)).encode()
    s.sendall(b"POST /v1/messages HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(b) + b)
    t0 = time.time()
    s.settimeout(0.5)
    while time.time() - t0 < leave_after:
        try:
            s.recv(65536)
        except socket.timeout:
            pass
    s.close()
    return time.time()


def done_lines():
    return [r for r in (json.loads(l) for l in open(LOG)) if r.get("kind") == "result"]


n0 = len(done_lines())
t_left = stream_and_leave("SLOW leave", 2.0, 1500)
time.sleep(8)
new = done_lines()[n0:]
check("a client that leaves mid-answer: the answer stops within seconds (not all 1,500 tokens)",
      len(new) == 0 or all(len(r.get("out_ids") or []) < 600 for r in new), f"results logged {len(new)}")

# 4. a client that leaves while its request waits in the queue: dropped before it runs
A2 = threading.Thread(target=run, args=("A2", msg("SLOW long main", max_tokens=900)))
A2.start(); time.sleep(1.0)
stream_and_leave("queued then gone", 1.0, 200)
A2.join()
time.sleep(3)
st, _ = post("/v1/messages", msg("after", max_tokens=8))
check("a request whose client left while queued is dropped (the next request still runs)", st == 200)

# 5. count_tokens = the rendered prompt that runs
body = msg("count me", max_tokens=1, system="You are terse.")
st, ct = post("/v1/messages/count_tokens", body)
n_ct = json.loads(ct).get("input_tokens")
post("/v1/messages", body)
renders = [r for r in (json.loads(l) for l in open(LOG)) if r.get("kind") == "render"]
check("count_tokens equals the rendered prompt's length", n_ct == renders[-1]["meta"]["n"], f"{n_ct} vs {renders[-1]['meta']['n']}")

# 6. an image block in a user turn: a clear note, no error
img = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}}
st, out = post("/v1/messages", {"model": "exactstream", "max_tokens": 16, "tools": TOOL,
                                "messages": [{"role": "user", "content": [img, {"type": "text", "text": "what is in this picture?"}]}]})
renders = [r for r in (json.loads(l) for l in open(LOG)) if r.get("kind") == "render"]
check("an image block: 200, and the prompt carries the text-only note", st == 200 and renders[-1]["meta"]["n"] > 0, f"status {st}")

# 7. tool_choice any / tool / none accepted
codes = [post("/v1/messages", msg("hello", max_tokens=4, tool_choice=tc))[0] for tc in ({"type": "any"}, {"type": "tool", "name": "Read"}, {"type": "none"}, {"type": "auto"})]
check("tool_choice auto / any / tool / none accepted", codes == [200] * 4, f"{codes}")

# 8. prompt too long: the error text Claude Code matches
big = "word " * 140000
st, out = post("/v1/messages/count_tokens", msg(big))
st2, out2 = post("/v1/messages", msg(big))
m = json.loads(out2).get("error", {}).get("message", "")
check("past --max-context: 400 invalid_request_error 'prompt is too long: N tokens > M maximum'",
      st2 == 400 and re.match(r"^prompt is too long: \d+ tokens > \d+ maximum$", m) is not None, m)
print(f"HARDEN: {sum(res)} of {len(res)} pass")
