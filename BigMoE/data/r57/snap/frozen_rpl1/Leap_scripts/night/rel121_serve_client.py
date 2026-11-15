"""rel121_serve_client.py (Leap 11-13): part E of rel121_check.sh. Five requests to serve.py --mem 45 (the cache on):
a long conversation's first turn (a miss: stored), a short request (<= 256 tokens: P2, which bypasses the cache), the long
conversation's follow-up (a hit, checked against a cold prefill by CONV_CACHE_CHECK=1), a second short one, a second
follow-up (a hit). Usage: rel121_serve_client.py <port> <long prompt file> <out jsonl>"""
import json, sys, time, urllib.request

port, longf, out = sys.argv[1], sys.argv[2], sys.argv[3]


def ask(msgs):
    body = json.dumps({"model": "exactstream", "messages": msgs, "max_tokens": 64}).encode()
    t0 = time.time()
    r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                                      {"content-type": "application/json"}), timeout=900)
    d = json.load(r)
    return r.status, d["choices"][0]["message"]["content"], d.get("usage", {}), round(time.time() - t0, 2)


res = []


def go(label, msgs):
    s, a, u, dt = ask(msgs)
    res.append(dict(label=label, status=s, prompt_tokens=u.get("prompt_tokens"), wall_s=dt))
    print(f"{label}: HTTP {s}, prompt tokens {u.get('prompt_tokens')}, wall {dt} s", flush=True)
    return a


A = [{"role": "user", "content": open(longf).read()}]
B = [{"role": "user", "content": "Name three primary colours."}]
a = go("A1 long, first turn (miss)", A)
A += [{"role": "assistant", "content": a}, {"role": "user", "content": "Which of those claims is the strongest? One sentence."}]
b = go("B1 short (P2)", B)
B += [{"role": "assistant", "content": b}, {"role": "user", "content": "Which of them is the warmest?"}]
a = go("A2 long, follow-up (hit)", A)
A += [{"role": "assistant", "content": a}, {"role": "user", "content": "And the weakest? One sentence."}]
go("B2 short (P2)", B)
go("A3 long, second follow-up (hit)", A)
open(out, "w").write("\n".join(json.dumps(r) for r in res) + "\n")
print(f"five requests: HTTP {[r['status'] for r in res]}, prompt tokens {[r['prompt_tokens'] for r in res]}")
