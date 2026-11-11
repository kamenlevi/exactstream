"""Replays a serve.py request log (EXACTSTREAM_LOG_REQUESTS): posts every recorded request body, in order, to the same
path with the same client headers (user-agent, x-app, anthropic-beta / -version, originator), and reads each reply to the
end (streamed replies end when the connection closes). The prompts are the recorded ones whatever the server answers now,
so a conversation cache sees the same shared prefixes as the original sessions. Writes one JSON line per request.
Usage: replay.py <port> <recorded log> <out.jsonl>"""
import http.client, json, sys, time

PORT, LOG, OUT = int(sys.argv[1]), sys.argv[2], sys.argv[3]
KEEP = ("user-agent", "x-app", "anthropic-beta", "anthropic-version", "originator")
reqs = [r for r in (json.loads(l) for l in open(LOG)) if r.get("kind") == "request"]
with open(OUT, "w") as f:
    for k, r in enumerate(reqs):
        hdr = {"content-type": "application/json"}
        hdr.update({h: v for h, v in (r.get("headers") or {}).items() if h in KEEP})
        t0 = time.time()
        try:
            c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=1800)
            c.request("POST", r["path"], json.dumps(r["body"]), hdr)
            resp = c.getresponse()
            data = resp.read()
            st = resp.status
        except Exception as e:
            st, data = f"{type(e).__name__}: {e}", b""
        rec = {"k": k, "path": r["path"], "status": st, "s": round(time.time() - t0, 2), "bytes": len(data),
               "stream": bool(r["body"].get("stream")), "model": r["body"].get("model")}
        f.write(json.dumps(rec) + "\n")
        f.flush()
        print(f"{k:2d} {r['path']:28s} {st} {rec['s']:7.2f} s {len(data)} bytes", flush=True)
print(f"REPLAY: {len(reqs)} requests")
