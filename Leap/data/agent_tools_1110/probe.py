"""Tool-call probe: 30 calls (10 requests x the Anthropic, Responses and chat-completions endpoints) against serve.py with six
tools. Per call: format (a tool call came back), right tool, right arguments (strings case-insensitive and trimmed, numbers
by value, arrays as sets). Usage: probe.py <port> <out.jsonl>"""
import json, sys, time, urllib.request

PORT, OUT = int(sys.argv[1]), sys.argv[2]
S = lambda **p: {"type": "object", "properties": p, "required": list(p)}
TOOLS = [("get_weather", "Get the current weather for a city.", S(city={"type": "string"}, unit={"type": "string", "enum": ["celsius", "fahrenheit"]})),
         ("add_numbers", "Add two numbers.", S(a={"type": "number"}, b={"type": "number"})),
         ("search_files", "Find files whose names match a glob pattern.", S(pattern={"type": "string"}, max_results={"type": "integer"})),
         ("send_email", "Send an email.", S(to={"type": "string"}, subject={"type": "string"}, body={"type": "string"})),
         ("create_event", "Create a calendar event.", S(title={"type": "string"}, date={"type": "string", "description": "YYYY-MM-DD"},
                                                        attendees={"type": "array", "items": {"type": "string"}})),
         ("convert_currency", "Convert an amount between currencies (ISO codes).", S(amount={"type": "number"}, **{"from": {"type": "string"}}, to={"type": "string"}))]
CASES = [("What's the weather in Paris, in celsius?", "get_weather", {"city": "Paris", "unit": "celsius"}),
         ("Add 17 and 25.", "add_numbers", {"a": 17, "b": 25}),
         ("Find files matching *.py, at most 5 results.", "search_files", {"pattern": "*.py", "max_results": 5}),
         ("Email bob@example.com with the subject Lunch and the body: See you at noon", "send_email",
          {"to": "bob@example.com", "subject": "Lunch", "body": "See you at noon"}),
         ("Create an event called Standup on 2026-11-12 with Ann and Bo.", "create_event",
          {"title": "Standup", "date": "2026-11-12", "attendees": ["Ann", "Bo"]}),
         ("Convert 100 USD to EUR.", "convert_currency", {"amount": 100, "from": "USD", "to": "EUR"}),
         ("How hot is it in Tokyo, in fahrenheit?", "get_weather", {"city": "Tokyo", "unit": "fahrenheit"}),
         ("What is 3.5 plus 4.25?", "add_numbers", {"a": 3.5, "b": 4.25}),
         ("Search for files named report*, max 3 results.", "search_files", {"pattern": "report*", "max_results": 3}),
         ("Change 250 GBP into JPY.", "convert_currency", {"amount": 250, "from": "GBP", "to": "JPY"})]


def post(path, body):
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                               headers={"content-type": "application/json", "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(r, timeout=600) as f:
        return json.loads(f.read())


def call(api, q):
    if api == "anthropic":
        r = post("/v1/messages", {"model": "exactstream", "max_tokens": 256, "messages": [{"role": "user", "content": q}],
                                  "tools": [{"name": n, "description": d, "input_schema": s} for n, d, s in TOOLS]})
        c = [b for b in r["content"] if b["type"] == "tool_use"]
        return (c[0]["name"], c[0]["input"]) if c else (None, None)
    if api == "responses":
        r = post("/v1/responses", {"model": "exactstream", "input": q,
                                   "tools": [{"type": "function", "name": n, "description": d, "parameters": s} for n, d, s in TOOLS]})
        c = [i for i in r["output"] if i["type"] == "function_call"]
        return (c[0]["name"], json.loads(c[0]["arguments"])) if c else (None, None)
    r = post("/v1/chat/completions", {"model": "exactstream", "max_tokens": 256, "messages": [{"role": "user", "content": q}],
                                      "tools": [{"type": "function", "function": {"name": n, "description": d, "parameters": s}} for n, d, s in TOOLS]})
    c = r["choices"][0]["message"].get("tool_calls") or []
    return (c[0]["function"]["name"], json.loads(c[0]["function"]["arguments"])) if c else (None, None)


def same(a, b):
    if isinstance(b, str):
        return isinstance(a, str) and a.strip().lower() == b.strip().lower()
    if isinstance(b, (int, float)) and not isinstance(b, bool):
        try:
            return abs(float(a) - float(b)) < 1e-9
        except Exception:
            return False
    if isinstance(b, list):
        return isinstance(a, list) and sorted(str(x).strip().lower() for x in a) == sorted(str(x).strip().lower() for x in b)
    return a == b


tally = {"calls": 0, "format": 0, "tool": 0, "args": 0}
with open(OUT, "w") as f:
    for api in ("anthropic", "responses", "chat"):
        for q, tname, targs in CASES:
            t0 = time.time()
            try:
                name, args = call(api, q)
                err = None
            except Exception as e:
                name, args, err = None, None, f"{type(e).__name__}: {e}"
            ok_f, ok_t = name is not None, name == tname
            ok_a = ok_t and isinstance(args, dict) and all(k in args and same(args[k], v) for k, v in targs.items())
            tally["calls"] += 1; tally["format"] += ok_f; tally["tool"] += ok_t; tally["args"] += ok_a
            rec = {"api": api, "q": q, "want": [tname, targs], "got": [name, args], "format": ok_f, "tool": ok_t, "args": ok_a,
                   "s": round(time.time() - t0, 1), "err": err}
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(("OK  " if ok_a else "BAD ") + f"{api:9s} {tname:16s} got {name} {json.dumps(args)[:90]}" + (f" ERR {err}" if err else ""), flush=True)
print(f"PROBE: {tally['calls']} calls; format {tally['format']}, right tool {tally['tool']}, right args {tally['args']}")
