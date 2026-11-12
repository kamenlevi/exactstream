#!/usr/bin/env python3
"""serve.py: the engine as a local OpenAI- and Anthropic-compatible server (127.0.0.1 only).

  .venv/bin/python serve.py                          # the 32 GB chat config on http://127.0.0.1:8080
  .venv/bin/python serve.py --mem 45 --port 8081

  OpenAI clients:     base URL http://127.0.0.1:8080/v1, any API key, model "exactstream"
                      (GET /v1/models, POST /v1/chat/completions, streaming as server-sent events)
  Anthropic clients:  ANTHROPIC_BASE_URL=http://127.0.0.1:8080, any key (POST /v1/messages, streaming;
                      POST /v1/messages/count_tokens)

The engine loads once and runs every request through the same decode loop as run.py (engine/loop/r57_screen_<snapshot>.py),
unchanged: serve.py hands the loop one prompt item per request (the model's chat template over the request's messages,
non-thinking, greedy) and streams the tokens each check commits (the session-vocabulary hook the chat and lossy configs run;
the exact configs send the answer when it is done). One request runs at a time; the others wait in a queue. Text only:
tool definitions and tool calls are not mapped yet, and every request prefills its whole prompt (no cache across
requests). The engine's own log goes to out/<tag>.log.

Work in progress (not in the 1.1.0 package): tool calls (agentio.py: Anthropic Messages, OpenAI Responses and chat
completions with tools), and each prompt item also carries sys_end / side / agent (agentio.meta) for a later conversation
cache. EXACTSTREAM_LOG_REQUESTS=<file>: each request's body, rendered ids and result, one JSON line each."""
import argparse, importlib.abc, importlib.util, itertools, json, os, queue, runpy, signal, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import run as R                                            # read_config, tokenizer, drafter_cached, DRAFTER, KIND_SET
import agentio as AI                                       # tool calls: the three wire formats <-> the chat template

NMAX = 99999                                               # requests per server run (the loop's prompt list is fixed at start)
MODEL = "exactstream"


class Job:
    def __init__(self, ids, cap, user="", meta=None):
        self.ids, self.cap, self.user = ids, cap, user   # user: the last user message (the loop's type rule reads it)
        self.meta = meta or {}                             # agentio.meta: sys_end, side, lane, conv, prefix_key
        self.q = queue.Queue()                             # ("text", piece) ... then ("end", stats) or ("error", message)
        self.stop = False                                  # the client went away: end at the next check
        self.out, self.shown, self.t0, self.t1 = [], "", time.perf_counter(), None
        self.n1 = 0                                        # tokens in the first commit (the live tok/s counts after it)
        self.why = "client left"                           # why stop was set: the client left, or a stop sequence


def text_of(content):
    """a message's text: a string, or the text of its parts (other parts as a short placeholder)"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for p in content:
        if not isinstance(p, dict):
            parts.append(str(p))
        elif p.get("type") in ("text", "input_text", "output_text"):
            parts.append(p.get("text", ""))
        elif p.get("type") == "tool_result":
            parts.append(text_of(p.get("content")))
        elif p.get("type") == "tool_use":
            parts.append(f"[tool call {p.get('name')}: {json.dumps(p.get('input'))}]")
        else:
            parts.append(f"[{p.get('type')} not supported]")
    return "\n".join(x for x in parts if x)


def last_user(messages):
    """the last user message's text (the loop's type rule reads it, as in run.py)"""
    for m in reversed(messages or []):
        if m.get("role") == "user":
            return text_of(m.get("content"))
    return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="chat", choices=("exact", "lossy", "chat"))
    ap.add_argument("--mem", default="32", choices=("45", "32", "24"))
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--max-tokens", type=int, default=65536,
                    help="the most any request may generate (default 65536; a request's own max_tokens can ask for less)")
    ap.add_argument("--drafter", choices=("auto", "dfl", "mtp"), help="dfl: every request takes the engine's code path (DFlash2), mtp: its chat path (the MTP head), auto: the engine's type rule (default; --kind code: dfl)")
    ap.add_argument("--kind", default="chat", choices=tuple(R.KIND_SET))
    ap.add_argument("--max-context", type=int, default=131072, help="the longest prompt accepted (tokens); longer ones get the API's 'prompt is too long' error")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--_screen", help=argparse.SUPPRESS)  # tests: a stand-in loop instead of the engine's
    a = ap.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"

    C, cenv, cpath = R.read_config(a.config, a.mem)
    store = os.path.join(ROOT, "engine", "models", "bare")
    if not os.path.isdir(os.path.join(store, "records3")):
        sys.exit("serve.py: engine/models/bare is not linked to a converted store (install.sh --store <dir>)")
    if R.drafter_cached() is None:
        sys.exit(f"serve.py: the block drafter's base {R.DRAFTER[0]} @ {R.DRAFTER[1][:7]} is not in the Hugging Face cache")
    snap = R.snapshot(cpath)                               # the config's decode loop (24 GB: w24), as in run.py
    screen = a._screen or os.path.join(ROOT, "engine", "loop", f"r57_screen_{snap}.py")
    out = os.path.join(ROOT, "out")
    os.makedirs(out, exist_ok=True)
    tag = time.strftime("serve_%m%d_%H%M%S")
    s = R.KIND_SET[a.kind]
    pre = s if s != "math" else "math500"
    keys = [f"{pre}/{tag}_{i:05d}" for i in range(1, NMAX + 1)]
    env = {"R57_PRESET": "45", "R57_PROMPTS": str(NMAX), "R57_REPS": "1", "R57_WARM": "0" if a.no_warmup else "1",
           "R57_TAG": tag, "R57_REFTOK": "", "R57_KEYS": ",".join(keys), "R57_QOM_SETS": "humaneval,mbpp",
           "HF_HUB_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    env.update(cenv)
    env["BIGMOE_EXT_BUILD"] = "build"
    env["BIGMOE_IDLE_MS"] = os.environ.get("BIGMOE_IDLE_MS", "50")  # idle: the ext's loader naps (200 us) after 50 ms without
                                                           # work instead of spinning a core; timing only (exact; Leap 11-11:
                                                           # idle CPU 100 -> 1.5 %). BIGMOE_IDLE_MS=0 in the shell keeps the spin.
    if a.mem in ("32", "24"):                              # PF-A, as chat.py: layer-major prefill of long prompts (agents send
        env["R57_PFLM"] = os.environ.get("R57_PFLM", "1")   # 10K+ tokens); the 45 GB path has its own prefill (PF-D)
    os.environ.update(env)
    cache_on = os.environ.get("CONV_CACHE", "1") != "0"   # BigMoE's conversation cache (engine/loop/convcache.py): a new
    os.environ.setdefault("CONV_CACHE_GB", {"24": "0.6", "32": "2.5", "45": "6"}[a.mem])   # turn prefills only its new tokens
    CCm = [None]                                           # the module once installed; False = off (not there / failed)

    term = os.fdopen(os.dup(1), "w", buffering=1)          # serve.py's own lines; the engine's prints go to the log

    def say(m):
        term.write(time.strftime("%H:%M:%S ") + m + "\n")

    say("Powered by MTPLX https://github.com/youssofal/mtplx")
    tok = R.tokenizer(store)
    jobs, cur, state = queue.PriorityQueue(), [None], {"ready": False, "i": 0, "agent_t": 0.0}
    lastby = {}                                            # lane -> the last finished request there (GET /stats, /statusline)
    jseq, RANK = itertools.count(), {"main": 0, "sub": 1, "side": 2}   # one engine queue: the main conversation first
    loglock = threading.Lock()
    LOG_HEADERS = ("user-agent", "x-app", "anthropic-beta", "anthropic-version", "originator")

    def G():
        return sys.modules["__main__"].__dict__

    def finish():
        j, cur[0] = cur[0], None
        if j is None:
            return
        g = G()
        r = g["RUNS"][-1] if g.get("RUNS") else {}
        if not j.out:                                      # a config without the session hook: the run's tokens at the end
            try:
                arm = next(iter(g["TOK"][r["key"]]))
                j.out = list(g["TOK"][r["key"]][arm][-1])
            except Exception:
                pass
        text = tok.decode(j.out, skip_special_tokens=True)
        if text[len(j.shown):]:
            j.q.put(("text", text[len(j.shown):]))
        n = r.get("n", len(j.out))
        eos = bool(j.out) and tok.convert_ids_to_tokens(int(j.out[-1])) in ("<|im_end|>", "<|endoftext|>")
        cc = CCm[0].last() if CCm[0] else {}
        stats = dict(n=n, prompt=len(j.ids), tps=r.get("tps"), pfs=r.get("pfs"), cache=cc,
                     peak=(r["mem"][6] if r.get("mem") and len(r["mem"]) > 6 else None),
                     reason="stop" if eos else "length", cut=j.stop)
        j.q.put(("end", stats))
        lane = j.meta.get("lane", "main")
        lastby[lane] = dict(n=n, prompt=len(j.ids), tps=stats["tps"], think=(j.t1 - j.t0) if j.t1 is not None else None,
                            cached=cc.get("c", 0) if cc.get("hit") else 0, lane=lane, t=time.time())
        say(f"done: {len(j.ids)} prompt tokens -> {n} tokens, whole decode {stats['tps'] or 0:.1f} tok/s, first token "
            f"{stats['pfs'] if stats['pfs'] is None else round(stats['pfs'], 2)} s, peak {stats['peak']} GB"
            + (f" ({j.why})" if j.stop else "")
            + (f"; cache {'hit' if cc.get('hit') else 'miss'} c {cc.get('c')} ({cc.get('src')}), prefilled {cc.get('prefilled')}, "
               f"restore {cc.get('restore_ms')} ms" + (f", check {cc.get('check')}" if cc.get('check') is not None else "")
               if cc else ""))

    drafter = a.drafter or ("dfl" if a.kind == "code" or cache_on else "auto")   # with the cache: DFlash2 throughout (an
                                                           # MTP-head turn can restore only from an MTP-head turn's entry)
    forced = {"dfl": "code", "mtp": "chat"}.get(drafter)

    def next_item():
        finish()
        if cache_on and CCm[0] is None:                    # once, before the first request's prefill
            try:
                import convcache
                CCm[0] = convcache if convcache.install(G()) else False
            except Exception as e:
                CCm[0] = False
                say(f"conversation cache off: {type(e).__name__}: {e}")
        g = G()
        g["HARD"] = 1e12                                   # no job budget in a server
        if forced and not getattr(next_item, "ruled", False):   # the loop's type rule: forced for serve.py's requests
            orig = g["rule"]
            g["rule"] = lambda item, _o=orig: forced if str(item.get("key", "")).startswith(f"{pre}/{tag}_") else _o(item)
            next_item.ruled = True
        if not state["ready"]:
            state["ready"] = True
            say(f"ready on http://127.0.0.1:{a.port} (config {a.config}-{a.mem}, C {C})")
        while True:
            _, _, j = jobs.get()
            if not j.stop:
                break
            j.q.put(("end", dict(n=0, prompt=len(j.ids), tps=None, pfs=None, peak=None, reason="stop", cut=True)))   # left while waiting
        g["CAP"] = j.cap
        state["i"] += 1
        if state["i"] > NMAX:
            raise SystemExit(0)
        cur[0] = j
        j.t0 = time.perf_counter()
        if CCm[0]:
            CCm[0].hint(sys_end=j.meta.get("sys_end", 0), lane=j.meta.get("lane", "main"), side=j.meta.get("side", False),
                        conv=j.meta.get("conv"))
        return [dict(key=keys[state["i"] - 1], set=s, prompt=j.user, user=j.user, prompt_ids=j.ids,
                     sys_end=j.meta.get("sys_end", 0), side=j.meta.get("side", False), agent=j.meta)]

    def live():
        """GET /stats (Leap 11-11, Kamen's status line): what the engine is doing now, from the job's own timestamps (stats
        only: nothing here touches the engine or the tokens). thinking = the prompt pass, before the first token."""
        j, now = cur[0], time.perf_counter()
        done = [r for k, r in list(lastby.items()) if k != "side"]   # the last answer of the conversation or a subagent;
        L = max(done, key=lambda r: r["t"]) if done else lastby.get("side", {})   # Claude Code's small background calls
        d = {"state": "idle" if state["ready"] else "loading", "queued": jobs.qsize(), "last": dict(L)}   # only before one
        if j is not None:
            cc = CCm[0].last() if CCm[0] else {}           # the cache's lookup for THIS prompt (its n = the prompt length)
            d.update(lane=j.meta.get("lane", "main"), prompt=len(j.ids),
                     cached=cc.get("c", 0) if cc.get("hit") and cc.get("n") == len(j.ids) else 0)
            if j.t1 is None:
                d.update(state="thinking", think_s=round(now - j.t0, 1))
            else:
                n_, dt = len(j.out), now - j.t1
                d.update(state="writing", think_s=round(j.t1 - j.t0, 1), out=n_,
                         tps=round((n_ - j.n1) / dt, 1) if n_ > j.n1 and dt > 0.3 else None)
        return d

    def status_line():
        """GET /statusline: one short line for Claude Code's status line (tools/statusline.sh prints it)"""
        d = live()
        mark = "◐◓◑◒"[int(time.time()) % 4]
        TAG = {"side": " · background", "sub": " · subagent"}   # Claude Code's small calls / a subagent's conversation
        tag = TAG.get(d.get("lane"), "")
        qn = f" · {d['queued']} queued" if d.get("queued") else ""
        if d["state"] == "loading":
            return f"⏳ loading the engine {mark}"
        if d["state"] == "thinking":
            c = d.get("cached") or 0
            return f"⏳ thinking {d['think_s']:.0f}s {mark} · prompt {d['prompt']:,} tok" + (f" ({c:,} cached)" if c else "") + tag + qn
        if d["state"] == "writing":
            head = f"⚡ {d['tps']:.1f} tok/s" if d.get("tps") else "⚡ writing"
            return f"{head} · {d['out']:,} tok · thought {d['think_s']:.1f}s" + tag + qn
        L = d["last"]
        if not L:
            return "✓ ready" + qn
        parts = [f"{L['n']:,} tok"] + ([f"{L['tps']:.1f} tok/s"] if L.get("tps") else []) + \
                ([f"thought {L['think']:.1f}s"] if L.get("think") is not None else [])
        return "✓ last: " + " · ".join(parts) + TAG.get(L.get("lane"), "") + qn

    def on_commit(new):
        j = cur[0]
        if j is None:
            return
        if j.t1 is None:
            j.t1, j.n1 = time.perf_counter(), len(new)
        j.out.extend(int(x) for x in new)
        text = tok.decode(j.out, skip_special_tokens=True).rstrip("�")
        if text.startswith(j.shown) and len(text) > len(j.shown):
            j.q.put(("text", text[len(j.shown):]))
            j.shown = text
        if j.stop:
            G()["CAP"] = len(j.out)

    def _items(mod):
        base = mod.load

        def load(nm):
            if sys._getframe(1).f_code.co_filename == screen:   # no wait on a research heat-pause file (as chat.py)
                G()["PAUSE"] = ""
            if nm == f"dec_{s}" and sys._getframe(1).f_code.co_filename == screen:
                return next_item()
            return base(nm)
        mod.load = load

    def _gated(mod):                                       # the engine's per-pass wait on that file, off too
        mod._PAUSE = ""

    def _drafter(mod):
        sd = mod.snapshot_download

        def snapshot_download(repo_id, *args, **kw):
            if repo_id == R.DRAFTER[0] and not kw.get("revision"):
                kw["revision"] = R.DRAFTER[1]
                kw.setdefault("local_files_only", True)
            return sd(repo_id, *args, **kw)
        mod.snapshot_download = snapshot_download

    def _session(mod):
        cls = mod.SessionVocab
        cm_ = cls.committed

        def committed(self, ids, *a_, **kw):
            r_ = cm_(self, ids, *a_, **kw)
            on_commit(ids)
            return r_
        cls.committed = committed
    patch = {"evalsuite.data": _items, "dflash_model_mlx": _drafter, "session_vocab": _session, "bigmoe.gated": _gated}

    class _Hook(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name not in patch:
                return None
            fn = patch.pop(name)
            sys.meta_path.remove(self)
            try:
                sp = importlib.util.find_spec(name)
            finally:
                if patch:
                    sys.meta_path.insert(0, self)
            if sp is None:
                return None
            orig = sp.loader.exec_module

            def exec_module(m):
                orig(m)
                fn(m)
            sp.loader.exec_module = exec_module
            return sp
    sys.meta_path.insert(0, _Hook())

    def prompt_ids(messages, system=None):
        msgs = [{"role": "system", "content": system}] if system else []
        for m in messages:
            role = m.get("role", "user")
            role = "system" if role == "developer" else ("user" if role == "tool" else role)
            msgs.append({"role": role, "content": text_of(m.get("content"))})
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False)
        return tok.encode(text, add_special_tokens=False)

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"                      # a streamed reply ends when the connection closes

        def log_message(self, fmt, *args):
            pass

        def _json(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def _sse_start(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

        def _comment(self):                                # an SSE comment line: keeps a waiting stream alive, and
            self.wfile.write(b": ping\n\n")                  # raises once the client has gone
            self.wfile.flush()

        def _sse(self, data, event=None):
            self.wfile.write(((f"event: {event}\n" if event else "") + f"data: {data}\n\n").encode())
            self.wfile.flush()

        def do_GET(self):
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                return self._json(200, {"object": "list", "data": [{"id": MODEL, "object": "model", "created": 0,
                                                                    "owned_by": "exactstream"}]})
            if self.path.rstrip("/") in ("", "/health"):
                return self._json(200, {"status": "ready" if state["ready"] else "loading"})
            if self.path.rstrip("/") == "/stats":
                return self._json(200, live())
            if self.path.rstrip("/") == "/statusline":
                b = status_line().encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
                return
            self._json(404, {"error": {"type": "not_found", "message": self.path}})

        def _log(self, rec):
            if os.environ.get("EXACTSTREAM_LOG_REQUESTS"):
                line = json.dumps(dict(rec, t=round(time.time(), 3), rid=getattr(self, "rid", None))) + "\n"
                with loglock, open(os.environ["EXACTSTREAM_LOG_REQUESTS"], "a") as lf_:
                    lf_.write(line)

        def _meta(self, body, tools, ids, se, max_tokens):
            m = AI.meta(body, self.headers, tools, ids, se, max_tokens, time.time() - state["agent_t"] < 1800)
            if m["lane"] == "main" and tools:
                state["agent_t"] = time.time()
            self._log({"kind": "render", "meta": m, "ids": ids})
            return m

        def _run(self, ids, max_tokens, user="", meta=None):
            j = Job(ids, max(1, min(int(max_tokens or a.max_tokens), a.max_tokens)), user, meta)
            jobs.put((RANK.get(j.meta.get("lane"), 0), next(jseq), j))
            return j

        def do_POST(self):
            try:
                body = self._body()
            except Exception as e:
                return self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
            p = self.path.split("?")[0].rstrip("/")
            self.rid = AI.new_id("req_")
            self._log({"kind": "request", "path": p, "body": body,
                       "headers": {k.lower(): v for k, v in self.headers.items() if k.lower() in LOG_HEADERS}})
            if p in ("/v1/chat/completions", "/chat/completions"):
                return self._openai(body)
            if p in ("/v1/messages", "/messages"):
                return self._anthropic(body)
            if p in ("/v1/responses", "/responses"):
                return self._responses(body)
            if p in ("/v1/messages/count_tokens", "/messages/count_tokens"):
                try:
                    msgs, tools = AI.from_anthropic(body)
                    return self._json(200, {"input_tokens": len(AI.render(tok, msgs, tools)[0])})
                except Exception as e:
                    return self._bad(e, "anthropic")
            self._json(404, {"error": {"type": "not_found", "message": self.path}})

        def _pieces(self, j, ping=None):                   # ping: called every 10 s while nothing arrives (a queued or
            while True:                                    # prefilling request); it raises when the client has gone
                try:
                    kind, v = j.q.get(timeout=10 if ping else None)
                except queue.Empty:
                    ping()
                    continue
                if kind == "text":
                    yield v
                else:
                    self.stats = v
                    return

        def _too_long(self, n, kind):
            msg = f"prompt is too long: {n} tokens > {a.max_context} maximum"
            if kind == "anthropic":
                return self._json(400, {"type": "error", "error": {"type": "invalid_request_error", "message": msg}})
            return self._json(400, {"error": {"message": msg, "type": "invalid_request_error", "code": "context_length_exceeded"}})

        def _bad(self, e, kind):
            msg = f"{type(e).__name__}: {e}"
            if kind == "anthropic":
                return self._json(400, {"type": "error", "error": {"type": "invalid_request_error", "message": msg}})
            return self._json(400, {"error": {"message": msg, "type": "invalid_request_error"}})

        def _events(self, j, tools, stops=None, ping=None):
            """the model's text as ("text", s) and ("tool", name, args) events, as they arrive; a stop sequence ends the
            answer (self.stop_hit)"""
            P, seen, S = AI.ToolParser(tools), [], AI.StopText(stops)
            self.stop_hit = None
            for piece in self._pieces(j, ping):
                s = S.feed(piece)
                if S.hit is not None and self.stop_hit is None:
                    self.stop_hit, j.why, j.stop = S.hit, "stop sequence", True   # the loop ends it at its next check
                for e in P.feed(s):
                    seen.append(e)
                    yield e
            for e in P.feed(S.finish()) + P.finish():
                seen.append(e)
                yield e
            self._log({"kind": "result", "events": seen, "out_ids": j.out, "stats": getattr(self, "stats", None)})

        def _drain(self, j):
            j.stop = True
            for _ in self._pieces(j):
                pass

        def _openai(self, body):                           # OpenAI chat completions, with or without tools
            try:
                msgs, tools = AI.from_chat(body)
                ids, se = AI.render(tok, msgs, tools)
            except Exception as e:
                return self._bad(e, "openai")
            if len(ids) >= a.max_context:
                return self._too_long(len(ids), "openai")
            mt = body.get("max_completion_tokens") or body.get("max_tokens")
            j = self._run(ids, mt, last_user(body.get("messages", [])), self._meta(body, tools, ids, se, mt))
            cid, created = AI.new_id("chatcmpl-"), int(time.time())

            def usage():
                s = self.stats
                return {"prompt_tokens": s["prompt"], "completion_tokens": s["n"], "total_tokens": s["prompt"] + s["n"]}
            if not body.get("stream"):
                text, calls = "", []
                for e in self._events(j, tools, body.get("stop")):
                    if e[0] == "text":
                        text += e[1]
                    else:
                        calls.append({"id": AI.new_id("call_"), "type": "function",
                                      "function": {"name": e[1], "arguments": json.dumps(e[2])}})
                msg = {"role": "assistant", "content": text or None}
                if calls:
                    msg["tool_calls"] = calls
                return self._json(200, {"id": cid, "object": "chat.completion", "created": created, "model": MODEL,
                                        "choices": [{"index": 0, "message": msg,
                                                     "finish_reason": "tool_calls" if calls else "stop" if self.stop_hit is not None else self.stats["reason"]}],
                                        "usage": usage()})
            self._sse_start()

            def chunk(delta, fin=None):
                return json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL,
                                   "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]})
            try:
                self._sse(chunk({"role": "assistant", "content": ""}))
                k = 0
                for e in self._events(j, tools, body.get("stop"), ping=self._comment):
                    if e[0] == "text":
                        self._sse(chunk({"content": e[1]}))
                    else:
                        self._sse(chunk({"tool_calls": [{"index": k, "id": AI.new_id("call_"), "type": "function",
                                                         "function": {"name": e[1], "arguments": json.dumps(e[2])}}]}))
                        k += 1
                self._sse(chunk({}, "tool_calls" if k else "stop" if self.stop_hit is not None else self.stats["reason"]))
                if (body.get("stream_options") or {}).get("include_usage"):
                    self._sse(json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL,
                                          "choices": [], "usage": usage()}))
                self._sse("[DONE]")
            except (BrokenPipeError, ConnectionResetError):
                self._drain(j)

        def _anthropic(self, body):                        # Anthropic Messages (Claude Code), with tools
            try:
                msgs, tools = AI.from_anthropic(body)
                ids, se = AI.render(tok, msgs, tools)
            except Exception as e:
                return self._bad(e, "anthropic")
            if len(ids) >= a.max_context:
                return self._too_long(len(ids), "anthropic")
            j = self._run(ids, body.get("max_tokens"), last_user(body.get("messages", [])),
                          self._meta(body, tools, ids, se, body.get("max_tokens")))
            mid, model = AI.new_id("msg_"), body.get("model", MODEL)
            reason = {"stop": "end_turn", "length": "max_tokens"}
            stops = body.get("stop_sequences")

            def why(n_tools):
                if n_tools:
                    return "tool_use"
                return "stop_sequence" if self.stop_hit is not None else reason[self.stats["reason"]]
            if not body.get("stream"):
                content = []
                for e in self._events(j, tools, stops):
                    if e[0] == "text":
                        if content and content[-1]["type"] == "text":
                            content[-1]["text"] += e[1]
                        else:
                            content.append({"type": "text", "text": e[1]})
                    else:
                        content.append({"type": "tool_use", "id": AI.new_id("toolu_"), "name": e[1], "input": e[2]})
                s = self.stats
                tool_used = any(c["type"] == "tool_use" for c in content)
                return self._json(200, {"id": mid, "type": "message", "role": "assistant", "model": model,
                                        "content": content or [{"type": "text", "text": ""}],
                                        "stop_reason": why(tool_used), "stop_sequence": self.stop_hit,
                                        "usage": {"input_tokens": s["prompt"], "output_tokens": s["n"]}})
            self._sse_start()

            def ev(o):
                self._sse(json.dumps(o), o["type"])
            try:
                ev({"type": "message_start", "message": {"id": mid, "type": "message", "role": "assistant", "model": model,
                                                         "content": [], "stop_reason": None, "stop_sequence": None,
                                                         "usage": {"input_tokens": len(ids), "output_tokens": 0}}})
                ev({"type": "ping"})
                idx, open_text, n_tools = 0, False, 0
                for e in self._events(j, tools, stops, ping=lambda: ev({"type": "ping"})):
                    if e[0] == "text":
                        if not open_text:
                            ev({"type": "content_block_start", "index": idx, "content_block": {"type": "text", "text": ""}})
                            open_text = True
                        ev({"type": "content_block_delta", "index": idx, "delta": {"type": "text_delta", "text": e[1]}})
                    else:
                        if open_text:
                            ev({"type": "content_block_stop", "index": idx})
                            idx, open_text = idx + 1, False
                        ev({"type": "content_block_start", "index": idx,
                            "content_block": {"type": "tool_use", "id": AI.new_id("toolu_"), "name": e[1], "input": {}}})
                        ev({"type": "content_block_delta", "index": idx,
                            "delta": {"type": "input_json_delta", "partial_json": json.dumps(e[2])}})
                        ev({"type": "content_block_stop", "index": idx})
                        idx, n_tools = idx + 1, n_tools + 1
                if open_text:
                    ev({"type": "content_block_stop", "index": idx})
                elif idx == 0:                             # an empty answer still carries one (empty) text block
                    ev({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
                    ev({"type": "content_block_stop", "index": 0})
                s = self.stats
                ev({"type": "message_delta", "delta": {"stop_reason": why(n_tools), "stop_sequence": self.stop_hit},
                    "usage": {"output_tokens": s["n"]}})
                ev({"type": "message_stop"})
            except (BrokenPipeError, ConnectionResetError):
                self._drain(j)

        def _responses(self, body):                        # OpenAI Responses (Codex), with function tools
            try:
                msgs, tools = AI.from_responses(body)
                ids, se = AI.render(tok, msgs, tools)
            except Exception as e:
                return self._bad(e, "openai")
            if len(ids) >= a.max_context:
                return self._too_long(len(ids), "openai")
            j = self._run(ids, body.get("max_output_tokens"), "", self._meta(body, tools, ids, se, body.get("max_output_tokens")))
            base = {"id": AI.new_id("resp_"), "object": "response", "created_at": int(time.time()),
                    "model": body.get("model", MODEL), "output": [], "parallel_tool_calls": False, "tool_choice": "auto",
                    "tools": []}
            items = []

            def usage():
                s = self.stats
                return {"input_tokens": s["prompt"], "output_tokens": s["n"], "total_tokens": s["prompt"] + s["n"],
                        "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}
            if not body.get("stream"):
                text = ""
                for e in self._events(j, tools):
                    if e[0] == "text":
                        text += e[1]
                    else:
                        items.append({"type": "function_call", "id": AI.new_id("fc_"), "call_id": AI.new_id("call_"),
                                      "name": e[1], "arguments": json.dumps(e[2]), "status": "completed"})
                AI.remember(base["id"], [i["call_id"] for i in items])
                if text.strip():
                    items.insert(0, {"type": "message", "id": AI.new_id("msg_"), "role": "assistant", "status": "completed",
                                     "content": [{"type": "output_text", "text": text, "annotations": []}]})
                return self._json(200, dict(base, status="completed", output=items, usage=usage()))
            self._sse_start()
            seq, cur = [0], {"msg": None, "text": ""}

            def ev(t, **kw):
                seq[0] += 1
                d = {"type": t, "sequence_number": seq[0]}
                d.update(kw)
                self._sse(json.dumps(d), t)

            def close_msg():
                if cur["msg"] is None:
                    return
                i, iid = cur["msg"]
                part = {"type": "output_text", "text": cur["text"], "annotations": []}
                ev("response.output_text.done", item_id=iid, output_index=i, content_index=0, text=cur["text"])
                ev("response.content_part.done", item_id=iid, output_index=i, content_index=0, part=part)
                item = {"type": "message", "id": iid, "role": "assistant", "status": "completed", "content": [part]}
                ev("response.output_item.done", output_index=i, item=item)
                items.append(item)
                cur["msg"], cur["text"] = None, ""
            try:
                ev("response.created", response=dict(base, status="in_progress"))
                ev("response.in_progress", response=dict(base, status="in_progress"))
                for e in self._events(j, tools, ping=self._comment):
                    if e[0] == "text":
                        if cur["msg"] is None:
                            i, iid = len(items), AI.new_id("msg_")
                            ev("response.output_item.added", output_index=i,
                               item={"type": "message", "id": iid, "role": "assistant", "status": "in_progress", "content": []})
                            ev("response.content_part.added", item_id=iid, output_index=i, content_index=0,
                               part={"type": "output_text", "text": "", "annotations": []})
                            cur["msg"] = (i, iid)
                        i, iid = cur["msg"]
                        cur["text"] += e[1]
                        ev("response.output_text.delta", item_id=iid, output_index=i, content_index=0, delta=e[1])
                    else:
                        close_msg()
                        i, fid, args = len(items), AI.new_id("fc_"), json.dumps(e[2])
                        item = {"type": "function_call", "id": fid, "call_id": AI.new_id("call_"), "name": e[1],
                                "arguments": "", "status": "in_progress"}
                        AI.remember(base["id"], [item["call_id"]])
                        ev("response.output_item.added", output_index=i, item=item)
                        ev("response.function_call_arguments.delta", item_id=fid, output_index=i, delta=args)
                        ev("response.function_call_arguments.done", item_id=fid, output_index=i, arguments=args)
                        done = dict(item, arguments=args, status="completed")
                        ev("response.output_item.done", output_index=i, item=done)
                        items.append(done)
                close_msg()
                ev("response.completed", response=dict(base, status="completed", output=items, usage=usage()))
            except (BrokenPipeError, ConnectionResetError):
                self._drain(j)

    def _stop(signum, frame):                              # SIGINT (Ctrl-C, or `kill -INT` on a server started from a
        raise KeyboardInterrupt                            # script, which starts it with SIGINT ignored) and SIGTERM (`kill`,
    signal.signal(signal.SIGINT, _stop)                    # launchd): the shutdown path below (srv.shutdown(), "stopped")
    signal.signal(signal.SIGTERM, _stop)

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    say(f"loading the engine (about a minute); requests wait until it is ready: http://127.0.0.1:{a.port}")

    log = os.path.join(out, f"{tag}.log")
    lf = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    sys.stdout.flush(); sys.stderr.flush()
    os.dup2(lf, 1)
    os.dup2(lf, 2)
    os.chdir(os.path.join(ROOT, "engine"))
    sys.argv = [screen, str(C), str(a.max_tokens)]
    sys.path.insert(0, os.path.dirname(screen))
    rc = 0
    try:
        runpy.run_path(screen, run_name="__main__")
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else 0
    except KeyboardInterrupt:
        rc = 130
    except BaseException as e:
        say(f"the engine stopped: {type(e).__name__}: {e} (log: {log})")
        if cur[0] is not None:
            cur[0].q.put(("end", dict(n=len(cur[0].out), prompt=len(cur[0].ids), tps=None, pfs=None, peak=None,
                                      reason="length", cut=True)))
        rc = 1
    finally:
        srv.shutdown()
        say(f"stopped; log: {log}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
