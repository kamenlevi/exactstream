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
requests). The engine's own log goes to out/<tag>.log."""
import argparse, importlib.abc, importlib.util, json, os, queue, runpy, signal, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import run as R                                            # read_config, tokenizer, drafter_cached, DRAFTER, KIND_SET

NMAX = 99999                                               # requests per server run (the loop's prompt list is fixed at start)
MODEL = "exactstream"


class Job:
    def __init__(self, ids, cap, user=""):
        self.ids, self.cap, self.user = ids, cap, user   # user: the last user message (the loop's type rule reads it)
        self.q = queue.Queue()                             # ("text", piece) ... then ("end", stats) or ("error", message)
        self.stop = False                                  # the client went away: end at the next check
        self.out, self.shown, self.t0, self.t1 = [], "", time.perf_counter(), None


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
    os.environ.update(env)                                          # work instead of spinning a core; timing only (exact).
                                                                    # BIGMOE_IDLE_MS=0 in the shell keeps the spin.

    term = os.fdopen(os.dup(1), "w", buffering=1)          # serve.py's own lines; the engine's prints go to the log

    def say(m):
        term.write(time.strftime("%H:%M:%S ") + m + "\n")

    say("Powered by MTPLX https://github.com/youssofal/mtplx")
    tok = R.tokenizer(store)
    jobs, cur, state = queue.Queue(), [None], {"ready": False, "i": 0}

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
        stats = dict(n=n, prompt=len(j.ids), tps=r.get("tps"), pfs=r.get("pfs"),
                     peak=(r["mem"][6] if r.get("mem") and len(r["mem"]) > 6 else None),
                     reason="stop" if eos else "length", cut=j.stop)
        j.q.put(("end", stats))
        say(f"done: {len(j.ids)} prompt tokens -> {n} tokens, whole decode {stats['tps'] or 0:.1f} tok/s, first token "
            f"{stats['pfs'] if stats['pfs'] is None else round(stats['pfs'], 2)} s, peak {stats['peak']} GB"
            + (" (client left)" if j.stop else ""))

    drafter = a.drafter or ("dfl" if a.kind == "code" else "auto")
    forced = {"dfl": "code", "mtp": "chat"}.get(drafter)

    def next_item():
        finish()
        g = G()
        g["HARD"] = 1e12                                   # no job budget in a server
        if forced and not getattr(next_item, "ruled", False):   # the loop's type rule: forced for serve.py's requests
            orig = g["rule"]
            g["rule"] = lambda item, _o=orig: forced if str(item.get("key", "")).startswith(f"{pre}/{tag}_") else _o(item)
            next_item.ruled = True
        if not state["ready"]:
            state["ready"] = True
            say(f"ready on http://127.0.0.1:{a.port} (config {a.config}-{a.mem}, C {C})")
        j = jobs.get()
        g["CAP"] = j.cap
        state["i"] += 1
        if state["i"] > NMAX:
            raise SystemExit(0)
        cur[0] = j
        j.t0 = time.perf_counter()
        return [dict(key=keys[state["i"] - 1], set=s, prompt=j.user, user=j.user, prompt_ids=j.ids)]

    def on_commit(new):
        j = cur[0]
        if j is None:
            return
        if j.t1 is None:
            j.t1 = time.perf_counter()
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

        def _sse(self, data, event=None):
            self.wfile.write(((f"event: {event}\n" if event else "") + f"data: {data}\n\n").encode())
            self.wfile.flush()

        def do_GET(self):
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                return self._json(200, {"object": "list", "data": [{"id": MODEL, "object": "model", "created": 0,
                                                                    "owned_by": "exactstream"}]})
            if self.path.rstrip("/") in ("", "/health"):
                return self._json(200, {"status": "ready" if state["ready"] else "loading"})
            self._json(404, {"error": {"type": "not_found", "message": self.path}})

        def _run(self, ids, max_tokens, user=""):
            j = Job(ids, max(1, min(int(max_tokens or a.max_tokens), a.max_tokens)), user)
            jobs.put(j)
            return j

        def do_POST(self):
            try:
                body = self._body()
            except Exception as e:
                return self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
            p = self.path.split("?")[0].rstrip("/")
            if p in ("/v1/chat/completions", "/chat/completions"):
                return self._openai(body)
            if p in ("/v1/messages", "/messages"):
                return self._anthropic(body)
            if p in ("/v1/messages/count_tokens", "/messages/count_tokens"):
                ids = prompt_ids(body.get("messages", []), text_of(body.get("system")))
                return self._json(200, {"input_tokens": len(ids)})
            self._json(404, {"error": {"type": "not_found", "message": self.path}})

        def _pieces(self, j):
            while True:
                kind, v = j.q.get()
                if kind == "text":
                    yield v
                else:
                    self.stats = v
                    return

        def _openai(self, body):
            ids = prompt_ids(body.get("messages", []))
            j = self._run(ids, body.get("max_completion_tokens") or body.get("max_tokens"), last_user(body.get("messages", [])))
            cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())
            if not body.get("stream"):
                text = "".join(self._pieces(j))
                st_ = self.stats
                return self._json(200, {"id": cid, "object": "chat.completion", "created": created, "model": MODEL,
                                        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                                     "finish_reason": st_["reason"]}],
                                        "usage": {"prompt_tokens": st_["prompt"], "completion_tokens": st_["n"],
                                                  "total_tokens": st_["prompt"] + st_["n"]}})
            self._sse_start()

            def chunk(delta, fin=None):
                return json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL,
                                   "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]})
            try:
                self._sse(chunk({"role": "assistant", "content": ""}))
                for piece in self._pieces(j):
                    self._sse(chunk({"content": piece}))
                self._sse(chunk({}, self.stats["reason"]))
                if (body.get("stream_options") or {}).get("include_usage"):
                    st_ = self.stats
                    self._sse(json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL,
                                          "choices": [], "usage": {"prompt_tokens": st_["prompt"], "completion_tokens":
                                                                   st_["n"], "total_tokens": st_["prompt"] + st_["n"]}}))
                self._sse("[DONE]")
            except (BrokenPipeError, ConnectionResetError):
                j.stop = True
                for _ in self._pieces(j):
                    pass

        def _anthropic(self, body):
            ids = prompt_ids(body.get("messages", []), text_of(body.get("system")))
            j = self._run(ids, body.get("max_tokens"), last_user(body.get("messages", [])))
            mid = "msg_" + uuid.uuid4().hex[:24]
            reason = {"stop": "end_turn", "length": "max_tokens"}
            if not body.get("stream"):
                text = "".join(self._pieces(j))
                st_ = self.stats
                return self._json(200, {"id": mid, "type": "message", "role": "assistant", "model": body.get("model", MODEL),
                                        "content": [{"type": "text", "text": text}], "stop_reason": reason[st_["reason"]],
                                        "stop_sequence": None,
                                        "usage": {"input_tokens": st_["prompt"], "output_tokens": st_["n"]}})
            self._sse_start()
            try:
                self._sse(json.dumps({"type": "message_start", "message": {
                    "id": mid, "type": "message", "role": "assistant", "model": body.get("model", MODEL), "content": [],
                    "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": len(ids), "output_tokens": 0}}}),
                    "message_start")
                self._sse(json.dumps({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
                          "content_block_start")
                for piece in self._pieces(j):
                    self._sse(json.dumps({"type": "content_block_delta", "index": 0,
                                          "delta": {"type": "text_delta", "text": piece}}), "content_block_delta")
                st_ = self.stats
                self._sse(json.dumps({"type": "content_block_stop", "index": 0}), "content_block_stop")
                self._sse(json.dumps({"type": "message_delta", "delta": {"stop_reason": reason[st_["reason"]],
                                                                          "stop_sequence": None},
                                      "usage": {"output_tokens": st_["n"]}}), "message_delta")
                self._sse(json.dumps({"type": "message_stop"}), "message_stop")
            except (BrokenPipeError, ConnectionResetError):
                j.stop = True
                for _ in self._pieces(j):
                    pass

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
