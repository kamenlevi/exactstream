#!/usr/bin/env python3
"""chat.py: chat with the 125B model in the terminal and watch its speed live.

  .venv/bin/python chat.py                            # the 32 GB chat config, up to 65,536 tokens per answer
  .venv/bin/python chat.py --kind code                # code requests: the DFlash2 drafter (--drafter dfl)
  .venv/bin/python chat.py --mem 45 --max-tokens 1024

The engine loads once (about a minute, with a short discarded warm-up) and stays loaded between messages. Every answer
runs through the same decode loop as run.py (engine/loop/r57_screen_<snapshot>.py), unchanged: chat.py hands the loop one
prompt item per message (the harness's chat template over the conversation so far, non-thinking) and reads the tokens each
check commits through the session-vocabulary hook the chat configs already run, to print them as they arrive, with live
tok/s in the window title. After each answer it prints the loop's own numbers: tokens, whole decode tok/s, first token, lifetime peak.
Ctrl-C stops the current answer (the process keeps running), /clear starts a new conversation, /exit (or Ctrl-D) quits,
/file PATH [message] sends the file in a fenced block and then the message. A built-in system line tells the model it
is a terminal chat without tools (--system adds to it).
A paste is one message, newlines included and of any length; Enter sends it. Backspace, Ctrl-U and Ctrl-W edit a message.
At --mem 32 and 24 long prompts prefill layer-major (PF-A: a faster first token, a little more memory; the same tokens
unless the prompt runs less than ~200 tokens past a multiple of 1,024); R57_PFLM=0 in the shell turns that off.
The engine's own log goes to out/<tag>.log; each answer's numbers, message and text go to out/<tag>.jsonl. Ctrl-Z is caught
(a suspended chat would keep its memory)."""
import argparse, codecs, importlib.abc, importlib.util, json, os, runpy, select, signal, sys, termios, time, unicodedata

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import run as R                                            # read_config, tokenizer, drafter_cached, DRAFTER, KIND_SET

NMAX = 999                                                 # messages per session (the loop's prompt list is fixed at start)
SYSTEM = ("You are chatting with the user in a terminal. You have no tools: you cannot run commands, browse, or open files. "
          "When the user shares a file, its full contents are in their message.")   # without it a long file drew agent-style
                                                           # replies with made-up tool calls (Tue 22:27)
FENCE_LANG = {".py": "python", ".js": "javascript", ".mjs": "javascript", ".ts": "typescript", ".tsx": "tsx", ".jsx": "jsx",
              ".html": "html", ".htm": "html", ".css": "css", ".json": "json", ".md": "markdown", ".sh": "bash",
              ".zsh": "bash", ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".hpp": "cpp", ".m": "objective-c",
              ".mm": "objective-cpp", ".swift": "swift", ".rs": "rust", ".go": "go", ".java": "java", ".kt": "kotlin",
              ".rb": "ruby", ".php": "php", ".lua": "lua", ".sql": "sql", ".yaml": "yaml", ".yml": "yaml",
              ".toml": "toml", ".xml": "xml", ".svg": "xml", ".tex": "latex", ".r": "r", ".jl": "julia"}


class UI:
    """the terminal (a copy of the original stdout; the engine's own prints go to the log). The live numbers go to the
    window title: a status line drawn at the bottom (a scroll region plus cursor save / restore) broke on lines wider than
    the terminal. The size comes from this copy: shutil.get_terminal_size() reads fd 1, which carries the engine's log."""

    def __init__(self):
        self.f = os.fdopen(os.dup(1), "w", buffering=1)
        self.tty = self.f.isatty()

    def size(self):
        try:
            return os.get_terminal_size(self.f.fileno())
        except OSError:
            return os.terminal_size((100, 30))

    def write(self, s):
        self.f.write(s)
        self.f.flush()

    def status(self, s):
        if not self.tty:
            return
        self.f.write(f"\033]0;{s}\007")                     # OSC 0: the window title
        self.f.flush()

    def close(self):
        if self.tty:
            self.f.write("\033]0;chat.py\007")
        self.f.write("\n")
        self.f.flush()


class State:
    def __init__(self):
        self.i = 0              # messages handed to the loop
        self.active = False     # an answer is being generated
        self.stop = False       # Ctrl-C during an answer
        self.pending = False    # an answer finished but is not reported yet
        self.ids, self.shown, self.t0, self.t1, self.last = [], "", 0.0, None, 0.0

    def begin(self):
        self.active, self.stop, self.pending = True, False, True
        self.ids, self.shown, self.t0, self.t1, self.last = [], "", time.perf_counter(), None, 0.0


class Keys:
    """One message from the terminal. The tty's own line mode holds at most 1,024 bytes per line (MAX_CANON) and cuts a
    longer paste, so chat.py reads keys in non-canonical mode and edits the message itself: Enter sends it; Backspace,
    Ctrl-U (erase the message) and Ctrl-W (erase a word) edit it; Ctrl-D on an empty message quits; Ctrl-C and Ctrl-Z stay
    signals. A paste is one message, newlines included (bracketed paste; on a terminal without it, lines that arrive
    together count as one paste), and Enter then sends it. Arrow keys and other escape sequences are ignored. The
    terminal's settings come back after every message, on Ctrl-C, on errors and on SIGTERM. Not a terminal (a pipe): one
    line per message, as before."""
    P0, P1 = "\x1b[200~", "\x1b[201~"                   # bracketed paste: the terminal wraps a paste in these

    def __init__(self, ui):
        self.ui, self.on, self.dirty = ui, ui.tty and os.isatty(0), False

    def read(self, prompt):
        """the next message (None at the end of input)"""
        if not self.on:
            self.ui.write(prompt)
            return sys.stdin.readline() or None
        old = termios.tcgetattr(0)
        raw = termios.tcgetattr(0)
        raw[3] &= ~(termios.ICANON | termios.ECHO | termios.IEXTEN)   # ISIG stays on: Ctrl-C / Ctrl-Z are signals
        raw[6][termios.VMIN], raw[6][termios.VTIME] = 1, 0
        term0 = signal.getsignal(signal.SIGTERM)

        def on_term(signum, frame):                        # SIGTERM while reading: leave through the finally below
            raise SystemExit(143)
        signal.signal(signal.SIGTERM, on_term)
        try:
            termios.tcsetattr(0, termios.TCSANOW, raw)
            self.ui.write("\033[?2004h" + prompt)          # bracketed paste on
            self.dirty = False
            return self._read(prompt)
        finally:
            termios.tcsetattr(0, termios.TCSANOW, old)
            self.ui.write("\033[?2004l")                   # bracketed paste off
            signal.signal(signal.SIGTERM, term0)

    def _read(self, prompt):
        W = max(20, self.ui.size().columns)
        dec = codecs.getincrementaldecoder("utf-8")("replace")
        buf, recs, out = [], [], []                        # characters; (column before, width, crossed a row) each; echo
        s = {"col": len(prompt)}

        def put(ch):                                       # add a character to the message and echo it
            c0, crossed = s["col"], ch == "\n"
            if ch == "\n":
                cells, shown, s["col"] = [], "\n", 0
            elif ch == "\t":
                cells, shown = [1, 1, 1, 1], "    "
            else:
                w = 0 if unicodedata.combining(ch) else 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
                cells, shown = [w], ch
            for w in cells:                                # the terminal wraps a character that does not fit the row
                if s["col"] + w > W:
                    s["col"], crossed = 0, True
                s["col"] += w
            buf.append(ch)
            recs.append((c0, sum(cells), crossed))
            out.append(shown)

        def redraw(lead):                                  # the message again from a fresh prompt
            text = "".join(buf)
            buf.clear(), recs.clear()
            out.append(lead + prompt)
            s["col"] = len(prompt)
            for ch in text:
                put(ch)

        def erase(n):                                      # remove the last n characters and their echo
            n = min(n, len(buf))
            if n <= 0:
                return
            col, simple = s["col"], True
            for c0, w, crossed in reversed(recs[-n:]):
                if crossed or col >= W or col != c0 + w:   # not on the cursor's row: redraw instead
                    simple = False
                    break
                col = c0
            k = s["col"] - col
            del buf[-n:], recs[-n:]
            if simple:
                out.append("\b" * k + " " * k + "\b" * k)
                s["col"] = col
            else:
                redraw("\n")

        esc, paste, prev = None, False, ""
        while True:
            data = os.read(0, 65536)
            if not data:                                   # the terminal went away
                return "".join(buf) or None
            if self.dirty:                                 # a Ctrl-Z note was printed: show the message again
                self.dirty = False
                redraw("")
            text = dec.decode(data)
            for i, ch in enumerate(text):
                if esc is not None:                        # an escape sequence: the paste markers count, the rest is ignored
                    esc += ch
                    if len(esc) == 2:
                        if ch not in "[O":
                            esc = None
                    elif esc[1] == "O" or "@" <= ch <= "~" or len(esc) > 16:
                        paste = True if esc == self.P0 else False if esc == self.P1 else paste
                        esc = None
                    continue
                if ch == "\x1b":
                    esc = ch
                elif ch in "\r\n":
                    if ch == "\n" and prev == "\r":         # CR LF: one newline
                        pass
                    elif paste or i + 1 < len(text) or select.select([0], [], [], 0.03)[0]:
                        put("\n")                         # in a paste, or more input right behind it: a newline
                    else:
                        self.ui.write("".join(out) + "\n")
                        return "".join(buf)
                elif ch in "\x7f\x08":                    # Backspace
                    erase(1)
                elif ch == "\x15":                         # Ctrl-U: erase the message
                    if buf:
                        out.append("\r\033[K" + prompt if not any(r[2] for r in recs) else "\n" + prompt)
                    buf.clear(), recs.clear()
                    s["col"] = len(prompt)
                elif ch == "\x17":                         # Ctrl-W: erase the word before the cursor
                    k = len(buf)
                    while k and buf[k - 1] in " \t\n":
                        k -= 1
                    while k and buf[k - 1] not in " \t\n":
                        k -= 1
                    erase(len(buf) - k)
                elif ch == "\x04":                         # Ctrl-D: quit on an empty message
                    if not buf:
                        self.ui.write("".join(out) + "\n")
                        return None
                elif ch == "\t" or ch >= " ":
                    put(ch)
                prev = ch
            self.ui.write("".join(out))
            out.clear()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="chat", choices=("exact", "lossy", "chat"))
    ap.add_argument("--mem", default="32", choices=("45", "32", "24"))
    ap.add_argument("--max-tokens", type=int, default=65536,
                    help="the most one answer may generate (default 65536; a long answer uses memory as a long prompt does)")
    ap.add_argument("--kind", default="chat", choices=tuple(R.KIND_SET),
                    help="which benchmark family's switches apply (code also implies --drafter dfl)")
    ap.add_argument("--drafter", choices=("auto", "dfl", "mtp"),
                    help="dfl: every message takes the engine's code path (the DFlash2 block drafter); mtp: its chat path (the "
                         "model's MTP head); auto: the engine's type rule picks per message (default; --kind code: dfl)")
    ap.add_argument("--system", help="more system text, after the built-in line (a terminal chat without tools; shared "
                    "files are in the message)")
    ap.add_argument("--no-warmup", action="store_true", help="skip the discarded warm-up (the first answer then runs cold)")
    ap.add_argument("--_screen", help=argparse.SUPPRESS)  # tests: a stand-in loop instead of the engine's
    a = ap.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"                     # before anything imports huggingface_hub (one process here)

    C, cenv, cpath = R.read_config(a.config, a.mem)
    store = os.path.join(ROOT, "engine", "models", "bare")
    if not os.path.isdir(os.path.join(store, "records3")):
        sys.exit("chat.py: engine/models/bare is not linked to a converted store (install.sh --store <dir>)")
    if R.drafter_cached() is None:
        sys.exit(f"chat.py: the block drafter's base {R.DRAFTER[0]} @ {R.DRAFTER[1][:7]} is not in the Hugging Face cache "
                 f"(see WEIGHTS.md)")
    snap = R.snapshot(cpath)                               # the config's decode loop (24 GB: w24), as in run.py
    screen = a._screen or os.path.join(ROOT, "engine", "loop", f"r57_screen_{snap}.py")
    out = os.path.join(ROOT, "out")
    os.makedirs(out, exist_ok=True)
    tag = time.strftime("chat_%m%d_%H%M%S")
    s = R.KIND_SET[a.kind]
    pre = s if s != "math" else "math500"
    keys = [f"{pre}/{tag}_{i:03d}" for i in range(1, NMAX + 1)]
    env = {"R57_PRESET": "45", "R57_PROMPTS": str(NMAX), "R57_REPS": "1", "R57_WARM": "0" if a.no_warmup else "1",
           "R57_TAG": tag, "R57_REFTOK": "", "R57_KEYS": ",".join(keys), "R57_QOM_SETS": "humaneval,mbpp",
           "HF_HUB_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    env.update(cenv)                                       # the config's own line wins, as in run.py
    env["BIGMOE_EXT_BUILD"] = "build"
    env["BIGMOE_IDLE_MS"] = os.environ.get("BIGMOE_IDLE_MS", "50")  # idle (a chat waiting for its next message): the ext's
                                                           # loader naps after 50 ms without work instead of spinning a core;
                                                           # timing only (exact). BIGMOE_IDLE_MS=0 in the shell keeps the spin.
    if a.mem in ("32", "24"):                              # PF-A (R57_PFLM=1): layer-major prefill on prompts over one chunk;
        env["R57_PFLM"] = os.environ.get("R57_PFLM", "1")  # bitwise with the default prefill unless the last 1,024-token chunk
    os.environ.update(env)                                 # is under ~200 tokens (Leap 11-11: 115 differs, 300 equal), a
                                                           # 2.4-3.8x faster first token on long prompts, +0.3-3.2 GB.
                                                           # R57_PFLM=0 in the shell turns it off.

    ui, st = UI(), State()
    kbd = Keys(ui)                                         # the message reader (keys: the loop's item keys)
    ui.write("Powered by MTPLX\nhttps://github.com/youssofal/mtplx\n")   # MTPLX's NOTICE (Apache-2.0 section 4(d))
    ui.write(f"ExactStream chat: config {a.config}-{a.mem} (C {C}), up to {a.max_tokens} tokens per answer. Loading the "
             f"engine (about a minute)...\n")
    tok = R.tokenizer(store)
    msgs = [{"role": "system", "content": SYSTEM + ("\n\n" + a.system if a.system else "")}]
    sysn = len(msgs)

    def G():                                              # the loop's globals (it runs as __main__ via runpy)
        return sys.modules["__main__"].__dict__

    def report():
        st.active = False
        if not st.pending:
            return
        st.pending = False
        g = G()
        r = g["RUNS"][-1] if g.get("RUNS") else None
        text = tok.decode(st.ids, skip_special_tokens=True)
        if text[len(st.shown):]:
            ui.write(text[len(st.shown):])
        msgs.append({"role": "assistant", "content": text.strip()})
        if r:
            peak = r["mem"][6] if r.get("mem") and len(r["mem"]) > 6 else float("nan")
            pfs = r.get("pfs")
            ui.write(f"\n\n[{r['n']} tokens | whole decode {r['tps']:.1f} tok/s | first token "
                     f"{pfs if pfs is None else round(pfs, 2)} s | lifetime peak {peak:.2f} GB | drafter {r.get('drafter')}, "
                     f"{r.get('kept') or 0:.2f} tokens per check"
                     + (" | stopped" if st.stop else "")
                     + (f" | {r['bad'].strip()}" if r.get("bad") and "0.0 s waited" not in r["bad"] else "") + "]\n")
            ui.status(f"last answer: {r['n']} tokens, {r['tps']:.1f} tok/s, first token "
                      f"{pfs if pfs is None else round(pfs, 2)} s, peak {peak:.1f} GB")
        try:                                               # one line per answer (the loop writes no run record in a chat):
            rec = {k: r.get(k) for k in ("key", "type", "drafter", "P", "n", "tps", "kept", "checks", "cyc", "ck", "drf",
                                          "miss", "io", "pfs", "bad", "cut")} if r else {}
            rec.update(time=time.strftime("%H:%M:%S"), peak=(r["mem"][6] if r and r.get("mem") and len(r["mem"]) > 6 else None),
                       stopped=st.stop, config=f"{a.config}-{a.mem}", user=msgs[-2]["content"] if len(msgs) > 1 else "",
                       answer=text)
            with open(os.path.join(out, f"{tag}.jsonl"), "a") as jf:
                jf.write(json.dumps(rec) + "\n")
        except Exception:
            pass

    drafter = a.drafter or ("dfl" if a.kind == "code" else "auto")
    forced = {"dfl": "code", "mtp": "chat"}.get(drafter)  # the loop's prompt type: "code" -> DFlash2, "chat" -> MTP head

    def next_item():
        report()
        g = G()
        g["HARD"] = 1e12                                   # the research runner's 600 s job budget: none in a chat
        g["CAP"] = a.max_tokens                            # (a Ctrl-C lowers it for one answer)
        if forced and not getattr(next_item, "ruled", False):   # the loop calls rule(item) right after it gets the item;
            orig = g["rule"]                               # chat.py's messages get the forced type, everything else the
            g["rule"] = lambda item, _o=orig: forced if str(item.get("key", "")).startswith(f"{pre}/{tag}_") else _o(item)
            next_item.ruled = True                         # engine's own rule (the loop file is unchanged)
        while True:
            ui.write("\n")
            try:
                line = kbd.read("> ")
            except KeyboardInterrupt:
                ui.write("\n(Ctrl-C: type /exit to quit)")
                continue
            if line is None:
                raise SystemExit(0)
            text = line.strip()
            if not text:
                continue
            if text in ("/exit", "/quit"):
                raise SystemExit(0)
            if text == "/clear":
                del msgs[sysn:]
                ui.write("(new conversation)")
                continue
            if text == "/file" or text.startswith("/file "):
                text = file_message(text[5:].strip())
                if text is None:
                    continue
            break
        msgs.append({"role": "user", "content": text})
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False)
        ids = tok.encode(prompt, add_special_tokens=False)
        st.i += 1
        if st.i > NMAX:
            ui.write("(the session's message limit is reached: restart chat.py)\n")
            raise SystemExit(0)
        ui.status(f"prefill: {len(ids)} prompt tokens")
        ui.write("\n")
        st.begin()
        return [dict(key=keys[st.i - 1], set=s, prompt=text, user=text, prompt_ids=ids)]

    def file_message(arg):
        """/file PATH [message]: the file in a fenced block (the language from its extension), then the message; a path
        with spaces goes in quotes; ~ expands. None (and a note) if the file cannot be read."""
        if arg[:1] in ("'", '"') and arg.find(arg[0], 1) > 0:
            j = arg.find(arg[0], 1)
            path, rest = arg[1:j], arg[j + 1:].strip()
        else:
            path, _, rest = arg.partition(" ")
            rest = rest.strip()
        path = os.path.expanduser(path)
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                body = fh.read()
        except OSError as e:
            ui.write(f"(/file: {e.strerror or e}: {path or 'no path given'}; usage: /file PATH [message])")
            return None
        lang = FENCE_LANG.get(os.path.splitext(path)[1].lower(), "")
        fence = "```"
        while fence in body:
            fence += "`"
        name = os.path.basename(path)
        ui.write(f"(/file: {name}, {body.count(chr(10)) + (not body.endswith(chr(10)) and bool(body))} lines, "
                 f"{len(tok.encode(body, add_special_tokens=False))} tokens)\n")
        return f"{name}:\n{fence}{lang}\n{body.rstrip(chr(10))}\n{fence}" + ("\n\n" + rest if rest else "")   # the request last

    def on_new_prompt(ids):
        if st.active:
            st.t0 = time.perf_counter()

    def on_commit(new):
        if not st.active:
            return
        now = time.perf_counter()
        if st.t1 is None:
            st.t1 = now
        st.ids.extend(int(x) for x in new)
        text = tok.decode(st.ids, skip_special_tokens=True).rstrip("�")
        if text.startswith(st.shown) and len(text) > len(st.shown):
            ui.write(text[len(st.shown):])
            st.shown = text
        n, d = len(st.ids), now - st.t1
        if now - st.last > 0.1 or n < 3:
            st.last = now
            head = f"{n} tokens | {(n - 1) / d:.1f} tok/s" if d > 0 and n > 1 else f"{n} tokens"
            ui.status(head + f" | first token {st.t1 - st.t0:.2f} s" + (" | stopping" if st.stop else " | Ctrl-C stops"))
        if st.stop:
            G()["CAP"] = n                                 # the loop ends this answer at its next check

    def _items(mod):                                       # the harness's item loader: the loop's main loop asks for each
        base = mod.load                                    # prompt item; chat.py answers with the next message

        def load(nm):
            caller = sys._getframe(1)
            if caller.f_code.co_filename == screen:        # before the warm-up's and every answer's run(): the loop waits
                G()["PAUSE"] = ""                          # on a research heat-pause file; a person's chat does not
            if nm == f"dec_{s}" and caller.f_code.co_filename == screen:
                return next_item()
            return base(nm)
        mod.load = load

    def _gated(mod):                                       # the engine's per-pass wait on that file, off too ("" never
        mod._PAUSE = ""                                    # exists); macOS and the fans protect the Mac itself

    def _drafter(mod):                                     # the drafter's loader gets DRAFTER's revision, as in run.py
        sd = mod.snapshot_download

        def snapshot_download(repo_id, *args, **kw):
            if repo_id == R.DRAFTER[0] and not kw.get("revision"):
                kw["revision"] = R.DRAFTER[1]
                kw.setdefault("local_files_only", True)
            return sd(repo_id, *args, **kw)
        mod.snapshot_download = snapshot_download

    def _session(mod):                                     # the per-check commits (observation only: the original runs)
        cls = mod.SessionVocab
        np_, cm_ = cls.new_prompt, cls.committed

        def new_prompt(self, ids, *a_, **kw):
            on_new_prompt(ids)
            return np_(self, ids, *a_, **kw)

        def committed(self, ids, *a_, **kw):
            r_ = cm_(self, ids, *a_, **kw)
            on_commit(ids)
            return r_
        cls.new_prompt, cls.committed = new_prompt, committed
    patch = {"evalsuite.data": _items, "dflash_model_mlx": _drafter, "session_vocab": _session, "bigmoe.gated": _gated}

    class _Hook(importlib.abc.MetaPathFinder):             # patches each module right after the loop imports it (run.py's)
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

    def on_sigint(signum, frame):
        if st.active:
            st.stop = True
            return
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, on_sigint)

    def on_sigtstp(signum, frame):                         # Ctrl-Z would leave the loaded engine (31+ GB) suspended in memory
        ui.write("\n(Ctrl-Z is off here: a suspended chat keeps its 31 GB; use Ctrl-C to stop an answer, /exit to quit)\n")
        kbd.dirty = True                                  # a message being typed is shown again at the next key
    signal.signal(signal.SIGTSTP, on_sigtstp)

    log = os.path.join(out, f"{tag}.log")
    lf = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    sys.stdout.flush(); sys.stderr.flush()
    os.dup2(lf, 1)                                         # the engine's own prints (Python and native) go to the log
    os.dup2(lf, 2)
    os.chdir(os.path.join(ROOT, "engine"))
    sys.argv = [screen, str(C), str(a.max_tokens)]
    sys.path.insert(0, os.path.dirname(screen))
    ui.status("loading the engine")
    rc = 0
    try:
        runpy.run_path(screen, run_name="__main__")
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else 0
    except KeyboardInterrupt:
        rc = 130
    except BaseException as e:                             # an engine error: say where the log is
        ui.write(f"\nchat.py: the engine stopped: {type(e).__name__}: {e}\n")
        rc = 1
    finally:
        try:
            report()
        except Exception:
            pass
        ui.close()
        ui.write(f"log: {log}\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
