"""chat.py's message reader in a real pseudo-terminal, on the stand-in loop (no GPU). Each case types or pastes into the
pty and checks the message chat.py handed the loop (out/<tag>.jsonl 'user', in order). Also: the terminal is in
non-canonical mode while chat.py waits for a message and back in canonical mode with echo while it answers; Ctrl-C at the
prompt; SIGTERM while reading (the finally's bracketed-paste-off reaches the terminal); Ctrl-D quits.
Usage: pty_test.py <installed clone> <mock_screen.py>"""
import fcntl, glob, json, os, pty, re, signal, struct, sys, termios, threading, time

CL, MOCK = sys.argv[1], sys.argv[2]
LAYOUT_ONLY = len(sys.argv) > 3 and sys.argv[3] == "layout"
ROWS, COLS = 40, 120
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vt as VTM
OUT = os.path.join(CL, "out")
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail else ""), flush=True)


class Session:
    def __init__(self):
        before = set(glob.glob(os.path.join(OUT, "*.jsonl")))
        env = {k: v for k, v in os.environ.items() if k not in ("COLUMNS", "LINES")}   # the size must come from the tty
        env["MOCK_STORE"] = os.path.join(CL, "engine", "models", "bare")
        self.pid, self.fd = pty.fork()
        if self.pid == 0:
            os.chdir(CL)
            os.execve(os.path.join(CL, ".venv", "bin", "python"), ["python", "chat.py", "--_screen", MOCK], env)
        fcntl.ioctl(self.fd, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
        self.buf, self.lock, self.alive = b"", threading.Lock(), True
        threading.Thread(target=self._pump, daemon=True).start()
        self.before = before

    def _pump(self):
        while True:
            try:
                d = os.read(self.fd, 65536)
            except OSError:
                break
            if not d:
                break
            with self.lock:
                self.buf += d
        self.alive = False

    def expect(self, pat, t=60, since=0):
        end = time.time() + t
        while time.time() < end:
            with self.lock:
                i = self.buf.find(pat, since)
            if i >= 0:
                return i + len(pat)
            time.sleep(0.05)
        return -1

    def mark(self):
        with self.lock:
            return len(self.buf)

    def send(self, b, gap=0.0):
        if gap:
            for x in b:
                os.write(self.fd, bytes([x]))
                time.sleep(gap)
        else:
            os.write(self.fd, b)

    def prompt(self, since):                               # the next "> " after an answer (or the first one)
        return self.expect(b"\x1b[?2004h> ", 60, since)

    def lflag(self):
        return termios.tcgetattr(self.fd)[3]

    def users(self):
        new = sorted(set(glob.glob(os.path.join(OUT, "*.jsonl"))) - self.before)
        return [json.loads(l)["user"] for f in new for l in open(f)] if new else []

    def wait(self, t=30):
        end = time.time() + t
        while time.time() < end:
            p, st = os.waitpid(self.pid, os.WNOHANG)
            if p:
                return os.waitstatus_to_exitcode(st)
            time.sleep(0.1)
        return None


def render(s, label):
    with s.lock:
        out = s.buf.decode(errors="replace")
    v = VTM.VT(ROWS, COLS, "keep")
    v.feed(out)
    shown = "\n".join(v.text())
    games = ("\x1b7" in out) + ("\x1b8" in out) + len(re.findall(r"\x1b\[\d*;\d*r", out))
    titles = [x for x in v.titles if "tok/s" in x]
    check(f"{label}: no scroll region or cursor save / restore, live numbers in the window title, the answer and its "
          "numbers on screen", games == 0 and len(titles) > 0 and "tokens | whole decode" in shown and "word word" in shown,
          f"cursor / region sequences {games}, title updates {len(titles)}, last title {titles[-1][:60] if titles else None!r}")


if LAYOUT_ONLY:
    s = Session()
    m = s.prompt(0)
    s.send(b"a long one\r")
    s.expect(b"tokens | whole decode", 60, m)
    s.prompt(m)
    time.sleep(0.3)
    render(s, "2,000-character single-line answer")
    s.send(b"/exit\r")
    s.wait(30)
    print(f"PTY TEST: {sum(results)} of {len(results)} pass")
    sys.exit(0)

s = Session()
m = s.prompt(0)
check("starts and shows a prompt", m > 0)
s.send(b"hi\r")
a0 = s.expect(b"tokens | whole decode", 60, m)
with s.lock:
    seg = s.buf[m:a0]
check("the built-in system line reaches the prompt (the stand-in loop answers SYS-OK)", b"SYS-OK" in re.sub(rb"\x1b\][^\x07]*\x07", b"", seg))
s.prompt(a0)
time.sleep(0.3)
lf = s.lflag()
check("waiting for a message: non-canonical, no tty echo", not (lf & termios.ICANON) and not (lf & termios.ECHO), hex(lf))

LONG = " ".join(f"w{i:04d}" for i in range(600))[:2999] + "z"          # 3,000 characters (chat.py strips edge spaces)
cases = [
    ("3-line bracketed paste, then Enter", b"\x1b[200~first line\rsecond line\rthird line\x1b[201~", b"\r",
     "first line\nsecond line\nthird line"),
    ("3,000-character one-line bracketed paste, then Enter", b"\x1b[200~" + LONG.encode() + b"\x1b[201~", b"\r", LONG),
    ("3,000-character one-line paste without markers", LONG.encode() + b"\r", b"", LONG),
    ("3 lines arriving together without markers", b"alpha\rbeta\rgamma\r", b"", "alpha\nbeta\ngamma"),
    ("Backspace while typing", b"helo\x7flo there", b"\r", "hello there"),
    ("Ctrl-U erases the message", b"garbage\x15clean", b"\r", "clean"),
    ("Ctrl-W erases a word", b"one two\x17three", b"\r", "one three"),
    ("arrow keys are ignored", b"ab\x1b[Dc\x1b[A", b"\r", "abc"),
    ("UTF-8, wide characters and their Backspace", "héllo 世界!".encode() + b"\x7f\x7f" + "界".encode(), b"\r", "héllo 世界"),
    ("Backspace across a wrapped row (redraw)", b"x" * 150 + b"\x7f" * 60, b"\r", "x" * 90),
]
for name, body, enter, want in cases:
    m = s.mark()
    if not s.alive:
        with s.lock:
            print("CHILD GONE; output tail:", repr(s.buf[-1500:].decode(errors="replace")))
        break
    s.send(body)
    if enter:
        time.sleep(0.2)
        s.send(enter)
    got_answer = s.expect(b"tokens | whole decode", 60, m)   # one write (the answer's own text has title updates between its tokens)
    m2 = s.prompt(got_answer if got_answer > 0 else m)
    users = s.users()
    got = users[-1] if users else None
    check(name, got == want, f"{len(got or '')} chars" if got == want else f"got {got!r:.120} want {want!r:.120}")

# /file PATH message: the message, then the file in a fenced block
TF = os.path.expanduser(os.environ.get("PTY_TEST_FILE", ""))       # a real file to send with /file (not in git)
if TF and os.path.exists(TF):
    body = open(TF, encoding="utf-8", errors="replace").read()
    want = os.path.basename(TF) + ":\n```html\n" + body.rstrip("\n") + "\n```\n\nfix the bug please"     # the file first, the request last
    m = s.mark()
    s.send(b"/file " + os.environ["PTY_TEST_FILE"].encode() + b" fix the bug please")
    time.sleep(0.2)
    s.send(b"\r")
    note = s.expect(b"(/file: " + os.path.basename(TF).encode() + b", ", 30, m)
    got_answer = s.expect(b"tokens | whole decode", 60, m)
    s.prompt(got_answer if got_answer > 0 else m)
    got = s.users()[-1]
    with s.lock:
        nl = s.buf[note - 30:note + 30].decode(errors="replace") if note > 0 else ""
    check("/file with a message: the file in an html fence, then the message; a size note", got == want and note > 0,
          f"{len(got)} chars, {body.count(chr(10))} lines; note {nl[nl.find('(/file'):].split(')')[0] + ')'!r}")
    m = s.mark()
    s.send(b"/file /nonexistent/x.py\r")
    check("/file on a missing file: a note, nothing sent", s.expect(b"No such file or directory", 10, m) > 0 and s.prompt(m) > 0)

# canonical mode with echo while an answer streams (the mock answers ~4 s to a message with "long" in it)
m = s.mark()
s.send(b"a long one\r")
s.expect(b"word word", 30, m)
lf = s.lflag()
check("while answering: canonical mode and echo restored", bool(lf & termios.ICANON) and bool(lf & termios.ECHO), hex(lf))
s.prompt(m)
time.sleep(0.3)
render(s, "whole session incl. a 2,000-character single-line answer")

# Ctrl-C at the prompt
m = s.mark()
s.send(b"half typed\x03")
check("Ctrl-C at the prompt: the note, then a new prompt", s.expect(b"(Ctrl-C: type /exit to quit)", 10, m) > 0 and s.prompt(m) > 0)
time.sleep(0.3)
lf = s.lflag()
check("after Ctrl-C: reading again in non-canonical mode", not (lf & termios.ICANON), hex(lf))

# Ctrl-D on an empty message quits
s.send(b"\x04")
rc = s.wait(30)
check("Ctrl-D on an empty message quits, rc 0", rc == 0, f"rc {rc}")
with s.lock:
    tail = s.buf[-400:]
check("bracketed paste switched off at the end", b"\x1b[?2004l" in tail)

# SIGTERM while waiting for a message
s2 = Session()
m = s2.prompt(0)
time.sleep(0.5)
os.kill(s2.pid, signal.SIGTERM)
rc = s2.wait(30)
with s2.lock:
    after = s2.buf[m:]
check("SIGTERM while reading: exits through the finally (paste mode off, terminal restored)", b"\x1b[?2004l" in after, f"rc {rc}")
print(f"PTY TEST: {sum(results)} of {len(results)} pass")
