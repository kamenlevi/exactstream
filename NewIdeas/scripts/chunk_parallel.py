"""Kamen's idea (any-order generation), chunk-level form, kill test (CPU, Kamen's own Claude Code sessions as the proxy for the
125B's agentic output, as Chat1000 used them): how much of the output could be written as PARALLEL streams after a skeleton?
Per assistant output unit (Bash command, Write content, Edit new_string, prose text), split into top-level blocks (code: blank-line
separated top-level blocks, def/class/function starts; prose: paragraphs / list items). Plan-then-fill: the skeleton = the first
line of each block (written first, sequentially); then blocks are filled in parallel on up to S_MAX streams (LPT schedule).
Critical path = skeleton + makespan; S_eff = total / critical path. Characters stand in for tokens (ratios only).
Pre-registered: corpus-wide S_eff (S_MAX 4) >= 2.0 -> strong (go to a quality test); <= 1.3 -> KILL for agentic output."""
import glob, json, os, re, heapq, collections
S_MAX = int(os.environ.get("S_MAX", "4"))
MINBLK = 40                                       # characters; tiny blocks merge into the previous one
files = [f for d in ("~/.claude/projects", "~/.claude-account-1/projects")
         for f in glob.glob(os.path.expanduser(d) + "/*/*.jsonl") if 300_000 < os.path.getsize(f) < 40_000_000]
START = re.compile(r"^(def |class |async def |function |[A-Za-z_][\w]*\s*\(\)\s*\{|#{1,3} |\d+\. |- |\* |@)")

def blocks(text, kind):
    lines = text.split("\n")
    out, cur = [], []
    for ln in lines:
        new = (ln.strip() == "" and cur) or (kind != "prose" and START.match(ln) and cur and not ln.startswith((" ", "\t")))
        if kind == "prose":
            new = (ln.strip() == "" and cur) or (START.match(ln.strip()) and cur)
        if new:
            out.append("\n".join(cur)); cur = []
        if ln.strip() or cur:
            cur.append(ln)
    if cur:
        out.append("\n".join(cur))
    merged = []
    for b in out:
        if merged and len(b) < MINBLK:
            merged[-1] += "\n" + b
        else:
            merged.append(b)
    return [b for b in merged if b.strip()]

def crit(bl):
    if len(bl) <= 1:
        return sum(len(b) for b in bl), sum(len(b) for b in bl)
    skel = sum(len(b.split("\n")[0]) for b in bl)
    rest = sorted((len(b) - len(b.split("\n")[0]) for b in bl), reverse=True)
    h = [0] * min(S_MAX, len(rest))
    for r in rest:
        heapq.heapreplace(h, h[0] + r) if h else None
    total = sum(len(b) for b in bl)
    return total, skel + max(h)

tot = collections.Counter(); cp = collections.Counter(); big = collections.Counter(); bigcp = collections.Counter()
nunits = collections.Counter()
for f in files:
    for line in open(f, errors="ignore"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        m = r.get("message") or {}
        if r.get("type") != "assistant" or not isinstance(m.get("content"), list):
            continue
        for c in m["content"]:
            if c.get("type") == "text":
                kind, text = "prose", c.get("text") or ""
            elif c.get("type") == "tool_use":
                inp = c.get("input") or {}
                name = c.get("name")
                text = inp.get("command") if name == "Bash" else inp.get("content") if name == "Write" else \
                    inp.get("new_string") if name == "Edit" else None
                if not isinstance(text, str):
                    continue
                kind = {"Bash": "bash", "Write": "write", "Edit": "edit"}[name]
            else:
                continue
            if not text.strip():
                continue
            t, c_ = crit(blocks(text, kind))
            tot[kind] += t; cp[kind] += c_; nunits[kind] += 1
            if t >= 2000:                          # long outputs: where the time goes
                big[kind] += t; bigcp[kind] += c_
T, C = sum(tot.values()), sum(cp.values())
print(f"S_MAX={S_MAX}, {len(files)} session files; units {dict(nunits)}")
for k in tot:
    print(f"  {k:6s} chars {tot[k]:>11,}  share {tot[k] / T:5.1%}  S_eff {tot[k] / cp[k]:.2f}  | outputs >= 2K chars: share of kind "
          f"{big[k] / max(tot[k], 1):5.1%}, S_eff {big[k] / max(bigcp[k], 1):.2f}")
print(f"  ALL    chars {T:>11,}  S_eff {T / C:.2f}  (Amdahl: time saved = 1 - 1/S_eff = {1 - C / T:.0%})")
