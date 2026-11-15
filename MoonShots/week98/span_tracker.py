"""WEEK98 I (MoonShots, Sun 15 Nov 2026): the incremental comment / docstring span tracker for the lossy relaxation rule
(LEVERS_A_MOONSHOTS.md H / I, pre-registered before this file was written), and its test against Python's tokenize + ast.

SpanTracker.update(token_id) runs a small character state machine over the token's decoded text (a table built once per
tokenizer), so the engine calls it once per ACCEPTED token and keeps the state across checks; in_span() is the rule's
gate: the prefix ends inside a '#' comment or inside a docstring (after its third opening quote, before any closing quote).
Markdown ``` fences: prose outside a code block is never in span. A triple-quoted string is a docstring only if it opens at
the start of a line after a line ending in ':' (or as the block's first statement); other triple-quoted literals are strings.
usage: ~/BigMoE/.venv/bin/python -I span_tracker.py        (the test over every HE+ / MBPP answer in the BANK)"""
import ast, glob, io, json, os, time, tokenize
import numpy as np

H = os.path.expanduser


class SpanTracker:
    __slots__ = ("text", "region", "ls", "bt", "mode", "q", "esc", "doc", "qrun", "qc", "q_ls", "q_pc", "crun",
                 "prev_colon", "last_ns")

    def __init__(self, token_text):
        self.text = token_text                            # token id -> decoded text (a list or dict), built once
        self.reset()

    def reset(self):
        self.region, self.ls, self.bt = "P", True, 0      # P prose, F fence info line, C code
        self._code_reset()
        self.region = "P"

    def _code_reset(self):
        self.mode, self.q, self.esc, self.doc = "N", "", False, False   # N normal, K comment, S string, T triple string
        self.qrun, self.qc, self.q_ls, self.q_pc, self.crun = 0, "", False, False, 0
        self.prev_colon, self.last_ns, self.ls, self.bt = True, "", True, 0

    def in_span(self):
        return self.region == "C" and (self.mode == "K" or (self.mode == "T" and self.doc and self.crun == 0))

    def update(self, token_id):
        for ch in self.text[token_id]:
            self.step(ch)

    def _eol(self):                                       # a newline in code, normal mode
        if self.last_ns:
            self.prev_colon = self.last_ns == ":"
        self.ls, self.bt, self.last_ns = True, 0, ""

    def step(self, ch):
        if self.region == "P":
            if ch == "\n":
                self.ls, self.bt = True, 0
            elif self.ls and ch == "`":
                self.bt += 1
                if self.bt == 3:
                    self.region, self.bt = "F", 0
            elif not (self.ls and self.bt == 0 and ch in " \t"):
                self.ls, self.bt = False, 0
            return
        if self.region == "F":                            # the info string after an opening fence
            if ch == "\n":
                self.region = "C"; self._code_reset()
            return
        m = self.mode
        if m == "N":
            if self.qrun:
                if ch == self.qc:
                    self.qrun += 1
                    if self.qrun == 3:
                        self.mode, self.q, self.doc = "T", self.qc, self.q_ls and self.q_pc
                        self.qrun, self.crun, self.esc = 0, 0, False
                    return
                if self.qrun == 1:                        # one quote then another char: a single-quoted string
                    self.mode, self.q, self.qrun, self.esc = "S", self.qc, 0, False
                    self._s(ch)
                    return
                self.qrun, self.ls, self.last_ns = 0, False, self.qc   # "" : an empty string; go on in normal mode
            if ch == "\n":
                self._eol(); return
            if self.ls and ch == "`":
                self.bt += 1
                if self.bt == 3:
                    self.region, self.ls, self.bt = "P", False, 0       # closing fence
                return
            if ch in " \t\r":
                return
            if ch == "#":
                self.mode = "K"; return
            if ch in "\"'":
                self.qrun, self.qc, self.q_ls, self.q_pc = 1, ch, self.ls, self.prev_colon
                return
            self.ls, self.bt, self.last_ns = False, 0, ch
            return
        if m == "K":
            if ch == "\n":
                self.mode = "N"; self._eol()
            return
        if m == "S":
            self._s(ch); return
        if self.esc:                                      # m == "T"
            self.esc, self.crun = False, 0; return
        if ch == "\\":
            self.esc, self.crun = True, 0; return
        if ch == self.q:
            self.crun += 1
            if self.crun == 3:
                self.mode, self.crun, self.doc, self.ls, self.last_ns = "N", 0, False, False, self.q
            return
        self.crun = 0

    def _s(self, ch):
        if self.esc:
            self.esc = False
        elif ch == "\\":
            self.esc = True
        elif ch == self.q:
            self.mode, self.ls, self.last_ns = "N", False, self.q
        elif ch == "\n":                                  # an unterminated string ends at the line (Python would error)
            self.mode = "N"; self._eol()


def reference(text):
    """per char: 1 if a prefix ending at this char is inside a comment / docstring; plus a class map for the char
    (0 prose, 1 code, 2 ordinary string, 3 comment, 4 docstring, -1 excluded block); and the excluded block count"""
    n = len(text)
    inside = np.zeros(n, np.int8); cls = np.zeros(n, np.int8); excluded = 0; blocks = 0
    lines = text.split("\n"); pos = 0; starts = []
    for ln in lines:
        starts.append(pos); pos += len(ln) + 1
    i = 0
    while i < len(lines):
        if lines[i].lstrip(" \t").startswith("```") and lines[i].strip("` \t") != "" or lines[i].strip() == "```":
            j = i + 1
            while j < len(lines) and not lines[j].lstrip(" \t").startswith("```"):
                j += 1
            c0 = starts[i + 1] if i + 1 < len(lines) else n; c1 = starts[j] if j < len(lines) else n
            content = text[c0:c1]; blocks += 1
            ls = [0]
            for ln in content.split("\n"):
                ls.append(ls[-1] + len(ln) + 1)
            off = lambda r, c: c0 + ls[r - 1] + c
            try:
                toks = list(tokenize.generate_tokens(io.StringIO(content).readline))
                tree = ast.parse(content)
            except Exception:
                cls[c0:c1] = -1; excluded += 1; i = j + 1; continue
            cls[c0:c1] = 1
            docs = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
                    e = node.body[0]
                    if isinstance(e, ast.Expr) and isinstance(e.value, ast.Constant) and isinstance(e.value.value, str):
                        docs.add((e.lineno, e.col_offset))
            for tk in toks:
                if tk.type == tokenize.COMMENT:
                    a, b = off(*tk.start), off(*tk.end)
                    cls[a:b] = 3; inside[a:b] = 1
                elif tk.type == tokenize.STRING:
                    a, b = off(*tk.start), off(*tk.end)
                    s = text[a:b]; q0 = a + (len(s) - len(s.lstrip("rRbBuUfF")))
                    if (tk.start in docs) and text[q0:q0 + 3] in ('"""', "'''"):
                        cls[a:b] = 4; inside[q0 + 2:b - 3] = 1
                    else:
                        cls[a:b] = 2
            i = j + 1
        else:
            i += 1
    return inside, cls, excluded, blocks


def main():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(sorted(glob.glob(H("~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-2B-MLX-4bit/snapshots/*/tokenizer.json")))[0])
    split = json.load(open(H("~/BigMoE/data/deepdraft/bank/split.json")))
    cache = {}
    def ttext(i):
        if i not in cache:
            cache[i] = tok.decode([int(i)])
        return cache[i]
    class Lazy(dict):
        def __missing__(self, k):
            v = ttext(k); self[k] = v; return v
    table = Lazy()
    names = ["prose", "code", "ordinary string", "comment", "docstring"]
    tot = {k: [0, 0] for k in names}; excl_rows = 0; excl_blocks = 0; nblocks = 0; ntok = 0; tsec = 0.0
    tp = fn = fp = 0
    for st in ("heplus", "mbpp"):
        for nm in split[st]:
            z = np.load(H(f"~/datasets/deepdraft/bank/{nm}")); ids = [int(x) for x in np.asarray(z["ids"])]; gen = int(z["gen"])
            pieces = [ttext(i) for i in ids[gen:]]; text = "".join(pieces)
            inside, cls, ex, nb = reference(text); excl_blocks += ex; nblocks += nb
            tr = SpanTracker(table); off = 0
            for k, i in enumerate(ids[gen:]):
                if off > 0:
                    c = int(cls[off - 1]); ref = bool(inside[off - 1]); got = tr.in_span()
                    if c == -1:
                        excl_rows += 1
                    else:
                        tot[names[c]][0] += int(got); tot[names[c]][1] += 1
                        tp += int(got and ref); fn += int(ref and not got); fp += int(got and not ref)
                t0 = time.perf_counter(); tr.update(i); tsec += time.perf_counter() - t0; ntok += 1
                off += len(pieces[k])
    ref_in = tp + fn
    res = {"recall": round(100 * tp / max(ref_in, 1), 3), "precision": round(100 * tp / max(tp + fp, 1), 3),
           "ref_in_span_rows": ref_in, "false_in_span_rows": fp,
           "tracker_in_span_rate_by_reference_class_%": {k: round(100 * v[0] / max(v[1], 1), 3) for k, v in tot.items()},
           "rows_by_reference_class": {k: v[1] for k, v in tot.items()},
           "excluded_blocks": excl_blocks, "blocks": nblocks, "excluded_rows": excl_rows,
           "us_per_token": round(1e6 * tsec / max(ntok, 1), 2), "tokens": ntok}
    code_fp = tot["code"][0]
    res["false_in_span_of_code_rows_%"] = round(100 * code_fp / max(tot["code"][1], 1), 3)
    res["LINES"] = {"recall>=99": res["recall"] >= 99.0, "false_in_span_code<=0.1%": res["false_in_span_of_code_rows_%"] <= 0.1,
                    "speed<=10us": res["us_per_token"] <= 10.0}
    print(json.dumps(res, indent=1))
    json.dump(res, open(H("~/MoonShots/week98/span_tracker.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
