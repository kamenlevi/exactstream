"""CHAN: Lunatic's CODE || COMMENTARY quality pre-screen on the 2B (~/FableGod/lunatic/IDEAS_1117c_channels.md 3.5-3.6;
results/LEVERS_A.md section CHAN, pre-registered before data). Real decoding of the 2B (not LAB).

The loop is MoonShots' J(CC) exact loop (week98/jcc_decode.py, read-only reference): Qwen3.5-2B (Sandbox98's), batch 1,
one cached target pass per committed token, s30 drafts made lazily (31 drafts + the bonus = 32-row checks), a draft kept
while it equals the row's argmax. Panel: EvalSuite's code panel (264 keys), the items' user text in the 2B's chat template
(thinking off), the items' max_tokens; an answer stops at an end token.
  --mode plain   J(CC)'s exact arm itself (the identity check against MoonShots' arm0)
  --mode chan    CODE || COMMENTARY:
    trunk   the code channel: the prompt + the answer's code only. A SLOT is where the committed token would take the span
            tracker (MoonShots' week98/span_tracker.py, read-only) into a '#' comment or a docstring; the trunk then
            commits its best token that opens neither (the target's logits in order; a draft is kept only if it equals
            that token). After a slot, openers stay masked (no new slot) until the trunk commits a non-blank character.
    branch  one per slot: prompt + the trunk's tokens before the slot + the opener (no earlier comments), greedy with the
            same draft / check loop until the tracker leaves the comment / docstring (caps 128 / 512; a capped docstring
            is closed). Then a probe: the branch's own next line (plain greedy, <= 48 tokens), for comment consistency.
    output  'text' = the trunk alone (the code channel); 'spliced' = each branch put back at its slot (full-line comments
            and docstrings at the slot's line with its indentation, inline comments at the end of the slot's line).
Per key: ids, texts, checks (kept per check) + the cut check, the class of every check-ending commit, slots (position,
the trunk's check index, kind, branch checks, the comment text, the probe line, the trunk's next line, indents).
Writes ~/Sandbox98/results/chan/<name>.jsonl (resumable). Run with python -I -B (nothing is written under ~/MoonShots).
usage: ~/BigMoE/.venv/bin/python -I -B sb/chan.py --mode chan [--keys 0:264] [--sets humaneval,humanevalx,mbpp] [--tag T]"""
import argparse, json, os, sys, time
import numpy as np

H = os.path.expanduser
sys.path.insert(0, H("~/Sandbox98")); sys.path.insert(0, H("~/MoonShots/week98"))
from span_tracker import SpanTracker                      # noqa: E402  (read-only, MoonShots')
import mlx.core as mx                                     # noqa: E402

OUT = H("~/Sandbox98/results/chan")
ITEMS = H("~/EvalSuite/data/items")
PANEL = H("~/EvalSuite/tools/lossy_panel.json")
NDRAFT = 31
SLOTS = SpanTracker.__slots__[1:]
CAP = {"#": 128, "doc": 512}
PROBE_CAP = 48


def panel_items():
    its = {}
    for f in ("dec_humaneval", "dec_humanevalx", "dec_mbpp"):
        for l in open(os.path.join(ITEMS, f + ".jsonl")):
            r = json.loads(l)
            its[r["key"]] = r
    return [its[k] for k in json.load(open(PANEL))["code"]]


def clone(tr):
    p = SpanTracker(tr.text)
    for s in SLOTS:
        setattr(p, s, getattr(tr, s))
    return p


def cls_of(p, ch):                                        # MoonShots' jcc_decode._cls
    if p.region != "C":
        return "prose"
    if p.mode == "K":
        return "comment"
    if p.mode == "T":
        return "docstring" if p.doc else "string"
    if p.mode == "S" or p.qrun:
        return "string"
    if ch == "#":
        return "comment"
    if ch is not None and ch in "\"'":
        return "string"
    return "code"


def where(tr, text):                                      # MoonShots' jcc_decode.where
    first = next((i for i, ch in enumerate(text) if ch not in " \t\r\n"), None)
    if first is None:
        return cls_of(tr, None)
    p = clone(tr)
    for ch in text[:first]:
        p.step(ch)
    return cls_of(p, text[first])


def in_cd(p):
    """inside a '#' comment or a docstring, quotes included (the tracker's in_span() turns False at a docstring's
    first closing quote; this stays True until the third)"""
    return p.region == "C" and (p.mode == "K" or (p.mode == "T" and p.doc))


def opens(tr, text):
    """the token's characters take the tracker into a '#' comment or a docstring (at any character)"""
    if in_cd(tr):
        return False
    p = clone(tr)
    for ch in text:
        p.step(ch)
        if in_cd(p):
            return True
    return False


def kind_after(tr, text):
    """'doc' when the opener's characters leave the tracker in a docstring, else '#'"""
    p = clone(tr)
    for ch in text:
        p.step(ch)
    return "doc" if p.mode == "T" else "#"


def complete_line(s):
    """the first non-empty line of s that is followed by a newline, or None"""
    parts = s.split("\n")
    for ln in parts[:-1]:
        if ln.strip():
            return ln
    return None


def first_line(s):
    for ln in s.split("\n"):
        if ln.strip():
            return ln
    return ""


def indent_of(ln):
    return len(ln) - len(ln.lstrip(" \t"))


class Run:
    def __init__(self, model, tok, C, D, table, eos):
        self.model, self.tok, self.C, self.D, self.table, self.eos = model, tok, C, D, table, eos
        self.t = C.tm()

    def decode(self, ctx, tr, cap, mode, stop_in_span=False):
        """J(CC)'s exact loop from a prefill of ctx. mode 'plain' / 'trunk' (openers masked, slots recorded) / 'branch'
        (stop once the tracker leaves the comment / docstring). -> dict of the generated part"""
        from mlx_lm.models.cache import make_prompt_cache
        C, D, t, table = self.C, self.D, self.t, self.table
        cache = make_prompt_cache(self.model)
        ids = list(ctx)
        post = t(mx.array(np.asarray(ctx, np.int32))[None], cache=cache)
        lg = t.embed_tokens.as_linear(post[:, -1:])[0, 0].astype(mx.float32)
        Wl = [post[0]]
        gen, checks, cls_seq, end_cls, slots = [], [], [], [], []
        st = {"cut": None, "muted": False, "left": False}

        def choose(lg):
            a = int(mx.argmax(lg).item())
            if mode != "trunk" or not opens(tr, table[a]):
                return a, None
            order = np.array(mx.argsort(-lg)[:256]).tolist()
            for c in order:
                if not opens(tr, table[int(c)]):
                    return int(c), a
            return a, None

        def commit(x, masked):
            if masked is not None and not st["muted"]:
                slots.append(dict(pos=len(gen), ctx_len=len(ids), opener=masked, check=len(checks), tr=clone(tr)))
                st["muted"] = True
            c = where(tr, table[x])
            tr.update(x)
            if st["muted"] and any(ch not in " \t\r\n" for ch in table[x]):
                st["muted"] = False
            gen.append(x); ids.append(x); cls_seq.append(c)
            if stop_in_span and not in_cd(tr):
                st["left"] = True

        def done():
            return gen[-1] in self.eos or len(gen) >= cap or st["left"]

        commit(*choose(lg))
        while not done():
            S = len(ids)
            W = mx.concatenate(Wl + [mx.zeros_like(Wl[0][:1])], 0)[None]
            ida = mx.array(np.asarray(ids, np.int32))
            h, K1, V1 = C.depth1(D, W, ida, S - 2)
            anc, ck, cv = mx.array([S - 2]), [], []
            k = 0
            while True:
                if k < NDRAFT:
                    if k >= 1:
                        h = C.deeper(D, h, mx.array(np.asarray(ids, np.int32)), anc, k + 1, K1, V1, ck, cv)
                    dtok = mx.argmax(C.lm_logits(D.norm(h))[0, 0])
                post = t(mx.array([[ids[-1]]], dtype=mx.int32), cache=cache)
                lg = t.embed_tokens.as_linear(post)[0, 0].astype(mx.float32)
                if k < NDRAFT:
                    mx.eval(dtok, lg, post, h)
                    d = int(dtok.item())
                else:
                    mx.eval(lg, post)
                    d = None
                Wl.append(post[0])
                a, masked = choose(lg)
                kept = d is not None and d == a
                commit(a, masked)
                if not kept or done():
                    if not kept:
                        checks.append(k + 1); end_cls.append(cls_seq[-1] if k < NDRAFT else "bonus")
                    else:
                        st["cut"] = k + 1
                    break
                k += 1
        return dict(gen=gen, checks=checks, cut=st["cut"], cls=cls_seq, end_cls=end_cls, slots=slots, cache=cache, ids=ids)

    def probe(self, out, after):
        """plain greedy after a branch's end until the text after the comment (after + new) holds a complete non-empty
        line (<= PROBE_CAP tokens) -> the new text"""
        t, table = self.t, self.table
        ids, cache = out["ids"], out["cache"]
        txt = ""
        for _ in range(PROBE_CAP):
            if complete_line(after + txt) is not None:
                break
            post = t(mx.array([[ids[-1]]], dtype=mx.int32), cache=cache)
            a = int(mx.argmax(t.embed_tokens.as_linear(post)[0, 0]).item())
            if a in self.eos:
                break
            ids.append(a); txt += table[a]
        return txt


def comment_span(tr0, text):
    """step a clone of the slot's tracker over the branch text (opener + continuation): -> (start, end, kind, closed);
    start = the '#' or the docstring's first quote, end exclusive ('#': at its newline; docstring: after the closing
    quotes); kind '#' / 'doc'; closed False when the text ran out inside the span (a cap)"""
    p = clone(tr0)
    start = kind = None
    for i, ch in enumerate(text):
        before = in_cd(p)
        p.step(ch)
        now = in_cd(p)
        if start is None and now and not before:
            if p.mode == "K":
                start, kind = i, "#"
            else:
                start, kind = max(0, i - 2), "doc"
        elif start is not None and before and not now:
            return start, (i if kind == "#" else i + 1), kind, True
    return start, len(text), kind, False


def splice(T, items):
    """T: the trunk text; items: (P, pre, body) with body = the comment / docstring text; insert from the last one"""
    out = T
    for P, pre, body in sorted(items, key=lambda x: x[0], reverse=True):
        line_pre = out[:P].split("\n")[-1]
        if pre.strip():                                   # code before the '#' in the opener: the trunk wrote its own code; inline
            E = out.find("\n", P)
            E = len(out) if E < 0 else E
            out = out[:E] + "  " + body + out[E:]
        elif "\n" in pre:                                 # the opener carried a line break: a new line at P
            ins = pre + body
            if not out[P:].startswith("\n"):
                ins += "\n" + pre.split("\n")[-1]
            out = out[:P] + ins + out[P:]
        elif line_pre.strip() == "":                      # a full-line comment / docstring at the slot's line
            out = out[:P] + pre + body + "\n" + line_pre + out[P:]
        else:                                             # an inline comment: at the end of the slot's line
            E = out.find("\n", P)
            E = len(out) if E < 0 else E
            sp = pre if pre.strip() == "" and pre else "  "
            out = out[:E] + sp + body + out[E:]
    return out


def run_key(R, it, mode, chat_ids):
    tok, table = R.tok, R.table
    prompt = chat_ids(tok, it["user"])
    cap = int(it["max_tokens"])
    t0 = time.perf_counter()
    tr = SpanTracker(table)
    out = R.decode(prompt, tr, cap, "trunk" if mode == "chan" else "plain")
    ans = out["gen"]
    stop = "eos" if ans and ans[-1] in R.eos else "cap"
    text = tok.decode([x for x in ans if x not in R.eos])
    row = dict(key=it["key"], set=it["set"], mode=mode, n=len(ans), stop=stop, ids=ans, text=text, checks=out["checks"],
               cut_check=out["cut"], end_cls=out["end_cls"], cls={c: out["cls"].count(c) for c in set(out["cls"])},
               prompt_len=len(prompt))
    if mode == "chan":
        tt = [table[x] for x in ans if x not in R.eos]
        T = "".join(tt)
        offs = np.cumsum([0] + [len(s) for s in tt]).tolist()
        sl, items = [], []
        trunk_ids = out["ids"]
        for s in out["slots"]:
            k_op = s["opener"]
            kind0 = kind_after(s["tr"], table[k_op])
            trb = clone(s["tr"])
            trb.update(k_op)
            ctx = trunk_ids[:s["ctx_len"]] + [k_op]
            b = R.decode(ctx, trb, CAP[kind0], "branch", stop_in_span=True)
            btxt = table[k_op] + "".join(table[x] for x in b["gen"] if x not in R.eos)
            start, end, kind, closed = comment_span(s["tr"], btxt)
            if start is None:                             # should not happen (the opener entered the span)
                sl.append(dict(pos=s["pos"], check=s["check"], error="no span")); continue
            body = btxt[start:end]
            if kind == "doc" and not closed:
                q = body.lstrip("rRbBuUfF")[:3]
                body = body + ("\n" if not body.endswith("\n") else "") + (q if q in ('"""', "'''") else '"""')
            P = offs[min(s["pos"], len(offs) - 1)]
            pre = btxt[:start]
            items.append((P, pre, body))
            after = btxt[end:]
            if closed and b["gen"] and b["gen"][-1] not in R.eos:
                after += R.probe(b, after)
            probe_line = complete_line(after) or first_line(after)
            trunk_next = complete_line(T[P:]) or first_line(T[P:])
            line_pre = T[:P].split("\n")[-1]
            full_line = not pre.strip() and (line_pre.strip() == "" or "\n" in pre)
            sl.append(dict(pos=s["pos"], check=s["check"], kind=kind, closed=closed, branch_n=len(b["gen"]),
                           branch_checks=b["checks"], branch_cut=b["cut"], body=body[:400], pre=pre,
                           probe=probe_line.strip()[:200], trunk_next=trunk_next.strip()[:200],
                           agree=probe_line.strip() == trunk_next.strip(), full_line=full_line,
                           indent_trunk=indent_of(first_line(T[T.rfind("\n", 0, P) + 1:])), indent_probe=indent_of(probe_line),
                           branch_end_cls=b["end_cls"]))
            del b
            mx.clear_cache()
        row["slots"] = sl
        row["spliced"] = splice(T, items)
        row["trunk_text_join"] = T
    row["secs"] = round(time.perf_counter() - t0, 2)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("plain", "chan"), required=True)
    ap.add_argument("--keys", default="0:264"); ap.add_argument("--sets", default="humaneval,humanevalx,mbpp")
    ap.add_argument("--mem", type=float, default=3.0); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    from sb import core as C
    from sb.gen import chat_ids
    C.limit_memory(a.mem)
    model, tok = C.load_target()
    D = C.load_drafter("s30")

    class Lazy(dict):
        def __missing__(self, k):
            v = tok.decode([int(k)]); self[k] = v; return v
    R = Run(model, tok, C, D, Lazy(), set(tok.eos_token_ids))
    its = panel_items()
    lo, hi = (int(x) for x in a.keys.split(":"))
    sets = set(a.sets.split(","))
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, a.mode + (f"_{a.tag}" if a.tag else "") + ".jsonl")
    done = {json.loads(l)["key"] for l in open(path)} if os.path.exists(path) else set()
    t_all, n_all = time.perf_counter(), 0
    for it in its[lo:hi]:
        if it["key"] in done or it["set"] not in sets:
            continue
        C.wait_pause()
        row = run_key(R, it, a.mode, chat_ids)
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")
        n_all += row["n"] + sum(s.get("branch_n", 0) for s in row.get("slots", []))
        print(f"[chan {a.mode}] {row['key']}: trunk {row['n']} tok ({row['stop']}), {len(row['checks'])} checks"
              + (f", {len(row['slots'])} slots, branches {sum(s.get('branch_n', 0) for s in row['slots'])} tok" if a.mode == "chan" else "")
              + f", {row['secs']} s, peak {mx.get_peak_memory() / 2 ** 30:.2f} GB", flush=True)
        mx.clear_cache()
    print(f"[chan {a.mode}] DONE: {n_all} tokens in {time.perf_counter() - t_all:.0f} s; peak {mx.get_peak_memory() / 2 ** 30:.2f} GB",
          flush=True)


if __name__ == "__main__":
    main()
