"""WEEK98 J(CC) (MoonShots, Sun 15 Nov 2026), as pre-registered in LEVERS_A_MOONSHOTS.md: close-call acceptance end to
end on Qwen3.5-2B (Sandbox98's target, mlx-community 4-bit) with s30 drafting (Sandbox98's native MTP head trained 30
deep), 32-row checks (31 drafts + the bonus). One arm per process: --delta 0 (exact) / 1 / 2 (nats).

Rule = the engine's relax_delta.relaxed_k (ccx45cc1): a draft is kept while it equals its row's argmax, or
logit(d) - max logit >= -delta (every row may relax); the first row not kept commits its argmax; after 31 kept drafts the
bonus row's argmax is committed.
Drafts are made lazily, one per committed token: a draft at depth k reads only committed tokens, so the committed text
equals block drafting's. The target runs one cached pass per committed token (batch 1), so delta 0 = the 2B's own greedy.
Panel: EvalSuite's code panel (tools/lossy_panel.json `code`, 264 keys), the items' `user` text in the 2B's chat template
(thinking off; Sandbox98's sb/gen.chat_ids), the items' max_tokens; an answer stops at an end token.
Per key: the answer ids and text, kept per check, every R1-only commit (position, depth, draft, argmax, dlm, class),
and the class counts of all committed tokens (code / comment / docstring / string / prose; the span tracker's state at
the token's first non-blank character, whitespace tokens where they start).
--selfcheck: self-check (b), the loop's drafts vs the teacher-forced chain (sb.core.chain on a full forward's W).
Read-only: Sandbox98 sb/, EvalSuite items and panel. Writes ~/MoonShots/week98/jcc/ (resumable: done keys are skipped).
usage: ~/BigMoE/.venv/bin/python -I -B jcc_decode.py --delta 1 [--keys 0:264] [--mem 3] [--tag smoke] [--selfcheck]"""
import argparse, json, os, sys, time
import numpy as np

H = os.path.expanduser
sys.path.insert(0, H("~/Sandbox98")); sys.path.insert(0, H("~/MoonShots/week98"))
from span_tracker import SpanTracker                      # noqa: E402

OUT = H("~/MoonShots/week98/jcc")
ITEMS = H("~/EvalSuite/data/items")
PANEL = H("~/EvalSuite/tools/lossy_panel.json")
NDRAFT = 31
SLOTS = SpanTracker.__slots__[1:]
CLASSES = ("code", "comment", "docstring", "string", "prose")


def panel_items():
    its = {}
    for f in ("dec_humaneval", "dec_humanevalx", "dec_mbpp"):
        for l in open(os.path.join(ITEMS, f + ".jsonl")):
            r = json.loads(l)
            its[r["key"]] = r
    return [its[k] for k in json.load(open(PANEL))["code"]]


def _clone(tr):
    p = SpanTracker(tr.text)
    for s in SLOTS:
        setattr(p, s, getattr(tr, s))
    return p


def _cls(p, ch):
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


def where(tr, text):
    """the class of a token about to be committed: the tracker's state at its first non-blank character"""
    first = next((i for i, ch in enumerate(text) if ch not in " \t\r\n"), None)
    if first is None:
        return _cls(tr, None)                             # a whitespace token: where it starts
    p = _clone(tr)
    for ch in text[:first]:
        p.step(ch)
    return _cls(p, text[first])


def decode_one(it, delta, D, model, tok, C, table, log_checks=False):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from sb.gen import chat_ids
    t = C.tm()
    prompt = chat_ids(tok, it["user"])
    cap, eos = int(it["max_tokens"]), set(tok.eos_token_ids)
    cache = make_prompt_cache(model)
    ids = list(prompt)
    post = t(mx.array(np.asarray(prompt, np.int32))[None], cache=cache)          # prefill: W rows 0..P-1
    lg = t.embed_tokens.as_linear(post[:, -1:])[0, 0].astype(mx.float32)
    a = int(mx.argmax(lg).item())
    Wl = [post[0]]
    tr = SpanTracker(table)
    cls_n = {c: 0 for c in CLASSES}
    ev, checks, chk_log = [], [], []
    cut = None

    def commit(x, relaxed=None):
        c = where(tr, table[x])
        cls_n[c] += 1
        if relaxed is not None:
            relaxed["cls"] = c
            ev.append(relaxed)
        tr.update(x)
        ids.append(x)

    commit(a)                                             # the first answer token: the prefill's argmax (no draft)
    t0 = time.perf_counter()
    while ids[-1] not in eos and len(ids) - len(prompt) < cap:
        S = len(ids)                                      # check start: ids[S-1] (the bonus) not yet through the target
        W = mx.concatenate(Wl + [mx.zeros_like(Wl[0][:1])], 0)[None]                # [1, S, d]; row S-1 unused
        ida = mx.array(np.asarray(ids, np.int32))
        h, K1, V1 = C.depth1(D, W, ida, S - 2)
        anc, ck, cv = mx.array([S - 2]), [], []
        drafts, k = [], 0
        while True:
            if k < NDRAFT:
                if k >= 1:
                    h = C.deeper(D, h, mx.array(np.asarray(ids, np.int32)), anc, k + 1, K1, V1, ck, cv)
                dtok = mx.argmax(C.lm_logits(D.norm(h))[0, 0])
            post = t(mx.array([[ids[-1]]], dtype=mx.int32), cache=cache)
            lg = t.embed_tokens.as_linear(post)[0, 0].astype(mx.float32)
            amx, mxl = mx.argmax(lg), lg.max()
            if k < NDRAFT:
                dl = lg[dtok] - mxl
                mx.eval(dtok, amx, dl, post, h)
                d, a, dlv = int(dtok.item()), int(amx.item()), float(dl.item())
                drafts.append(d)
            else:
                mx.eval(amx, post)
                d, a, dlv = None, int(amx.item()), None
            Wl.append(post[0])
            pos = len(ids) - len(prompt)
            if d is not None and (d == a or (delta > 0 and dlv >= -delta)):   # delta 0: exact (a tie is not kept)
                commit(d, None if d == a else dict(pos=pos, depth=k, draft=d, argmax=a, dlm=round(dlv, 4)))
                kept = True
            else:
                commit(a)
                kept = False
            done = ids[-1] in eos or len(ids) - len(prompt) >= cap
            if not kept or done:
                if log_checks:
                    chk_log.append((S - 2, drafts))
                if not kept:
                    checks.append(k + 1)                  # the check ended at a miss or at the bonus (k = 31)
                else:
                    cut = k + 1                           # cut by the stop (an end token kept as a draft, or the cap)
                break
            k += 1
    dt = time.perf_counter() - t0
    ans = ids[len(prompt):]
    stop = "eos" if ans and ans[-1] in eos else "cap"
    row = dict(key=it["key"], set=it["set"], delta=delta, n=len(ans), stop=stop, ids=ans,
               text=tok.decode([x for x in ans if x not in eos]), checks=checks, cut_check=cut, relaxed=ev,
               cls=cls_n, secs=round(dt, 2), prompt_len=len(prompt))
    return row, ids, chk_log


def selfcheck(D, C, ids, chk_log):
    """self-check (b): the loop's drafts vs the teacher-forced chain on the full forward's W of the same answer"""
    import mlx.core as mx
    _, post, _, _ = C.target_states(ids, taps=(), logits=False)
    need = {a: len(dr) for a, dr in chk_log}
    a0 = min(need)
    pred = {}

    def on_depth(j, h, anc):
        an = np.array(anc).tolist()
        rows = [i for i, x in enumerate(an) if need.get(x, 0) >= j]
        if not rows:
            return
        hh = h[:, mx.array(rows)]
        am = np.array(mx.argmax(C.lm_logits(D.norm(hh))[0], -1)).tolist()
        for i, v in zip(rows, am):
            pred[(an[i], j)] = int(v)
    C.chain(D, post[None], mx.array(np.asarray(ids, np.int32)), NDRAFT, on_depth, a0=a0)
    same = tot = 0
    for a, dr in chk_log:
        for j, d in enumerate(dr, start=1):
            if (a, j) in pred:
                tot += 1; same += int(pred[(a, j)] == d)
    return same, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, required=True)
    ap.add_argument("--keys", default="0:264")
    ap.add_argument("--mem", type=float, default=3.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    from sb import core as C
    import mlx.core as mx
    C.limit_memory(a.mem)
    model, tok = C.load_target()
    D = C.load_drafter("s30")

    class Lazy(dict):
        def __missing__(self, k):
            v = tok.decode([int(k)]); self[k] = v; return v
    table = Lazy()
    its = panel_items()
    lo, hi = (int(x) for x in a.keys.split(":"))
    os.makedirs(OUT, exist_ok=True)
    name = f"arm{a.delta:g}" + (f"_{a.tag}" if a.tag else "")
    path = os.path.join(OUT, name + ".jsonl")
    done = {json.loads(l)["key"] for l in open(path)} if os.path.exists(path) else set()
    sc_same = sc_tot = 0
    t_all, n_all = time.perf_counter(), 0
    for it in its[lo:hi]:
        if it["key"] in done:
            continue
        C.wait_pause()
        row, ids, chk = decode_one(it, a.delta, D, model, tok, C, table, log_checks=a.selfcheck)
        if a.selfcheck:
            s, n = selfcheck(D, C, ids, chk)
            row["selfcheck"] = [s, n]; sc_same += s; sc_tot += n
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")
        n_all += row["n"]
        kp = np.mean(row["checks"]) if row["checks"] else float("nan")
        print(f"[jcc d{a.delta:g}] {row['key']}: {row['n']} tok ({row['stop']}), {len(row['checks'])} checks, "
              f"kept {kp:.2f}, R1 {len(row['relaxed'])}, {row['n'] / max(row['secs'], 1e-9):.1f} tok/s"
              + (f", selfcheck {row['selfcheck'][0]}/{row['selfcheck'][1]}" if a.selfcheck else "")
              + f", peak {mx.get_peak_memory() / 2 ** 30:.2f} GB", flush=True)
        mx.clear_cache()
    if a.selfcheck:
        print(f"[jcc] selfcheck (b): {sc_same}/{sc_tot} = {100 * sc_same / max(sc_tot, 1):.2f} % of drafts equal the "
              f"teacher-forced chain's (line >= 99 %)", flush=True)
    print(f"[jcc d{a.delta:g}] done: {n_all} tokens in {time.perf_counter() - t_all:.0f} s; "
          f"peak {mx.get_peak_memory() / 2 ** 30:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
