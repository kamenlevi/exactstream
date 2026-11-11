"""R1 CENSUS (MoonShots, Wed 2026-11-04; the orchestrator 21:4x, Fable FULL's round 2 R2.2 (a), ~/FableGod/full/FULL.md
"## Round 2", lines pre-registered there): can one answer be written as parts in one pass? How many recorded code answers
hold >= 2 independent top-level parts, and how much of the answer is inside them?
Sources: the HPD-0 walk's 24 HumanEval+ windows and 100 MBPP windows (~/TrainLab/data/d02b_walk_hpd0_{code,mbpp}.npz; the
answer = the assistant turn of the window, decoded with the 125B's byte-level vocab; windows are cut at ~458 tokens, so a
long answer is seen as its prefix), the Magicoder OSS-Instruct items with lang 'python' (~/TrainLab/ext/magicoder_oss75k/,
the 'solution' text), and the d16w agent windows (~/TrainLab/data/d16w/agent_*.npz).
Method (FULL R2.2 (a)): the answer's ```python blocks (an unclosed block runs to the end); ast.parse, dropping up to 40
trailing lines when the text is cut; parts = top-level FunctionDef / AsyncFunctionDef / ClassDef (methods belong to their
class); a part's parallel text = its body after the signature and the docstring (for a class: its methods' bodies + other
class statements); edges = Name / Attribute references in one part to another part's name (sensitivity: also a class's
non-dunder method names); independent = no edge between members; the best set = the maximum-weight independent set by
parallel text. Imports, signatures, docstrings, prose and driver code stay sequential. Tokens: characters (FULL's chars / 4
stand-in; shares are ratios). An answer whose code does not parse has no parts.
Lines (FULL R2.2 (a)): PASS (R1 goes to (b)) if on the target workload >= 30 % of answers have >= 2 independent parts
holding >= 50 % of their tokens; KILL if < 15 % do. FULL's prediction: HumanEval+ 5-10 %, MBPP <= 10 %, Magicoder 40-60 %
(50-70 % of tokens in parts), agent write / edit 20-40 %.
Also reported: the concurrent share = (best set's text - its largest part) / the answer (the text that can run beside
another stream; the time saving's upper bound at k streams).
usage: [BigMoE venv] python r1_census.py -> stdout + data/r1_census.txt"""
import ast, itertools, json, os, re, time
import numpy as np

H = os.path.expanduser("~")
TD = f"{H}/TrainLab/data"
MAGIC = f"{H}/TrainLab/ext/magicoder_oss75k/data-oss_instruct-decontaminated.jsonl"
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


vocab = json.load(open(f"{H}/BigMoE/models/bare/vocab.json"))
ID2 = {v: k for k, v in vocab.items()}


def _b2u():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


BD = _b2u()


def dec(ids):
    out = bytearray()
    for t in ids:
        s = ID2.get(int(t))
        out += f"<|{int(t)}|>".encode() if s is None else bytes(BD[c] for c in s)
    return out.decode("utf-8", errors="replace")


FENCE = re.compile(r"```(?:python|py|Python)?[ \t]*\n(.*?)(?:\n```|$)", re.S)


def py_blocks(ans):
    """the answer's python code: ```python / ```py / bare ``` blocks (other languages skipped); an unclosed block runs on"""
    out = []
    for m in re.finditer(r"```([A-Za-z0-9_+-]*)[ \t]*\n", ans):
        lang = m.group(1).lower()
        if lang not in ("", "python", "py", "python3"):
            continue
        end = ans.find("\n```", m.end())
        body = ans[m.end():] if end < 0 else ans[m.end():end]
        if lang == "" and not re.search(r"^\s*(def|class|import|from)\s", body, re.M):
            continue
        out.append(body)
    return out


def parse_tolerant(code, max_drop=40):
    ls = code.split("\n")
    for cut in range(len(ls), max(len(ls) - max_drop, 0), -1):
        try:
            return ast.parse("\n".join(ls[:cut])), "\n".join(ls[:cut]), len(ls) - cut
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            continue
    return None, "", len(ls)


def line_chars(ls, a, b):
    """chars of 1-based lines a..b inclusive (+1 per newline)"""
    return sum(len(ls[i]) + 1 for i in range(a - 1, min(b, len(ls))))


def fn_body_chars(node, ls):
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    if not body:
        return 0
    return line_chars(ls, body[0].lineno, node.end_lineno)


def part_chars(node, ls):
    if isinstance(node, ast.ClassDef):
        tot, body = 0, node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                and isinstance(body[0].value.value, str):
            body = body[1:]
        for st in body:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                tot += fn_body_chars(st, ls)
            else:
                tot += line_chars(ls, st.lineno, st.end_lineno)
        return tot
    return fn_body_chars(node, ls)


def refs(node):
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
    return out


def best_set(w, adj):
    """maximum-weight independent set (exact for <= 12 parts, greedy beyond)"""
    n = len(w)
    if n <= 12:
        best, bw = (), 0.0
        for r in range(1, n + 1):
            for S in itertools.combinations(range(n), r):
                if any(b in adj[a] for a, b in itertools.combinations(S, 2)):
                    continue
                s = sum(w[i] for i in S)
                if s > bw:
                    best, bw = S, s
        return best
    S = []
    for i in sorted(range(n), key=lambda i: -w[i]):
        if all(i not in adj[j] for j in S):
            S.append(i)
    return tuple(S)


def census(ans):
    """-> dict(parsed, n_parts, best parts, share, conc, share_m (method-name edges), cut)"""
    blocks = py_blocks(ans)
    total = max(len(ans), 1)
    if not blocks:
        return dict(code=False, parsed=False, n=0, k=0, share=0.0, conc=0.0, k_m=0, share_m=0.0, k_b=0, share_b=0.0,
                    share_c=0.0, share_bc=0.0, cut=0, names=[], code_frac=0.0, conc_b=0.0)
    parts, mparts, cut, ok, code_chars = [], [], 0, False, 0
    for b in blocks:
        tree, txt, dropped = parse_tolerant(b)
        if tree is None:
            continue
        ok = True; cut += dropped; code_chars += len(txt)
        ls = txt.split("\n")
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                meth = {st.name for st in node.body if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and not st.name.startswith("__")} if isinstance(node, ast.ClassDef) else set()
                parts.append(dict(name=node.name, w=part_chars(node, ls), refs=refs(node) - {node.name}, meth=meth))
                if isinstance(node, ast.ClassDef):                   # reading B: each method its own part
                    for st in node.body:
                        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            mparts.append(dict(name=st.name, w=fn_body_chars(st, ls), refs=refs(st) - {st.name}, meth=set()))
                else:
                    mparts.append(parts[-1])
    if not ok:
        return dict(code=True, parsed=False, n=0, k=0, share=0.0, conc=0.0, k_m=0, share_m=0.0, k_b=0, share_b=0.0,
                    share_c=0.0, share_bc=0.0, cut=0, names=[], code_frac=0.0, conc_b=0.0)
    res = dict(code=True, parsed=True, n=len(parts), cut=cut)
    for tag, P_, with_meth in (("", parts, False), ("_m", parts, True), ("_b", mparts, False)):
        adj = {i: set() for i in range(len(P_))}
        for i, a in enumerate(P_):
            for j, b in enumerate(P_):
                if i != j and (b["name"] in a["refs"] or (with_meth and b["meth"] & a["refs"])):
                    adj[i].add(j); adj[j].add(i)
        S = best_set([p["w"] for p in P_], adj) if P_ else ()
        S = tuple(i for i in S if P_[i]["w"] > 0)
        ws = [P_[i]["w"] for i in S]
        res["k" + tag] = len(S)
        res["share" + tag] = sum(ws) / total if len(S) >= 2 else 0.0
        if tag in ("", "_b"):
            res["share" + ("_c" if tag == "" else "_bc")] = sum(ws) / max(code_chars, 1) if len(S) >= 2 else 0.0
        if tag == "_b":
            res["names"] = [(P_[i]["name"], P_[i]["w"]) for i in S]
        if tag in ("", "_b"):
            res["conc" + tag] = (sum(ws) - max(ws)) / total if len(S) >= 2 else 0.0
    res["code_frac"] = code_chars / total
    return res


def walk_answers(path):
    W = np.load(path)
    out = []
    for i, nm in enumerate(W["names"]):
        t = dec(W[f"{i}_ids"])
        a = t.split("<|248045|>assistant")[-1]
        a = re.sub(r"<\|248068\|>.*?<\|248069\|>", "", a, flags=re.S)
        a = a.split("<|248046|>")[0]
        out.append((str(nm), a.strip("\n")))
    return out


def report(name, R, n_items):
    n = len(R)
    two = [r for r in R if r["k"] >= 2]
    line = [r for r in R if r["k"] >= 2 and r["share"] >= 0.5]
    line_m = [r for r in R if r["k_m"] >= 2 and r["share_m"] >= 0.5]
    say(f"  {name}: {n} answers ({n_items}); python code {sum(r['code'] for r in R)}, parsed {sum(r['parsed'] for r in R)} "
        f"(cut text: {sum(r['cut'] > 0 for r in R)}); parts per parsed answer {np.mean([r['n'] for r in R if r['parsed']] or [0]):.2f}; "
        f">= 2 independent parts: {100 * len(two) / max(n, 1):.1f} %; LINE (>= 2 independent parts holding >= 50 % of the answer): "
        f"{100 * len(line) / max(n, 1):.1f} % ({100 * len(line_m) / max(n, 1):.1f} % with method-name edges); token share in the best "
        f"set, mean over all answers {100 * np.mean([r['share'] for r in R] or [0]):.1f} % (over the >= 2 answers "
        f"{100 * np.mean([r['share'] for r in two] or [0]):.1f} %); concurrent share (best set - its largest part), mean "
        f"{100 * np.mean([r['conc'] for r in R] or [0]):.1f} %")
    lb = [r for r in R if r["k_b"] >= 2 and r["share_b"] >= 0.5]
    say(f"    sensitivities: methods as their own parts (reading B): >= 2 independent parts {100 * sum(r['k_b'] >= 2 for r in R) / max(n, 1):.1f} %, "
        f"LINE {100 * len(lb) / max(n, 1):.1f} %; share of the CODE (prose excluded) in the best set >= 50 %: "
        f"{100 * sum(r['k'] >= 2 and r['share_c'] >= 0.5 for r in R) / max(n, 1):.1f} % (reading B "
        f"{100 * sum(r['k_b'] >= 2 and r['share_bc'] >= 0.5 for r in R) / max(n, 1):.1f} %); code share of the answer "
        f"{100 * np.mean([r.get('code_frac', 0) for r in R] or [0]):.0f} %; concurrent share, reading B, mean "
        f"{100 * np.mean([r['conc_b'] for r in R] or [0]):.1f} %")
    return 100 * len(line) / max(n, 1)


t0 = time.time()
say(f"== R1 CENSUS (MoonShots, {time.strftime('%a %Y-%m-%d %H:%M')}): parts of one answer (FULL R2.2 (a)); tokens = characters")
res = {}
he = walk_answers(f"{TD}/d02b_walk_hpd0_code.npz")
res["HumanEval+ (HPD-0 walk, 24)"] = report("HumanEval+ (HPD-0 walk windows)", [census(a) for _, a in he], "24 windows")
mb = walk_answers(f"{TD}/d02b_walk_hpd0_mbpp.npz")
res["MBPP (HPD-0 walk, 100)"] = report("MBPP (HPD-0 walk windows)", [census(a) for _, a in mb], f"{len(mb)} windows")
MG, nraw = [], 0
with open(MAGIC) as f:
    for ln in f:
        nraw += 1
        d = json.loads(ln)
        if d.get("lang") == "python":
            MG.append(census(d["solution"]))
            if os.environ.get("SMOKE") and len(MG) >= 2000:
                break
res["Magicoder python"] = report("Magicoder OSS-Instruct, lang python", MG, f"of {nraw} items")
rng = np.random.default_rng(4)
ex = [i for i, r in enumerate(MG) if r["k_b"] >= 2]
say("    Magicoder examples with >= 2 independent parts (reading B; part: body chars), 4 drawn: " + " | ".join(
    ", ".join(f"{nm_}: {w_}" for nm_, w_ in MG[i]["names"]) + f" (answer share {100 * MG[i]['share_b']:.0f} %)"
    for i in rng.choice(ex, min(4, len(ex)), replace=False)))
# the agent windows: what code they carry
AG = []
for f in sorted(os.listdir(f"{TD}/d16w")):
    if f.startswith("agent_"):
        t = dec(np.load(f"{TD}/d16w/{f}")["ids"])
        turns = [x.split("<|248046|>")[0] for x in t.split("<|248045|>assistant")[1:]]
        content = re.findall(r"<parameter=content>\n?(.*?)(?:</parameter>|$)", t, re.S)
        tops = sum(len(re.findall(r"^(?:public |private |internal |export |async )*(?:func|def|class|struct|enum|function)\s", c, re.M)) for c in content)
        AG.append((f, len(turns), sum(len(py_blocks(x)) for x in turns), len(content), tops))
say("  d16w agent windows (bash / edit / text / write): " + "; ".join(f"{f[6:-4]} {nt} assistant turns, {npy} python blocks, "
    f"{nc} file contents ({tp} top-level decls)" for f, nt, npy, nc, tp in AG))
say(f"  -> the agent windows carry no Python: the census does not apply to them")
he_v, mb_v, mg_v = res["HumanEval+ (HPD-0 walk, 24)"], res["MBPP (HPD-0 walk, 100)"], res["Magicoder python"]
v = lambda x: "PASS" if x >= 30 else ("KILL" if x < 15 else "BETWEEN")
say(f"Lines (FULL R2.2 (a): PASS >= 30 % of answers with >= 2 independent parts holding >= 50 % of their tokens, KILL < 15 %): "
    f"HumanEval+ {he_v:.1f} % -> {v(he_v)}, MBPP {mb_v:.1f} % -> {v(mb_v)}, Magicoder python {mg_v:.1f} % -> {v(mg_v)} "
    f"[prediction 5-10 / <= 10 / 40-60 %]")
say(f"({time.time() - t0:.0f} s)")
if not os.environ.get("SMOKE"):
    open(os.path.expanduser("~/MoonShots/data/r1_census.txt"), "w").write("\n".join(lines) + "\n")
