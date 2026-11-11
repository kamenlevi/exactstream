"""D-08 CALC screen: share of the 125B's GSM8K CoT completion tokens that a calculator determines exactly."""
import glob, re
import numpy as np
from tokenizers import Tokenizer

tk = Tokenizer.from_file("/Users/kamen/BigMoE/models/bare/tokenizer.json")
NUM = r"\(?\s*\d[\d,]*(?:\.\d+)?\s*\)?"
EXPR = re.compile(r"((?:" + NUM + r"\s*[-+*/]\s*)+" + NUM + r")\s*=\s*\$?\s*$")


def norm(t):
    for a, b in ((r"\times", "*"), (r"\cdot", "*"), ("×", "*"), ("÷", "/"), (r"\div", "/"), (" x ", " * "), ("{", ""), ("}", "")):
        t = t.replace(a, b)
    return t[-80:]


def results(v):
    out = []
    if abs(v - round(v)) < 1e-9:
        i = int(round(v))
        out += [str(i), f"{i:,}"]
    else:
        out.append(f"{v:.4f}".rstrip("0").rstrip("."))
        out.append(f"{v:.2f}")
    return out


tot = det = det_pos = mism = 0
examples = []
for f in sorted(glob.glob("/Users/kamen/Leap/data/dfl2d_math/*.npz")):
    z = np.load(f)
    ids, g = z["ids"].tolist(), int(z["gen"])
    n = len(ids)
    start = g - int(z["off"])  # gen = absolute index where the answer starts (the first version took the last gen tokens)
    tot += n - start
    i = start
    while i < n:
        m = EXPR.search(norm(tk.decode(ids[start:i])))
        if not m:
            i += 1
            continue
        expr = m.group(1).replace(",", "")
        try:
            v = eval(expr, {"__builtins__": {}}, {}) if re.fullmatch(r"[\d\s.+\-*/()]+", expr) else None
        except Exception:
            v = None
        if v is None:
            i += 1
            continue
        cont = tk.decode(ids[i:i + 12]).lstrip()
        hit = next((r for r in results(float(v)) if cont.startswith(r)), None)
        if hit is None:
            mism += 1
            if len(examples) < 6:
                examples.append((expr.strip(), round(float(v), 4), cont[:20]))
            i += 1
            continue
        k = 1
        while k < 12 and len(tk.decode(ids[i:i + k]).lstrip()) < len(hit):
            k += 1
        det += k
        det_pos += 1
        i += k
print(f"completion tokens {tot}; calculator-determined tokens matching the 125B {det} = {100 * det / tot:.2f} %"
      f" ({det_pos} results, {det / max(det_pos, 1):.2f} tokens each); expressions where the 125B wrote something else: {mism}")
for e in examples:
    print("  mismatch:", e)
