#!/usr/bin/env python3 -I
"""drafter_ensemble.py (FableGod, mission 15). CPU, no numpy. On the 12 code windows with the 125B's classes: are two
DFlash2 variants wrong at the same positions? Walks on the same windows, teacher-forced along the 125B's answer
(~/TrainLab/data/d02b_walk_<variant>_code.npz with stored drafts and cf). Reports, on the positions both walks tested,
the accuracy of each, of 'either right' (an oracle pick) and of 'pick the draft with the higher cf', by class."""
import zipfile, struct, array, json, os, math, re
TD = os.path.expanduser("~/TrainLab/data"); OUT = os.path.expanduser("~/FableGod/decision_map/out")
def npz(path):
    z = zipfile.ZipFile(path); out = {}
    for i in z.infolist():
        d = z.read(i.filename); hl = struct.unpack("<H", d[8:10])[0]; hdr = d[10:10 + hl].decode("latin1"); body = d[10 + hl:]
        descr = re.search(r"'descr': '([^']+)'", hdr).group(1)
        shape = tuple(int(x) for x in re.findall(r"\d+", re.search(r"'shape': \(([^)]*)\)", hdr).group(1)))
        name = i.filename[:-4]
        if descr.startswith("<U"):
            w = int(descr[2:]); n = shape[0] if shape else 1
            out[name] = [body[j * w * 4:(j + 1) * w * 4].decode("utf-32-le").rstrip("\x00") for j in range(n)]
        elif descr == "<f2": n = len(body) // 2; out[name] = (list(struct.unpack(f"<{n}e", body)), shape)
        else:
            a = array.array({"<i2": "h", "<i4": "i", "<i8": "q", "<f4": "f", "<f8": "d"}[descr]); a.frombytes(body); out[name] = (a, shape)
    return out
base = json.load(open(f"{OUT}/baseline.json")); CL = {}
for nm, b in base.items():
    if b["dom"] != "code": continue
    cls = []
    for lp in b["lp"]:
        p1, p2 = math.exp(lp[0]), math.exp(lp[1]); cls.append("sure" if p1 >= 0.99 else ("tie" if p1 - p2 <= 0.1 else "mid"))
    CL[nm] = (b["a"], cls)
def per_position(walkfile):
    W = npz(f"{TD}/{walkfile}"); res = {}
    for i, nm in enumerate(W["names"]):
        if nm not in CL or f"{i}_dcf" not in W: continue
        a, cls = CL[nm]; ids = W[f"{i}_ids"][0]; checks, (nc, _) = W[f"{i}_checks"]; dcf, _ = W[f"{i}_dcf"]; dtok, _ = W[f"{i}_dtok"]
        for c in range(nc):
            k, m, A = checks[c * 3], checks[c * 3 + 1], checks[c * 3 + 2]
            for j in range(min(k + 1, m)):
                p = A + j
                if p + 1 >= len(ids) or not (0 <= p - a < len(cls)): continue
                res[(nm, p)] = (j < k, dcf[c * 15 + j], dtok[c * 15 + j], j + 1, cls[p - a])
    return res
A = per_position("d02b_walk_hpd0_code.npz"); print("hpd0 (L1-20) tested positions:", len(A))
for other in ("d02b_walk_hpdH_code.npz", "d02b_walk_hpdU_code.npz", "d02b_walk_kh_code.npz", "d02b_walk_d13S_code.npz", "d02b_walk_e5_code.npz"):
    if not os.path.exists(f"{TD}/{other}"): print(other, "absent"); continue
    B = per_position(other); common = [k for k in A if k in B]
    if not common: print(other, "no common positions (different windows or no stored cf)"); continue
    out = [f"{other}: common {len(common)}"]
    for cls in ("all", "sure", "mid", "tie"):
        K = [k for k in common if cls == "all" or A[k][4] == cls]
        if not K: continue
        ra = sum(A[k][0] for k in K) / len(K); rb = sum(B[k][0] for k in K) / len(K)
        either = sum(A[k][0] or B[k][0] for k in K) / len(K)
        pick = sum((A[k][0] if A[k][1] >= B[k][1] else B[k][0]) for k in K) / len(K)
        agree = sum(A[k][2] == B[k][2] for k in K) / len(K)
        out.append(f"{cls}: A {100 * ra:.1f} B {100 * rb:.1f} either {100 * either:.1f} pick-by-cf {100 * pick:.1f} same draft {100 * agree:.0f} % (n {len(K)})")
    print("; ".join(out))
