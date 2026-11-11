"""Fit12 (Wed 4 Nov 21:2x; the orchestrator 21:18): import the TRL store files of layer 41, experts N..511, that Leap's
batched Metal encoder wrote to /Users/kamen/Leap/data/trl41_for_fit12 (same inputs as the store: b10rcap R0..R7 + the
store's L41/_hx.npz; Leap's own check: bitwise == /Users/kamen/Fit12/scripts/fit_trl_store.py's C path on L18 1,536/1,536
and L41 e10/e50/e90). Read-only on Leap's dir. Nothing is copied unless every check passes:
  0. no fit_trl_store.py process runs (no second writer), and my store holds all three files of experts 0..N-1;
  1. each source file of experts N..511 exists and is whole: safetensors size = 8 + header + data end, the same tensors
     (names, dtypes, shapes, offsets) as my own file of that projection, metadata == mine with its own expert number;
  2. every file name present in both dirs is byte-identical (Leap's e10/e50/e90 test files and the N-2, N-1 overlap of
     its production run, which must be there);
  3. the spot experts my C path re-encoded into /Users/kamen/Fit12/data/trl_spot/L41 (trl_spot.sh) are byte-identical
     to Leap's.
Then, in expert order, each missing file is copied to <name>.tmp.imp.safetensors inside the store's L41 dir, compared
byte for byte with its source and renamed into place (os.replace, same volume; a file that exists is never replaced), so
Breakthrough's trl_gate (exact names + whole-file check, every 2 min) never sees a partial file and passes only after the
last rename. Usage: [BigMoE venv] python -B trl_import.py N SPOT_EXPERTS(e.g. 300,511) [--dry]"""
import filecmp, json, os, shutil, struct, subprocess, sys, time

L = 41
SRC = "/Users/kamen/Leap/data/trl41_for_fit12"
DST = f"/Users/kamen/Fit12/data/trl/L{L}"
SPOT = f"/Users/kamen/Fit12/data/trl_spot/L{L}"
LOG = "/Users/kamen/Fit12/logs/summary.txt"
PK = (("gate", 2), ("up", 2), ("down", 3))


def name(e, p, k):
    return f"e{e:03d}_{p}_k{k}.safetensors"


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    meta = h.pop("__metadata__", None)
    return n, h, meta


def whole(path, ref_t, ref_meta, e):
    """(ok, why): the file is whole and the one its name says."""
    if not os.path.exists(path):
        return False, "missing"
    try:
        n, t, meta = header(path)
    except Exception as ex:                       # truncated / not a safetensors file
        return False, f"header unreadable ({type(ex).__name__})"
    end = max(v["data_offsets"][1] for v in t.values())
    if os.path.getsize(path) != 8 + n + end:
        return False, f"size {os.path.getsize(path)} != 8 + {n} + {end}"
    if t != ref_t:
        return False, f"tensors differ from mine: {t}"
    if meta != dict(ref_meta, expert=str(e)):
        return False, f"metadata {meta} != mine with expert {e}"
    return True, ""


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry" in sys.argv
    N = int(pos[0])
    spots = [int(x) for x in pos[1].split(",")] if len(pos) > 1 and pos[1] else []
    t0 = time.time()
    bad = []

    # 0. no second writer; my store is whole below N
    if subprocess.run(["pgrep", "-f", "fit_trl_stor[e].py"], capture_output=True).returncode == 0:
        sys.exit("import: a fit_trl_store.py process runs; not started")
    low = [name(e, p, k) for e in range(N) for p, k in PK if not os.path.exists(os.path.join(DST, name(e, p, k)))]
    if low:
        sys.exit(f"import: my store lacks {len(low)} files below N={N} (first {low[:3]}); not started")

    # 1. every source file N..511 whole and the one its name says
    ref = {(p, k): header(os.path.join(DST, name(0, p, k)))[1:] for p, k in PK}
    for e in range(N, 512):
        for p, k in PK:
            ok, why = whole(os.path.join(SRC, name(e, p, k)), *ref[(p, k)], e)
            if not ok:
                bad.append(f"{name(e, p, k)}: {why}")
    print(f"[import] check 1: {3 * (512 - N)} source files for experts {N}..511: {len(bad)} bad", flush=True)

    # 2. every name in both dirs byte-identical; the N-2, N-1 overlap must be there
    src_names = sorted(f for f in os.listdir(SRC) if f.endswith(".safetensors") and ".tmp." not in f)
    both = [f for f in src_names if os.path.exists(os.path.join(DST, f))]
    diff = [f for f in both if not filecmp.cmp(os.path.join(SRC, f), os.path.join(DST, f), shallow=False)]
    need = [name(e, p, k) for e in range(max(0, N - 2), N) for p, k in PK]
    lack = [f for f in need if f not in both]
    bad += [f"{f}: differs from mine" for f in diff] + [f"{f}: overlap file missing in Leap's dir" for f in lack]
    ov = sorted({int(f[1:4]) for f in both})
    print(f"[import] check 2: {len(both)} files in both dirs (experts {ov}): {len(diff)} differ, "
          f"{len(lack)} of the {len(need)} N-2..N-1 overlap files missing", flush=True)

    # 3. my C path's spot re-encodes == Leap's
    nsp = 0
    for e in spots:
        for p, k in PK:
            a, b = os.path.join(SPOT, name(e, p, k)), os.path.join(SRC, name(e, p, k))
            if not (os.path.exists(a) and os.path.exists(b)):
                bad.append(f"spot {name(e, p, k)}: missing ({'mine' if not os.path.exists(a) else 'Leap'})")
            elif not filecmp.cmp(a, b, shallow=False):
                bad.append(f"spot {name(e, p, k)}: my C re-encode differs from Leap's")
            else:
                nsp += 1
    print(f"[import] check 3: spot experts {spots}: {nsp} of {3 * len(spots)} files byte-identical to my C path",
          flush=True)
    if not spots:
        bad.append("no spot experts given")

    if bad:
        print("[import] NOT IMPORTED:\n  " + "\n  ".join(bad[:20]) + (f"\n  ... {len(bad)} in all" if len(bad) > 20 else ""))
        sys.exit(1)
    if dry:
        print(f"[import] dry run: all checks pass; would copy the missing files of experts {N}..511")
        return

    # copy: temp name inside the store dir, byte check, rename; expert order (the last rename completes the layer)
    done = skip = 0
    for e in range(N, 512):
        for p, k in PK:
            s, d = os.path.join(SRC, name(e, p, k)), os.path.join(DST, name(e, p, k))
            if os.path.exists(d):
                skip += 1                         # compared in check 2
                continue
            tmp = d[:-len(".safetensors")] + ".tmp.imp.safetensors"
            shutil.copyfile(s, tmp)
            if not filecmp.cmp(s, tmp, shallow=False):
                os.remove(tmp)
                sys.exit(f"import: the copy of {name(e, p, k)} differs from its source; stopped after {done} files")
            if os.path.exists(d):
                os.remove(tmp)
                skip += 1
                continue
            os.replace(tmp, d)
            done += 1
    n = sum(1 for f in os.listdir(DST) if f.endswith(".safetensors") and ".tmp." not in f)
    msg = (f"SCR trl import: L{L} experts {N}..511 from Leap's Metal encoder ({SRC}): {done} files copied, {skip} "
           f"already there; checks: {3 * (512 - N)} source files whole with my tensors + metadata, {len(both)} shared "
           f"files byte-identical (experts {ov}), spot {spots} == my C path ({nsp} files); L{L} now {n} files "
           f"({time.time() - t0:.0f} s)")
    with open(LOG, "a") as f:
        f.write(time.strftime("%H:%M:%S") + " " + msg + "\n")
    print("[import] " + msg, flush=True)


if __name__ == "__main__":
    main()
