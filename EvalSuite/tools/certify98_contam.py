"""CERTIFY98's contamination check (docs/PREREG_certify98_2026-11-15.md): test windows against a candidate's training data,
on token ids. CPU only, numpy only, one process (well under 1 GB for ~3K training files).
  .venv/bin/python tools/certify98_contam.py --test <npz|glob|@listfile> ... --train <npz|glob|@listfile> ...
      [--span 50] [--ngram 13] [--label L] [--forbid humaneval,mbpp,...]
Each npz needs `ids` (prompt + answer token ids); `src` is read when present.
- Source tags: the training files' src values are counted; a src or file name containing a --forbid pattern (default
  humaneval, mbpp, he_, tf_code) is FORBIDDEN, which makes the run VOID.
- Overlap: a test window sharing any --span-token run (default 50) with a training file is FLAGGED. Hash matches are
  verified token by token. Runs found in >= 5 % of the training files are template boilerplate (a chat template's fixed
  text), not content: they never flag, and their count is reported.
- The set is VOID if more than 5 % of its windows are flagged; else flagged windows are dropped and counted.
- Reported: the --ngram (default 13) overlap rate, the share of the test windows' n-grams found in the training data
  (hash-based, boilerplate n-grams excluded the same way).
Last line: `CONTAM <label>: CLEAN | FLAGGED <n> of <K> (dropped: ...) | VOID (...)`; exit 0 CLEAN / FLAGGED, 1 VOID,
2 bad input."""
import glob, os, sys
import numpy as np

B1 = np.uint64(0x9E3779B97F4A7C15)          # odd 64-bit base; arithmetic wraps mod 2^64
BOILER = 0.05


def files(args):
    out = []
    for a in args:
        if a.startswith("@"):
            out += [l.strip() for l in open(a[1:]) if l.strip()]
        else:
            g = sorted(glob.glob(os.path.expanduser(a)))
            out += g if g else [a]
    return out


def ids_of(f):
    z = np.load(f, mmap_mode="r")
    src = str(z["src"]) if "src" in z.files else ""
    return np.asarray(z["ids"]).astype(np.uint64), src


def hashes(x, w):
    """polynomial hashes of every w-token run of x (uint64, wrapping)"""
    if len(x) < w:
        return np.zeros(0, np.uint64)
    pw = np.array([pow(int(B1), w - 1 - k, 1 << 64) for k in range(w)], np.uint64)
    win = np.lib.stride_tricks.sliding_window_view(x, w)
    with np.errstate(over="ignore"):
        return (win * pw).sum(axis=1, dtype=np.uint64)


def main(argv):
    def opt(name, default):
        return argv[argv.index(name) + 1] if name in argv else default

    def group(name):
        if name not in argv:
            return []
        i = argv.index(name) + 1; out = []
        while i < len(argv) and not argv[i].startswith("--"):
            out.append(argv[i]); i += 1
        return out

    test, train = files(group("--test")), files(group("--train"))
    span, ng, label = int(opt("--span", 50)), int(opt("--ngram", 13)), opt("--label", "test set")
    forbid = [p for p in opt("--forbid", "humaneval,mbpp,he_,tf_code").lower().split(",") if p]
    if not test or not train:
        print(__doc__); return 2
    # training side: per-file distinct hashes (for the boilerplate count) and the span index for verification
    srcs, bad, T, Hs, Fs, Ps, Us, Gs = {}, [], [], [], [], [], [], []
    for i, f in enumerate(train):
        x, src = ids_of(f)
        srcs[src] = srcs.get(src, 0) + 1
        if any(p in (src + " " + os.path.basename(f)).lower() for p in forbid):
            bad.append(os.path.basename(f))
        T.append(x)
        h = hashes(x, span); Hs.append(h); Fs.append(np.full(len(h), i, np.int32)); Ps.append(np.arange(len(h), dtype=np.int32))
        Us.append(np.unique(h)); Gs.append(np.unique(hashes(x, ng)))      # per-file distinct runs: counts = files holding it
    H, F, P = np.concatenate(Hs), np.concatenate(Fs), np.concatenate(Ps)
    o = np.argsort(H, kind="stable"); H, F, P = H[o], F[o], P[o]
    hv, cnt = np.unique(np.concatenate(Us), return_counts=True)            # boilerplate: in >= 5 % of the training files
    boiler = set(hv[cnt >= BOILER * len(train)].tolist())
    G, gc = np.unique(np.concatenate(Gs), return_counts=True)              # the n-gram side, for the overlap rate
    gboiler = G[gc >= BOILER * len(train)]
    flagged, rates, ex = [], [], []
    for f in test:
        x, _ = ids_of(f)
        h = hashes(x, span)
        j = np.searchsorted(H, h); j = np.minimum(j, len(H) - 1)
        hit = np.flatnonzero(H[j] == h)
        real = []
        for q in hit:
            if int(h[q]) in boiler:
                continue
            k = j[q]
            while k < len(H) and H[k] == h[q]:                 # verify token by token against each candidate
                fi, pi = F[k], P[k]
                if np.array_equal(T[fi][pi:pi + span], x[q:q + span]):
                    real.append((q, os.path.basename(train[fi]), int(pi))); break
                k += 1
            if real:
                break
        if real:
            flagged.append(os.path.basename(f)); ex.append(real[0])
        g = np.unique(hashes(x, ng)); g = g[~np.isin(g, gboiler)]
        rates.append(float(np.isin(g, G).mean()) if len(g) else 0.0)
    K = len(test)
    print(f"{label}: {K} test windows vs {len(train)} training files; span {span}, n-gram {ng}")
    print(f"  training src tags: {dict(sorted(srcs.items(), key=lambda t: -t[1]))}; forbidden: {bad[:5]}{' ...' if len(bad) > 5 else ''}")
    print(f"  boilerplate runs skipped (in >= {100 * BOILER:.0f} % of training files): {len(boiler)} of span {span}, "
          f"{len(gboiler)} of n-gram {ng}")
    print(f"  {ng}-gram overlap rate: mean {100 * np.mean(rates):.2f} %, max {100 * np.max(rates):.2f} %")
    for w_, e in zip(flagged[:10], ex[:10]):
        print(f"  FLAGGED {w_}: its tokens {e[0]}..{e[0] + span - 1} equal {e[1]} at {e[2]}")
    if bad:
        print(f"CONTAM {label}: VOID ({len(bad)} forbidden training files)")
        return 1
    if len(flagged) > 0.05 * K:
        print(f"CONTAM {label}: VOID ({len(flagged)} of {K} windows flagged > 5 %; re-draw)")
        return 1
    if flagged:
        print(f"CONTAM {label}: FLAGGED {len(flagged)} of {K} (dropped: {', '.join(flagged)})")
        return 0
    print(f"CONTAM {label}: CLEAN (0 of {K} windows share a {span}-token run with the training data)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
