"""BigMoE's 127-node bank on the dev sets (HE+ 24 / MBPP 100; information, no gate): the dump-manifest check first, then
certify98_tree.py's read-out per dump, and the nesting: a 127-node k = 8 dump's tree-16 and tree-32 must equal the
32-node dump's exactly (BigMoE: the first 32 pops are BITWISE the 32-node dumps'). CPU only.
  .venv/bin/python tools/certify98_bank127.py > results/certify98/tree_bank127_2026-11-16.txt"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify98_dumpcheck as D                          # noqa: E402
import certify98_tree as T                               # noqa: E402

CE = T.CE
SETS = ("heplus24", "mbppbank")
H = os.path.expanduser("~")


def same(R, B, N):
    """tree-N of R == tree-N of B: per-window (kept, checks), the ratio vs chain-16, and (N = 16) the coverage counts"""
    ok = np.array_equal(R[f"_t{N}"], B[f"_t{N}"]) and R[f"tree{N}_vs_chain16"] == B[f"tree{N}_vs_chain16"]
    if N == 16:
        ok &= all(np.array_equal(a, b) for a, b in zip(R["cov_tree16"], B["cov_tree16"]))
    return ok


def main():
    bank = [f"{CE}/tree_t30b_{s}_n127k{k}.npz" for s in SETS for k in (8, 16)]
    lines, ok = D.check(f"{CE}/dumps_sha256.json", bank)
    print("## 1. The dump manifest, before any scoring (tools/certify98_dumpcheck.py)")
    print("\n".join(lines).replace(H, "~"))
    if not ok:
        print("BANK127: nothing scored (the manifest check failed)"); return 1
    print("\n## 2. certify98_tree.py per dump (the topk dumps are the 32-depth dev TOPK files)")
    allnest = True
    for s in SETS:
        tk = f"{CE}/topk_t30b_{s}.npz"
        B = T.score(tk, f"{CE}/tree_t30b_{s}.npz")
        R = {k: T.score(tk, f"{CE}/tree_t30b_{s}_n127k{k}.npz") for k in (8, 16)}
        for k in (8, 16):
            T.show(R[k], f"{s} n127 k{k}")
        nest = {N: same(R[8], B, N) for N in (16, 32)}
        allnest &= all(nest.values()) and not B["bad"]
        print(f"  nesting vs tree_t30b_{s}.npz (32 nodes, k 8): "
              + ", ".join(f"tree-{N} {'IDENTICAL' if v else 'DIFFERS'}" for N, v in nest.items()))
        print(f"  k 16 vs k 8 at the same tree size (information, MTP Head's T16): "
              + "  ".join(f"tree-{N} {R[16][f'tree{N}']:.3f} / {R[8][f'tree{N}']:.3f} = x{R[16][f'tree{N}'] / R[8][f'tree{N}']:.3f}"
                          for N in T.NS))
        print(f"  kept per check vs chain-32 ({R[8]['chain32']:.3f}), k 8: "
              + "  ".join(f"tree-{N} x{R[8][f'tree{N}'] / R[8]['chain32']:.3f}" for N in T.NS))
        ok &= not R[8]["bad"] and not R[16]["bad"]
    print(f"\nBANK127: {'PASS' if ok and allnest else 'FAIL'} (manifest PASS; scorer checks "
          f"{'clean' if ok else 'FAILED'}; nesting {'IDENTICAL' if allnest else 'DIFFERS'})")
    return 0 if ok and allnest else 1


if __name__ == "__main__":
    sys.exit(main())
