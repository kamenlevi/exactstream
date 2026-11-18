"""patch_c25_prof.py (Leap, Mon 16 Nov 2026 13:5x): adds libmtlprof (Breakthrough's CHECK55 hook, copied to
/Users/kamen/BigMoE/data/r57/snap/frozen_s98n/Leap_third_party/mtlprof) to c25_loop.py as an opt-in: with C25_PROF set, the loop loads it BEFORE anything imports mlx
(the hook names pipelines as MLX creates them); c25_engine then records, per check, the GPU busy time (the union of the
command buffers' GPU intervals), the thread CPU per phase, dispatches / command buffers, and per arm the kernels by name.
Anchored; the anchor must match once; idempotent."""
import os
P = os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98n/Leap_scripts/week98/c25/c25_loop.py")
s = open(P).read()
A = 'sys.path.insert(1, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98n/Leap_third_party"))\n'
B = ('if os.environ.get("C25_PROF"):                          # C25 (Leap 11-16): libmtlprof before mlx (patch_c25_prof.py)\n'
     '    sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98n/Leap_third_party/mtlprof"))\n'
     '    import mtlprof as _c25mp\n'
     '    _c25mp.load(int(os.environ.get("C25_PROFLIB", "2")))\n')
if B in s:
    print("already patched")
else:
    assert s.count(A) == 1, s.count(A)
    s = s.replace(A, A + B)
    open(P, "w").write(s)
    print("patched")
