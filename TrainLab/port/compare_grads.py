"""PORT gradient parity read-out (MTP Head, Thu 11-05): per-parameter norms and estimated relative differences between
port/ref_mlx.py grad (MLX) and port/test_parity.py grad (port) JSONs; largest differences first.
Usage: python compare_grads.py MLX.json PORT.json"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grad_stats as GS
print("\n".join(GS.compare(sys.argv[1], sys.argv[2])))
