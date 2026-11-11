# 36 GB Mac simulation, in-process only (orchestrator Tue 13:3x): when SIM36_WS_GB is set, MLX reports that GPU working set
# (a 36 GB Mac's ~27 GB) as max_recommended_working_set_size, which the engine wires up to (R57_WIRED=1), and the MLX memory
# limit is set to 1.5x it (MLX's default rule). No system setting is touched. On this directory's PYTHONPATH only.
import os

if os.environ.get("SIM36_WS_GB"):
    try:
        import mlx.core as mx
        _ws = int(float(os.environ["SIM36_WS_GB"]) * 1e9)
        _real = mx.device_info

        def device_info():
            d = dict(_real())
            d["max_recommended_working_set_size"] = _ws
            return d

        mx.device_info = device_info
        mx.set_memory_limit(int(1.5 * _ws))
    except Exception as e:                                   # never break the interpreter for a test hook
        print(f"sim36 sitecustomize: not applied ({type(e).__name__}: {e})", flush=True)
