Fit12 v3p (2026-10-25 22:4x): a copy of /Users/kamen/Fit12/v3k (sources only) + PFR engine calls (grep "Fit12 PFR" in
ext/bigmoe_ext.cpp): reset_layer(L, pools) = new pool buffers for layer L, every slot empty; invalidate_tables() = every
published residency-table line's tag -1, so no pass reads a stale table. Additive: nothing changes unless they are called.
Build: cmake -S ext -B ext/build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=/Users/kamen/BigMoE/.venv/bin/python -DNB_CMAKE_DIR=/Users/kamen/BigMoE/.nb215/nanobind/cmake && cmake --build ext/build -j 2
