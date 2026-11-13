# Leap 11-12 (PF-B one-build, the 45 GB night test): when the engine imports `pflm`, load the one-build candidate
# (scripts/prefill15/onebuild/pflm.py) instead of engine/loop/pflm.py, so the pinned clone stays untouched. Active only when
# PYTHONPATH includes this folder (arms R1-R3; the reference arm R0 runs without it). As prof/sitecustomize.py.
import importlib.abc, importlib.util, os, sys

_PF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pflm.py")


class _PflmFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "pflm":
            return importlib.util.spec_from_file_location("pflm", _PF)
        return None


sys.meta_path.insert(0, _PflmFinder())
