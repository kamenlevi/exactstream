# Leap 11-11 (prefill 1.5K stage split): when the engine imports `pflm`, load the instrumented copy next to this file
# instead of engine/loop/pflm.py (the pinned clone stays untouched). Active only when PYTHONPATH includes this folder.
import importlib.abc, importlib.util, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))


class _PflmFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "pflm":
            return importlib.util.spec_from_file_location("pflm", os.path.join(_HERE, "pflm_prof.py"))
        return None


sys.meta_path.insert(0, _PflmFinder())
