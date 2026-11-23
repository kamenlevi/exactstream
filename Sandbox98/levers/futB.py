"""BREAK98 Screen 1, variant B (FAR's FUTURE-STATE ROWS): at depth j <= 8 the head adds P(norm(the frozen 2B's state at
placeholder offset j under its anchor)), P zero-initialised. Placeholder: the mean of the 2B's input-embedding table (an embedding override).
Mechanism and storage: sb/futstate.py; states from sb/futgen.py --variant B."""
from sb.futstate import make

setup, standin, PREFIX, MODULE_LR = make("B")
