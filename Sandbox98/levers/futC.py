"""BREAK98 Screen 1, variant C (FAR's FUTURE-STATE ROWS): at depth j <= 8 the head adds P(norm(the frozen 2B's state at
placeholder offset j under its anchor)), P zero-initialised. Placeholder: <|vision_pad|> (id 248055, never seen: the 'any foreign vector' control).
Mechanism and storage: sb/futstate.py; states from sb/futgen.py --variant C."""
from sb.futstate import make

setup, standin, PREFIX, MODULE_LR = make("C")
