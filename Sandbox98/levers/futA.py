"""BREAK98 Screen 1, variant A (FAR's FUTURE-STATE ROWS): at depth j <= 8 the head adds P(norm(the frozen 2B's state at
placeholder offset j under its anchor)), P zero-initialised. Placeholder: <|fim_pad|> (id 248063, the 125B's id too).
Mechanism and storage: sb/futstate.py; states from sb/futgen.py --variant A."""
from sb.futstate import make

setup, standin, PREFIX, MODULE_LR = make("A")
