"""MTP Head margin arm 'mg1k' (WEEK98 S8b MARGIN): delta 1, LAM 1, GAMMA 1, exact survival. Implementation:
margin_base.py."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from margin_base import make

standin, setup = make(1.0, 1.0, 1.0, False)
