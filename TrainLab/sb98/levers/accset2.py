"""MTP Head accset arm 'cs2' (WEEK98 S7 ACCSET): delta 2, LAM 1, close-call survival. Implementation: accset_base.py."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from accset_base import make

standin, setup = make(2.0, 1.0, True)
