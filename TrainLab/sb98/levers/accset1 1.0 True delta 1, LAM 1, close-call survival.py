"""MTP Head accset arm 'accset1 1.0 True delta 1, LAM 1, close-call survival' (WEEK98 S7 ACCSET): accset1 1.0 True delta 1, LAM 1, close-call survival. Implementation: accset_base.py."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from accset_base import make

standin, setup = make(, 1.0, )
