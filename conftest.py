import os
import sys

# Make the repo modules (model, data, util, ...) importable from the test suite.
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
