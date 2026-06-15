import os
import sys

# Make the repo root importable so tests can `import Problem` and `import src.goldcollector`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
