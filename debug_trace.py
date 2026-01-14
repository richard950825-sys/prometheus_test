import traceback
import sys
from evo.core import EvoCore

try:
    print("Starting generation...")
    c = EvoCore()
    c.generate('sessions/demo6', n=1, reset=True)
    print("Generation complete.")
except Exception:
    print("Caught exception:")
    traceback.print_exc()
with open("debug_error.log", "w") as f:
    f.write(traceback.format_exc())
