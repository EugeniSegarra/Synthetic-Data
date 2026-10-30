"""Original filename retained as an entry point to the corrected package."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().with_name("synth_data.py")), run_name="__main__")
