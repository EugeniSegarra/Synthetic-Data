"""Compatibility entry point. Prefer `python -m synthetic_market`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from synthetic_market.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["demo", "--out", "outputs/demo"]))
