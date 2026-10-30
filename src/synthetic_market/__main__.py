"""Allow python -m synthetic_market without import-time side effects."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
