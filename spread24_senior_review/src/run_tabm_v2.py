"""Retired V2 CLI.

V2.0 artifacts remain available for historical reproduction, but this executable
entrypoint is intentionally fail-closed so no run can bypass the V2.1 YAML
configuration/provenance contract or write new V2.1 artifacts.
"""
from __future__ import annotations

import sys

RETIREMENT_MESSAGE = (
    "V2 CLI is retired and cannot execute training/evaluation. "
    "Use: python run_tabm_v21.py ..."
)


def main(argv=None) -> int:
    del argv
    print(RETIREMENT_MESSAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
