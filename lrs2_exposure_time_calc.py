#!/usr/bin/env python3
"""Backward-compatible entry point for the LRS2 exposure-time calculator."""

from lrs2_etc.calculator import *  # noqa: F401,F403
from lrs2_etc.calculator import main


if __name__ == "__main__":
    raise SystemExit(main())
