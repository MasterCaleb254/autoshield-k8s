#!/usr/bin/env python3
"""
Run dataset tests with PYTHONPATH set to src for cross-platform use.
Usage: python src/scripts/run_test_dataset.py -v
"""
import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
src_path = str(repo_root / "src")
os.environ.setdefault("PYTHONPATH", src_path)

import pytest

if __name__ == "__main__":
    args = sys.argv[1:] or ["tests/test_dataset.py", "-v"]
    sys.exit(pytest.main(args))
