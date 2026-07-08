#!/usr/bin/env python3
"""
Legacy entry point — kept for backward compatibility.
Use the canonical module entry point instead:
    python -m src.dataset.prepare [args]

This thin wrapper delegates to src/dataset/prepare.main().
"""

import sys
from pathlib import Path

# Ensure project root is on path for imports
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.prepare import main

if __name__ == '__main__':
    main()
